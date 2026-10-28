"""A题径向热质传递：节点型有限体积 + Radau/BDF，原始附件只读。

修订版要点（相对历史版本）：
1. 环境延拓主方案为【末次观测值延续】：t>14400 s 后恒为附件1末次观测值
   50.165 ℃ / 0.04986 kg/kg（np.interp 末端自然截断）；'platform' 为
   恒温平台 50 ℃ / 0.05 kg/kg 的敏感性对照。
2. 问题2 结果文件覆盖【整个烘干过程】：与问题3 同一模型积分到全域阈值事件，
   按每 1 s 导出温度与水分浓度；论文表3/表4 仍取前 3 h 稀疏时刻。
3. 大规模时刻采样分块执行，避免 dense_output 一次性展开导致内存爆掉。
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
try:
    import numpy, scipy, sympy, mpmath  # 优先使用系统已安装的完整包（cp313）
except ImportError:
    sys.path.insert(0, str(ROOT / '依赖'))  # 缺失时才回退到随题附带的依赖
import argparse, json, csv, time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix, coo_matrix
from scipy.interpolate import PchipInterpolator
from scipy.special import expi
import openpyxl

DATA = ROOT.parent.parent / '附件'      # …/A题/附件
OUT  = ROOT.parent / '计算结果'          # …/A题_终稿/计算结果
OUT.mkdir(parents=True, exist_ok=True)

def read_data():
    tables = []
    for name in ('附件1.xlsx', '附件2.xlsx'):
        w = openpyxl.load_workbook(DATA / name, read_only=True, data_only=True)
        tables.append(np.array(list(w.active.values)[1:], dtype=float)); w.close()
    return tables

ENV, RAD = read_data()
ENV_TAIL_T, ENV_TAIL_C = float(ENV[-1,1]), float(ENV[-1,2])   # 末次观测值 50.165 / 0.04986

class Model:
    def __init__(self, case, n=80, tail='last', temp_offset=0., mass_scale=1., diff_scale=1., shrink=True, latent=0., flux='kirchhoff', mesh='uniform'):
        self.case, self.n = case, n
        self.x = np.linspace(0, 1, n+1)
        self.mesh = mesh
        if mesh == 'surface': self.x = 1-(1-self.x)**1.5
        elif mesh != 'uniform': raise ValueError(mesh)
        self.faces = np.r_[0., (self.x[:-1]+self.x[1:])/2, 1.]
        self.vol = np.diff(self.faces**2)/2
        self.tail, self.temp_offset = tail, temp_offset
        self.hm = 8e-7 * mass_scale
        self.diff_scale, self.shrink, self.latent = diff_scale, shrink, latent
        self.flux = flux
        self.radius = PchipInterpolator(RAD[:,0], RAD[:,1]*.01)
        m = n+1
        sp = lil_matrix((2*m+1, 2*m+1), dtype=int)
        for i in range(m):
            js = list(range(max(0,i-1),min(m,i+2)))
            for j in js:
                sp[i,j] = sp[i,m+j] = sp[m+i,j] = sp[m+i,m+j] = 1
        sp[-1,m-1] = sp[-1,2*m-1] = 1
        self.sparsity = sp.tocsr()

    def env(self, t):
        # t<=14400 s：附件1 分段线性插值；t>14400 s：np.interp 末端截断 =
        # 末次观测值延续（主方案）。tail=='platform' 时改为恒温平台 50/0.05。
        T = np.interp(t, ENV[:,0], ENV[:,1])
        C = np.interp(t, ENV[:,0], ENV[:,2])
        if t > ENV[-1,0] and self.tail == 'platform':
            T, C = 50., .05
        return T+self.temp_offset, C

    def R(self, t):
        return float(self.radius(min(t,RAD[-1,0]))) if self.case==4 and self.shrink else .02

    def props(self, T, C):
        # Numerical guard only: accepted solutions are independently checked nonnegative.
        C = np.maximum(C, 1e-10)
        if self.case == 1:
            rho = np.full_like(C,820.); cp = np.full_like(C,2600.)
            k = np.full_like(C,.36); D = 7e-9*np.exp(-.89/C)
        elif self.case in (2,3):
            rho = 650+128*C; cp = 1450+2736*C/(C+1)
            k = .21+.38*C/(C+1)
            D = 2.4e-3*np.exp(-.45/C-3850/(T+273.15))
        else:
            rho = 760+90*C; cp = 1850+2150*C/(C+1)
            k = .12+.20*C/(C+1)
            D = 4.2e-4*np.exp(-.30/C-3850/(T+273.15))
        return rho, cp, k, D*self.diff_scale

    def operator(self, u, coeff, boundary_flux, R, face_coef=None):
        if face_coef is None: face_coef = 2*coeff[:-1]*coeff[1:]/(coeff[:-1]+coeff[1:])
        flux = np.zeros(self.n+2)
        flux[1:-1] = self.faces[1:-1]*face_coef*np.diff(u)/np.diff(self.x)/R**2
        flux[-1] = boundary_flux/R
        return np.diff(flux)/self.vol

    def rhs(self, t, y):
        m = self.n+1; T,C = y[:m],y[m:2*m]
        rho,cp,k,D = self.props(T,C); Ta,Ca = self.env(t); R = self.R(t)
        mass_out = self.hm*(C[-1]-Ca)
        # Optional surface-latent-loss scenario: rho_d is fixed initial dry bulk density.
        rho_d0 = (820. if self.case==1 else (650+128*2.55 if self.case in (2,3) else 760+90*2.55))/3.55
        rho_d = rho_d0*(.02/R)**2
        heat_in = 25*(Ta-T[-1])-self.latent*rho_d*mass_out
        dT = self.operator(T,k,heat_in,R)/(rho*cp)
        face_D = None
        if self.flux == 'kirchhoff':
            a,A = (.89,7e-9) if self.case==1 else (.45,2.4e-3) if self.case in (2,3) else (.30,4.2e-4)
            cc = np.maximum(C,1e-10)
            F = cc*np.exp(-a/cc)+a*expi(-a/cc)
            dc = np.diff(cc)
            eff = np.exp(-a/((cc[:-1]+cc[1:])/2))
            np.divide(np.diff(F),dc,out=eff,where=np.abs(dc)>1e-7)
            face_D = A*eff*self.diff_scale
            if self.case!=1: face_D *= np.exp(-3850/((T[:-1]+T[1:])/2+273.15))
        dC = self.operator(C,D,-mass_out,R,face_D)
        return np.r_[dT,dC,2*mass_out/R]

    def solve(self, limit, rtol=2e-9, max_step=60., method='BDF'):
        m = self.n+1
        y0 = np.r_[np.full(m,28.),np.full(m,2.55),0.]
        def hit(t,y): return np.max(y[m:2*m])-.15
        hit.terminal = True; hit.direction = -1
        started = time.perf_counter()
        self.run_config = dict(method=method, mesh=self.mesh, tail=self.tail,
                               rtol=rtol, atol=rtol*.02, max_step=max_step)
        # 问题2/3/4 都用全域阈值事件：问题2 需要积分到烘干结束才能导出全程结果。
        sol = solve_ivp(self.rhs,(0,limit),y0,method=method,rtol=rtol,atol=rtol*.02,
                        max_step=max_step,dense_output=True,jac=self.jacobian,
                        events=hit if self.case in (2,3,4) else None)
        if not sol.success: raise RuntimeError(sol.message)
        hit_reached = bool(self.case in (2,3,4) and sol.t_events is not None and len(sol.t_events[0]))
        print(json.dumps({'case':self.case,'n':self.n,'tail':self.tail,'last_h':sol.t[-1]/3600,
                          'threshold':hit_reached,'steps':len(sol.t),
                          'runtime_s':round(time.perf_counter()-started,2)},ensure_ascii=False),flush=True)
        return sol

    def jacobian(self,t,y):
        # Six-color local finite difference; exact zero cumulative-loss column is excluded.
        # Avoid automatic Jacobian step inflation in spatially flat states.
        m=self.n+1;f0=self.rhs(t,y);rows=[];cols=[];values=[]
        for field in (0,1):
            step=1e-4 if field==0 else 1e-6
            for color in range(3):
                nodes=np.arange(color,m,3);yy=y.copy();yy[field*m+nodes]+=step
                delta=(self.rhs(t,yy)-f0)/step
                for j in nodes:
                    affected=np.r_[np.arange(max(0,j-1),min(m,j+2)),m+np.arange(max(0,j-1),min(m,j+2))]
                    if j==m-1:affected=np.r_[affected,2*m]
                    rows.extend(affected.tolist());cols.extend([field*m+int(j)]*len(affected));values.extend(delta[affected].tolist())
        return coo_matrix((values,(rows,cols)),shape=(2*m+1,2*m+1)).tocsc()

def sample(model, sol, times, radii_cm, chunk=4000):
    """分块采样：dense_output 一次展开全部时刻会占用数 GB 内存。"""
    m = model.n+1
    T = np.full((len(times), len(radii_cm)), np.nan)
    C = np.full_like(T, np.nan)
    for a in range(0, len(times), chunk):
        b = min(a+chunk, len(times))
        tt = np.asarray(times[a:b], dtype=float)
        vals = sol.sol(tt)
        for i in range(len(tt)):
            actual = model.x*model.R(tt[i])*100
            T[a+i] = np.interp(radii_cm, actual, vals[:m,i], right=np.nan)
            C[a+i] = np.interp(radii_cm, actual, vals[m:2*m,i], right=np.nan)
    return T, C

def export_matrix(filename, times, radii, values, surface=None):
    with (OUT/filename).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['时间/s\\到药材中心的距离/cm']+[f'{x:g}' for x in radii]+(['药材表面'] if surface is not None else []))
        for i,t in enumerate(times):
            row=[float(t)]+[None if not np.isfinite(x) else round(float(x),4) for x in values[i]]
            if surface is not None: row.append(round(float(surface[i]),4))
            writer.writerow(row)

def save_case(model, sol):
    from types import SimpleNamespace
    q = model.case; m = model.n+1; end = float(sol.t[-1]); crossing = end
    hit_required = q in (2,3,4)
    if hit_required and not len(sol.t_events[0]):
        raise RuntimeError('积分上限内未达标，不能导出为最终达标结果。')
    if q==1 and end < 1800:
        raise RuntimeError('实际积分未覆盖结果表时间范围，禁止外推导出。')
    if q==2 and end < 10800:
        raise RuntimeError('问题2 积分未覆盖前3小时正文表区间，禁止导出。')
    if hit_required and len(sol.t_events[0]):
        # Strict inequality: integrate to the first whole second AFTER the threshold crossing.
        stop = float(np.floor(end)+1)
        cont = solve_ivp(model.rhs,(end,stop),sol.y[:,-1],method=model.run_config['method'],
                         rtol=2e-9,atol=4e-11,dense_output=True,jac=model.jacobian)
        if not cont.success: raise RuntimeError(cont.message)
        old_dense = sol.sol
        def combined(times):
            ts=np.atleast_1d(times); ret=old_dense(np.minimum(ts,crossing)); mask=ts>crossing
            if mask.any(): ret[:,mask]=cont.sol(ts[mask])
            return ret[:,0] if np.ndim(times)==0 else ret
        sol = SimpleNamespace(sol=combined, t=np.r_[sol.t,stop], t_events=sol.t_events)
        end = stop
    # 输出时刻
    if q==1:
        times = np.arange(1,1801,dtype=float)
    elif q==2:
        # 问题2：整个烘干过程，每 1 s
        times = np.arange(1,np.floor(end)+1,1.)
        if not len(times) or end-times[-1]>1e-7: times=np.r_[times,end]
    else:
        # 问题3/4：每 60 s + 整数秒停止时刻
        times = np.arange(60,np.floor(end/60)*60+1,60.)
        if not len(times) or end-times[-1]>1e-7: times=np.r_[times,end]
    radii = np.round(np.arange(0,2.0001,.1),1)
    T,C = sample(model,sol,times,radii)
    surface = sample(model,sol,times,np.array([1.0]))[1][:,0] if False else None
    if q==4:
        # 实际表面值：ξ=1 节点
        surf = np.empty(len(times))
        for a in range(0,len(times),4000):
            b=min(a+4000,len(times)); vals=sol.sol(np.asarray(times[a:b],dtype=float))
            surf[a:b]=vals[2*m-1]
        export_matrix(f'q{q}_C.csv',times,radii,C,surf)
    else:
        export_matrix(f'q{q}_C.csv',times,radii,C)
    if q in (1,2): export_matrix(f'q{q}_T.csv',times,radii,T)
    # 正文稀疏表：问题1 取 100..1800 s；问题2 取前 3 h 每 0.5 h；问题3/4 取每 6 h + 停止时刻
    paper_t = (np.array([100,300,600,900,1200,1500,1800.]) if q==1
               else np.arange(1800,10801,1800.) if q==2
               else np.r_[np.arange(21600,end,21600.),end])
    pr = np.arange(0,2.00001,.5); pT,pC = sample(model,sol,paper_t,pr)
    if q==4:
        psv = np.empty(len(paper_t))
        for a in range(0,len(paper_t),4000):
            b=min(a+4000,len(paper_t)); psv[a:b]=sol.sol(np.asarray(paper_t[a:b],dtype=float))[2*m-1]
        export_matrix(f'q{q}_正文_C.csv',paper_t,pr,pC,psv)
    else:
        export_matrix(f'q{q}_正文_C.csv',paper_t,pr,pC)
    if q in (1,2): export_matrix(f'q{q}_正文_T.csv',paper_t,pr,pT)
    checks_t = np.unique(np.r_[0,np.linspace(0,end,1001),paper_t]); v = sol.sol(checks_t)
    mean = (2*model.vol)@v[m:2*m]
    conservation = float(np.max(np.abs(mean+v[-1]-2.55)))
    result = {'case':q,'n':model.n,'run_config':model.run_config,'env_tail':[ENV_TAIL_T,ENV_TAIL_C],
              'C_end_max':float(v[m:2*m,-1].max()),'end_s':end,'end_h':end/3600,
              'crossing_s':crossing,'crossing_h':crossing/3600,
              'threshold_reached':bool(hit_required and len(sol.t_events[0])),
              'R_end_cm':model.R(end)*100,'C_end_center':float(v[m,-1]),'C_end_surface':float(v[2*m-1,-1]),
              'T_end_center':float(v[0,-1]),'T_end_surface':float(v[m-1,-1]),
              'min_C':float(v[m:2*m].min()),'mass_residual':conservation,
              'radial_monotonic_violation':float(np.maximum(np.diff(v[m:2*m],axis=0),0).max()),
              'rows':len(times),'paper_times_s':paper_t.tolist(),
              'paper_T':[[None if not np.isfinite(x) else float(x) for x in row] for row in pT],
              'paper_C':[[None if not np.isfinite(x) else float(x) for x in row] for row in pC],
              'paper_surface_C':v[2*m-1,np.searchsorted(checks_t,paper_t)].tolist()}
    np.savez_compressed(OUT/f'q{q}_curve.npz',t=checks_t,x=model.x,T=v[:m],C=v[m:2*m],
                        R=np.array([model.R(t) for t in checks_t]),mean=mean)
    (OUT/f'q{q}_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if q==4:
        np.savetxt(OUT/'q4_半径辅助.csv',np.c_[times,[model.R(t) for t in times]],delimiter=',',header='time_s,radius_m',comments='')
    print(json.dumps({k:result[k] for k in ('case','end_h','crossing_h','C_end_center','C_end_surface','min_C','mass_residual','rows')},ensure_ascii=False),flush=True)
    return result

def main():
    global OUT
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=1280)
    p.add_argument('--case', type=int, choices=range(5), default=0)
    p.add_argument('--method', choices=['BDF','Radau'], default='Radau')
    p.add_argument('--tail', choices=['last','platform'], default='last')
    p.add_argument('--output', type=Path, default=ROOT.parent/'临时运行')
    args = p.parse_args()
    if args.n < 4: p.error('网格区间数至少为4')
    OUT = args.output.resolve(); OUT.mkdir(parents=True, exist_ok=True)
    np.savetxt(OUT/'环境数据_SI.csv', ENV, delimiter=',', header='time_s,T_C,C_kg_kg', comments='')
    np.savetxt(OUT/'半径数据_SI.csv', np.c_[RAD[:,0],RAD[:,1]*.01], delimiter=',', header='time_s,radius_m', comments='')
    cases = [args.case] if args.case else [1,2,3,4]
    for q in cases:
        model = Model(q, args.n, tail=args.tail, mesh='surface' if q in (1,2,4) else 'uniform')
        sol = model.solve(1800 if q==1 else 3600*120, method=args.method)
        save_case(model, sol)

if __name__=='__main__': main()
