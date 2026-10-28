"""A题径向热质传递：节点型有限体积 + BDF，原始附件只读。"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
try:
    import numpy, scipy, sympy, mpmath  # 优先使用系统已安装的完整包（cp313）
except ImportError:
    sys.path.insert(0, str(ROOT.parent / '求解' / '依赖'))  # 本工作区已有依赖；独立使用时安装requirements.txt
import argparse, json, csv, time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix, coo_matrix
from scipy.interpolate import PchipInterpolator
from scipy.special import expi
import openpyxl

DATA = ROOT / '附件'
OUT = ROOT / '计算结果'
OUT.mkdir(exist_ok=True)

def read_data():
    tables = []
    for name in ('附件1.xlsx', '附件2.xlsx'):
        w = openpyxl.load_workbook(DATA / name, read_only=True, data_only=True)
        tables.append(np.array(list(w.active.values)[1:], dtype=float)); w.close()
    return tables

ENV, RAD = read_data()

class Model:
    def __init__(self, case, n=80, tail='last', temp_offset=0., mass_scale=1., diff_scale=1., shrink=True, latent=0., flux='kirchhoff', mesh='uniform', env_interp='linear'):
        self.case, self.n = case, n
        self.x = np.linspace(0, 1, n+1)
        self.mesh = mesh
        if mesh == 'surface': self.x = 1-(1-self.x)**1.5
        elif mesh != 'uniform': raise ValueError(mesh)
        self.faces = np.r_[0., (self.x[:-1]+self.x[1:])/2, 1.]
        self.vol = np.diff(self.faces**2)/2
        self.tail, self.temp_offset = tail, temp_offset
        if tail not in ('nominal','last','mean'): raise ValueError(tail)
        if env_interp not in ('linear','pchip'): raise ValueError(env_interp)
        self.env_interp = env_interp
        self.env_curve = PchipInterpolator(ENV[:,0],ENV[:,1:],axis=0,extrapolate=False)
        self.tail_mean = ENV[ENV[:,0]>=ENV[-1,0]-3600,1:].mean(axis=0)
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
        T = np.interp(t, ENV[:,0], ENV[:,1])
        C = np.interp(t, ENV[:,0], ENV[:,2])
        if self.env_interp == 'pchip': T,C=self.env_curve(np.clip(t,ENV[0,0],ENV[-1,0]))
        if t > ENV[-1,0] and self.tail == 'nominal': T, C = 50., .05
        if t > ENV[-1,0] and self.tail == 'mean': T, C = self.tail_mean
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
        self.run_config = dict(method=method, mesh=self.mesh, tail=self.tail, env_interp=self.env_interp, rtol=rtol, atol=rtol*.02, max_step=max_step)
        sol = solve_ivp(self.rhs,(0,limit),y0,method=method,rtol=rtol,atol=rtol*.02,
                        max_step=max_step,dense_output=True,jac=self.jacobian,
                        events=hit if self.case in (3,4) else None)
        if not sol.success: raise RuntimeError(sol.message)
        print(json.dumps({'case':self.case,'n':self.n,'last_h':sol.t[-1]/3600,'steps':len(sol.t),
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

def sample(model,sol,times,radii_cm):
    vals = sol.sol(times); m=model.n+1
    T,C=[],[]
    for i,t in enumerate(times):
        actual = model.x*model.R(t)*100
        T.append(np.interp(radii_cm,actual,vals[:m,i],right=np.nan))
        C.append(np.interp(radii_cm,actual,vals[m:2*m,i],right=np.nan))
    return np.asarray(T),np.asarray(C)

def export_matrix(filename,times,radii,values,surface=None):
    with (OUT/filename).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['时间/s\\到药材中心的距离/cm']+[f'{x:g}' for x in radii]+(['药材表面'] if surface is not None else []))
        for i,t in enumerate(times):
            row=[float(t)]+[None if not np.isfinite(x) else round(float(x),4) for x in values[i]]
            if surface is not None: row.append(round(float(surface[i]),4))
            writer.writerow(row)

def save_case(model,sol,full_process=False):
    from types import SimpleNamespace
    q=model.case; m=model.n+1; end=float(sol.t[-1]); crossing=end
    has_event=bool(sol.t_events is not None and len(sol.t_events[0]))
    if q==2 and full_process and not has_event:
        raise RuntimeError('问题2全程输出需使用同附录3的全域达标事件终点。')
    if q in (3,4) and not len(sol.t_events[0]):
        raise RuntimeError('积分上限内未达标，不能导出为最终达标结果。')
    if q in (1,2) and end < (1800 if q==1 else 10800):
        raise RuntimeError('实际积分未覆盖结果表时间范围，禁止外推导出。')
    if has_event and (q in (3,4) or (q==2 and full_process)):
        # Strict inequality: integrate to the first whole second AFTER the threshold crossing.
        stop=float(np.floor(end)+1)
        cont=solve_ivp(model.rhs,(end,stop),sol.y[:,-1],method=model.run_config['method'],rtol=2e-9,atol=4e-11,dense_output=True,jac=model.jacobian)
        if not cont.success: raise RuntimeError(cont.message)
        old_dense=sol.sol
        def combined(times):
            ts=np.atleast_1d(times);ret=old_dense(np.minimum(ts,crossing));mask=ts>crossing
            if mask.any():ret[:,mask]=cont.sol(ts[mask])
            return ret[:,0] if np.ndim(times)==0 else ret
        sol=SimpleNamespace(sol=combined,t=np.r_[sol.t,stop],t_events=sol.t_events)
        end=stop
    if q in (1,2): times=np.arange(1,1801 if q==1 else int(end)+1 if full_process else 10801,dtype=float)
    else:
        # export only true in-domain times plus the threshold event, not an extrapolated later minute.
        times=np.arange(60,np.floor(end/60)*60+1,60.)
        if not len(times) or end-times[-1]>1e-7: times=np.r_[times,end]
    radii=np.round(np.arange(0,2.0001,.1),1)
    # Full-process question 2 has over 200k seconds: evaluate dense output in chunks.
    batches=[sample(model,sol,times[i:i+400],radii) for i in range(0,len(times),400)]
    T=np.vstack([b[0] for b in batches]);C=np.vstack([b[1] for b in batches])
    export_matrix(f'q{q}_C.csv',times,radii,C,sol.sol(times)[2*m-1] if q==4 else None)
    if q in (1,2): export_matrix(f'q{q}_T.csv',times,radii,T)
    paper_t=np.array([100,300,600,900,1200,1500,1800.]) if q==1 else np.arange(1800,10801,1800.) if q==2 else np.r_[np.arange(21600,end,21600.),end]
    pr=np.arange(0,2.00001,.5); pT,pC=sample(model,sol,paper_t,pr)
    export_matrix(f'q{q}_正文_C.csv',paper_t,pr,pC,sol.sol(paper_t)[2*m-1] if q==4 else None)
    if q in (1,2): export_matrix(f'q{q}_正文_T.csv',paper_t,pr,pT)
    checks_t=np.unique(np.r_[0,np.linspace(0,end,1001),paper_t]); v=sol.sol(checks_t)
    mean=(2*model.vol)@v[m:2*m]
    conservation=float(np.max(np.abs(mean+v[-1]-2.55)))
    result={'case':q,'n':model.n,'run_config':model.run_config,'coverage':'full_process' if (q>=3 or full_process) else 'requested_short_window','C_end_max':float(v[m:2*m,-1].max()),'end_s':end,'end_h':end/3600,'crossing_s':crossing,'crossing_h':crossing/3600,'threshold_reached':has_event,
            'R_end_cm':model.R(end)*100,'C_end_center':float(v[m,-1]),'C_end_surface':float(v[2*m-1,-1]),
            'T_end_center':float(v[0,-1]),'T_end_surface':float(v[m-1,-1]),
            'min_C':float(v[m:2*m].min()),'mass_residual':conservation,
            'radial_monotonic_violation':float(np.maximum(np.diff(v[m:2*m],axis=0),0).max()),
            'rows':len(times),'paper_times_s':paper_t.tolist(),'paper_T':[[None if not np.isfinite(x) else float(x) for x in row] for row in pT],
            'paper_C':[[None if not np.isfinite(x) else float(x) for x in row] for row in pC],
            'paper_surface_C':v[2*m-1,np.searchsorted(checks_t,paper_t)].tolist()}
    np.savez_compressed(OUT/f'q{q}_curve.npz',t=checks_t,x=model.x,T=v[:m],C=v[m:2*m],R=np.array([model.R(t) for t in checks_t]),mean=mean)
    (OUT/f'q{q}_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if q==4:
        np.savetxt(OUT/'q4_半径辅助.csv',np.c_[times,[model.R(t)*100 for t in times]],delimiter=',',header='time_s,radius_cm',comments='')
    return result

def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=1280);p.add_argument('--case',type=int,choices=range(5),default=0)
    p.add_argument('--method',choices=['BDF','Radau'],default='Radau')
    p.add_argument('--output',type=Path,default=ROOT/'临时运行');args=p.parse_args()
    if args.n < 4: p.error('网格区间数至少为4')
    OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
    np.savetxt(OUT/'环境数据_SI.csv',ENV,delimiter=',',header='time_s,T_C,C_kg_kg',comments='')
    np.savetxt(OUT/'半径数据_SI.csv',np.c_[RAD[:,0],RAD[:,1]*.01],delimiter=',',header='time_s,radius_m',comments='')
    cases=[args.case] if args.case else [1,2,4]
    for q in cases:
        model=Model(3 if q==2 else q,args.n,mesh='surface')
        sol=model.solve(1800 if q==1 else 3600*120,method=args.method)
        if q==2: save_case(model,sol)
        model.case=q
        print(json.dumps(save_case(model,sol,full_process=q==2),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
