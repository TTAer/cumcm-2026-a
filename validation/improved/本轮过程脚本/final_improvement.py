"""统一最后观测主方案；只重算Q2/Q3共享过程，复用其他主结果。"""
from solve import *
import solve as s
import shutil, hashlib, gc
DEST=ROOT.parent/'A题_改进交付版'
for sub in ('计算结果','提交结果','验证','求解代码'):(DEST/sub).mkdir(parents=True,exist_ok=True)
OLD=ROOT.parent/'A题_终稿'
for q in (1,2):
    shutil.copy2(OLD/'提交结果'/f'result{q}.xlsx',DEST/'提交结果'/f'result{q}.xlsx')
for p in (ROOT/'计算结果').glob('q1_*'):shutil.copy2(p,DEST/'计算结果'/p.name)
for p in (ROOT/'资料复核改进/last_Radau_1280_surface').glob('q4_*'):shutil.copy2(p,DEST/'计算结果'/p.name)
shutil.copy2(ROOT/'资料复核改进/last_Radau_1280_surface/q4_半径辅助.csv',DEST/'计算结果/q4_半径辅助.csv')
s.OUT=DEST/'计算结果'
m=Model(3,1280,tail='last',mesh='surface');sol=m.solve(432000,method='Radau')
s.save_case(m,sol)
m.case=2;s.save_case(m,sol,full_process=True)
checks={'Q2_Q3共享积分':True,'q2旧CSV与本次CSV':{}}
for field in ('T','C'):
    a=np.loadtxt(OLD/'计算结果'/f'q2_{field}.csv',delimiter=',',skiprows=1)
    b=np.loadtxt(s.OUT/f'q2_{field}.csv',delimiter=',',skiprows=1)
    delta=np.abs(a-b)
    checks['q2旧CSV与本次CSV'][field]={'最大差':float(delta.max()),'不同单元格数':int(np.count_nonzero(delta))}
    del a,b,delta
checks['终点证明']={}
for tt in (int(sol.t[-1]),float(sol.t[-1])):
    v=sol.sol(tt)[m.n+1:2*(m.n+1)]
    checks['终点证明'][str(tt)]={'全域最大C':float(v.max()),'最大值节点':int(v.argmax())}
(DEST/'验证/主过程重算.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False),flush=True)
del sol,m;gc.collect()

# Retain full failure diagnostics, including physical location and excluded-domain mask.
fine=np.load(ROOT/'资料复核改进/nominal_Radau_1280_surface/q4_full_precision.npz')
m=Model(4,640,mesh='surface');sol=m.solve(432000,method='BDF')
tt=fine['t'];T,C=sample(m,sol,tt,np.arange(21)/10);v=sol.sol(tt)
T=np.c_[T,v[m.n]];C=np.c_[C,v[2*(m.n+1)-1]]
diag={}
for label,a,b in [('T',T,fine['T']),('C',C,fine['C'])]:
    d=np.abs(a-b);i,j=np.unravel_index(np.nanargmax(d),d.shape)
    diag[label]={'最大差':float(d[i,j]),'时间_s':float(tt[i]),'列':int(j),'粗值':float(a[i,j]),'细值':float(b[i,j]),'实际表面最大差':float(np.max(d[:,-1])),'超过2_5e_5数量':int(np.count_nonzero(d>2.5e-5)),'域外掩码一致':bool(np.array_equal(np.isnan(a),np.isnan(b)))}
(DEST/'验证/Q4原失败定位.json').write_text(json.dumps(diag,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(diag,ensure_ascii=False),flush=True)
