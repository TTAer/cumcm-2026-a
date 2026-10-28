from solve import *
BASE=ROOT.parent/'A题_改进交付版';out=[]
for q in (3,4):
    m=Model(q,640,tail='last',mesh='surface');sol=m.solve(432000,method='BDF')
    f=np.load(BASE/'计算结果'/f'q{q}_curve.npz')
    mask=f['t']<=sol.t[-1];tt=f['t'][mask];v=sol.sol(tt)
    a=np.stack([np.interp(f['x'],m.x,v[k:k+m.n+1,j]) for j in range(len(tt)) for k in (0,m.n+1)]).reshape(len(tt),2,-1)
    diffT=abs(a[:,0].T-f['T'][:,mask]);diffC=abs(a[:,1].T-f['C'][:,mask])
    summary=json.loads((BASE/'计算结果'/f'q{q}_summary.json').read_text(encoding='utf-8'))
    record={'q':q,'共同配置':'last、surface；640/BDF对1280/Radau','时刻数':len(tt),'细网格节点数':len(f['x']),'最大温差':float(diffT.max()),'最大含水率差':float(diffC.max()),'交叉时间差_s':float(sol.t[-1]-summary['crossing_s']),'细场最小C':float(f['C'].min()),'细场最大径向反增':float(np.maximum(np.diff(f['C'],axis=0),0).max())}
    if q==4:
        raw=np.load(BASE/'计算结果/q4_full_precision.npz');tt2=raw['t'];T,C=sample(m,sol,tt2,np.arange(21)/10);values=sol.sol(tt2)
        T=np.c_[T,values[m.n]];C=np.c_[C,values[2*(m.n+1)-1]]
        d=abs(C-raw['C']);i,j=np.unravel_index(np.nanargmax(d),d.shape)
        record['分钟点含真实表面']={'最大温差':float(np.nanmax(abs(T-raw['T']))),'最大含水率差':float(d[i,j]),'发生时刻_s':float(tt2[i]),'列索引':int(j),'域外掩码一致':bool(np.array_equal(np.isnan(C),np.isnan(raw['C'])))}
    out.append(record);print(json.dumps(record,ensure_ascii=False),flush=True)
(BASE/'验证/主方案独立粗网格对照.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')

# Boundary mapping check independent of rounded worksheet comparisons.
rows=np.genfromtxt(BASE/'计算结果/q4_C.csv',delimiter=',',skip_header=1)
m=Model(4,1280,tail='last',mesh='surface');r=np.arange(21)/10
expected=np.array([r>m.R(t)*100+1e-12 for t in rows[:,0]])
assert np.array_equal(np.isnan(rows[:,1:22]),expected)
assert np.all(rows[:,1:][np.isfinite(rows[:,1:])]>=0)
same=np.array([abs(m.R(t)*100-1.2)<1e-12 for t in rows[:,0]])
assert np.array_equal(rows[same,13],rows[same,-1])
(BASE/'验证/Q4物理坐标核验.json').write_text(json.dumps({'域外留空':'通过','非负性':'通过','半径1_2cm时对应表面一致':'通过','检查行数':len(rows)},ensure_ascii=False,indent=2),encoding='utf-8')
