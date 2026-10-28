# -*- coding: utf-8 -*-
"""修订版验证：网格收敛、解析解对照、质量守恒、参数敏感性。

环境主方案 = 末次观测值延续（tail='last'，即附件1末次观测 50.165 ℃ / 0.04986）。
敏感性对照含：环境改用恒温平台 50 ℃ / 0.05（tail='platform'）。
输出到 …/A题_终稿/验证/。
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import sympy as sy
from solve import Model, sample, ENV, RAD, ENV_TAIL_T, ENV_TAIL_C

VOUT = HERE.parent / '验证'
VOUT.mkdir(parents=True, exist_ok=True)

def run():
    records = []
    # ---------- 符号核验 ----------
    r,t = sy.symbols('r t', positive=True)
    R = sy.Function('R')(t); U = sy.Function('U')
    material = sy.simplify(sy.diff(U(r/R,t), t) + (sy.diff(R,t)/R)*r*sy.diff(U(r/R,t), r))
    a,c = sy.symbols('a c', positive=True)
    primitive = c*sy.exp(-a/c) + a*sy.Ei(-a/c)
    assert sy.simplify(sy.diff(primitive,c) - sy.exp(-a/c)) == 0
    (VOUT/'符号核验.txt').write_text(
        '均匀径向收缩的物质导数（应为0，即材料迁移与网格运动相消）：\n'+str(material)+
        '\n扩散通量原函数 dF/dC=exp(-a/C) 核验通过。\n'
        f'环境主方案=末次观测值延续: {ENV_TAIL_T} ℃, {ENV_TAIL_C} kg/kg\n', encoding='utf-8')

    # ---------- 网格收敛 ----------
    outputs = {}
    for q in (1,2,3,4):
        for n in (80,160,320,640):
            model = Model(q, n)
            lim = 1800 if q==1 else 10800 if q==2 else 432000
            sol = model.solve(lim, rtol=2e-9, max_step=60.)
            rec = {'问题':q, '网格区间数':n, '结束时间_h':round(sol.t[-1]/3600, 6)}
            if q in (1,2):
                tt = (np.array([100,300,600,900,1200,1500,1800.]) if q==1
                      else np.arange(1800,10801,1800.))
                TT,CC = sample(model, sol, tt, np.arange(0,2.001,.1))
                outputs[(q,n)] = (TT,CC)
                if n > 80:
                    prev = outputs[(q,n//2)]
                    rec['温度较前网格最大差_C'] = float(np.max(abs(TT-prev[0])))
                    rec['含水率较前网格最大差'] = float(np.max(abs(CC-prev[1])))
            records.append(rec)
            print(json.dumps(rec, ensure_ascii=False), flush=True)

    # ---------- 解析解对照（恒温边界圆柱 Robin 级数） ----------
    from scipy.optimize import brentq
    from scipy.special import j0, j1
    model = Model(1, 320); model.env = lambda t: (50., 50.)
    model.tail = 'platform'
    sol = model.solve(1800, rtol=2e-9, max_step=30.)
    Bi = 25*.02/.36; alpha = .36/(820*2600)
    grid = np.linspace(1e-7, 180, 10000)
    f = grid*j1(grid) - Bi*j0(grid)
    roots = [brentq(lambda z: z*j1(z)-Bi*j0(z), grid[i], grid[i+1])
             for i in range(len(grid)-1) if f[i]*f[i+1] < 0]
    roots = np.array(roots)
    coef = 2*j1(roots)/(roots*(j0(roots)**2 + j1(roots)**2))
    errs = []
    for tt in (100,300,600,1800):
        exact = 50 - 22*np.sum((coef*np.exp(-roots**2*alpha*tt/.02**2))[:,None]*j0(roots[:,None]*model.x), axis=0)
        errs.append(float(np.max(abs(sol.sol(tt)[:321] - exact))))
    benchmark = {'恒温边界圆柱解析解最大温度误差_C': max(errs), '模式数': int(len(roots))}
    print(json.dumps(benchmark, ensure_ascii=False), flush=True)

    # ---------- 参数敏感性（160 格） ----------
    sensitivity = []
    base_scen = [('基准（末次观测延续）', {}),
                 ('环境改用恒温平台50_0.05', {'tail':'platform'}),
                 ('烘房温度减1度', {'temp_offset':-1}),
                 ('烘房温度加1度', {'temp_offset':1}),
                 ('传质系数减10%', {'mass_scale':.9}),
                 ('传质系数加10%', {'mass_scale':1.1}),
                 ('扩散系数减10%', {'diff_scale':.9}),
                 ('扩散系数加10%', {'diff_scale':1.1})]
    for q in (3,4):
        scen = list(base_scen)
        if q == 4:
            scen.append(('附录4参数但不收缩', {'shrink':False}))
        for name, kw in scen:
            model = Model(q, 160, **kw)
            lim = 864000 if name == '附录4参数但不收缩' else 432000
            sol = model.solve(lim, rtol=2e-8, max_step=120.)
            sensitivity.append({'问题':q, '情景':name,
                                '达标时间_h':round(sol.t[-1]/3600, 4),
                                '是否达标':bool(sol.t_events is not None and len(sol.t_events[0]))})
            print(json.dumps(sensitivity[-1], ensure_ascii=False), flush=True)
    # 诊断：端点调和平均通量
    for q in (3,4):
        model = Model(q, 160, flux='harmonic')
        sol = model.solve(432000, rtol=2e-8, max_step=120.)
        sensitivity.append({'问题':q, '情景':'诊断：端点调和平均通量，非最终结果',
                            '达标时间_h':round(sol.t[-1]/3600, 4),
                            '是否达标':bool(sol.t_events is not None and len(sol.t_events[0]))})
        print(json.dumps(sensitivity[-1], ensure_ascii=False), flush=True)

    result = {'环境主方案': {'尾部': '末次观测值延续', 'T_a_C': ENV_TAIL_T, 'C_a_kg_kg': ENV_TAIL_C},
              '环境对照方案': {'尾部': '恒温平台', 'T_a_C': 50.0, 'C_a_kg_kg': 0.05},
              '网格与容差': records, '解析解对照': benchmark, '敏感性': sensitivity}
    (VOUT/'验证汇总.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('WROTE', VOUT/'验证汇总.json', flush=True)

if __name__ == '__main__':
    run()
