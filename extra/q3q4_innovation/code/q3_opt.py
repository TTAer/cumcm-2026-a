# -*- coding: utf-8 -*-
"""Q3 创新：三段恒温制度的代理建模 → 三目标 Pareto → 真模型回代验证 → 决策推荐。

目标（全部为最小化）：
  f1 = t60                   达标时间 / h
  f2 = SEC                   单位去水量对流供热量 / MJ·kg^-1（显热口径，见说明）
  f3 = dose = -ln(Q_ret)     品质降解剂量（Arrhenius，文献 Ea，未标定）
报告属性（不作为目标）：峰值热流 q_max、末期物料温度、过程最高物料温度。
"""
import json, sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K   # noqa: E402

LAM_VAP = 2.383e6           # J/kg 50 degC 附近水的汽化潜热（仅用于物理折算，未进入模型）
N_CAND = 8000
VERIFY_N = 160
VERIFY_RTOL = 1e-7


# ---------- 正交 Legendre 基 ----------
def legendre_vals(x, deg):
    """返回 [P0(x), P1(x), ..., Pdeg(x)]（标准 Legendre，未正交化）。"""
    v = [np.ones_like(x)]
    if deg >= 1: v.append(x)
    for n in range(2, deg + 1):
        v.append(((2 * n - 1) * x * v[n - 1] - (n - 1) * v[n - 2]) / n)
    return v


def pce_design(u, deg=3):
    """正交归一化 Legendre 张量积基（三项及以下）。u:(n,3) in [-1,1]。"""
    nrm = [math.sqrt(2 * k + 1) for k in range(deg + 1)]
    V = [legendre_vals(u[:, d], deg) for d in range(3)]
    cols, names = [], []
    for i in range(deg + 1):
        for j in range(deg + 1 - i):
            for k in range(deg + 1 - i - j):
                cols.append(nrm[i] * V[0][i] * nrm[j] * V[1][j] * nrm[k] * V[2][k])
                names.append(f'P{i}{j}{k}')
    return np.column_stack(cols), names


def fit_pce(X, Y, deg=3):
    pass


def normalize(X, box):
    """box = (lo, hi) 标量（三变量同区间）。"""
    lo, hi = float(box[0]), float(box[1])
    return 2.0 * (np.asarray(X, float) - lo) / (hi - lo) - 1.0


def pareto_min(F):
    """返回 3 目标最小化的非支配点索引。"""
    order = np.lexsort((F[:, 2], F[:, 1], F[:, 0]))
    Fs = F[order]
    front = []
    for i in range(len(Fs)):
        p = Fs[i]
        if front:
            Kf = Fs[front]
            if np.any(np.all(Kf <= p, axis=1)):
                continue
            keep = ~np.all(p <= Kf, axis=1)
            if not keep.all():
                front = [front[j] for j in range(len(front)) if keep[j]]
        front.append(i)
    return order[np.array(front, dtype=int)]


def topsis(R, w=None):
    """R: (m,k) 成本型指标。返回贴近度与熵权。"""
    R = np.asarray(R, float)
    m, k = R.shape
    Z = R / np.sqrt((R ** 2).sum(axis=0, keepdims=True))
    P = Z / Z.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        E = -(P * np.log(np.where(P > 0, P, 1.0))).sum(axis=0) / math.log(m)
    if w is None:
        w = (1 - E) / (1 - E).sum()
    best, worst = Z.min(axis=0), Z.max(axis=0)
    dp = np.sqrt((w * (Z - best) ** 2).sum(axis=1))
    dn = np.sqrt((w * (Z - worst) ** 2).sum(axis=1))
    return dn / (dp + dn), w, E


# ---------- 主流程 ----------
def load_doe():
    rows = []
    for p in sorted(K.RES.glob('q3_doe_shard*.json')):
        rows += json.loads(p.read_text(encoding='utf-8'))
    ok = [r for r in rows if r.get('reached')]
    bad = [r for r in rows if not r.get('reached')]
    ok.sort(key=lambda r: r['idx'])
    return ok, bad


def main():
    doe, bad = load_doe()
    print(f'DOE 成功 {len(doe)} 条，失败/未达标 {len(bad)} 条', flush=True)
    X = np.array([r['temps'] for r in doe], float)
    t60 = np.array([r['t60_h'] for r in doe], float)
    sec = np.array([r['SEC_J_per_kg'] for r in doe], float)
    dose = np.array([r['dose'] for r in doe], float)
    qmax = np.array([max(r['traj']['q_W_m2']) for r in doe], float)
    Tmax = np.array([max(r['traj']['TbarK']) - 273.15 for r in doe], float)

    # ---- 三目标代理（PCE 3 阶，训练/测试划分）----
    is_test = np.array([r['label'].startswith('corner') for r in doe])
    U = normalize(X, K.T_BOX)
    B, names = pce_design(U, 3)

    def fit_one(y):
        c, *_ = np.linalg.lstsq(B[~is_test], y[~is_test], rcond=None)
        pred = B @ c
        ss = ((y[~is_test] - pred[~is_test]) ** 2).sum()
        r2 = 1 - ss / ((y[~is_test] - y[~is_test].mean()) ** 2).sum()
        rmse_te = float(np.sqrt(((y[is_test] - pred[is_test]) ** 2).mean())) if is_test.any() else None
        return c, float(r2), rmse_te, pred

    sur = {}
    for nm, y in (('t60_h', t60), ('SEC_MJ', sec / 1e6), ('dose', dose)):
        c, r2, rmse_te, pred = fit_one(y)
        # 5 折交叉验证（仅在 128 个 Sobol 训练点上）
        rng0 = np.random.default_rng(7)
        fold = rng0.integers(0, 5, size=int((~is_test).sum()))
        Utr, ytr = U[~is_test], y[~is_test]
        errs = []
        for f in range(5):
            te_f = fold == f
            cf, *_ = np.linalg.lstsq(B[~is_test][~te_f], ytr[~te_f], rcond=None)
            errs.append((B[~is_test][te_f] @ cf - ytr[te_f]) ** 2)
        cv = float(np.sqrt(np.concatenate(errs).mean()))
        sur[nm] = dict(c=c.tolist(), r2=r2, test_rmse=rmse_te, cv5_rmse=cv,
                       train_max_err=float(np.abs(pred[~is_test] - y[~is_test]).max()))
        print(f'  代理 {nm:<8} 训练R²={r2:.6f}  结构点留出RMSE={rmse_te:.5f}  '
              f'5折CV_RMSE={cv:.5f}', flush=True)

    # ---- 代理上求 Pareto ----
    rng = np.random.default_rng(20260912)
    Uc = rng.uniform(-1, 1, size=(N_CAND, 3))
    Bc, _ = pce_design(Uc, 3)
    Fc = np.column_stack([Bc @ np.array(sur['t60_h']['c']),
                          Bc @ np.array(sur['SEC_MJ']['c']),
                          Bc @ np.array(sur['dose']['c'])])
    idx = pareto_min(Fc)
    print(f'  代理 Pareto 点数 = {len(idx)}', flush=True)

    # 沿 f1 分位抽稀到 ~16 个 + 三个极值
    Fp, Up = Fc[idx], Uc[idx]
    order = np.argsort(Fp[:, 0])
    pick = set()
    for q in np.linspace(0, len(order) - 1, 14).round().astype(int):
        pick.add(int(order[q]))
    pick.add(int(np.argmin(Fp[:, 1]))); pick.add(int(np.argmin(Fp[:, 2])))
    pick = sorted(pick)

    lo, hi = float(K.T_BOX[0]), float(K.T_BOX[1])
    cand = lo + (Up[pick] + 1) / 2 * (hi - lo)

    # ---- 真模型回代验证 ----
    ver = []
    for i, T in enumerate(cand):
        d = K.run_schedule(3, tuple(T), n=VERIFY_N, rtol=VERIFY_RTOL, max_step=200.,
                           method='BDF', limit_h=150.0, want_traj=True)
        d['src'] = 'pareto_verify'
        ver.append(d)
        print(f'  验证 {i+1:2d}/{len(cand)} T=({T[0]:6.2f},{T[1]:6.2f},{T[2]:6.2f}) '
              f't60={d["t60_h"]:8.4f} h SEC={d["SEC_J_per_kg"]/1e6:7.4f} Qret={d["Q_ret"]:.6f}',
              flush=True)

    # ---- 附加点：观测温区上限 / 恒温基准 / 箱角点 / 箱内最速点 ----
    extra = []
    Tmin = cand[int(np.argmin([r['t60_h'] for r in ver]))]
    for lab, T in (('obs_upper', (K.T_OBS_MAX, K.T_OBS_MAX, K.T_OBS_MAX)),
                   ('const_50p165', (50.165, 50.165, 50.165)),
                   ('corner_hot', (K.T_BOX[1],) * 3),
                   ('corner_cold', (K.T_BOX[0],) * 3),
                   ('min_time_box', tuple(Tmin))):
        d = K.run_schedule(3, tuple(T), n=VERIFY_N, rtol=VERIFY_RTOL, max_step=200.,
                           method='BDF', limit_h=150.0, want_traj=True)
        d['src'] = lab
        extra.append(d)
        print(f'  附加 {lab:<14} T=({T[0]:6.2f},{T[1]:6.2f},{T[2]:6.2f}) '
              f't60={d["t60_h"]:8.4f} h Qret={d["Q_ret"]:.6f}', flush=True)

    allv = ver + extra
    V = np.array([[r['t60_h'], r['SEC_J_per_kg'] / 1e6, r['dose']] for r in allv])
    vidx = pareto_min(V)

    # ---- TOPSIS 决策 ----
    clo, w, E = topsis(V[vidx])
    w = np.asarray(w, float)
    best = int(vidx[int(np.argmax(clo))])
    rec = dict(
        min_time=allv[int(vidx[int(np.argmin(V[vidx][:, 0]))])],
        min_energy=allv[int(vidx[int(np.argmin(V[vidx][:, 1]))])],
        max_quality=allv[int(vidx[int(np.argmin(V[vidx][:, 2]))])],
        topsis=allv[best], topsis_weights=dict(zip(['t60', 'SEC', 'dose'], w.tolist())),
        topsis_entropy=dict(zip(['t60', 'SEC', 'dose'], np.asarray(E).tolist())),
        topsis_closeness=float(np.max(clo)),
    )

    # ---- Ea 稳健性：用已保存轨迹重算 dose，重做 Pareto 与推荐 ----
    ea_res = {}
    for ea in K.EA_SET:
        dd = np.array([K.dose_from_traj(r['traj'], ea) for r in allv])
        V2 = np.column_stack([V[:, 0], V[:, 1], dd])
        v2 = pareto_min(V2)
        c2, w2, _ = topsis(V2[v2])
        b2 = int(v2[int(np.argmax(c2))])
        ea_res[f'{int(ea)}'] = dict(
            n_front=int(len(v2)),
            min_time_t60_h=float(V2[v2][:, 0].min()),
            min_dose_Qret=float(math.exp(-V2[v2][:, 2].min())),
            topsis_temps=list(map(float, allv[b2]['temps'])),
            topsis_t60_h=float(allv[b2]['t60_h']),
            topsis_Qret=float(math.exp(-float(V2[b2][2]))),
            topsis_weights=dict(zip(['t60', 'SEC', 'dose'], np.asarray(w2).tolist())),
        )
        print(f'  Ea={ea/1000:.0f} kJ/mol -> 前沿 {len(v2)} 点，TOPSIS 推荐 T='
              f'{ea_res[f"{int(ea)}"]["topsis_temps"]}', flush=True)

    # ---- 基准对照 ----
    base = K.baseline()
    out = dict(
        meta=dict(doe_n=len(doe), doe_failed=len(bad), n_run=80, rtol=1e-6,
                  verify_n=VERIFY_N, verify_rtol=VERIFY_RTOL,
                  box=list(K.T_BOX), switch_h=list(K.SWITCH_H),
                  quality_model=dict(k_ref_per_h=K.K_REF_PER_H, T_ref_K=K.T_REF_K,
                                     Ea_J_per_mol=K.EA_REF, source='文献量级设定，未标定'),
                  note='f2=SEC 为模型内对流供热量/去水量（显热口径）；本模型不含潜热项，'
                       '故另报 SEC_phys=SEC+lambda。'),
        surrogate={k: {kk: vv for kk, vv in v.items() if kk != 'c'} for k, v in sur.items()},
        surrogate_coef={k: v['c'] for k, v in sur.items()},
        baseline=dict(q3_t60_h=base['q3']['t60_h'], q3_t_cross_s=base['q3']['t_cross_s'],
                      q3_SEC_MJ=base['q3']['SEC_J_per_kg'] / 1e6, q3_Qret=base['q3']['Q_ret']),
        doe=dict(temps=X.tolist(), t60_h=t60.tolist(), SEC_MJ=(sec / 1e6).tolist(),
                 dose=dose.tolist(), qmax_W_m2=qmax.tolist(), Tmax_C=Tmax.tolist(),
                 labels=[r['label'] for r in doe]),
        pareto_surrogate=dict(temps=(lo + (Up[pick] + 1) / 2 * (hi - lo)).tolist(),
                              f=Fp[pick].tolist()),        pareto_verified=[dict(temps=r['temps'], t60_h=r['t60_h'],
                              SEC_MJ=r['SEC_J_per_kg'] / 1e6, SEC_phys_MJ=r['SEC_J_per_kg'] / 1e6 + LAM_VAP / 1e6,
                              dose=r['dose'], Q_ret=r['Q_ret'], src=r['src'],
                              qmax_W_m2=max(r['traj']['q_W_m2']),
                              Tbar_max_C=max(r['traj']['TbarK']) - 273.15) for r in allv],
        pareto_verified_front=[dict(temps=allv[i]['temps'], t60_h=allv[i]['t60_h'],
                                    SEC_MJ=allv[i]['SEC_J_per_kg'] / 1e6, dose=allv[i]['dose'],
                                    Q_ret=allv[i]['Q_ret'], src=allv[i]['src']) for i in vidx],
        recommend={k: (dict(temps=v['temps'], t60_h=v['t60_h'], SEC_MJ=v['SEC_J_per_kg'] / 1e6,
                            SEC_phys_MJ=v['SEC_J_per_kg'] / 1e6 + LAM_VAP / 1e6,
                            dose=v['dose'], Q_ret=v['Q_ret'],
                            qmax_W_m2=max(v['traj']['q_W_m2']),
                            Tbar_max_C=max(v['traj']['TbarK']) - 273.15,
                            Xbar_end=v['Xbar_end'])
                      if isinstance(v, dict) and 'temps' in v else v)
                  for k, v in rec.items()},
        ea_robustness=ea_res,
        lam_vap_J_per_kg=LAM_VAP,
    )
    K.jdump(out, 'q3_优化结果.json')
    print('\n已写出 results/q3_优化结果.json', flush=True)

    # 文本摘要
    L = []
    L.append(f'题目口径基准：t60 = {base["q3"]["t60_h"]:.4f} h（{base["q3"]["t60_s"]:.0f} s）')
    L.append(f'恒温 50.165 全程：t60 = {extra[1]["t60_h"]:.4f} h')
    L.append(f'观测温区上限恒温 {K.T_OBS_MAX}：t60 = {extra[0]["t60_h"]:.4f} h')
    L.append(f'工艺扩展上界恒温 {K.T_BOX[1]:.0f}：t60 = {extra[2]["t60_h"]:.4f} h')
    L.append(f'工艺扩展下界恒温 {K.T_BOX[0]:.0f}：t60 = {extra[3]["t60_h"]:.4f} h')
    L.append(f'TOPSIS 熵权 = ' + json.dumps(rec['topsis_weights'], ensure_ascii=False))
    L.append('')
    L.append(f'{"T1":>7}{"T2":>7}{"T3":>7}{"t60/h":>10}{"SEC MJ/kg":>12}{"SEC_phys":>10}{"Qret":>10}{"qmax":>9}{"Tmax":>8}')
    for i in vidx:
        r = allv[int(i)]
        L.append(f'{r["temps"][0]:7.2f}{r["temps"][1]:7.2f}{r["temps"][2]:7.2f}'
                 f'{r["t60_h"]:10.4f}{r["SEC_J_per_kg"]/1e6:12.4f}'
                 f'{r["SEC_J_per_kg"]/1e6+LAM_VAP/1e6:10.4f}{r["Q_ret"]:10.4f}'
                 f'{max(r["traj"]["q_W_m2"]):9.1f}{max(r["traj"]["TbarK"])-273.15:8.2f}')
    (K.RES / 'q3_Pareto前沿.txt').write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L), flush=True)
    np.savez_compressed(K.RES / 'q3_pareto.npz',
                        temps=np.array([r['temps'] for r in allv]),
                        t60=np.array([r['t60_h'] for r in allv]),
                        SEC=np.array([r['SEC_J_per_kg'] for r in allv]) / 1e6,
                        dose=np.array([r['dose'] for r in allv]),
                        is_front=np.isin(np.arange(len(allv)), vidx))
    # 轨迹（供画图）：逐点存为二维数组
    def _mat(key):
        m = max(len(r['traj'][key]) for r in allv)
        A = np.full((len(allv), m), np.nan)
        for i, r in enumerate(allv):
            A[i, :len(r['traj'][key])] = r['traj'][key]
        return A
    tb = K.S.Model(3, 160, tail='last', mesh='surface', env_interp='pchip')
    tbs = tb.solve(150 * 3600., rtol=1e-6, max_step=300., method='BDF')
    tbq = K.qoi(tb, tbs, want_traj=True, rtol=1e-6, max_step=300.)
    np.savez_compressed(
        K.RES / 'q3_trajs.npz',
        temps=np.array([r['temps'] for r in allv]),
        src=np.array([r['src'] for r in allv]),
        t_h=_mat('t_h'), Xbar=_mat('Xbar'), TbarK=_mat('TbarK'),
        Ta=_mat('Ta'), Ts=_mat('Ts'), q=_mat('q_W_m2'),
        base_t_h=np.array(tbq['traj']['t_h']), base_Xbar=np.array(tbq['traj']['Xbar']),
        base_Ta=np.array(tbq['traj']['Ta']), base_TbarK=np.array(tbq['traj']['TbarK']))


if __name__ == '__main__':
    main()
