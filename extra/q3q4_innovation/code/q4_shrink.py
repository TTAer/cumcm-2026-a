# -*- coding: utf-8 -*-
"""Q4 创新：收缩本构（外生附件2 → 内生 R(X)）+ 孔隙率演化 + 表皮阻力情景。

三块内容
  D1-a 收缩本构回归：R/R0 = 1+k(X-X0)（径向线性）与 (R/R0)^3 = 1+b(X-X0)（体积线性），
       含"前 40%/60% 拟合、后段预测"的检验，以及 X=0 外推。
  D1-b 内生收缩：把附件2 的 R(t) 换成由本模型平均含水率经本构推得的 R(t)，做自洽重算；
       并做"只用前 40% 标定的本构预测全程时长"的预测性检验。
  D3   孔隙率演化（总空隙/水/气相）与 ρ_s 敏感性；表皮阻力情景下的达标时间变化。
"""
import json, sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K    # noqa: E402
from scipy.integrate import solve_ivp     # noqa: E402

T_H = 90 * 3600.0          # 固定积分上限 / s
RTOL = 1e-6
MAX_STEP = 300.0
DT_HIST = 60.0             # 历史采样步长 = 达标判据网格
DT_TAB = 120.0             # 内生 R(t) 表采样步长
RHO_S_SET = (1200.0, 1300.0, 1400.0, 1500.0, 1600.0)
RHO_S_REF = 1400.0
K_CRUST_SET = (0.0, 1.0, 2.0, 4.0)


# ---------- 固定上限求解与历史提取 ----------
def solve_to(model, t_h=T_H, rtol=RTOL, max_step=MAX_STEP, method='BDF'):
    y0 = np.r_[np.full(model.n + 1, 28.), np.full(model.n + 1, 2.55), 0.]
    sol = solve_ivp(model.rhs, (0., t_h), y0, method=method, rtol=rtol, atol=rtol * 0.02,
                    max_step=max_step, dense_output=True, jac=model.jacobian)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def history(model, sol, t_h=T_H, dt=DT_HIST):
    t = np.arange(0.0, t_h + 1e-9, dt)
    Y = sol.sol(t)
    mm = model.n + 1
    C, T = Y[mm:2 * mm], Y[:mm]
    w = 2.0 * model.vol
    return dict(t=t, Cmax=C.max(axis=0), Xbar=w @ C, Ts=T[-1],
                R=np.array([model.R(tt) for tt in t]), T=T, C=C)


def t60_of(h):
    hit = np.flatnonzero(h['Cmax'] < 0.15)
    if not len(hit):
        return None
    k = int(hit[0])
    return dict(t60_s=float(h['t'][k]), t60_h=float(h['t'][k] / 3600.0),
                g_prev=float(h['Cmax'][k - 1] - 0.15) if k else None,
                g_stop=float(h['Cmax'][k] - 0.15), k=k,
                Xbar_end=float(h['Xbar'][k]), R_end_cm=float(h['R'][k] * 100),
                Cmax_end=float(h['Cmax'][k]))


def slope_k(X, ratio):
    """径向线性过初值本构的斜率 k：ratio = 1 + k (X - X0)。"""
    dX = np.asarray(X, float) - K.X0
    return float((dX * (np.asarray(ratio, float) - 1.0)).sum() / (dX ** 2).sum())


def slope_beta(X, ratio):
    """体积线性过初值本构的斜率 beta：(R/R0)^3 = 1 + beta (X - X0)。"""
    dX = np.asarray(X, float) - K.X0
    return float((dX * (np.asarray(ratio, float) ** 3 - 1.0)).sum() / (dX ** 2).sum())


def slope_k2(X, ratio):
    """过初值的二次径向本构：ratio = 1 + k1 dX + k2 dX^2。返回 (k1, k2)。"""
    dX = np.asarray(X, float) - K.X0
    A = np.column_stack([dX, dX ** 2])
    c, *_ = np.linalg.lstsq(A, np.asarray(ratio, float) - 1.0, rcond=None)
    return float(c[0]), float(c[1])


def build_R(X, k=None, beta=None, k2=None):
    """由含水率序列构造 R(t) 序列（非增、限幅）。k2 非空时用二次径向本构。"""
    dX = np.asarray(X, float) - K.X0
    if beta is not None:
        r = np.cbrt(np.clip(1.0 + beta * dX, 1e-9, None))
    elif k2 is not None:
        r = 1.0 + k * dX + k2 * dX ** 2
    else:
        r = 1.0 + k * dX
    r = np.clip(r, 0.2, 1.0)
    return K.R0 * np.minimum.accumulate(r)


# ---------- 收缩本构回归 ----------
def fit_forms(X, ratio):
    x0 = float(X[0]); r0 = float(ratio[0]); dX = X - x0
    out = {}
    k1 = float((dX * (ratio - r0)).sum() / (dX ** 2).sum()); p1 = r0 + k1 * dX
    y = ratio ** 3
    b2 = float((dX * (y - 1.0)).sum() / (dX ** 2).sum()); p2 = np.cbrt(np.clip(1.0 + b2 * dX, 1e-12, None))
    A = np.column_stack([np.ones_like(X), X])
    c3, *_ = np.linalg.lstsq(A, ratio, rcond=None); p3 = A @ c3
    c4, *_ = np.linalg.lstsq(A, y, rcond=None); p4 = np.cbrt(np.clip(A @ c4, 1e-12, None))
    k1q, k2q = slope_k2(X, ratio)
    p5 = 1.0 + k1q * dX + k2q * dX ** 2
    for nm, pred, par, f0 in (
            ('径向线性_过初值', p1, dict(k=k1), lambda x: r0 + k1 * (x - x0)),
            ('径向二次_过初值', p5, dict(k1=k1q, k2=k2q), lambda x: 1 + k1q * (x - x0) + k2q * (x - x0) ** 2),
            ('体积线性_过初值', p2, dict(beta=b2), lambda x: (1.0 + b2 * (x - x0)) ** (1 / 3)),
            ('径向线性_含截距', p3, dict(a=float(c3[0]), b=float(c3[1])), lambda x: c3[0] + c3[1] * x),
            ('体积线性_含截距', p4, dict(a=float(c4[0]), b=float(c4[1])),
             lambda x: (c4[0] + c4[1] * x) ** (1 / 3))):
        rms = float(np.sqrt(((pred - ratio) ** 2).mean()))
        v0 = float(f0(0.0))
        out[nm] = dict(params=par, RMS_R_over_R0=rms, RMS_cm=rms * K.R0 * 100,
                       max_err_cm=float(np.abs(pred - ratio).max() * K.R0 * 100),
                       R2=float(1 - ((pred - ratio) ** 2).sum() / ((ratio - ratio.mean()) ** 2).sum()),
                       R_at_X0_cm=(v0 * K.R0 * 100) if v0 > 0 else None,
                       X0_admissible=bool(v0 > 0))
    return out


def _cbrt_pos(v):
    """(R/R0)^3 = v 的实数解；v<=0 视为该本构在此外推点不可用，返回 nan。"""
    v = np.asarray(v, float)
    out = np.where(v > 0, np.cbrt(np.where(v > 0, v, 1.0)), np.nan)
    return out


def predict_test(X, ratio, frac):
    """前 frac 段拟合、其余段预测。"""
    x0, x_end = float(X[0]), float(X[-1])
    x_split = x0 - frac * (x0 - x_end)
    tr, te = X >= x_split, X < x_split
    ratio_s = ratio[tr]
    k = slope_k(X[tr], ratio_s)
    b = slope_beta(X[tr], ratio_s)
    k1q, k2q = slope_k2(X[tr], ratio_s)
    out = {}
    for nm, f, par in (
            ('径向线性', lambda x: 1.0 + k * (x - x0), dict(k=k)),
            ('径向二次', lambda x: 1.0 + k1q * (x - x0) + k2q * (x - x0) ** 2, dict(k1=k1q, k2=k2q)),
            ('体积线性', lambda x: _cbrt_pos(1.0 + b * (x - x0)), dict(beta=b))):
        pred = f(X[te]) if te.any() else np.array([])
        bad = int(np.isnan(pred).sum()) if te.any() else 0
        ok = ~np.isnan(pred) if te.any() else np.array([], bool)
        rend = float(f(np.array([x_end]))[0])
        out[nm] = dict(fit_frac=frac, X_split=float(x_split), n_fit=int(tr.sum()), n_test=int(te.sum()),
                       params=par,
                       test_RMS_cm=float(np.sqrt(((pred[ok] - ratio[te][ok]) ** 2).mean()) * K.R0 * 100)
                       if ok.any() else None,
                       test_max_cm=float(np.abs(pred[ok] - ratio[te][ok]).max() * K.R0 * 100)
                       if ok.any() else None,
                       n_inadmissible=bad,
                       R_end_pred_cm=rend * K.R0 * 100 if np.isfinite(rend) else None,
                       R_end_true_cm=float(ratio[-1] * K.R0 * 100))
    return out


# ---------- 内生收缩重算 ----------
def resolve_R(n, tab_t, R_arr, label):
    sel = np.unique(np.r_[np.arange(0, len(tab_t), max(1, int(DT_TAB / DT_HIST))), len(tab_t) - 1])
    m = K.TabRModel(n, tab_t[sel], R_arr[sel])
    h = history(m, solve_to(m))
    t = t60_of(h)
    print(f'  [{label}] t60 = {t["t60_h"]:.4f} h  R_end = {t["R_end_cm"]:.4f} cm', flush=True)
    return t, h


def main():
    out = {}
    # ===== 基准（附件2 + 观测环境）=====
    mb = K.S.Model(4, 160, tail='last', mesh='surface', env_interp='pchip')
    hb = history(mb, solve_to(mb))
    tb = t60_of(hb)
    print(f'Q4 基准(附件2+观测环境): t60 = {tb["t60_h"]:.4f} h ({tb["t60_s"]:.0f} s)', flush=True)
    kk = tb['k'] + 1
    out['baseline'] = dict(t60_s=tb['t60_s'], t60_h=tb['t60_h'], g_prev=tb['g_prev'],
                           g_stop=tb['g_stop'], Xbar_end=tb['Xbar_end'],
                           R_end_cm=tb['R_end_cm'], Cmax_end=tb['Cmax_end'])
    Xf, Rf = hb['Xbar'][:kk], hb['R'][:kk]
    ratio = Rf / K.R0

    # ===== D1-a 本构回归 =====
    out['constitutive'] = dict(
        forms=fit_forms(Xf, ratio),
        prediction_tests={f'{f:.1f}': predict_test(Xf, ratio, f) for f in (0.3, 0.4, 0.6)},
        n_points=int(kk))
    print('  本构 RMS(cm):', {k: round(v['RMS_cm'], 5) for k, v in out['constitutive']['forms'].items()},
          flush=True)
    for f, r in out['constitutive']['prediction_tests'].items():
        print(f'  前 {float(f)*100:.0f}% 拟合 -> 后段预测: 径向 RMS={r["径向线性"]["test_RMS_cm"]:.4f} cm '
              f'体积 RMS={r["体积线性"]["test_RMS_cm"]:.4f} cm', flush=True)

    # ===== D1-b 内生收缩 =====
    kk_full = slope_k(Xf, ratio)
    bb_full = slope_beta(Xf, ratio)
    k1q, k2q = slope_k2(Xf, ratio)
    endo = {}
    for nm, kw in (('径向线性_全域标定', dict(k=kk_full)),
                   ('径向二次_全域标定', dict(k=k1q, k2=k2q)),
                   ('体积线性_全域标定', dict(beta=bb_full))):
        Rnew = build_R(hb['Xbar'], **kw)
        t, h = resolve_R(160, hb['t'], Rnew, nm)
        endo[nm] = dict(t60_h=t['t60_h'], delta_h=t['t60_h'] - tb['t60_h'],
                        R_end_cm=t['R_end_cm'], Xbar_end=t['Xbar_end'],
                        max_R_dev_cm=float(np.abs(Rnew[:kk] - Rf).max() * 100),
                        rms_R_dev_cm=float(np.sqrt(((Rnew[:kk] - Rf) ** 2).mean()) * 100),
                        params=kw)
        print(f'    与附件2 的 R 偏差: max={endo[nm]["max_R_dev_cm"]:.5f} cm, '
              f'RMS={endo[nm]["rms_R_dev_cm"]:.5f} cm', flush=True)

    # 预测性情景：只用前 40% 标定的本构，套在基准 Xbar 上重建 R 并重算
    pt40 = out['constitutive']['prediction_tests']['0.4']
    k40 = pt40['径向线性']['params']['k']
    R40 = build_R(hb['Xbar'], k=k40)
    t40, _ = resolve_R(160, hb['t'], R40, '前40%标定本构_径向线性')
    endo['前40%标定_径向线性'] = dict(k=k40, t60_h=t40['t60_h'], delta_h=t40['t60_h'] - tb['t60_h'],
                                     R_end_cm=t40['R_end_cm'],
                                     max_R_dev_cm=float(np.abs(R40[:kk] - Rf).max() * 100),
                                     rms_R_dev_cm=float(np.sqrt(((R40[:kk] - Rf) ** 2).mean()) * 100))
    out['endogenous'] = endo

    # ===== D3 孔隙率演化（双口径 + 相容性诊断）=====
    # 口径 A：只用附录4 的体密度经验式 rho_b(C)=760+90C（自洽，恒非负）
    # 口径 B：只用附件2 的几何收缩 rho_d(t)=rho_d0 (R0/R)^2
    rho_b = 760.0 + 90.0 * hb['Xbar']
    rho_d_geo = K.RHO_D0 * (K.R0 / hb['R']) ** 2
    por = {}
    for rs in RHO_S_SET:
        fA_s, fA_w = rho_b / ((1 + hb['Xbar']) * rs), rho_b * hb['Xbar'] / ((1 + hb['Xbar']) * K.RHO_W)
        fA_g = 1.0 - fA_s - fA_w
        fB_s, fB_w = rho_d_geo / rs, rho_d_geo * hb['Xbar'] / K.RHO_W
        fB_g = 1.0 - fB_s - fB_w
        por[f'{rs:.0f}'] = dict(
            rho_s=rs,
            A_phi_void_0=float(1 - fA_s[0]), A_phi_void_end=float(1 - fA_s[kk - 1]),
            A_phi_water_0=float(fA_w[0]), A_phi_water_end=float(fA_w[kk - 1]),
            A_phi_gas_0=float(fA_g[0]), A_phi_gas_end=float(fA_g[kk - 1]), A_min_phi_gas=float(fA_g[:kk].min()),
            B_phi_void_0=float(1 - fB_s[0]), B_phi_void_end=float(1 - fB_s[kk - 1]),
            B_phi_water_0=float(fB_w[0]), B_phi_water_end=float(fB_w[kk - 1]),
            B_phi_gas_0=float(fB_g[0]), B_phi_gas_end=float(fB_g[kk - 1]), B_min_phi_gas=float(fB_g[:kk].min()),
            rho_d_geo_0=float(rho_d_geo[0]), rho_d_geo_end=float(rho_d_geo[kk - 1]),
            rho_d_app_0=float(rho_b[0] / (1 + hb['Xbar'][0])),
            rho_d_app_end=float(rho_b[kk - 1] / (1 + hb['Xbar'][kk - 1])))
    adm_A = [r for r in RHO_S_SET if por[f'{r:.0f}']['A_min_phi_gas'] > 0]
    adm_B = [r for r in RHO_S_SET if por[f'{r:.0f}']['B_min_phi_gas'] > 0]

    # 相容性诊断：几何收缩 (R/R0) 与 密度式所隐含的 (R/R0)
    R_over_R0_att = hb['R'] / K.R0
    R_over_R0_den = np.sqrt(K.RHO_D0 * (1.0 + hb['Xbar']) / rho_b)
    dev = R_over_R0_att[:kk] - R_over_R0_den[:kk]
    out['porosity'] = dict(
        by_rho_s=por, admissible_rho_s_A=adm_A, admissible_rho_s_B=adm_B, rho_s_ref=RHO_S_REF,
        consistency=dict(
            note='附件2 的几何收缩 (R/R0) 与附录4 密度式 ρ_b=760+90C 在干质量守恒下隐含的 (R/R0) '
                 '= sqrt(ρ_d0(1+C)/ρ_b(C)) 之间的一致性诊断',
            max_abs_dev=float(np.abs(dev).max()),
            max_abs_dev_at_h=float(hb['t'][:kk][int(np.argmax(np.abs(dev)))] / 3600.0),
            RMS_dev=float(np.sqrt((dev ** 2).mean())),
            R_over_R0_attach_end=float(R_over_R0_att[kk - 1]),
            R_over_R0_density_end=float(R_over_R0_den[kk - 1]),
            ratio_end=float(R_over_R0_att[kk - 1] / R_over_R0_den[kk - 1]),
            min_B_phi_gas=float(min(por[f'{r:.0f}']['B_min_phi_gas'] for r in RHO_S_SET)),
        ),
        curve=dict(t_h=(hb['t'][:kk] / 3600).tolist(), Xbar=hb['Xbar'][:kk].tolist(),
                   R_cm=(hb['R'][:kk] * 100).tolist(), rho_b=rho_b[:kk].tolist(),
                   rho_d_geo=rho_d_geo[:kk].tolist(),
                   A_phi_void=(1 - rho_b[:kk] / ((1 + hb['Xbar'][:kk]) * RHO_S_REF)).tolist(),
                   A_phi_water=(rho_b[:kk] * hb['Xbar'][:kk] / ((1 + hb['Xbar'][:kk]) * K.RHO_W)).tolist(),
                   A_phi_gas=(1 - rho_b[:kk] / ((1 + hb['Xbar'][:kk]) * RHO_S_REF)
                              - rho_b[:kk] * hb['Xbar'][:kk] / ((1 + hb['Xbar'][:kk]) * K.RHO_W)).tolist(),
                   B_phi_gas=(1 - rho_d_geo[:kk] / RHO_S_REF
                              - rho_d_geo[:kk] * hb['Xbar'][:kk] / K.RHO_W).tolist(),
                   R_over_R0_attach=R_over_R0_att[:kk].tolist(),
                   R_over_R0_density=R_over_R0_den[:kk].tolist()))
    pc = out['porosity']['curve']
    print(f'  相容性诊断: max|Δ(R/R0)| = {out["porosity"]["consistency"]["max_abs_dev"]:.4f} '
          f'@ {out["porosity"]["consistency"]["max_abs_dev_at_h"]:.1f} h; '
          f'口径B 最小气相孔隙率 = {out["porosity"]["consistency"]["min_B_phi_gas"]:.4f}', flush=True)

    # ===== D3 表皮阻力情景 =====
    crust = {'0': dict(k_crust=0.0, t60_h=tb['t60_h'], delta_h=0.0)}
    for kc in K_CRUST_SET[1:]:
        mc = K.CrustModel(4, 160, (50.165,) * 3, k_crust=kc, c_crust=0.5, use_obs=True,
                          tail='last', mesh='surface', env_interp='pchip')
        tc = t60_of(history(mc, solve_to(mc)))
        crust[f'{kc:.0f}'] = dict(k_crust=kc, c_crust=0.5, t60_h=tc['t60_h'],
                                  delta_h=tc['t60_h'] - tb['t60_h'], Xbar_end=tc['Xbar_end'])
        print(f'  表皮阻力 k_crust={kc}: t60={tc["t60_h"]:.4f} h (Δ={tc["t60_h"]-tb["t60_h"]:+.4f} h)',
              flush=True)
    out['crust'] = crust

    K.jdump(out, 'q4_收缩与孔隙率.json')
    np.savez_compressed(
        K.RES / 'q4_curves.npz',
        t_h=hb['t'][:kk] / 3600.0, Xbar=hb['Xbar'][:kk], R_cm=hb['R'][:kk] * 100,
        rho_b=np.array(pc['rho_b']), rho_d_geo=np.array(pc['rho_d_geo']),
        A_phi_void=np.array(pc['A_phi_void']), A_phi_water=np.array(pc['A_phi_water']),
        A_phi_gas=np.array(pc['A_phi_gas']), B_phi_gas=np.array(pc['B_phi_gas']),
        R_over_R0_attach=np.array(pc['R_over_R0_attach']),
        R_over_R0_density=np.array(pc['R_over_R0_density']))
    print('\n已写出 results/q4_收缩与孔隙率.json 与 q4_curves.npz', flush=True)


if __name__ == '__main__':
    main()
