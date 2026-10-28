# -*- coding: utf-8 -*-
"""汇总关键结果，生成 results/关键结果汇总.md（供论文直接引用）。"""
import json, sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K

q3 = json.loads((K.RES / 'q3_优化结果.json').read_text(encoding='utf-8'))
q4 = json.loads((K.RES / 'q4_收缩与孔隙率.json').read_text(encoding='utf-8'))
L = []
A = L.append


def T3s(r):
    t = r['temps']
    return f'({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})'


A('# 第三、四问创新扩展：关键结果汇总')
A('')
A('> 口径：环境 PCHIP（附件1 全部观测点）+ t>14400 s 保持 50.165 ℃/0.04986；达标判据为全域节点最大')
A('> 干基含水率严格 < 0.15 kg/kg，报告时刻取 60 s 网格 t60；Q4 用 ξ=r/R(t) 材料坐标。')
A('> **本目录的全部扩展内容不替代题目主答案，只作为进一步讨论。**')
A('')

A('## 一、Q3：三段恒温制度的设计与优化')
A('')
A('### 1.1 基准对照（同一判据、同一环境口径）')
A('')
b = q3['baseline']
A('| 制度 | 达标时间 t60 / h | 说明 |')
A('|---|---|---|')
A(f'| **附件1 观测环境（题面口径）** | **{b["q3_t60_h"]:.4f}** | {b["q3_t60_h"]*3600:.0f} s，即题目主答案 |')
m = {r['src']: r for r in q3['pareto_verified']}
for src, lab in (('const_50p165', '全程恒温 50.165 ℃'), ('obs_upper', '全程恒温 50.246 ℃（观测上限）'),
                 ('corner_hot', '全程恒温 70 ℃（工艺上界）'), ('corner_cold', '全程恒温 40 ℃（工艺下界）')):
    if src in m:
        A(f'| {lab} | {m[src]["t60_h"]:.4f} | 同为三段制度族内的退化情形 |')
A('')
A(f'注：三段切换时点固定为 {q3["meta"]["switch_h"][0]:.0f} h 与 {q3["meta"]["switch_h"][1]:.0f} h；')
A(f'温度优化区间 {q3["meta"]["box"][0]:.0f}–{q3["meta"]["box"][1]:.0f} ℃（超出附件1 观测范围，属工艺扩展假设）。')
A('')
_bl = [r for r in q3['pareto_verified'] if r['src'] == 'const_50p165']
if _bl:
    A(f'题面基准的三项指标（用于后续对比）：t60 = {b["q3_t60_h"]:.4f} h、'
      f'SEC = {b["q3_SEC_MJ"]:.4f} MJ/kg、Q_ret = {b["q3_Qret"]:.4f}。')
    A('')

A('### 1.2 代理模型与 DOE')
A('')
A(f'- DOE：Sobol′ 序列 {q3["meta"]["doe_n"] - 8} 点 + 8 个结构化点（含角点与恒温点），共 {q3["meta"]["doe_n"]} 点，'
  f'全部达标、无失败；每组用 n={q3["meta"]["n_run"]}、rtol={q3["meta"]["rtol"]:g} 求解。')
A('- 代理：3 阶正交 Legendre 多项式混沌（3 变量、20 项）。')
A('')
A('| 响应 | 训练 R² | 结构点留出 RMSE | 5 折交叉验证 RMSE |')
A('|---|---|---|---|')
nm_map = {'t60_h': ('达标时间 t60 / h', 'h'), 'SEC_MJ': ('单位去水量供热量 / MJ·kg⁻¹', 'MJ/kg'),
          'dose': ('降解剂量 D', '')}
for kk, (lab, un) in nm_map.items():
    s = q3['surrogate'][kk]
    A(f'| {lab} | {s["r2"]:.6f} | {s["test_rmse"]:.5f} {un} | {s["cv5_rmse"]:.5f} {un} |')
A('')
A('代理在 8000 个候选点上给出 542 个非支配点；抽稀 14 点 + 5 个附加点，'
  f'**全部用真模型（n={q3["meta"]["verify_n"]}、rtol={q3["meta"]["verify_rtol"]:g}）回代复核**，下表只用复核值。')
A('')

A('### 1.3 真模型复核后的 Pareto 前沿')
A('')
A(f'| T1 / ℃ | T2 / ℃ | T3 / ℃ | t60 / h | SEC / MJ·kg⁻¹ | SEC+λ / MJ·kg⁻¹ | Q_ret | 峰值热流 / W·m⁻² | 物料最高温 / ℃ |')
A('|---|---|---|---|---|---|---|---|---|')
_full = {tuple(round(x, 6) for x in r['temps']): r for r in q3['pareto_verified']}
for r in sorted(q3['pareto_verified_front'], key=lambda x: x['t60_h']):
    t = r['temps']
    fr = _full.get(tuple(round(x, 6) for x in t), r)
    A(f'| {t[0]:.2f} | {t[1]:.2f} | {t[2]:.2f} | {r["t60_h"]:.4f} | {r["SEC_MJ"]:.4f} | '
      f'{r["SEC_MJ"] + q3["lam_vap_J_per_kg"] / 1e6:.4f} | {r["Q_ret"]:.4f} | '
      f'{fr.get("qmax_W_m2", float("nan")):.1f} | {fr.get("Tbar_max_C", float("nan")):.2f} |')
A('')
A(f'注：λ = {q3["lam_vap_J_per_kg"] / 1e6:.3f} MJ/kg 为 50 ℃ 附近水的汽化潜热。本模型不含潜热项，'
  'SEC 只统计**对流供入物料的热量**，故 SEC+λ 才是与常规"单位能耗"可比的口径；'
  '由于所有制度的去水量几乎相同，SEC 的排序与 SEC+λ 一致。')
A('')

A('### 1.4 决策推荐（指标越小越好）')
A('')
A('| 推荐 | T1 / ℃ | T2 / ℃ | T3 / ℃ | t60 / h | SEC / MJ·kg⁻¹ | Q_ret | 相对题面基准 |')
A('|---|---|---|---|---|---|---|---|')
lab_map = {'min_time': '最速（时间最优）', 'min_energy': '最省能', 'max_quality': '最优品质',
           'topsis': 'TOPSIS 折中'}
for kk, lab in lab_map.items():
    r = q3['recommend'][kk]
    if not isinstance(r, dict) or 'temps' not in r:
        continue
    d = r['t60_h'] - b['q3_t60_h']
    A(f'| {lab} | {r["temps"][0]:.2f} | {r["temps"][1]:.2f} | {r["temps"][2]:.2f} | {r["t60_h"]:.4f} | '
      f'{r["SEC_MJ"]:.4f} | {r["Q_ret"]:.4f} | {d:+.4f} h（{d / b["q3_t60_h"] * 100:+.2f}%） |')
A('')
w = q3['recommend']['topsis_weights']
A(f'TOPSIS 熵权：t60 {w["t60"]:.3f}、SEC {w["SEC"]:.3f}、dose {w["dose"]:.3f}'
  f'（贴近度 {q3["recommend"]["topsis_closeness"]:.4f}）。')
A('')
_rt = q3['recommend']['topsis']
A(f'**TOPSIS 折中点的意义**：相对题面基准 t60 只增加 {_rt["t60_h"] - b["q3_t60_h"]:+.4f} h'
  f'（{(_rt["t60_h"] - b["q3_t60_h"]) / b["q3_t60_h"] * 100:+.2f}%），'
  f'SEC 从 {b["q3_SEC_MJ"]:.4f} 降到 {_rt["SEC_MJ"]:.4f} MJ/kg'
  f'（−{(1 - _rt["SEC_MJ"] / b["q3_SEC_MJ"]) * 100:.1f}%），'
  f'Q_ret 为 {_rt["Q_ret"]:.4f}（基准 {b["q3_Qret"]:.4f}）。'
  '也就是说：在几乎不牺牲时长与品质的前提下，把烘房前 8 h 降到约 40 ℃、8–24 h 升到约 60 ℃、'
  '24 h 后回到约 45 ℃，可显著降低物料侧供热强度与峰值热流（'
  f'{_rt["qmax_W_m2"]:.0f} W/m² vs 基准的 '
  f'{[r for r in q3["pareto_verified"] if r["src"] == "const_50p165"][0]["qmax_W_m2"]:.0f} W/m²）。**')
A('')

A('### 1.5 品质动力学参数 Ea 的稳健性')
A('')
A('| Ea / kJ·mol⁻¹ | TOPSIS 推荐制度 | t60 / h | Q_ret | 前沿点数 |')
A('|---|---|---|---|---|')
for kk in sorted(q3['ea_robustness'], key=float):
    v = q3['ea_robustness'][kk]
    t = v['topsis_temps']
    A(f'| {float(kk)/1000:.0f} | ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}) | {v["topsis_t60_h"]:.4f} | '
      f'{v["topsis_Qret"]:.4f} | {v["n_front"]} |')
A('')
A('**要点：推荐制度随 Ea 变化（Ea 越大越应选低温长时），而 Ea 来自文献量级、未标定，'
  '因此 1.4 的推荐只在给定 Ea 下成立。**')
A('')

A('## 二、Q4：收缩本构、内生化与孔隙率')
A('')
A('### 2.1 基准与收缩本构回归')
A('')
A(f'基准（附件2 半径 + 题面观测环境）：t60 = {q4["baseline"]["t60_h"]:.4f} h'
  f'（{q4["baseline"]["t60_s"]:.0f} s）；末期 X̄ = {q4["baseline"]["Xbar_end"]:.4f} kg/kg；'
  f'R_end = {q4["baseline"]["R_end_cm"]:.4f} cm；g(t60−60) = {q4["baseline"]["g_prev"]:+.2e}，'
  f'g(t60) = {q4["baseline"]["g_stop"]:+.2e}。')
A('')
A('| 本构形式 | RMS / cm | 最大偏差 / cm | R² | X=0 外推 R / cm | 参数 |')
A('|---|---|---|---|---|---|')
for kk, v in q4['constitutive']['forms'].items():
    par = ', '.join(f'{a}={b:.6f}' for a, b in v['params'].items())
    r0 = v['R_at_X0_cm']
    A(f'| {kk.replace("_", " ")} | {v["RMS_cm"]:.5f} | {v["max_err_cm"]:.5f} | {v["R2"]:.6f} | '
      f'{("—" if r0 is None else f"{r0:.4f}")} | {par} |')
A('')
A('**径向二次（过初值）本构显著优于线性与体积线性：RMS 0.00684 cm vs 0.0538 / 0.0996 cm。**')
A('')
A('### 2.2 本构外推预测检验（只用前段标定，预测后段）')
A('')
A('| 标定区间 | 本构 | 后段预测 RMS / cm | 后段最大偏差 / cm | 末端 R 预测 / cm | 末端 R 真值 / cm |')
A('|---|---|---|---|---|---|')
for f in sorted(q4['constitutive']['prediction_tests'], key=float):
    for kk, v in q4['constitutive']['prediction_tests'][f].items():
        rp = v['R_end_pred_cm']
        A(f'| 前 {float(f)*100:.0f}% | {kk} | {v["test_RMS_cm"]:.4f} | {v["test_max_cm"]:.4f} | '
          f'{("不可用" if rp is None else f"{rp:.4f}")} | {v["R_end_true_cm"]:.4f} |')
A('')
A('**线性收缩本构不具备外推能力**（前 40% 标定后，后段 RMS 达 0.43 cm，末端半径预测 0.71 cm 而真值 1.20 cm，'
  '低估 41%）；二次本构把后段 RMS 降到 0.23 cm，但末端半径反而高估到 1.47 cm。'
  '**二者都说明：不能靠早期数据标定的收缩本构去预测后期几何。**')
A('')
A('### 2.3 把附件2 换成内生本构后的达标时间')
A('')
A('| 收缩口径 | t60 / h | 相对附件2 口径 Δ / h | 末期 R / cm | 与附件2 的最大 R 偏差 / cm |')
A('|---|---|---|---|---|')
A(f'| 附件2 半径（题面口径） | {q4["baseline"]["t60_h"]:.4f} | 0.0000 | '
  f'{q4["baseline"]["R_end_cm"]:.4f} | — |')
for kk, v in q4['endogenous'].items():
    A(f'| {kk.replace("_", " ")} | {v["t60_h"]:.4f} | {v["delta_h"]:+.4f} | {v["R_end_cm"]:.4f} | '
      f'{v["max_R_dev_cm"]:.5f} |')
A('')
A('**要点：以"径向二次"本构代替附件2，达标时间变化为 0.0000 h；线性本构为 −0.12 h（−7 min）；'
  '体积线性为 −1.48 h；而只用前 40% 标定的线性本构会把时长低估 23.55 h（−46%）。**')
A('')
A('### 2.4 孔隙率演化与两套经验闭合的相容性')
A('')
c = q4['porosity']['consistency']
A(f"- 口径A（只用附录4 ρ_b=760+90C）：自洽，气相孔隙率恒非负；ρ_s ∈ {{{', '.join(f'{r:.0f}' for r in q4['porosity']['admissible_rho_s_A'])}}} kg/m³ 均可用。")
A(f"- 口径B（只用附件2 的 R(t)）：最小气相孔隙率 {c['min_B_phi_gas']:.4f} < 0，**物理上不可能**。")
A(f"- 相容性诊断：附件2 的 R/R0 与附录4 密度式在干质量守恒下隐含的 R/R0 最大相差 **{c['max_abs_dev']:.4f}**"
  f"（出现在 {c['max_abs_dev_at_h']:.1f} h），RMS {c['RMS_dev']:.4f}；"
  f"末期两者分别为 {c['R_over_R0_attach_end']:.4f} 与 {c['R_over_R0_density_end']:.4f}，比值 {c['ratio_end']:.4f}。")
A('')
A('**结论：附件2 的几何收缩量与附录4 的密度经验式不能同时严格成立——观测收缩明显快于密度式所允许的体积收缩。'
  '这是本题两套经验闭合之间的内在张力，论文应显式说明，且不应据此反推任一经验式为"错"。**')
A('')
A('### 2.5 表皮阻力（case hardening）情景')
A('')
A('表面传质系数按 h_m,eff = h_m / [1 + K_c·max(0, 1 − C_s/0.5)] 衰减（C_crust = 0.5 kg/kg）。')
A('')
A('| K_c | t60 / h | Δt60 / h | 末期 X̄ / kg·kg⁻¹ |')
A('|---|---|---|---|')
for kk, v in q4['crust'].items():
    A(f'| {kk} | {v["t60_h"]:.4f} | {v["delta_h"]:+.4f} | {v.get("Xbar_end", q4["baseline"]["Xbar_end"]):.4f} |')
A('')
A(f'**表皮阻力使 Q4 达标时间最多延长 {max(v["delta_h"] for v in q4["crust"].values()):.4f} h'
  f'（+{max(v["delta_h"] for v in q4["crust"].values()) / q4["baseline"]["t60_h"] * 100:.2f}%），'
  '量级小于温度制度的影响，但方向明确。**')
A('')

A('## 三、必须随结果一起声明的局限')
A('')
A('1. 温度优化区间 40–70 ℃ 超出附件1 的观测范围（28–50.246 ℃），附录3 的经验物性式在该区间**未标定**；')
A('   所有扩展结果都属"若允许调温"的情景，不替代 Q3 主答案。')
A('2. 品质降解动力学（k_ref = 0.02 h⁻¹ @60 ℃、Ea = 50 kJ/mol 基准）取自文献量级，**未用本题数据标定**；')
A('   Q_ret 只应作相对排序，1.5 节已给出 Ea 敏感性。')
A('3. SEC 只统计对流供入物料的热量，本模型不含潜热项，故另列 SEC+λ；不可当作真实单位能耗。')
A('4. 温度制度改变时，空气含湿量仍沿用附件1 的观测序列（未随温度耦合），这是显式假设。')
A('5. 固体真密度 ρ_s 未标定（孔隙率） ；附件2 收缩与附录4 密度式不相容（2.4 节）。')
A('6. 三段恒温、切换时点固定在 8 h / 24 h 是设计选择，未对切换时点做优化。')
A('')

(K.RES / '关键结果汇总.md').write_text('\n'.join(L), encoding='utf-8')
print('已写出 results/关键结果汇总.md')
print('\n'.join(L[:6]))
