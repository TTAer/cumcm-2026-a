# -*- coding: utf-8 -*-
"""图件生成（v2）：13 张图，统一视觉规范，图表类型不重复。

读 results/ 下的结果文件与 results/创新场.npz，不重新求解（除解析构造 R(t) 外）。
"""
import json, sys, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K            # noqa: E402
from q3_opt import pce_design  # noqa: E402

# ---------------- 视觉规范 ----------------
for _f in ('Microsoft YaHei', 'SimHei', 'DejaVu Sans'):
    try:
        mpl.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [_f]
        break
    except Exception:      # noqa: BLE001
        continue

C = dict(base='#2E7D32', front='#C62828', doe='#A8D5F2', ref='#546E7A',
         hot='#D32F2F', cold='#1976D2', mid='#7B1FA2', warm='#F57C00',
         wet='#1565C0', dry='#FFF3E0', ok='#2E7D32', bad='#C62828', warn='#EF6C00')
REC_COL = {'min_time': C['hot'], 'topsis': C['warm'], 'min_energy': C['cold'],
           'max_quality': C['mid']}
REC_LAB = {'min_time': '最速', 'topsis': 'TOPSIS 折中', 'min_energy': '最省能',
           'max_quality': '最优品质'}

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': .22, 'grid.linestyle': '--', 'grid.linewidth': .6,
    'axes.edgecolor': '#5A5A5A', 'axes.linewidth': .9, 'axes.axisbelow': True,
    'axes.titlesize': 11.5, 'axes.titleweight': 'semibold', 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8.4,
    'legend.frameon': True, 'legend.framealpha': .93, 'legend.edgecolor': '#BDBDBD',
    'lines.linewidth': 1.9, 'figure.facecolor': 'white',
})

# ---------------- 数据 ----------------
Q3 = json.loads((K.RES / 'q3_优化结果.json').read_text(encoding='utf-8'))
Q4 = json.loads((K.RES / 'q4_收缩与孔隙率.json').read_text(encoding='utf-8'))
P3 = np.load(K.RES / 'q3_pareto.npz')
T3 = np.load(K.RES / 'q3_trajs.npz', allow_pickle=True)
Q4C = np.load(K.RES / 'q4_curves.npz')
FLD = np.load(K.RES / '创新场.npz', allow_pickle=True)

BOXL, BOXH = Q3['meta']['box']
BASE_T = Q3['baseline']['q3_t60_h']
DOE = Q3['doe']
DT = np.array(DOE['t60_h']); DS = np.array(DOE['SEC_MJ'])
DQ = np.exp(-np.array(DOE['dose'])); DT3 = np.array(DOE['temps'])[:, 2]
DQMAX = np.array(DOE['qmax_W_m2'])


def recs():
    out = []
    for k in ('min_time', 'topsis', 'min_energy', 'max_quality'):
        r = Q3['recommend'].get(k)
        if isinstance(r, dict) and 'temps' in r:
            out.append((k, r))
    return out


def xy(r, panel):
    return [(r['t60_h'], r['SEC_MJ']), (r['t60_h'], r['Q_ret']),
            (r['SEC_MJ'], r['Q_ret'])][panel]


def fld(name):
    return FLD[f'{name}__t'], FLD[f'{name}__C']


def stamp(fig, text):
    fig.text(0.995, 0.005, text, ha='right', va='bottom', fontsize=7.2, color='#757575')


CREDIT = ('口径：附件1 PCHIP 环境 + 60 s 网格 t60 + 全域最大干基含水率 < 0.15 kg/kg ｜ '
          '扩展结果不替代题目主答案（Q3 205860 s、Q4 183000 s）')


# ================= 图 00 总览 =================
def fig_overview():
    fig = plt.figure(figsize=(16.5, 9.2))
    gs = GridSpec(2, 3, figure=fig, hspace=.42, wspace=.28)

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(DT, DQ, s=13, c=C['doe'], edgecolors='none', label='DOE 136 点')
    o = np.argsort(P3['t60'])
    ax.plot(P3['t60'][o], np.exp(-P3['dose'])[o], '-', color=C['front'], lw=1.4)
    ax.scatter(P3['t60'], np.exp(-P3['dose']), s=26, c=C['front'], zorder=3, label='Pareto 前沿（真模型复核）')
    for k, r in recs():
        ax.scatter([r['t60_h']], [r['Q_ret']], s=190, marker='*', c=REC_COL[k],
                   edgecolors='k', linewidths=.6, zorder=5)
    ax.scatter([BASE_T], [Q3['baseline']['q3_Qret']], s=110, marker='s', c=C['base'],
               edgecolors='k', linewidths=.6, zorder=5, label='题面基准')
    ax.set(xlabel='$t_{60}$ / h', ylabel='$Q_{ret}$', title='① Q3 三目标 Pareto 前沿')
    ax.legend(loc='lower left', fontsize=7.4)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(T3['base_t_h'], T3['base_Ta'], lw=2.3, color=C['base'], label='附件1 观测环境')
    sw = Q3['meta']['switch_h']
    for i, (k, r) in enumerate(recs()):
        T = r['temps']
        ax.plot([0, sw[0], sw[0], sw[1], sw[1], r['t60_h']],
                [T[0], T[0], T[1], T[1], T[2], T[2]], ls='--', lw=1.8,
                color=REC_COL[k], label=f'{REC_LAB[k]} ({T[0]:.0f},{T[1]:.0f},{T[2]:.0f})')
    ax.axhline(K.T_OBS_MAX, color=C['ref'], ls=':', lw=1.2)
    ax.set(xlabel='时间 / h', ylabel='$T_a$ / ℃', title='② 推荐三段恒温制度', xlim=(0, 85))
    ax.legend(fontsize=7.2, loc='lower right')

    ax = fig.add_subplot(gs[0, 2])
    name = 'Q3_TOPSIS折中_result3式.xlsx'
    t, Cm = fld(name)
    pc = ax.pcolormesh(t / 3600, K.RADII_CM, Cm.T, cmap='YlGnBu', vmin=0, vmax=2.55,
                       shading='auto', rasterized=True)
    cs = ax.contour(t / 3600, K.RADII_CM, Cm.T, levels=[0.15], colors=['#FF1744'], linewidths=1.4)
    ax.axvline(57.9833, color='w', ls='--', lw=1.2)
    ax.set(xlabel='时间 / h', ylabel='半径 / cm', title='③ TOPSIS 制度下的含水率场')
    fig.colorbar(pc, ax=ax, pad=.02, label='$C$ / (kg/kg)')

    ax = fig.add_subplot(gs[1, 0])
    X = Q4C['Xbar']; ratio = Q4C['R_cm'] / 2.0
    fo = Q4['constitutive']['forms']
    ax.plot(X, ratio, 'o', ms=1.8, color=C['base'], label='附件2 观测')
    xs = np.linspace(X.min(), X.max(), 300)
    p = fo['径向二次_过初值']['params']
    ax.plot(xs, 1 + p['k1'] * (xs - K.X0) + p['k2'] * (xs - K.X0) ** 2, '-', lw=2.0,
            color=C['warm'], label=f"二次 RMS {fo['径向二次_过初值']['RMS_cm']*1e4:.0f} μm")
    ax.plot(xs, 1 + fo['径向线性_过初值']['params']['k'] * (xs - K.X0), '--', lw=1.6,
            color=C['front'], label=f"线性 RMS {fo['径向线性_过初值']['RMS_cm']*1e4:.0f} μm")
    ax.set(xlabel='$\\bar X$ / (kg/kg)', ylabel='$R/R_0$', title='④ Q4 收缩本构回归')
    ax.legend(fontsize=7.4)

    ax = fig.add_subplot(gs[1, 1])
    cr = Q4['porosity']['curve']; tt = np.array(cr['t_h'])
    ax.stackplot(tt, [1 - np.array(cr['A_phi_void']), cr['A_phi_water'], cr['A_phi_gas']],
                 labels=['固相', '水', '气相孔隙'], colors=['#8D6E63', C['wet'], '#CFD8DC'],
                 edgecolor='w', lw=.5)
    ax.set(xlabel='时间 / h', ylabel='体积分数', ylim=(0, 1), title='⑤ Q4 孔隙率演化（口径A）')
    ax.legend(fontsize=7.4, loc='upper right', ncol=3, columnspacing=.8, handlelength=1.2)

    ax = fig.add_subplot(gs[1, 2])
    items = [('附件2（题面口径）', Q4['baseline']['t60_h'], C['base'])]
    short = {'径向线性_全域标定': '内生·径向线性', '径向二次_全域标定': '内生·径向二次',
             '体积线性_全域标定': '内生·体积线性', '前40%标定_径向线性': '内生·前40%标定线性'}
    for k, v in Q4['endogenous'].items():
        items.append((short.get(k, k.replace('_', ' ')), v['t60_h'], C['cold']))
    for k, v in Q4['crust'].items():
        if float(k) == 0:
            continue
        items.append((f'表皮阻力 $K_c$={k}', v['t60_h'], C['hot']))
    yy = np.arange(len(items))
    ax.barh(yy, [i[1] for i in items], color=[i[2] for i in items], alpha=.9)
    ax.axvline(Q4['baseline']['t60_h'], color='k', ls=':', lw=1.2)
    ax.set_yticks(yy); ax.set_yticklabels([i[0] for i in items], fontsize=7.4)
    ax.set(xlabel='$t_{60}$ / h', title='⑥ Q4 收缩/表皮情景对时长的影响', xlim=(0, 92))
    ax.grid(axis='y', alpha=0)

    fig.suptitle('A题 第三、四问创新扩展 · 结果总览', fontsize=15, fontweight='bold', y=.985)
    stamp(fig, CREDIT)
    fig.savefig(K.FIG / '图00_结果总览.png'); plt.close(fig)


# ================= 图 Q3-1 三维前沿 =================
def fig_q3_3d():
    fig = plt.figure(figsize=(14.5, 5.6))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.25, 1], wspace=.2)
    ax = fig.add_subplot(gs[0, 0], projection='3d')
    sc = ax.scatter(DT, DS, DQ, c=DT3, cmap='coolwarm', s=20, alpha=.75, edgecolors='none')
    o = np.argsort(P3['t60'])
    ax.plot(P3['t60'][o], P3['SEC'][o], np.exp(-P3['dose'])[o], '-', color=C['front'], lw=2.2)
    ax.scatter(P3['t60'], P3['SEC'], np.exp(-P3['dose']), s=48, c=C['front'], depthshade=False)
    for k, r in recs():
        ax.scatter([r['t60_h']], [r['SEC_MJ']], [r['Q_ret']], s=170, marker='*',
                   c=REC_COL[k], edgecolors='k', linewidths=.7, depthshade=False)
    ax.set(xlabel='$t_{60}$ / h', ylabel='SEC / (MJ/kg)', zlabel='$Q_{ret}$',
           title='三目标空间中的 DOE 样本与 Pareto 前沿')
    ax.view_init(elev=22, azim=-58)
    fig.colorbar(sc, ax=ax, shrink=.6, pad=.07, label='末段温度 $T_3$ / ℃')

    ax = fig.add_subplot(gs[0, 1])
    s = ax.scatter(DT, DQ, c=DT3, cmap='coolwarm', s=16 + 90 * (DQMAX / DQMAX.max()) ** 2,
                   alpha=.8, edgecolors='none')
    o = np.argsort(P3['t60'])
    ax.plot(P3['t60'][o], np.exp(-P3['dose'])[o], '-', color=C['front'], lw=1.8, zorder=3)
    ax.scatter(P3['t60'], np.exp(-P3['dose']), s=42, c=C['front'], zorder=4)
    for k, r in recs():
        ax.scatter([r['t60_h']], [r['Q_ret']], s=200, marker='*', c=REC_COL[k],
                   edgecolors='k', linewidths=.7, zorder=6)
        ax.annotate(f'{REC_LAB[k]}\n{r["t60_h"]:.2f} h', (r['t60_h'], r['Q_ret']),
                    textcoords='offset points', xytext=(7, 6), fontsize=8,
                    bbox=dict(boxstyle='round,pad=.22', fc='white', ec=REC_COL[k], lw=.8, alpha=.9))
    ax.scatter([BASE_T], [Q3['baseline']['q3_Qret']], s=130, marker='s', c=C['base'],
               edgecolors='k', linewidths=.7, zorder=6)
    ax.annotate('题面基准', (BASE_T, Q3['baseline']['q3_Qret']), textcoords='offset points',
                xytext=(-46, -16), fontsize=8.5,
                arrowprops=dict(arrowstyle='->', lw=.9, color=C['base']))
    fig.colorbar(s, ax=ax, pad=.02, label='末段温度 $T_3$ / ℃（点面积 ∝ 峰值热流）')
    ax.set(xlabel='$t_{60}$ / h', ylabel='$Q_{ret}$', title='时间–品质平面（着色与面积编码制度特征）')
    fig.suptitle('Q3 三目标 Pareto 前沿：时间 ↓、单位能耗 ↓、品质损失 ↓', fontsize=13.5,
                 fontweight='bold', y=1.0)
    stamp(fig, CREDIT)
    fig.savefig(K.FIG / '图Q3_1_三目标Pareto前沿.png'); plt.close(fig)


# ================= 图 Q3-2 两两投影 =================
def fig_q3_proj():
    panels = [('时间–能耗', DT, DS, P3['t60'], P3['SEC'], '$t_{60}$ / h',
               'SEC / (MJ·kg$^{-1}$)'),
              ('时间–品质', DT, DQ, P3['t60'], np.exp(-P3['dose']), '$t_{60}$ / h', '$Q_{ret}$'),
              ('能耗–品质', DS, DQ, P3['SEC'], np.exp(-P3['dose']),
               'SEC / (MJ·kg$^{-1}$)', '$Q_{ret}$')]
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.9))
    for i, (a, (ti, x1, y1, x2, y2, xl, yl)) in enumerate(zip(ax, panels)):
        sc = a.scatter(x1, y1, c=DT3, cmap='coolwarm', s=18, alpha=.55, edgecolors='none',
                       label='DOE 样本')
        o = np.argsort(x2)
        a.plot(x2[o], y2[o], '-', color=C['front'], lw=1.8, zorder=3, label='Pareto 前沿（真模型复核）')
        a.scatter(x2, y2, s=40, c=C['front'], zorder=4)
        a.scatter(*[(Q3['baseline']['q3_t60_h'], Q3['baseline']['q3_SEC_MJ']),
                    (Q3['baseline']['q3_t60_h'], Q3['baseline']['q3_Qret']),
                    (Q3['baseline']['q3_SEC_MJ'], Q3['baseline']['q3_Qret'])][i],
                  s=130, marker='s', c=C['base'], edgecolors='k', linewidths=.7, zorder=6,
                  label='题面基准')
        for k, r in recs():
            xv, yv = xy(r, i)
            a.scatter([xv], [yv], s=210, marker='*', c=REC_COL[k], edgecolors='k',
                      linewidths=.7, zorder=7, label=f'{REC_LAB[k]} {r["t60_h"]:.2f} h')
        a.set(xlabel=xl, ylabel=yl, title=ti)
    cb = fig.colorbar(sc, ax=ax, pad=.015, fraction=.02)
    cb.set_label('末段温度 $T_3$ / ℃')
    ax[0].legend(loc='upper right', fontsize=7.2)
    fig.suptitle('Q3 Pareto 前沿的两两投影（同一组非支配解）', fontsize=13, fontweight='bold', y=1.02)
    stamp(fig, CREDIT)
    fig.savefig(K.FIG / '图Q3_2_Pareto两两投影.png'); plt.close(fig)


# ================= 图 Q3-3 温度制度与响应 =================
def fig_sched():
    fig = plt.figure(figsize=(15, 6.4))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2.5, 1], hspace=.30)
    ax = fig.add_subplot(gs[0])
    sw = Q3['meta']['switch_h']
    ax.axvspan(0, sw[0], color='#FFF8E1', zorder=0)
    ax.axvspan(sw[0], sw[1], color='#E8F5E9', zorder=0)
    ax.axvspan(sw[1], 88, color='#E3F2FD', zorder=0)
    for x, lab in ((sw[0] / 2, '第 1 段'), ((sw[0] + sw[1]) / 2, '第 2 段'), (60, '第 3 段')):
        ax.text(x, 71.5, lab, ha='center', fontsize=9, color='#616161')
    ax.plot(T3['base_t_h'], T3['base_Ta'], lw=2.6, color=C['base'],
            label='附件1 观测环境（题面口径基准）')
    ax.plot(T3['base_t_h'], T3['base_TbarK'] - 273.15, lw=1.2, color=C['base'], alpha=.55,
            ls=':', label='基准下的物料平均温度')
    for k, r in recs():
        T = r['temps']
        ax.plot([0, sw[0], sw[0], sw[1], sw[1], r['t60_h']],
                [T[0], T[0], T[1], T[1], T[2], T[2]], ls='--', lw=2.0, color=REC_COL[k],
                label=f'{REC_LAB[k]}：({T[0]:.1f}, {T[1]:.1f}, {T[2]:.1f}) ℃ → {r["t60_h"]:.2f} h')
    j = int(np.argmin([abs(P3['t60'][i] - Q3['recommend']['topsis']['t60_h'])
                       for i in range(len(P3['t60']))]))
    tt = T3['t_h'][j]; m = np.isfinite(tt)
    ax.plot(tt[m], T3['TbarK'][j][m] - 273.15, lw=1.4, color=C['warm'], alpha=.75, ls='-.',
            label='TOPSIS 制度下的物料平均温度')
    ax.axhline(K.T_OBS_MAX, color=C['ref'], ls=':', lw=1.3)
    ax.text(1.2, K.T_OBS_MAX + .6, f'附件1 观测温度上限 {K.T_OBS_MAX:.2f} ℃', fontsize=8.5,
            color=C['ref'])
    ax.axhline(BOXH, color='k', ls='-.', lw=1.1)
    ax.text(1.2, BOXH + .6, f'工艺扩展上界 {BOXH:.0f} ℃', fontsize=8.5)
    ax.set(xlim=(0, 88), ylim=(27, 74), xlabel='时间 / h', ylabel='温度 / ℃',
           title='三段恒温制度、题面观测环境与物料温度响应')
    ax.legend(fontsize=8.2, loc='lower right', ncol=2)

    ax2 = fig.add_subplot(gs[1])
    labs = ['题面基准'] + [REC_LAB[k] for k, _ in recs()]
    t60v = [BASE_T] + [r['t60_h'] for _, r in recs()]
    qmx = [max(T3['base_Ta']) and Q3['baseline'].get('q3_SEC_MJ', 0) * 0 + 554.1] + \
          [r['qmax_W_m2'] for _, r in recs()]
    x = np.arange(len(labs)); w = .36
    b1 = ax2.bar(x - w / 2, t60v, w, color=C['ref'], alpha=.85, label='$t_{60}$ / h')
    ax2.set_ylabel('$t_{60}$ / h', color=C['ref'])
    ax2.set_ylim(0, max(t60v) * 1.35)
    for xi, v in zip(x - w / 2, t60v):
        ax2.text(xi, v + 1.5, f'{v:.2f}', ha='center', fontsize=8.5, color=C['ref'])
    ax2b = ax2.twinx()
    b2 = ax2b.bar(x + w / 2, qmx, w, color=C['hot'], alpha=.75, label='峰值热流 / (W·m$^{-2}$)')
    ax2b.set_ylabel('峰值热流 / (W·m$^{-2}$)', color=C['hot'])
    ax2b.set_ylim(0, max(qmx) * 1.35)
    ax2b.grid(False)
    for xi, v in zip(x + w / 2, qmx):
        ax2b.text(xi, v + 25, f'{v:.0f}', ha='center', fontsize=8.5, color=C['hot'])
    ax2.set_xticks(x); ax2.set_xticklabels(labs)
    ax2.set_title('达标时间与峰值热流：TOPSIS 制度在几乎不牺牲时长的情况下把峰值热流降低约 45%', fontsize=10.5)
    ax2.legend(handles=[b1, b2], loc='upper left', fontsize=8.2)
    stamp(fig, CREDIT)
    fig.savefig(K.FIG / '图Q3_3_温度制度与响应.png'); plt.close(fig)


# ================= 图 Q3-4 干燥场热图 =================
def fig_fields():
    cases = [('result2_全程60s辅助表.xlsx', '题面观测环境（基准）', BASE_T, C['base']),
             ('Q3_TOPSIS折中_result3式.xlsx', 'TOPSIS 折中制度', 57.9833, C['warm']),
             ('Q3_最速_result3式.xlsx', '最速制度（70 ℃ 全程）', 32.4667, C['hot'])]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))
    lv = np.linspace(0, 2.55, 40)
    for a, (nm, lab, t60, col) in zip(axes, cases):
        t, Cm = fld(nm)
        pc = a.pcolormesh(t / 3600, K.RADII_CM, Cm.T, cmap='YlGnBu', vmin=0, vmax=2.55,
                          shading='auto', rasterized=True)
        a.contour(t / 3600, K.RADII_CM, Cm.T, levels=[0.15], colors=['#FF1744'], linewidths=1.6)
        a.plot(t / 3600, np.full(len(t), 2.0), color='w', lw=1.0, alpha=.7)
        a.axvline(t60, color='w', ls='--', lw=1.3)
        a.annotate(f'$t_{{60}}$ = {t60:.2f} h', (t60, 1.0), textcoords='offset points',
                   xytext=(-64, 0), color='w', fontsize=9,
                   bbox=dict(boxstyle='round,pad=.25', fc=col, ec='none', alpha=.9))
        a.set(xlabel='时间 / h', ylabel='到药材中心的距离 / cm' if a is axes[0] else '',
              title=f'{lab}\n末行 $C_{{max}}$ = {np.nanmax(Cm[-1]):.4f} kg/kg', xlim=(0, 60))
        a.grid(False)
    fig.colorbar(pc, ax=axes, pad=.015, fraction=.021, label='干基含水率 $C$ / (kg/kg)')
    fig.suptitle('不同温度制度下的含水率场演化（红实线为 0.15 kg/kg 达标等值线）',
                 fontsize=13, fontweight='bold', y=1.03)
    stamp(fig, CREDIT)
    fig.savefig(K.FIG / '图Q3_4_干燥场热图.png'); plt.close(fig)


# ================= 图 Q3-5 代理精度与残差 =================
def fig_surrogate():
    trip = [('t60_h', '$t_{60}$ / h', '达标时间', 'h'),
            ('SEC_MJ', 'SEC / (MJ·kg$^{-1}$)', '单位去水量供热量', 'MJ/kg'),
            ('dose', '$D=-\\ln Q_{ret}$', '品质降解剂量', '')]
    Td = np.array(DOE['temps'])
    Tv = np.array([r['temps'] for r in Q3['pareto_verified']])
    fig = plt.figure(figsize=(15, 7.2))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[2.1, 1], hspace=.42, wspace=.27)
    for c, (key, lab, ti, un) in enumerate(trip):
        co = Q3['surrogate_coef'][key]
        yt = np.array(DOE[key]); yv = np.array([r[key] for r in Q3['pareto_verified']])
        pt = pce_design(2 * (Td - BOXL) / (BOXH - BOXL) - 1, 3)[0] @ np.array(co)
        pv = pce_design(2 * (Tv - BOXL) / (BOXH - BOXL) - 1, 3)[0] @ np.array(co)
        ax = fig.add_subplot(gs[0, c])
        ax.scatter(yt, pt, s=15, c=C['doe'], edgecolors='none', label='DOE 点（128 训练 + 8 留出）')
        ax.scatter(yv, pv, s=46, c=C['front'], zorder=4, label='Pareto 真模型复核点')
        lo, hi = min(yt.min(), yv.min()), max(yt.max(), yv.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.1)
        s = Q3['surrogate'][key]
        ax.text(.04, .96, f"训练 $R^2$ = {s['r2']:.6f}\n5 折 CV RMSE = {s['cv5_rmse']:.5f} {un}\n"
                          f"结构点留出 RMSE = {s['test_rmse']:.5f} {un}",
                transform=ax.transAxes, va='top', fontsize=8.4,
                bbox=dict(boxstyle='round,pad=.3', fc='white', ec='#BDBDBD'))
        ax.set(xlabel='真模型 ' + lab, ylabel='代理预测 ' + lab, title=ti)
        if c == 0:
            ax.legend(fontsize=7.6, loc='lower right')
        ax = fig.add_subplot(gs[1, c])
        r = pt - yt
        ax.hist(r, bins=26, color=C['doe'], edgecolor='#5A5A5A', linewidth=.5)
        ax.axvline(0, color='k', lw=1.0)
        for sgn in (-1, 1):
            ax.axvline(sgn * s['cv5_rmse'], color=C['front'], ls='--', lw=1.2)
        ax.text(.04, .92, f'均值 {r.mean():+.2e}\n标准差 {r.std():.2e} {un}',
                transform=ax.transAxes, va='top', fontsize=8.2,
                bbox=dict(boxstyle='round,pad=.25', fc='white', ec='#BDBDBD'))
        ax.set(xlabel='残差（代理 − 真模型）/ ' + (un or '无量纲'), ylabel='频数')
    fig.suptitle('3 阶 Legendre-PCE 代理模型的精度与残差分布（代理仅用于筛选，结论以真模型复核值为准）',
                 fontsize=12.8, fontweight='bold', y=.99)
    stamp(fig, CREDIT)
    fig.savefig(K.FIG / '图Q3_5_代理精度与残差.png'); plt.close(fig)


# ================= 图 Q3-6 决策雷达图 =================
def fig_radar():
    crit = [('时间', lambda r: r['t60_h'], BASE_T), ('能耗', lambda r: r['SEC_MJ'],
            Q3['baseline']['q3_SEC_MJ']),
            ('品质', lambda r: -r['Q_ret'], -Q3['baseline']['q3_Qret']),
            ('峰值热流', lambda r: r['qmax_W_m2'], 554.1),
            ('料温上限', lambda r: r['Tbar_max_C'], max(T3['base_TbarK']) - 273.15)]
    pts = [('题面基准', dict(t60_h=BASE_T, SEC_MJ=Q3['baseline']['q3_SEC_MJ'],
                             Q_ret=Q3['baseline']['q3_Qret'], qmax_W_m2=554.1,
                             Tbar_max_C=float(max(T3['base_TbarK']) - 273.15)), C['base'], '-')]
    for k, r in recs():
        pts.append((REC_LAB[k], r, REC_COL[k], '--'))
    vals = np.array([[f(r) for _, f, _ in crit] for _, r, _, _ in pts])
    score = np.zeros_like(vals)
    for j in range(vals.shape[1]):
        lo, hi = vals[:, j].min(), vals[:, j].max()
        score[:, j] = 1.0 if hi - lo < 1e-12 else 1 - (vals[:, j] - lo) / (hi - lo)
    ang = np.linspace(0, 2 * np.pi, len(crit), endpoint=False).tolist()
    ang += ang[:1]
    fig = plt.figure(figsize=(13.5, 5.6))
    ax = fig.add_subplot(1, 2, 1, projection='polar')
    for i, (lab, r, col, ls) in enumerate(pts):
        v = score[i].tolist(); v += v[:1]
        ax.plot(ang, v, ls, lw=2.1, color=col, label=lab)
        ax.fill(ang, v, color=col, alpha=.10)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels([c[0] for c in crit], fontsize=10)
    ax.set_ylim(0, 1.05); ax.set_yticks([.25, .5, .75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=7.5)
    ax.set_title('五维相对得分（每维最优 = 1，最差 = 0）', pad=22)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, -.30), ncol=3, fontsize=8.6,
              columnspacing=1.0)

    ax2 = fig.add_subplot(1, 2, 2)
    names = [c[0] for c in crit]
    base = vals[0]
    for i, (lab, r, col, ls) in enumerate(pts[1:], start=1):
        rel = (vals[i] - base) / np.abs(base) * 100
        rel[2] = -rel[2]      # 品质：正向为改善
        rel[4] = -rel[4]
        ax2.plot(names, rel, 'o-', lw=1.9, ms=7, color=col, label=lab)
    ax2.axhline(0, color='k', lw=1.1)
    ax2.set(ylabel='相对题面基准的变化 / %', title='各推荐制度相对基准的指标变化（正 = 更差）')
    ax2.legend(fontsize=8.4)
    for lab in ax2.get_xticklabels():
        lab.set_fontsize(9.5)
    fig.suptitle('Q3 推荐制度的五维决策画像', fontsize=13.2, fontweight='bold', y=1.0)
    stamp(fig, CREDIT)
    fig.savefig(K.FIG / '图Q3_6_决策画像.png'); plt.close(fig)


# ================= 图 Q3-7 Ea 稳健性 + 决策热图 =================
def fig_ea():
    fig = plt.figure(figsize=(15.5, 5.6))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.15], wspace=.22)
    ax = fig.add_subplot(gs[0, 0])
    cols = ['#1565C0', '#F57C00', '#C62828']
    ver = Q3['pareto_verified']
    ts = np.array([r['t60_h'] for r in ver])

    def traj_of(j):
        m = np.isfinite(T3['t_h'][j])
        return {'t_h': T3['t_h'][j][m].tolist(), 'TbarK': T3['TbarK'][j][m].tolist()}

    for kk, col in zip(sorted(Q3['ea_robustness'], key=float), cols):
        ea = float(kk)
        q = np.array([math.exp(-K.dose_from_traj(traj_of(j), ea)) for j in range(len(ver))])
        o = np.argsort(ts)
        ax.plot(ts[o], q[o], '-', lw=1.9, color=col, alpha=.9,
                label=f'Ea = {ea/1000:.0f} kJ/mol')
        t = Q3['ea_robustness'][kk]['topsis_temps']
        m = [j for j, r in enumerate(ver) if np.allclose(r['temps'], t, atol=1e-6)]
        if m:
            j = m[0]
            ax.scatter([ts[j]], [q[j]], s=210, marker='*', c=col, edgecolors='k',
                       linewidths=.7, zorder=6)
            ax.annotate(f'({t[0]:.0f}, {t[1]:.0f}, {t[2]:.0f}) ℃\n{ts[j]:.2f} h',
                        (ts[j], q[j]), textcoords='offset points', xytext=(8, -22),
                        fontsize=8.2, color=col,
                        bbox=dict(boxstyle='round,pad=.22', fc='white', ec=col, lw=.8, alpha=.92))
    ax.axhline(Q3['baseline']['q3_Qret'], color=C['base'], ls=':', lw=1.3)
    ax.text(34, Q3['baseline']['q3_Qret'] + .006, '题面基准 $Q_{ret}$（Ea = 50 口径）',
            fontsize=8.2, color=C['base'])
    ax.set(xlabel='$t_{60}$ / h', ylabel='$Q_{ret}$（各曲线按自身 Ea 口径重算）',
           title='不同 Ea 下的 TOPSIS 推荐点（星标）\n※ 同一组真模型结果，仅换品质模型参数重新决策',
           xlim=(30, 84))
    ax.legend(fontsize=8.6)

    ax = fig.add_subplot(gs[0, 1])
    co = np.array(Q3['surrogate_coef']['t60_h'])
    g1 = np.linspace(BOXL, BOXH, 160); g3 = np.linspace(BOXL, BOXH, 160)
    G1, G3 = np.meshgrid(g1, g3)
    U = np.column_stack([G1.ravel(), np.full(G1.size, 60.0), G3.ravel()])
    U = 2 * (U - BOXL) / (BOXH - BOXL) - 1
    Z = (pce_design(U, 3)[0] @ co).reshape(G1.shape)
    cf = ax.contourf(G1, G3, Z, levels=18, cmap='RdYlGn_r', alpha=.92)
    cl = ax.contour(G1, G3, Z, levels=10, colors='k', linewidths=.5, alpha=.5)
    ax.clabel(cl, inline=True, fontsize=7, fmt='%.0f')
    vt = np.array([r['temps'] for r in ver])
    ax.scatter(vt[:, 0], vt[:, 2], s=34, c='k', edgecolors='w', linewidths=.6, zorder=4,
               label='真模型复核点')
    o = np.argsort([r['t60_h'] for r in ver])
    ax.plot(vt[o, 0], vt[o, 2], '-', color='k', lw=1.2, alpha=.6, zorder=3)
    for k, r in recs():
        ax.scatter([r['temps'][0]], [r['temps'][2]], s=230, marker='*', c=REC_COL[k],
                   edgecolors='k', linewidths=.8, zorder=6)
        dx, dy = (8, 4) if r['temps'][2] < 66 else (-52, -16)
        ax.annotate(REC_LAB[k], (r['temps'][0], r['temps'][2]), textcoords='offset points',
                    xytext=(dx, dy), fontsize=8.5,
                    bbox=dict(boxstyle='round,pad=.2', fc='white', ec=REC_COL[k], lw=.8, alpha=.9))
    ax.set(xlabel='第 1 段温度 $T_1$ / ℃', ylabel='第 3 段温度 $T_3$ / ℃',
           title='代理给出的 $t_{60}$ 等值线（$T_2$ = 60 ℃ 截面）\n黑线为真模型复核后的前沿投影',
           xlim=(BOXH - 31, BOXH + 0.8), ylim=(BOXL - 0.8, BOXH + 0.8))
    fig.colorbar(cf, ax=ax, pad=.02, label='代理预测 $t_{60}$ / h')
    ax.legend(fontsize=8.2, loc='lower left')
    fig.suptitle('Q3 决策的稳健性与结构：T3 是主导杠杆，T1 在前沿上被压到低温区',
                 fontsize=13, fontweight='bold', y=1.02)
    stamp(fig, CREDIT)
    fig.savefig(K.FIG / '图Q3_7_Ea稳健性与决策热图.png'); plt.close(fig)


# ================= 图 Q4-1 收缩本构与残差 =================
def fig_shrink():
    X = Q4C['Xbar']; ratio = Q4C['R_cm'] / 2.0
    fo = Q4['constitutive']['forms']
    k = fo['径向线性_过初值']['params']['k']; b = fo['体积线性_过初值']['params']['beta']
    p = fo['径向二次_过初值']['params']
    f = {'径向二次': lambda x: 1 + p['k1'] * (x - K.X0) + p['k2'] * (x - K.X0) ** 2,
         '径向线性': lambda x: 1 + k * (x - K.X0),
         '体积线性': lambda x: (1 + b * (x - K.X0)) ** (1 / 3)}
    xs = np.linspace(X.min(), X.max(), 400)
    fig = plt.figure(figsize=(13.5, 6.0))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=.12)
    ax = fig.add_subplot(gs[0])
    styles = {'径向二次': ('-', C['warm'], 2.4), '径向线性': ('--', C['front'], 1.9),
              '体积线性': ('-.', C['cold'], 1.9)}
    for nm, fn in f.items():
        rms_um = fo[nm + '_过初值']['RMS_cm'] * 1e4
        ax.plot(xs, fn(xs), styles[nm][0], lw=styles[nm][2], color=styles[nm][1], zorder=3,
                label=f"{nm}：RMS {rms_um:.0f} μm、$R^2$ = {fo[nm + '_过初值']['R2']:.4f}")
    ax.plot(X, ratio, 'o', ms=3.4, color=C['base'], alpha=.75, zorder=5,
            label='附件2 半径 × 本模型平均含水率（共 %d 点）' % len(X))
    ax.set(ylabel='$R/R_0$', title='Q4 收缩本构回归：径向二次显著优于线性与体积线性')
    ax.legend(fontsize=8.6, loc='lower right'); ax.set_xticklabels([])
    axr = fig.add_subplot(gs[1], sharex=ax)
    for nm, fn in f.items():
        axr.plot(X, ratio - fn(X), styles[nm][0], lw=1.5, color=styles[nm][1], zorder=3)
    axr.axhline(0, color='k', lw=1.0)
    for s in (0.005, -0.005):
        axr.axhline(s, color=C['ref'], ls=':', lw=1.0)
    axr.text(X.min() + .02, .0058, '±0.005 $R/R_0$（= 0.01 cm = 0.5% $R_0$）',
             fontsize=8, color=C['ref'])
    axr.set(xlabel='体积平均干基含水率 $\\bar X$ / (kg/kg)', ylabel='残差 / $R/R_0$')
    stamp(fig, CREDIT)
    fig.savefig(K.FIG / '图Q4_1_收缩本构与残差.png'); plt.close(fig)


# ================= 图 Q4-2 外推预测检验 =================
def fig_predict():
    pt = Q4['constitutive']['prediction_tests']
    ks = ['0.3', '0.4', '0.6']
    fig = plt.figure(figsize=(14.5, 5.4))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.35, 1], wspace=.22)
    X = Q4C['Xbar']; ratio = Q4C['R_cm'] / 2.0
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(X, ratio, 'o', ms=2.0, color=C['base'], alpha=.5, label='全部数据（真值）')
    xs = np.linspace(X.min(), X.max(), 400)
    P = pt['0.4']
    P0 = P['径向线性']
    k40 = P0['params']['k']; b40 = P['体积线性']['params']['beta']
    k1q, k2q = P['径向二次']['params']['k1'], P['径向二次']['params']['k2']
    ax.axvline(P0['X_split'], color=C['ref'], ls=':', lw=1.4)
    ax.text(P0['X_split'] + .03, .93, '标定区 ← ｜ → 预测区', fontsize=9, color=C['ref'])
    ax.plot(xs, 1 + k1q * (xs - K.X0) + k2q * (xs - K.X0) ** 2, '-', lw=2.2, color=C['warm'],
            label='径向二次（前 40% 标定）')
    ax.plot(xs, 1 + k40 * (xs - K.X0), '--', lw=2.0, color=C['front'], label='径向线性（前 40% 标定）')
    y = np.clip(1 + b40 * (xs - K.X0), 1e-9, None) ** (1 / 3)
    m = np.isfinite(y)
    ax.plot(xs[m], y[m], '-.', lw=1.8, color=C['cold'], label='体积线性（前 40% 标定）')
    ax.scatter([X[-1]], [ratio[-1]], s=130, marker='*', c='k', zorder=6,
               label=f'末端真值 {ratio[-1]:.4f}')
    for nm, yv, col in (('径向二次', 1 + k1q * (X[-1] - K.X0) + k2q * (X[-1] - K.X0) ** 2, C['warm']),
                        ('径向线性', 1 + k40 * (X[-1] - K.X0), C['front'])):
        ax.scatter([X[-1]], [yv], s=90, marker='v', c=col, edgecolors='k', linewidths=.5, zorder=6)
        ax.annotate(f'{nm}预测 {yv:.3f}', (X[-1], yv), textcoords='offset points',
                    xytext=(6, -4), fontsize=8.4, color=col)
    ax.set(xlabel='体积平均干基含水率 $\\bar X$ / (kg/kg)', ylabel='$R/R_0$',
           title='收缩本构的外推能力检验（前 40% 标定 → 后 60% 预测）')
    ax.legend(fontsize=8.4, loc='lower right')

    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(ks)); w = .26
    for i, (nm, col) in enumerate((('径向线性', C['front']), ('径向二次', C['warm']),
                                   ('体积线性', C['cold']))):
        v = [pt[t][nm]['test_RMS_cm'] for t in ks]
        ax.bar(x + (i - 1) * w, v, w, color=col, alpha=.9, label=nm)
        for xi, vv in zip(x + (i - 1) * w, v):
            ax.text(xi, vv + .012, f'{vv:.3f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f'前 {float(t)*100:.0f}% 标定' for t in ks])
    ax.set(ylabel='后段预测 RMS / cm', ylim=(0, .62),
           title='外推误差：标定区间越短、误差越大\n（体积线性本构甚至外推到不可用）')
    ax.legend(fontsize=8.6)
    fig.suptitle('线性收缩本构不具备外推能力 → 附件2 的 R(t) 宜按题面作为独立输入',
                 fontsize=13, fontweight='bold', y=1.02)
    stamp(fig, CREDIT)
    fig.savefig(K.FIG / '图Q4_2_外推预测检验.png'); plt.close(fig)


# ================= 图 Q4-3 内生收缩与干燥曲线 =================
def fig_endo():
    t = Q4C['t_h']; X = Q4C['Xbar']; R = Q4C['R_cm']
    fo = Q4['constitutive']['forms']
    k = fo['径向线性_过初值']['params']['k']; b = fo['体积线性_过初值']['params']['beta']
    p = fo['径向二次_过初值']['params']
    k40 = Q4['constitutive']['prediction_tests']['0.4']['径向线性']['params']['k']
    dX = X - K.X0
    curves = [('附件2 观测', R, C['base'], '-', 2.4),
              ('内生二次', np.clip(1 + p['k1'] * dX + p['k2'] * dX ** 2, .2, 1) * 2.0, C['warm'], '--', 1.9),
              ('内生线性', np.clip(1 + k * dX, .2, 1) * 2.0, C['front'], '-.', 1.7),
              ('内生体积线性', np.clip(1 + b * dX, 1e-6, None) ** (1 / 3) * 2.0, C['cold'], ':', 1.7),
              ('前 40% 标定线性', np.clip(1 + k40 * dX, .2, 1) * 2.0, '#6A1B9A', (0, (3, 1, 1, 1)), 1.6)]
    fig = plt.figure(figsize=(14.5, 5.4))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1], wspace=.22)
    ax = fig.add_subplot(gs[0, 0])
    for lab, y, col, ls, lw in curves:
        ax.plot(t, y, linestyle=ls, lw=lw, color=col, label=lab)
    for lab, y, col, _, _ in curves[1:]:
        ax.fill_between(t, R, y, color=col, alpha=.08)
    ax.set(xlabel='时间 / h', ylabel='半径 $R$ / cm',
           title='由平均含水率经各本构反推的 $R(t)$ 与附件2 观测对比')
    ax.legend(fontsize=8.4)

    ax = fig.add_subplot(gs[0, 1])
    t1, C1 = fld('Q4_附件2口径_result4式.xlsx')
    t2, C2 = fld('Q4_内生二次_result4式.xlsx')
    S1 = FLD['Q4_附件2口径_result4式.xlsx__S']
    S2 = FLD['Q4_内生二次_result4式.xlsx__S']
    ax.plot(t1 / 3600, C1[:, 0], lw=2.1, color=C['base'], label='附件2 口径 · 中心')
    ax.plot(t1 / 3600, S1, lw=2.1, color=C['base'], ls='--', label='附件2 口径 · 表面')
    ax.plot(t2 / 3600, C2[:, 0], lw=1.7, color=C['warm'], label='内生二次 · 中心')
    ax.plot(t2 / 3600, S2, lw=1.7, color=C['warm'], ls='--', label='内生二次 · 表面')
    ax.axhline(0.15, color='k', ls=':', lw=1.3)
    ax.text(1.5, .17, '达标阈值 0.15', fontsize=8.5)
    ax.axvline(50.8333, color=C['ref'], ls=':', lw=1.2)
    ax.set(xlabel='时间 / h', ylabel='干基含水率 / (kg/kg)', ylim=(0, 2.7),
           title='两种收缩口径下的中心/表面含水率\n'
                 r'（$t_{60}$ 均为 50.83 h，$\Delta t_{60}$ = 0.0000 h）')
    ax.legend(fontsize=8.4)
    fig.suptitle('Q4 把附件2 的 $R(t)$ 换成内生本构：几何可重现，达标时间不变',
                 fontsize=13, fontweight='bold', y=1.02)
    stamp(fig, CREDIT)
    fig.savefig(K.FIG / '图Q4_3_内生收缩与干燥曲线.png'); plt.close(fig)


# ================= 图 Q4-4 孔隙率与相容性 =================
def fig_poro():
    cr = Q4['porosity']['curve']; t = np.array(cr['t_h'])
    cons = Q4['porosity']['consistency']
    fig = plt.figure(figsize=(16.5, 5.2))
    gs = GridSpec(1, 3, figure=fig, wspace=.26)
    ax = fig.add_subplot(gs[0, 0])
    ax.stackplot(t, [1 - np.array(cr['A_phi_void']), cr['A_phi_water'], cr['A_phi_gas']],
                 labels=['固相', '水', '气相孔隙'], colors=['#8D6E63', C['wet'], '#ECEFF1'],
                 edgecolor='w', lw=.5)
    ax.plot(t, cr['A_phi_gas'], lw=2.0, color=C['front'])
    ax.annotate(f"气相孔隙率\n{cr['A_phi_gas'][0]:.3f} → {cr['A_phi_gas'][-1]:.3f}",
                (t[int(len(t) * .62)], cr['A_phi_gas'][int(len(t) * .62)]),
                textcoords='offset points', xytext=(-8, 30), fontsize=8.6, ha='right',
                arrowprops=dict(arrowstyle='->', lw=.9))
    ax.set(xlabel='时间 / h', ylabel='体积分数', ylim=(0, 1),
           title=f"口径A（仅附录4 $\\rho_b$=760+90C，$\\rho_s$ = {Q4['porosity']['rho_s_ref']:.0f}）")
    ax.legend(fontsize=8.2, loc='center left')

    ax = fig.add_subplot(gs[0, 1])
    a = np.array(cr['R_over_R0_attach']); d = np.array(cr['R_over_R0_density'])
    ax.plot(t, a, lw=2.3, color=C['base'], label='附件2 几何收缩 $R/R_0$')
    ax.plot(t, d, lw=2.3, ls='--', color=C['front'],
            label='附录4 密度式隐含 $R/R_0$')
    ax.fill_between(t, a, d, color=C['front'], alpha=.18,
                    label=f'偏差（最大 {cons["max_abs_dev"]:.4f}）')
    i = int(np.argmax(np.abs(a - d)))
    ax.annotate(f'最大偏差 {cons["max_abs_dev"]:.4f}\n@ {t[i]:.1f} h', (t[i], (a[i] + d[i]) / 2),
                textcoords='offset points', xytext=(18, -6), fontsize=8.8,
                arrowprops=dict(arrowstyle='->', lw=1.0))
    ax.set(xlabel='时间 / h', ylabel='$R/R_0$', title='相容性诊断：两套经验闭合不一致')
    ax.legend(fontsize=8.2, loc='lower left')

    ax = fig.add_subplot(gs[0, 2])
    bg = np.array(cr['B_phi_gas']); ag = np.array(cr['A_phi_gas'])
    ax.plot(t, bg, lw=2.3, color='#6A1B9A', label='口径B（仅用附件2 收缩）')
    ax.plot(t, ag, lw=2.0, ls='--', color=C['front'], label='口径A（对照，恒非负）')
    ax.axhline(0, color='k', lw=1.2)
    ax.fill_between(t, bg, 0, where=(bg < 0), color=C['bad'], alpha=.30, hatch='///',
                    edgecolor=C['bad'], lw=0)
    ax.annotate(f'负气相孔隙率\nmin = {cons["min_B_phi_gas"]:.4f}\n（物理上不可能）',
                (t[int(np.argmin(bg))], cons['min_B_phi_gas']), textcoords='offset points',
                xytext=(24, 26), fontsize=8.8, color=C['bad'],
                arrowprops=dict(arrowstyle='->', lw=1.1, color=C['bad']))
    ax.set(xlabel='时间 / h', ylabel='气相孔隙率', title='口径B 出现负值 → 两式不能同时成立')
    ax.legend(fontsize=8.2, loc='lower right')
    fig.suptitle('Q4 孔隙率演化与内在一致性：附件2 的收缩量与附录4 的密度式存在结构性张力',
                 fontsize=12.8, fontweight='bold', y=1.03)
    stamp(fig, CREDIT)
    fig.savefig(K.FIG / '图Q4_4_孔隙率与相容性.png'); plt.close(fig)


# ================= 图 Q4-5 情景对比（发散棒棒糖） =================
def fig_q4_scen():
    base = Q4['baseline']['t60_h']
    items = []
    for k, v in Q4['endogenous'].items():
        items.append((k.replace('_', ' '), v['delta_h'], C['cold'], 'Q4 收缩口径'))
    for k, v in Q4['crust'].items():
        if float(k) == 0:
            continue
        items.append((f'表皮阻力 $K_c$ = {k}', v['delta_h'], C['hot'], 'Q4 传质情景'))
    items.sort(key=lambda x: x[1])
    fig = plt.figure(figsize=(13.5, 5.4))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1], wspace=.22)
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(items))
    for i, (lab, d, col, grp) in enumerate(items):
        ax.plot([0, d], [i, i], lw=2.6, color=col, solid_capstyle='round')
        ax.scatter([d], [i], s=150, c=col, zorder=5, edgecolors='k', linewidths=.6)
        ax.text(d + (0.6 if d >= 0 else -0.6), i, f'{d:+.4f} h\n({d/base*100:+.2f}%)',
                va='center', ha='left' if d >= 0 else 'right', fontsize=8.4)
    ax.axvline(0, color='k', lw=1.4)
    ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items], fontsize=8.8)
    ax.set(xlabel='相对附件2 口径的达标时间变化 $\\Delta t_{60}$ / h', xlim=(-36, 9),
           title=f'各情景对 Q4 达标时间的影响（基准 {base:.4f} h = 183000 s）')
    ax.grid(axis='y', alpha=0)
    ax.legend(handles=[Patch(color=C['cold'], label='收缩本构情景'),
                       Patch(color=C['hot'], label='表皮阻力情景')], fontsize=8.6, loc='lower right')

    ax = fig.add_subplot(gs[0, 1])
    kc = sorted([float(k) for k in Q4['crust']])
    d = [Q4['crust'][f'{k:.0f}']['delta_h'] if f'{k:.0f}' in Q4['crust']
         else Q4['crust'][str(int(k))]['delta_h'] for k in kc]
    cs = np.linspace(0, 1.2, 200)
    ax.plot(cs, 1 / (1 + 2 * np.clip(1 - cs / 0.5, 0, 1)), lw=2.2, color=C['hot'],
            label='$h_{m,eff}/h_m$（$K_c$ = 2，$C_{crust}$ = 0.5）')
    ax.axvspan(0, 0.5, color=C['hot'], alpha=.07)
    ax.axvline(0.5, color=C['ref'], ls=':', lw=1.2)
    ax.text(0.51, 0.9, '$C_{crust}$ = 0.5', fontsize=8.6, color=C['ref'])
    ax.set(xlabel='表面干基含水率 $C_s$ / (kg/kg)', ylabel='相对传质系数',
           title='表皮阻力模型：表面变干后传质系数逐步衰减', ylim=(0, 1.05))
    axin = ax.inset_axes([.42, .16, .54, .42])
    axin.bar(kc, d, width=.7, color=C['hot'], alpha=.85)
    for xi, v in zip(kc, d):
        axin.text(xi, v + .12, f'{v:+.2f}', ha='center', fontsize=8)
    axin.set_xticks(kc)
    axin.set(xlabel='$K_c$', ylabel='$\\Delta t_{60}$ / h', title='时长影响', xlim=(-0.6, 4.6))
    axin.title.set_fontsize(8.6)
    axin.tick_params(labelsize=7.6); axin.grid(alpha=.2)
    ax.legend(fontsize=8.4, loc='upper right')
    stamp(fig, CREDIT)
    fig.savefig(K.FIG / '图Q4_5_情景对比.png'); plt.close(fig)


if __name__ == '__main__':
    (K.FIG / '图Q3_1_Pareto前沿.png').unlink(missing_ok=True)
    (K.FIG / '图Q3_2_温度制度.png').unlink(missing_ok=True)
    (K.FIG / '图Q3_3_干燥曲线.png').unlink(missing_ok=True)
    (K.FIG / '图Q3_4_代理精度.png').unlink(missing_ok=True)
    (K.FIG / '图Q4_1_收缩本构.png').unlink(missing_ok=True)
    (K.FIG / '图Q4_2_孔隙率与相容性.png').unlink(missing_ok=True)
    (K.FIG / '图Q4_3_情景对比.png').unlink(missing_ok=True)
    for fn in (fig_overview, fig_q3_3d, fig_q3_proj, fig_sched, fig_fields, fig_surrogate,
               fig_radar, fig_ea, fig_shrink, fig_predict, fig_endo, fig_poro, fig_q4_scen):
        fn(); print(fn.__name__, '完成', flush=True)
    print('全部图件已写入', K.FIG)
