# -*- coding: utf-8 -*-
"""Generate paper figures from the revised q*_curve.npz."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent           # …/A题_终稿
CALC = ROOT / '计算结果'
OUT = ROOT / '图'
OUT.mkdir(exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load(q):
    d = np.load(CALC / f'q{q}_curve.npz')
    return d['t'], d['x'], d['T'], d['C'], d['R'], d['mean']

def r_physical(x, R):
    return x[:, None] * R[None, :] * 100  # cm

# ---------------- 图1/图2：问题1 温度与水分径向分布 ----------------
t1, x1, T1, C1, R1, mean1 = load(1)
times1 = np.array([100, 300, 600, 900, 1200, 1500, 1800.])
idx1 = [np.argmin(np.abs(t1 - tt)) for tt in times1]
rr1 = r_physical(x1, R1)[:, 0]
cmap = plt.cm.viridis(np.linspace(0.05, 0.95, len(times1)))

fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=130)
for k, j in enumerate(idx1):
    ax.plot(rr1, T1[:, j], color=cmap[k], lw=1.8, label=f'{int(times1[k])} s')
ax.set_xlabel('到药材中心的距离 / cm'); ax.set_ylabel('温度 / ℃')
ax.set_title('图 1  问题 1：预热阶段药材温度径向分布')
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / '图1_q1_温度.png'); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=130)
for k, j in enumerate(idx1):
    ax.plot(rr1, C1[:, j], color=cmap[k], lw=1.8, label=f'{int(times1[k])} s')
ax.set_xlabel('到药材中心的距离 / cm'); ax.set_ylabel('水分浓度 / (kg/kg)')
ax.set_title('图 2  问题 1：预热阶段药材水分浓度径向分布')
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / '图2_q1_水分.png'); plt.close(fig)

# ---------------- 图3：问题2 温度与水分径向分布（3h内） ----------------
t2, x2, T2, C2, R2, mean2 = load(2)
times2 = np.arange(1800, 10801, 1800.)
idx2 = [np.argmin(np.abs(t2 - tt)) for tt in times2]
rr2 = r_physical(x2, R2)[:, 0]
cmap2 = plt.cm.viridis(np.linspace(0.05, 0.95, len(times2)))

fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), dpi=130)
for k, j in enumerate(idx2):
    axs[0].plot(rr2, T2[:, j], color=cmap2[k], lw=1.6, label=f'{times2[k]/3600:.1f} h')
    axs[1].plot(rr2, C2[:, j], color=cmap2[k], lw=1.6, label=f'{times2[k]/3600:.1f} h')
axs[0].set_xlabel('到药材中心的距离 / cm'); axs[0].set_ylabel('温度 / ℃')
axs[0].set_title('(a) 温度径向分布'); axs[0].grid(alpha=0.3); axs[0].legend(fontsize=7, ncol=2)
axs[1].set_xlabel('到药材中心的距离 / cm'); axs[1].set_ylabel('水分浓度 / (kg/kg)')
axs[1].set_title('(b) 水分浓度径向分布'); axs[1].grid(alpha=0.3); axs[1].legend(fontsize=7, ncol=2)
fig.suptitle('图 3  问题 2：整个烘干过程前 3 h 的径向分布', y=1.02)
fig.tight_layout(); fig.savefig(OUT / '图3_q2_温度水分.png'); plt.close(fig)

# ---------------- 图4：问题3 水分径向分布（每隔6h） ----------------
t3, x3, T3, C3, R3, mean3 = load(3)
times3 = np.arange(21600, 21600*9+1, 21600.)
idx3 = [np.argmin(np.abs(t3 - tt)) for tt in times3]
rr3 = r_physical(x3, R3)[:, 0]
cmap3 = plt.cm.plasma(np.linspace(0.05, 0.95, len(times3)))

fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=130)
for k, j in enumerate(idx3):
    ax.plot(rr3, C3[:, j], color=cmap3[k], lw=1.7, label=f'{times3[k]/3600:.0f} h')
ax.axhline(0.15, color='r', ls='--', lw=1.3, label='达标阈值 0.15')
ax.set_xlabel('到药材中心的距离 / cm'); ax.set_ylabel('水分浓度 / (kg/kg)')
ax.set_title('图 4  问题 3：烘干过程水分浓度径向分布（每隔 6 h）')
ax.legend(fontsize=7.5, ncol=2); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / '图4_q3_水分.png'); plt.close(fig)

# ---------------- 图5：问题3 干燥曲线（中心/表面 + 阈值） ----------------
tc3 = float(t3[-1]) / 3600
fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=130)
ax.plot(t3/3600, C3[0, :], lw=2, label='中心 (r=0)')
ax.plot(t3/3600, C3[-1, :], lw=2, label='表面 (r=R)')
ax.plot(t3/3600, mean3, lw=1.6, ls='-.', label='体积平均含水率')
ax.axhline(0.15, color='r', ls='--', lw=1.3)
ax.axvline(tc3, color='r', ls=':', lw=1.2)
ax.annotate(f'烘干结束 {tc3:.2f} h', xy=(tc3, 0.15), xytext=(tc3-26, 0.9),
            arrowprops=dict(arrowstyle='->', color='k'), fontsize=9)
ax.set_xlabel('时间 / h'); ax.set_ylabel('水分浓度 / (kg/kg)')
ax.set_title('图 5  问题 3：干燥曲线与达标时间')
ax.set_xlim(0, t3[-1]/3600); ax.set_ylim(0, 2.7)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / '图5_q3_干燥曲线.png'); plt.close(fig)

# ---------------- 图6：问题4 收缩对比 + 半径曲线 ----------------
t4, x4, T4, C4, R4, mean4 = load(4)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), dpi=130)
axs[0].plot(t3/3600, C3[0, :], lw=2, label='问题3 中心（固定半径）')
axs[0].plot(t4/3600, C4[0, :], lw=2, label='问题4 中心（径向收缩）')
axs[0].axhline(0.15, color='r', ls='--', lw=1.2)
axs[0].axvline(tc3, color='C0', ls=':', lw=1.2)
axs[0].axvline(float(t4[-1])/3600, color='C1', ls=':', lw=1.2)
axs[0].set_xlabel('时间 / h'); axs[0].set_ylabel('中心水分浓度 / (kg/kg)')
axs[0].set_title('(a) 中心含水率对比'); axs[0].legend(fontsize=8.5); axs[0].grid(alpha=0.3)
axs[1].plot(t4/3600, R4*100, lw=2, color='C2')
axs[1].set_xlabel('时间 / h'); axs[1].set_ylabel('药材半径 / cm')
axs[1].set_title('(b) 半径收缩曲线（附件2）'); axs[1].grid(alpha=0.3)
axs[1].set_ylim(1.0, 2.1)
fig.suptitle('图 6  问题 4：收缩对干燥过程的影响', y=1.02)
fig.tight_layout(); fig.savefig(OUT / '图6_q4_收缩对比.png'); plt.close(fig)

print('figures saved to', OUT)
for p in sorted(OUT.glob('*.png')):
    print(' ', p.name, p.stat().st_size, 'bytes')
