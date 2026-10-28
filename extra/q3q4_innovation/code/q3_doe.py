# -*- coding: utf-8 -*-
"""Q3 三段恒温制度的 DOE（分片运行）。

用法：python q3_doe.py --shard i --nshards m
每个分片把结果写到 results/q3_doe_shard{i}.json
"""
import argparse, json, time, sys, traceback
from pathlib import Path
import numpy as np
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K   # noqa: E402

N_SOBOL = 128
N_TOTAL = N_SOBOL + 8
N_RUN = 80           # DOE 用网格数
RTOL = 1e-6
MAX_STEP = 300.0
SEED = 20260912


def design():
    """返回 (序号, 温度三元组, 标签) 列表。含结构化角点与两个机理性对照。"""
    lo, hi = K.T_BOX
    eng = qmc.Sobol(d=3, scramble=True, seed=SEED)
    u = eng.random(N_SOBOL)
    pts = lo + u * (hi - lo)
    structured = np.array([
        [lo, lo, lo], [hi, hi, hi], [hi, lo, lo], [lo, hi, lo], [lo, lo, hi],
        [hi, hi, lo], [hi, lo, hi], [50.165, 50.165, 50.165],
    ])
    allp = np.vstack([pts, structured])
    labels = [f'sobol{i:03d}' for i in range(N_SOBOL)] + [f'corner{i}' for i in range(8)]
    return list(zip(range(N_TOTAL), allp, labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    a = ap.parse_args()
    todo = [d for d in design() if d[0] % a.nshards == a.shard]
    out = []
    tic = time.perf_counter()
    for idx, T, lab in todo:
        t0 = time.perf_counter()
        try:
            d = K.run_schedule(3, tuple(T), n=N_RUN, rtol=RTOL, max_step=MAX_STEP,
                               method='BDF', limit_h=150.0, want_traj=True)
            d['idx'] = idx
            d['label'] = lab
            d['wall_s'] = round(time.perf_counter() - t0, 2)
            out.append(d)
            print(f'[{a.shard}] {idx:3d} {lab:<10} T=({T[0]:6.2f},{T[1]:6.2f},{T[2]:6.2f}) '
                  f'reached={d["reached"]} t60={d.get("t60_h", float("nan")):8.4f} h '
                  f'SEC={d.get("SEC_J_per_kg", float("nan"))/1e6:8.3f} MJ/kg '
                  f'Qret={d.get("Q_ret", float("nan")):.6f}  {d["wall_s"]}s', flush=True)
        except Exception as e:              # noqa: BLE001
            out.append(dict(idx=idx, label=lab, temps=list(map(float, T)), reached=False,
                            error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc()))
            print(f'[{a.shard}] {idx:3d} {lab:<10} FAILED {e}', flush=True)
    p = K.RES / f'q3_doe_shard{a.shard}.json'
    p.write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
    print(f'[{a.shard}] wrote {p.name}  n={len(out)}  wall={time.perf_counter()-tic:.1f}s', flush=True)


if __name__ == '__main__':
    main()
