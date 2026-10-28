# -*- coding: utf-8 -*-
"""一键复现：按顺序跑完第三、四问创新扩展的全部计算与图件。

用法（在本目录下）：
    python run_all.py            # 全量
    python run_all.py --quick    # 跳过 DOE（沿用已有 results/q3_doe_shard*.json）
"""
import argparse, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script, args=()):
    tic = time.perf_counter()
    print(f'\n===== {script} {" ".join(args)} =====', flush=True)
    r = subprocess.run([sys.executable, str(HERE / script), *args], cwd=str(HERE))
    print(f'----- {script} 退出码 {r.returncode}  用时 {time.perf_counter()-tic:.1f}s', flush=True)
    if r.returncode != 0:
        raise SystemExit(f'{script} 失败')
    return time.perf_counter() - tic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='跳过 DOE 重跑')
    a = ap.parse_args()
    t0 = time.perf_counter()
    if not a.quick:
        run('q3_doe.py', ['--shard', '0', '--nshards', '1'])
    run('q3_opt.py')
    run('q4_shrink.py')
    run('make_result_workbooks.py')
    run('make_figures.py')
    run('summary.py')
    print(f'\n全部完成，总用时 {time.perf_counter()-t0:.1f}s')


if __name__ == '__main__':
    main()
