# -*- coding: utf-8 -*-
"""CSV -> result{q}.xlsx（openpyxl write_only；问题2 全程 20 万行时作为兜底）。

用法：python export_results.py 1|2|3|4
表头/工作表名与附件3模板一致；数值已按四位小数写入。
"""
import csv, sys
from pathlib import Path
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

ROOT = Path(__file__).resolve().parent.parent
CALC = ROOT / '计算结果'
OUT = ROOT / '提交结果'
OUT.mkdir(parents=True, exist_ok=True)
q = int(sys.argv[1]) if len(sys.argv) > 1 else 1

def read_csv(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        r = csv.reader(f)
        header = next(r)
        return header, [row for row in r]

def build_header(header):
    h = [header[0]]
    for v in header[1:]:
        h.append(v if v == '药材表面' else float(v))
    return h

def conv(row):
    vals = []
    for i, v in enumerate(row):
        if v == '':
            vals.append(None)
        else:
            x = float(v)
            vals.append(int(x) if (i == 0 and x == int(x)) else x)
    return vals

fields = [('温度', 'T'), ('水分浓度', 'C')] if q <= 2 else [('Sheet1', 'C')]
wb = openpyxl.Workbook(write_only=True)
for name, field in fields:
    header, rows = read_csv(CALC / f'q{q}_{field}.csv')
    ws = wb.create_sheet(name)
    ws.append(build_header(header))
    n = 0
    for row in rows:
        ws.append(conv(row)); n += 1
    print(f'  sheet {name}: {n} rows x {len(header)} cols', flush=True)
wb.save(OUT / f'result{q}.xlsx')
print(f'saved result{q}.xlsx', flush=True)
