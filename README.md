# CUMCM 2026 A 题整理

## 项目简介
（待补充：题目名称、背景、本文采用的方法与主要结论）

## 目录结构
- src/：求解与导出代码（solve / validate / make_figures / export_results）
- results/figures/：论文图表
- results/tables/：计算结果 CSV/JSON
- results/workbooks/：提交结果 result1~4.xlsx
- validation/：终稿与改进版的验证/核验材料
- docs/：建模思路、修订说明、坑点记录
- extra/q3q4_innovation/：第三、四问创新分析（Pareto、收缩、反演等）

## 复现
    pip install -r requirements.txt
    python src/solve.py
    python src/make_figures.py
    python src/export_results.py

export_results.mjs 需要 Node.js，可选。

## 数据与合规声明
- 原始赛题、附件数据未上传，请从竞赛官方渠道获取。
- 论文正文、承诺书未上传，避免参赛信息与版权问题。

## AI 使用声明
部分代码/文字使用 DeepSeek harness、Codex 等工具辅助生成，均经人工检查、修改与验证。详见 docs/ai-usage.md。