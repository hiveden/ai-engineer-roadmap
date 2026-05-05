# 02-原理 · Demos

第 2 章 · LR 原理 + 损失函数配套交互 demo。

## 索引

| Demo | 章节 | 端口 | 教学钩 |
|---|---|---|---|
| [`01-decision-boundary.py`](./01-decision-boundary.py) | 01-原理 | 2733 | 拖 w₁/w₂/b/τ → 决策边界直线实时移动 + 概率热图 + sigmoid 阈值映射 |
| [`02-loss-landscape.py`](./02-loss-landscape.py) | 02-损失函数 | 2734 | MSE 非凸 vs log-loss 凸（多极小 vs 单极小）+ 单样本 −log(p) 惩罚曲线 |

## 启动

```bash
.venv/bin/marimo edit "01-ML/03-LogReg/02-原理/demos/01-decision-boundary.py" --port 2733 --headless --no-token
.venv/bin/marimo edit "01-ML/03-LogReg/02-原理/demos/02-loss-landscape.py"     --port 2734 --headless --no-token
```

## 自测（不开浏览器）

```bash
for f in "01-ML/03-LogReg/02-原理/demos"/0[1-2]-*.py; do
  name=$(basename "$f" .py)
  .venv/bin/marimo export script "$f" -o "/tmp/_${name}_lint.py" \
    && .venv/bin/python "/tmp/_${name}_lint.py" \
    && echo "[$name] ✓"
done
```

## 设计约定

沿用 01-foundation/demos：altair + 反应式 slider/dropdown，无 layouts.json（Step 5 由用户在 marimo edit 拖布局）。
