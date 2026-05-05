# 03-api · Demos

第 3 章 · API 介绍 + 癌症分类配套交互 demo。

## 索引

| Demo | 章节 | 端口 | 教学钩 |
|---|---|---|---|
| [`01-c-penalty-effect.py`](./01-c-penalty-effect.py) | 01-API 介绍 | 2735 | 拖 C / 切 L1/L2 → 决策边界 + coef_ 条形图（L1 稀疏化噪声特征归零）|
| [`02-cancer-pipeline.py`](./02-cancer-pipeline.py) | 02-癌症分类案例 | 2736 | 威斯康星乳腺癌 minimal pipeline + top-10 \|w\| 排行 + 阈值调节 |

## 启动

```bash
.venv/bin/marimo edit "01-ML/03-LogReg/03-api/demos/01-c-penalty-effect.py" --port 2735 --headless --no-token
.venv/bin/marimo edit "01-ML/03-LogReg/03-api/demos/02-cancer-pipeline.py"  --port 2736 --headless --no-token
```

## 自测

```bash
for f in "01-ML/03-LogReg/03-api/demos"/0[1-2]-*.py; do
  name=$(basename "$f" .py)
  .venv/bin/marimo export script "$f" -o "/tmp/_${name}_lint.py" \
    && .venv/bin/python "/tmp/_${name}_lint.py" \
    && echo "[$name] ✓"
done
```
