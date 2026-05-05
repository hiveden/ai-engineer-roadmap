# 05-客户流失 · Demos

| Demo | 章节 | 端口 | 教学钩 |
|---|---|---|---|
| [`01-churn-pipeline.py`](./01-churn-pipeline.py) | 03-案例实现 | 2750 | 7000 客户合成 Telco 数据 → drop ID + one-hot + 标准化 + LR + class_weight + 评估 + top-10 系数排行 |

## 启动

```bash
.venv/bin/marimo edit "01-ML/03-LogReg/05-客户流失/demos/01-churn-pipeline.py" --port 2750 --headless --no-token
```

## 数据说明

由于仓库未携带 `WA_Fn-UseC_-Telco-Customer-Churn.csv`，demo 内用 numpy 合成 7000 客户 × 12 特征的近似分布（流失率 ~26%、Contract / tenure / MonthlyCharges 与 Churn 的相关性遵循真实数据规律）。如需用真实数据：把 cell 2 替换为 `pd.read_csv(...)`，drop 同名列并保留 one-hot 流程即可。
