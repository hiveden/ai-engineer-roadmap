# 01 · KNN

K-Nearest Neighbor，懒惰学习器（lazy learner）的代表——不算参数、不训练，预测时算距离投票。
KNN 是 ML 第一根算法柱子，覆盖**距离度量、bias-variance、特征工程、超参调优** 4 个核心议题。

## 章节地图

| # | 章节 | 知识焦点 | 互动 demo |
|---|---|---|---|
| 1 | [`01-intro/`](./01-intro/) | 算法思想、距离、k 调优、加权、回归 | `02-proximity` / `03-k-tuning` / `04-regression` |
| 2 | [`02-api/`](./02-api/) | sklearn `KNeighborsClassifier` / `KNeighborsRegressor` | — （文档已含完整代码） |
| 3 | [`03-distance/`](./03-distance/) | 欧氏 / 曼哈顿 / 切比雪夫 / 闵可夫斯基 | `01-distance-zoo` |
| 4 | [`04b-scaling/`](./04b-scaling/) | 量纲问题、归一化、标准化、高斯分布 | `02-knn-scaling` |
| 5 | [`04c-iris-case/`](./04c-iris-case/) | 完整 ML pipeline 6 步实战 | `01-iris-pipeline` |
| 6 | [`05-hyperparameter/`](./05-hyperparameter/) | 交叉验证、GridSearchCV、手写数字 | `01-cv-gridsearch` |

## demo 启动速查

每个 demo 是独立 marimo notebook。建议端口分配（避免冲突）：

```bash
# 01-intro
marimo run 01-intro/demos/02-proximity.py     --port 2718
marimo run 01-intro/demos/03-k-tuning.py      --port 2719
marimo run 01-intro/demos/04-regression.py    --port 2720

# 03-distance
marimo run 03-distance/demos/01-distance-zoo.py --port 2724

# 04b-scaling
marimo run 04b-scaling/demos/02-knn-scaling.py --port 2723

# 04c-iris-case
marimo run 04c-iris-case/demos/01-iris-pipeline.py --port 2725

# 05-hyperparameter
marimo run 05-hyperparameter/demos/01-cv-gridsearch.py --port 2726
```

`marimo run` 是只读演示模式，多 tab 友好。要改代码用 `marimo edit`（单 tab 限制）。

## 学完掌握

完成本章你应该能：

- **直觉**：把"分类问题"翻译成"在特征空间里给查询点找邻居投票"，2D 决策边界图能凭直觉读出
- **算法**：手算 KNN 投票流程；理解 lazy learner 与正常 fit-predict 范式的区别
- **距离**：知道 4 种距离的几何含义，以及 sklearn `metric='minkowski'` 的 p 参数控制
- **k 调优**：理解 bias-variance 谱（k=1 过拟合 → k=√N 适中 → k=N 退化为多数类），用 LOOCV / GridSearchCV 数据驱动选 k
- **预处理**：知道为什么 KNN 必须先标准化（量纲敏感），以及 fit/transform 的训练-测试边界
- **回归**：知道分类与回归的统一性（投票 vs 平均），以及评估指标差异（accuracy vs MAE/RMSE/R²）
- **工程**：能写出 6 步标准 pipeline 代码（load → split[stratify] → scale → fit → eval → predict）

## 与下一站的衔接

- → **02-LR 线性回归**：从"懒得训练"切到"显式拟合参数"，引入梯度下降 + 损失函数视角
- → **04-DecisionTree**：另一种非参数模型，不靠距离靠分裂规则
- → **04b-scaling 已铺垫**：所有距离 / 梯度类算法的预处理通用逻辑
