---
tags: [回归/案例, sklearn/API, 实操]
---

# API 复习 · LinearRegression vs SGDRegressor

> 维度：API
> 知识点级别：【知道】案例前再过一遍两套回归 API 的参数和属性，避免实战时翻文档
> 章节底稿全文见 [`README.md`](./README.md)（PPT slide 90 跨章引用 + 笔记 §06 线性回归 API）

## ━━━━━━━━ 底稿 ━━━━━━━━

### PPT

> Slide 90（跨章引用） · 线性回归 正规方程法 + 梯度下降 API 对比

```python
sklearn.linear_model.LinearRegression(fit_intercept=True)
```

- **参数**：`fit_intercept`（是否计算偏置）
- **属性**：`LinearRegression.coef_`（回归系数）、`LinearRegression.intercept_`（偏置）

```python
sklearn.linear_model.SGDRegressor(
    loss="squared_error",       # 注：原 PPT 写 "squared_loss"，sklearn 1.0+ 已改为 "squared_error"
    fit_intercept=True,
    learning_rate='constant',
    eta0=0.01,
)
```

`SGDRegressor` 类实现了随机梯度下降学习，支持不同的损失函数和正则化惩罚项来拟合线性回归模型。

- **参数**：`loss` / `fit_intercept` / `learning_rate`（如 `'invscaling'`：eta = eta0 / pow(t, power_t=0.25)）/ `eta0`
- **属性**：`coef_` / `intercept_`

> sklearn 提供两种实现的 API，根据需要选择使用。
>
> **备注**：LinearRegression 是正规方程法，但**不求逆矩阵**，而是用 **奇异值分解（SVD）**。

### 笔记

> 【知道】线性回归 API

- 通过正规方程优化：`LinearRegression(fit_intercept=True)`
- 梯度下降优化：`SGDRegressor(loss="squared_error", fit_intercept=True, learning_rate='constant', eta0=0.01)`
- 两者共有属性：`coef_`（回归系数）、`intercept_`（偏置）

---

## ━━━━━━━━ 讲解 ━━━━━━━━

进案例前先把两套回归 API 的参数和属性过一遍，避免实战时翻文档。完整对比见 [`02-api/`](../02-api/)。

### 两套 API 一表对比

| 维度 | `LinearRegression` | `SGDRegressor` |
|---|---|---|
| 求解方式 | 正规方程（内部 SVD） | 随机梯度下降 |
| 适用规模 | 小到中（< 10 万样本） | 大数据 / 在线学习 |
| 损失函数 | 固定 MSE | `loss` 可选（squared_error / huber / epsilon_insensitive） |
| 学习率 | 无 | `learning_rate` + `eta0` 控制 |
| 是否需要标准化 | 不强制（但有数值优势） | **强制**（GD 对量纲敏感） |
| 是否凸优化 | 是 | 是（收敛到同一最优解） |

> Tech Lead 视角：小数据先用 `LinearRegression`，省得调学习率。等样本破百万、内存吃紧再换 SGD。

### LinearRegression 关键签名

```python
from sklearn.linear_model import LinearRegression

LinearRegression(
    fit_intercept=True,   # 是否学习偏置 b
    copy_X=True,          # 是否复制 X（避免被原地修改）
    n_jobs=None,          # 多核加速（大特征矩阵下生效）
)
```

- `fit_intercept=False` 的场景：你已经在 X 里手动加了一列全 1，或者业务上确认 y 截距必为 0
- 99% 情况保持默认即可

### SGDRegressor 关键参数

```python
from sklearn.linear_model import SGDRegressor

SGDRegressor(
    loss="squared_error",      # 损失函数（默认 MSE）
    fit_intercept=True,
    learning_rate="invscaling",# constant / optimal / invscaling / adaptive
    eta0=0.01,                 # 初始学习率
    max_iter=1000,             # 最大迭代轮数
    random_state=22,
)
```

**`learning_rate` 的四种策略**：

| 策略 | 含义 | 何时用 |
|---|---|---|
| `constant` | 学习率恒定 = `eta0` | 调试期，想看清收敛曲线 |
| `optimal` | 按公式 `1/(alpha*(t+t0))` 衰减 | 默认推荐 |
| `invscaling` | `eta0 / pow(t, power_t)` | 收敛后期需要细调 |
| `adaptive` | 损失不降则减半 | 想要稳健自动衰减 |

`eta0` 太大 → 振荡不收敛；太小 → 训练慢。0.001 到 0.1 之间扫一遍是惯例。

### 共有属性（训练完才有）

```python
model.coef_         # 权重向量 w，shape = (n_features,)
model.intercept_    # 偏置 b（标量或长度为 1 的数组）
```

这两个就是模型的全部参数。**保存这俩值就能在 Go / Java / JS 里复现推理**——见 [`05-代码实现.md`](./05-代码实现.md) 末尾的部署小节。

### 共有方法

```python
model.fit(X_train, y_train)         # 训练
y_pred = model.predict(X_test)      # 预测
score = model.score(X_test, y_test) # 默认返回 R²（不是 MSE，注意）
```

> 坑点：`LinearRegression.score()` 返回的是 **R²**（决定系数），不是 MSE。要 MSE 自己调 `mean_squared_error`。这点和 `KNeighborsClassifier.score()` 返回 accuracy 是统一约定——「score 越大越好」。

> Sources：
> - PPT Slide 90（05-metrics 章末跨章引用）
> - 笔记（README §06 线性回归 API）
> - [scikit-learn · LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
> - [scikit-learn · SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)
