# 第 6 章 · 波士顿房价预测案例

> 本文件 = 章 PPT 完整底稿（复习记忆页）
> 维度：**代码**
>
> **按知识点拆分的讲解版**：
>
> 1. [`01-API复习.md`](./01-API复习.md) — 【知道】LinearRegression 再过一遍
> 2. [`02-案例背景.md`](./02-案例背景.md) — 【知道】
> 3. [`03-案例分析.md`](./03-案例分析.md) — 【实操】
> 4. [`04-性能评估.md`](./04-性能评估.md) — 【实操】
> 5. [`05-代码实现.md`](./05-代码实现.md) — 【实操】完整 PPT 代码

## 底稿

> 06 · 【实操】波士顿房价预测案例

> 【知道】线性回归 API

```python
sklearn.linear_model.LinearRegression(fit_intercept=True)
```

- 通过正规方程优化
- 参数：fit_intercept，是否计算偏置
- 属性：`LinearRegression.coef_`（回归系数），`LinearRegression.intercept_`（偏置）

```python
sklearn.linear_model.SGDRegressor(loss="squared_error", fit_intercept=True, learning_rate='constant', eta0=0.01)
```

- 参数：loss（损失函数类型），fit_intercept（是否计算偏置），learning_rate（学习率）
- 属性：`SGDRegressor.coef_`（回归系数），`SGDRegressor.intercept_`（偏置）

> 【实操】波士顿房价预测

**目标**：用线性回归预测波士顿地区房价中位数（千美元）。

#### 案例背景介绍

**经典数据集**：13 个特征 + 1 个目标值（房价中位数 MEDV），506 条样本。

> ⚠️ **sklearn 1.2+ 已移除 `load_boston`**（因数据集含种族变量被认定有伦理问题）。
> 推荐用 **`fetch_california_housing`** 或 **`fetch_openml('boston')`** 替代。
> 本章代码以 California 数据集演示流程，原 PPT 的 boston 字段说明仍保留作历史参考。

**Boston 13 特征**（历史参考）：

| 简称 | 含义 |
|---|---|
| CRIM | 城镇人均犯罪率 |
| ZN | 大宅区域比例 |
| INDUS | 非零售商业用地比例 |
| CHAS | 是否邻近查尔斯河（0/1） |
| NOX | 一氧化氮浓度 |
| RM | 平均房间数 |
| AGE | 1940 年前自住房比例 |
| DIS | 到 5 个就业中心加权距离 |
| RAD | 高速公路通达指数 |
| TAX | 每万美元房产税率 |
| PTRATIO | 师生比 |
| B | 黑人比例（已废弃字段） |
| LSTAT | 低收入人口比例 |
| **MEDV** | **目标：房价中位数（千美元）** |

> 给定的这些特征，是专家们得出的影响房价的结果属性。我们此阶段不需要自己去探究特征是否有用，只需要使用这些特征。到后面量化很多特征需要我们自己去寻找。

#### 案例分析

回归当中的数据大小不一致，是否会导致结果影响较大。所以需要做标准化处理。

- 数据分割与标准化处理
- 回归预测
- 线性回归的算法效果评估

#### 回归性能评估

均方误差（Mean Squared Error, MSE）评价机制：

$$MSE = \frac{1}{m}\sum_{i=1}^{m}(y^i - \hat{y})^2$$

sklearn 中的 API：`sklearn.metrics.mean_squared_error(y_true, y_pred)`

- 均方误差回归损失
- y_true：真实值
- y_pred：预测值
- return：浮点数结果

#### 代码实现

**完整 6 步 pipeline**（用 California 数据集替代 Boston）：

```python
"""
线性回归 · 加州房价预测（替代废弃的 Boston 数据集）
"""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 1 加载数据
data = fetch_california_housing()
X, y = data.data, data.target  # X: (20640, 8), y: 房价（10万美元）

# 2 拆分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=22,
)

# 3 标准化（LR + GD 都建议）
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 4 训练 - 方案 A：正规方程（LinearRegression 内部用 SVD）
model_ne = LinearRegression()
model_ne.fit(X_train_s, y_train)

# 4' 训练 - 方案 B：随机梯度下降
model_sgd = SGDRegressor(
    loss="squared_error", learning_rate="invscaling",
    eta0=0.01, max_iter=1000, random_state=22,
)
model_sgd.fit(X_train_s, y_train)

# 5 评估
for name, model in [("正规方程", model_ne), ("SGD", model_sgd)]:
    y_pred = model.predict(X_test_s)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: MSE={mse:.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
    print(f"  权重前 3: {model.coef_[:3]}")
    print(f"  截距: {model.intercept_}")

# 6 预测新样本
new_x = np.array([[8.3, 41, 6.9, 1.0, 322, 2.5, 37.88, -122.23]])  # 8 维
new_x_s = scaler.transform(new_x)
print("预测房价（10万美元）:", model_ne.predict(new_x_s))
```

**典型输出**（参考值，不同 random_state 略有差异）：

```
正规方程: MSE=0.555  MAE=0.533  RMSE=0.745  R²=0.596
SGD:      MSE=0.557  MAE=0.534  RMSE=0.746  R²=0.594
```

**两种解法结果几乎一致**——验证了正规方程和 SGD 在凸优化问题上都收敛到同一最优解。
