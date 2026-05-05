# 第 5 章 · 回归评估方法

> 本文件 = 章 PPT 完整底稿（复习记忆页）
> 维度：**概念**
>
> **按知识点拆分的讲解版**：
>
> 1. [`01-MAE.md`](./01-MAE.md) — 【知道】平均绝对误差
> 2. [`02-MSE.md`](./02-MSE.md) — 【知道】均方误差
> 3. [`03-RMSE.md`](./03-RMSE.md) — 【知道】均方根误差
> 4. [`04-三种指标比较.md`](./04-三种指标比较.md) — 【了解】

## 底稿

### PPT 原文（slide 82-90）

#### Slide 82 · 学习目标

- 掌握常用的回归评估方法
- 了解不同评估方法的特点

#### Slide 83 · 线性回归模型评估 – MAE / MSE / RMSE 三种

**为什么要进行线性回归模型的评估**：我们希望衡量预测值和真实值之间的差距，会用到 MAE、MSE、RMSE 多种测评函数进行评价。

**1) 平均绝对误差** Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \lvert y_i - \hat{y}_i \rvert$$

- n 为样本数量, y 为实际值, ŷ 为预测值
- MAE 越小模型预测越准确

```python
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_predict)
```

**2) 均方误差** Mean Squared Error (MSE)

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- n 为样本数量, y 为实际值, ŷ 为预测值
- MSE 越小模型预测越准确

```python
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_predict)
```

**3) 均方根误差** Root Mean Squared Error (RMSE)

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} = \sqrt{\text{MSE}}$$

- n 为样本数量, y 为实际值, ŷ 为预测值
- RMSE 越小模型预测越准确
- **RMSE 是 MSE 的平方根，某些情况下比 MSE 更有用**

> **RMSE 不适合当 loss 函数 —— 求导难！**

#### Slide 84 · 评估函数 vs loss 函数

- 一些函数被被拿来用作模型评估函数，也同时可以被当做 loss 函数

#### Slide 85 · 三种指标对比（场景 1：噪声小）

- RMSE 的计算公式中有一个平方项，因此大的误差将被平方，会增加 RMSE 的值，**大多数情况下 RMSE > MAE**
- 在数据点弥散度小时，**MAE 和 RMSE 非常接近**
- 对误差的**平方级惩罚**

> 用直线 y = 2x + 5 拟合数据时观察。

#### Slide 86 · 三种指标对比（场景 2：含异常点）

- 对比第一张图，数据点弥散度变大，**所有指标都变大了**
- **RMSE 几乎达到 MAE 值的两倍**——RMSE 对异常点更为敏感
- **MSE 对异常点特别敏感**，几乎爆炸（正常数据和异常数据的误差共同决定了该误差，但**异常数据的贡献占了大头**！！！）
- 对误差的平方级**乘法**（PPT 原文 typo，应为"惩罚"——同 slide 85 表述）

#### Slide 87 · 异常点存在时为何推荐 MAE

> 当数据存在大量异常点，相较于 MAE，使用 **MSE 损失函数**异常点对 loss 值的贡献过大。
>
> 在梯度下降的优化过程中，为了让 loss 降得更低（即避免异常值产生很大的 loss），训练结束后即使 MSE loss 降到最低，**模型的拟合会偏向异常值**，从而降低对正常数据分布的拟合精度。
>
> 此时推荐 **MAE**，模型失真不严重。

> **补充说明**：作为 loss 函数，MAE 和 MSE 对模型优化的差异显著。

#### Slide 88 · 综合结论

> 一般使用 **MAE 和 RMSE** 这两个指标。

| 指标 | 反映 | 对大误差 | 对异常点 |
|---|---|---|---|
| **MAE** | "真实"的平均误差 | 不敏感 | 不敏感 |
| **RMSE** | 放大大误差点的影响 | 敏感 | 敏感 |

> 两者都能反映出预测值和真实值之间的误差。

#### Slide 90 · 线性回归 API 对比

```python
sklearn.linear_model.LinearRegression(fit_intercept=True)
```

- **参数**：`fit_intercept`（是否计算偏置）
- **属性**：`coef_`（回归系数）、`intercept_`（偏置）

```python
sklearn.linear_model.SGDRegressor(
    loss="squared_error",       # 注：原 PPT 写 "squared_loss"，sklearn 1.0+ 已改为 "squared_error"
    fit_intercept=True,
    learning_rate='constant',
    eta0=0.01,
)
```

`SGDRegressor` 实现了随机梯度下降学习，支持不同的损失函数和正则化惩罚项来拟合线性回归模型。

- **参数**：
  - `loss`（损失函数类型）
  - `fit_intercept`（是否计算偏置）
  - `learning_rate`（学习率策略）：可配置随迭代次数减小，例如 `'invscaling 逆缩放'`：eta = eta0 / pow(t, power_t=0.25)
  - `eta0=0.01`（学习率的初值）
- **属性**：`coef_`、`intercept_`

> sklearn 提供两种实现的 API，根据需要选择使用。
>
> **备注**：LinearRegression 是正规方程法，但**不求逆矩阵**，而是用 **奇异值分解（SVD）**。

---

### 笔记（已整理）

> 05 · 回归评估方法

**学习目标**：

1. 掌握常用的回归评估方法
2. 了解不同评估方法的特点

**为什么要进行线性回归模型的评估**：我们希望衡量预测值和真实值之间的差距，会用到 MAE、MSE、RMSE 多种测评函数进行评价。

> 【知道】平均绝对误差

**Mean Absolute Error (MAE)**

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \lvert y_i - \hat{y}_i \rvert$$

- 上面的公式中：n 为样本数量，y 为实际值，$\hat{y}$ 为预测值
- MAE 越小模型预测约准确
- 单位 = 目标变量单位（房价是元，体重是 kg），**最容易解释**

Sklearn 中 MAE 的 API：

```python
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_predict)
```

> 【知道】均方误差

**Mean Squared Error (MSE)**

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- 上面的公式中：n 为样本数量，y 为实际值，$\hat{y}$ 为预测值
- MSE 越小模型预测约准确
- 单位 = 目标变量单位的**平方**（不直观），但**对大误差敏感**——线性回归的损失函数就是它

Sklearn 中 MSE 的 API：

```python
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_predict)
```

> 【知道】均方根误差

**Root Mean Squared Error (RMSE)**

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} = \sqrt{\text{MSE}}$$

- 上面的公式中：n 为样本数量，y 为实际值，$\hat{y}$ 为预测值
- RMSE 越小模型预测约准确
- **单位回到目标变量本身**（解决了 MSE 平方单位的问题），同时保留 MSE 对大误差敏感的特性

> 【了解】三种指标的比较

我们绘制了一条直线 **y = 2x + 5** 用来拟合 **y = 2x + 5 + e** 这些数据点，其中 e 为噪声。

| 指标 | 单位 | 对离群值 | sklearn API | 默认用 |
|---|---|---|---|---|
| **MAE** | 同 y | **不敏感**（线性增长） | `mean_absolute_error` | 异常值多时 |
| **MSE** | y² | **极敏感**（平方放大） | `mean_squared_error` | LR 训练损失 |
| **RMSE** | 同 y | 敏感（开方后仍保留差异） | `root_mean_squared_error`（或 `np.sqrt(mse)`） | 报告评估时

**关键差异**（一句话）：RMSE 平方放大大误差，所以**对离群点敏感**；MAE 线性求和，离群点影响有限。

举例：误差 [1, 3] → MAE=2，RMSE=$\sqrt{5}$≈2.24。误差越离散 RMSE 越超过 MAE。

**选哪个**：
- 数据干净 / 想惩罚大误差 → **RMSE**（默认报告指标）
- 异常点多但已知是噪声 → **MAE**（不被几个极端值绑架）
- 不同样本量的两个模型不要直接比 RMSE
