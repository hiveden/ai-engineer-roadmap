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
sklearn.linear_model.SGDRegressor(loss="squared_loss", fit_intercept=True, learning_rate='constant', eta0=0.01)
```

- 参数：loss（损失函数类型），fit_intercept（是否计算偏置），learning_rate（学习率）
- 属性：`SGDRegressor.coef_`（回归系数），`SGDRegressor.intercept_`（偏置）

> 【实操】波士顿房价预测

⚠️ 待补充

#### 案例背景介绍

数据介绍。

⚠️ 待补充

⚠️ 待补充

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

⚠️ PPT 笔记此处代码块为空（待补）。

```python

```

1.2.0 以上版本实现：

```python

```
