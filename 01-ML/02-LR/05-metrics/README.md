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
