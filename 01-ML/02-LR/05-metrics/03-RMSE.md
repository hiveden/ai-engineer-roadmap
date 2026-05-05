---
tags: [回归/评估, 评估/RMSE]
---

# 均方根误差 RMSE

> 维度：概念
> 知识点级别：【知道】RMSE = √MSE，单位回到 y；工业默认报告指标；不能当 loss（求导难）
> 章节底稿全文见 [`README.md`](./README.md)（PPT slide 83 RMSE 段, 84）

## ━━━━━━━━ 底稿 ━━━━━━━━

### PPT

> Slide 83 · 线性回归模型评估（RMSE 部分）

**均方根误差** Root Mean Squared Error (RMSE)

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} = \sqrt{\text{MSE}}$$

- n 为样本数量, y 为实际值, ŷ 为预测值
- RMSE 越小模型预测越准确
- **RMSE 是 MSE 的平方根，某些情况下比 MSE 更有用**

> **RMSE 不适合当 loss 函数 —— 求导难！**

> Slide 84 · 评估函数 vs loss 函数

一些函数被被拿来用作模型评估函数，也同时可以被当做 loss 函数。

### 笔记

> 【知道】均方根误差

- n 为样本数量，y 为实际值，$\hat{y}$ 为预测值
- RMSE 越小模型预测约准确
- **单位回到目标变量本身**（解决了 MSE 平方单位的问题），同时保留 MSE 对大误差敏感的特性

---

## ━━━━━━━━ 讲解 ━━━━━━━━

### 直觉

一句话：**RMSE = MSE 但单位回到 y** —— 报告评估指标时的工业默认。

MSE 有个硬伤：单位是 y²，没法解释。开根号一切恢复正常：

| 指标 | 房价数值 | 含义 |
|---|---|---|
| MSE | 25000000 | 「元²」业务方一脸懵 |
| **RMSE** | 5000 | 「元」 → 「平均偏差大约 5000 元」可解释 |
| MAE | 4200 | 「元」 → 「平均偏差 4200 元」更字面 |

注意 **RMSE 通常比 MAE 略大**：因为平方再开根号会「放大」误差分布的离散度。两者相等只在所有误差都相同时成立。

### 解决了 MSE 单位问题，保留对大误差敏感

开根号是**单调函数**，不改变排序：

- 模型 A 的 MSE < 模型 B 的 MSE → 模型 A 的 RMSE 也 < 模型 B 的 RMSE
- 离群点对 MSE 的「平方放大」效应 → 开根号后**仍然存在**，只是数值不爆炸

→ **RMSE = MSE 的可解释版本**，二者的「敏感性结论」完全一致。

### RMSE 不能当 loss 函数

PPT 强调过：**RMSE 不适合当 loss —— 求导难**。

为什么？

$$\frac{\partial \text{RMSE}}{\partial w} = \frac{1}{2\sqrt{\text{MSE}}} \cdot \frac{\partial \text{MSE}}{\partial w}$$

多了一个 $\frac{1}{2\sqrt{\text{MSE}}}$ 因子，且 MSE = 0 时**梯度爆炸**（除以 0）。MSE 没这个问题，所以训练永远用 MSE，**评估**才用 RMSE。

> 训练用 MSE，报告用 RMSE —— 一对黄金搭档。

### sklearn API（注意版本）

#### sklearn 1.5+（2024 起的现代写法）

```python
from sklearn.metrics import root_mean_squared_error

rmse = root_mean_squared_error(y_test, y_predict)
```

#### sklearn 1.4 及以前（旧写法，**已废弃**）

```python
from sklearn.metrics import mean_squared_error

# 老写法 1：squared=False（在 1.6 中已移除）
rmse = mean_squared_error(y_test, y_predict, squared=False)

# 老写法 2：手动开根号（永远兼容）
import numpy as np
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
```

→ **生产代码用新 API `root_mean_squared_error`**，老代码看到 `squared=False` 知道在做什么即可。

### 数值小例

误差 [1, 3]：

$$\text{RMSE} = \sqrt{\frac{1 + 9}{2}} = \sqrt{5} \approx 2.24$$

加一个离群点 [1, 3, 100]：

$$\text{RMSE} = \sqrt{\frac{1 + 9 + 10000}{3}} \approx 57.76$$

变化倍数 = 57.76 / 2.24 ≈ **25.8 倍**。

| 指标 | 离群点前 | 离群点后 | 倍数 |
|---|---|---|---|
| MAE | 2 | 34.7 | 17× |
| **RMSE** | 2.24 | 57.76 | **25.8×** |
| MSE | 5 | 3336.7 | 667× |

→ RMSE 的离群点敏感度**介于 MAE 和 MSE 之间，但单位还是 y**，所以是工业上的默认报告指标。

### 报告评估的首选

业内默认结论：

> **训练用 MSE，报告用 RMSE，异常点多用 MAE。**

写在 paper / 技术报告 / Kaggle 提交时，第一指标基本都是 RMSE。例外只在两种场景：
- 异常点多 → 改 MAE
- 分类问题 → 改 Accuracy / F1 / AUC（见 [`../../03-LogReg/`](../../03-LogReg/)）

### 衔接

- 三个指标的全方位对比 → [`04-三种指标比较.md`](./04-三种指标比较.md)
- RMSE 在加州房价案例的实际数值 → [`06-boston/`](../06-boston/)
- 跑 demo 看不同 outlier 比例下 RMSE 的变化 → [`demos/metric-vs-outlier.py`](./demos/metric-vs-outlier.py)

> Sources：
> - PPT Slide 83 (RMSE 段), 84
> - 笔记（README §05 均方根误差）
> - sklearn 文档 `sklearn.metrics.root_mean_squared_error`（1.5+）
