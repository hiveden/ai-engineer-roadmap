# 第 7 章 a · 欠拟合与过拟合

> 本文件 = 章 PPT 完整底稿（复习记忆页）
> 维度：**概念**
>
> **按知识点拆分的讲解版**：
>
> 1. [`01-欠拟合与过拟合.md`](./01-欠拟合与过拟合.md) — 【理解】
> 2. [`02-代码认识过拟合.md`](./02-代码认识过拟合.md) — 【实践】
> 3. [`03-原因和解决办法.md`](./03-原因和解决办法.md) — 【理解】

## 底稿

> 07 · 正则化

**学习目标**：

1. 掌握过拟合、欠拟合的概念
2. 掌握过拟合、欠拟合产生的原因

> 正则化作为解决方案，独立成章 → [`07b-regularization/`](../07b-regularization/)

> 【理解】欠拟合与过拟合

**过拟合**：一个假设 **在训练数据上能够获得比其他假设更好的拟合，但是在测试数据集上却不能很好地拟合数据**（体现在准确率下降），此时认为这个假设出现了过拟合的现象。（模型过于复杂）

**欠拟合**：一个假设 **在训练数据上不能获得更好的拟合，并且在测试数据集上也不能很好地拟合数据**，此时认为这个假设出现了欠拟合的现象。（模型过于简单）

过拟合和欠拟合的区别：

| 维度 | 欠拟合 underfit | 良好 good fit | 过拟合 overfit |
|---|---|---|---|
| **模型复杂度** | 太简单 | 适中 | 太复杂 |
| **训练误差** | 大 | 小 | **极小**（接近 0） |
| **测试误差** | 大 | 小 | **大**（远大于训练） |
| **典型表现** | 直线拟合曲线数据 | 多项式 2-3 次 | 多项式 10+ 次抖动剧烈 |
| **bias / variance** | high bias | balanced | high variance |

- 欠拟合在训练集和测试集上的误差都较大
- 过拟合在训练集上误差较小，而测试集上误差较大

**判断方法**：画"训练误差 vs 测试误差随模型复杂度变化"曲线（learning curve）：
- 训练 ↓、测试 ↑ → 过拟合区
- 训练 ↑、测试 ↑ → 欠拟合区
- 两者都低且接近 → 甜蜜点

> 【实践】通过代码认识过拟合和欠拟合

**绘制数据**（生成抛物线 + 噪声的合成数据）：

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(666)
x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)
y = 0.5 * x**2 + x + 2 + np.random.normal(0, 1, 100)
plt.scatter(x, y); plt.show()
```

**仅用 X（一次项）拟合 → 欠拟合**：

```python
estimator = LinearRegression()
estimator.fit(X, y)
y_predict = estimator.predict(X)

plt.scatter(x, y)
plt.plot(np.sort(x), y_predict[np.argsort(x)], color="r")
plt.show()
```

```python
# 计算均方误差
from sklearn.metrics import mean_squared_error
mean_squared_error(y, y_predict)

# 3.0750025765636577
```

**添加二次项，绘制图像 → 良好拟合**：

```python
X2 = np.hstack([X, X**2])
estimator = LinearRegression()
estimator.fit(X2, y)
y_predict2 = estimator.predict(X2)

plt.scatter(x, y)
plt.plot(np.sort(x), y_predict2[np.argsort(x)], color="r")
plt.show()
```

```python
# 计算均方误差和准确率
from sklearn.metrics import mean_squared_error
mean_squared_error(y, y_predict2)

# 1.0987392142417858
```

**再次加入高次项（X^3 ~ X^10），绘制图像，观察均方误差结果 → 过拟合**：

```python
X5 = np.hstack([X**i for i in range(1, 11)])  # 10 次多项式
estimator = LinearRegression()
estimator.fit(X5, y)
y_predict5 = estimator.predict(X5)

plt.scatter(x, y)
plt.plot(np.sort(x), y_predict5[np.argsort(x)], color="r")
plt.show()

mean_squared_error(y, y_predict5)
# 训练集 MSE 继续下降到 ~0.85，但拟合曲线已经在数据点之间剧烈抖动
```

通过上述观察发现，随着加入的高次项越来越多，拟合程度越来越高，均方误差也随着加入越来越小。说明已经不再欠拟合了。

**问题**：如何判断出现过拟合呢？

将数据集进行划分：对比 X、X²、X⁵ 的测试集的均方误差。

X 的测试集均方误差：

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)
estimator = LinearRegression()
estimator.fit(X_train, y_train)
y_predict = estimator.predict(X_test)

mean_squared_error(y_test, y_predict)
# 3.153139806483088
```

X² 的测试集均方误差：

```python
X_train, X_test, y_train, y_test = train_test_split(X2, y, random_state=5)
estimator = LinearRegression()
estimator.fit(X_train, y_train)
y_predict = estimator.predict(X_test)
mean_squared_error(y_test, y_predict)
# 1.111873885731967
```

X⁵ 的测试集的均方误差：

```python
X_train, X_test, y_train, y_test = train_test_split(X5, y, random_state=5)
estimator = LinearRegression()
estimator.fit(X_train, y_train)
y_predict = estimator.predict(X_test)
mean_squared_error(y_test, y_predict)
# 1.4145580542309835
```

> 【理解】原因以及解决办法

**欠拟合产生原因**：学习到数据的特征过少。

**解决办法**：

1. **添加其他特征项**，有时出现欠拟合是因为特征项不够导致的，可以添加其他特征项来解决
2. **添加多项式特征**，模型过于简单时的常用套路，例如将线性模型通过添加二次项或三次项使模型泛化能力更强

**过拟合产生原因**：原始特征过多，存在一些嘈杂特征，模型过于复杂是因为模型尝试去兼顾所有测试样本。

**解决办法**：

1. 重新清洗数据，导致过拟合的一个原因有可能是数据不纯，如果出现了过拟合就需要重新清洗数据
2. 增大数据的训练量，还有一个原因就是我们用于训练的数据量太小导致的，训练数据占总数据的比例过小
3. **正则化**
4. 减少特征维度
