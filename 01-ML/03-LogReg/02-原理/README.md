# 第 2 章 · 原理

> 维度：**算法 + 数学**
>
> **按知识点拆分的讲解版**：
>
> 1. [`01-原理.md`](./01-原理.md) — 【理解】Logistic Regression
> 2. [`02-损失函数.md`](./02-损失函数.md) — 【知道】对数似然损失

## 底稿

> 02 · 逻辑回归原理

**学习目标**：

1. 理解逻辑回归算法的原理
2. 知道逻辑回归的损失函数

> 【理解】原理

**Logistic Regression**：分类模型。把线性回归的输出 $z = wx + b$ 喂给 sigmoid，得到 $(0, 1)$ 区间的概率值。

**两步走**：

1. 线性模型按特征重要性算一个分 $z = wx + b$
2. sigmoid 把 $z$ 压到 $(0,1)$ 当概率；阈值（默认 0.5）切成 0/1 两类

**假设函数**：

$$h_w(x) = \sigma(wx + b) = \frac{1}{1 + e^{-(wx + b)}}$$

**Sigmoid**：

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

性质：$z \to +\infty$ 趋近 1；$z \to -\infty$ 趋近 0；$z = 0$ 时为 0.5。所以"线性回归 + 阈值切分" = 二分类。

**决策**：$h_w(x) \geq 0.5 \Rightarrow \hat{y} = 1$，否则 $\hat{y} = 0$。等价于 $wx + b \geq 0$，即决策边界是 $wx + b = 0$（线性超平面）。

**代码 API**：

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)
clf.predict(X_test)        # 返回 0/1 标签（阈值 0.5）
clf.predict_proba(X_test)  # 返回 [P(y=0), P(y=1)] 概率
```

> 【知道】损失函数

**问题**：MSE 用在分类上，loss 曲面非凸（多个局部最小值），梯度下降会卡住。需要换损失。

**对数似然损失**（log loss / binary cross-entropy）——单样本：

$$\mathrm{loss}(h_w(x), y) = \begin{cases} -\log(h_w(x)) & y = 1 \\ -\log(1 - h_w(x)) & y = 0 \end{cases}$$

**直觉**：真实标签是 1 时，预测概率越接近 1 损失越小；接近 0 损失趋于 $+\infty$。预测错了被无情放大。

**合并写法**（用 $y \in \{0, 1\}$ 当开关）：

$$\mathrm{loss}(h_w(x), y) = -y \log(h_w(x)) - (1 - y) \log(1 - h_w(x))$$

**整体损失**（$m$ 个样本求和/平均）：

$$J(w) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(h_w(x_i)) + (1 - y_i) \log(1 - h_w(x_i)) \right]$$

**为什么这个损失好**：与 sigmoid 配合后 $J(w)$ 是凸函数，梯度下降稳定收敛到全局最优。

**代码 API**：

```python
from sklearn.metrics import log_loss
log_loss(y_true, y_pred_proba)  # y_pred_proba 是 predict_proba 的输出
```
