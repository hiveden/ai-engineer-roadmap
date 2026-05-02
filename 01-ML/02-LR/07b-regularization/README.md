# 第 7 章 b · 正则化

> 本文件 = 章 PPT 完整底稿（复习记忆页）
> 维度：**数学 + 代码**
> 学习目标见 [`07a-overfit/README.md`](../07a-overfit/README.md)
>
> **按知识点拆分的讲解版**：
>
> 1. [`01-L1正则化.md`](./01-L1正则化.md) — Lasso
> 2. [`02-L2正则化.md`](./02-L2正则化.md) — Ridge
> 3. [`03-正则化案例.md`](./03-正则化案例.md) — 【实操】

## 底稿

> 07 · 正则化

> 【理解】正则化

在解决回归过拟合中，我们选择正则化。但是对于其他机器学习算法如分类算法来说也会出现这样的问题，除了一些算法本身作用之外（决策树、神经网络），我们更多的也是去自己做特征选择，包括之前说的删除、合并一些特征。

**核心思想**：在原损失函数 $L(W)$ 后**加一个惩罚项**，让权重 $w_i$ 不要变得太大。

$$L_{\text{new}} = L(W) + \lambda \cdot \text{penalty}(W)$$

- 当模型为了拟合噪声把某个 $w_i$ 学得很大时（即过拟合症状），惩罚项也会变大
- 优化器为了**总损失**最小，不得不在"拟合训练" 和 "权重小" 之间折中
- $\lambda$（lambda）是控制正则化强度的超参，越大越偏向"权重压小"

**如何解决过拟合**：

| 手段 | 何时用 |
|---|---|
| **L1 正则**（Lasso） | 想让特征自动稀疏化（部分 $w_i = 0$ 删特征） |
| **L2 正则**（Ridge） | 想让权重整体小但不为 0（更稳的标准做法） |
| 减少高次项 | 多项式回归手动控制次数 |
| 加数据 | 从根上降低过拟合风险 |
| 早停 early stopping | GD 训练时验证集误差开始上升就停 |

**在学习的时候，数据提供的特征有些影响模型复杂度或者这个特征的数据点异常较多，所以算法在学习的时候尽量减少这个特征的影响（甚至删除某个特征的影响），这就是正则化。**

注：调整时候，算法并不知道某个特征影响，而是去调整参数得出优化的结果。

> L1 正则化

- 假设 $L(W)$ 是未加正则项的损失，$\lambda$ 是一个超参，控制正则化项的大小
- 则最终的损失函数：$L = L(W) + \lambda \cdot \sum_{i=1}^{n} \lvert w_i \rvert$

**作用**：用来进行特征选择，主要原因在于 L1 正则化会使得较多的参数为 0，从而产生稀疏解，可以将 0 对应的特征遗弃，进而用来选择特征。一定程度上 L1 正则也可以防止模型过拟合。

**几何直觉**：L1 的等高线是**菱形**（顶点在坐标轴上），原损失等高线（椭圆）与菱形相切时，
切点很容易**落在顶点**——而顶点意味着某些 $w_i = 0$。这就是 L1 产生稀疏解的几何原因。

**L1 正则为什么可以产生稀疏解（可以特征选择）**

稀疏性：向量中很多维度值为 0。

- 对其中的一个参数 $w_i$ 计算梯度，其他参数同理，α 是学习率，sign(wi) 是符号函数

**梯度下降更新规则**：

$$w_i \leftarrow w_i - \alpha \cdot \frac{\partial L(W)}{\partial w_i} - \alpha \lambda \cdot \text{sign}(w_i)$$

注意右边那项 $\alpha \lambda \cdot \text{sign}(w_i)$：
- $w_i > 0$ 时减一个常数（往 0 靠）
- $w_i < 0$ 时加一个常数（往 0 靠）
- $w_i = 0$ 时不动 → **一旦归零就保持归零**

→ 这是一种"**等量收缩**"，与 L2 的"按比例收缩"不同。等量收缩能让小权重直接归零（特征剔除）。

L1 的梯度：

$$L = L(W) + \lambda \cdot \sum_{i=1}^{n} \lvert w_i \rvert$$

$$\frac{\partial L}{\partial w_i} = \frac{\partial L(W)}{\partial w_i} + \lambda \cdot \text{sign}(w_i)$$

LASSO 回归：

```python
from sklearn.linear_model import Lasso
```

> L2 正则化

- 假设 $L(W)$ 是未加正则项的损失，$\lambda$ 是一个超参，控制正则化项的大小
- 则最终的损失函数：$L = L(W) + \lambda \cdot \sum_{i=1}^{n} w_i^2$

**作用**：主要用来防止模型过拟合，可以减小特征的权重。

**优点**：越小的参数说明模型越简单，越简单的模型则越不容易产生过拟合现象。

**梯度下降更新规则**：

$$w_i \leftarrow w_i - \alpha \cdot \frac{\partial L(W)}{\partial w_i} - 2 \alpha \lambda \cdot w_i = (1 - 2\alpha\lambda) w_i - \alpha \cdot \frac{\partial L(W)}{\partial w_i}$$

注意 $(1 - 2\alpha\lambda) w_i$：每次更新都把 $w_i$ **乘一个 < 1 的因子**（按比例收缩），
所以权重越来越小但**不会直接归零**——这就是 L2 与 L1 的本质区别。

Ridge 回归：

```python
from sklearn.linear_model import Ridge
```

> 正则化案例

**对比 LR / Lasso / Ridge 在多项式过拟合数据上的表现**：

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 构造抛物线 + 噪声数据
np.random.seed(666)
x = np.random.uniform(-3, 3, size=100).reshape(-1, 1)
y = 0.5 * x.ravel()**2 + x.ravel() + 2 + np.random.normal(0, 1, 100)

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=5)
```

**方案 1：纯多项式 LR（10 次项）→ 过拟合基线**：

```python
poly_lr = Pipeline([
    ("poly", PolynomialFeatures(degree=10)),
    ("scaler", StandardScaler()),
    ("lr", LinearRegression()),
])
poly_lr.fit(X_train, y_train)
print("LR  train MSE:", mean_squared_error(y_train, poly_lr.predict(X_train)))
print("LR  test  MSE:", mean_squared_error(y_test, poly_lr.predict(X_test)))
print("LR  权重:", poly_lr.named_steps["lr"].coef_)
# 训练 ↓ 测试 ↑（过拟合特征：训练 0.8 / 测试 1.5+）
```

**方案 2：Lasso（L1）→ 自动剔除高次项**：

```python
poly_lasso = Pipeline([
    ("poly", PolynomialFeatures(degree=10)),
    ("scaler", StandardScaler()),
    ("lasso", Lasso(alpha=0.1, max_iter=10000)),
])
poly_lasso.fit(X_train, y_train)
print("Lasso 权重:", poly_lasso.named_steps["lasso"].coef_)
# 大部分高次项权重 = 0（稀疏解）
# 测试 MSE 接近 1.0（接近真实噪声水平）
```

**方案 3：Ridge（L2）→ 整体收缩，权重平滑**：

```python
poly_ridge = Pipeline([
    ("poly", PolynomialFeatures(degree=10)),
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0)),
])
poly_ridge.fit(X_train, y_train)
print("Ridge 权重:", poly_ridge.named_steps["ridge"].coef_)
# 所有权重都被压小但不为 0
# 测试 MSE 接近 1.0
```

**调 alpha**：alpha 越大正则越强 → 训练误差↑ 但测试误差先↓再↑（U 形）。
工程实践用 `LassoCV` / `RidgeCV` 自动 CV 选 alpha。
