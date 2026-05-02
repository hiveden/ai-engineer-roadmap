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

### PPT 原文（slide 109-120）

#### Slide 109 · 欠拟合与过拟合 – 正则化（思想）

> **当前的模型问题**：模型试图充分利用所有的输入特征来进行预测（寻找输入和标签之间的映射规律），无论特征是否有效。
>
> **每个特征对应的权重（的绝对值）可以认为是模型对于某个特征的"重视程度"**（即该特征是否关键）。
>
> **面临的问题**：
>
> 1. 无效特征也被模型利用了，导致过拟合
> 2. 模型对某单一特征过于重视（权重特别大），忽略其他特征，导致过拟合
>
> **对策**：降低模型复杂度，避免过拟合。

- **L1 正则**：在损失函数中添加权重绝对值之和作为惩罚项，旨在促使模型参数稀疏化（即让部分权重变为零），从而实现**特征选择**
- **L2 正则**：在损失函数中添加权重平方和作为惩罚项，让模型权重整体衰减（分布平滑），防止模型对任何单一特征过度敏感

> **备注**：L2 让模型不要光盯着一个特征（比如房屋面积），也看看其他的特征。权重等比衰减，大权重衰减更多，从而让不同特征的受重视程度被拉近。

#### Slide 110 · 正则项如何起作用

> **备注**：根据公式，第一项是正常的梯度项，决定了模型参数的大小；第二项是 L1 正则项，作用是对所有的参数往 0 靠近。两者作用在打架——对于一些不太重要的特征，其权重参数本来就不大，容易被正则项变成 0，从而对应的特征完全失效。

#### Slide 111 · L1 正则化

- **λ 叫做惩罚系数**，该值越大则权重调整的幅度就越大，即对特征权重惩罚力度就越大
- L1 正则化：在损失函数中添加 L1 正则化项 → $\lambda (\lvert w_1 \rvert + \lvert w_2 \rvert + \cdots + \lvert w_n \rvert)$
- L1 正则化会使得权重**趋向于 0，甚至等于 0**，使得某些特征失效，达到**特征筛选**的目的
- L1 正则化项对某权重 w 的导数：用 sign 函数（在 w=0 处导数不存在）
- 函数形状：y = |x|（V 字形）

#### Slide 112 · L2 正则化

- **λ 叫做惩罚系数**，该值越大则权重调整的幅度就越大
- L2 正则化：在损失函数中添加 L2 正则化项 → $\lambda (w_1^2 + w_2^2 + \cdots + w_n^2)$
- L2 正则化会使得权重**趋向于 0，一般不等于 0**

> **备注**：类比**橡皮筋**——把权重往 0 拉但不会拉到 0。

#### Slide 113 · API

- 使用 L1 正则化的线性回归模型是 **Lasso 回归**：`from sklearn.linear_model import Lasso`
- 使用 L2 正则化的线性回归模型是 **岭回归**：`from sklearn.linear_model import Ridge`

#### Slide 114 · L1 实现 – Lasso 回归

```python
from sklearn.linear_model import Lasso

def dm04_模型过拟合_L1正则化():
    np.random.seed(666)
    x = np.random.uniform(-3, 3, size=100)
    y = 0.5 * x**2 + x + 2 + np.random.normal(0, 1, size=100)

    # alpha 惩罚力度越来越大 → k 值越来越小，最后会欠拟合
    estimator = Lasso(alpha=0.005, normalize=True)  # normalize=True 已弃用

    X = x.reshape(-1, 1)
    X3 = np.hstack([X, X**2, X**3, X**4, X**5, X**6, X**7, X**8, X**9, X**10])
    estimator.fit(X3, y)
    print('estimator.coef_', estimator.coef_)

    y_predict = estimator.predict(X3)
    myret = mean_squared_error(y, y_predict)
    print('myret-->', myret)

    plt.scatter(x, y)
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')
    plt.show()
```

> Lasso 回归 L1 正则——**会将高次方项系数变为 0**

#### Slide 115 · L2 实现 – Ridge 回归

```python
from sklearn.linear_model import Ridge

def dm05_模型过拟合_L2正则化():
    np.random.seed(666)
    x = np.random.uniform(-3, 3, size=100)
    y = 0.5 * x**2 + x + 2 + np.random.normal(0, 1, size=100)

    estimator = Ridge(alpha=0.005, normalize=True)

    X = x.reshape(-1, 1)
    X3 = np.hstack([X, X**2, X**3, X**4, X**5, X**6, X**7, X**8, X**9, X**10])
    estimator.fit(X3, y)
    print('estimator.coef_', estimator.coef_)

    y_predict = estimator.predict(X3)
    myret = mean_squared_error(y, y_predict)
    print('myret-->', myret)

    plt.scatter(x, y)
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')
    plt.show()
```

> Ridge 线性回归 L2 正则——**不会将系数变为 0**，但是对高次方项系数影响较大。
>
> **工程开发中 L1、L2 使用建议**：一般倾向使用 **L2** 正则。

#### Slide 116 · 总结

| 现象 | 描述 | 解决方法 |
|---|---|---|
| **欠拟合** | 训练集表现不好，测试集也不好 | 添加其他特征项 / 添加多项式特征 |
| **过拟合** | 训练集表现好，测试集不好 | 重新清洗数据 / 增大训练量 / **正则化** / 减少特征维度 |

#### Slide 117 · 正则化小结

- **正则化**：异常点数据会造成权重系数过大、过小，尽量减少这些特征的影响（甚至删除某个特征的影响），这就是正则化
- 为了减少过拟合的影响，控制模型的参数（尤其是高次项的权重参数）
- **L1 正则化（Lasso 回归）**：会**直接把高次项前面的系数变为 0**
- **L2 正则化（岭回归）**：把高次项前面的系数变成**特别小的值**

#### Slide 118 · 自检题（欠拟合 vs 过拟合）

> 1. 下列关于欠拟合与过拟合的描述正确的是？
>
> A）欠拟合：模型学习到的特征过少，无法准确的预测未知样本
> B）过拟合：模型学习到的特征过多，导致模型只能在训练样本上得到较好的预测结果，而在未知样本上的效果不好
> C）欠拟合可以通过增加特征来解决
> D）过拟合可以通过正则化、异常值检测、特征降维等方法来解决
>
> **答案：ABCD**
>
> 解析：A 欠拟合出现的原因 / B 过拟合出现的原因 / C 增加模型复杂度 / D 降低模型复杂度。

#### Slide 119 · 自检题（正则化）

> 2. 下列关于过拟合问题的解决方式以及描述正确的是？
>
> A）使用岭回归能够防止训练所得的模型发生过拟合
> B）使用 Lasso 回归也能防止模型产生过拟合，这时所得模型的权重系数部分为 0
> C）L2 正则化能够让模型产生一些平滑的权重系数
> D）Early stopping 是当模型训练到某个固定的验证错误率阈值时，及时停止模型训练
>
> **答案：ABCD**

#### Slide 120 · 自检题（填空）

> 填空：
>
> ① `sklearn.linear_model.Ridge()` 岭回归 API 中：alpha 表示正则化系数，正则化系数越大，表示正则化力度 **越大**，所得模型的权重系数 **越小**；反之，所得模型的权重系数 **越大**。
>
> ② `sklearn.linear_model.SGDRegressor()` 使用随机梯度下降法优化的线性回归 API：当它的参数 `penalty` 为 `l2`、参数 `loss` 为 `squared_loss` 时，达到的效果与上述的岭回归 API 相同，只不过 SGDRegressor 只能使用 **普通的随机梯度下降法**去优化损失，而 Ridge 的选择则更加丰富。

---

### 笔记（已整理）

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
