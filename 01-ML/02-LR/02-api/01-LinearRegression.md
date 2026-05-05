---
tags: [回归/线性回归, sklearn/API, sklearn/LinearRegression, 实操]
---

# LinearRegression API

> 维度：代码
> 知识点级别：【实操】4 行代码跑通线性回归（导入 / 实例化 / fit / predict）；coef_ 是权重 w，intercept_ 是截距 b
> 章节底稿全文见 [`README.md`](./README.md)（PPT slide 15-17）

## ━━━━━━━━ 底稿 ━━━━━━━━

### PPT

> Slide 15 · 一元线性回归案例

预测小明身高。需求：小明身高是 176，请预测体重？

| 编号 | 身高 | 体重 |
|---|---|---|
| 1 | 160 | 56.3 |
| 2 | 166 | 60.6 |
| 3 | 172 | 65.1 |
| 4 | 174 | 68.5 |
| 5 | 180 | 75 |
| 6 | 176 | ？ |

> 对于这个回归案例如何利用 API 快速求解呢？

> Slide 16 · 线性回归 API 介绍（5 步流程图）

PPT 原图把流程拆成 5 步：

1. **导入**：`from sklearn.linear_model import LinearRegression`
2. **准备数据**：x = [[160], [166], …]（X 必须 2D），y = [56.3, 60.6, …]
3. **实例化**：`estimator = LinearRegression()`
4. **训练**：`estimator.fit(x, y)` 从数据中获取规律
5. **预测**：`estimator.predict([[176]])`

查看模型参数：
- 斜率 `estimator.coef_`
- 截距 `estimator.intercept_`

> **思考**：这里要不要标准化？（一元、量纲单一可不做；多元量纲悬殊必须做，详见 04b-scaling）

> Slide 17 · 线性回归 API 介绍（完整代码）

```python
from sklearn.linear_model import LinearRegression

def dm01_lr预测小明身高():
    # 1 准备数据 身高和体重
    x = [[160], [166], [172], [174], [180]]
    y = [56.3, 60.6, 65.1, 68.5, 75]

    # 2 实例化 线性回归模型 estimator
    estimator = LinearRegression()

    # 3 训练 线性回归模型 fit()  h(w) = w1·x1 + w2·x2 + b
    estimator.fit(x, y)

    # 4 打印 线性回归模型参数 coef_ intercept_
    print('estimator.coef_-->', estimator.coef_)
    print('estimator.intercept_-->', estimator.intercept_)

    # 5 模型预测 predict()
    myres = estimator.predict([[176]])
    print('myres-->', myres)
```

### 笔记

> 【实操】线性回归 API 的应用

预测小明身高。已知 5 个样本（身高 → 体重），需求：小明身高 176，预测体重。

```python
from sklearn.linear_model import LinearRegression

# 1 准备数据：x 必须是 2D（n 样本 × n 特征），y 是 1D
x = [[160], [166], [172], [174], [180]]  # 身高
y = [56.3, 60.6, 65.1, 68.5, 75]          # 体重

# 2 实例化（内部用 SVD 解正规方程）
model = LinearRegression()

# 3 训练
model.fit(x, y)
print('w:', model.coef_)       # 权重
print('b:', model.intercept_)  # 截距

# 4 预测
print('预测：', model.predict([[176]]))
```

通过线性回归 API 可快速的找到一条红色直线，是怎么求解的呢？

---

## ━━━━━━━━ 讲解 ━━━━━━━━

### 直觉

一句话：**`LinearRegression()` 就是把 [`01-intro/02-线性回归`](../01-intro/02-线性回归.md) 的公式 $\hat{y} = \mathbf{w}^T \mathbf{x} + b$ 装进一个对象** —— `fit()` 算 w 和 b，`predict()` 套公式。整套 API 4 行能跑通。

### API 速查表

#### 构造参数（高频 3 个，其它默认基本不动）

| 参数 | 默认 | 含义 |
|---|---|---|
| `fit_intercept` | `True` | 是否学截距 b。设 `False` 强制 b=0（仅当数据已中心化时用）|
| `copy_X` | `True` | fit 时复制 X，避免被覆盖。内存敏感时设 `False` |
| `n_jobs` | `None` | 多目标回归时并行 CPU 数。单目标无效 |
| `positive` | `False` | 强制所有 w ≥ 0（用于物理约束，如成本不能为负）|

#### 训练后属性（带下划线后缀 `_` 是 sklearn 的 fit 后属性约定）

| 属性 | shape | 含义 |
|---|---|---|
| `coef_` | `(n_features,)` 单目标 / `(n_targets, n_features)` 多目标 | 权重向量 w |
| `intercept_` | 标量 / `(n_targets,)` | 偏置 b |
| `n_features_in_` | int | 训练时见过的特征数（predict 维度不匹配会报错）|
| `rank_` | int | X 的秩（判断特征是否共线）|
| `singular_` | array | X 的奇异值（来自 SVD）|

#### 核心方法

| 方法 | 作用 |
|---|---|
| `fit(X, y, sample_weight=None)` | 训练。`sample_weight` 给样本加权（重要样本权重大）|
| `predict(X)` | 预测。返回 `(n_samples,)` 数组 |
| `score(X, y)` | 返回 R²（决定系数，1 为完美拟合，0 为均值线水平，可负）|

### 完整 5 步代码范例（带详细注释）

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# 1. 准备数据
#    X 必须 2D：哪怕只有 1 个特征也得是 [[160], [166], ...] 而不是 [160, 166, ...]
#    原因：sklearn 内部按 (n_samples, n_features) 处理，1D 它没法判断是 1 样本 N 特征还是 N 样本 1 特征
x = [[160], [166], [172], [174], [180]]
y = [56.3, 60.6, 65.1, 68.5, 75]      # y 是 1D（多目标时才用 2D）

# 2. 实例化
#    fit_intercept=True 默认会学 b；如果数据已中心化（减过均值）可以设 False 省一点
estimator = LinearRegression()

# 3. 训练 fit()
#    内部调 scipy.linalg.lstsq → SVD 分解求最小二乘解
#    时间复杂度 O(n_samples · n_features²)，n_features 大于几千会慢
estimator.fit(x, y)

# 4. 查看模型参数
#    coef_ 是数组（每个特征一个权重），即使只有 1 个特征也是 array([0.9x])
#    intercept_ 是标量（除非多目标回归）
print('coef_      ->', estimator.coef_)        # ≈ [0.92...]
print('intercept_ ->', estimator.intercept_)   # ≈ -93.7...

# 5. 预测 predict()
#    输入也必须 2D，[[176]] 表示 1 个样本 1 个特征
y_hat = estimator.predict([[176]])
print('predict    ->', y_hat)                  # ≈ [69.x]

# 顺手验证：手动套公式 y = w·x + b 应该和 predict 一致
manual = estimator.coef_[0] * 176 + estimator.intercept_
print('manual     ->', manual)
```

### 常见坑

#### 1. X 必须 2D

1D 数组会报 `Reshape your data either using array.reshape(-1, 1) if your data has a single feature`。修复：`np.array(x).reshape(-1, 1)` 或直接写嵌套 list。

#### 2. `fit_intercept=False` 不要乱设

默认 `True` 永远是安全的。设 `False` 只在两种场景：(a) 数据已经手动中心化（X 和 y 都减了均值）；(b) 业务上确信 x=0 时 y 必须等于 0（如经过原点的物理定律）。乱设会让模型被迫穿过原点，欠拟合。

#### 3. SVD 而不是直接求逆

正规方程理论解是 $\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$，但 sklearn **不**这么算 —— 它走 `scipy.linalg.lstsq` 的 SVD 路径。原因：当特征共线（$X^T X$ 接近奇异）时直接求逆数值不稳定甚至失败；SVD 用伪逆，在共线情况下也能给出最小范数解。代价是 SVD 比求逆贵 2-3 倍，但稳定性优先。详见 [`04a-analytical/`](../04a-analytical/)。

#### 4. `LinearRegression` 不做特征缩放

解析解对量纲不敏感（数学上 $w_i$ 会自动缩放抵消），但**多元 + 量纲悬殊**时数值精度会受影响。建议养成 `StandardScaler` + LR 的习惯，详见 [`../../01-KNN/04b-scaling/`](../../01-KNN/04b-scaling/)。

#### 5. `coef_` 维度坑

单目标 y 是 1D → `coef_` shape `(n_features,)`；多目标 y 是 2D `(n_samples, n_targets)` → `coef_` shape `(n_targets, n_features)`。下游代码做 `coef_[0]` 之前先看 `.shape`。

### 跨章节链接

- 数学原理（损失函数 MSE）→ [`03b-math/`](../03b-math/)
- 解析解（SVD / 正规方程怎么解）→ [`04a-analytical/`](../04a-analytical/)
- 大数据场景的替代品 `SGDRegressor`（梯度下降，适合 n_samples > 10⁵ 或 online learning）→ [`04b-gd/`](../04b-gd/)
- 特征缩放配合 → [`../../01-KNN/04b-scaling/`](../../01-KNN/04b-scaling/)
- 与 KNN 的训练/预测代价对比 → [`01-intro/02-线性回归.md`](../01-intro/02-线性回归.md) 末尾对比表

> Sources：
> - PPT Slide 15-17
> - 笔记（README §02-api 线性回归 API 的应用）
> - sklearn 文档 `sklearn.linear_model.LinearRegression`
