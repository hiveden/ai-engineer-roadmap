# 第 2 章 · 线性回归 API

> 本文件 = 章 PPT 完整底稿（复习记忆页）
> 维度：**代码**
>
> **按知识点拆分的讲解版**：
>
> 1. [`01-LinearRegression.md`](./01-LinearRegression.md) — 【实操】sklearn API + 身高体重 demo

## ━━━━━━━━ 底稿 ━━━━━━━━

### PPT 原文（slide 15-17）━━━━━━━━

#### Slide 15 · 一元线性回归案例

> 预测小明身高。需求：小明身高是 176，请预测体重？

| 编号 | 身高 | 体重 |
|---|---|---|
| 1 | 160 | 56.3 |
| 2 | 166 | 60.6 |
| 3 | 172 | 65.1 |
| 4 | 174 | 68.5 |
| 5 | 180 | 75 |
| 6 | 176 | ？ |

> 对于这个回归案例如何利用 API 快速求解呢？

#### Slide 16 · 线性回归 API 介绍（5 步流程图）

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

#### Slide 17 · 线性回归 API 介绍（完整代码）

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

---

### 笔记（已整理）━━━━━━━━

> 02 · 线性回归问题的求解

**学习目标**：

1. 知道线性回归 API 的使用

> 原 PPT 还预告了"损失函数 / 导数矩阵 / 正规方程 / 梯度下降"4 项——这些是后续章节内容，
> 已剥离到对应章节：损失/数学复习见 [`03b-math/`](../03b-math/)，
> 解析解见 [`04a-analytical/`](../04a-analytical/)，梯度下降见 [`04b-gd/`](../04b-gd/)。

> 【实操】线性回归 API 的应用

预测小明身高。

已知数据（5 个样本，见下方代码 `x` / `y`）：

| 身高 (cm) | 体重 (kg) |
|---|---|
| 160 | 56.3 |
| 166 | 60.6 |
| 172 | 65.1 |
| 174 | 68.5 |
| 180 | 75.0 |

需求：小明身高是 176，请预测体重？

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
