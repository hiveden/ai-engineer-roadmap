# 第 2 章 · 线性回归 API

> 本文件 = 章 PPT 完整底稿（复习记忆页）
> 维度：**代码**
>
> **按知识点拆分的讲解版**：
>
> 1. [`01-LinearRegression.md`](./01-LinearRegression.md) — 【实操】sklearn API + 身高体重 demo

## 底稿

> 02 · 线性回归问题的求解

**学习目标**：

1. 知道线性回归 API 的使用

> 原 PPT 还预告了"损失函数 / 导数矩阵 / 正规方程 / 梯度下降"4 项——这些是后续章节内容，
> 已剥离到对应章节：损失/数学复习见 [`03b-math/`](../03b-math/)，
> 解析解见 [`04a-analytical/`](../04a-analytical/)，梯度下降见 [`04b-gd/`](../04b-gd/)。

> 【实操】线性回归 API 的应用

预测播仔身高。

已知数据（5 个样本，见下方代码 `x` / `y`）：

| 身高 (cm) | 体重 (kg) |
|---|---|
| 160 | 56.3 |
| 166 | 60.6 |
| 172 | 65.1 |
| 174 | 68.5 |
| 180 | 75.0 |

需求：播仔身高是 176，请预测体重？

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
