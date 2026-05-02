# 第 4 章 a · 解析解（正规方程法）

> 本文件 = 章 PPT 完整底稿（复习记忆页）
> 维度：**数学**
>
> **按知识点拆分的讲解版**：
>
> 1. [`01-一元解析解.md`](./01-一元解析解.md) — 【了解】
> 2. [`02-多元正规方程.md`](./02-多元正规方程.md) — 【了解】

## 底稿

> 04 · 一元 / 多元线性回归的解析解

> 【了解】一元线性回归的解析解

**目标**：找一组 (k, b) 使损失 $L(k, b) = \sum_{i=1}^{n}(y_i - k x_i - b)^2$ 最小。

**思路**：偏导 = 0 的位置就是极小值。分别对 k、b 求偏导：

$$\frac{\partial L}{\partial b} = -2 \sum (y_i - k x_i - b) = 0$$

$$\frac{\partial L}{\partial k} = -2 \sum x_i (y_i - k x_i - b) = 0$$

**联立求解**（设 $\bar{x}, \bar{y}$ 为均值）：

$$k = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}, \qquad b = \bar{y} - k \bar{x}$$

> 也叫**最小二乘法 OLS**（Ordinary Least Squares）的闭式解。**直接代公式即得最优，不需要迭代**。

身高体重例（5 个样本）：
| $x$ (身高) | $y$ (体重) |
|---|---|
| 160 | 56.3 |
| 166 | 60.6 |
| 172 | 65.1 |
| 174 | 68.5 |
| 180 | 75.0 |

代入公式：$\bar{x}=170.4$，$\bar{y}=65.1$，算出 $k \approx 0.93$，$b \approx -93.5$。
预测播仔 (x=176)：$\hat{y} = 0.93 \times 176 - 93.5 \approx 70.2$ kg。

> 【了解】多元线性回归的解析解 - 正规方程法

**矩阵化表示**：m 个样本、n 个特征。

- 特征矩阵 $X \in \mathbb{R}^{m \times (n+1)}$（多一列全 1 吸收偏置 b 进 $w_0$）
- 权重向量 $\mathbf{w} \in \mathbb{R}^{n+1}$
- 标签向量 $\mathbf{y} \in \mathbb{R}^{m}$

预测：$\hat{\mathbf{y}} = X \mathbf{w}$
损失：$L(\mathbf{w}) = (\mathbf{y} - X\mathbf{w})^T (\mathbf{y} - X\mathbf{w})$

**对 $\mathbf{w}$ 求导令零**：

$$\frac{\partial L}{\partial \mathbf{w}} = -2 X^T (\mathbf{y} - X\mathbf{w}) = 0$$

整理得 **正规方程**（Normal Equation）：

$$\boxed{\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}}$$

**优点**：
- 一步出解，**无需调学习率 / 不需要迭代**
- 数学优雅，是 LR 闭式解的"教科书答案"

**缺点**：
- 需要 $X^T X$ **可逆**——特征数过多 / 多重共线性时失败
- $(X^T X)^{-1}$ 计算复杂度 $O(n^3)$，特征数 ~10⁴ 以上就慢得离谱
- 大数据集 $X$ 内存放不下时不可用

→ 工程实践中**特征 < 10⁴ 用正规方程，否则用梯度下降**（见 04b）。
sklearn 的 `LinearRegression` 内部用 SVD 分解避免直接求逆，是正规方程的稳定版。
