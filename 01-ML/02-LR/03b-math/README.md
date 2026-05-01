# 第 3 章 b · 导数和矩阵复习

> 本文件 = 章 PPT 完整底稿（复习记忆页）
> 维度：**数学**
>
> **按知识点拆分的讲解版**：
>
> 1. [`01-数据表述.md`](./01-数据表述.md) — 【知道】常见数据表述
> 2. [`02-导数.md`](./02-导数.md) — 【知道】
> 3. [`03-偏导.md`](./03-偏导.md) — 【知道】
> 4. [`04-向量.md`](./04-向量.md) — 【知道】
> 5. [`05-矩阵.md`](./05-矩阵.md) — 【知道】

## 底稿

> 03 · 复习导数和矩阵

> 【知道】常见的数据表述

- 为什么要学习标量、向量、矩阵、张量？
  - 因机器学习、深度学习中经常用，不要因是数学就害怕
  - 宗旨：用到就学什么，不要盲目的展开、大篇幅学数学
- **标量** scalar：一个独立存在的数，只有大小没有方向
- **向量** vector：向量指一列顺序排列的元素。默认是列向量
- **矩阵** matrix：二维数组
- **张量** Tensor：多维数组，张量是基于向量和矩阵的推广

> 【知道】导数

当函数 y = f(x) 的自变量 x 在一点 $x_0$ 上产生一个增量 Δx 时，函数输出值的增量 Δy 与自变量增量 Δx 的比值在 Δx 趋于 0 时的极限 a 如果存在，a 即为在 $x_0$ 处的导数，记作 $f^\prime(x_0)$ 或 df($x_0$)/dx。

导数是函数的局部性质。一个函数在某一点的导数描述了这个函数在这一点附近的变化率。

函数在某一点的导数就是该函数所代表的曲线在这一点上的切线斜率。

**常见函数的导数**：

| 公式 | 例子 |
|---|---|
| $(C)^\prime = 0$ | $(5)^\prime = 0$，$(10)^\prime = 0$ |
| $(x^\alpha)^\prime = \alpha x^{\alpha-1}$ | $(x^3)^\prime = 3x^2$，$(x^5)^\prime = 5x^4$ |
| $(a^x)^\prime = a^x \ln a$ | $(2^x)^\prime = 2^x \ln 2$，$(7^x)^\prime = 7^x \ln 7$ |
| $(e^x)^\prime = e^x$ | $(e^x)^\prime = e^x$ |
| $(\log_a x)^\prime = \frac{1}{x \ln a}$ | $(\log_{10} x)^\prime = \frac{1}{x \ln 10}$ |
| $(\ln x)^\prime = \frac{1}{x}$ | — |
| $(\sin x)^\prime = \cos x$ | — |
| $(\cos x)^\prime = -\sin x$ | — |

**导数的四则运算**：

| 公式 | 例子 |
|---|---|
| $[u(x) \pm v(x)]^\prime = u^\prime(x) \pm v^\prime(x)$ | $(e^x + 4\ln x)^\prime = e^x + \frac{4}{x}$ |
| $[u(x) \cdot v(x)]^\prime = u^\prime(x) v(x) + u(x) v^\prime(x)$ | $(\sin x \cdot \ln x)^\prime = \cos x \ln x + \frac{\sin x}{x}$ |
| $\left[\frac{u(x)}{v(x)}\right]^\prime = \frac{u^\prime(x) v(x) - u(x) v^\prime(x)}{v^2(x)}$ | $\left(\frac{e^x}{\cos x}\right)^\prime = \frac{e^x \cos x + e^x \sin x}{\cos^2 x}$ |
| $\{g[h(x)]\}^\prime = g^\prime(h) \cdot h^\prime(x)$ | $(\sin 2x)^\prime = \cos 2x \cdot (2x)^\prime = 2\cos 2x$ |

**复合函数求导**：g(h) 是外函数，h(x) 是内函数。先对外函数求导，再对内函数求导。

**导数求极值**：导数为 0 的位置是函数的极值点。

> 【知道】偏导

⚠️ 待补充

> 【知道】向量

向量运算。

⚠️ 待补充

> 【知道】矩阵

矩阵运算。

⚠️ 待补充
