# 第 3 章 · AdaBoost

> 维度：**算法 + 数学**
>
> **按知识点拆分的讲解版**（待补）：
>
> 1. `01-简介.md` — 【知道】算法思想
> 2. `02-推导.md` — 【理解】公式推导
> 3. `03-构建过程.md` — 【理解】手算构建过程
> 4. `04-葡萄酒实战.md` — 【实践】

## 底稿

> 03 · AdaBoost

**学习目标**：

1. 理解 AdaBoost 算法思想
2. 知道 AdaBoost 构建过程
3. 实践案例

> 【知道】AdaBoost 算法简介

**Adaptive Boosting**：基于 Boosting 思想，通过**逐步提高被前一轮分类错误样本的权重**来训练强分类器。

**两个动态权重**：

| 对象 | 调整方式 |
|---|---|
| **样本权重** $D_t$ | 错分样本权重 ↑，正确样本权重 ↓ |
| **学习器权重** $\alpha_t$ | 误差率小 → 权重大；误差率大 → 权重小 |

**核心步骤**：

- **权值调整**：错分样本权重提高，让后续学习器更关注难样本
- **加权多数表决**：好学习器权重大，差的小

> 【理解】AdaBoost 算法推导

**强学习器**（分类输出 $> 0$ 为正类，$< 0$ 为负类）：

$$H(x) = \mathrm{sign}\left(\sum_{t=1}^{m} \alpha_t h_t(x)\right)$$

其中 $\alpha_t$ 是第 $t$ 个弱学习器的权重，$h_t(x) \in \{-1, +1\}$。

**模型权重公式**：

$$\alpha_t = \frac{1}{2} \ln\frac{1 - \varepsilon_t}{\varepsilon_t}$$

$\varepsilon_t$ 是第 $t$ 个弱学习器在当前样本权重下的加权错误率。$\varepsilon_t < 0.5$ 时 $\alpha_t > 0$，越准 $\alpha_t$ 越大。

**样本权重更新公式**：

$$D_{t+1}(i) = \frac{D_t(i)}{Z_t} \cdot \begin{cases} e^{-\alpha_t} & h_t(x_i) = y_i \\ e^{\alpha_t} & h_t(x_i) \neq y_i \end{cases}$$

- $Z_t$：归一化因子（保证 $\sum_i D_{t+1}(i) = 1$）
- 正确样本权重乘 $e^{-\alpha_t} < 1$（衰减），错误样本乘 $e^{\alpha_t} > 1$（放大）

> 【理解】AdaBoost 构建过程

**示例**：10 个样本一维特征，弱分类器为单一阈值决策桩（decision stump）。

**第 1 轮**：

1. 初始权重均匀：$D_1(i) = 0.1$
2. 遍历分裂点 $\{0.5, 1.5, \ldots, 8.5\}$，找加权错误率最小者：以 2.5 为分裂点，3 个错样本，$\varepsilon_1 = 0.3$
3. $\alpha_1 = \frac{1}{2}\ln\frac{0.7}{0.3} \approx 0.4236$
4. 更新权重：正确样本 $\times e^{-0.4236} \approx 0.6547$；错误样本 $\times e^{0.4236} \approx 1.5275$
5. 归一化 $Z_1 = 0.06547 \times 7 + 0.15275 \times 3 \approx 0.9165$
6. 最终：正确样本权重 $\approx 0.0714$，错误样本 $\approx 0.1667$

**第 2 轮**：在新权重下找最优分裂点 → 8.5，$\varepsilon_2 \approx 0.214$，$\alpha_2 \approx 0.6496$ → 再更新权重。

**第 3 轮**：$\varepsilon_3 \approx 0.182$，$\alpha_3 \approx 0.7514$。

**强学习器**：$H(x) = \mathrm{sign}(0.4236 h_1 + 0.6496 h_2 + 0.7514 h_3)$。

**注意**：第 $t+1$ 轮选分裂点时，错误率是**加权**的（$\sum D_t(i) \cdot \mathbb{1}[h(x_i) \neq y_i]$），不是简单计数。

> 【实践】AdaBoost 葡萄酒数据

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # 决策桩
    n_estimators=500,
    learning_rate=0.1,
)
ada.fit(X_train, y_train)
```

**结论**：AdaBoost 比单决策树过拟合更轻、测试集略胜，与 bagging 准确率相近。
