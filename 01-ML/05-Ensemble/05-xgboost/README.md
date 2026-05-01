# 第 5 章 · XGBoost

> 维度：**算法 + 数学 + 代码**
>
> **按知识点拆分的讲解版**（待补）：
>
> 1. `01-算法思想.md` — 【知道】对 GBDT 的改进
> 2. `02-目标函数.md` — 【理解】推导
> 3. `03-API.md` — 【了解】XGBClassifier
> 4. `04-红酒预测.md` — 【实践】

## 底稿

> 05 · XGBoost（Extreme Gradient Boosting）

**学习目标**：

1. 知道 XGBoost 算法思想
2. 理解 XGBoost 目标函数
3. 了解 API
4. 实现红酒品质预测

**地位**：Kaggle 王牌，回归/分类问题最强基线之一。

> 【知道】XGBoost 算法思想

XGBoost 是对 GBDT 的三点改进：

1. **泰勒二阶展开**求解损失（GBDT 只用一阶）
2. **加正则项**控制树复杂度（防过拟合）
3. **自创分裂指标**（从损失函数推导，分裂时考虑复杂度增益）

> 【理解】XGBoost 目标函数

**整体目标**（前 $K$ 棵树）：

$$\mathrm{Obj} = \sum_{i=1}^{n} L(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$

第一项是样本损失，第二项是 $K$ 棵树的复杂度正则。

**树复杂度**：

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2$$

- $T$：叶子结点数（$\gamma$ 调节）
- $w$：叶子输出值向量（$\lambda$ 调节 L2）

**第 $t$ 轮目标**（前 $t-1$ 棵已固定）：

$$\mathrm{Obj}^{(t)} = \sum_{i=1}^{n} L(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t) + \mathrm{const}$$

**泰勒二阶展开**（对 $\hat{y}_i^{(t-1)}$ 处展开）：

$$L(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) \approx L(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)$$

其中：

$$g_i = \partial_{\hat{y}^{(t-1)}} L, \quad h_i = \partial^2_{\hat{y}^{(t-1)}} L$$

**去常数项**后的目标：

$$\mathrm{Obj}^{(t)} \approx \sum_{i=1}^{n} \left[g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)\right] + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

**视角变换**（按样本 → 按叶子）：设 $I_j = \{i \mid q(x_i) = j\}$ 为落到叶子 $j$ 的样本集合，$f_t(x_i) = w_{q(x_i)}$：

$$\mathrm{Obj}^{(t)} = \sum_{j=1}^{T} \left[\left(\sum_{i \in I_j} g_i\right) w_j + \frac{1}{2}\left(\sum_{i \in I_j} h_i + \lambda\right) w_j^2\right] + \gamma T$$

记 $G_j = \sum_{i \in I_j} g_i$，$H_j = \sum_{i \in I_j} h_i$：

$$\mathrm{Obj}^{(t)} = \sum_{j=1}^{T} \left[G_j w_j + \frac{1}{2}(H_j + \lambda) w_j^2\right] + \gamma T$$

**对 $w_j$ 求导得最优叶子值**：

$$w_j^* = -\frac{G_j}{H_j + \lambda}$$

**代回得目标最小值**（打分函数 / scoring function）：

$$\mathrm{Obj}^* = -\frac{1}{2} \sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma T$$

**分裂增益**（树构建时选最佳分裂点）：

$$\mathrm{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma$$

- $\mathrm{Gain} > 0$ 才分裂
- 停止条件：达到最大深度 / 叶子样本数过低 / 增益不足

> 【了解】XGBoost API

```python
from xgboost import XGBClassifier

bst = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.3,
    objective="binary:logistic",  # multi:softmax / reg:squarederror
    reg_lambda=1,    # L2
    gamma=0,         # 分裂阈值
)
bst.fit(X_train, y_train)
```

> 【实践】红酒品质预测

**数据**：11 个特征，3269 条数据，品质 1–6。

```python
# 流程骨架
# 1. 读 wine.csv → X, y
# 2. train_test_split
# 3. XGBClassifier(objective="multi:softmax").fit(X_train, y_train)
# 4. GridSearchCV 调 n_estimators + max_depth + learning_rate
```
