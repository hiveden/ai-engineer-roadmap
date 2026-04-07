# 03-线性回归：万物皆可“加权求和”的工程哲学

> **资深架构视角：** 
> 别把线性回归看成简单的“画线”。在神经网络时代，一个感知机（Perceptron）本质上就是一个线性回归。
> 它是所有复杂预测（如房价预估、点击率预测、股票走势）的基础原子。
> 本章我们将通过“解析解”与“迭代解”的权衡，理解模型是如何通过“自反馈”实现优化的。

---

## 模块一：核心隐喻 —— 模型就是一个“加权累加器”

### 1. 业务逻辑转换
在传统 Java 开发中，你可能会写：
```java
double predictPrice(House h) {
    if (h.area > 100 && h.isSchoolDistrict) return 500.0; // Hardcoded!
    return 300.0;
}
```
在 ML 时代，我们通过线性回归将其转化为 **加权公式**：
$$ \text{Price} = w_1 \cdot \text{Area} + w_2 \cdot \text{RoomCount} + w_3 \cdot \text{SchoolScore} + b $$
*   **权重 ($w$)**：代表每个特征对结果的“话语权”。
*   **偏置 ($b$)**：代表“起步价”或者“底噪”。
*   **学习的目标**：就是为了找出这组最优的 $w$ 和 $b$。

---

## 模块二：求解策略 —— 离线批量 vs. 在线流式

如何求出最优的 $w$？工程上有两种完全不同的“降维打击”策略：

### 1. 正规方程 (Normal Equation) —— “离线全量计算”
*   **数学本质**：通过矩阵运算 $w = (X^T X)^{-1} X^T y$ 一步到位算出精确解。
*   **工程视角**：相当于写一个巨大的 SQL 全量 join 并求逆矩阵。
*   **优点**：不需要调参，精度最高。
*   **缺点**：计算复杂度是 $O(N^3)$。当你的特征维度（Column 数）超过 1 万时，内存会爆掉，计算会卡死。
*   **适用场景**：特征少、数据量适中（< 10万条）的离线分析任务。

### 2. 梯度下降 (Gradient Descent) 👑 —— “在线迭代优化”
*   **工程比喻**：**“盲人下山”**。每走一步，就环顾四周找坡度最陡的方向踩一小步（学习率 $\alpha$），直到走到山底（误差最小）。
*   **核心组件**：
    *   **损失函数 (Loss Function)**：衡量“当前预测值”和“真实值”差距的标尺（通常是 MSE - 均方误差）。
    *   **梯度 (Gradient)**：损失函数在当前点的“变化率”。梯度为 0 时，我们就达到了最优解（山底）。
*   **主流变体**：**SGD (随机梯度下降)**。它不需要每次遍历全量数据，而是每次只看一个 Batch（如 32 条记录）就更新一次参数。
*   **架构价值**：它是 **大规模训练的唯一解**。无论数据有几亿条，只要不断迭代，模型总会变强。

---

## 模块三：模型评估指标 —— 别被平均值骗了

在代码上线后，我们需要一个监控指标。

1.  **MAE (平均绝对误差)**：
    *   **解释**：预测值与真实值差的绝对值平均数。
    *   **业务价值**：最符合人类直觉。比如预测误差是 5 万，老板一听就懂。
2.  **MSE (均方误差) / RMSE** 👑：
    *   **解释**：误差先平方再平均。
    *   **业务价值**：**“严惩离群点”**。如果有一个预测错了 10 倍，MSE 会因为平方效应而爆炸。
    *   **工程建议**：在训练模型时，我们用 MSE（因为它对错误零容忍）；在向业务方汇报时，我们用 MAE（因为它更好懂）。

---

## 模块四：实战案例 —— 生产级波士顿房价预测 (1.2+ 版本兼容)

由于 `load_boston` 已从最新库中移除，资深开发者应习惯于直接处理原始数据源。

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# 1. 生产环境数据加载 (以 CSV 为例)
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]

# 2. 架构规范：训练/测试集切分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 标准化 (特征对齐，加速梯度下降)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. 训练：选择 SGD (梯度下降) 而非正规方程
# 参数: constant 代表固定步长, eta0 是初始步长
model = SGDRegressor(learning_rate='constant', eta0=0.01)
model.fit(X_train, y_train)

# 5. 推理与评估
y_pred = model.predict(X_test)
error = mean_squared_error(y_test, y_pred)
print(f"均方误差 (MSE): {error:.4f}")
```

---

## 模块五：正则化 (Regularization) —— 解决过拟合的“紧箍咒”

为什么模型在测试环境很猛，上线就拉胯？因为 **过拟合 (Overfitting)**。模型把噪声也学进去了。

**正则化本质**：在损失函数后面加一个小尾巴，惩罚那些太大的权重 $w$。

### 1. L1 正则化 (Lasso) —— “暴力裁员”
*   **效果**：会产生**稀疏矩阵**（很多不重要的特征权重直接变 0）。
*   **工程价值**：**特征筛选**。如果你的特征有 1000 个，Lasso 能帮你找出最有用的 10 个，其他的直接“裁掉”，减少计算压力。

### 2. L2 正则化 (Ridge) —— “集体降薪” 👑
*   **效果**：不把权重变 0，但会让它们都变得非常小（接近 0）。
*   **工程价值**：**防止模型变得“偏执”**。它让模型变得平滑，不会因为某个特征的微小波动而剧烈震荡。它是工业界的默认标准（Baseline）。

---

## 架构师面试题

**Q：为什么梯度下降算法一定要先做特征标准化 (Standardization)？**

**A：** 
1.  **梯度下降的“碗”状理论**：如果特征 A 范围是 1~10，特征 B 范围是 1~1,000,000。损失函数的等高线会变成一个“极其扁平的长椭圆”。
2.  **震荡问题**：在这种情况下，梯度会不停地在窄轴方向来回震荡，很难走到中心点（山底），导致模型**收敛极其缓慢甚至无法收敛**。
3.  **对齐价值**：通过标准化，我们将“长椭圆”变回“圆碗”，梯度可以直接顺着坡度最陡的方向俯冲到山底，大大提升训练效率。
