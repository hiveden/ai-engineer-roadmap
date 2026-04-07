# 04-逻辑回归：二分类问题的“定海神针”与工程博弈

> **资深架构视角：** 
> 别被它的名字骗了，逻辑回归 (Logistic Regression) 是不折不扣的分类算法。
> 在大模型横行的今天，它依然是工业界最稳健的分类器（如广告点击率预测、金融风控）。
> 本章我们将理解如何将“加权求和”转化为“概率判定”，以及在商业场景中如何权衡“错杀”与“漏掉”。

---

## 模块一：算法本质 —— 给线性回归穿上 Sigmoid 外套

### 1. 从“连续”到“概率”的映射
线性回归输出的是一个连续值（从 $-\infty$ 到 $+\infty$），但分类问题需要的是一个 **[0, 1] 之间的概率**。
我们引入 **Sigmoid 函数**：
$$ \text{Output} = \frac{1}{1 + e^{-(\mathbf{w \cdot x} + b)}} $$

*   **开发者视角**：Sigmoid 就是一个 **“挤压函数（Squashing Function）”**。
    *   如果线性预测值很大（如 100），输出接近 1（100% 概率）。
    *   如果线性预测值很小（如 -100），输出接近 0（0.1% 概率）。
    *   如果线性预测值为 0，输出刚好是 0.5。

### 2. 为什么不用线性回归做分类？
*   **非凸优化问题**：如果用线性回归的 MSE 损失函数处理分类，损失函数会变得“坑洼不平”（非凸），梯度下降会陷入局部最优解。
*   **Log-Loss (对数损失)** 👑：分类器追求的是“对正确的自信”。如果真实标签是 1，模型预测是 0.1，Log-Loss 会给予极大的惩罚。

---

## 模块二：工程评估指标 —— 告别“准确率”陷阱

在 Software 1.0 中，我们测试逻辑是否正确。在分类模型中，**准确率 (Accuracy) 是最危险的指标。**

> **场景**：假设你在写一个银行欺诈检测系统，1 万笔交易中只有 10 笔是欺诈（0.1%）。
> 如果你写一句 `return false;` (预测全为正常)，你的准确率高达 99.9%。但你的系统完全没用！

### 1. 混淆矩阵 (Confusion Matrix) —— 架构师的解题思路
| | 预测为 1 (Positive) | 预测为 0 (Negative) |
| :--- | :--- | :--- |
| **真实为 1** | **TP (真正例)** - 抓住了坏人 | **FN (伪反例)** - 漏掉了坏人 (代价极大!) |
| **真实为 0** | **FP (伪正例)** - 冤枉了坏人 (用户投诉) | **TN (真反例)** - 正常放行 |

### 2. 查准率 (Precision) vs. 召回率 (Recall) 👑
*   **查准率 (Precision)**：**“抓得准不准”**。
    *   公式：$TP / (TP + FP)$。
    *   场景：垃圾邮件过滤。宁可漏掉一点，也不要把老板的紧急邮件划为垃圾。
*   **召回率 (Recall)**：**“抓得全不全”**。
    *   公式：$TP / (TP + FN)$。
    *   场景：癌症预测。宁可让健康人复查，也绝不能放过任何一个癌症患者。

### 3. AUC (Area Under Curve) —— 排序能力的终极标尺
*   **解释**：AUC 衡量的是模型能不能把正例排在负例前面的能力。
*   **业务价值**：对于推荐系统，我们不在乎预测概率是 0.6 还是 0.8，只要“推荐列表里感兴趣的商品排在前面”就行。AUC 越接近 1，排序越准。

---

## 模块三：实战代码 —— 癌症分类与客户流失预测

在工业界，我们不仅要预测结果，还要拿到 **概率值 (`predict_proba`)**，这样后端可以根据业务灵活调整阈值。

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# 1. 加载乳腺癌数据集 (示例)
# 真实数据通常涉及缺失值处理: data.replace('?', np.nan).dropna()
data = pd.read_csv('breast-cancer-wisconsin.csv')
X = data.iloc[:, 1:-1] # 提取医学指标
y = data.Class        # 2代表良性, 4代表恶性

# 2. 标准化 (逻辑回归对量纲敏感!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. 实例化模型 (C为正则化力度, C越小正则化越强)
model = LogisticRegression(solver='liblinear', C=1.0)
model.fit(X_train, y_train)

# 4. 获取概率输出 (极其关键!)
# predict_proba 返回 [负例概率, 正例概率]
probs = model.predict_proba(X_test)[:, 1]

# 5. 深度评估
print(classification_report(y_test, model.predict(X_test)))
print(f"AUC Score: {roc_auc_score(y_test, probs):.4f}")
```

---

## 模块四：架构设计 —— 阈值博弈 (Thresholding)

默认情况下，`predict()` 使用 **0.5** 作为阈值。但作为架构师，你需要提供“调节杆”。

*   **保守策略 (High Precision)**：如果是一个涉及高额提现的风控模型，我们将阈值设为 **0.9**。只有 90% 确定是本人，才允许提现。
*   **激进策略 (High Recall)**：如果是双 11 的大促引流，我们将阈值设为 **0.1**。只要有一点点购买意向，就发优惠券。

---

## 架构师面试题

**Q：逻辑回归在处理类别特征 (Categorical Features) 时，为什么要先做 One-Hot 编码？**

**A：** 
1.  **数值误解**：如果把“颜色”编码为 红=1, 蓝=2, 绿=3。模型会错误地认为 `3 > 1`，从而推导出“绿色比红色更重要”，或者“红色+蓝色=绿色”。
2.  **空间拆解**：One-Hot 编码将一个特征拆成三个独立的二元特征（IsRed, IsBlue, IsGreen）。这样模型可以为每种颜色分配独立的权重（$w$），通过权重来表达不同颜色的业务贡献，这才是符合逻辑的线性叠加。
