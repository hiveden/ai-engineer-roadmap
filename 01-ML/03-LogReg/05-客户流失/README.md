# 第 5 章 · 电信客户流失预测

> 维度:**代码**
>
> **按知识点拆分的讲解版**：
>
> 1. [`01-数据集介绍.md`](./01-数据集介绍.md)
> 2. [`02-处理流程.md`](./02-处理流程.md)
> 3. [`03-案例实现.md`](./03-案例实现.md)

## 底稿

> 05 · 【实践】电信客户流失预测

**学习目标**：

1. 了解案例的背景信息
2. 知道案例的处理流程
3. 动手实现电信客户流失案例的代码

> 数据集介绍

电信运营商客户行为数据，**预测用户是否流失（churn）**。

- 规模：7043 条 × 21 列
- 目标列：`Churn`（Yes / No 二分类）
- 特征类型混合：
  - **人口属性**：`gender`、`SeniorCitizen`、`Partner`、`Dependents`
  - **业务属性**：`tenure`（在网月数）、`PhoneService`、`MultipleLines`、`InternetService`、`OnlineSecurity`、`OnlineBackup`、`DeviceProtection`、`TechSupport`、`StreamingTV`、`StreamingMovies`
  - **合约/账单**：`Contract`（月付/年付/两年）、`PaperlessBilling`、`PaymentMethod`、`MonthlyCharges`、`TotalCharges`
  - **ID**：`customerID`（建模时丢弃）

业务诉求：找出高流失风险用户，集中资源做留存动作（折扣 / 客服回访 / 套餐升级），而不是给全量用户撒券。

> 处理流程

1. **数据基本处理**
   - 查看 dtype / 缺失 / 标签分布（流失类通常类别不平衡）
   - 类别特征 one-hot 编码
   - `customerID` 丢弃，`TotalCharges` 字符串转数值
2. **特征筛选**
   - 看每个特征对 `Churn` 的边际分布（透视表 / 相关性）
   - 初筛影响大的特征构造 `X` / `y`
3. **模型训练**
   - LogisticRegression 拟合
   - 交叉验证 + GridSearch 调 `C` / `penalty`
4. **模型评估**
   - Accuracy / Precision / Recall（流失场景看 Recall）
   - ROC / AUC

> 案例实现

PPT 笔记此处代码块为空，下面是按处理流程整理的 sklearn 骨架（方便后续 lab 对照）：

```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# 1. 加载
df = pd.read_csv("churn.csv")

# 2. 基本处理
df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# 3. one-hot
X = pd.get_dummies(df.drop(columns=["Churn"]), drop_first=True)
y = df["Churn"]

# 4. 切分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. 标准化（连续特征 tenure / MonthlyCharges / TotalCharges 量纲差异大）
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 6. 模型 + 网格搜索
param_grid = {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"]}
grid = GridSearchCV(
    LogisticRegression(max_iter=1000, solver="lbfgs"),
    param_grid, cv=5, scoring="roc_auc",
)
grid.fit(X_train_s, y_train)

# 7. 评估
y_pred = grid.predict(X_test_s)
y_proba = grid.predict_proba(X_test_s)[:, 1]
print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))
```

要点：
- `stratify=y` 保留训练/测试集流失比例，避免标签倾斜
- `scoring="roc_auc"` 比 accuracy 更适合不平衡分类
- 业务侧关注 **Recall on churn=1**（漏掉一个高风险用户的代价 > 错挽留一个）
