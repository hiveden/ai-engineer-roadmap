# 第 5 章 · 泰坦尼克号案例

> 维度：**代码**
>
> **按知识点拆分的讲解版**（待补）：
>
> 1. `01-案例背景.md` — 数据与目标
> 2. `02-API.md` — `DecisionTreeClassifier` 关键参数
> 3. `03-实现.md` — 完整 pipeline 骨架

## 底稿

> 05 · 泰坦尼克号生存预测

**学习目标**：

1. 能用 sklearn 决策树解决二分类问题
2. 掌握关键超参数的工程含义

> 【实践】案例背景

1912 年泰坦尼克号沉没，2224 人中 1502 人遇难。任务：根据**票类、班次、姓名、年龄、上船港口、房间、性别**等特征，预测乘客是否生还。

**业务侧观察**：妇女、儿童、社会地位高者存活率显著更高 → 决策树天然适合表达这类"分群规则"。

**特征工程要点**：

- 类别特征（Sex、Embarked）用 `pd.get_dummies` 或 `OrdinalEncoder`
- 缺失值（Age、Cabin）填充或丢弃
- 决策树**不需要**标准化

> 【知道】API 介绍

```python
sklearn.tree.DecisionTreeClassifier(
    criterion='gini',       # 'gini' (CART) 或 'entropy' (ID3 风格)
    max_depth=None,         # 树最大深度，控过拟合
    min_samples_split=2,    # 节点继续分裂的最小样本数
    min_samples_leaf=1,     # 叶子节点最少样本数
    random_state=None,
)
```

**关键参数工程含义**：

| 参数 | 作用 | 调参直觉 |
|---|---|---|
| `criterion` | 分裂准则 | gini 默认（更快）；entropy 在某些数据上略好 |
| `max_depth` | 限制深度 | 数据量大 / 特征多时设 10-100；过深易过拟合 |
| `min_samples_split` | 内部节点最小分裂样本 | 大数据集（10w+）可设 10 |
| `min_samples_leaf` | 叶子最少样本 | 大数据集设 5+，避免叶子过碎 |
| `random_state` | 随机种子 | 复现实验 |

> 【实践】案例实现

**Pipeline 骨架**：

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. 加载 + 清洗
df = pd.read_csv('titanic.csv')
df = df[['Pclass', 'Sex', 'Age', 'Survived']].dropna()
X = pd.get_dummies(df[['Pclass', 'Sex', 'Age']])
y = df['Survived']

# 2. 切分
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 训练
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_tr, y_tr)

# 4. 评估
print(accuracy_score(y_te, clf.predict(X_te)))
```

**决策树的产品化优势**：

- **可解释性**：可视化（`sklearn.tree.plot_tree`）后能给业务方看每条规则
- **无需标准化 / 量化**：开箱即用
- **混合类型友好**：数值 + 类别都能处理
- **快推理**：从根到叶 $O(\log n)$
