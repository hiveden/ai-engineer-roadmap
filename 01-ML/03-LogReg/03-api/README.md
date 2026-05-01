# 第 3 章 · 逻辑回归 API

> 维度：**代码**
>
> **按知识点拆分的讲解版**：
>
> 1. [`01-API介绍.md`](./01-API介绍.md) — 【知道】LogisticRegression
> 2. [`02-癌症分类案例.md`](./02-癌症分类案例.md) — 【实践】

## 底稿

> 03 · 逻辑回归 API

**学习目标**：

1. 知道逻辑回归的 API
2. 动手实现癌症分类案例

> 【知道】API 介绍

```python
sklearn.linear_model.LogisticRegression(solver='liblinear', penalty='l2', C=1.0)
```

**关键参数**：

- `solver`：损失函数优化方法
  - `liblinear`：小数据集更快（默认）
  - `sag` / `saga`：大数据集更快
  - `newton-cg` / `lbfgs` / `sag` / `saga`：仅支持 L2 / 无正则
  - `liblinear` / `saga`：支持 L1 正则
- `penalty`：正则化种类，`l1` 或 `l2`
- `C`：正则化力度（**越小正则越强**，与岭回归 `alpha` 相反）

**约定**：sklearn 默认把**类别数量少**的当作正例。

> 【实践】癌症分类案例

**数据介绍**：威斯康星乳腺癌（Breast Cancer Wisconsin）

- 699 条样本，11 列：`id` + 9 个肿瘤医学特征 + 1 个标签
- 包含 16 个缺失值，用 `?` 标出
- 标签：`2` = 良性，`4` = 恶性

**步骤**：

```
1. 获取数据
2. 基本数据处理
   2.1 缺失值处理
   2.2 确定特征值、目标值
   2.3 分割数据
3. 特征工程（标准化）
4. 机器学习（逻辑回归）
5. 模型评估
```

**代码骨架**：

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. 获取数据
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
                'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
                'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
data = pd.read_csv('breast-cancer-wisconsin.data', names=column_names)

# 2.1 缺失值处理：'?' → NaN → dropna
data = data.replace(to_replace='?', value=np.nan)
data = data.dropna()

# 2.2 特征值 / 目标值
x = data.iloc[:, 1:-1]   # 去掉 id 和 label
y = data['Class']

# 2.3 分割数据
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)

# 3. 标准化（量纲不一致 → LR 对尺度敏感）
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4. 逻辑回归
estimator = LogisticRegression()
estimator.fit(x_train, y_train)

# 5. 模型评估
y_pred = estimator.predict(x_test)
print('预测值：', y_pred)
print('准确率：', estimator.score(x_test, y_test))
print('w:', estimator.coef_)
print('b:', estimator.intercept_)
```

**坑点提醒**：

- `?` 是字符串，`replace` 后整列 `dtype` 仍可能是 `object`，必要时 `astype(float)`
- 类别不均衡场景下，光看 `score`（准确率）会被误导 → 下一章学精确率 / 召回率 / F1
