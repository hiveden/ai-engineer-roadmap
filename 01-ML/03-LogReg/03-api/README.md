# 第 3 章 · 逻辑回归 API

> 维度：**代码**
>
> **按知识点拆分的讲解版**：
>
> 1. [`01-API介绍.md`](./01-API介绍.md) — 【知道】LogisticRegression
> 2. [`02-癌症分类案例.md`](./02-癌症分类案例.md) — 【实践】

## ━━━━━━━━ 底稿 ━━━━━━━━

### PPT

> 从 [`../04逻辑回归.pptx`](../04逻辑回归.pptx) 提取（slide 23-29）。图占位标 〔图〕；排版整理；文字保留 PPT 原话。

> Slide 23 · 目录（章间导航）

- 逻辑回归简介（应用场景，数学知识）
- 逻辑回归原理
- 逻辑回归API函数和案例
- 分类问题评估（混淆矩阵、精确率、召回率、F1-score、AUC指标、ROC曲线）
- 电信客户流失预测案例

> Slide 24 · 学习目标

- 知道逻辑回归的API
- 动手实现癌症分类案例

> Slide 25 · 逻辑回归API函数和案例 – API介绍

```
sklearn.linear_model.LogisticRegression(solver='liblinear', penalty='l2', C = 1.0)
```

solver（计算引擎）:
1. liblinear 对小数据集场景训练速度更快，sag 和 saga 对大数据集更快一些。
2. 正则化：
   1. sag、saga 支持 L2 正则化或者没有正则化
   2. liblinear 和 saga 支持 L1 正则化

penalty：正则化的种类，l1 或者 l2

C：正则化力度

默认将类别数量少的当做正例

〔图：API 参数界面截图〕

> **Notes**（讲师 —— 此处备注错位，属于 ROC/AUC 内容）：讲清楚 1 横纵坐标  2 曲线含义（不同阈值下，类比头发长度），宽松的含义  3（0,1）坐标的含义  4 对角线 随机猜测  5 面积的含义  6 如果比随机猜测还糟糕，则反过来理解。
> 比如以头发长短判断性别，阈值变化；AUC=0.5 是随机猜测；AUC=1是完美分类器，（0,1）点代表纵坐标分母 FN=0，横坐标分子 FP=0。

> Slide 26 · 逻辑回归API函数和案例 – 案例癌症分类预测（数据描述）

逻辑回归API函数和案例 – 案例癌症分类预测

数据描述

（1）699条样本，共11列数据，
      第一列用语检索的id，
      后9列分别是与肿瘤相关的医学特征，
      最后一列表示肿瘤类型的数值。

（2）包含16个缺失值，用"?"标出。

（3）2表示良性，4表示恶性

> ⚠ PPT 原文"用语检索" 疑似 typo，应为"用于检索"，保留原文。

> Slide 27 · 逻辑回归API函数和案例 – 癌症分类预测（案例分析步骤）

逻辑回归API函数和案例 – 癌症分类预测

案例分析

1. 获取数据
2. 基本数据处理
   2.1 缺失值处理
   2.2 确定特征值,目标值
   2.3 分割数据
3. 特征工程(标准化)
4. 机器学习(逻辑回归)
5. 模型评估

> Slide 28 · 逻辑回归API函数和案例 – 癌症分类预测（代码实现）

逻辑回归API函数和案例 – 癌症分类预测

代码实现（PPT 原文单行连写）：

```python
def dm_LogisticRegression():
    # 1 获取数据
    data = pd.read_csv('./data/breast-cancer-wisconsin.csv')
    data.info()
    # 2 基本数据处理
    # 2.1 缺失值处理
    data = data.replace(to_replace="?", value=np.NaN)
    data = data.dropna()
    # 2.2 确定特征值,目标值
    x = data.iloc[:, 1:-1]
    print('x.head()-->\n', x.head())
    y = data["Class"]
    print('y.head()-->\n', y.head())
    # 2.3 分割数据
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)
    # 3 特征工程(标准化)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 4 机器学习(逻辑回归)
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)
    # 5 模型评估
    y_predict = estimator.predict(x_test)
    print('y_predict-->', y_predict)
    accuracy = estimator.score(x_test, y_test)
    print('accuracy-->', accuracy)
```

> Slide 29 · 习题 第 3 章

1、下列关于逻辑回归API的使用正确的是？（多选）

A）需要在sklearn的线性模型linear_model中导出使用
B）可以通过solver参数指定损失的优化方法
C）可以通过penalty参数指定使用哪种正则化方式
D）它默认将样本中类别数较多的一类当做正例

答案解析：它默认将样本中类别数较少的一类当做正例

答案：ABC

### 笔记

> 【知道】API 介绍

```python
sklearn.linear_model.LogisticRegression(solver='liblinear', penalty='l2', C=1.0)
```

> 旁注：上述 `solver='liblinear'` 是旧版 sklearn 默认；sklearn ≥1.0 实际默认为 `'lbfgs'`。

**关键参数**：

- `solver`：损失函数优化方法
  - `liblinear`：小数据集更快（默认）
  - `sag` / `saga`：大数据集更快
  - `newton-cg` / `lbfgs` / `sag` / `saga`：仅支持 L2 / 无正则
  - `liblinear` / `saga`：支持 L1 正则
- `penalty`：正则化种类，`l1` 或 `l2`
- `C`：正则化力度（**越小正则越强**，与岭回归 `alpha` 相反）

**约定**：sklearn 默认把**类别数量少**的当作正例。

> PPT 原话（API 介绍，slide 25）：
>
> sklearn.linear_model.LogisticRegression(solver='liblinear', penalty='l2', C = 1.0)
> solver （计算引擎）:
>           1 liblinear 对小数据集场景训练速度更快，sag 和 saga 对大数据集更快一些。
>           2 正则化：
> 	1 sag、saga 支持 L2 正则化或者没有正则化
> 	2 liblinear 和 saga 支持 L1 正则化
> penalty：正则化的种类，l1 或者 l2
> C：正则化力度
> 默认将类别数量少的当做正例
>
> 〔图：图片3.jpg —— 图形用户界面, 文本, 应用程序〕

> 【实践】癌症分类案例

**数据介绍**：威斯康星乳腺癌（Breast Cancer Wisconsin）

- 699 条样本，11 列：`id` + 9 个肿瘤医学特征 + 1 个标签
- 包含 16 个缺失值，用 `?` 标出
- 标签：`2` = 良性，`4` = 恶性

> PPT 原话（数据描述，slide 26）：
>
> （1）699条样本，共11列数据，
>       第一列用语检索的id，
>       后9列分别是与肿瘤相关的医学特征，
>       最后一列表示肿瘤类型的数值。
>
> （2）包含16个缺失值，用"?"标出。
>
> （3）2表示良性，4表示恶性
> 数据描述
>
> 〔图：图片5.jpg —— 数据集表格预览〕
>
> 备注："用语检索" 疑似 typo，应为"用于检索"，保留原文。

> PPT 原话（案例分析步骤，slide 27）：
>
> 案例分析
> 1.获取数据
> 2.基本数据处理
>     2.1 缺失值处理
>     2.2 确定特征值,目标值
> 2.3 分割数据
> 3.特征工程(标准化)
> 4.机器学习(逻辑回归)
> 5.模型评估

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

> PPT 原话（代码实现，slide 28，单行连写 + 全角引号原样保留）：
>
> ```python
> def dm_LogisticRegression():    # 1 获取数据    data = pd.read_csv('./data/breast-cancer-wisconsin.csv')    data.info()    # 2 基本数据处理    # 2.1 缺失值处理    data = data.replace(to_replace="?", value=np.NaN)    data = data.dropna()    # 2.2 确定特征值,目标值    x = data.iloc[:, 1:-1]    print('x.head()-->\n', x.head())    y = data["Class"]    print('y.head()-->\n', y.head())    # 2.3 分割数据    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)    # 3 特征工程(标准化)    transfer = StandardScaler()    x_train = transfer.fit_transform(x_train)    x_test = transfer.transform(x_test)    # 4 机器学习(逻辑回归)    estimator = LogisticRegression()    estimator.fit(x_train, y_train)    # 5 模型评估    y_predict = estimator.predict(x_test)    print('y_predict-->', y_predict)    accuracy = estimator.score(x_test, y_test)    print('accuracy-->', accuracy)
> ```
>
> 旁注：PPT 原文为单行连写 / 全角引号，按 sklearn 标准排版补全（见上方"代码骨架"）

> 【习题】第 3 章（PPT slide 29）

1、下列关于逻辑回归API的使用正确的是？（多选）

〔图：图片2.jpg —— 图形用户界面, 文本, 应用程序〕

A）需要在sklearn的线性模型linear_model中导出使用

B）可以通过solver参数指定损失的优化方法

C）可以通过penalty参数指定使用哪种正则化方式

D）它默认将样本中类别数较多的一类当做正例
答案解析：它默认将样本中类别数较少的一类当做正例
答案：ABC
