# 第 5 章 · 电信客户流失预测

> 维度：**代码**
>
> **按知识点拆分的讲解版**：
>
> 1. [`01-数据集介绍.md`](./01-数据集介绍.md)
> 2. [`02-处理流程.md`](./02-处理流程.md)
> 3. [`03-案例实现.md`](./03-案例实现.md)

## ━━━━━━━━ 底稿 ━━━━━━━━

### PPT

> 从 [`../04逻辑回归.pptx`](../04逻辑回归.pptx) 提取（slide 60-70）。图占位标 〔图〕；排版整理；文字保留 PPT 原话。

> Slide 60 · 目录（章间导航）

- 逻辑回归简介（应用场景，数学知识）
- 逻辑回归原理
- 逻辑回归API函数和案例
- 分类问题评估（混淆矩阵、精确率、召回率、F1-score、AUC指标、ROC曲线）
- 电信客户流失预测案例

> Slide 61 · 学习目标

- 了解案例的背景信息
- 知道案例的处理流程
- 动手实现电信客户流失案例的代码

> Slide 62 · 案例 – 电信客户流失预测（案例需求 + 数据集介绍）

案例 – 电信客户流失预测

案例需求

已知：用户个人，通话，上网等信息数据
需求：通过分析特征属性确定用户流失的原因，以及哪些因素可能导致用户流失。建立预测模型来判断用户是否流失，并提出用户挽回策略。

数据集介绍

〔图：数据集列名截图〕
〔图：Kaggle 数据集徽标〕

标签

> **Notes**（讲师）：如果预测流失，营销团队会采用大折扣挽回；如果预测不会流失，则可能爱里不搭。

> Slide 63 · 案例 – 电信客户流失预测（数据集介绍续）

案例 – 电信客户流失预测

数据集介绍

〔图：数据列类型一览〕
〔图：各列具体内容预览〕

标签

标签  object str (非二分类标签)  bool

挑选最有用的特征(支付方式)

> **Notes**（讲师）：7043行   16列   没有空值数据；Gender gender_male gender_female；churn churn_yes churn_no；然后舍弃

> Slide 64 · 案例 – 电信客户流失预测（案例步骤分析）

案例 – 电信客户流失预测

案例步骤分析

1、数据基本处理 / 清洗
主要是查看数据行/列数量
对类别数据数据进行one-hot处理
查看标签分布情况

2、特征筛选
分析哪些特征对标签值影响大
对标签进行分组统计，对比0/1标签分组后的均值  等等
初步筛选出对标签影响比较大的特征，形成x、y

3、模型训练
样本均衡情况下模型训练
样本不平衡情况下模型训练
交叉验证网格搜素等方式模型训练

4、模型评估
精确率
Roc_AUC指标计算

> ⚠ PPT 原文"网格搜素" 疑似 typo（"搜索"），保留原文。

> Slide 65 · 案例 – 电信客户流失预测 – 1 数据基本处理（代码）

案例 – 电信客户流失预测 – 1数据基本处理（PPT 原文单行连写）：

```python
import numpy as np
import pandas as pd

def dm01_数据基本处理():
    churn_pd = pd.read_csv('./data/churn.csv')
    churn_pd.info()
    print('churn_pd.describe()-->\n', churn_pd.describe())
    print('churn_pd-->\n', churn_pd)
    # 1 处理类别型的数据 类别型数据做one-hot编码
    churn_pd = pd.get_dummies(churn_pd)
    print('churn_pd-->\n', churn_pd)
    churn_pd.info()
    # 2 去除列 Churn_no gender_Male # inplace=True 在原来的数据上进行删除
    churn_pd.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)
    churn_pd.info()
    # 3 列标签重命名 打印列名
    print('churn_pd.columns', churn_pd.columns)
    churn_pd.rename(columns = {'Churn_Yes':'flag'}, inplace=True)
    print('churn_pd.columns', churn_pd.columns)
    # 4 查看标签的分布情况 0.26用户流失
    value_counts = churn_pd.flag.value_counts(1)
    print('value_counts-->\n', value_counts)
    print('从标签的分类中可以看出: 属于标签分类不平衡样本')
```

> Slide 66 · 案例 – 电信客户流失预测 – 特征筛选（代码）

案例 – 电信客户流失预测 – 特征筛选（PPT 原文单行连写）：

```python
import matplotlib.pyplot as plt
import seaborn as sns

def dm02_特征筛选():
    churn_pd = pd.read_csv('./data/churn.csv')
    # 1 处理类别型的数据 类别型数据做one-hot编码
    churn_pd = pd.get_dummies(churn_pd)
    # 2 去除列 Churn_no gender_Male # nplace=True 在原来的数据上进行删除
    churn_pd.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)
    # 3 列标签重命名 打印列名
    churn_pd.rename(columns={'Churn_Yes': 'flag'}, inplace=True)
    # 4 查看标签的分布情况 0.26用户流失
    value_counts = churn_pd.flag.value_counts(1)
    # 5 查看Contract_Month 是否月签约流失情况
    sns.countplot(data=churn_pd, y = "Contract_Month", hue='flag')
    plt.show()
```

> ⚠ PPT 注释 "nplace=True" 疑似 typo（应为 inplace），按 PPT 原样保留。

> Slide 67 · 案例 – 电信客户流失预测 – 3 模型训练与评测（代码）

案例 – 电信客户流失预测 – 3 模型训练与评测（PPT 原文单行连写，末尾 import 错位）：

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

def dm03_模型训练和评测():
    # 1 数据基本处理
    churn_pd = pd.read_csv('./data/churn.csv')
    # 1-1 处理类别型的数据 类别型数据做one-hot编码
    churn_pd = pd.get_dummies(churn_pd)
    # 1-2 去除列 Churn_no gender_Male # nplace=True 在原来的数据上进行删除
    churn_pd.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)
    # 1-3 列标签重命名 打印列名
    churn_pd.rename(columns={'Churn_Yes': 'flag'}, inplace=True)
    # 2 特征处理
    # 2-1 确定目标值和特征值
    x = churn_pd[['Contract_Month', 'internet_other', 'PaymentElectronic']]
    y = churn_pd['flag']
    # 2-2 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
 random_state=100)
    # 3 实例化模型 训练模型 模型预测
    estimator =  LogisticRegression()
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_test)
    # 4 模型评估
    my_accuracy_score = accuracy_score(y_test, y_pred)
    print('my_accuracy_score-->', my_accuracy_score)
    my_score = estimator.score(x_test, y_test)
    print('my_score-->', my_score)
    # 计算AUC值
    my_roc_auc_score = roc_auc_score(y_test, y_pred)
    print('my_roc_auc_score-->', my_roc_auc_score)

from sklearn.metrics import classification_report
    result = classification_report(y_test, y_pred, target_names=['flag0', 'flag1'])
    print('classification_report result->\n', result)
```

> ⚠ PPT 原文末尾 `from sklearn.metrics import classification_report` 在函数体之外（缩进异常），且 result/print 又回到缩进，PPT 原写法即如此（疑似复制粘贴错位），按 PPT 原样保留。

> Slide 68 · 分析报告

classification_report 在用户流失分析的案例

分析报告

〔图：classification_report 输出表格（当非流失客户作为正例时）〕
〔图：classification_report 输出表格（当流失客户作为正例时）〕

当非流失客户作为正例时
当流失客户作为正例时

> Slide 69 · （纯图示页）

〔图：分类对比示意图〕

> Slide 70 · （纯图示页）

〔图：分析结果汇总表〕

> 备注：Slide 69-70 为纯图示页，无可提取文字内容，按图占位处理。

### 笔记

> 【了解】数据集介绍

电信运营商客户行为数据，**预测用户是否流失（churn）**。

- 规模：7043 条 × 21 列
- 目标列：`Churn`（Yes / No 二分类）
- 特征类型混合：
  - **人口属性**：`gender`、`SeniorCitizen`、`Partner`、`Dependents`
  - **业务属性**：`tenure`（在网月数）、`PhoneService`、`MultipleLines`、`InternetService`、`OnlineSecurity`、`OnlineBackup`、`DeviceProtection`、`TechSupport`、`StreamingTV`、`StreamingMovies`
  - **合约/账单**：`Contract`（月付/年付/两年）、`PaperlessBilling`、`PaymentMethod`、`MonthlyCharges`、`TotalCharges`
  - **ID**：`customerID`（建模时丢弃）

业务诉求：找出高流失风险用户，集中资源做留存动作（折扣 / 客服回访 / 套餐升级），而不是给全量用户撒券。

> PPT 原话（案例需求 + 数据集介绍，slide 62）：
>
> 案例需求
> 已知：用户个人，通话，上网等信息数据
> 需求：通过分析特征属性确定用户流失的原因，以及哪些因素可能导致用户流失。建立预测模型来判断用户是否流失，并提出用户挽回策略。
>
> 〔图：图片7.jpg〕
> 数据集介绍
>
> 〔图：图片8.jpg —— 图片包含 徽标〕
> 标签
>
> 讲师备注：如果预测流失，营销团队会采用大折扣挽回；如果预测不会流失，则可能爱里不搭。

> PPT 原话（数据集介绍续，slide 63）：
>
> 数据集介绍
>
> 〔图：图片6.jpg〕
> 〔图：图片7.jpg〕
> 标签
>
> 〔图：图片5.jpg〕
> 标签  object str (非二分类标签)  bool
> 挑选最有用的特征(支付方式)
>
> 讲师备注：7043行   16列   没有空值数据；Gender  gender_male   gender_female；churn  churn_yes   churn_no；然后舍弃

> 【知道】处理流程

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

> PPT 原话（案例步骤分析，slide 64）：
>
> 案例步骤分析
> 1、数据基本处理 / 清洗
> 主要是查看数据行/列数量
> 对类别数据数据进行one-hot处理
> 查看标签分布情况
> 2、特征筛选
> 分析哪些特征对标签值影响大
> 对标签进行分组统计，对比0/1标签分组后的均值  等等
> 初步筛选出对标签影响比较大的特征，形成x、y
> 3、模型训练
> 样本均衡情况下模型训练
> 样本不平衡情况下模型训练
> 交叉验证网格搜素等方式模型训练
> 4、模型评估
> 精确率
> Roc_AUC指标计算
>
> 备注："网格搜素" 疑似 typo（"搜索"），保留原文。

> 【实践】案例实现

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

> 【代码】1 数据基本处理（PPT slide 65）

```python
import numpy as np
import pandas as pd

def dm01_数据基本处理():
    churn_pd = pd.read_csv('./data/churn.csv')
    churn_pd.info()
    print('churn_pd.describe()-->\n', churn_pd.describe())
    print('churn_pd-->\n', churn_pd)
    # 1 处理类别型的数据 类别型数据做one-hot编码
    churn_pd = pd.get_dummies(churn_pd)
    print('churn_pd-->\n', churn_pd)
    churn_pd.info()
    # 2 去除列 Churn_no gender_Male # inplace=True 在原来的数据上进行删除
    churn_pd.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)
    churn_pd.info()
    # 3 列标签重命名 打印列名
    print('churn_pd.columns', churn_pd.columns)
    churn_pd.rename(columns = {'Churn_Yes':'flag'}, inplace=True)
    print('churn_pd.columns', churn_pd.columns)
    # 4 查看标签的分布情况 0.26用户流失
    value_counts = churn_pd.flag.value_counts(1)
    print('value_counts-->\n', value_counts)
    print('从标签的分类中可以看出: 属于标签分类不平衡样本')
```

> 旁注：PPT 原文为单行连写，按 sklearn 标准排版补全。
>
> PPT 原文（单行连写）：
>
> ```python
> import numpy as npimport pandas as pddef dm01_数据基本处理():    churn_pd = pd.read_csv('./data/churn.csv')    churn_pd.info()    print('churn_pd.describe()-->\n', churn_pd.describe())    print('churn_pd-->\n', churn_pd)    # 1 处理类别型的数据 类别型数据做one-hot编码    churn_pd = pd.get_dummies(churn_pd)    print('churn_pd-->\n', churn_pd)    churn_pd.info()    # 2 去除列 Churn_no gender_Male # inplace=True 在原来的数据上进行删除    churn_pd.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)    churn_pd.info()    # 3 列标签重命名 打印列名    print('churn_pd.columns', churn_pd.columns)    churn_pd.rename(columns = {'Churn_Yes':'flag'}, inplace=True)    print('churn_pd.columns', churn_pd.columns)    # 4 查看标签的分布情况 0.26用户流失    value_counts = churn_pd.flag.value_counts(1)    print('value_counts-->\n', value_counts)    print('从标签的分类中可以看出: 属于标签分类不平衡样本')
> ```

> 【代码】2 特征筛选（PPT slide 66）

```python
import matplotlib.pyplot as plt
import seaborn as sns

def dm02_特征筛选():
    churn_pd = pd.read_csv('./data/churn.csv')
    # 1 处理类别型的数据 类别型数据做one-hot编码
    churn_pd = pd.get_dummies(churn_pd)
    # 2 去除列 Churn_no gender_Male # nplace=True 在原来的数据上进行删除
    churn_pd.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)
    # 3 列标签重命名 打印列名
    churn_pd.rename(columns={'Churn_Yes': 'flag'}, inplace=True)
    # 4 查看标签的分布情况 0.26用户流失
    value_counts = churn_pd.flag.value_counts(1)
    # 5 查看Contract_Month 是否月签约流失情况
    sns.countplot(data=churn_pd, y="Contract_Month", hue='flag')
    plt.show()
```

> 旁注：PPT 原文为单行连写 / 全角引号，按 sklearn 标准排版补全；注释 "nplace=True" 疑似 typo（应为 inplace），按 PPT 原样保留。
>
> PPT 原文（单行连写、全角引号）：
>
> ```python
> import matplotlib.pyplot as pltimport seaborn as snsdef dm02_特征筛选():    churn_pd = pd.read_csv('./data/churn.csv')    # 1 处理类别型的数据 类别型数据做one-hot编码    churn_pd = pd.get_dummies(churn_pd)    # 2 去除列 Churn_no gender_Male # nplace=True 在原来的数据上进行删除    churn_pd.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)    # 3 列标签重命名 打印列名    churn_pd.rename(columns={'Churn_Yes': 'flag'}, inplace=True)    # 4 查看标签的分布情况 0.26用户流失    value_counts = churn_pd.flag.value_counts(1)      # 5 查看Contract_Month 是否月签约流失情况    sns.countplot(data=churn_pd, y = "Contract_Month", hue='flag')    plt.show()
> ```

> 【代码】3 模型训练与评测（PPT slide 67）

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

def dm03_模型训练和评测():
    # 1 数据基本处理
    churn_pd = pd.read_csv('./data/churn.csv')
    # 1-1 处理类别型的数据 类别型数据做one-hot编码
    churn_pd = pd.get_dummies(churn_pd)
    # 1-2 去除列 Churn_no gender_Male # nplace=True 在原来的数据上进行删除
    churn_pd.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)
    # 1-3 列标签重命名 打印列名
    churn_pd.rename(columns={'Churn_Yes': 'flag'}, inplace=True)
    # 2 特征处理
    # 2-1 确定目标值和特征值
    x = churn_pd[['Contract_Month', 'internet_other', 'PaymentElectronic']]
    y = churn_pd['flag']
    # 2-2 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
 random_state=100)
    # 3 实例化模型 训练模型 模型预测
    estimator =  LogisticRegression()
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_test)
    # 4 模型评估
    my_accuracy_score = accuracy_score(y_test, y_pred)
    print('my_accuracy_score-->', my_accuracy_score)
    my_score = estimator.score(x_test, y_test)
    print('my_score-->', my_score)
    # 计算AUC值
    my_roc_auc_score = roc_auc_score(y_test, y_pred)
    print('my_roc_auc_score-->', my_roc_auc_score)

from sklearn.metrics import classification_report
    result = classification_report(y_test, y_pred, target_names=['flag0', 'flag1'])
    print('classification_report result->\n', result)
```

> 旁注：PPT 原文为单行连写，按 sklearn 标准排版补全；末尾 `from sklearn.metrics import classification_report` 在 PPT 中处于函数体之外（缩进异常），且 result/print 又回到缩进，PPT 原 PPT 写法即如此（疑似复制粘贴错位），按 PPT 原样保留。
>
> PPT 原文（单行连写、末尾 import 错位）：
>
> ```python
> from sklearn.model_selection import train_test_splitfrom sklearn.linear_model import LogisticRegressionfrom sklearn.metrics import accuracy_score, roc_auc_scoredef dm03_模型训练和评测():    # 1 数据基本处理    churn_pd = pd.read_csv('./data/churn.csv')    # 1-1 处理类别型的数据 类别型数据做one-hot编码    churn_pd = pd.get_dummies(churn_pd)    # 1-2 去除列 Churn_no gender_Male # nplace=True 在原来的数据上进行删除    churn_pd.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)    # 1-3 列标签重命名 打印列名    churn_pd.rename(columns={'Churn_Yes': 'flag'}, inplace=True)    # 2 特征处理    # 2-1 确定目标值和特征值    x = churn_pd[['Contract_Month', 'internet_other', 'PaymentElectronic']]    y = churn_pd['flag']    # 2-2 数据集划分    x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size=0.3,
>  random_state=100)
>     # 3 实例化模型 训练模型 模型预测    estimator =  LogisticRegression()    estimator.fit(x_train, y_train)    y_pred = estimator.predict(x_test)    # 4 模型评估    my_accuracy_score = accuracy_score(y_test, y_pred)    print('my_accuracy_score-->', my_accuracy_score)    my_score = estimator.score(x_test, y_test)    print('my_score-->', my_score)    # 计算AUC值    my_roc_auc_score = roc_auc_score(y_test, y_pred)    print('my_roc_auc_score-->', my_roc_auc_score)
>
> from sklearn.metrics import classification_report
>     result = classification_report(y_test, y_pred, target_names=['flag0', 'flag1'])
>     print('classification_report result->\n', result)
> ```

> 【分析】分析报告（PPT slide 68）

〔图：图片8.jpg —— 背景图案〕
分析报告
classification_report 在用户流失分析的案例

〔图：图片3.jpg —— 表格〕
〔图：图片6.jpg —— 手机屏幕截图〕
当非流失客户作为正例时
当流失客户作为正例时

> 【图示】PPT slide 69

〔图：图片5.jpg —— 背景图案〕
〔图：图片7.jpg —— 图片包含 文本〕

\#

> **Notes**（讲师）：类比 举例：样本极少  统计误差大：比如  预测中国男女比例

> 【图示】PPT slide 70

〔图：图片4.jpg —— 表格〕

> 备注：纯图示页。
