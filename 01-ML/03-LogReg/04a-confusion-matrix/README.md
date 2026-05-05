# 第 4 章 a · 混淆矩阵

> 维度：**概念 + 数学**
>
> **按知识点拆分的讲解版**：
>
> 1. [`01-混淆矩阵.md`](./01-混淆矩阵.md) — 【理解】TP / FP / TN / FN

## ━━━━━━━━ 底稿 ━━━━━━━━

### PPT

> 从 [`../04逻辑回归.pptx`](../04逻辑回归.pptx) 提取（slide 30-36）。图占位标 〔图〕；排版整理；文字保留 PPT 原话。

> Slide 30 · 目录（章间导航）

- 逻辑回归简介（应用场景，数学知识）
- 逻辑回归原理
- 逻辑回归API函数和案例
- 分类问题评估（混淆矩阵、精确率、召回率、F1-score、AUC指标、ROC曲线）
- 电信客户流失预测案例

> Slide 31 · 学习目标（04a / 04b / 04c 共享）

- 理解混淆矩阵的构建方法
- 掌握精确率，召回率和F1score的计算方法

> **Notes**：如果预测流失，营销团队会采用大折扣挽回；如果预测不会流失，则可能爱里不搭。

> Slide 32 · 引子 – 只做预测准确率能满足各种场景需要吗？

只做预测准确率能满足各种场景需要吗？
比如上述癌症检测的案例
        癌症患者有没有被全部预测（检测）出来。

> **Notes**：7043行 16列 没有空值数据；Gender gender_male gender_female；churn churn_yes churn_no；然后舍弃

> Slide 33 · 分类问题的评估方法 – 混淆矩阵（概念）

分类问题的评估方法 – 混淆矩阵

什么是混淆矩阵？

某医院疾病检测案例

混淆矩阵四个指标

真实值是 正例 的样本中，被分类为 正例 的样本数量有多少，叫做真正例（TP，True Positive）
真实值是 正例 的样本中，被分类为 假例 的样本数量有多少，叫做伪反例（FN，False Negative）
真实值是 假例 的样本中，被分类为 正例 的样本数量有多少，叫做伪正例（FP，False Positive）
真实值是 假例 的样本中，被分类为 假例 的样本数量有多少，叫做真反例（TN，True Negative）

注意：TP+FN+FP+TN = 总样本数量

命名方式：

〔图：命名方式说明图〕

```
混淆矩阵布局（行=真实，列=预测）：

              预测 正例    预测 负例
真实 正例  |   TP        |   FN       |
真实 负例  |   FP        |   TN       |
```

> Slide 34 · 分类评估方法 – 混淆矩阵（举例题面）

分类评估方法 – 混淆矩阵

混淆矩阵-举个例子

已知：样本集10样本，有 6 个恶性肿瘤样本，4 个良性肿瘤样本，我们假设恶性肿瘤为正例

注意：TP+FN+FP+TN = 总样本数量

模型A：预测对了 3 个恶性肿瘤样本，4 个良性肿瘤样本
请计算：TP、FN、FP、TN

模型B：预测对了 6 个恶性肿瘤样本，1个良性肿瘤样本
请计算：TP、FN、FP、TN

模型A：
- 真正例 TP 为：
- 伪反例 FN 为：
- 伪正例 FP 为：
- 真反例 TN：

模型B：
- 真正例 TP 为：
- 伪反例 FN 为：
- 伪正例 FP 为：
- 真反例 TN：

> Slide 35 · 分类评估方法 – 混淆矩阵（答案揭晓）

分类评估方法 – 混淆矩阵

混淆矩阵-举个例子

已知：样本集10样本，有 6 个恶性肿瘤样本，4 个良性肿瘤样本，我们假设恶性肿瘤为正例

注意：TP+FN+FP+TN = 总样本数量

模型A：预测对了 3 个恶性肿瘤样本，4 个良性肿瘤样本

- 真正例 TP 为：3
- 伪反例 FN 为：3
- 伪正例 FP 为：0
- 真反例 TN：4

模型B：预测对了 6 个恶性肿瘤样本，1个良性肿瘤样本

- 真正例 TP 为：6
- 伪反例 FN 为：0
- 伪正例 FP 为：3
- 真反例 TN：1

> Slide 36 · 分类评估方法 – 混淆矩阵（程序计算）

分类评估方法 – 混淆矩阵

混淆矩阵 - 举个例子（PPT 原文单行连写）：

```python
from sklearn.metrics import confusion_matrix
import pandas as pd

def dm01_混淆矩阵四个指标():
    # 样本集中共有6个恶性肿瘤样本, 4个良性肿瘤样本
    y_true = ["恶性", "恶性", "恶性", "恶性", "恶性", "恶性", "良性", "良性", "良性", "良性"]
    labels = ["恶性", "良性"]
    dataframe_labels = ["恶性(正例)", "良性(反例)"]
    # 1. 模型 A: 预测对了3个恶性肿瘤样本, 4个良性肿瘤样本
    print("模型A:")
    print("-" * 13)
    y_pred1= ["恶性", "恶性", "恶性", "良性", "良性", "良性", "良性", "良性", "良性", "良性"]
    result = confusion_matrix(y_true, y_pred1,  labels=labels)
    print(pd.DataFrame(result, columns=dataframe_labels,  index=dataframe_labels))
    # 2. 模型 B: 预测对了6个恶性肿瘤样本, 1个良性肿瘤样本
    print("模型B:")
    print("-" * 13)
    y_pred2= ["恶性", "恶性", "恶性", "恶性", "恶性", "恶性", "恶性", "恶性", "恶性", "良性"]
    result = confusion_matrix(y_true, y_pred2, labels=labels)
    print(pd.DataFrame(result,  columns=dataframe_labels，  index=dataframe_labels))
```

> ⚠ PPT 原文末行 `dataframe_labels` 后有中文全角逗号"，"作为分隔符（typo），按 PPT 原样保留。

### 笔记

> 04 · 分类评估方法

**学习目标**（04a / 04b / 04c 共享）：

1. 理解混淆矩阵的构建方法
2. 掌握精确率，召回率和 F1-score 的计算方法
3. 知道 ROC 曲线和 AUC 指标

> 【引子】只做预测准确率能满足各种场景需要吗？（PPT slide 32）

只做预测准确率能满足各种场景需要吗？
比如上述癌症检测的案例
        癌症患者有没有被全部预测（检测）出来。

> 【理解】混淆矩阵

混淆矩阵作用在测试集上，把"真实标签 × 预测标签"的二维交叉计数列出来。二分类下 4 个格子，多分类下 N×N。

**命名规则**：第二个字母（P/N）= 预测的类别；第一个字母（T/F）= 预测对了没有。所以 TP = 预测为正且预测对了；FP = 预测为正但预测错了（实际是负）。

**4 个核心概念**（约定正例 = 我们关心的少数类，例如恶性肿瘤、欺诈交易）：

- **TP**（True Positive，真正例）：实际正 + 预测正 → 命中
- **FN**（False Negative，伪反例）：实际正 + 预测负 → 漏报（病人被判为健康）
- **FP**（False Positive，伪正例）：实际负 + 预测正 → 误报（健康人被判为病人）
- **TN**（True Negative，真反例）：实际负 + 预测负 → 正确放过

**矩阵布局**（行 = 真实，列 = 预测）：

|              | 预测 正例 | 预测 负例 |
|--------------|----------|----------|
| **真实 正例** | TP        | FN        |
| **真实 负例** | FP        | TN        |

**恒等式**：`TP + FN + FP + TN = 测试集总样本数`。

**例子**：6 个恶性 + 4 个良性，正例 = 恶性。

| 模型 | TP | FN | FP | TN | 解读 |
|------|----|----|----|----|------|
| A    | 3  | 3  | 0  | 4  | 不敢报恶性 → 漏诊一半 |
| B    | 6  | 0  | 3  | 1  | 宁可错杀 → 恶性全抓但误报 3 个良性 |

> 同一个准确率 `(TP+TN)/总数` 可能掩盖完全不同的错误结构 —— 这就是后面要引入 Precision / Recall / F1 的原因。

> PPT 原话（混淆矩阵概念，slide 33）：
>
> 某医院疾病检测案例
> 什么是混淆矩阵？
>
> 〔图：图片18.jpg —— 表格〕
> 命名方式：
>
> 〔图：图片15.jpg —— 文本〕
> 混淆矩阵四个指标
> 真实值是 正例 的样本中，被分类为 正例 的样本数量有多少，叫做真正例（TP，True Positive）
> 真实值是 正例 的样本中，被分类为 假例 的样本数量有多少，叫做伪反例（FN，False Negative）
> 真实值是 假例 的样本中，被分类为 正例 的样本数量有多少，叫做伪正例（FP，False Positive）
> 真实值是 假例 的样本中，被分类为 假例 的样本数量有多少，叫做真反例（TN，True Negative）
> 注意：TP+FN+FP+TN = 总样本数量
>
> 讲师备注：应用场景  缺陷检测   垃圾邮件分类

> PPT 原话（混淆矩阵 - 举个例子题面，slide 34）：
>
> 已知：样本集10样本，有 6 个恶性肿瘤样本，4 个良性肿瘤样本，我们假设恶性肿瘤为正例
> 模型A：预测对了 3 个恶性肿瘤样本，4 个良性肿瘤样本
> 请计算：TP、FN、FP、TN
> 模型B：预测对了 6 个恶性肿瘤样本，1个良性肿瘤样本
> 请计算：TP、FN、FP、TN
> 真正例 TP 为：
> 伪反例 FN 为：
> 伪正例 FP 为：
> 真反例 TN：
> 真正例 TP 为：
> 伪反例 FN 为：
> 伪正例 FP 为：
> 真反例 TN：
>
> 〔图：图片15.jpg —— 表格〕
> 注意：TP+FN+FP+TN = 总样本数量

> PPT 原话（混淆矩阵 - 答案揭晓，slide 35）：
>
> 真正例 TP 为：3
> 伪反例 FN 为：3
> 伪正例 FP 为：0
> 真反例 TN：4
> 真正例 TP 为：6
> 伪反例 FN 为：0
> 伪正例 FP 为：3
> 真反例 TN：1

> 【代码】混淆矩阵 - 程序计算（PPT slide 36）

```python
from sklearn.metrics import confusion_matrix
import pandas as pd

def dm01_混淆矩阵四个指标():
    # 样本集中共有6个恶性肿瘤样本, 4个良性肿瘤样本
    y_true = ["恶性", "恶性", "恶性", "恶性", "恶性", "恶性", "良性", "良性", "良性", "良性"]
    labels = ["恶性", "良性"]
    dataframe_labels = ["恶性(正例)", "良性(反例)"]
    # 1. 模型 A: 预测对了3个恶性肿瘤样本, 4个良性肿瘤样本
    print("模型A:")
    print("-" * 13)
    y_pred1 = ["恶性", "恶性", "恶性", "良性", "良性", "良性", "良性", "良性", "良性", "良性"]
    result = confusion_matrix(y_true, y_pred1, labels=labels)
    print(pd.DataFrame(result, columns=dataframe_labels, index=dataframe_labels))
    # 2. 模型 B: 预测对了6个恶性肿瘤样本, 1个良性肿瘤样本
    print("模型B:")
    print("-" * 13)
    y_pred2 = ["恶性", "恶性", "恶性", "恶性", "恶性", "恶性", "恶性", "恶性", "恶性", "良性"]
    result = confusion_matrix(y_true, y_pred2, labels=labels)
    print(pd.DataFrame(result, columns=dataframe_labels， index=dataframe_labels))
```

> 旁注：PPT 原文为单行连写 / 中文全角引号 + 末行 dataframe_labels 后有中文全角逗号"，"作为分隔符（typo），按 sklearn 标准排版补全；末行全角逗号按 PPT 原样保留。
>
> PPT 原文（单行连写、引号混合全角/半角）：
>
> ```python
> from sklearn.metrics import confusion_matriximport pandas as pddef dm01_混淆矩阵四个指标():    # 样本集中共有6个恶性肿瘤样本, 4个良性肿瘤样本    y_true = ["恶性", "恶性", "恶性", "恶性", "恶性", "恶性", "良性", "良性", "良性", "良性"]    labels = ["恶性", "良性"]    dataframe_labels = ["恶性(正例)", "良性(反例)"]    # 1. 模型 A: 预测对了3个恶性肿瘤样本, 4个良性肿瘤样本    print("模型A:")    print("-" * 13)    y_pred1= ["恶性", "恶性", "恶性", "良性", "良性", "良性", "良性", "良性", "良性", "良性"]    result = confusion_matrix(y_true, y_pred1,  labels=labels)    print(pd.DataFrame(result, columns=dataframe_labels,  index=dataframe_labels))    # 2. 模型 B: 预测对了6个恶性肿瘤样本, 1个良性肿瘤样本    print("模型B:")    print("-" * 13)    y_pred2= ["恶性", "恶性", "恶性", "恶性", "恶性", "恶性", "恶性", "恶性", "恶性", "良性"]    result = confusion_matrix(y_true, y_pred2, labels=labels)    print(pd.DataFrame(result,  columns=dataframe_labels，  index=dataframe_labels))
> ```
