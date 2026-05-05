---
tags: [算法/逻辑回归, 代码/sklearn, 评估/分类报告]
---

# 分类报告 API

> 维度：**代码**
> 知识点级别：【实操】sklearn classification_report 使用
> 章节底稿全文见 [`README.md`](./README.md)（PPT slide 56）

## ━━━━━━━━ 底稿 ━━━━━━━━

### PPT

> Slide 56 · 分类评估报告API

分类评估报告api

```python
sklearn.metrics.classification_report(y_true, y_pred, labels=[], target_names=None)
```

- y_true：真实目标值
- y_pred：估计器预测目标值
- labels：指定类别对应的数字
- target_names：目标类别名称
- return：每个类别精确率与召回率

### 笔记

> 分类评估报告 API

一次性给出**每个类别**的 Precision / Recall / F1 / Support，避免手动从混淆矩阵抠数。

```python
sklearn.metrics.classification_report(
    y_true,                # 真实标签
    y_pred,                # 预测标签（不是概率）
    labels=[],             # 指定要展示的类别编号
    target_names=None,     # 类别可读名（替换数字标签）
)
```

**关键点**：

- 输入是**离散预测**（`predict` 的输出），不是概率
- 输出 `macro avg`（各类等权平均）和 `weighted avg`（按 support 加权）
- `target_names` 长度必须与 `labels` 一致
- 不平衡场景重点看**少数类的 Recall**

---

## ━━━━━━━━ 讲解 ━━━━━━━━

### 生活锚 · 体检综合报告单

去医院体检完，护士不会让你拿着血常规、肝功、血糖、血脂 4 张化验单自己去对照参考值——她直接把一张 A4 报告单递到你手上：

- 上面 12 行指标，每行 4 列：项目名 / 你的值 / 参考范围 / 是否异常
- 末尾还有一行"综合评估"——总体看哪些指标拖后腿
- 整页一目了然，你 30 秒就知道这次体检主要问题在哪

要是没有这张报告单，你得拿着 4 张化验单一个个查参考值、做减法、判超标——3 张以上几乎必看错一项。本节要的就是给分类模型出这种"一张报告单"的工具。

### 业务问题

混淆矩阵 4 格 + Precision / Recall / F1 在 04a / 04b 已展开。手算流程是：拿混淆矩阵 → 4 个除法算 P/R → 调和平均算 F1 → 每个类重复一遍。3 类以上 + 不平衡场景，手算几乎必错。

`classification_report` 把这一切打成**一张可读表**，一行调用就够了。

### 【代码】API 解读

```python
from sklearn.metrics import classification_report

print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["未点击", "点击"],
    digits=3,           # 小数位，默认 2
))
```

输出长这样：

```
              precision    recall  f1-score   support

         未点击      0.95      0.98      0.96       400
          点击      0.80      0.62      0.70        50

    accuracy                          0.94       450
   macro avg      0.88      0.80      0.83       450
weighted avg      0.93      0.94      0.93       450
```

每一行 / 列读法：

- **每个类一行**：该类作为正类时的 P / R / F1 + support（该类样本数）
- **accuracy**：所有样本的整体准确率，只有一个数字横跨 P/R/F1 列
- **macro avg**：各类的 P/R/F1 **算术平均**，每类等权——少数类被放大
- **weighted avg**：各类按 support **加权平均**——多数类主导

### macro vs weighted：选哪个

| 场景 | 选 | 理由 |
|---|---|---|
| 不平衡数据，关心少数类 | **macro avg** | 少数类和多数类一样有发言权 |
| 平衡数据 / 关心整体 | **weighted avg** ≈ accuracy | 反映平均水平 |
| 报论文 / 比赛榜单 | 两个都报 | 信息全 |

广告 CTR 数据点击率 1%——多数类（不点击）就算瞎猜也 99% 准。**weighted avg 会假装一切美好**，**macro avg 会暴露少数类的烂 Recall**。生产环境永远盯 macro，accuracy 别看。

### 【代码】关键参数

- **`labels`**：指定输出哪些类、按什么顺序。多分类不传时按字母序排，可能不是你想要的
- **`target_names`**：显示的类名，长度**必须与 `labels` 一致**（少了报错）。建议传业务可读名
- **`output_dict=True`**：返回 dict 而不是字符串，方便程序化抓取（如写到 MLflow / 飞书）
- **`zero_division=0`**：当某类没有任何预测时，P 的分母为 0。默认 warn 并给 0；显式传 0 / 1 可以静默
- **`digits`**：小数位数，默认 2，写论文调到 3 ~ 4

### 【代码】完整可运行 demo

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# 造一个 95:5 不平衡的二分类数据
X, y = make_classification(
    n_samples=2000, n_features=10, weights=[0.95, 0.05],
    random_state=42,
)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
y_pred = clf.predict(X_te)   # 注意：这里要 predict（标签），不是 predict_proba

print(classification_report(
    y_te, y_pred,
    labels=[0, 1],
    target_names=["majority", "minority"],
    digits=3,
))
```

### 常见坑

| 坑 | 现象 | 修法 |
|---|---|---|
| 把 `predict_proba` 输出传进去 | 报错或全是 0 | 改传 `predict` 的离散标签 |
| `target_names` 长度 ≠ `labels` 长度 | `ValueError` | 对齐两个列表长度 |
| 不平衡场景只看 accuracy | 模型瞎猜也 99%，假装没问题 | 看 macro avg + 少数类 Recall |
| `labels` 漏了某个出现的类 | 该类不被打印但仍算入 accuracy 分母 | 传全集或省略 `labels` |

**这一节的关键启示**：一行打印 P/R/F1，不平衡场景永远盯 macro avg 和少数类 Recall。

→ 下一步：[`05-auc计算api.md`](./05-auc计算api.md) 用 sklearn 算 AUC。

### 术语

| 故事里的元素 | 术语名 | 主场 |
|---|---|---|
| 一行打印 P/R/F1 的 sklearn 函数 | `classification_report` | 本节 |
| 各类等权平均 | macro average | 本节 |
| 按样本数加权平均 | weighted average | 本节 |
| 该类的样本数 | support | 本节 |
| 精确率 / 召回率 / F1 分数 | Precision / Recall / F1-score | 04a / 04b 主场 |

> Sources：
> - PPT Slide 56
> - sklearn.metrics.classification_report
