---
tags: [算法/逻辑回归, 代码/sklearn, 评估/AUC]
---

# AUC 计算 API

> 维度：**代码**
> 知识点级别：【实操】roc_auc_score + roc_curve 配合使用
> 章节底稿全文见 [`README.md`](./README.md)（PPT slide 55）

## ━━━━━━━━ 底稿 ━━━━━━━━

### PPT

> Slide 55 · 分类评估方法 – AUC计算API

分类评估方法 – ROC曲线、AUC指标

AUC的计算api

```python
from sklearn.metrics import roc_auc_score
sklearn.metrics.roc_auc_score(y_true, y_predict)
```

- 计算ROC曲线面积，即AUC值
- y_true：每个样本的真实类别，必须为0(反例),1(正例)标记
- y_predict：预测得分，可以是正例的估计概率、置信值或者分类器方法的返回值

### 笔记

> AUC 计算 API

```python
from sklearn.metrics import roc_auc_score

sklearn.metrics.roc_auc_score(
    y_true,    # 真实类别，必须是 0/1（反例/正例）
    y_score,   # 预测得分：正例概率 / 置信度 / decision_function 返回值
)
```

**关键点**：

- `y_score` 传**概率或得分**（如 `model.predict_proba(X)[:, 1]`），**不是 `predict` 的 0/1 标签**——传标签会退化为单点 AUC，失去曲线意义
- 标签必须是 0/1；其他取值（如 1/2）需先重映射
- 多分类用 `multi_class='ovr'` / `'ovo'` 参数
- 配套绘图：`sklearn.metrics.roc_curve(y_true, y_score)` → 返回 `fpr, tpr, thresholds`

---

## ━━━━━━━━ 讲解 ━━━━━━━━

### 生活锚 · 电子血压计

家里老人量血压，没人还在用水银柱听诊器：

- **手动版**：袖带充气、放气、听科氏音、看刻度——每一步都可能错，自己量给自己听根本听不准
- **电子版**：袖带绑好、按一下"开始"——30 秒后屏幕直接跳出"高压 128 / 低压 82 / 心率 76"，连判读都帮你做完

测的还是同一根血管，原理没变，但调用方式从"5 步手动操作"压成"按一个按钮读一个数"。出错的可能性也从"5 处都可能错"压成"袖带没绑紧"这一处。本节的 AUC 计算 API 就是这种"按一下出数"的工具——03 节那条手算 7 个点连线再求面积的流程，被压成一行函数调用。

### 业务问题

03 节算 AUC 是手算面积，工程不会真的去积分。`roc_auc_score` 一行算完，配套 `roc_curve` 还能拿到画图所需的所有点。两者搭配是评估二分类的标配。

### 【代码】API 解读

```python
from sklearn.metrics import roc_auc_score, roc_curve

auc = roc_auc_score(y_true, y_score)
fpr, tpr, thresholds = roc_curve(y_true, y_score)
```

- **`y_true`**：真实标签，**必须是 0/1**。1/2、True/False 这类要先 `(y == 正类).astype(int)` 转
- **`y_score`**：连续打分，**必须是模型对"正类"的预测概率或决策值**——
  - LR / 树模型：`model.predict_proba(X)[:, 1]`（取第 2 列即正类概率）
  - SVM：`model.decision_function(X)`（不需要是概率，单调即可）
- **返回**：单个 float，AUC 值

`roc_curve` 返回三个数组：

- `fpr`、`tpr`：可直接 `plt.plot(fpr, tpr)` 画图
- `thresholds`：与 (fpr, tpr) 一一对应的阈值，长度比样本多 1（首位是哨兵阈值 `np.inf`，对应 (0, 0) 点）

### 【代码】反例：传 0/1 标签的退化

最高频的坑——把 `predict` 的 0/1 标签当 `y_score` 传：

```python
# ❌ 反例：传 0/1 标签
y_pred = clf.predict(X_te)            # 离散 0/1
roc_auc_score(y_te, y_pred)           # 退化为单点 AUC

# ✅ 正例：传正类概率
y_score = clf.predict_proba(X_te)[:, 1]
roc_auc_score(y_te, y_score)
```

为什么退化？标签只有 0 和 1 两种取值，等价于"只有一个阈值"——ROC 曲线只能产生一个非平凡点，AUC 退化为该点对应的梯形面积，远低于真实曲线下面积。

模型实际能力 AUC = 0.92，传错变成 0.78——你以为模型变烂了，其实是评估姿势错。

### 常见坑

| 坑 | 现象 | 修法 |
|---|---|---|
| 传 0/1 标签当 `y_score` | AUC 偏低 | 改用 `predict_proba(X)[:, 1]` |
| `predict_proba` 取错列 | AUC ≈ 1 - 真实值 | 取**第 2 列**（正类列），不是 `[:, 0]` |
| 标签是 1/2 而不是 0/1 | `ValueError` 或结果错 | `(y == 2).astype(int)` 重映射 |
| 全是同一类 | `ValueError: Only one class present` | 检查数据划分，确保 train/test 都包含正负 |
| 多分类直接调用 | 报错 | 加 `multi_class='ovr'` 或 `'ovo'` 参数 |

### 【代码】多分类

`roc_auc_score` 在 sklearn ≥ 0.22 起支持多分类，需指定策略：

- `multi_class='ovr'`（One-vs-Rest）：每个类对其余所有类算一个 AUC，再平均
- `multi_class='ovo'`（One-vs-One）：每两类一对算 AUC，再平均

```python
roc_auc_score(y_true, y_score_matrix, multi_class='ovr', average='macro')
```

数学细节不在本章范围。

### 【代码】完整可运行 demo

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=2000, n_features=10,
                           weights=[0.9, 0.1], random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
y_score = clf.predict_proba(X_te)[:, 1]

auc = roc_auc_score(y_te, y_score)
fpr, tpr, thresholds = roc_curve(y_te, y_score)

plt.plot(fpr, tpr, label=f"LR (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="random")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.title("ROC curve")
plt.show()
```

**这一节的关键启示**：`y_score` 必须传概率不是 0/1 标签——这是 AUC API 头号坑。

### 术语

| 故事里的元素 | 术语名 | 主场 |
|---|---|---|
| 一行算 AUC 的 sklearn 函数 | `roc_auc_score` | 本节 |
| 返回 fpr / tpr / thresholds 数组 | `roc_curve` | 本节 |
| 模型给正类的概率 | `predict_proba(X)[:, 1]` | 本节 |
| SVM 给的非概率得分 | `decision_function` 输出 | 本节 |
| 多分类 AUC 策略 | One-vs-Rest / One-vs-One | 多分类章主场 |

> Sources：
> - PPT Slide 55
> - sklearn.metrics.roc_auc_score
> - sklearn.metrics.roc_curve
