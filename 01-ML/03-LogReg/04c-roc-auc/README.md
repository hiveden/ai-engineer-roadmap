# 第 4 章 c · ROC 曲线 + AUC 指标

> 维度：**数学 + 代码**
> 学习目标见 [`04a-confusion-matrix/README.md`](../04a-confusion-matrix/README.md)
>
> **按知识点拆分的讲解版**：
>
> 1. [`01-roc曲线.md`](./01-roc曲线.md) — 【知道】
> 2. [`02-绘制roc.md`](./02-绘制roc.md) — 【实操】
> 3. [`03-auc.md`](./03-auc.md) — 【知道】AUC 值
> 4. [`04-分类报告api.md`](./04-分类报告api.md) — 【代码】
> 5. [`05-auc计算api.md`](./05-auc计算api.md) — 【代码】

## 底稿

> 04 · 分类评估方法（ROC + AUC）

> 【知道】ROC 曲线和 AUC 指标

样本不平衡时，单看 Recall / Precision 易被多数类淹没。ROC + AUC 的核心：**抛开单一阈值，直接评估模型对正负样本的整体辨别能力**。

> ROC 曲线

ROC（Receiver Operating Characteristic）曲线把分类器在**所有阈值**下的表现画成一条线。两个轴分别看正负样本：

| 轴 | 含义 | 公式 |
|---|---|---|
| 纵轴 **TPR**（True Positive Rate） | 正样本中被预测为正的概率 = 召回率 Recall | TP / (TP + FN) |
| 横轴 **FPR**（False Positive Rate） | 负样本中被预测为正的概率 = 误报率 | FP / (FP + TN) |

**直觉**：TPR 越高 = 抓正样本越准；FPR 越低 = 误伤负样本越少。理想模型 TPR=1 且 FPR=0，即落在 **(0, 1)** 点。

四个特殊点（横轴 FPR，纵轴 TPR）：

| 点 | 含义 |
|---|---|
| (0, 0) | 全预测为负：正样本全错，负样本全对 |
| (1, 1) | 全预测为正：正样本全对，负样本全错 |
| (0, 1) | 完美分类器 |
| (1, 0) | 全反着来（正负全错） |

对角线 y = x 是**随机猜测**基线（TPR = FPR），曲线必须显著高于对角线才算有判别力。

> 绘制 ROC 曲线

**核心机制**：模型输出的是概率，调阈值 → 划分正负 → 算一组 (FPR, TPR) → 连成曲线。

**画法骨架**（4 步）：

1. 模型对每个样本输出**预测概率** `y_score`
2. 按 `y_score` **降序排序**
3. 从高到低**遍历每个分数作为阈值**：≥ 阈值判正，< 阈值判负
4. 每个阈值算一组 (FPR, TPR) → 全部点连线

**广告点击案例**（6 样本，2 正 4 负）：按概率降序遍历，每跨过一个正样本 TPR 跳一格，每跨过一个负样本 FPR 跳一格。最终得到点序列 `(0, 0.5) → (0, 1) → (0.25, 1) → (0.5, 1) → (0.75, 1) → (1, 1)`，连线就是 ROC。

**几何直觉**：

- 排序后**正样本集中在前** → 曲线先沿纵轴爬到顶 → 形状贴左上角 → 模型好
- 正负样本**交错** → 曲线接近对角线 → 模型烂

> AUC 值

**AUC**（Area Under Curve）= ROC 曲线下面积，把"形状好不好"压成一个 **[0, 1]** 数字。

**解读速查**：

| AUC | 含义 |
|---|---|
| 1.0 | 完美分类器（几乎不存在） |
| 0.5 | 随机猜测，模型无效 |
| < 0.5 | 比随机还差，预测反着用即可 |
| 0.7 ~ 0.85 | 工程实践中常见的"可用"区间 |
| > 0.9 | 强判别力（警惕数据泄漏） |

**概率解释**：随机抽一个正样本和一个负样本，AUC = 模型给正样本打分 > 给负样本打分的概率。

**核心定位**：AUC 评估的是**正负样本的辨别能力**，与阈值选择无关 → 适合不平衡场景。

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

## 【实践】电信客户流失预测
