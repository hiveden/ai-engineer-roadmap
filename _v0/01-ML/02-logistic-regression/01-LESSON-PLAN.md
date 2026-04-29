# LESSON-PLAN · 第 2 节 · 逻辑回归（demo-02 乳腺癌）

> **受众**：未来某次 session 开始时需要带 Alex 过逻辑回归 / demo-02 的 Claude。
>
> **这是备课资料库，不是教案。** 讲解方式由当时的 Claude 判断；这份文档确保你不会遇到"学生突然问了个合理问题，你没备到"。
>
> **10 节课里的第 2 节**。算法清单见 [`demos/README.md`](../README.md)。上一节见 [`demos/demo-01-house-price/LESSON-PLAN.md`](../demo-01-house-price/LESSON-PLAN.md)。

---

## 0. 课前须知（防复读失败）

### 0.1 这节课的位置

| 维度 | 值 |
|---|---|
| 算法 | 逻辑回归（LogisticRegression） |
| 任务类型 | 二分类（输出离散类别 + 概率） |
| 数据集 | 乳腺癌（569, 30） |
| roadmap 阶段 | 阶段一·传统 ML 工程化·第 2 层·接口层 |
| 目标深度（来自 demos/README.md） | **L3** — 能讲清楚每步动机 + 能做选型判断，不要求手写代码 |
| 在 10 节课里 | 第 2 节（对比回归 vs 分类的最基础一刀） |
| 前置 | 第 1 节线性回归 demo-01 |

### 0.2 当前学习状态（2026-04-19 快照）

- A 组 10 概念 + B 组 5 工程坑 Level 2 通关于 session-04
- demo-01 跑通于 session-06，4 步流程能自述
- demo-02 step 1/2/3 跑通于 session-07
- 过拟合 / 泛化 Level 2 于 session-07（原话："记住了答案没记住规律，不能泛化"）
- 置信度 Level 2 于 session-07：
  - 知道 `predict_proba` 返回两列（类别 0 概率、类别 1 概率）
  - 知道 `predict()` 按概率最大选类别（argmax）
  - 知道置信度 = 那行里最大的那个数（不是偏移量 0.88-0.5）
  - 知道 `classes_` 是 sklearn 列顺序的权威出口
- step4_evaluate.py **未跑过**，这是下一步起点

**如果又回炉重讲已通关概念 = 浪费时间**。先 attune。

---

## 1. 讲义集成（迁移自 `01-Machine-Learning-Foundation/04-Logistic-Regression-and-Classification-Metrics.md`）

> 原文档已删除，内容迁入本备课。Sigmoid 公式、Log-Loss、MLE 等数学推导未迁入。

### 1.1 核心隐喻 · 给线性回归穿上 Sigmoid 外套

线性回归输出连续值（-∞ 到 +∞），但分类需要 [0, 1] 的概率。逻辑回归多了一步"挤压"：

```
第 1 步：和线性回归一样，算出一个任意实数（w·x + b）
第 2 步：套进 sigmoid 函数，压到 0~1 区间
第 3 步：把这个 0~1 的数当作"类别 1 的概率"
```

> ⚠️ **不写 sigmoid 公式**。学生追问"sigmoid 是什么"：下限答案 "一个数学函数，能把任何数字压到 0 到 1 之间。具体长什么样不重要，知道它干这个事就行"。

**为什么不直接用线性回归 + 0.5 阈值做分类**：
- 线性回归对离群点敏感，一个极端值能把决策边界拉歪
- 输出不是概率，不能做置信度判断
- 工程上损失函数不"凸"，优化容易陷局部最优

学生追问到此：停。不讲 Log-Loss / MLE。

### 1.2 评估指标 · 告别"准确率陷阱"

**准确率（Accuracy）是最危险的指标**——在类别不平衡时会骗人。

> **权威例**：1 万笔交易里只有 10 笔欺诈（0.1%）。写一行 `return false`（全预测正常），准确率 99.9%。系统完全没用。

#### 1.2.1 混淆矩阵（Confusion Matrix）

二分类下 4 个格子：

```
                预测 1（正）     预测 0（负）
真实 1（正）    TP 抓住坏人      FN 漏掉坏人（代价极大！）
真实 0（负）    FP 冤枉好人      TN 正常放行
```

- TP = True Positive = 真正例
- FN = False Negative = 假负例（漏报）
- FP = False Positive = 假正例（误报）
- TN = True Negative = 真负例

**sklearn 规约**：`confusion_matrix(y_true, y_pred)` 返回 `[[TN, FP], [FN, TP]]`。记这个矩阵形状，别靠背。

#### 1.2.2 Precision / Recall / F1

| 指标 | 公式（文字版） | 一句话含义 | 业务场景 |
|---|---|---|---|
| Precision（精确率） | TP / (TP + FP) | 说"是"的里面，真的是的比例 → "抓得准不准" | 垃圾邮件过滤（宁漏不错杀） |
| Recall（召回率） | TP / (TP + FN) | 真是的里面，被抓出来的比例 → "抓得全不全" | 癌症诊断（宁错杀不漏掉） |
| F1 | 2·P·R / (P+R)（调和平均） | Precision 和 Recall 的综合 | 两者都重要 |

**关键认知**：Precision 和 Recall 是**跷跷板**关系——提一个往往压另一个。业务场景决定谁重要。

#### 1.2.3 AUC / ROC（选讲）

- AUC = "把正例排在负例前面的能力"
- 范围 0-1：0.5 随机、1.0 完美
- **权威坑**：`roc_auc_score(y_true, y_score)` 的 `y_score` 必须是**概率或决策值**（`predict_proba(X)[:, 1]`），**不是硬标签**

学生现在不主动问不讲。demo-02 step4 也没用 AUC。

#### 1.2.4 classification_report

sklearn 一个函数输出每个类的 Precision / Recall / F1 + support：

```
             precision    recall  f1-score   support
   恶性(0)       0.98      0.93      0.95        42
   良性(1)       0.96      0.99      0.97        72

    accuracy                           0.96       114
   macro avg     0.97      0.96      0.96       114
weighted avg     0.97      0.96      0.96       114
```

- macro avg = 每类无权均值
- weighted avg = 按 support 加权均值
- 类别不平衡时 macro 和 weighted 会差很多——是否不平衡的"一眼判断"

### 1.3 阈值博弈 · 架构师视角的调节杆

默认 `predict()` 用 **0.5 阈值**（二分类下概率最大者）。但业务可以自定义：

```python
# 保守策略：概率 > 0.9 才判为 1
y_pred_strict = (model.predict_proba(X)[:, 1] > 0.9).astype(int)

# 激进策略：概率 > 0.1 就判为 1
y_pred_loose = (model.predict_proba(X)[:, 1] > 0.1).astype(int)
```

**业务场景映射**：
- 高额风控（宁可漏过不误杀）→ High Precision → 阈值提高
- 医疗筛查（宁可误诊不漏诊）→ High Recall → 阈值降低
- 大促引流（广撒网）→ 阈值降低

**这是工程师最该掌握的一块**——算法不变，阈值调优就能大幅改变业务行为。

### 1.4 面试题档案（备用）

**Q：逻辑回归处理类别特征为什么要先做 One-Hot 编码？**

> 学生翻看原培训班或 01-MLF 旧版 04 时会看到这个问题。备用答案：
>
> 如果把"颜色"编码成 红=1, 蓝=2, 绿=3，模型会错误认为"3 > 1"即"绿色比红色重要"，甚至"红色+蓝色=绿色"（数值位置有序性）。One-Hot 拆成三个独立的 0/1 特征（IsRed, IsBlue, IsGreen），让模型给每个颜色独立的权重。
>
> demo-02 数据全是数值特征，不涉及这个坑。但做决策树（第 4 节）时这就是一个必备技能。

---

## 2. Demo 代码逐步拆解

> 这一节是**直接资料对照**——每个 step 学生可能问什么、权威答案在哪。

### 2.1 step1_data.py · 加载数据、看结构

**代码做的事**：
```python
raw = load_breast_cancer()
df = pd.DataFrame(raw.data, columns=raw.feature_names)
df["Diagnosis"] = raw.target  # 0=恶性, 1=良性
print(df.iloc[:5, list(range(6)) + [-1]])
print(f"{df.shape[0]} 行，{df.shape[1]} 列")
counts = df["Diagnosis"].value_counts()
print(f"良性(1)：{counts[1]} 例")
print(f"恶性(0)：{counts[0]} 例")
```

**Bunch 对象字段**（和 demo-01 一致，分类数据集多了 target_names）：
- `.data` → (569, 30) 特征矩阵
- `.target` → (569,) 标签 0/1
- `.target_names` → ['malignant', 'benign']
- `.feature_names` → 30 个特征名

**数据集元数据（sklearn 官方）**：
- 569 条，30 个特征（肿瘤测量值：半径、纹理、周长、面积等）
- 二分类：0=恶性 malignant 212 例、1=良性 benign 357 例
- 比例 ≈ 1.68:1（大致平衡）

**学生可能问**：
- "每一行代表什么" → 一次肿瘤检测（不是一个病人——同一病人检测两次占两行）
- "Diagnosis 和前 30 列的本质区别" → 前 30 列是模型的输入（X 特征），Diagnosis 是要预测的目标（y 标签）
- "357:212 算平衡吗" → 大致平衡（1.68:1）。失衡通常指 10:1 以上
- "为什么要看分布" → 失衡数据会让"全预测多数类"的傻瓜模型准确率虚高。先看分布 = 先防被 accuracy 骗

### 2.2 step2_split.py · 切 X/y + 切 train/test

**代码做的事**：
```python
X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**和 demo-01 完全一样的结构**。API 统一就是 sklearn 的工程价值——这是该埋的伏笔。

**train_test_split 权威签名**（和 demo-01 一致，详见那份备课 §2.2）：
- `test_size` / `train_size` / `random_state` / `shuffle` / `stratify`

**demo-02 代码没用 stratify**：
- 数据比例 1.68:1 大致平衡，不加也凑活
- 结果：训练集 286:169 ≈ 1.69:1，和全集 1.68:1 几乎一样（大数定律，样本够大随机切分比例自然接近总体）
- 如果是 95:5 的失衡数据，不加 stratify 可能切出"测试集恶性样本只剩 2 条"这种事故

**学生可能问**：
- "训练集 1.69 和全集 1.68 为什么这么接近" → 样本够大（569）随机切分自然接近总体分布。不是参数在起作用
- "stratify 什么时候必须加" → 类别不平衡时、小数据集时、分类任务

### 2.3 step3_train.py · 创建模型 + 训练

**代码做的事**：
```python
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

sample_X = X_test[:5]
sample_y = y_test.values[:5]
predictions = model.predict(sample_X)
probas = model.predict_proba(sample_X)

for i in range(5):
    confidence = max(probas[i]) * 100
    print(f"真实: {label[sample_y[i]]}  预测: {label[predictions[i]]}  置信度: {confidence:.1f}%")
```

**LogisticRegression 权威签名**：
```
LogisticRegression(C=1.0,              # 正则化强度的倒数
                   penalty='l2',
                   solver='lbfgs',      # 默认
                   max_iter=100,        # demo 调大为 10000 防 ConvergenceWarning
                   random_state=None,
                   class_weight=None,   # 类别不平衡可用 'balanced'
                   fit_intercept=True,
                   tol=0.0001)
```

**关键参数深度解析**：

| 参数 | 说明 | 易混淆点 |
|---|---|---|
| `C` | 正则化强度的**倒数**。C 越小正则越强。默认 1.0 | 原培训班讲 `alpha`（α 越大正则越强），两者关系 `C = 1/α` |
| `max_iter` | 最大迭代次数。默认 100 | 数据复杂时默认不够，报 ConvergenceWarning。调大就行 |
| `solver` | 优化器 | 见下表 |
| `penalty` | 正则化类型 | 默认 'l2' |
| `class_weight` | 类别权重 | 不平衡数据用 'balanced' |

**Solver 对比表**（学生问到再讲）：

| Solver | 支持 L1 | 支持多分类 | 适用数据 |
|---|---|---|---|
| lbfgs（默认） | ❌ | ✅ | 中小数据 |
| liblinear | ✅ | ❌ | 小数据、稀疏数据 |
| saga | ✅ | ✅ | 大数据 |
| sag | ❌ | ✅ | 大数据 |

**方法 / 属性清单**：

| 方法 | 返回 | 说明 |
|---|---|---|
| `fit(X, y)` | self | 拟合 |
| `predict(X)` | (n_samples,) | **离散类标签**（0 或 1） |
| `predict_proba(X)` | (n_samples, n_classes) | **每个类的概率，各行加起来=1** |
| `decision_function(X)` | ndarray | 原始决策分值（sigmoid 前） |
| `score(X, y)` | float | Accuracy |

| 属性 | shape | 含义 |
|---|---|---|
| `classes_` | (n_classes,) | **类标签顺序。predict_proba 的列严格按此！** |
| `coef_` | (n_classes, n_features)；二分类时 (1, n_features) | 决策函数系数 |
| `intercept_` | (n_classes,) | 截距 |
| `n_iter_` | ndarray | 实际迭代次数 |

**predict_proba 列顺序（重要 · session-07 教学重点）**：

```python
# classes_ = [0, 1]
# proba shape = (n_samples, 2)
proba = [[0.12, 0.88], ...]
#         ^     ^
#         |     |
#       类别 0   类别 1
#       恶性     良性
```

- 按 `classes_` 排序（类别值升序）
- `classes_` 按数值升序（int [0,1]）或字母序（str ['benign','malignant']）
- **验证方式**：`print(model.classes_)`——这是 API 规约，**直讲不让猜**

**置信度 = 那行里最大的数**：
- `predict()` 内部是 argmax(predict_proba)——哪个概率大选哪个
- 所以置信度 = 选中那个类别的概率 = `max(predict_proba 那一行)`
- 二分类下 argmax 等价于 "> 0.5"；多分类失效，argmax 永远成立
- ⚠️ **坑点**：置信度**不是偏移量**（0.88 - 0.5 = 0.38）——session-07 用户推错过

**学生可能问**（和置信度相关的深度讨论见 [`02-NOTES.md`](./02-NOTES.md)）：
- "列顺序怎么定的" → 按 classes_ 升序，用 `print(model.classes_)` 验证
- "> 0.5 就是 1 吗" → 二分类对，多分类要用 argmax
- "置信度是 0.88 - 0.5 吗" → 不是。置信度就是 0.88 本身
- "predict_proba 为什么加起来是 1" → 因为非此即彼，样本必属其中一类，所有类概率和必为 1
- "置信度等于真实概率吗" → LogisticRegression 自带良好校准，两者约等于。其他模型（RandomForest、SVM）可能偏差大，需要 CalibratedClassifierCV

### 2.4 step4_evaluate.py · 评估分类模型

**代码做的事**：
```python
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"准确率：{acc:.1%}（{int(acc*len(y_test))}/{len(y_test)} 判对）")

print(classification_report(y_test, y_pred, target_names=["恶性(0)", "良性(1)"]))
```

**关键 API**：

| 函数 | 签名 | 返回 |
|---|---|---|
| `accuracy_score(y_true, y_pred)` | 两个 1D 数组 | 0-1 浮点数 |
| `precision_score(y_true, y_pred, average='binary')` | 同上 + average 选项 | 0-1 |
| `recall_score(y_true, y_pred, average='binary')` | 同上 | 0-1 |
| `f1_score(y_true, y_pred, average='binary')` | 同上 | 0-1 |
| `classification_report(y_true, y_pred, target_names=[...])` | 同上 + 类名 | 文本报告（或 dict） |
| `confusion_matrix(y_true, y_pred)` | 两个 1D 数组 | ndarray |
| `roc_auc_score(y_true, y_score)` | y_score 必须是概率 | 0-1 |

**average 参数**（多分类时）：
- `'binary'`（默认）：只算正类（pos_label）
- `'macro'`：各类无权均值
- `'weighted'`：按样本数加权
- `'micro'`：全局 TP/FP 计算

**demo-02 真实表现**：accuracy ≈ 96%（114 条里对 110 左右）。对这个精心整理的教学数据集来说属于正常表现。真实临床应用数据会更乱，指标会低。

**学生可能问**：
- "accuracy 96% 好吗" → 对这个数据集合理。但 accuracy 在失衡场景会骗人，看 classification_report 里的 recall 更稳
- "recall 和 precision 哪个重要" → 场景决定。医疗筛查（漏诊致命）要 recall；垃圾邮件（错杀老板邮件）要 precision
- "能不能画混淆矩阵" → 可以：`from sklearn.metrics import confusion_matrix; confusion_matrix(y_test, y_pred)`。demo-02 代码没画，要画就补一行
- "为什么这个 demo 只用了 accuracy 没用 F1" → 教学简化。真实项目评估永远不只看一个指标
- "什么时候该用 AUC" → 需要跨阈值评估、或者业务要调优阈值时

### 2.5 NOTES.md · 置信度专题

`demo-02-breast-cancer/NOTES.md` 是 session-07 产出的独立专题文档，核心三句：
1. 分类有 `predict_proba`，回归没有
2. 回归想要置信度要自己造（bootstrap / quantile / conformal，只提名字不展开）
3. 分类 = SELECT COUNT(*)（开箱即用），回归 = "数据新鲜度"（自己造）

**文档状态**：
- 已简化版（砍掉 sigmoid / 校准 / Agent 应用等超前内容）
- 保留在 demo-02 目录里作为扩展材料
- 学生追问置信度深度问题时可以引用

---

## 3. 知识点清单（按层，资料必须全）

### 3.1 概念层（苏格拉底为主）

| 知识点 | 已通关？ | 讲法建议 |
|---|---|---|
| 模型是函数 | 3（session-01） | 不重讲 |
| 分类任务输出离散类别 | 2（session-04） | step3 确认 |
| 二分类 / 正例 / 负例 | 2（session-04） | step1 用"良性/恶性"带出 |
| 类别不平衡 | 2（session-04） | step1 `value_counts` 时确认"为什么要看分布" |
| 过拟合 / 泛化 | 2（session-07） | step4 评估时 spot-check |
| 置信度 = max(predict_proba) | 2（session-07） | 已通关，不重讲 |
| Sigmoid 外套心智 | 待建立 | step3 直讲（§1.1） |
| Precision vs Recall 跷跷板 | 待建立 | step4 讲 classification_report 时带出 |
| 阈值博弈 | 待建立 | step4 讲完指标后引入 |
| 类别不平衡时 accuracy 骗人 | 部分通关（session-04 类别不平衡已 L2） | step4 用 1 万交易 10 欺诈例子巩固 |

### 3.2 API 层（直讲）

- `load_breast_cancer()` → Bunch（含 target_names）
- `LogisticRegression(C=, max_iter=, solver=, class_weight=)` + `.fit` + `.predict` + `.predict_proba` + `.classes_` + `.coef_` + `.intercept_`
- `train_test_split(..., stratify=y)` 分类场景建议加
- `accuracy_score` / `precision_score` / `recall_score` / `f1_score` / `classification_report` / `confusion_matrix` / `roc_auc_score`
- `StandardScaler` + `Pipeline`（进阶，demo-02 没用但真实项目要用）

### 3.3 规约层（直讲）

- `predict_proba` shape = `(n_samples, n_classes)`，每行加起来=1
- 列顺序按 `classes_` 升序排
- `classes_` 验证方式：`print(model.classes_)`
- `predict` = argmax(predict_proba)
- 二分类下 argmax ≡ "概率 > 0.5"；多分类只有 argmax 成立
- `confusion_matrix` 返回 `[[TN, FP], [FN, TP]]`
- `roc_auc_score` 的 y_score 必须是概率不是硬标签
- `C` = 1/α（和 Ridge/Lasso 的 alpha 相反关系）
- 大写 X 小写 y 的 ML 惯例（同 demo-01）

### 3.4 工程层（L3 目标）

- 阈值博弈（见 §1.3）—— **用户护城河之一**
- 类别不平衡工程手段：重采样 / `class_weight='balanced'` / stratify 切分
- 为什么建议先 StandardScaler：逻辑回归梯度下降求解器对尺度敏感（尤其 sag/saga）
- LogisticRegression 的部署形态和线性回归一样：coef_ + intercept_ + classes_ + feature schema
- 跨语言落地：ONNX 支持、PMML 在金融行业常见
- Agent 工程里的应用：意图分类、路由、审核——LR 是最便宜的选项

### 3.5 延展层（学生问到再讲）

| 话题 | 下限答案 | 权威出处 |
|---|---|---|
| Sigmoid 是什么 | "把任何数字压到 0-1 的函数" | 不展开 |
| Log-Loss 损失函数 | 不主动提 | 学生不问就不讲 |
| MLE 极大似然 | 不主动提 | 原培训班深讲，仓库不覆盖 |
| ROC 曲线怎么画 | "不同阈值扫一遍，画出 FPR-TPR 曲线" | 学生要手撕再展开 |
| AUC 深讲 | "曲线下面积，0.5 随机，1.0 完美" | 足够 |
| 校准（Calibration） | "LogisticRegression 自带良好校准，RF/SVM 才需要 CalibratedClassifierCV" | sklearn calibration 官方 |
| 多分类（softmax） | "逻辑回归也能做多分类，sklearn 自动用 one-vs-rest" | 不展开 |
| One-Hot 编码 | 见 §1.4 面试题 | 决策树那节会碰 |
| 精确率 vs 准确率 | Accuracy = 总对率，Precision = 预测为正的命中率 | 必须讲清，易混淆 |

---

## 4. 教学节奏建议

### 4.1 四个 step 的推进建议

| step | 建议时长 | 苏格拉底 or 直讲 | 关键动作 |
|---|---|---|---|
| step1 | 10-15 min | 直讲（Bunch 字段）+ 苏格拉底（一行 = 什么，分布平衡吗） | 带出 "先看分布防 accuracy 陷阱" |
| step2 | 5-10 min | 直讲（stratify）+ 苏格拉底（为什么切） | 如果 demo-01 step2 已讲透，这里秒过 |
| step3 | 20-30 min | 直讲（classes_、predict_proba 规约）+ 苏格拉底（为什么多了置信度） | 本节最重的一块。带出 "sigmoid 外套" 心智。引向 NOTES.md 的置信度讨论 |
| step4 | 20-30 min | 直讲（accuracy/precision/recall 公式+含义）+ 苏格拉底（哪个指标在医疗场景下更重要） | 压轴：阈值博弈（§1.3）。带出"调阈值=调业务策略"的工程直觉 |

### 4.2 和 demo-01 的衔接讲法

跑完 step2 时让学生观察：demo-02 代码结构和 demo-01 几乎一模一样，只有 step3 的算法构造函数不同。这是 sklearn 统一 API 的体感——该埋的伏笔。

跑完整个 demo-02 时：让他说"回归和分类的三个区别"（输出类型、评估指标、有无置信度）。这是 C 组算法适配表的第一刀，从手感来而不是从背表来。

---

## 5. 检测方案

> 评分标准：0 没听过 / 1 听过讲不清 / 2 能一句话讲清 / 3 能用工程类比向同事解释 + 知道场景

### 5.1 Level 2 检测题（demo-02 跑完该过）

- "demo-02 和 demo-01 的 ML 流程有哪里不一样" → 算法、y 类型、评估指标、多了 predict_proba
- "为什么 accuracy 在失衡数据上不靠谱" → 全预测多数类就能虚高
- "predict_proba 返回什么" → (n_samples, n_classes) 的概率矩阵
- "怎么知道第一列是哪个类" → 看 model.classes_
- "Precision 和 Recall 区别" → 一个是"抓得准"，一个是"抓得全"
- "置信度怎么算出来" → predict_proba 那一行里最大的那个数

### 5.2 Level 3 检测题（逻辑回归这节课的终点）

选型判断：
- "给你一个任务：预测用户下周会不会流失" → Logistic，二分类
- "预测用户下周登录次数" → Linear，连续
- "垃圾邮件识别怎么选指标" → Precision（宁可漏过不错杀老板邮件）
- "癌症筛查怎么选指标" → Recall（宁可误诊不漏诊）
- "风控高额提现怎么选阈值" → 提高（保守 high precision）
- "大促优惠券怎么选阈值" → 降低（激进 high recall）

架构类：
- "LogisticRegression 训练完，线上 Java 服务要存什么" → coef_ + intercept_ + classes_ + feature schema
- "怎么让线上的决策阈值可调而不重训模型" → 不调用 model.predict()，自己用 predict_proba 加阈值逻辑
- "类别不平衡场景的 3 种处理手段" → 重采样 / class_weight / 换指标
- "为什么 LogisticRegression 的 predict_proba 可以直接当概率用" → 它是概率模型，自带良好校准

---

## 6. 逻辑回归章特有的术语易混点

- `C` vs `alpha`：C 是正则强度的倒数（C 越小正则越强），α 是正则强度正比（α 越大正则越强）。两者互为 1/x 关系。原培训班讲的是 α，sklearn 里用的是 C
- accuracy vs precision：accuracy 是总对率（判对 / 总数），precision 是"说是的里面真是的比例"（TP / (TP+FP)）。中文里"准确率/精确率"容易互换混用
- 正例 / 反例 / 正类 / 负类：不同文献术语混用，同义；用时明确指出 1 和 0 哪个是正例即可
- predict_proba 列顺序不是硬码的：永远用 `model.classes_` 验证

---

## 7. 附录

### 7.1 数据集对照表

| 维度 | 仓库 demo-02（sklearn 内置） | 原培训班 04 章（Wisconsin CSV） |
|---|---|---|
| 形状 | (569, 30) | (699, 11) |
| sklearn API | `load_breast_cancer()` | `pd.read_csv('breast-cancer-wisconsin.csv')` |
| 加载方式 | 直接内置，离线可用 | 需要下载 CSV 或从 UCI 在线拉 |
| 缺失值 | 已清洗，无缺失 | 16 个 "?" 缺失值，需 `data.replace('?', np.nan).dropna()` |
| target | 0=恶性 / 1=良性 | 2=良性 / 4=恶性（值不同！） |
| 原因 | 入门教学简化 | 原始数据集，保留训练数据处理步骤 |

学生如果翻原培训班看到 `data.Class` 且值是 2/4——直接告诉他是同一数据集的原始版本，sklearn 版已做清洗和标签重映射。

### 7.2 sklearn 官方文档链接

- LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- CalibratedClassifierCV: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
- load_breast_cancer: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
- accuracy_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
- precision_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
- recall_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
- f1_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
- classification_report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
- roc_auc_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
- confusion_matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
- Calibration 用户指南: https://scikit-learn.org/stable/modules/calibration.html
- Cross-validation 用户指南: https://scikit-learn.org/stable/modules/cross_validation.html

### 7.3 项目内相关文档

- 上节课：[`demos/demo-01-house-price/LESSON-PLAN.md`](../demo-01-house-price/LESSON-PLAN.md)
- 置信度专题：[`02-NOTES.md`](./02-NOTES.md)（session-07 产出）
- 阶段一 LESSON-PLAN（学习方法总纲 + 开发者视角 · 合并自原 Orientation + Overview）：[`01-LESSON-PLAN.md`](../00-mental-model/01-LESSON-PLAN.md)
- demos 总清单：[`demos/README.md`](../README.md)

**已删除的讲义**：
- `01-Machine-Learning-Foundation/04-Logistic-Regression-and-Classification-Metrics.md` 的内容已迁入本文 §1 和 §2
- 原培训班对照：`assets/source-materials/第4阶段-机器学习/逻辑回归.md`（gitignored，本地可访问）

### 7.4 相关 session 日志

- session-01 · 2026-04-07 · "模型是函数" 心智建立（回归/分类第一次碰）
- session-04 · 2026-04-13 · 分类 vs 回归 Level 2、类别不平衡 Level 2
- session-06 · 2026-04-14 · demo-02 脚本创建（未跑）
- session-07 · 2026-04-19 · demo-02 step 1/2/3 跑通，置信度 Level 2，NOTES.md 写好。本备课在此会话产出

### 7.5 迁移记录

本文档从以下源迁移/整合：

1. 讲义 `01-Machine-Learning-Foundation/04-Logistic-Regression-and-Classification-Metrics.md`（已删除）
   - §1 Sigmoid 外套 → 本文 §1.1（去公式）
   - §2 评估指标 → 本文 §1.2（混淆矩阵 / P/R/F1 / AUC / classification_report）
   - §3 实战代码 → 融入本文 §2（按 demo-02 step 重组）
   - §4 阈值博弈 → 本文 §1.3
   - §5 面试题 → 本文 §1.4

2. `demos/LESSON-PLAN.md`（双 demo 合并版，已删）
   - 分类相关部分全部迁入本文

3. `demos/demo-02-breast-cancer/NOTES.md`
   - 保留不动，作为置信度专题扩展材料

4. sklearn 官方文档
   - API 签名、参数含义、形状规约 → 本文 §2 和 §3
   - Calibration 权威定义 → 本文 §3.5 延展层

5. 原培训班 `assets/source-materials/第4阶段-机器学习/逻辑回归.md`
   - 用作对照（§7.1）——内容不迁入（有数学推导）
