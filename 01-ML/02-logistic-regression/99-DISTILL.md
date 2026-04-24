## 99 · 逻辑回归 · 学习蒸馏

> **这份文档是什么**：学完逻辑回归（demo-02 乳腺癌二分类）后沉淀下来的算法要点。
>
> **前置依赖**：
> - [`../00-mental-model/99-DISTILL.md`](../00-mental-model/99-DISTILL.md)（心智模型蒸馏）—— 特征/标签、监督学习、训练/测试/推理
> - [`../01-linear-regression/99-DISTILL.md`](../01-linear-regression/99-DISTILL.md)（线性回归学习蒸馏）—— 加权求和心智、"线性"和"回归"两个字的含义与历史
>
> 本文档只讲逻辑回归**独有**的部分，不重复前置。
>
> **阅读视角**：第三人称客观陈述。踩过的坑升维成「⚠️ 陷阱」旁注。
>
> **脚注说明**：`[^n]` 鼠标悬浮预览外网权威源，点击跳转文末。学习轨迹另列「学习轨迹」小节。

---

### 零、本章位置

| 维度 | 值 |
|---|---|
| 算法 | 逻辑回归（Logistic Regression）[^1] |
| 任务类型 | **二分类**（输出：类别 + 概率） |
| 在 10 节课里 | 第 2 节 —— 分类任务原型 |
| 和上一节的对比 | 线性回归输出连续数字；逻辑回归输出"是/否 + 置信度" |
| 为什么排第二 | API 和线性回归完全一样（`.fit` / `.predict`），只换算法类名 —— 把 "**回归 vs 分类**" 这把刀切透 |

---

### 一、做什么：输出一个"是/否"的概率

**任务类型**：**二分类**。给一条记录，输出"属于哪一类" + "多大把握"。

**典型业务**：

| 业务 | 正例（1） | 负例（0） |
|---|---|---|
| 垃圾邮件识别 | 垃圾 | 正常 |
| 信用卡反欺诈 | 欺诈 | 正常 |
| 病理诊断 | 恶性 | 良性 |
| 信贷评分卡 | 违约 | 不违约 |
| 用户流失预测 | 会流失 | 不会流失 |

**本章用的数据**：sklearn 自带的**乳腺癌数据集**，569 条肿瘤检测记录，30 个特征（半径、纹理、周长等），标签是 0（恶性）/ 1（良性）。

**和线性回归的直接对比**：

|  | 线性回归 | 逻辑回归 |
|---|---|---|
| 输出 | 任意连续数字（`-∞ ~ +∞`） | 概率（`0 ~ 1`）+ 类别 |
| `.predict()` 返回 | 浮点数（房价 2.35） | 类别（0 或 1） |
| 特有 API | — | `.predict_proba()` 返回每类概率 |
| 评估指标 | MAE / MSE / R² | Accuracy / Precision / Recall / F1 |

---

### 二、内部长什么样：线性回归 + 压到 (0, 1) + 阈值判断

逻辑回归**唯一要记住的心智**，三步：

```
第 1 步：和线性回归一模一样，算出一个任意实数    z = w·x + b
第 2 步：把 z 通过一个"压缩函数"压到 (0, 1)     p = f(z)
第 3 步：把 p 当作"类别 1 的概率"                p > 0.5 → 判 1
```

> 📎 **第 2 步的"压缩函数"叫什么**：**Sigmoid**[^3]（S 型函数）。本文不展开它的数学形态——只记"它把任意实数压到 (0, 1)"这一句就够了。想进一步了解，点脚注跳 Wikipedia。
>
> 入门阶段记住"干这个事"的函数存在即可。它在神经网络里的正式名叫 **激活函数（activation function）**，在统计 GLM 里叫 **链接函数（link function）**——以后学到再正式认识。

#### 数据流

```
    特征 x₁...x₃₀                  线性组合                压缩函数                类别决策
  ┌────────────┐            ┌─────────────────┐       ┌──────────┐         ┌────────────┐
  │ radius     │            │ z = w₁·x₁ + ... │       │          │         │ p > 0.5 ?  │
  │ texture    │ ───────►   │    + w₃₀·x₃₀    │ ──z──►│ Sigmoid  │ ──p───►│  是 → 1    │
  │ ...        │            │    + b          │       │ 压到(0,1)│         │  否 → 0    │
  │ 30 列      │            └─────────────────┘       └──────────┘         └────────────┘
  └────────────┘              任意实数                  (0, 1) 概率         类别 + 置信度

  ↑ 和线性回归一样           ↑ 和线性回归完全相同         ↑ 新加的"压缩"         ↑ 新加的"阈值判断"
```

**一句话记住逻辑回归**：**线性回归 + 压到 (0, 1) + 阈值判断**。算法的增量只在后两步。

#### 为什么不直接用线性回归 + 0.5 阈值做分类

技术上可以，但会踩几个坑：
- 线性回归对离群点敏感，一个极端值能把决策边界拉歪
- 输出不是概率（可能是 -3.2、5.8），不能当置信度用
- 损失函数（MSE）用在分类上数学性质不好，优化容易陷局部最优

加一层压缩正是为了解决这些问题——把任意实数规范化成 (0, 1) 的概率值。

#### 术语映射

| 符号 | 名字 | 角色 |
|---|---|---|
| `w` | 权重 | 和线性回归一样——每个特征的话语权 |
| `b` | 偏置 | 和线性回归一样——起步价 |
| `z` | 决策值（decision function） | 线性组合的原始输出 |
| `p` | 概率 | 压缩后的类别 1 概率 |
| `0.5` | 阈值（threshold） | 判决分界线（可调） |

> 📎 **关于"逻辑回归"这四个字本身怎么拆**（Logistic ≠ logic；Verhulst 1838 / Berkson 1944 历史；名字误导的判断准则）—— 见文末 **QA Q0**。

---

### 三、评估：分类指标家族

分类没有"平均误差"的概念（预测"恶性"对不对是非题），评估看的是**4 种结果的分布**。

#### 混淆矩阵 (Confusion Matrix)

所有分类评估的**起点**[^5]。二分类下是一张 2×2 的表：

```
                   预测 1 (正例)       预测 0 (负例)
              ┌──────────────────┬──────────────────┐
  真实 1 (正) │  TP 真正例        │  FN 假负例 🚨    │
              │ （抓住坏人）       │ （漏掉坏人）      │
              ├──────────────────┼──────────────────┤
  真实 0 (负) │  FP 假正例        │  TN 真负例        │
              │ （冤枉好人）       │ （正常放行）      │
              └──────────────────┴──────────────────┘
```

- **TP** (True Positive) — 预测正例，真的是正例（抓对了）
- **FN** (False Negative) — 预测负例，实际是正例（**漏报**）
- **FP** (False Positive) — 预测正例，实际是负例（**误报**）
- **TN** (True Negative) — 预测负例，真的是负例（放行对了）

**sklearn 规约**：`confusion_matrix(y_true, y_pred)` 返回 `[[TN, FP], [FN, TP]]`。

#### Accuracy / Precision / Recall / F1

| 指标 | 公式 | 一句话 | 业务锚 |
|---|---|---|---|
| **Accuracy** 准确率 | `(TP+TN) / 总数` | 判对的占多少 | 考试总分 |
| **Precision** 精确率[^6] | `TP / (TP+FP)` | 说"是"的里面，真是的比例 | 命中率 |
| **Recall** 召回率[^6] | `TP / (TP+FN)` | 真是的里面，抓出来的比例 | 抓捕率 |
| **F1 分数**[^7] | `2·P·R / (P+R)` | Precision 和 Recall 的调和平均 | 综合分 |

#### 💡 F1 反复遗忘？用"综合分"锚住它

这个概念在学习过程中**反复遗忘了 3 次**——用业务锚强化一次：

**F1 = 学生综合分**。单看语文 100 分（高 Precision）或数学 100 分（高 Recall）都不够，要两科都不差才算"综合优秀"。F1 专治"偏科"——只要有一个指标低，F1 就会被拉下来。

**为什么是调和平均（harmonic mean）不是算术平均**？

```
算术平均：  (P + R) / 2     ← 一个 0 一个 1 得 0.5（看起来不差）
调和平均：  2·P·R / (P + R) ← 一个 0 一个 1 得 0     （彻底判死）
```

**调和平均对极端值更敏感**——只要 P 或 R 有一个拉垮，F1 必垮。这是"综合分"最想要的性质：**不让偏科蒙混过关**。

**什么时候看 F1**：不确定业务更在意 Precision 还是 Recall、两者都不能太差时。medical screening 盯 Recall，spam filter 盯 Precision，**普通业务默认看 F1**。

#### ⚠️ 陷阱：准确率（Accuracy）是最危险的指标

**经典反例**：1 万笔交易里只有 10 笔欺诈（0.1%）。写一行 `return 正常`（全预测负例），**准确率 99.9%**——但系统完全没用，10 笔欺诈一个没抓到。

这就是**类别不平衡陷阱**。任何分类任务必须先看数据分布：

```python
df["Diagnosis"].value_counts()
# 良性 357
# 恶性 212
# 比例 1.68:1 ← 大致平衡，Accuracy 还能用
# 比例 100:1 或更悬殊 ← Accuracy 必骗人
```

#### 📊 Precision vs Recall 的跷跷板

**Precision 和 Recall 是此消彼长的关系**[^6]。画成跷跷板：

```
  阈值提高（模型变保守，只预测非常肯定的正例）
  ────────────────────────────────────────────►
                                                 
   Precision ↑↑↑                Recall ↓↓↓      
   (抓得准)                     (漏得多)        
                                                 
  ◄────────────────────────────────────────────
  阈值降低（模型变激进，宁可错杀不放过）
                                                 
   Precision ↓↓↓                Recall ↑↑↑      
   (冤得多)                     (抓得全)        
```

**业务锚对照**：

| 场景 | 漏过代价 | 错杀代价 | 盯谁 | 阈值方向 |
|---|---|---|---|---|
| 癌症筛查 | 极高（死人） | 低（复查） | **Recall** | 降低 |
| 垃圾邮件过滤 | 低（删一封） | 高（老板邮件丢） | **Precision** | 提高 |
| 反欺诈 | 高（赔钱） | 中（好客户被冻） | Recall 略重 | 略降 |
| 高额贷款风控 | 高（坏账几十万） | 中（好客户没贷到） | **Precision** | 提高 |
| 两者都在意 | — | — | **F1** | 平衡 |

#### 💡 aha moment · 漏诊 vs 误诊是钥匙

选 Precision 还是 Recall，别背表，按这个路径推：

```
  这类业务里，犯哪种错更可怕？
            │
    ┌───────┴───────┐
    ↓               ↓
 漏过（FN）        错杀（FP）
 更可怕            更可怕
    │               │
    ↓               ↓
  盯 Recall      盯 Precision
  （降阈值）      （提阈值）
```

**医疗场景的具体推理路径**：

```
Q: recall 恶性 0.91、良性 0.99——模型在哪类上更弱？
↓
（第一反应）"不知道，我没这个概念"
↓
（回退到业务锚）"漏诊 vs 误诊，哪个更可怕？"
↓
"漏诊"（真有癌症没检出 → 病人延误治疗可能死）
↓
漏诊 = FN = 真是正例但被预测为负例
↓
Recall = TP / (TP + FN) —— 分母包含 FN
↓
漏诊多 → FN 多 → Recall 低
↓
所以医院盯 recall，越高越好
↓
回头看：recall 恶性 0.91 < 良性 0.99 → 模型在恶性上更弱 ✓
```

**记忆锚**：**漏诊 = FN，FN 在 Recall 的分母，所以医疗场景死盯 Recall**。

#### ⚠️ 陷阱：Precision ≠ Accuracy，中文名称容易混

中文"**准确率**"和"**精确率**"只差一个字，英文是完全不同的指标：

```
Accuracy  = (TP+TN) / 全部         ← 分母是"总数"
Precision = TP / (TP+FP)           ← 分母是"模型说是的总数"
Recall    = TP / (TP+FN)           ← 分母是"真是的总数"
```

**三个指标的分母完全不同**。记忆锚：**分母定义了指标的意义**。

#### classification_report：一行看全

```
                precision   recall   f1-score   support
   恶性(0)         0.98      0.93       0.95        42
   良性(1)         0.96      0.99       0.97        72

   accuracy                             0.96       114
   macro avg       0.97      0.96       0.96       114
   weighted avg    0.97      0.96       0.96       114
```

**看点**：
- 每个类分别看 P/R/F1（不平衡时单一数字会骗人）
- `macro avg`（每类无权平均）vs `weighted avg`（按样本数加权）差很多 = 类别不平衡的信号
- `support` 是该类在测试集中的真实数量

#### 💡 卡点实录 · "43 是哪来的"

demo-02 跑完，`classification_report` 配合 `confusion_matrix` 一起看才能读懂：

```
                precision   recall   f1-score   support
   恶性(0)         0.98      0.93       0.95        42
   良性(1)         0.96      0.99       0.97        72
                                       ─────       ────
   accuracy                             0.96       114
   macro avg       0.97      0.96       0.96       114
   weighted avg    0.97      0.96       0.96       114

                  预测恶性(0)    预测良性(1)
   真实恶性(0)         39            3     ← 42 = 39 + 3 （support 恶性）
   真实良性(1)          1           71     ← 72 = 1 + 71  （support 良性）
                       ────          ────
                        40           74    ← 这两列也有讲法
```

**关键对应关系**（回答"43 是哪来的"这类数字溯源）：

| 数字 | 从哪算 | 代表什么 |
|---|---|---|
| `support 恶性 = 42` | 真实恶性行的和（39+3） | 测试集中真实恶性样本数 |
| `support 良性 = 72` | 真实良性行的和（1+71） | 测试集中真实良性样本数 |
| `114`（accuracy 的 support） | 总样本数 | 测试集大小 |
| `预测恶性 = 40`（40 = 39+1） | 预测恶性列的和 | 模型说"恶性"的总数 |
| `Recall 恶性 = 0.93` | `39 / 42` | TP / (TP+FN) —— 真恶性里抓到的比例 |
| `Precision 恶性 = 0.98` | `39 / 40` | TP / (TP+FP) —— 说恶性里真恶性的比例 |

**记忆锚 · 三个指标的分母完全不同**：

| 指标 | 分母 | 具体数字（demo-02） | 分母的意义 |
|---|---|---|---|
| **Accuracy** | 全部样本 | 114 | 总判对率 |
| **Precision** | 模型说"正"的 | 40（预测恶性列的和） | 命中率 |
| **Recall** | 真是"正"的 | 42（真实恶性行的和） | 抓捕率 |

**一句话**：**Accuracy 看全部，Precision 看"预测列"，Recall 看"真实行"**。分母不同 → 分母定义了指标的意义。

#### 💡 macro avg vs weighted avg：一眼判断类别平衡

看上表两行：`macro avg = 0.96` ≈ `weighted avg = 0.96` → 两类大致平衡。

**如果两者差很多**（例如 macro=0.60 / weighted=0.88）：
- weighted 被多数类拉高 → 少数类表现差
- **这就是类别不平衡的信号** → 必须看混淆矩阵找少数类的 FN/FP

---

### 四、置信度：分类独有的宝藏

这是 `predict_proba` 带来的**分类独有能力**——每次预测都自带"多大把握"的数字。

#### predict_proba 返回什么

```python
probas = model.predict_proba(X_test[:3])
#
# [[0.12, 0.88],   ← 样本 1：12% 恶性，88% 良性 → predict 返回 1
#  [0.95, 0.05],   ← 样本 2：95% 恶性，5% 良性  → predict 返回 0
#  [0.51, 0.49]]   ← 样本 3：51% 恶性，49% 良性 → predict 返回 0（差一点就翻）
#
# shape = (n_samples, n_classes)
# 每行加起来 = 1（非此即彼）
# 列顺序按 model.classes_ 升序
```

**置信度 = `max(那行)`**——选中类别的那个概率。

#### ⚠️ 陷阱：置信度**不是**偏移量（学习过程中真实出现过的错误推理）

**错误的直觉推理**（学习过程中真实出现过）：

> "0.5 是两类摇摆的中点，所以概率是 0.88 表示**比中点多出 0.38**——这 0.38 应该就是置信度？"

看起来合理，但**错了**。

```
❌ 错推理：置信度 = |0.88 - 0.5| = 0.38
✅ 正解：  置信度 = 0.88
```

**为什么错**：置信度的定义是"选中类别的概率"，不是"偏离阈值的距离"。阈值是决策边界（可调），概率是模型输出（固定）——两者是独立的。

**正确心智**：

| proba | predict | 置信度 | 业务解读 |
|---|---|---|---|
| `[0.12, 0.88]` | 1 | **0.88** | 88% 把握是良性 |
| `[0.95, 0.05]` | 0 | **0.95** | 95% 把握是恶性（非常肯定）|
| `[0.51, 0.49]` | 0 | **0.51** | 51% 把握是恶性（勉强过线）|
| `[0.50, 0.50]` | 0 或 1 | **0.50** | 完全不知道 |

**验证办法**：置信度永远 ≥ 0.5（二分类下）——如果你算出来小于 0.5 说明推错了。

#### ⚠️ 陷阱：predict_proba 列顺序不能猜

列顺序按 `model.classes_` 升序排，**必须用 `print(model.classes_)` 验证**——不要假设"第一列就是类别 0"。

```python
print(model.classes_)  # → [0 1]   意味着第 0 列 = 恶性，第 1 列 = 良性
```

#### 💡 分类 vs 回归的置信度差别

| | 分类 | 回归 |
|---|---|---|
| 置信度 API | `predict_proba()` **开箱即用** | 没有内置 API |
| 想要的话 | — | 自己造（bootstrap / quantile regression / conformal prediction）|
| 类比 | `SELECT COUNT(*)` 直接查 | "数据新鲜度"要自己存时间戳 |

**工程意义**：做 Agent 系统时，**分类式节点**（意图识别/路由/审核）自带置信度可用于"不确定时兜底转人工"；**回归式节点**（预测数量/延迟/金额）需要自己设计区间估计。

---

### 五、阈值博弈：工程师的调节杆

默认 `predict()` 用 **0.5 阈值**做决策。但阈值**不是固定的**——工程上可以自定义：

```python
# 保守策略（高 Precision）：概率 > 0.9 才判正例
y_pred_strict = (model.predict_proba(X)[:, 1] > 0.9).astype(int)

# 激进策略（高 Recall）：概率 > 0.1 就判正例
y_pred_loose  = (model.predict_proba(X)[:, 1] > 0.1).astype(int)
```

#### 📊 阈值如何影响业务

```
  阈值 = 0.1             阈值 = 0.5             阈值 = 0.9
  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
  │宁错杀不放过   │      │默认平衡       │      │宁放过不错杀   │
  │              │      │              │      │              │
  │ Recall  ↑↑↑  │      │ Recall  ●●●  │      │ Recall  ↓↓↓  │
  │ Precision ↓↓ │      │ Precision ●● │      │ Precision ↑↑ │
  │              │      │              │      │              │
  │ 医疗筛查 →   │      │ 默认起点     │      │ ← 高额风控   │
  │ 反欺诈  →    │      │              │      │ ← 垃圾邮件   │
  │ 大促引流 →   │      │              │      │              │
  └──────────────┘      └──────────────┘      └──────────────┘
```

#### 💡 架构师视角：阈值可调 ≠ 重训模型

**工程价值**：**阈值改了不需要重训模型**。线上 Java/Go 服务只要改一个浮点数，业务策略就变。这意味着：

- A/B 测试不同阈值 → 不需要多训模型
- 业务方想调"严/松" → 调个参数就行
- 动态阈值（按用户分层、时间段）→ 服务层的事，不打扰算法团队

**最佳实践**：线上服务不直接调 `model.predict()`，而是用 `predict_proba() + 可配置阈值`——算法层和业务层解耦。

---

### 六、实操代码 · 4 步跑通分类模型

讲到这里所有心智都齐了。看看 demo-02 实际怎么跑——整个 pipeline 约 20 行核心代码（比回归多几行评估代码）：

```python
# ─────────────────────────────────────────
# Step 1 · 加载数据
# ─────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
import pandas as pd

raw = load_breast_cancer()
df = pd.DataFrame(raw.data, columns=raw.feature_names)
df["Diagnosis"] = raw.target                # 30 特征 + 1 标签
# df.shape = (569, 31)
# df["Diagnosis"].value_counts() → 良性 357 / 恶性 212 （1.68:1 大致平衡）

# ─────────────────────────────────────────
# Step 2 · 切分 X/y 和 train/test
# ─────────────────────────────────────────
from sklearn.model_selection import train_test_split

X = df.drop("Diagnosis", axis=1)            # 30 个特征
y = df["Diagnosis"]                         # 0=恶性, 1=良性
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 严重不平衡时加 stratify=y（见七节 stratify 陷阱）

# ─────────────────────────────────────────
# Step 3 · 创建模型 · 训练
# ─────────────────────────────────────────
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=10000)  # ← 调大防 ConvergenceWarning
model.fit(X_train, y_train)                 # ← 核心一行：训练循环在这

# 训练完模型内部：
# model.coef_       → shape (1, 30)        30 个权重
# model.intercept_  → shape (1,)           偏置 b
# model.classes_    → [0, 1]               类顺序（predict_proba 列的权威出口）

# ─────────────────────────────────────────
# Step 4 · 预测 + 评估（分类比回归多）
# ─────────────────────────────────────────
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)              # 类别 shape (114,)
probas = model.predict_proba(X_test)        # 概率 shape (114, 2)

# 评估 · 三板斧
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.1%}")               # ≈ 96%

print(classification_report(y_test, y_pred, target_names=["恶性", "良性"]))
# 每类分别看 P/R/F1 + support

print(confusion_matrix(y_test, y_pred))     # [[39, 3], [1, 71]]
# [[TN, FP], [FN, TP]]

# 置信度（分类独有）
confidence = probas.max(axis=1)             # 每条样本的置信度 shape (114,)
```

#### 💡 和 demo-01（线性回归）的骨架对比

```
  Step            demo-01 (回归)                      demo-02 (分类)
  ────────────────────────────────────────────────────────────────────────
  1 · 加载        fetch_california_housing()          load_breast_cancer()
  2 · 切分        train_test_split(...)               train_test_split(...)           ← 一字不改
  3 · 训练        LinearRegression()                  LogisticRegression(max_iter=10000)
  4 · 预测        model.predict() 只给数字            predict() + predict_proba()    ← 分类多一个
  4 · 评估        mean_absolute_error                 accuracy + classification_report
                                                      + confusion_matrix              ← 分类多两个
```

**核心变化只有三处**：

1. **Step 1** 换数据集加载函数（业务不同）
2. **Step 3** 换算法类名 + 可能调超参（算法不同）
3. **Step 4** 换评估指标家族（回归 vs 分类任务类型不同）

**其他代码一字不改**——这就是上一章讲的 "换算法 = 换一行" 的 sklearn 统一 API 工程价值。分类和回归的 4 步骨架在**代码结构层完全一致**，只是在三个点插入了任务类型相关的 API。

#### 📍 代码每一步对应前面哪节

| Step | 核心 API | 在本 DISTILL 哪里讲过 |
|---|---|---|
| 1 · 加载 | `load_breast_cancer()` | 一节（数据集元信息） |
| 1 · 分布 | `value_counts()` | 三节 ⚠️ Accuracy 陷阱（先看分布防失衡） |
| 2 · 切分 | `train_test_split()` | 七节 🪤 stratify 陷阱 |
| 3 · 训练 | `LogisticRegression(max_iter=10000).fit()` | 二节（压到 (0,1) + 阈值判断）+ 七节（max_iter 卡点） |
| 3 · 检查 | `coef_` / `intercept_` / `classes_` | 二节 + 四节（classes_ 验证） |
| 4 · 预测类别 | `model.predict()` | 四节（`argmax(proba)`） |
| 4 · 预测概率 | `model.predict_proba()` | 四节（置信度的来源） |
| 4 · 置信度 | `probas.max(axis=1)` | 四节 ⚠️ 不是偏移量 |
| 4 · 评估 | `accuracy_score` / `classification_report` / `confusion_matrix` | 三节（分类指标家族 + "43 哪来的"溯源） |
| 4 · 阈值调 | `(probas[:, 1] > threshold)` | 五节（阈值博弈） |

**闭卷复习提示**：看这张表如果能**反推 API 对应的心智**（从动作名推回概念），说明已经打通骨架 → 心智的双向映射。做到这里 demo-02 就到 **L3**。

---

### 七、工程落地

#### 模型文件里存什么

```
{
  "coef_":              [[w1, w2, ..., w30]],      // (1, 30)
  "intercept_":         [b],                        // (1,)
  "classes_":           [0, 1],                     // 类顺序的权威出口
  "feature_names_in_":  ["radius", "texture", ...], // 列顺序
  "n_features_in_":     30
}
```

和线性回归几乎一样，只多了 `classes_` 字段（分类特有，标记类别顺序）。

#### 关键超参（sklearn LogisticRegression）[^4]

| 参数 | 默认 | 作用 | 坑 |
|---|---|---|---|
| `C` | `1.0` | 正则化强度的**倒数**（C 越小正则越强） | ⚠️ 和原培训班的 `α` 相反：`C = 1/α` |
| `max_iter` | `100` | 优化器最大迭代次数 | 数据复杂时默认不够，报 ConvergenceWarning，调大（如 10000）即可 |

#### 💡 卡点实录 · `ConvergenceWarning` 长什么样

默认 `max_iter=100` 对 demo-02 这种 30 维的数据**不够用**。直接跑会看到：

```
ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
```

**两条解法（官方文档建议两者选一或都做）**：
1. **调大 `max_iter`**（demo-02 用的就是这个：`max_iter=10000`）—— 最省事
2. **特征标准化**（`StandardScaler`）—— 尺度统一后收敛速度可快 10~100 倍，根本不需要调大 max_iter

**为什么默认 100 不够**：梯度下降在特征尺度不一致时走得很慢（像在很扁的椭圆里来回震荡）。收敛需要的迭代次数和特征尺度差异强相关。

**生产建议**：永远用 `StandardScaler + LogisticRegression` 的 `Pipeline`，而不是靠 `max_iter=10000` 硬顶。
| `solver` | `'lbfgs'` | 优化器 | 默认够用；大数据用 `saga` |
| `penalty` | `'l2'` | 正则化类型 | 默认 L2；想特征稀疏用 L1 |
| `class_weight` | `None` | 类别权重 | 不平衡数据用 `'balanced'` 自动加权 |

#### 🪤 陷阱：特征缩放要做

逻辑回归的求解器（尤其 `saga` / `sag`）对特征尺度敏感，不做 `StandardScaler` 会：
- 收敛极慢（`max_iter` 再大也没用）
- 正则项偏袒大尺度特征，权重失真

**实战默认**：`StandardScaler` + `LogisticRegression` 组成 `Pipeline`。

#### 🪤 陷阱：`stratify` 参数什么时候必须加

`train_test_split` 默认**随机切分**，比例不保证。分类任务在数据量小或类别不平衡时必须加 `stratify=y`，让训练/测试集的类别比例和全集一致：

```python
# 不加 stratify（demo-02 的做法）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 结果：训练集 286:169 ≈ 1.69:1，全集 1.68:1 —— 差不多，因为样本够大

# 加 stratify（分类任务建议标配）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# 结果：每份的类别比例严格贴近全集
```

**什么时候必须加**：

| 场景 | 必须加 stratify？ | 原因 |
|---|---|---|
| 样本大 + 类别平衡（demo-02 · 1.68:1 · 569 条） | 可不加 | 大数定律保证随机切分比例接近全集 |
| 样本小（< 1000） | **必须** | 随机切容易偏 |
| 类别不平衡（10:1 或更甚） | **必须** | 不加可能切出"测试集正例只剩 2 条"的事故 |
| 回归任务 | 用不上 | stratify 是分类专属（y 是类别才能分层） |

#### 🪤 陷阱：类别不平衡的三种工程手段

1. **重采样**（SMOTE 过采样少数类 / 欠采样多数类）
2. **`class_weight='balanced'`** —— 让模型在损失函数里给少数类加权
3. **换指标 + 调阈值** —— 不改数据不改模型，改评估方式

工业界首选 2（代码最少），严重不平衡再叠 1。

---

### 八、何时该用，何时千万别用

#### ✅ 适用场景

- **二分类 baseline**——任何分类任务的第一个基线模型
- **要可解释性**——每个特征的权重直接告诉业务方影响方向
- **决策边界近似线性**——线性可分或近似线性可分的数据
- **推理延迟要求极低**——亚毫秒级（30 次乘法 + 30 次加法 + 1 次 Sigmoid）
- **需要可调阈值的业务**——风控、反欺诈、医疗筛查

#### ❌ 千万别用

- **决策边界明显非线性**（异或、环形、嵌套簇）—— 换决策树 / 随机森林 / 神经网络
- **特征维度极高 + 稀疏** + 需要自动特征选择 —— 换 L1 Lasso Logistic 或树模型
- **多分类**（>2 类）—— 技术上 sklearn 会自动 OvR，但 Softmax / 决策树更自然
- **图像、文本原始输入**—— 深度学习是更合适的选择

#### 📊 线性可分 vs 非线性决策边界

```
 ✅ 线性可分（LR 擅长）           ❌ 非线性（LR 失败）
                                 
   │  ○ ○ ○ ○                     │  ○ × × ○
   │    ○ ○ ○                     │  × ○ ○ ×
   │ ─────────── ← 一条直线        │  × ○ ○ ×     ← 一条直线拆不开
   │   × × ×                      │  ○ × × ○
   │ × × × × ×                    │                
   └────────────►                 └────────────►
   例：邮件 spam 特征              例：异或 / 环形分布
```

---

### QA

#### Q0. "逻辑回归"这四个字什么意思？

这是学这个算法的**根问题**。两个字都带误导性。

**"逻辑"**—— **和中文"逻辑"（logic，逻辑推理）完全无关**。原词 "Logistic" 来自 **Logistic function**（逻辑函数）——1838 年比利时数学家 Verhulst 研究人口增长时发明的 S 型曲线[^2]。Verhulst 用 "logistic" 这个词是因为它和 "logarithmic"（对数）对应，词源来自古希腊语 λογιστικός（算术）。

**"回归"**—— 历史遗留误导。1944 年 Joseph Berkson 推广这个模型时沿用了 "regression" 一词[^1]，但**它做的是分类，不是回归**（详见 [01 线性回归 DISTILL · Q0](../01-linear-regression/99-DISTILL.md) 的"回归"历史）。

**合起来一句话**：

> **用 S 型曲线把线性组合压成 (0, 1) 概率，再按阈值做分类。名字叫"回归"是历史遗留，干的是分类活儿。**

**判断准则 · 看 `.predict()` 吐什么**（sklearn 里"名字叫 Regression 但做分类"不止逻辑回归一个）：

| 类名 | 实际任务 | `.predict` 返回 |
|---|---|---|
| `LinearRegression` | 回归 | 连续数字 |
| `LogisticRegression` | **分类** | 类别标签（名字骗人） |
| `SoftmaxRegression` | 多分类 | 类别标签（同上） |

**记忆锚**：sklearn 类名里带 Regression **不等于** 回归任务——永远以 `.predict()` 输出类型为准。

#### Q1. 逻辑回归和线性回归到底差在哪？

两步差别：

1. 多套了一层 **Sigmoid**：把任意实数压到 `(0, 1)`
2. 最终用**阈值判断**输出类别（不是直接返回数字）

线性组合那一层（`w·x + b`）**完全一样**。sklearn 里两个 API 几乎一致（`.fit` / `.predict`），只多一个 `.predict_proba()`。

#### Q2. 那个"压缩函数"具体长什么样，要不要学？

**当前阶段不用**。本章的 L2 目标是理解"**有这么一个步骤把实数压到 (0, 1)**"——压缩函数的名字叫 **Sigmoid**，想查点脚注跳 Wikipedia。

数学形态、S 型曲线特性、`σ(z) = 1 / (1 + e^(-z))` 公式都属于 L3+ 内容（触及数学层）。按 `01-LESSON-PLAN.md` 的学习路径"**数学层放最后**"，以后学神经网络（Sigmoid / Tanh / ReLU 这类**激活函数**家族）或 GLM（**链接函数**）时自然会正式认识它。

本章 L2 的要求就是："**知道有这个压缩步骤、能解释为什么要压**（线性输出不是概率、离群点敏感）"。

#### Q3. predict 和 predict_proba 有什么区别？

- `predict(X)` 返回**类别**（0 或 1）—— 背后是 `argmax(predict_proba)`
- `predict_proba(X)` 返回**概率矩阵** —— shape `(n_samples, n_classes)`，每行加起来 = 1

二分类时，`predict` ≡ `predict_proba[:, 1] > 0.5`。但做风控/医疗时通常绕过 `predict`，直接 `predict_proba + 自定义阈值`——阈值博弈靠这个。

#### Q4. 置信度是 `|proba - 0.5|` 吗？

**不是**。置信度 = `max(predict_proba 那行)`，就是选中类别的那个概率本身。

```
proba = [0.12, 0.88]
✅ 置信度 = 0.88
❌ 置信度 = 0.88 - 0.5 = 0.38
```

#### Q5. Accuracy 为什么是"最危险的指标"？

在**类别不平衡**时会骗人。经典反例：1 万笔交易里 10 笔欺诈（0.1%），写 `return 正常` 全预测负例 → Accuracy 99.9%，但 10 笔欺诈一个没抓到，系统完全没用。

**解药**：看 `classification_report` 里每类的 Precision / Recall，或者直接看混淆矩阵。

#### Q6. Precision 和 Recall 为什么是跷跷板？

两者的**分母不同**[^6]：
- Precision = `TP / (TP + FP)` —— 分母是"模型说是的总数"
- Recall = `TP / (TP + FN)` —— 分母是"真是的总数"

调高阈值 → 只在最肯定的情况下说"是"：FP 减少（Precision ↑），但 FN 增加（Recall ↓）。反之亦然。两个分子都是 TP，一动都动，永远此消彼长。

#### Q7. 医疗场景选什么指标？为什么？

**Recall**（召回率）。

理由：**漏诊（FN）代价 >> 误诊（FP）代价**。
- 漏诊 = 真有癌症但没检出 → 病人延误治疗可能死
- 误诊 = 没癌症但被判有 → 复查一下就澄清

Recall 盯的正是"真是的里面有多少被抓出来"，分母是 TP+FN——最大化 Recall = 最小化 FN。

#### Q8. C 参数怎么调？它是正则化强度吗？

⚠️ **是倒数**。`C = 1/α`：
- `C` 越大 → 正则越弱（模型复杂度更高，更容易过拟合）
- `C` 越小 → 正则越强（模型更简单，可能欠拟合）

和原培训班的 `α`（alpha，正比于正则强度）**相反**。sklearn 默认 `C=1.0`，调参时通常在 `[0.01, 10]` 范围内扫。

#### Q9. LogisticRegression 的 `predict_proba` 可以直接当真实概率用吗？

**LogisticRegression 自带良好校准**[^4]——模型输出的 0.88 大致就是"88% 真实概率"，可以直接用于置信度判断。

但其他模型不一定：
- **SVM**：输出是距离，不是概率，要配 `CalibratedClassifierCV`
- **RandomForest**：概率有偏差，也要校准
- **朴素贝叶斯**：概率过于极端（接近 0 或 1），要校准

这是 LR 的一个独特工程价值——概率可用、可解释、可监控。

#### Q10. 业务想要"可调阈值而不重训模型"怎么实现？

线上服务不调 `model.predict()`，改为：

```python
proba = model.predict_proba(X)[:, 1]  # 正类概率
threshold = config.get("threshold", 0.5)  # 配置化
y_pred = (proba > threshold).astype(int)
```

阈值做成可配置项（配置中心 / 环境变量）。调整业务策略时只改配置，不动模型，算法层和业务层解耦。

---

### 学习轨迹（内部溯源）

- [highlights/2026-04-14-session-05-反面教材-—-C组教学失败.md](../../highlights/2026-04-14-session-05-反面教材-—-C组教学失败.md) — 第 [40]~[52] 条讨论了 "逻辑回归名字带回归但做分类"。关键原话："名字叫'回归'是历史遗留，容易误导"
- [learning-sessions/2026-04-19-session-08.md](../../learning-sessions/2026-04-19-session-08.md) — demo-02 step4_evaluate 完成，Accuracy / Precision / Recall / F1 / 混淆矩阵 Level 2 通关；业务锚"漏诊 vs 误诊"扎根；自主诊断出"评估指标是重的一环，demo-01 完全漏掉"的元反思
- [02-logistic-regression/02-NOTES.md](./02-NOTES.md) — session-07 产出的"分类 vs 回归置信度"专题。核心三句：分类有 predict_proba、回归没有要自己造、工程价值对应 Agent 架构
- [02-logistic-regression/03-DATA-FLOW.md](./03-DATA-FLOW.md) — demo-02 四步数据流图（含 shape 标注），session-08 产出，L2.5 通关
- [02-logistic-regression/01-LESSON-PLAN.md](./01-LESSON-PLAN.md) — 备课稿，含完整延展层（Log-Loss、MLE、AUC/ROC 深讲、Solver 对比）。本 DISTILL 只保留实际学过的部分
- [_shared/EVAL-METRICS.md](../_shared/EVAL-METRICS.md) — 跨 demo 共用的评估指标速查表（session-08 产出）
- [02-logistic-regression/lab/](./lab/) — step1~4 实际跑过的代码（含 step4_evaluate 的混淆矩阵打印）

---

[^1]: **逻辑回归定义 + Joseph Berkson 1944 命名** — [Wikipedia: Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression)
[^2]: **Logistic 函数的历史（Verhulst 1838 · 人口增长 S 型曲线 · 词源和"逻辑推理"无关）** — [Wikipedia: Logistic function](https://en.wikipedia.org/wiki/Logistic_function)
[^3]: **Sigmoid 函数（S 型，压缩到 (0, 1)）** — [Wikipedia: Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function)
[^4]: **sklearn LogisticRegression API（含 C / max_iter / solver / predict_proba / classes_）** — [scikit-learn docs: LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
[^5]: **混淆矩阵（TP/FP/TN/FN 四格）** — [Wikipedia: Confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
[^6]: **Precision 和 Recall 定义 + 跷跷板关系** — [Wikipedia: Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall)
[^7]: **F1 分数 = Precision 和 Recall 的调和平均** — [Wikipedia: F-score](https://en.wikipedia.org/wiki/F-score)
[^8]: **ROC 曲线 / AUC（跨阈值评估）** — [Wikipedia: Receiver operating characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
