# LESSON-PLAN · 第 4 节 · 决策树（Decision Tree）

> **三源合并产出**：
> 1. 原始培训笔记 `assets/source-materials/第4阶段-机器学习/决策树.md`
> 2. 第一版改编 `./05-Decision-Trees-and-Rule-Extraction.md`（合并后待删）
> 3. 联网搜索交叉验证 `./04-decision-tree-digest.md`
>
> 按模板 v0.2 填槽。LESSON-PLAN 给 Claude 查，不是给学员自读——数学、工程类比、反面教材一锅保留。

---

## §0 · 课前须知

### 0.1 这节课的位置

| 维度 | 值 |
|---|---|
| 算法 | 决策树 Decision Tree（CART / ID3 / C4.5 三代） |
| **可支持的任务类型** | 二分类 / 多分类 / 回归（CART 三任务通吃；ID3/C4.5 仅分类） |
| roadmap 阶段 | 阶段一 · 传统 ML 工程化 · 第 3 层场景层 |
| 目标深度 | L3（冲刺期只打 L2 手感，本文件覆盖 L3 以备回炉） |
| 在 10 节课里 | 第 4 节 |
| **推荐前置**（非强制） | 第 3 节 KNN — 有"训练/推理"心智即可进入，不依赖距离概念 |

### 0.2 当前学习状态（2026-04-24 快照）

> **情境锚句**：这一节开始时，学员已通关线性回归 / 逻辑回归 / KNN 的"模型即函数"心智 + sklearn 三行 API + 分类/回归区分 + Precision/Recall。本节是冲刺 Day 1 的第 1 个算法。没碰过：熵的数学定义、剪枝参数、可解释性 vs 黑盒权衡。回炉已通关的概念 = 浪费时间。

- 已 Level 2 通关的相关概念：
  - 模型即函数 / fit-predict 三行 API（L3）
  - 分类 vs 回归（L2.5）
  - Precision / Recall / 混淆矩阵（L2.5，F1 欠账）
  - KNN 五步法 + "WHERE 过滤 vs 距离综合"直觉（L2）
- 已跑通的 demo：
  - 01-linear-regression（房价）
  - 02-logistic-regression（乳腺癌）
  - 03-knn（进行中，L2）
- 本节课的起点：
  - **冲刺 Day 1 · L2 手感**（40 分钟）：digest → step1 → 直觉 → 3 道题
  - 本 LESSON-PLAN 覆盖 L3 完整知识面，冲刺只 pick §1.1 + §1.2 + §5.1-5.3 核心题
  - 后续 L3 回炉时走全量

### 0.3 候选数据集清单（备课覆盖全量，上课现场选）

> **情境锚句**：备课不预选，上课由 Claude 根据学员状态 pick。每个候选带教学价值对比 + URL。

| 候选 | 规模（行×列） | 特点 | 教学价值 | 适合上课场景 | 数据源 URL |
|---|---|---|---|---|---|
| **Titanic 泰坦尼克号** | 891 × 12 | 二分类 survive；混合数值+类别；含缺失值 | 原始笔记主例；能讲**缺失值处理 + 类别编码 + 可视化 if-else 规则**；规则能翻译成"女性儿童优先"的人话 | 第一轮主推 / 对齐原始笔记 | [Kaggle Titanic](https://www.kaggle.com/competitions/titanic/data) |
| **贷款申请 15 条**（原始笔记作业 3） | 15 × 4 | 多分类 approve/reject；纯类别特征；小到能手算 | **手推 ID3 信息熵 + 信息增益**的唯一合适样本；白板上完整走一遍 | 想手推数学 | 原始笔记 `决策树.md` 作业 3 |
| **sklearn load_iris** | 150 × 4 | 3 分类；全数值 | **决策边界可视化**最干净；树深 3 就 95%+ | 零基础第一次看树结构 | [sklearn load_iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) |
| **California Housing** | 20640 × 8 | 回归任务；房价中位数 | **CART 回归树**锚；"线性回归是直线，回归树是阶梯"戏剧对比 | 讲 CART 既分类又回归时 | [fetch_california_housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) |
| **UCI Adult / Census Income** | 48842 × 14 | 二分类 income >50K；含敏感特征（种族、性别） | **反面教材钩子**——演示树在敏感特征上 leak bias，对接 Apple Card / COMPAS | 讲公平性 + 可解释性双刃剑 | [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult) |
| **Breast Cancer Wisconsin** | 569 × 30 | 二分类；全数值 | 已在 demo-02 用过 | **强建议不复用**，除非"同一数据 LR/DT 横向对比" | [load_breast_cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) |

**现场选型规则**：
- 对齐原始笔记 / 规则人话叙事 → **Titanic**
- 手推数学 → **贷款 15 条**
- 可视化决策边界 → **Iris**
- CART 回归树戏剧对比 → **California Housing**
- 讲公平性 / 反面教材 → **Adult**

**冲刺 Day 1 默认 pick**：Iris（最小、干净、可视化），理由——L2 要求"跑通看输入输出 + 跟 KNN 差异"，Iris 恰好是 KNN 的老战场（session-11 曾用类似结构），同一数据换算法对比差异最直观。

---

## §1 · 钩子 · 业务锚 + 跑通 demo

### 1.1 一句话业务类比

> 给工程师的锚：**决策树 = 让机器从历史数据里自动长出来的 Drools 规则引擎**。

- 软件 1.0：产品经理拍脑袋写 `if (age > 30) reject()`，你手工堆嵌套
- 软件 2.0：喂给 sklearn 10 万条样本 + 标签，它自动长出一棵 20 层的 if-else 嵌套，每层选哪个字段 / 切在哪个阈值全是算出来的
- 推理 = 沿着 if-else 链路走一次，O(树深度)，亚毫秒
- **最大亮点**：树可以 `export_text` 反编译成人类可读的 if-else，甚至翻译回 SQL——**唯一一个能直接拿给法务看的 ML 模型**

**Alex 存量对应**：你 7 年写过的 Drools、规则引擎、审批流，都是这棵树的手工版。决策树只是把"谁先判"这一步从主观评估换成了信息增益暴力搜索。

### 1.2 最小可跑 demo

#### 1.2.A 候选 · Iris（推荐冲刺默认）

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

print("accuracy:", clf.score(X_test, y_test))
print(export_text(clf, feature_names=load_iris().feature_names))
```

- **预期输出**：accuracy ≈ 0.95+；一段类似
  ```
  |--- petal length (cm) <= 2.45
  |   |--- class: 0
  |--- petal length (cm) >  2.45
  |   |--- petal width (cm) <= 1.75
  ...
  ```
- **第一印象锚**：" fit 完一棵树能 `print` 出 if-else 文本"——这件事 KNN / 线性 / 逻辑全都做不到

#### 1.2.B 候选 · Titanic

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

df = pd.read_csv("titanic.csv")
df["Sex"] = (df["Sex"] == "female").astype(int)
df = df[["Pclass", "Sex", "Age", "Fare", "Survived"]].dropna()

X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)
print("accuracy:", clf.score(X_test, y_test))
print(export_text(clf, feature_names=list(X.columns)))
```

- **预期输出**：accuracy ≈ 0.80+；规则能读出"Sex=0（男）→ 多数死；Sex=1（女）+ Pclass<=2 → 多数生"
- **第一印象锚**：规则能翻译成"女性儿童优先"的人话——这就是金融/医疗审计场景要的东西

#### 1.2.C 候选 · 贷款 15 条（手推）

原始笔记作业 3，纯白板走一遍 ID3 信息增益，不用 sklearn。适合已经跑完 Iris 后想"手算验证"的学员。

---

## §2 · 递归缩放讲解

### 2.1 黑盒视图 · 输入 → 输出

> **情境锚句**：先别管里面怎么算。这个黑盒吃什么、吐什么。

- **输入**：`X shape (n_samples, n_features)`——可以混合数值 + 类别（类别需先编码）；`y shape (n_samples,)` 是类别标签（分类）或连续值（回归）
- **输出**：`predict(X_new)` → 类别（分类）或数值（回归）；`predict_proba(X_new)` → 各类概率
- **工程接口类比**：
  - 训练 = `POST /compile-rules`，吃数据吐一棵树（< 50KB 文件）
  - 推理 = `POST /predict`，O(树深度) 亚毫秒
  - **决策树独家**：`GET /rules` 导出整棵 if-else 文本，可审计

### 2.2 拆一层 · 核心机制

> **情境锚句**：打开黑盒第一层——最关键的一个动作。

**三步递归**：
1. 遍历每个特征的每个候选切点
2. 对每个切法算"不纯度下降量"（信息增益 / 信息增益率 / 基尼增益）
3. 选最大的那个切下去 → 数据分成左右两堆 → 递归到两堆上

**伪代码**：

```
function buildTree(data):
    if shouldStop(data):             # 纯了 / 深度到顶 / 样本太少
        return LeafNode(majorityClass(data))

    bestFeature, bestSplit = null, -inf
    for feature in data.features:
        for candidateSplit in candidateSplits(data, feature):
            gain = purityGain(data, feature, candidateSplit)
            if gain > bestSplit:
                bestFeature, bestSplit = feature, candidateSplit

    left, right = split(data, bestFeature, bestSplit)
    return InternalNode(
        feature = bestFeature,
        threshold = bestSplit,
        leftChild = buildTree(left),
        rightChild = buildTree(right)
    )
```

**关键工程点**：
- 每个节点都重跑一遍"遍历所有字段 × 所有切点"。因此训练是 O(n_features × n_samples × log(n_samples))
- 预测是 O(tree_depth)——亚毫秒
- `stopCondition` 就是 sklearn 的一堆 `max_depth` / `min_samples_split` / `min_samples_leaf` / `ccp_alpha` 超参

### 2.3 再拆一层 · 数学附录

> **情境锚句**：学员追问时才展开。主线不讲公式。

**信息熵** Entropy（ID3 用）：

$$H = -\sum_{i=1}^{k}p_i\log(p_i)$$

- 全一类 → 熵 = 0（最纯）
- 均匀分布 → 熵最大（最乱）

**信息增益** Information Gain（ID3 用）：

$$g(D,A) = H(D) - H(D|A)$$

- 切前熵 - 切后加权条件熵
- 越大越值得切
- **坑**：偏爱取值多的特征（user_id 一刀切成 N 个纯叶子）→ C4.5 引入信息增益率

**信息增益率** Gain Ratio（C4.5 用）：

$$\text{GainRatio}(D, A) = \frac{g(D, A)}{IV(A)}$$

其中 IV(A) 是特征 A 的内在熵，作为"取值数量"的惩罚。

**基尼指数** Gini Index（CART 用）：

$$\text{Gini} = 1 - \sum_{i=1}^{k}p_i^2$$

- 数学上和熵接近但**免去 log 运算**
- sklearn 默认用它，因为更快

**CART 回归平方损失**：

$$\text{Loss}(y, f(x)) = (f(x) - y)^2$$

叶子节点输出 = 该叶子所有样本 y 的均值。

**学员追问下限答案**：熵 / 基尼 / 信息增益 / 基尼增益——统一理解为**"不纯度的不同量化方法"**，像 MD5/SHA1/SHA256 都是哈希，区别只在实现细节。选哪个不是业务决策。

---

## §3 · 坏方案 → 好方案递进

### 3.1 坏方案 A / Baseline · 手写 Drools 规则

不用决策树，业务经理拍脑袋写规则：
```java
if (信用分 < 600) reject();
else if (历史违约 > 2) reject();
else if (月收入 < 5000 && !有房) reject();
else approve();
```

**什么时候崩**：
- 字段超过 10 个，规则优先级互相冲突
- 阈值（600、5000）没数据支撑，纯靠经验 / 拍脑袋
- 业务 A/B 测试新规则 vs 老规则谁更准 → 没有客观衡量
- 半年后规则膨胀到 200+ 条，没人敢删

### 3.2 坏方案 B · 用 KNN 做分类

KNN 能做分类，但：
- 推理慢：每次要扫全表算距离（除非用 ANN 索引）
- 不可解释：法务问"为啥拒绝用户 X" → "因为他最近的 5 个邻居里 3 个违约" → 监管：？
- 对无关特征敏感：多加一个"省份 id" 会把相似度结构破坏

### 3.3 决策树解决了什么

- **数据驱动选顺序**：字段优先级靠信息增益暴力搜出来，不拍脑袋
- **亚毫秒推理**：O(树深度)，几十层树推理都是 us 级
- **可审计**：`export_text` 反编译为 if-else，法务、合规、业务直接看
- **序列化小**：max_depth=8 的树通常 < 50KB，PMML/ONNX 轻松跨语言

**引入的新问题（代价）**：
- 单棵树精度天花板低（非线性拟合能力有限）
- 不加约束 100% 过拟合（每片叶子记 1 个样本）
- 对数据微扰敏感（同样的数据换 random_state 长出来不一样）
- → 引出集成学习（下一节 RF + XGBoost）

---

## §4 · 术语卡片

### 4.1 信息熵 Entropy

- **中英对照**：信息熵 / Entropy
- **业务锚**：不确定度。类比数据库里某张表的"数据混乱程度"——全一类很干净熵=0；一半一半很乱熵=1
- **一句话定义**：`H = -Σ p·log(p)`，衡量一堆样本标签分布的不确定度
- **在本算法里的角色**：ID3 用来评估"切一刀后左右两堆是不是更纯"
- **常见追问**：
  - Q: 熵的对数 log 底取什么？
    - 下限：底 2（bit）或 e（nat），都可以，比较相对大小不影响结论
  - Q: 熵 vs 交叉熵？
    - 下限：熵是单分布的不确定度；交叉熵是两个分布的差异度（LLM loss 就是交叉熵）

### 4.2 信息增益 Information Gain

- **中英对照**：信息增益 / Information Gain
- **业务锚**：**纯度提升量**。切一刀前熵多少，切完之后两堆加权熵多少，差多少就是增益
- **一句话定义**：`g(D,A) = H(D) - H(D|A)`
- **在本算法里的角色**：ID3 选分裂特征的唯一标准（选增益最大的）
- **常见追问**：
  - Q: 为什么信息增益偏爱取值多的特征？
    - 下限：user_id 有 10 万个取值，一刀切出 10 万个纯叶子，增益理论上最大但毫无泛化价值。C4.5 的信息增益率就是为了修这个坑
    - URL: [sklearn · Beware Default RF Importances](https://explained.ai/rf-importance/)

### 4.3 基尼指数 Gini Index

- **中英对照**：基尼指数 / Gini Index
- **业务锚**：不纯度的另一种度量。和熵作用一样，但不用算 log，快
- **一句话定义**：`Gini = 1 - Σ p²`
- **在本算法里的角色**：CART（sklearn 默认）用来评估分裂
- **常见追问**：
  - Q: Gini vs Entropy 实战选哪个？
    - 下限：精度实测几乎一样，Gini 快（无 log）所以是默认。选哪个不是业务决策
    - URL: [Gini vs Entropy 对比](https://www.garysieling.com/blog/sklearn-gini-vs-entropy-criteria/)

### 4.4 剪枝 Pruning

- **中英对照**：剪枝 / Pruning（预剪枝 Pre-pruning / 后剪枝 Post-pruning）
- **业务锚**：
  - 预剪枝 = **熔断器**，边长边判断不值得就停
  - 后剪枝 = **GC**，先长完再回头砍冗余
- **一句话定义**：决策树天生过拟合（每叶 1 样本），剪枝 = 人为限制复杂度
- **在本算法里的角色**：sklearn 的 `max_depth` / `min_samples_split` / `min_samples_leaf` 是预剪枝；`ccp_alpha` 是后剪枝
- **常见追问**：
  - Q: 生产上用预剪枝还是后剪枝？
    - 下限：90% 用预剪枝（`max_depth=5-10`），够用且可控。后剪枝 `ccp_alpha` 配 CV 选最优是精度优先场景
    - URL: [sklearn Post pruning with cost complexity](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html)
  - Q: max_depth 没设会怎样？
    - 下限：默认 `None` = 无限长，到每叶 1 样本（或者 `min_samples_leaf` 止）。训练 100% 上线崩盘

### 4.5 CART

- **中英对照**：CART / Classification And Regression Tree
- **业务锚**：MySQL 里的 InnoDB——主流标配
- **一句话定义**：Breiman 1984 提出，严格二叉树，既能分类（基尼）又能回归（MSE），sklearn 默认
- **在本算法里的角色**：`DecisionTreeClassifier` / `DecisionTreeRegressor` 都是它；XGBoost / LightGBM 的 base learner 也是它
- **常见追问**：
  - Q: ID3 / C4.5 / CART 三代选哪个？
    - 下限：生产只用 CART。ID3 / C4.5 是为了理解"分裂依据演进"，不是为了用
  - Q: 为什么 XGBoost 用 CART 不用 ID3？
    - 下限：CART 的平方损失直接对应 gradient boosting 的残差拟合，数学上严丝合缝

### 4.6 特征重要性 Feature Importance

- **中英对照**：特征重要性 / Feature Importance（MDI / Mean Decrease Impurity）
- **业务锚**：PM 最爱看的"哪个字段最关键"——但容易被用错
- **一句话定义**：`clf.feature_importances_` = 该特征在所有分裂中带来的基尼/熵减少量的加权和
- **在本算法里的角色**：事后归因
- **常见追问**：
  - Q: 为什么 RF 里 user_id 排第一名？
    - 下限：MDI 对高基数特征有偏。永远切 Permutation Importance 或 SHAP 做稳健替代
    - URL: [explained.ai · Beware Default RF Importances](https://explained.ai/rf-importance/)

---

## §5 · Questionnaire · 显式 active recall

> **情境锚句**：闭卷题库，按 Bloom 三层。Claude 上课时抽题。冲刺 Day 1 只用 5.1 + 5.3 各 1 题 + 场景题 1 题（共 3 道）。

### 5.1 记忆层（能复述）

1. 决策树的输入输出是什么 shape？输出和 KNN / 逻辑回归有啥本质差异？
2. 核心机制一句话：训练在干啥？推理在干啥？
3. 熵公式是什么？熵越大表示什么？熵 = 0 表示什么？
4. ID3 / C4.5 / CART 三者的分裂依据分别是什么？sklearn 默认哪个？
5. 预剪枝和后剪枝的区别？对应 sklearn 哪些参数？

### 5.2 应用层（能选型 / 能读代码）

1. 这行代码每个参数在干啥：
   ```python
   DecisionTreeClassifier(criterion='gini', max_depth=8, min_samples_split=20, min_samples_leaf=10)
   ```
2. 业务场景：银行信贷审批系统要求"每次决策必须能向监管解释"，你选决策树还是 XGBoost？为什么？
3. 一个 CSV 有字段 `[user_id, age, city, history_clicks, label]`，直接喂进 `DecisionTreeClassifier` 有什么风险？怎么改？
4. 训练完一棵 max_depth=10 的树，accuracy=100% / 测试 accuracy=60%。诊断 + 三种可能的修复方案。
5. `export_text` 打印出来的规则里有 `|--- class: 1`，这个 1 是什么？和分类概率 `predict_proba` 的关系？

### 5.3 迁移层（新场景 / 对比）

1. **KNN vs 决策树**：
   - 训练时各自在干啥？
   - 推理时各自在干啥？
   - 一个新业务"千万级用户实时风控"，哪个更合适？为什么？
2. 如果特征数从 10 涨到 1000（比如 embedding 向量），决策树会崩在哪？为什么 RAG 场景不用决策树？
3. 业务方说"预测这个用户会不会流失"，你只能选一个基础模型：逻辑回归 / 决策树 / KNN。怎么选？
4. 反面教材迁移：Apple Card 被曝性别歧视，你如果是 Goldman Sachs 的 ML 架构师，能用决策树的什么特性来查问题？
5. 单棵决策树的天花板是什么？为什么必须升到 RF / XGBoost？（下一节预告）

---

## §6 · 常见坑 / 反面教材

> **情境锚句**：学员必然踩的坑。Claude 要提前埋伏。

### 6.1 认知误解

- **误解 1 · "决策树 = 一定是可解释的"**：`max_depth=20` 的树人类读 100 条路径就晕了。**可解释性和"复杂度预算"强耦合**。超过 `max_depth=8` + 50 特征，你实际上在看一个灰盒
- **误解 2 · "树好像很智能，自己选字段"**：它只是暴力遍历。不是智能，是蛮力
- **误解 3 · "决策树不用特征工程"**：连续特征确实不用标准化（和 KNN / 逻辑回归不同），但**类别特征必须编码**（OneHot / Ordinal），sklearn 决策树不原生支持类别变量

### 6.2 工程坑

- **坑 1 · 类别特征工程短板**：`DecisionTreeClassifier` 不原生支持类别字段，必须预编码。高基数（城市、商品 ID）用 OneHot 会爆炸，OrdinalEncoder 会误导树当成有序值。工业解法是 `HistGradientBoostingClassifier(categorical_features=[...])` 或 LightGBM 原生类别支持。[URL](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_categorical.html)
- **坑 2 · feature_importances_ 骗人**：对高基数特征有偏（user_id 会排第一）。永远切 `sklearn.inspection.permutation_importance` 做稳健替代
- **坑 3 · 不设 max_depth 上线崩盘**：默认 `None` = 无限长。训练 100% 正常，测试集一看才发现过拟合
- **坑 4 · 跨语言部署状态漂移**：训练用 Python + 某种类别编码，线上 Java 推理时编码表不一致 → 预测漂移。**解法**：用 PMML 导出整个 `PMMLPipeline`（含预处理），或 ONNX 配合 onnxruntime

### 6.3 业界反面教材

- **Apple Card 性别歧视（2019）**：Goldman Sachs 的信用评分模型用 tree-based，即使未显式用性别，通过代理变量（邮编、消费类别）间接编码。**正是决策树可解释性被监管要求的原因**——能打开路径看是哪些特征驱动。[Washington Post](https://www.washingtonpost.com/business/2019/11/11/apple-card-algorithm-sparks-gender-bias-allegations-against-goldman-sachs/) / [AI Incident DB #92](https://incidentdatabase.ai/cite/92/)
- **COMPAS 种族偏见（ProPublica 2016）**：司法风险打分系统，黑人被告被错误标"高风险"概率是白人 2 倍。**决策树"可解释"也救不了算法公平**的经典案例。[ProPublica Machine Bias](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)
- **规则爆炸**：某银行从决策树导出规则灌进 Drools，规则数从 30 → 300 → 意大利面。**教训**：可解释性要配套"规则治理（Rule Governance）"流程

### 6.4 跨语言落地坑位

- **PMML 路径**（给 Java 老栈 / 金融监管）：`sklearn2pmml` + `jpmml-evaluator`。**不可替代**：PMML 序列化了完整 pipeline（含预处理），彻底解决训推一致性
- **ONNX 路径**（给 Go / Node / Rust）：`skl2onnx` + `onnxruntime-go`。P99 延迟 1-2ms
- **m2cgen 路径**（极致低延迟）：把树反编译成原生 Java/Go/C 源代码，跳过任何 runtime。决策树的"简单 + 可解释"带来的独家优势

---

## §7 · 数据流图 · 跑完 demo 后补

> **情境锚句**：本节为占位。`DATA-FLOW.md` 跑完 demo 后创建。

详见 `./DATA-FLOW.md`（跑完 demo 后创建）

---

## §8 · 上课调度档

> **情境锚句**：给 Claude 现场选路径用。

### 8.1 冲刺 Day 1 · L2 · 40 分钟档

1. §1.1 业务锚（2'）—— Drools 规则引擎类比，绑定 Alex 7 年存量
2. §1.2.A Iris 跑 step1（10'）—— 看 `export_text` 输出
3. §2.2 核心机制 + 伪代码（8'）—— 遍历 × 选最大增益 × 递归
4. §5.3.1 KNN vs 决策树差异题（10'）—— L2 验收第 2 条，不过回炉
5. §5.2.2 业务场景选型题（5'）—— 信贷审批选树还是 XGBoost
6. 收尾 · TWO-DAY-SPRINT 勾选 Day 1 第 1 个（5'）

### 8.2 L3 回炉 · 完整档（90 分钟）

1. §1.1 + §1.2.A Iris（10'）
2. §2.1 黑盒视图（5'）
3. §2.2 核心机制 + 伪代码（10'）
4. §2.3 数学附录 · 熵 / 增益 / 基尼（15'，按需展开）
5. §3 坏→好三档递进（10'）
6. §4 术语卡片 6 张（15'，挑 3-4 张深讲）
7. §5.1 + §5.2 + §5.3 各抽 2 题（15'）
8. §6 常见坑 · Apple Card + COMPAS + 规则爆炸（10'）
9. 收尾：§5.3.5 引到集成学习（下一节 RF / XGBoost）

### 8.3 架构对话档（45 分钟，跳过数学）

1. §1.1 业务锚（3'）
2. §1.2.B Titanic 跑（10'）—— 看规则翻译
3. §3 坏→好：Drools 规则 vs 数据驱动树（8'）
4. §6.4 跨语言落地三路径 · PMML / ONNX / m2cgen（12'）
5. §6.3 反面教材 · Apple Card + COMPAS（7'）
6. §5.3.4 业务迁移题（5'）

---

## 附：迭代记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-04-24 | v0.1 | 首版。基于三源合并（原始笔记 + 第一版改编 05-Decision-Trees-and-Rule-Extraction.md + digest）。冲刺 Day 1 起手。按模板 v0.2 填槽。 |
