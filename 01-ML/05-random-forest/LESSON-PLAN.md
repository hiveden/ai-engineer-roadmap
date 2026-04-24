# LESSON-PLAN · 第 5 节 · 随机森林（Random Forest / 05-random-forest）

> **🧪 试行模板 v0.2**（自 2026-04-20 / session-09 起）：骨架见 [`../_shared/LESSON-PLAN-TEMPLATE.md`](../_shared/LESSON-PLAN-TEMPLATE.md)。demo-03 KNN 是第 1 个试用算法，本节是第 2 个。
>
> **本文档定位**：给 Claude 备课查的 RAG 文档，不是给 Alex 自读的教材。数学 + 工程类比双保留。
>
> **三源基础**：详见 §8.9 定位表 + [`05-random-forest-digest.md`](./05-random-forest-digest.md)（L1 素材包，45+ URL 已落地）。

---

## §0 · 课前须知

### 0.1 这节课的位置

| 维度 | 值 |
|---|---|
| 算法 | 随机森林 Random Forest |
| **可支持的任务类型** | ① 二分类 / 多分类（`RandomForestClassifier`）② 回归（`RandomForestRegressor`）③ 多输出 multi-output（一次预测多个目标列）④ 特征重要性排序（作为其他算法的前置特征筛选器）⑤ 异常检测（变种 `IsolationForest`，机制同源） |
| roadmap 阶段 | 阶段一 · 传统 ML 工程化 · 第 5 节（集成学习 bagging 代表） |
| 目标深度 | L3（Alex 能独立做 RF vs XGBoost vs 逻辑回归的选型判断） |
| 在 10 节课里 | 第 5 节（03 KNN → 04 决策树 → **05 RF** → 06 XGBoost） |
| **推荐前置**（非强制） | 第 4 节 决策树（RF = 决策树集成，单树原理是 RF 的基础） |
| 可选前置（提供对照轴） | 第 3 节 KNN（用来对比"懒惰 vs 急切"、"要不要 scale"、"多线程并行 vs 单查询全表扫描"三条轴） |

> ⚠️ **不在此处定死数据集**。数据集选型见 §0.3 + 上课现场决策。

### 0.2 当前学习状态（2026-04-24 快照）

> **情境锚句**：这一节开始时，Alex 已经通关了 KNN 的五步法 / 距离几何直觉 / Software 1.0 vs 2.0 第一跳（豆瓣打分场景）；还没碰过**集成学习 ensemble**、**bootstrap 有放回采样**、**OOB**、**feature importance** 四个概念。如果又回炉重讲 KNN = 浪费时间。

- **已 Level 2 通关的相关概念**（来自 [session-11](../../learning-sessions/2026-04-21-session-11.md)）：
  - 样本 / query / 训练数据 三角色
  - 特征 feature / 维度 dimension
  - 分类 vs 回归
  - Software 1.0 WHERE 直觉 vs 2.0 距离/Embedding 直觉（L2.5）
  - 元能力：按停键、挑战钩子牢度、拒绝勉强类比（均 L3 或近 L3）

- **已跑通的 demo**：
  - demo-01 线性回归
  - demo-02 逻辑回归
  - demo-03 KNN（豆瓣 / Wine 对照）
  - demo-04 决策树（假设先讲，单棵树的分裂 / Gini / 深度控制已焊住）

- **本节课的起点**：**"一棵树不够就种一片林"**——从学员熟悉的决策树出发，直接抛一个问题："单棵决策树有什么毛病，为什么要种一大片？" 让 Alex 自己从"过拟合"跳到"投票"，再自然引出 bagging + 特征随机两重随机。

- **首次会话的 attune 要点**：
  - Alex 习惯"先问为什么，再看怎么做"——§1 业务锚（Quorum）要先立住，再跑代码
  - Alex 对"勉强类比"敏感（session-11 知乎关注列表翻车）——**Quorum 类比要先自检**：RF 树之间是否真的"独立 + 平权投票"？答案：bagging + 特征随机正是为了"独立"（decorrelate）；默认平权投票（分类投多数，回归取均值）；这类比可以过关
  - session-11 盲区里的 **F1 分数** 本节必须闭环（RF 在不平衡场景的 recall 讨论是天然触发点）

### 0.3 候选数据集清单（备课覆盖全量，上课现场选）

> **情境锚句**：这一章列出所有**可选**数据集——备课时不预选，上课时 Claude 根据 Alex 状态现场 pick。每个候选带教学价值 + URL。

| 候选 | 规模（行×列） | 任务 | 特点 | 教学价值（能讲什么钩子） | 适合上课场景 | 数据源 URL |
|---|---|---|---|---|---|---|
| **Titanic** 泰坦尼克 | 891 × 12 | 二分类 | 混合类型（数值+类别+缺失） | 原始培训笔记默认数据集；Sex / Pclass 是天然强特征，适合首次看 feature importance 排序 | 第一次接触 / 跟原教材对齐 | [Kaggle Titanic](https://www.kaggle.com/c/titanic) |
| **Wine Quality** 红酒品质 | 4898 × 12 | 多分类（1-10 品质） | 全数值 | **和 demo-03 KNN 对照组**——RF 不需要标准化，KNN 必须 scale；同一份数据两套预处理 | 想回炉 demo-03 对比 | [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) |
| **German Credit** 德国信贷 | 1000 × 20 | 二分类（违约） | 类别不平衡 7:3 | **业界最经典风控数据集**；能引爆 accuracy 虚高 + 必须看 recall / PR-AUC 的话题；和 Alex 的 AI Agent 转型方向（风控/反欺诈是常见落地领域）贴 | **默认推荐** / 想走风控主线 | [UCI Statlog German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) |
| **Credit Card Fraud** 信用卡欺诈 | 284807 × 30 | 二分类（欺诈） | **极端不平衡** 0.17% | 直接复现"全预测为 0 也有 99.83% accuracy"的反教材；PCA 脱敏后的 V1-V28 是讲 feature importance 陷阱的好素材 | 想炸不平衡话题 + SMOTE + class_weight | [Kaggle CCFraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **California Housing** 加州房价 | 20640 × 8 | 回归 | 全数值 + 有地理强相关 | 演示 `RandomForestRegressor`；可与 demo-01 线性回归对照（RF 捕捉非线性 + 交互项） | 想对比 RF 回归 vs 线性回归 | [sklearn fetch_california_housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) |

**现场选型规则**（供 Claude 上课参考）：

| Alex 状态 | 推荐 | 为什么 |
|---|---|---|
| 注意力满 / 首次 RF | **German Credit** | 规模适中（1000 条训练快）+ 风控故事线带情绪驱动 + 天然铺垫不平衡话题 |
| 注意力一般 / 想快跑通 | **Titanic** | 最小 + 最干净 + 跟原教材对齐，10 行代码出 0.8+ accuracy |
| 想建立工程直觉 / 炸机 | **Credit Card Fraud** | 28 万行 + 极端不平衡，accuracy 虚高一眼破功 |
| 想回炉 demo-03 对照 | **Wine Quality** | 和 KNN 同一份数据，RF 不 scale 也能干，直接体感差异 |
| 想讲回归变种 | **California Housing** | `RandomForestRegressor` 一行切换；和 demo-01 线性回归的预测残差对比非常直观 |

---

## §1 · 钩子 · 业务锚 + 跑通 demo（fast.ai top-down）

> **情境锚句**：这一节先让 Alex **跑通最小 demo 拿到结果 + 建立"一片林"的心智模型**，再讲原理。**不从定义开始**。

### 1.1 一句话业务类比

> **给工程师的锚**：**随机森林 = ZooKeeper Quorum 投票集群**
>
> 每个节点（决策树）只看**部分数据**（bagging 样本随机）+ **部分字段**（max_features 特征随机）就做出自己的判断；最后**多数派决议**（分类投票 / 回归取均值）。
>
> - Quorum：保证单节点故障不影响总体决议 → RF：保证单棵树过拟合不影响总体泛化
> - 投票要有效，成员必须"独立"（相关性低）→ RF 两重随机正是为了 decorrelate
> - 加节点到一定数量后收益饱和 → RF 的 n_estimators 也有饱和点（200-400 之后几乎不动）

**和 Alex 转型方向的锚**：
- 在 LLM Agent 栈里，RF 是"工具调用分类 / 意图识别"的**常用 fallback 模型**——轻量（几 MB 模型文件）、可解释（能说出"为什么走这个 tool"）、离线部署友好（CPU 纯推理）。LLM 不可用 / 成本敏感 / 需审计时兜底。
- 对比 Embedding + KNN 路线：RF 适合**有标签 + 特征工程能做**的场景；Embedding 适合**冷启动 + 语义理解**。两者不是替代，是栈里不同层。

**"这个类比有没有翻车可能"自检**（吸取 session-11 知乎例子失败的教训）：
- ✅ RF 树之间"独立性"是真的（bagging + 特征随机的核心设计目标就是 decorrelate）
- ✅ "平权投票"是默认行为（`voting='hard'` 等价），加权变种存在但不是主流
- ⚠️ 一个差异点：Quorum 节点**同构**（多副本一致），RF 树**异构**（每棵树看不同样本/特征，做出不同决策）——这个差异反而是优点（diversity 是 ensemble 能 work 的前提），讲的时候顺手点一下就行

### 1.2 最小可跑 demo（每个候选数据集各一份 10-15 行代码）

> **情境锚句**：**不预选数据集**——5 个候选各写一份最小 demo。上课时 Claude 根据现场 pick 一份跑。代码放 `lab/step1_minimal_<dataset>.py`。

#### 1.2.A 候选 A · Titanic

```python
# lab/step1_minimal_titanic.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
df = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Survived"]].dropna()
df["Sex"] = (df["Sex"] == "male").astype(int)  # 最朴素的 label encoding
X, y = df.drop(columns="Survived"), df["Survived"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
clf.fit(X_tr, y_tr)

print(f"Train: {clf.score(X_tr, y_tr):.3f}  Test: {clf.score(X_te, y_te):.3f}  OOB: {clf.oob_score_:.3f}")
print(classification_report(y_te, clf.predict(X_te), digits=3))
print("Feature importance:", dict(zip(X.columns, clf.feature_importances_.round(3))))
```

- **预期输出**：Test ~0.80-0.82，OOB ~0.81（**OOB 和 Test 基本贴合** = 关键钩子）；feature importance 里 `Sex`、`Fare`、`Age` 排前三
- **第一印象锚**：
  - "OOB 这个数是哪来的？我没做 CV 啊" → 引 §4.3 OOB 术语卡
  - "Sex 权重最高" → 引 §4.6 feature importance + §6.1 陷阱（Sex 是低基数，这次刚好靠谱，但要埋 MDI 陷阱的雷）

#### 1.2.B 候选 B · Wine Quality（和 KNN 对照）

```python
# lab/step1_minimal_wine.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
df = pd.read_csv(url, sep=";")
X, y = df.drop(columns="quality"), df["quality"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# 关键钩子：不 StandardScaler 也能干
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_tr, y_tr)

print(f"Wine quality test score (无 scaler): {clf.score(X_te, y_te):.3f}")
# 对照 demo-03 KNN：没标准化时 KNN 会崩，RF 岿然不动
```

- **预期输出**：test ~0.68-0.70（多分类 6 类，baseline 不会特别高）
- **第一印象锚**：
  - "为什么 RF 不用 scale，KNN 就必须" → 核心区分点：KNN 算距离（特征量纲直接影响距离），RF 每棵树**只看单列分裂**（只在乎大小顺序，不在乎量纲）→ 引 §2.2 核心机制 + §3 坏→好递进
  - 这是**和 demo-03 最快建立对比的方式**——同一份数据，RF 省了一个预处理步骤

#### 1.2.C 候选 C · German Credit（默认推荐 / 风控主线）

```python
# lab/step1_minimal_german.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# UCI German Credit，数值版
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric"
df = pd.read_csv(url, sep=r"\s+", header=None)
X, y = df.iloc[:, :-1], df.iloc[:, -1] - 1  # label 1/2 → 0/1，1=违约

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 关键钩子：class_weight="balanced" 的 A/B 对照
clf_naive = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_tr, y_tr)
clf_bal = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced").fit(X_tr, y_tr)

for name, clf in [("naive", clf_naive), ("balanced", clf_bal)]:
    print(f"\n=== {name} ===")
    print(classification_report(y_te, clf.predict(X_te), digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_te, clf.predict(X_te)))
```

- **预期输出**：
  - naive 版 accuracy ~0.76，但违约类（1）的 recall 只有 ~0.40-0.45
  - balanced 版 accuracy 略降 ~0.72，但违约类 recall 升到 ~0.55-0.60
- **第一印象锚**：
  - "为什么 accuracy 反而降了还叫好" → 风控场景 **recall 比 accuracy 重要**：漏掉一个违约 = 坏账；拒绝一个好客户 = 利息损失（量级小得多）→ 引 §4.7 class_weight + §6.2 不平衡 + session-11 盲区里的 **F1 分数**闭环
  - 这是**讲清评估指标战场**的最佳场景，而且和 Alex 未来工作（风控/反欺诈栈常见）贴

#### 1.2.D 候选 D · Credit Card Fraud（炸机版）

```python
# lab/step1_minimal_ccfraud.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score

# 需要从 Kaggle 下载（需登录），本地放 data/creditcard.csv
df = pd.read_csv("data/creditcard.csv")
X, y = df.drop(columns=["Class", "Time"]), df["Class"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 关键钩子：accuracy 99.9%+ 但 recall 可能 80% 也可能 60%
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
y_proba = clf.predict_proba(X_te)[:, 1]

print(classification_report(y_te, clf.predict(X_te), digits=3))
print(f"PR-AUC: {average_precision_score(y_te, y_proba):.4f}")
# 这里要学员自己念出来：accuracy 0.999X 看着漂亮，但这是因为欺诈只有 0.17%
```

- **预期输出**：accuracy ~0.9995，PR-AUC ~0.80-0.85（欺诈 recall ~0.75-0.80，precision ~0.85-0.92）
- **第一印象锚**：
  - "99.95% accuracy 好像很强" → **全预测 0 也有 99.83%**（引 digest §4.2 反面教材）
  - 引 §6.2 + 评估指标必须看 PR-AUC / F1 / Recall

#### 1.2.E 候选 E · California Housing（回归变种）

```python
# lab/step1_minimal_cahousing.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
pred = reg.predict(X_te)
print(f"R²: {r2_score(y_te, pred):.3f}  MAE: ${mean_absolute_error(y_te, pred) * 100000:.0f}")
# 对照 demo-01 线性回归的 R² ≈ 0.6，RF 能到 0.8+
```

- **预期输出**：R² ~0.80-0.82（对照线性回归的 0.6），MAE ~\$31k-\$33k
- **第一印象锚**：
  - "同一份数据，RF 为什么吊打线性回归" → RF 捕捉**非线性 + 交互项**（地理位置 × 房间数的组合效应）；线性回归只能拟合直线
  - `RandomForestRegressor` vs `RandomForestClassifier` **一行切换**，回归版聚合方式是**取均值**而不是投票

---

## §2 · 递归缩放讲解（Alammar 黑盒逐层拆开）

### 2.1 黑盒视图 · 输入 → 输出

> **情境锚句**：先别管林子里 100 棵树各自在干啥。整个 RF 这个黑盒吃什么、吐什么。

**输入**：
- `X`：`(n_samples, n_features)` 的二维数组 / DataFrame
  - 可以混合类型（数值 + 类别 label encoded）
  - **不需要标准化 / 归一化**（对比 KNN 的致命预处理）
  - 可以有缺失（sklearn 1.4+ 原生支持 `NaN`；老版本需预先 `SimpleImputer`）
- `y`：`(n_samples,)` 标签
  - 分类：整数 / 字符串
  - 回归：浮点

**输出**（分类）：
- `.predict(X)` → `(n_samples,)` 类别标签
- `.predict_proba(X)` → `(n_samples, n_classes)` 每类概率（= 所有树的投票比例）
- `.feature_importances_` → `(n_features,)` 归一化重要性（加起来 = 1）
- `.oob_score_` → 单个 float（开启 `oob_score=True` 后）

**工程接口类比**：
- **微服务**：`POST /predict`，请求体带特征向量，响应带概率分布；SLA 可做到单样本 <5ms（CPU，100 树）
- **ONNX / PMML 导出**：RF 的结构（每棵树的分裂阈值 + 叶子值）可序列化，跨语言加载（Go / Java / Node 推理）
  - sklearn → ONNX：`skl2onnx.convert_sklearn`
  - sklearn → PMML：`sklearn2pmml`（金融风控常见）
- **模型体积**：100 树 + max_depth=None + 1000 样本 → 序列化后几 MB 到几十 MB；生产环境常设 `max_depth=10-20` 控制体积

### 2.2 拆一层 · 核心机制

> **情境锚句**：打开黑盒第一层——"一片林子"到底怎么训练出来的。**三步**：两重随机造树、独立训练、聚合投票。

**伪代码**（工程师语言）：

```
// 训练阶段（并行可到 n_estimators 线程）
def fit(X, y, n_estimators=100, max_features="sqrt"):
    forest = []
    for i in range(n_estimators):
        # 第一重随机：样本随机（bagging）
        X_bag, y_bag = bootstrap_sample(X, y)   // 有放回抽 n 条，约 63% 唯一样本
        oob_mask = get_unselected(X, X_bag)      // 剩下 ~37% 就是 OOB

        // 第二重随机：特征随机
        // 注意：不是训练时一次性砍特征，而是每个分裂节点独立抽 k 列再挑最优
        tree = DecisionTree(feature_sampler=sample_k_features(max_features))
        tree.fit(X_bag, y_bag)
        forest.append((tree, oob_mask))
    return forest

// 预测阶段
def predict(forest, X):
    votes = [tree.predict(X) for tree, _ in forest]   // 各树独立投票
    return majority_vote(votes)                         // 分类：多数派；回归：均值
```

**三个必讲点**：

1. **"两重随机"是 RF 名字里的核心**
   - 只 bagging 样本 = Bagged Trees（Breiman 1996，比 RF 早）
   - 只特征随机 = 无意义（样本都一样，方差不降）
   - 两重叠加 → 树之间**相关性大幅下降**（decorrelate） → bagging 削方差才真正生效
   - 见 [sklearn MOOC · Random Forests](https://inria.github.io/scikit-learn-mooc/python_scripts/ensemble_random_forest.html)

2. **树之间独立 = 训练可以彻底并行**
   - `n_jobs=-1` 用满所有核心
   - 对比 XGBoost/GBDT：后面的树依赖前面的残差 → **串行**，训练时间是 RF 的 N 倍
   - 工程类比：Bagging = ZooKeeper Quorum（并行）；Boosting = 责任链模式（串行）

3. **聚合方式分类 vs 回归不同**
   - 分类：`majority_vote`（`predict`）；或 `predict_proba` 返回所有树的投票比例平均
   - 回归：`np.mean`（每棵树回归预测的均值）
   - sklearn 分类内部用的是"概率平均 + argmax"而不是 hard voting（细节差异，但更稳）

**bootstrap 采样的 63% 数学**：
- 每次有放回抽 n 次，某条样本**没被抽中**的概率 = `(1 - 1/n)^n`
- `n → ∞` 时 → `1/e ≈ 0.368`（37% 没抽到 = OOB）→ 63% 被至少抽到一次（单棵树训练集）
- 这直接引出 §4.3 OOB 免费验证集

### 2.3 再拆一层 · 数学附录

> **情境锚句**：这一节是**下限答案**——Alex 追问公式时才展开，不主动讲。保留数学基础以备查。

#### 2.3.1 单棵决策树的分裂准则（CART）

- **Gini 不纯度**（分类默认）：
  $$\text{Gini}(S) = 1 - \sum_{k=1}^{K} p_k^2$$
  - $p_k$ 是节点 $S$ 中类别 $k$ 的占比
  - 完全纯（全是一类）→ Gini = 0；均匀分布（K 类各 1/K）→ Gini = 1 - 1/K

- **分裂增益**：选择让 `Gini(parent) - 加权(Gini(left) + Gini(right))` 最大的特征 + 阈值

- **Entropy** 替代（`criterion="entropy"`）：$-\sum p_k \log_2 p_k$，和 Gini 数值不同但分裂排序几乎一致，默认 Gini 更快

- **回归版**（`RandomForestRegressor`）：用方差或 MAE 替代 Gini
  $$\text{Variance}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y}_S)^2$$

- **一句话翻译**：每个分裂点都在问"切哪个阈值能让两边最纯"，然后每棵树这样切到不能切为止（或到 max_depth）

#### 2.3.2 Breiman 泛化误差上界（为什么加树不过拟合）

- Breiman 2001 原论文 [PDF](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) 证明：
  $$PE^* \leq \frac{\bar{\rho}(1 - s^2)}{s^2}$$
  - $PE^*$：森林的泛化误差
  - $\bar{\rho}$：树之间的**平均相关性**（越低越好）
  - $s$：单棵树的**强度**（strength，单树准确率 - 0.5）

- **工程直觉翻译**：
  - 相关性 $\bar{\rho}$ 越低（两重随机让它低） + 单树越强（$s$ 高）→ 上界越紧 → 森林越准
  - **加树（n_estimators ↑）不会让这个上界变差**——这就是"加树不过拟合"的数学来源
  - 但这**不等于 RF 整体不能过拟合**：如果 max_depth=None 让单树过拟合（$s$ 虚高但只是背了训练集）→ 森林照样继承 bias

- **一句话翻译**：加树只降方差不降偏差；偏差天花板 = 单树的偏差

#### 2.3.3 max_features 的经验默认值

- 分类：$\sqrt{d}$（sklearn 1.1+ 默认；老版本 `"auto"` = sqrt，已 deprecated）
- 回归：$d/3$（R 和 h2o 的默认；sklearn 默认改为 `1.0` 即全特征——和原论文不一致，需手动改 `max_features="sqrt"` 或 `1/3`）
- **直觉来源**：让任意强特征都有"不被选到"的概率，强制其他特征也参与分裂决策 → 树间去相关

- **学员追问下限答案**：
  - "为什么是 sqrt 不是 log" → 经验值，不是理论推导。Breiman 2001 试了 1 / $\log_2 d$ / $\sqrt{d}$ 几种，$\sqrt{d}$ 综合最好。现实中 Alex 拿默认值即可，调参时可以网格搜 [`log2`, `sqrt`, `0.3`, `0.5`]。见 [Lorentzen · Feature Subsampling for RF](https://lorentzen.ch/index.php/2021/08/19/feature-subsampling-for-random-forest-regression/)。

---

## §3 · 坏方案 → 好方案递进（CS231n 风格）

> **情境锚句**：让 Alex 先尝一口"只用一棵树会怎样"，再体会为什么要种一片林。

### 3.1 坏方案 · 单棵决策树

> **起点**：假设 Alex 已通关 demo-04 决策树。直接复用他的单树 demo 作 baseline。

**单棵树两头为难**：

| `max_depth` 设置 | 现象 | 问题 |
|---|---|---|
| `None`（长满） | Train 100% / Test 70% | **过拟合**：每条样本都有自己的叶子，噪声全背下来 |
| `max_depth=3`（砍浅） | Train 82% / Test 78% | **欠拟合**：特征交互全丢，复杂规律学不到 |
| 中间某个值 | 靠人调 | 依赖人工调参 + 数据依赖大，换个 fold 最佳深度就变 |

**单树的根本矛盾**：**方差大（深树）或偏差大（浅树）二选一**。

**还有一个问题**：**解释性陷阱**——一棵树看着"if-else 可解释"，但这棵树对数据噪声极度敏感，换个随机 seed / 换 fold 得到完全不同的树，"可解释"的是**偶然生成的这棵**，不是底层规律。

### 3.2 好方案 · RF 解决了什么 + 代价

**解决了什么**：
1. **方差削减**（核心）：
   - 单树方差 $\sigma^2$，$B$ 棵**完全独立**的树平均后方差 $= \sigma^2 / B$（大数定律）
   - 现实中树不是完全独立（训练数据有重叠），但两重随机让相关性 $\rho$ 尽量低
   - 实际方差 ≈ $\rho \sigma^2 + (1-\rho) \sigma^2 / B$ → $B$ 很大时趋近 $\rho \sigma^2$
2. **自带验证集**：OOB 给免费泛化估计，不用专门切 CV（见 §4.3）
3. **鲁棒性**：
   - 对异常值不敏感（分裂只看大小顺序）
   - 对特征尺度不敏感（**不用 scale**）
   - 对缺失值友好（sklearn 1.4+ 原生支持）
4. **可并行训练**：`n_jobs=-1` 直接线性加速

**代价（引入的新问题）**：
1. **失去单树的可解释性**：100 棵不同的树 → 没法画出一张"if-else 决策图" → 只能靠**聚合统计**（feature importance / SHAP）解释
   - 这也是为什么 §4.6 feature importance 陷阱这么重要——Alex 后续**只能靠它**去理解模型
2. **模型体积**：单树几 KB → 100 树几 MB-几十 MB；部署到手机 / 边缘设备前要 prune
3. **推理延迟**：单样本要遍历所有 100 棵树；虽然每棵树 O(log n) 很快，但常数变 100 倍
   - 生产经验：单样本 <5ms 没问题，万 QPS 要考虑树数 / 深度 tradeoff
4. **仍可过拟合**：虽然"加树不过拟合"，但**深度不限 + 噪声数据 + 树数不足**时会过拟合。见 [mljar · Does RF overfit?](https://mljar.com/blog/random-forest-overfitting/)
5. **不擅长外推**：
   - RF 预测值总在训练标签范围内（每个叶子是训练样本的均值/投票）
   - 训练数据价格 [10万, 200万] → 测试一个 300 万的豪宅 → 永远预测不出 200 万以上
   - 线性回归可以外推（`y = wx + b`），RF 不行
   - **选型信号**：预测值可能超出历史范围 → 选线性 / XGBoost（同样不外推但精度高）

### 3.3 Software 1.0 → 2.0 映射（第二跳）

> session-11 Alex 通关了"WHERE 过滤 vs 距离"第一跳。RF 是第二跳。

| 视角 | Software 1.0 | Software 2.0（以 RF 为代表的传统 ML） |
|---|---|---|
| 决策逻辑 | 人写 if-else 规则（风控 rule engine） | 数据驱动学习出一堆 if-else 的集成 |
| 可解释性 | 规则本身 = 解释 | feature importance / SHAP（事后解释） |
| 更新方式 | 改代码 + 重新发布 | 重新训练 + 模型文件替换（热更新） |
| 失败模式 | 规则覆盖不到的 case | 训练分布外的 case（distribution shift） |

**和 LLM 时代的对接**：
- LLM Agent 的"工具调用分类"常用 RF + 特征（prompt 长度 / 关键词 / 历史行为） 当 baseline
- 当 LLM 不可用 / 延迟敏感 / 需审计时，RF 是 fallback
- 参考 IBM 对 RF 工业定位的讨论：[IBM · What Is Random Forest](https://www.ibm.com/think/topics/random-forest)

---

## §4 · 术语卡片（每张独立自包含）

> **情境锚句**：这一章是 RAG 式卡片。每张卡片 Claude 可被单独召回 + 完整上下文。追问答案就近放。

### 4.1 集成学习 ensemble learning

- **中英对照**：集成学习 / ensemble learning
- **业务锚**：**委员会决策 / 多模型投票**——类比微服务架构里的"多实例 + 负载均衡 + 投票决议"
- **一句话定义**：把多个弱学习器（weak learners）组合成一个强学习器（strong learner）的方法论
- **在本算法里的角色**：RF 是集成学习的 **bagging 代表**（和 boosting / stacking 并列）

- **常见追问**：
  - **Q1**：集成学习有哪几种主流范式？
    - **下限答案**：三种——**Bagging**（并行 + 有放回采样 + 平权投票，RF 代表）、**Boosting**（串行 + 全量数据 + 残差拟合，XGBoost/GBDT 代表）、**Stacking**（基模型 + meta 模型两层，Kaggle 常用）
    - **深入答案**：还有 **Voting**（直接投票多个异构模型）、**Blending**（Stacking 的简化版，用 holdout 代替 CV）
    - **URL**：[Xoriant · Ensemble Methods](https://www.xoriant.com/blog/gradient-boosting-in-machine-learning-xgboost-lightgbm-random-forest-explained)
  - **Q2**：为什么一堆弱学习器合起来能强？
    - **下限答案**：两个条件——① **每个弱学习器准确率 > 50%**（比随机好）② **弱学习器之间**相对独立**（错不在同一个地方）。满足 → 投票后错误率按数量指数下降（Condorcet 陪审团定理）
    - **工程类比**：ZooKeeper Quorum，5 节点容 2 错；不独立（5 节点抄一份代码同时崩）= 白搭

### 4.2 Bagging / Bootstrap Aggregating（自助聚合）

- **中英对照**：**自助聚合** / Bagging = Bootstrap Aggregating
- **业务锚**：**有放回抽样 + 平权投票 = 民调式采样**——做民调时从总人口里重复抽（可能抽到同一人 2 次），每次样本独立做一份统计，最后平均
- **一句话定义**：一种并行的集成学习方法——用 bootstrap 采样造 B 个训练子集，每个子集独立训一个弱学习器，最后投票/平均
- **在本算法里的角色**：RF = Bagging + 特征随机（Bagging 只做样本随机，不做特征随机）

- **常见追问**：
  - **Q1**：Bootstrap 为什么叫 bootstrap？
    - **下限答案**：统计学借来的词。1979 年 Efron 提出 bootstrap 方法用于统计量的抽样分布估计——"有放回从样本里重复抽，模拟从总体抽"。英语 "pull yourself up by your own bootstraps"（靠自己）
  - **Q2**：Bagging 和 RF 的区别是什么？
    - **下限答案**：Bagging = 只样本随机；RF = Bagging **加上**特征随机。sklearn 里 Bagging 是 `BaggingClassifier`（可以用任意基模型），RF 是 bagging 专门用 DecisionTree + 特征随机的特化
    - **URL**：[sklearn · BaggingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)
  - **Q3**：Bagging 和 Boosting 什么区别？
    - **下限答案**：
      - Bagging = **并行** + **有放回采样** + **平权投票**，降方差
      - Boosting = **串行** + **全量数据** + **加权投票 / 残差拟合**，降偏差
    - **工程类比**：Bagging = ZooKeeper Quorum（并行投票）；Boosting = 责任链 / Chain of Responsibility（前面没处理好的交给下一个）
    - **URL**：[GeeksforGeeks · RF vs XGBoost](https://www.geeksforgeeks.org/machine-learning/difference-between-random-forest-vs-xgboost/)

### 4.3 OOB / Out-of-Bag（袋外样本 / 袋外误差）

- **中英对照**：**袋外样本** / Out-of-Bag / OOB
- **业务锚**：**CI/CD 里的 canary 流量** / **免费单元测试**——训练时天然剩下的部分当验证集，不用额外切 CV
- **一句话定义**：每棵树 bootstrap 采样时**没被抽到**的那部分样本（约 37%），用来估计该树的泛化误差；所有树的 OOB 结果聚合 → 无偏的森林泛化误差估计
- **在本算法里的角色**：RF / 所有 Bagging 方法**独有的免费验证机制**

- **代码开启**：
  ```python
  clf = RandomForestClassifier(oob_score=True)
  clf.fit(X_tr, y_tr)
  print(clf.oob_score_)   # 直接出单个数，类似 test accuracy
  ```

- **常见追问**：
  - **Q1**：为什么是 37%？哪来的？
    - **下限答案**：bootstrap 从 n 条样本有放回抽 n 次，某条样本**没被抽中**的概率 = `(1 - 1/n)^n`。$n \to \infty$ 时极限 = $1/e \approx 0.368$。见 [Wikipedia · OOB error](https://en.wikipedia.org/wiki/Out-of-bag_error)
  - **Q2**：OOB 能代替 CV 吗？
    - **下限答案**：**大多数场景可以**。Breiman 证明大样本下 OOB 渐近等于 leave-one-out CV，而且训练时顺带算 → 比 5-fold CV 省 5 倍时间。
    - 但有坑：OOB 是**逐样本**评估（某条样本只用"没见过它的树"预测），不是训练-测试分离；**极小样本量时**（n<100）OOB 估计方差大，还是老老实实 CV。
    - **URL**：[sklearn · OOB Errors](https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html) / [Analytics Vidhya · OOB Score](https://www.analyticsvidhya.com/blog/2020/12/out-of-bag-oob-score-in-the-random-forest-algorithm/)
  - **Q3**：OOB score 和 test score 应该差多少？
    - **下限答案**：**基本贴合**（±1-2% 正常）。差距大 → 要么数据分布有 shift（train/test split 不随机）要么 OOB 没收敛（n_estimators 太小）

### 4.4 max_features / 特征随机采样

- **中英对照**：**最大特征数** / max_features
- **业务锚**：**每次只看部分字段做决策**——类比分布式数据库里的"每个分片只索引部分字段"，避免所有分片都挤着用同一个主键
- **一句话定义**：每个**分裂节点**独立随机抽 k 个特征（k = max_features），从这 k 个里挑最优分裂；不是训练前一次性砍特征
- **在本算法里的角色**：RF 名字里的**第二重随机**。没有它 = Bagged Trees（效果差）

- **默认值**：
  | 算法 | 分类默认 | 回归默认 |
  |---|---|---|
  | sklearn RF | `"sqrt"` | `1.0`（全特征，和原论文不一致） |
  | R randomForest | `sqrt(p)` | `p/3` |
  | Breiman 原论文推荐 | `sqrt(p)` | `p/3` |
  - **sklearn 回归默认是全特征**——这是历史遗留，**工程师要手动改 `max_features=1/3` 或 `"sqrt"`**，否则和原论文不一致

- **常见追问**：
  - **Q1**：为什么是 sqrt？
    - **下限答案**：经验值不是理论。Breiman 2001 试了 1 / $\log_2 d$ / $\sqrt{d}$ 几种，$\sqrt{d}$ 综合最好。拿默认即可。
  - **Q2**：为什么要"每个节点"独立抽，不是"每棵树"抽一次？
    - **下限答案**：每棵树一次性砍 → 某些特征在某棵树里永远不出现，贡献被低估；每个节点抽 → 所有特征都有机会在某个分裂点出现。后者样本效率更高。
  - **Q3**：设多少合适？
    - **下限答案**：默认（`sqrt`）基本够用。调参时试 `["sqrt", "log2", 0.3, 0.5]` 网格。特征数很少（d<10）时可能全特征反而好（因为 sqrt(10)=3 太少）。见 [Lorentzen · Feature Subsampling](https://lorentzen.ch/index.php/2021/08/19/feature-subsampling-for-random-forest-regression/)

### 4.5 n_estimators / 树数 + 饱和点

- **中英对照**：**树的数量** / n_estimators
- **业务锚**：**集群规模 / 投票人数**——过少投票不稳定，过多边际递减
- **一句话定义**：森林里决策树的数量；Bagging 的 B
- **在本算法里的角色**：**最重要的超参之一**（和 max_features + max_depth 并列）

- **工程经验值**：
  - 默认 100
  - 结构化数据一般 200-500 够用
  - **加树不过拟合**（Breiman 证明），但训练时间 + 模型体积线性增长
  - **有饱和点**——画 `n_estimators vs OOB score / CV score` 曲线找拐点

- **常见追问**：
  - **Q1**：怎么知道饱和点？
    - **下限答案**：
      ```python
      scores = []
      for n in [50, 100, 200, 400, 800]:
          clf = RandomForestClassifier(n_estimators=n, oob_score=True, warm_start=True).fit(X, y)
          scores.append(clf.oob_score_)
      # 画图找斜率趋近 0 的拐点
      ```
    - `warm_start=True` + 逐次 +N 避免从头训
  - **Q2**：500 还没饱和怎么办？
    - **下限答案**：**问题多半不在 n_estimators**——要么特征工程不够（加特征）要么单树太浅（提 max_depth）要么数据噪声大（ensemble 无能为力）。硬加树 → 边际收益可忽略 + 模型文件变大
    - **URL**：[Crunching the Data · Number of trees](https://crunchingthedata.com/number-of-trees-in-random-forests/)

### 4.6 feature_importances_ / 特征重要性（MDI vs Permutation）

- **中英对照**：**特征重要性** / feature importance
- **业务锚**：**埋点数据分析里的"贡献度排名"**——告诉你哪些字段对决策最关键，类比产品里的"核心漏斗归因"
- **一句话定义**：每个特征对模型预测的贡献度评分（归一化，和为 1）
- **在本算法里的角色**：**RF 的主要可解释性出口**——因为失去了单树可解释性，feature importance 是 Alex 后续最常用的"给老板解释"工具

- **⚠️ 两种算法 + 陷阱**：

| 方法 | 原理 | 优点 | 缺陷 |
|---|---|---|---|
| **MDI / Mean Decrease in Impurity**（sklearn 默认 `feature_importances_`） | 累加每次分裂的 Gini 增益 | 免费（训练时顺带算） | ① **偏向高基数特征**（连续数值 / 高唯一值类别会被系统性高估）② **训练集计算**，过拟合时噪声特征也刷高分 |
| **Permutation Importance**（`sklearn.inspection.permutation_importance`） | 在测试集上把某列打乱，看 score 掉多少 | ① 测试集评估，反映泛化贡献 ② 适用任何模型（不只 RF） | ① 慢（每特征 shuffle N 次）② 相关特征互相掩盖（permute A 时 B 还能撑住） |

- **用哪个**：
  - 快速探索 → MDI
  - **上线前 / 报告 / 风控审计 → 必须用 permutation importance**
  - 真要严谨 → SHAP（`TreeSHAP` 对 RF 有专用高效算法）

- **常见追问**：
  - **Q1**：为什么 MDI 偏向高基数？
    - **下限答案**：高基数列（连续数值 / 高唯一值 ID）每次分裂都能劈出纯节点，基尼增益累加爆表——**即使它毫无泛化价值**（极端例子：用 user_id 做特征，MDI 会排第一，但换个数据集 user_id 完全变了，毫无用）
    - **URL**：[explained.ai · Beware Default RF Importances](https://explained.ai/rf-importance/)（业界权威吐槽文）
  - **Q2**：我的 feature importance 排序看着合理，还需要换 permutation 吗？
    - **下限答案**：**合理≠正确**。"合理"可能只是你先验偏见对上了。permutation 成本就几秒，默认都跑一下对比：如果两者 top-5 重合 → 放心用 MDI；不重合 → 一定是 MDI 出 bug
  - **Q3**：有 ID 列 / 高基数列怎么办？
    - **下限答案**：**删掉**（训练时）。ID 列永远不应该进模型（信息泄漏 + MDI 炸）；高基数类别列（邮编 / 城市）用 target encoding / frequency encoding 替代原始 label encoding
    - **URL**：[sklearn · Permutation vs MDI](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html)

### 4.7 class_weight / 类别权重（不平衡处理）

- **中英对照**：**类别权重** / class_weight
- **业务锚**：**风控 / 反欺诈 / 医疗诊断** 常见——坏客户 / 欺诈 / 病例往往是少数类，默认训练会被淹没
- **一句话定义**：训练时给少数类样本更高权重，让模型"更在乎"漏掉少数类的惩罚
- **在本算法里的角色**：不平衡数据场景的**首选工具**（比重采样 SMOTE 简单 + 不改数据分布）

- **三种用法**：
  ```python
  # 1. 自动平衡（最常用）
  RandomForestClassifier(class_weight="balanced")
  # 等价于 weight[k] = n_samples / (n_classes * np.bincount(y)[k])

  # 2. 自动平衡 + 每棵树基于 bootstrap 子集
  RandomForestClassifier(class_weight="balanced_subsample")
  # 每棵树独立算 weight，对极度不平衡更稳

  # 3. 手动
  RandomForestClassifier(class_weight={0: 1, 1: 10})   # 类别 1 权重 10x
  ```

- **常见追问**：
  - **Q1**：class_weight 和 SMOTE 选哪个？
    - **下限答案**：
      - class_weight：**首选**。不改数据，只改 loss。简单 + 快 + 不引入合成样本的偏差
      - SMOTE（`imbalanced-learn` 库）：class_weight 效果不够时才用。合成样本可能跨越真实边界导致 noise
      - 极度不平衡（<1%）：两者叠加常见
    - **URL**：[Nature · RF + SMOTE for Fraud (2025)](https://www.nature.com/articles/s41598-025-00873-y)
  - **Q2**：为什么 balanced 比手动更常用？
    - **下限答案**：balanced 公式里有 `n_classes` 归一化，换数据集不用重调；手动写死 `{0:1, 1:10}` 下次数据分布变了就要手调

### 4.8 Boosting vs Bagging · 集成范式对比

- **业务锚**：**Quorum 集群（并行投票）vs 责任链（串行补救）**
- **为什么放在 RF 的 LESSON-PLAN 里**：下节课（06 XGBoost）就讲 boosting，这里先铺认知锚

| 维度 | Bagging（RF 代表） | Boosting（XGBoost 代表） |
|---|---|---|
| **训练方向** | 并行（树独立） | 串行（后面拟合前面残差） |
| **采样** | 有放回 bootstrap | 全量（可能加权） |
| **聚合** | 平权投票 / 均值 | 加权投票（表现好的树权重高） |
| **主要降** | 方差 variance | 偏差 bias |
| **基学习器要求** | 高方差 + 低偏差（深树） | 低方差 + 可接受偏差（浅树，`max_depth=3-8` 常见） |
| **默认性能** | 开箱 = 调优后 97-99% | 开箱 = 调优后 90-95% |
| **调优空间** | 小 | 大（精度天花板高 2-5%） |
| **可并行训练** | ✅ 天然 | ❌ 串行（但单树内可并行） |
| **噪声鲁棒性** | ✅ 强 | ⚠️ 弱（容易过拟合噪声） |
| **数据量** | 中小 + 噪声大 | 大 + 干净 |

- **选型决策（给 Alex 的 cheatsheet）**：

  ```
  时间紧 / 第一版 baseline     → RF
  噪声大 / 样本少                → RF
  可解释性 / 风控监管            → RF + SHAP
  训练多核可用但推理可等         → RF
  ---
  大数据 + 愿意调参              → XGBoost / LightGBM
  十亿级 CTR / 在线特征          → LightGBM
  GPU 训练                       → XGBoost
  ```

- **URL**：[apxml · XGBoost vs LightGBM vs CatBoost](https://apxml.com/posts/xgboost-vs-lightgbm-vs-catboost) / [MCP Analytics · XGBoost vs RF](https://mcpanalytics.ai/articles/xgboost-vs-random-forest-comparison)

### 4.9 eager learner vs lazy learner（对照 KNN）

- **业务锚**：**构建时编译 vs 运行时解释**——RF 训练时把决策逻辑"编译"好（每棵树结构定死），预测快；KNN 训练时啥都不做，预测时才全表扫描算距离
- **为什么放这**：session-11 挂了"KNN 是 lazy learner"术语没展开，RF 是天然对比点

| 维度 | Eager Learner（RF） | Lazy Learner（KNN） |
|---|---|---|
| 训练成本 | 高（训 100 棵树） | 近 0（只存训练集） |
| 预测成本 | 低（遍历 100 棵树 O(log n)） | 高（算 n 个距离 + 排序） |
| 模型体积 | 几 MB-几十 MB（树结构序列化） | = 训练集大小（要带全量） |
| 部署 | 一次训练，多次推理 | 预测时必须带训练集 → 大数据量 KNN 要换向量数据库 |
| 增量更新 | 要重训（或 online RF 变种） | 直接加样本进训练集 |

---

## §5 · Questionnaire · 显式 active recall

> **情境锚句**：这一章是闭卷题库，按 Bloom 三层。Claude 上课时按 Alex 节奏抽题。评分标准见 [`01-LESSON-PLAN.md`](../00-mental-model/01-LESSON-PLAN.md) + [`00-grading-rules.md`](../../00-grading-rules.md)。

### 5.1 记忆层（能复述）

1. 随机森林里的"两重随机"分别是哪两重？各自解决什么问题？
   - **标答**：① **样本随机**（bagging，bootstrap 有放回采样）→ 降方差 ② **特征随机**（max_features，每个分裂节点独立抽 k 列）→ 去相关化（decorrelate）
2. bootstrap 采样里，每棵树大约有多少比例的样本**没被抽到**？这些样本叫什么？
   - **标答**：约 37%（$1/e \approx 0.368$），叫 **OOB（Out-of-Bag）袋外样本**
3. RF 的分类和回归，预测聚合方式分别是什么？
   - **标答**：分类 = 多数投票（sklearn 实际用概率平均 + argmax）；回归 = 均值
4. `max_features="sqrt"` 是分类的默认。为什么不是全特征？
   - **标答**：全特征 → 每棵树第一刀都切最强特征 → 树间高度相关 → bagging 只剩重复投票没有多样性，方差削减失效
5. 加树（n_estimators ↑）会不会导致过拟合？为什么？
   - **标答**：不会（Breiman 证明 generalization error 随树数收敛到上界）。但**深度不限 + 单树过拟合 + 树数不足**时整个 RF 还是会过拟合——加树只降方差不降偏差，偏差天花板 = 单树的偏差

### 5.2 应用层（能选型 / 能读代码）

1. 给定一份 1000 × 20 的德国信贷数据，类别比例 7:3，下面这段代码有什么**必改**的地方？
   ```python
   clf = RandomForestClassifier(n_estimators=100).fit(X_tr, y_tr)
   print("accuracy:", clf.score(X_te, y_te))
   ```
   - **标答**：① `class_weight="balanced"`（不平衡场景）② 开 `oob_score=True` 省 CV ③ `n_jobs=-1` 用满核心 ④ 评估不能只看 accuracy，要看违约类 recall + F1 + PR-AUC
2. 为什么 RF 可以不做 StandardScaler，但 KNN 必须做？
   - **标答**：KNN 算距离，特征量纲直接影响距离值（一列 0-1 + 一列 0-10000 → 距离完全被第二列主宰）；RF 每棵树**只看单列分裂**，只在乎大小顺序不在乎量纲（分裂阈值会自动适配）
3. `feature_importances_` 排第一的是 `user_id`，合理吗？
   - **标答**：**不合理**。ID 列是超高基数类别，MDI 算法天然偏向高基数特征——每次分裂都能劈出纯节点 → 累计基尼增益爆表，但毫无泛化价值。修复：① 删 ID 列（根本不该进模型）② 对其他特征用 permutation importance 重算
4. 同一份数据 RF 训出来 OOB = 0.85，test = 0.72，差距这么大说明什么？
   - **标答**：**数据分布 shift** —— train/test split 不随机，或 test 是未来时段的数据。OOB 是 train 内部的估计，反映不了分布偏移。修复：① 检查 split 逻辑（时序数据用 time-based split）② 看 train/test 特征分布差异（KS 检验）
5. 我要做一个"RF 分类 + 在线推理 SLA <10ms" 的服务，该调什么参数 + 该导出什么格式？
   - **标答**：
     - 参数：`n_estimators=100-200`（别堆到 500+）+ `max_depth=15-20`（砍掉极深路径）+ `min_samples_leaf=5`（叶子别太碎）
     - 导出：**ONNX**（`skl2onnx.convert_sklearn`）→ Go/Java/Node 都有 onnxruntime；或 **PMML**（金融风控常见）
     - 缓存：特征预处理（encoding/填充）建 pipeline，一起序列化

### 5.3 迁移层（新场景 / 对比）

1. 一个"物品冷启动推荐"业务，没有用户历史行为序列，只有用户画像（年龄/性别/城市/注册来源）+ 物品属性（类目/价格/标签）。选 RF 还是 Transformer？为什么？
   - **标答**：**RF**。① 冷启动没历史行为 → Transformer 缺序列输入 ② 数据量通常不大（冷启动期） → RF 开箱即用 ③ 需要快速上线 + 可解释（产品经理要看"为什么推 X 给 Y"）→ RF + feature importance 够用。等数据量上来 + 有序列信号后再切 Transformer
2. 和 demo-03 KNN 相比，RF 和 KNN 分别适合什么场景？给 3 个选型维度。
   - **标答**：
     | 维度 | RF 赢 | KNN 赢 |
     |---|---|---|
     | 特征 | 有特征工程 / 混合类型 | 已有 embedding / 纯语义 |
     | 可解释 | feature importance 可用 | 不好解释（但可给"你的最近邻"当解释） |
     | 规模 | 中等（万到百万） | 小（千级） 或 有向量数据库（HNSW） |
     - **典型场景**：风控 / 推荐 baseline → RF；RAG / 语义搜索 → KNN + pgvector
3. 接到需求"预测未来 6 个月某豪宅价格"，历史数据价格范围 [10万, 200万]，目标房价可能 >300万。RF 合适吗？
   - **标答**：**不合适 / 至少要警告**。RF 不擅长外推——预测值永远在训练标签范围内（每个叶子是训练样本均值）→ 永远吐不出 200万以上。此场景选：① 线性回归 / Ridge（可外推）② 特征工程（log 转换目标）③ Gradient Boosting（同样不外推但精度高） + 业务上 cap 住
4. 下面哪个场景我该用 class_weight 而不是 SMOTE？反过来呢？
   - **标答**：
     - 用 **class_weight** 的场景：欺诈率 0.17%（极度不平衡，合成样本会造噪声） / 模型是 RF 这种原生支持 weight 的
     - 用 **SMOTE** 的场景：中等不平衡（5-15%） + 模型不支持 weight（某些黑盒）+ 样本量小到 class_weight 效果不够 → 用合成样本人为扩充
5. 业务场景：LLM Agent 做"用户意图分类"——已有 LLM 方案（准确率 92% 但延迟 800ms + 每次 $0.003），要做一个轻量 fallback。RF 能不能干？怎么搭？
   - **标答**：**能**。搭法：
     - 特征：prompt 长度 + TF-IDF top-100 + 用户历史行为统计（过去 7 天意图分布）
     - 模型：RF 200 树 / `class_weight="balanced"`（意图分布通常不均）
     - 评估：per-class F1（每个意图都要准，不只是整体 accuracy）
     - 路由：`p_max = max(proba); if p_max > 0.85 → 用 RF 结果; else → 走 LLM`
     - 部署：ONNX + Go 推理，延迟 <5ms
     - 监控：LLM 和 RF 结果不一致时记日志 → 持续再训

---

## §6 · 常见坑 / 反面教材

> **情境锚句**：Alex 已经犯过 / 必然会犯的错误。备课时 Claude 要提前埋伏。

### 6.1 feature_importance 被高基数特征假冠军

- **现象**：某团队用 RF 做用户流失预测，`feature_importances_` 显示 `user_id` 排第一。
- **真相**：user_id 是高基数分类列，MDI 天然偏向——每次分裂它都能劈出纯节点，累计基尼增益爆表，但**毫无泛化价值**。
- **修复**：
  1. ID 类列**绝不该进模型**（信息泄漏）——训练前 drop
  2. 保留的所有特征都用 `sklearn.inspection.permutation_importance` 对照 MDI
  3. 在**测试集**上算 permutation（训练集算 → 过拟合时噪声也刷高分）
- **URL**：[explained.ai · Beware Default RF Importances](https://explained.ai/rf-importance/) / [sklearn · Permutation vs MDI](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html)

### 6.2 不平衡数据 accuracy 虚高（信用卡欺诈场景）

- **现象**：Credit Card Fraud 数据集欺诈比例 0.17%。RF 默认参数直接训，accuracy 99.8%+——老板看 dashboard 表示满意。
- **真相**：模型把几乎所有样本预测为"非欺诈"，欺诈类 recall 接近 0%。一个欺诈都没抓到。
- **修复**：
  1. 训练层：`class_weight="balanced"` 或叠加 SMOTE（`imbalanced-learn`）
  2. 评估层：**换指标**——PR-AUC / F1 / per-class recall，**禁止单看 accuracy**
  3. 业务层：看 confusion matrix 看绝对数（漏了多少欺诈 = 多少钱）
- **URL**：[Nature · RF+SMOTE for CCFraud (2025)](https://www.nature.com/articles/s41598-025-00873-y) / [Keylabs · Handling Imbalanced Data](https://keylabs.ai/blog/handling-imbalanced-data-to-improve-precision/)

### 6.3 数据泄漏（训练 0.93 上线 0.62）

- **现象**：train AUC 0.95 / test AUC 0.93 / 上线 AUC 0.62。
- **真相**：特征里混入了**未来信息**——例如 "用户是否投诉" 这列训练时用事后标注，但**预测时间点**拿不到。RF 对噪声相关特征不敏感不代表对泄漏不敏感——有泄漏它照样会学，而且学得特别好（误导性最高）。
- **修复**：
  1. 所有特征都问一句"**预测时间点**这列能拿到吗"
  2. 时序数据用 **time-based split**，不是随机 split
  3. 检查高相关特征（`|corr(X_i, y)| > 0.9`）——往往是泄漏信号
  4. 预处理 scaler / encoder **包进 pipeline 后再进 CV**，避免 preprocessing leakage
- **URL**：[Springer · Data Leakage Risks (2025)](https://link.springer.com/article/10.1007/s10462-025-11326-3) / [ML Mastery · Data Leakage](https://machinelearningmastery.com/data-leakage-machine-learning/)

### 6.4 `max_features="auto"` 老代码迁移坑

- **现象**：老代码（sklearn 1.0 或更早）里写 `max_features="auto"` → 1.3+ 升级后报 DeprecationWarning，2.0 会直接报错。
- **真相**：`"auto"` 在分类中等价 `"sqrt"`，在回归中等价 `1.0`（全特征）——改名只是统一。但**回归情况从"全特征"改 `"sqrt"` 会改变行为**。
- **修复**：
  - 分类：`max_features="sqrt"`（行为不变）
  - 回归：确认历史行为——如果之前 `"auto"` 实际跑的是全特征，升级后写 `max_features=1.0`；如果要按 Breiman 原论文推荐改 `"sqrt"` 或 `1/3`，是**故意行为变更**
- **URL**：[sklearn 1.8 RandomForestClassifier 文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

### 6.5 跨语言推理踩坑（Python 训 → Go 推）

- **现象**：sklearn 训好 RF，保存 pickle，Go 服务加载 → 加载失败 / 结果不一致。
- **真相**：
  - pickle 是 Python 专属，Go 不认
  - pickle 还有版本问题（sklearn 1.2 训的 pickle 在 1.5 里可能炸）
- **修复**：
  - 推荐 **ONNX**：`pip install skl2onnx` → `convert_sklearn(clf, initial_types=...)` → Go 用 onnxruntime-go 加载
  - 金融场景：**PMML**（`sklearn2pmml`）+ Java PMML4.x 生态成熟
  - 特征预处理也要一起导出（`Pipeline(scaler, encoder, clf)` 整体 convert）——单独导出模型，Go 侧手写预处理极容易对不齐

### 6.6 回归默认 max_features=1.0 的历史坑

- **现象**：RandomForestRegressor 效果平平，改 `max_features="sqrt"` 或 `1/3` 后提升明显。
- **真相**：sklearn 回归默认 `max_features=1.0`（全特征）——和 Breiman 原论文 / R randomForest 的 `p/3` 不一致。全特征 = 退化成 Bagged Trees，去不了相关。
- **修复**：**回归场景默认手动设 `max_features="sqrt"` 或 `0.33`**，除非你特征特别少（d<10）或网格搜实测全特征更好

---

## §7 · 数据流图 · 跑完 demo 后补

> **情境锚句**：本节为占位。`DATA-FLOW.md` 是跑完 demo 的临时产出物，不在模板起手范围。

详见 [`DATA-FLOW.md`](./DATA-FLOW.md)（跑完 demo 后创建）

---

## §8 · 调度档位（本节课怎么上）

> **情境锚句**：这一章给 Claude 上课时按 Alex 能量档位 pick 一档执行。

### 8.1 90 分钟深度档（推荐首次）

| 阶段 | 时长 | 内容 |
|---|---|---|
| 热身 | 5min | session-11 自我热身题回顾 + 引出"一棵树不够怎么办" |
| §1 业务锚 + 跑 demo | 15min | Quorum 类比 + German Credit 跑通 + 看 OOB / accuracy / feature importance |
| §2 核心机制 | 20min | 两重随机 + 伪代码 + bootstrap 37% 数学 |
| §3 坏→好递进 | 15min | 单树过拟合对照 + RF 解决 + 代价（不外推 / 模型大） |
| §4 术语焊 | 20min | OOB / max_features / feature importance（含 MDI 陷阱） / class_weight |
| §5 Questionnaire 抽题 | 10min | 记忆 2 题 + 应用 1 题 + 迁移 1 题 |
| 收尾 | 5min | 当日盲区总结 + 下次起点 |

### 8.2 30 分钟复习档（后续 session）

| 阶段 | 时长 | 内容 |
|---|---|---|
| 热身 | 3min | 5 道 §5.1 记忆题闪问 |
| 盲区靶向 | 15min | 根据 session 报告盲区，抽 1-2 个术语卡深讲 |
| 迁移题 | 10min | §5.3 抽 1 题做独立选型判断 |
| 收尾 | 2min | 定级 + 下次起点 |

### 8.3 架构对话档（高能量时）

不走题库。放开谈：
- "LLM Agent 栈里 RF 作 fallback 的架构决策点"
- "风控从 rule engine → RF → XGBoost → LLM 的演进路径"
- "RF + SHAP 做可审计 AI 的工程设计"

自由展开，Claude 跟进追问 + 挖 L3 判断。

### 8.4 session-11 遗留盲区闭环点

| 盲区 | 在本 LESSON-PLAN 的闭环位置 |
|---|---|
| F1 分数 | §1.2.C German Credit demo + §4.7 class_weight + §6.2 不平衡反面教材 |
| Lazy learner vs Eager learner | §4.9 对照卡 |
| Metadata filtering 选型 | §5.3 第 5 题（LLM 意图分类 fallback） |
| L3 迁移验证题 | §5.3 全部 5 题 |

---

## §9 · 三源定位（本节素材来自哪）

> **情境锚句**：三份素材分别填不同槽位，缺一源则该节含金量打折。详细 URL 全量表见 [`05-random-forest-digest.md`](./05-random-forest-digest.md) §7。

### 9.1 三源映射

| 本 LESSON-PLAN 章节 | 主要来源 | 补充来源 |
|---|---|---|
| §1.1 业务锚（Quorum） | **第一版改编** `demos/06-Ensemble-and-Financial-Risk.md` §2 Bagging | 联网（IBM / Xoriant 的工业定位文） |
| §1.2 最小 demo | 原始培训笔记 `集成学习.md` 泰坦尼克 demo | 联网（UCI / Kaggle 数据源 URL） |
| §2.1 黑盒视图 | 第一版改编（工程接口类比） | — |
| §2.2 核心机制（两重随机） | **原始培训笔记** §"随机森林" 构建过程 | 联网（sklearn MOOC 为什么特征随机能 decorrelate） |
| §2.3 数学附录（Gini / Breiman 公式） | **原始培训笔记** + **Breiman 2001 原论文** | 联网（alexmolas / mljar 对"加树不过拟合"的工程澄清） |
| §3 坏→好递进 | 第一版改编（Software 1.0 vs 2.0） | 联网（RF 不外推 / Kaggle 2025 定位） |
| §4 术语卡定义 | **原始培训笔记**（严谨中文定义） | 第一版改编（业务锚素材） |
| §4 术语追问答案 | — | **联网**（必带 URL，经得起"你哪看来的"反问） |
| §5.1 记忆题 | 原始培训笔记课后练习 | — |
| §5.2 应用题 + §5.3 迁移题 | 第一版改编（工程场景） | 联网（Kaggle 冠军方案 / LLM Agent 架构） |
| §6 反面教材 | 原始笔记"注意"提示 + 改编版跨语言落地坑 | **联网**（explained.ai MDI 陷阱 / Nature 2025 CCFraud / Springer 2025 Data Leakage） |

### 9.2 本节**必须命中**的外部 URL（备课时已嵌入正文）

- **原论文 / 权威源**：
  - [Breiman 2001 · Random Forests PDF](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
  - [sklearn RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
  - [sklearn MOOC · RF](https://inria.github.io/scikit-learn-mooc/python_scripts/ensemble_random_forest.html)
- **MDI 陷阱**：
  - [explained.ai · Beware Default RF Importances](https://explained.ai/rf-importance/)
  - [sklearn · Permutation vs MDI](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html)
- **OOB**：
  - [sklearn · OOB Errors](https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html)
  - [Wikipedia · OOB error](https://en.wikipedia.org/wiki/Out-of-bag_error)
- **不平衡 / 反面教材**：
  - [Nature · RF+SMOTE for CCFraud (2025)](https://www.nature.com/articles/s41598-025-00873-y)
- **数据泄漏**：
  - [Springer · Data Leakage Risks (2025)](https://link.springer.com/article/10.1007/s10462-025-11326-3)
- **2024-2025 工业定位**：
  - [ML Contests · State of ML Competitions 2024](https://mlcontests.com/state-of-machine-learning-competitions-2024/)
  - [NVIDIA · Kaggle Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)

**完整 URL 清单（45+ 条）** 见 [`05-random-forest-digest.md`](./05-random-forest-digest.md) §7.3。

---

## 附：迭代记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-04-24 | v0.1 试行 | 首版。基于 `_shared/LESSON-PLAN-TEMPLATE.md` v0.2 + `05-random-forest-digest.md` L1 素材包三源融合。调度档位 §8 沿用 demo-03 KNN 的 90/30/架构三档结构。 |
