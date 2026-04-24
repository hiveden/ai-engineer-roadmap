# LESSON-PLAN · 第 1 节 · KNN 最近邻算法（demo-03）

> **🧪 本文档是模板 [`demos/_LESSON-PLAN-TEMPLATE.md`](../_LESSON-PLAN-TEMPLATE.md) v0.2 试行方案的第一个实例**。
>
> 备课资料库，不是教案。给 Claude 上课时查，不是给学员自读。
> 写作规则见模板 §8。

> 📎 **大纲接收**（2026-04-24 决策 · 同日按 day-02 KNN 大纲补充）：
> - **§6 归一化 Normalization / 标准化 Standardization**——Wine 数据集必做标准化，距离算法对量纲敏感
> - **§7 欠拟合 / 过拟合 / 泛化（首讲）**——顺线下课 day-02 KNN 大纲 §5，K 值过小 → 过拟合 / K 值过大 → 欠拟合
>
> 详见 [`../00-mental-model/00-机器学习概述-学习安排.md`](../00-mental-model/00-机器学习概述-学习安排.md) + [`../00-mental-model/01-LESSON-PLAN.md` §8.4](../00-mental-model/01-LESSON-PLAN.md)。

---

## §0 · 课前须知

### 0.1 这节课的位置

| 维度 | 值 |
|---|---|
| 算法 | **KNN / K-Nearest Neighbors / K 近邻** |
| 可支持的任务类型 | **分类**（主流）+ **回归**（均值）。不是聚类——K-Means 才是聚类 |
| roadmap 阶段 | 阶段一 · 传统 ML 工程化 · 第 1 节（按 README.md 清单），**实际顺序是 demo-01/02 之后的 demo-03** |
| 目标深度（见 [`demos/README.md`](../README.md)） | **L3** — 能讲清楚每步动机 + 能做选型判断，不要求手写代码 |
| 在 10 节课里 | 第 1 节（按算法清单）/ 第 3 个 demo（按 Alex 实际学习顺序） |
| 推荐前置（非强制） | demo-02 逻辑回归（已用过分类评估指标 P/R/F1，此处**不重讲**） |

> ⚠️ **不在此处定死"数据集"**。数据集选型是上课时的现场决策，见 §0.3。

### 0.2 当前学习状态（2026-04-20 快照，来自 session-09）

> **情境锚句**：这一节开始时，Alex 已经通关了 demo-01 线性回归 + demo-02 逻辑回归基础评估，正在理解"业务决策链 → 指标选型 → 阈值方向"。KNN 是第一次接触无参数 / lazy 模型，也是转向"向量检索 / RAG"的桥梁算法。如果回炉重讲 Accuracy/Precision/Recall 家族 = 浪费时间。

**已 Level 2+ 通关**：
- 监督学习 vs 无监督（A 组 L2，session-04）
- 分类 vs 回归（L2→L3 边缘，session-09）
- 过拟合 / 泛化 / 置信度（L2，session-07）
- 训练/测试划分 + 标准化必要性的**概念存在感**（demo-02 step2 / step3 带过，但没手跑过 StandardScaler）
- 决策阈值 threshold（L2，session-09 新增）
- 分类评估家族 P/R/F1/混淆矩阵（F1 仍 L1）

**本节课的起点**：
- 还没跑过任何 KNN 代码
- 没碰过 GridSearchCV
- 没碰过 StandardScaler（demo-02 用的是 LogisticRegression 加大 max_iter 绕过，不是标准化）
- 没碰过 `fit_transform` vs `transform` 的区别
- **F1 可以在本节 §5 Questionnaire 或 §3 评估时顺便闭环**（session-09 盲区）

### 0.3 候选数据集清单（备课覆盖全量，上课现场选）

> **情境锚句**：这一章列出 KNN 教学所有**可选**数据集——备课时不预选，上课时 Claude 根据学员状态 pick。每个候选带教学价值对比 + sklearn 官方 URL。

| 候选 | 规模（行×列）| 特点 | 教学价值（能讲什么钩子）| 适合上课场景 | 数据源 URL |
|---|---|---|---|---|---|
| **Iris 鸢尾花** | 150 × 4 | 3 分类，全数值，无缺失，低维线性可分 | 算法本体 + **多数投票**（K=3/5 票数对比一目了然）；可画二维决策边界 | 零基础 / 第一次见 KNN / 注意力低 | [sklearn load_iris](https://scikit-learn.org/stable/datasets/toy_dataset.html) |
| **Wine 葡萄酒** | 178 × 13 | 3 分类，**量纲差异巨大**（proline 上千，灰分 <3） | **标准化对 KNN 致命**——不做 scaling 直接训练可当场复现"精度崩盘" | 想体会 scaler 威力 / 铺 §4 术语卡 | [sklearn load_wine](https://scikit-learn.org/stable/datasets/toy_dataset.html) |
| **Digits 手写数字子集** | 1797 × 64 (8×8 像素) | 10 分类，MNIST 子集 | **维度诅咒入门**（64 维已开始退化）；**规模灾难预热**（预测耗时可 benchmark）| 想过渡到 ANN / Vector DB / RAG 的锚点 | [sklearn load_digits](https://scikit-learn.org/stable/datasets/toy_dataset.html) |
| **California Housing** | 20640 × 8 | **回归任务**，房价中位数，含经纬度 | **KNN 不只是分类器**——回归用 k 邻居均值；2 万行能体感 brute-force 变慢 | 第二轮 / 讲"为什么需要 ANN" | [sklearn fetch_california_housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) |
| ~~Breast Cancer~~ | ~~569 × 30~~ | ~~二分类~~ | ~~demo-02 已用，不建议复用~~ | ~~仅当要对比 KNN vs 逻辑回归时~~ | — |

**现场选型规则**（供 Claude 上课参考）：
- Alex 注意力低 / 只想跑通 → **Iris**
- Alex 精力好 / 想体会 "标准化为什么非做不可" → **Wine**（最戏剧性）
- Alex 要建立"KNN → ANN → Vector DB → RAG"的工程直觉 → **Digits**（维度起来了）
- Alex 想对比分类 vs 回归两种任务类型 → **California Housing**（KNN 回归）

**默认推荐**：第一轮走 **Wine**（教 scaler 的戏剧性最强），如果 Alex 想跳到架构讨论，转 **Digits**。

---

## §1 · 钩子 · 业务锚 + 跑通 demo（fast.ai top-down）

> **情境锚句**：这一章先让 Alex 跑通最小 demo 拿到结果，再讲原理——不从定义开始。KNN 的"算法本体"过于简单（5 步伪代码），真正的记忆钩子在"它是 Vector DB 的底层"。

### 1.1 一句话业务类比 · 给工程师的锚

**KNN = 向量数据库（Vector DB）的底层"最近邻检索"**

- **Software 1.0 视角**：KNN 是"带索引的相似度查询" — `SELECT label FROM training ORDER BY distance(query, features) LIMIT K`，然后对这 K 条做多数表决。
- **Software 2.0 视角**：KNN 是 **所有 RAG / 推荐系统 / 以图搜图** 的底层计算原语——把文本/图片变成 embedding 后，做的就是 KNN。
- **工程师最容易接受的心锚**：
  > "你要判断新用户是不是羊毛党？查他最像的 5 个老用户——5 个里 4 个是羊毛党 → 他也是。这就是 KNN。"

**为什么这个锚重要**：KNN 在大模型时代**不是一个可以跳过的小算法**——它是 RAG / 推荐 / 反欺诈的**检索命脉**（见 [Pinecone · What is a Vector Database](https://www.pinecone.io/learn/vector-database/)）。Alex 的方向是 AI Agent / 多 Agent 架构，RAG 是标配，KNN 就是地基。

### 1.2 最小可跑 demo（每候选数据集各一份）

> **情境锚句**：**不预选数据集**——把 §0.3 里所有候选各写一份最小 demo。上课时 Claude 根据现场选型 pick 其中一份跑。

#### 1.2.A 候选 · Iris（零基础款，10 行）

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_tr, y_tr)
print(f"准确率: {model.score(X_te, y_te):.2%}")
# 预期: 96~100%（Iris 太干净了）
```

- **预期输出**：`准确率: 100.00%`（或 96.67%）
- **第一印象锚**：
  > "这么简单就 100%？" — 对，Iris 太干净。要体会"什么时候会崩"，换 Wine。
  > "fit 都没算什么？" — 对，KNN 是 **lazy learner**，训练只存数据，计算推迟到 predict。

#### 1.2.B 候选 · Wine（标准化威力款，15 行）

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

X, y = load_wine(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# 先不做标准化跑一次
model_naive = KNeighborsClassifier(n_neighbors=5)
model_naive.fit(X_tr, y_tr)
print(f"未标准化: {model_naive.score(X_te, y_te):.2%}")  # 预期 ~60~75%

# 再做标准化跑一次
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_tr_s, y_tr)
print(f"标准化后: {model.score(X_te_s, y_te):.2%}")  # 预期 ~95%+
```

- **预期输出**：`未标准化: 72.22% / 标准化后: 97.22%`
- **第一印象锚**：
  > "差了 25 个百分点，光因为没 scale？" — 对。这就是 §4 "标准化对 KNN 致命" 这张卡片的戏剧性 demo。
  > **记忆钩子**：Wine 数据里 `proline` 特征范围上千，`灰分` <3。不标准化 → `proline` 的距离吞掉所有其他特征 → KNN 退化成 "按 proline 一列排序"。

#### 1.2.C 候选 · Digits（维度诅咒预热款）

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import time

X, y = load_digits(return_X_y=True)  # 1797 × 64
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_tr, y_tr)

t0 = time.time()
score = model.score(X_te, y_te)
print(f"准确率: {score:.2%}, 预测耗时: {(time.time()-t0)*1000:.1f}ms")
# 预期: ~98%, 耗时 50~200ms（1797 条训练 × 360 条测试 × 64 维）
```

- **预期输出**：`准确率: 98.61%, 预测耗时: 80.3ms`
- **第一印象锚**：
  > "才 1797 条训练数据就 80ms？那 100 万条呢？" — 80ms × (1,000,000 / 1797) ≈ **45 秒**/query。**工业界不能接受**，所以有了 ANN / HNSW / Vector DB（§3 展开）。

#### 1.2.D 候选 · California Housing（KNN 回归款）

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

X, y = fetch_california_housing(return_X_y=True)  # 20640 × 8
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)

model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_tr, y_tr)
y_pred = model.predict(X_te)
print(f"MAE: {mean_absolute_error(y_te, y_pred):.3f}")  # 单位：10 万美元
```

- **预期输出**：`MAE: 0.446`（平均误差约 4.5 万美元）
- **第一印象锚**：
  > "KNN 也能做回归？" — 能。分类是邻居投票，回归是邻居取平均。

---

## §2 · 递归缩放讲解（Alammar 黑盒逐层拆开）

### 2.1 黑盒视图 · 输入 → 输出

> **情境锚句**：先别管里面怎么算。这个黑盒吃什么、吐什么。

```
┌─────────────────────────────────────────┐
│            KNN 黑盒                      │
│                                          │
│ 输入（训练阶段）：                         │
│   X_train: shape (n, d) — n 条样本 d 维  │
│   y_train: shape (n,)   — n 个标签       │
│                                          │
│ 输入（预测阶段）：                         │
│   X_query: shape (m, d) — 待预测 m 条    │
│                                          │
│ 超参：                                     │
│   K: 考虑几个邻居（默认 5）               │
│   distance metric: 怎么算距离（默认欧氏）  │
│                                          │
│ 输出（分类）：y_pred shape (m,) 整数类别  │
│ 输出（回归）：y_pred shape (m,) 浮点数     │
└─────────────────────────────────────────┘
```

**工程接口类比**：一个 API 端点 `POST /predict`，body 带特征向量，返回类别 / 数值。
- 和逻辑回归比，**没有 model.coef_ / model.intercept_**——KNN 没有"学出来的参数"可保存，只有"训练数据本身"作为模型。
- `model.fit()` 只是**把数据存起来 + 建索引**（sklearn 默认用 KD-Tree 或 Ball-Tree），**不拟合任何参数**。这就是 lazy learner。

### 2.1.5 坐标系视觉先导（算距离的几何锚）

> **情境锚句**：这一节是**视觉脚手架**，在 §2.2 伪代码之前先让学员"看到"距离。
> **适用触发条件**：学员说出类似 "'算距离' 这三个字我建立不起链接" 的话——代数直觉缺失，需要视觉补给。
> **证据**：session-11 首次应用成功——Alex 在看到坐标系前卡在"距离"抽象词，看到图后立即自主跳出"类型不是唯一维度"的纠错。**视觉锚 > 代数锚**，对工程师尤其有效。

#### 步骤 A · 把每部电影变成"点"

每部电影 = 一组特征数字 = 一个坐标。先用 2 维（方便画图）：

```js
const 流浪地球3 = [180, 8.5]  // [时长, IMDB]
const 阿凡达    = [162, 7.9]
const 泰坦尼克  = [195, 7.9]
const 喜羊羊    = [85,  5.5]
```

#### 步骤 B · 画在坐标系里

```
IMDB
 9 |
   |        • 流浪地球3 (180, 8.5)
 8 |    • 阿凡达 (162, 7.9)    • 泰坦尼克 (195, 7.9)
   |
 7 |
   |
 6 |
   |              • 喜羊羊 (85, 5.5)
 5 +--------------------------------
   60    100   140   180    220   时长
```

**一眼看出**："流浪地球 3 离谁最近"变成**目测图上直线距离**——阿凡达最近、泰坦尼克其次、喜羊羊最远。

#### 步骤 C · 距离就是勾股定理

小学学过：两点 A=(x₁,y₁), B=(x₂,y₂) 的直线距离 = √((x₁-x₂)² + (y₁-y₂)²)

多维只是公式推广：

```js
// 对应位置相减、平方、求和、开方
function distance(a, b) {
  return Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0))
}
distance(流浪地球3, 阿凡达)   // ≈ 18.0（近 → 像）
distance(流浪地球3, 泰坦尼克) // ≈ 15.0
distance(流浪地球3, 喜羊羊)   // ≈ 95.0（远 → 不像）
```

**关键一句话**：**4 维 / 10 维 / 768 维（BERT embedding）都是同一套勾股定理**——维数变多，公式不变。

#### 步骤 D · 顺手解决学员的"为什么不先过滤"陷阱

学员（尤其 SQL 背景）下意识会做：
> "先 `WHERE 类型='科幻'`，再在科幻里算距离。"

**点破**：KNN 不做预筛。对应 SQL 是 **`ORDER BY distance LIMIT K`——没有 WHERE**。

反例最好用："一部**类型=悬疑、但时长/导演/IMDB 都极像流浪地球 3 的诺兰《信条》**" vs "一部**类型=科幻但烂到爆的 B 级片**"——按"距离综合"，《信条》距离更小，更值得参考；按"WHERE 类型='科幻'"，《信条》直接被砍。

顺带挂名字（先挂不展开，后续架构章展开）：**Metadata filtering / 元数据过滤**——pgvector / Pinecone / Milvus 都支持的"外层 WHERE 过滤"，但那是 KNN 外面包一层，不是 KNN 本身。

#### 步骤 E · 术语焊接点

学员看完坐标系后，用他的话可能说成"类型不是唯一维度"——这里焊接术语：

| 词 | 在 KNN 里 |
|---|---|
| **特征 feature / 维度 dimension** | 类型、时长、IMDB…每一列 |
| **距离 distance** | **多维综合出来的一个数**（不是维度本身，是多维运算的结果） |
| **样本 sample** | **一行数据**（= 一个 n 维空间里的点） |

**Software 1.0 vs 2.0 的第一跳**：
- 1.0：`WHERE` 是二值砍刀，非黑即白
- 2.0：`distance` 是连续评分，所有维度一起投票

这个跳跃是后续一切 Embedding / RAG / Vector DB 的认知地基——在 §2.1.5 这里第一次打。

---

### 2.2 拆一层 · 核心机制（五步伪代码）

> **情境锚句**：打开黑盒第一层——KNN 内部在做什么。5 步，没有任何玄学。

```java
// Java 视角看 KNN 预测（分类版）
public String predict(float[] query, List<Sample> trainingData, int K) {
    // 1. 全表扫描，计算 query 到所有训练样本的距离
    List<DistanceResult> dists = new ArrayList<>();
    for (Sample s : trainingData) {
        dists.add(new DistanceResult(euclidean(query, s.features), s.label));
    }

    // 2. 按距离升序排序
    dists.sort(Comparator.comparingDouble(d -> d.distance));

    // 3. 取 Top-K
    List<DistanceResult> topK = dists.subList(0, K);

    // 4. 多数表决（分类）或求均值（回归）
    return majorityVote(topK);  // 回归时改成 mean(topK)
}
```

**五步口诀**（让 Alex 背这个，不要背公式）：

1. **算距离**（query vs 全表）
2. **排序**
3. **取前 K**
4. **投票 / 求均值**
5. **输出**

**工程师直觉**：这等价于一个 `ORDER BY distance LIMIT K` + `GROUP BY label ORDER BY count DESC LIMIT 1`。
**训练阶段做什么**：只是把数据存起来（+ 可选建 KD-Tree 索引）。**不算距离不投票**——全部推迟到预测。这就是 "**lazy learner**" 的名字由来。

### 2.3 再拆一层 · 数学附录（下限答案层）

> **情境锚句**：这一章是**下限答案**——学员追问时才展开，不主动讲。原则遵循 [`01-LESSON-PLAN.md`](../00-mental-model/01-LESSON-PLAN.md) 的"数学层放最后"。

#### 2.3.1 欧氏距离 Euclidean Distance（默认）

$$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

**下限答案**：勾股定理的多维推广。"两点在 n 维空间里直线拉出来多远"。

#### 2.3.2 曼哈顿距离 Manhattan Distance（L1）

$$d(p, q) = \sum_{i=1}^{n} |p_i - q_i|$$

**下限答案**：出租车走街区——只能横平竖直，不能斜着走。对离群点更稳健。

#### 2.3.3 余弦相似度 Cosine Similarity（工业主流）

$$\text{cos}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$$

**下限答案**：只看"方向是否一致"，忽略"长度"。文本/embedding 场景的标配——长度≠相似度。**embedding 归一化后，余弦和欧氏等价**（见 [Weaviate · Distance Metrics in Vector Search](https://weaviate.io/blog/distance-metrics-in-vector-search)）。

#### 2.3.4 闵可夫斯基距离 Minkowski Distance（通用）

$$d(p, q) = \left(\sum_{i=1}^{n} |p_i - q_i|^r\right)^{1/r}$$

**下限答案**：欧氏和曼哈顿的母公式。r=1 是曼哈顿，r=2 是欧氏，r=∞ 是切比雪夫。**了解就行，sklearn 默认 r=2 欧氏**。

#### 2.3.5 为什么硬件亲和性很重要

**现代 CPU 的 SIMD / AVX-512 指令**和 **GPU 的 CUDA**对"向量点积（乘累加 FMA）"做了硬件级优化——`A·B = ΣAᵢBᵢ`可以一次时钟周期算几十维。这就是 FAISS、ONNX Runtime 用 C++ 做底层的根本原因。**纯 Java/Go 算距离慢 10-100 倍**，所以生产环境都交给专用引擎。

**追问延伸**：详见 `demos/02-KNN-and-Vector-Search.md`（迁移到本 LESSON-PLAN 后已删原文件）的旧版架构讨论。

---

## §3 · 坏方案 → 好方案递进（CS231n 风格）

> **情境锚句**：让 Alex 先尝一口"不用 ANN 会怎样"，再体会为什么工业界必须迁移。KNN 本体简单，真正的学习点在"它为什么不能直接上生产"。

### 3.1 坏方案 A · 不做特征标准化

**现象**：Wine 数据集，同样的 K=5，不做 scale 精度 72%，做了 scale 精度 97%。

**根因**：KNN 距离是所有特征的加和。`proline`（上千）直接吞掉 `灰分`（<3）→ KNN 退化成"按 proline 一列排序"，其他 12 个特征全废。

**修复**：`StandardScaler()` 或 `MinMaxScaler()`。**KNN 的特征预处理不是可选项，是必选项**。

**衍生工程坑**（§6 会展开）：Python 离线 `scaler.fit_transform` 训出来的 `mean_` 和 `var_` 必须**和在线系统同步**——如果 Java/Go 网关用了不一样的 mean/var，模型直接漂移。

参考：[Baeldung · kNN high-dim issues](https://www.baeldung.com/cs/k-nearest-neighbors)

### 3.2 坏方案 B · 暴力全表扫描（精确 KNN）

**现象**：Digits 数据集（1797 条 × 64 维），单次预测 80ms。线性外推到 100 万条 → **45 秒/query**。

**工业界实测**：1M × 768 维 embedding 场景下，暴力 KNN 延迟上百 ms，**推荐系统/RAG 全挂**（见 [ANN-Benchmarks](https://ann-benchmarks.com/)）。

**修复**：**ANN 近似最近邻**，把 O(n·d) 降到 O(log n) 或 O(√n)。代价：允许 1-5% 召回损失。

#### 三种主流 ANN 索引（只记名字和"谁用在什么场景"）

| 索引 | 机制 | 优势 | 代价 | 谁在用 |
|---|---|---|---|---|
| **HNSW** 分层小世界图 | 建多层导航图，跳着搜 | 召回最稳、延迟最低 | 内存占用大 | Milvus、Qdrant、pgvector 默认 |
| **IVF** 倒排文件 | 先聚类 K 个 centroid，查询只扫邻近桶 | 内存省、适合十亿规模 | 召回略低 | FAISS 默认 |
| **PQ** 乘积量化 | 向量压缩存储 | 省内存 | 召回再降 | 常和 IVF 组合成 IVF-PQ |
| **ScaNN**（Google）| 各向异性量化 | 小内存下反超 HNSW | 生态窄 | Google 内部、少数团队 |

参考：[FAISS 官方论文 (Meta, 2024)](https://arxiv.org/pdf/2401.08281) / [Pinecone · Faiss Missing Manual](https://www.pinecone.io/learn/series/faiss/) / [TiDB · IVF vs HNSW vs PQ](https://www.pingcap.com/article/approximate-nearest-neighbor-ann-search-explained-ivf-vs-hnsw-vs-pq/)

### 3.3 坏方案 C · 自己造轮子而不是用 Vector DB

**现象**：团队在 Redis / MySQL 里手写 embedding 距离计算。

**为什么错**：专用 Vector DB 已经把 ANN 索引、metadata 过滤、分片、分布式、监控都做了。手写 = 重复造轮子 + 性能不达标。

**2024-2025 Vector DB 选型决策树**（[LiquidMetal · Vector DB Comparison](https://liquidmetal.ai/casesAndBlogs/vector-comparison/) / [Tensorblue · 2025 Vector DB Report](https://tensorblue.com/blog/vector-database-comparison-pinecone-weaviate-qdrant-milvus-2025)）：

```
数据量 < 10M 且已有 Postgres？     → pgvector（最省事）
需要复杂 metadata filter？          → Qdrant（Rust 写，过滤最强）
十亿级 vector？                     → Milvus（分片成熟）
不想自己运维？                       → Pinecone（托管）
要内置 keyword + vector 混合检索？  → Weaviate
```

### 3.4 Software 1.0 vs 2.0 对比

| 维度 | Software 1.0 · MySQL LIKE 查询 | Software 2.0 · Vector DB KNN |
|---|---|---|
| 输入 | 关键字字符串 | 高维 embedding |
| 索引 | B+Tree / 倒排索引 | HNSW / IVF-PQ |
| 相似性 | 字符串匹配 | 向量距离（欧氏 / 余弦） |
| 表达能力 | 形式匹配 | **语义匹配**（"iPhone 充电线"能召回"Lightning 数据线"） |
| 何时用 | 精确查询、结构化字段 | 非结构化（图 / 文 / 音）、语义搜索、RAG |

---

## §4 · 术语卡片（每张独立自包含，追问就近放）

> **情境锚句**：这一章每张卡片是一个**可被单独检索**的语义单元。Alex 追问任意一个，Claude 直接跳到对应卡片就能给出完整答复 + URL 溯源。

### 4.1 K 值 · Hyperparameter

- **中英对照**：K 值 / `n_neighbors`
- **业务锚**：投票参与人数。3 人投票容易跟风，100 人投票又太稀释——**要找一个平衡点**。
- **一句话定义**：预测时，考虑"最近的几个邻居"来投票/取均值。
- **在 KNN 里的角色**：KNN 唯一的核心超参。其他都是 tuning 细节。
- **常见追问**：
  - **Q1：K 怎么选？**
    - 下限答案：`sqrt(n)` 起步 + CV 网格搜索；**分类用奇数避免平票**；小 K 过拟合（记住噪声），大 K 欠拟合（边界模糊）。
    - URL：[GeeksforGeeks · How to find optimal K in KNN](https://www.geeksforgeeks.org/machine-learning/how-to-find-the-optimal-value-of-k-in-knn/)
  - **Q2：K=1 什么意思？K=n 什么意思？**
    - K=1：完全跟最近那个邻居 → 极端过拟合，单个噪声就能误导
    - K=n：所有训练样本一起投票 → 等价于"永远返回训练集里最多的那一类"（多数类基线）
    - 合理范围：`K ∈ [3, √n]`

### 4.2 距离度量 · Distance Metric

- **中英对照**：距离度量 / distance metric
- **业务锚**：不同距离 = 不同的"相似怎么算"。和产品需求强相关。
- **一句话定义**：衡量两个向量"不像"程度的函数。距离越大 = 越不像。
- **在 KNN 里的角色**：决定"谁是邻居"。度量选错 → 邻居选错 → 预测全错。
- **sklearn 默认**：`metric='minkowski', p=2`（即欧氏距离）
- **常见追问**：
  - **Q1：什么时候用余弦而不是欧氏？**
    - 下限答案：**文本 / embedding 场景都用余弦**——只看方向不看长度。文档长短不该影响相似度。embedding 归一化后，余弦和欧氏等价。
    - URL：[Pinecone · Vector Similarity Explained](https://www.pinecone.io/learn/vector-similarity/) / [Weaviate · Distance Metrics](https://weaviate.io/blog/distance-metrics-in-vector-search)
  - **Q2：曼哈顿和欧氏差在哪？**
    - 下限答案：曼哈顿 = L1（直角走路），对离群点更稳健；欧氏 = L2（直线距离）。高维时差异缩小，低维时可对比试。
  - **Q3：切比雪夫、闵氏是什么？**
    - 下限答案：闵氏是母公式（r=1 曼哈顿，r=2 欧氏，r=∞ 切比雪夫）。**了解就行，默认欧氏够用**。

### 4.3 维度诅咒 · Curse of Dimensionality

- **中英对照**：维度诅咒 / curse of dimensionality
- **业务锚**：维度一高，"远和近"的感觉就失灵——就像在一个太大的城市里，所有地点"感觉都差不多远"。
- **一句话定义**：维度升高 → 空间体积指数膨胀 → 所有点两两距离趋于相等 → "最近"失去意义。
- **在 KNN 里的角色**：KNN 对维度极其敏感。**>20-50 维就开始退化**，embedding 动辄 768 / 1536 维 → 为什么需要 PCA 降维 / ANN 近似。
- **常见追问**：
  - **Q1：为什么会趋于相等？**
    - 下限答案：n 维单位球的体积几乎全集中在"球壳"，内部空荡荡。所有点都住在"边缘"，彼此距离都差不多。
    - URL（数学推导）：[Cornell CS4780 · kNN & Curse of Dimensionality](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote02_kNN.html)
  - **Q2：embedding 768 维怎么办？**
    - 下限答案：两条路——**降维**（PCA / UMAP 降到 100-300 维再 KNN）或 **ANN 近似**（HNSW 在高维下性能没那么差，靠图结构跳着搜）。

### 4.4 懒惰学习 · Lazy Learner

- **中英对照**：懒惰学习 / lazy learner（vs **eager learner**）
- **业务锚**：拖延症学生。平时不学，考试前狂翻笔记。
- **一句话定义**：训练阶段只存数据，不拟合参数；所有计算推迟到预测时。
- **在 KNN 里的角色**：解释"为什么 `model.fit()` 很快但 `model.predict()` 很慢"。
- **常见追问**：
  - **Q1：对应的 eager learner 是什么？**
    - 下限答案：逻辑回归 / 线性回归 / K-Means / 决策树 / 神经网络——训练时就"学"出参数固化，预测只做前向计算。
    - URL：[Medium · KNN and KMeans](https://harshkr21august.medium.com/knn-and-kmeans-b741dfccb69)
  - **Q2：lazy 有什么代价？**
    - 下限答案：① 预测慢（必须重算） ② 内存占用大（整个训练集常驻） ③ 不能在线推理场景直接用——百万级 QPS 下崩给你看。

### 4.5 特征标准化 · Feature Standardization

- **中英对照**：标准化 / standardization（Z-score）；归一化 / normalization（Min-Max）
- **业务锚**：让所有特征"换到同一张尺子上再比"。
- **一句话定义**：
  - 标准化 Z-score：`(x - mean) / std` → 均值 0、标准差 1 的正态分布
  - 归一化 Min-Max：`(x - min) / (max - min)` → [0, 1] 区间
- **在 KNN 里的角色**：**必做，不是可选**。Wine demo 可复现"不做就崩 25%"的戏剧效果。
- **sklearn API**：`StandardScaler()` / `MinMaxScaler()`
- **工业界首选**：`StandardScaler`（对离群点更稳健）
- **常见追问**：
  - **Q1：`fit_transform` 和 `transform` 有什么区别？**
    - 下限答案：
      - `fit_transform(X_train)`：**算 mean/var 并应用**，用在训练集
      - `transform(X_test)`：**只应用，不重新算**，用在测试集 / 在线预测
    - **如果测试集也 fit_transform**：等于用测试集的分布去评估，**数据泄漏 data leakage** → 评分虚高。
  - **Q2：离线训练的 mean/var 怎么到线上？**
    - 下限答案：**必须持久化**。MLOps 实践：
      - 把 scaler 对象序列化（`joblib.dump`）
      - 更好的做法：打包进 ONNX 计算图，让推理引擎原生处理
      - 最差做法：Python 训 + Java 网关复现 mean/var — 精度不一致 / 版本不同步 → **灾难级漂移**

### 4.6 交叉验证 + 网格搜索 · CV + GridSearch

- **中英对照**：交叉验证 / cross-validation (CV)；网格搜索 / grid search
- **业务锚**：CV = "轮流考试+平均分"，避免一次考试的运气成分。GridSearch = "把所有参数组合穷举试一遍"。
- **一句话定义**：
  - CV：训练集切成 k 份，轮流用 1 份做验证、k-1 份做训练，取 k 次评分均值
  - GridSearch：给一个参数空间（比如 `K ∈ [1..20]`），对每个组合做 CV 打分，选最优
- **在 KNN 里的角色**：**选 K 的唯一靠谱方式**。不用 CV 选 K = 靠感觉。
- **sklearn API**：`GridSearchCV(estimator, param_grid, cv=5)`
- **常见追问**：
  - **Q1：`cv=5` 是什么意思？**
    - 下限答案：5-fold CV，训练集切 5 份，轮 5 次。**cv=10 更稳但慢一倍**。
  - **Q2：GridSearch 找到的 K 就是最好的？**
    - 下限答案：是**在给定搜索空间和 CV 方式下最好**。如果空间只给 `range(1, 10)`，最优可能在 15——你永远不知道。**所以 GridSearch 前先用 `sqrt(n)` 大致划一个合理范围**。

### 4.7 近似最近邻 · ANN

- **中英对照**：近似最近邻 / Approximate Nearest Neighbor (ANN)（**不是** Artificial Neural Network）
- **业务锚**：快递分拣——不追求"绝对最近"，追求"足够近 + 快 1000 倍"。
- **一句话定义**：用特殊索引结构（HNSW / IVF / PQ）在可接受的召回损失下，把最近邻查询从 O(n) 降到 O(log n)。
- **在 KNN 里的角色**：让精确 KNN 从"教学玩具"变成"十亿级生产基建"。
- **常见追问**：
  - **Q1：召回损失是什么意思？**
    - 下限答案：暴力 KNN 能找到真正 Top-5；ANN 可能找到 4/5 真实的 + 1 个次近的。**召回率 95%+ 工业界普遍接受**。
    - URL：[ANN-Benchmarks](https://ann-benchmarks.com/)
  - **Q2：HNSW / IVF / PQ 是谁？**
    - 下限答案：见 §3.2 表格——**HNSW 用图跳着搜（主流）；IVF 用聚类分桶；PQ 压缩向量省内存**。

### 4.8 向量数据库 · Vector DB

- **中英对照**：向量数据库 / Vector Database
- **业务锚**：MySQL 是"结构化数据仓库"，Vector DB 是"语义数据仓库"。都是数据库，存的东西不同。
- **一句话定义**：专门存储 + 检索高维向量的数据库。内置 ANN 索引、metadata 过滤、分片、持久化。
- **在 AI 时代的角色**：RAG / 推荐 / 以图搜图的**标准基建**。
- **主流产品**（2024-2025）：
  - **pgvector** — Postgres 扩展，<10M 向量首选
  - **Qdrant** — Rust 写，metadata 过滤最强
  - **Milvus** — 十亿级向量
  - **Pinecone** — 托管服务
  - **Weaviate** — 内置混合检索
- **常见追问**：
  - **Q1：为什么不直接用 Redis / MySQL？**
    - 下限答案：Redis 和 MySQL 没有 **ANN 索引**。即使你自己装扩展（pgvector），性能也只在 <10M 级别能打。十亿级必须专用 Vector DB。
    - URL：[Pinecone · What is a Vector Database](https://www.pinecone.io/learn/vector-database/)
  - **Q2：embedding 哪里来？**
    - 下限答案：预训练模型。文本用 `text-embedding-3-small`（OpenAI）/ `bge-m3`（开源）；图像用 CLIP / ResNet。**Embedding 模型决定检索质量上限**。

### 4.9 Embedding 漂移 · Embedding Drift

- **中英对照**：embedding 漂移 / embedding drift
- **业务锚**：升级了翻译软件却没告诉所有员工——新老文档的"意思"对不上了。
- **一句话定义**：embedding 模型换版本后，新向量和旧向量在同一个空间里**不可比**——必须全量重建索引。
- **在 KNN / Vector DB 里的角色**：最隐蔽的生产杀手。
- **常见追问**：
  - **Q1：怎么预防？**
    - 下限答案：embedding 模型版本**必须**和向量库元数据强绑定（存进 metadata）。升级时**全量 re-embed + rebuild index**，不能部分更新。
    - URL（真实案例 · 1.2 亿向量 18 小时重建）：[Embedding Drift: The Quiet Killer of RAG](https://dev.to/dowhatmatters/embedding-drift-the-quiet-killer-of-retrieval-quality-in-rag-systems-4l5m)

---

## §5 · Questionnaire · 显式 active recall（fast.ai 风格）

> **情境锚句**：这一章是**闭卷题库**，按 Bloom 认知分层。Claude 上课时按 Alex 节奏随机抽题。评分标准见 [`01-LESSON-PLAN.md`](../00-mental-model/01-LESSON-PLAN.md)。

### 5.1 记忆层（能复述 · 5 题）

1. KNN 预测五步法说出来？（算距离 → 排序 → 取前 K → 投票/均值 → 输出）
2. KNN 为什么叫 "lazy learner"？（训练只存数据，计算推迟到预测）
3. KNN 分类和回归的区别是什么？（分类投票，回归取均值）
4. sklearn 里默认的距离度量是什么？（欧氏，`metric='minkowski', p=2`）
5. 为什么分类时 K 常用奇数？（避免 2 类各 K/2 票的平票情况）

### 5.2 应用层（能选型 / 读代码 · 5 题）

1. 看这行：`KNeighborsClassifier(n_neighbors=5)`，5 是超参还是参数？（**超参**，你选的；参数是模型自己学的，KNN 没有参数）
2. `fit_transform(X_train)` vs `transform(X_test)` 为什么不一样？（前者算+应用 mean/var，后者只应用——防数据泄漏）
3. 给你一个文本语义搜索场景（768 维 BERT embedding），该用哪种距离？为什么？（**余弦**——文档长短不该影响相似度；归一化后余弦≈欧氏）
4. Wine 数据集不做标准化直接 KNN 会发生什么？为什么？（精度崩盘；`proline` 量纲上千吞掉其他 12 个特征）
5. 业务场景：反欺诈，已知"疑似欺诈"的 K 个历史账号里 3/5 是真欺诈 → KNN 预测这个新账号是欺诈。这个系统**死盯哪个指标**？为什么？（Recall——漏判欺诈损失钱，复习 session-09 的决策链）

### 5.3 迁移层（新场景 / 对比 · 5 题）

1. **KNN 和 K-Means 是不是同一个 K？**（不是！KNN = 监督分类，K 是邻居数；K-Means = 无监督聚类，K 是簇数。名字撞车纯巧合）
2. 你有 100 万 × 768 维 embedding，暴力 KNN 单次预测多久？还能用吗？（几百 ms 到秒级；不能用；换 HNSW / Vector DB）
3. pgvector 什么时候够用，什么时候要换 Milvus？（<10M 向量 + 已有 Postgres → pgvector；十亿级 → Milvus）
4. 一个 RAG 系统原本召回率 95%，某天升级 embedding 模型后降到 40%——最可能是什么原因？（embedding drift——新旧 embedding 不在同一向量空间，必须 rebuild index）
5. KNN 和逻辑回归对比，各自强项是什么？（逻辑回归：训练慢预测快、有参数可解释、适合线性可分；KNN：训练快预测慢、无参数、适合非线性边界和"相似度检索"场景）

---

## §6 · 常见坑 / 反面教材

> **情境锚句**：Alex 已经犯过 / 必然会犯的错误。备课时 Claude 要提前埋伏，讲到对应概念时顺手引爆。

### 6.1 学员容易误解（概念坑）

- **坑 1 · KNN 和 K-Means 是同一个 K**
  - 正确：KNN 的 K = 投票邻居数；K-Means 的 K = 簇数。**除了都叫 K，没有任何关系**。
  - 触发时机：§4.4 / §5.3.1 / 如果 Alex 已学过 K-Means（目前没）

- **坑 2 · KNN 有"训练过程"**
  - 正确：KNN 是 lazy learner，`fit()` 只是把数据存起来（+ 建 KD-Tree 索引）。**不拟合任何参数**。
  - 触发时机：§2.1 / §2.2

- **坑 3 · 维度诅咒是"维度越多越好"**
  - 正确：反过来——维度越高，"最近"越失去意义。**大于 20-50 维就开始退化**。
  - 触发时机：§4.3

### 6.2 工程落地坑

- **坑 4 · 不做标准化直接训**
  - 后果：量纲大的特征吞掉其他特征，KNN 退化成单维度排序
  - 防御：`StandardScaler()` 是必做，不是选做
  - 触发时机：§1.2.B Wine demo / §3.1 / §4.5

- **坑 5 · 测试集也 `fit_transform`**
  - 后果：数据泄漏 → 评分虚高 → 生产环境崩
  - 防御：训练 `fit_transform`，测试/预测 `transform` 只应用不重新算
  - 触发时机：§4.5.Q1

- **坑 6 · 离线训 + 在线推理的 mean/var 不同步**
  - 后果：Python 算出来的 scaler 参数没同步到 Java/Go 网关 → 预测漂移
  - 防御：scaler 对象序列化（`joblib.dump`）或打包进 ONNX
  - 触发时机：§4.5.Q2

- **坑 7 · Embedding drift · 1.2 亿向量 18 小时重建案例**
  - 后果：升级 embedding 模型没版本化 → 新老向量混杂 → 召回率崩
  - 防御：embedding 模型版本 + 向量库元数据强绑定；升级时全量 rebuild
  - URL：[Embedding Drift Killer of RAG](https://dev.to/dowhatmatters/embedding-drift-the-quiet-killer-of-retrieval-quality-in-rag-systems-4l5m)
  - 触发时机：§4.9

- **坑 8 · Vector DB 跨区容灾（AWS 2025 宕机）**
  - 后果：托管 Vector DB（Pinecone）缺少跨 region failover → AWS 大规模宕机时 RAG 应用全线停摆
  - 防御：Vector DB 不是"无状态 CDN"，需要传统 DB 的灾备设计
  - URL：[Zilliz · AWS Outage Wake-Up Call](https://zilliz.com/blog/the-aws-outage-was-a-wake-up-call-for-vector-database-cross-region-disaster-recovery)
  - 触发时机：§3.3 讨论 Vector DB 选型时

- **坑 9 · ingest 用 L2，query 用 cosine**
  - 后果：距离度量不一致 → "近邻"完全错位 → 静默错误
  - 防御：索引创建时固定一个 metric，查询时对齐
  - URL：[Vector DB Debugging: FAISS/pgvector/Qdrant/Redis](https://kitemetric.com/blogs/vector-database-debugging-fixing-subtle-failures-in-faiss-pgvector-qdrant-and-redis)
  - 触发时机：§4.2

### 6.3 反面教材引用

- demo-02 session-07 "把 `predict_proba` 第二列当置信度偏移量" 这类"数字符号搞错"的模式在 KNN 里对应："误把 K=1 当成最强"（实则极端过拟合）。
- 若未来 KNN 实战产生翻车，追加到 `learning-sessions/` 对应 session 并在此引用。

---

## §7 · 数据流图 · 跑完 demo 后补

> **情境锚句**：本节为占位。`DATA-FLOW.md` 是跑完 demo 后的临时产出物，不在模板起手范围。

详见 `DATA-FLOW.md`（未创建，上课后 Alex 闭卷口述 / Claude 转图时产出）

---

## §8 · 上课现场调度建议（可选章节）

> **情境锚句**：这一节是**给上课时的 Claude 看的运行手册**——按 Alex 状态和本节目标组合切片。非模板标准章节，试行中。

### 8.1 90 分钟完整版（Alex 精力好）

1. §1.1 业务锚（5min）
2. §1.2.B Wine demo 对比（10min）— 戏剧效果拉满
3. §2.1 黑盒 → §2.2 五步机制（10min）
4. §3.1 标准化坑（穿插 §4.5 术语）（15min）
5. §3.2 暴力 KNN 崩 → §3.3 Vector DB（20min）
6. §5 Questionnaire 抽 5 题（15min）
7. 收尾 + 记 NOTES（15min）

### 8.2 30 分钟压缩版（Alex 注意力低）

1. §1.1 业务锚（3min）
2. §1.2.A Iris demo（5min）— 跑通就够
3. §2.2 五步机制（5min）
4. §4.1 + §4.5 两张术语卡（10min）
5. §5.1 记忆层 2 题收尾（7min）

### 8.3 架构对话版（Alex 想聊 AI Agent 方向）

1. §1.1 业务锚（RAG 地基定位）（5min）
2. 跳过本体，直接 §3.2 + §3.3（ANN / Vector DB 讨论）（20min）
3. §4.7 + §4.8 + §4.9 三张架构卡（20min）
4. §6.2 工程坑集锦（15min）

### 8.4 F1 闭环时机（session-09 盲区）

§5.2.5 题里反欺诈选型 + §5.1 分类回归区分 → 自然引出"为什么综合看 F1"。或在 §6.3 里提 demo-02 评估指标的对接。

---

## 附：本 LESSON-PLAN 的数据源（按 v0.2 模板 §8.9 三源定位表）

| 来源 | 文件 / URL | 本 LESSON-PLAN 用在哪 |
|---|---|---|
| **原始培训笔记** | `assets/source-materials/第4阶段-机器学习/KNN算法.md` | §2.2 五步机制 · §2.3 距离公式 · §4.1 K 值 · §4.5 标准化 · §4.6 CV+GridSearch · §5.1 记忆层 |
| **第一版改编** | `demos/02-KNN-and-Vector-Search.md`（**合并后将删除**） | §1.1 业务锚（Software 1.0 vs 2.0）· §2.1 黑盒视图 · §2.3.5 硬件亲和性 · §3.3 Vector DB · §3.4 对比表 · §4.8 Vector DB |
| **联网搜索交叉验证** | 详见下方 URL 清单 | §0.3 候选数据集 · §3.2 ANN 索引三件套 · §3.3 选型决策树 · §4 追问下限答案 URL · §6.2 工程坑 URL |

### 联网 URL 清单（session-09 agent 交付）

**数据集**：
- [sklearn toy datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html)
- [sklearn fetch_california_housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

**ANN / Vector DB 技术**：
- [Pinecone · Faiss Missing Manual](https://www.pinecone.io/learn/series/faiss/)
- [Pinecone · What is a Vector Database](https://www.pinecone.io/learn/vector-database/)
- [Pinecone · HNSW 教程](https://www.pinecone.io/learn/series/faiss/hnsw/)
- [Pinecone · Vector Similarity Explained](https://www.pinecone.io/learn/vector-similarity/)
- [FAISS Meta 2024 论文](https://arxiv.org/pdf/2401.08281)
- [ANN-Benchmarks](https://ann-benchmarks.com/)
- [TiDB · IVF vs HNSW vs PQ](https://www.pingcap.com/article/approximate-nearest-neighbor-ann-search-explained-ivf-vs-hnsw-vs-pq/)
- [Milvus · FAISS/HNSW/ScaNN 对比](https://milvus.io/ai-quick-reference/what-is-the-role-of-faiss-hnsw-and-scann-in-ai-databases)
- [Qdrant · Benchmarks](https://qdrant.tech/benchmarks/)
- [Tiger Data · pgvector vs Qdrant](https://www.tigerdata.com/blog/pgvector-vs-qdrant)
- [LiquidMetal · Vector DB Comparison](https://liquidmetal.ai/casesAndBlogs/vector-comparison/)
- [Tensorblue · 2025 Vector DB Report](https://tensorblue.com/blog/vector-database-comparison-pinecone-weaviate-qdrant-milvus-2025)

**距离度量**：
- [Weaviate · Distance Metrics in Vector Search](https://weaviate.io/blog/distance-metrics-in-vector-search)

**学院派 / 数学**：
- [Cornell CS4780 · kNN & Curse of Dimensionality](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote02_kNN.html)

**教程 / Q&A**：
- [GeeksforGeeks · Optimal K](https://www.geeksforgeeks.org/machine-learning/how-to-find-the-optimal-value-of-k-in-knn/)
- [GeeksforGeeks · KNN supervised or unsupervised](https://www.geeksforgeeks.org/machine-learning/is-knn-supervised-or-unsupervised/)
- [Baeldung · KNN high-dim issues](https://www.baeldung.com/cs/k-nearest-neighbors)
- [Medium · KNN vs KMeans](https://harshkr21august.medium.com/knn-and-kmeans-b741dfccb69)

**现代实践 / 工业场景**：
- [ziyang.io · Industrial Leap from KNN to ANN](https://ziyang.io/blog/2025-practical-knn)
- [arXiv 2025 · FAISS vs ScaNN in biology](https://arxiv.org/html/2507.16978v1)
- [MDPI 2023 Sensors · KNN+LDA in fraud detection](https://www.mdpi.com/1424-8220/23/18/7788)

**业界翻车 / 反面教材**：
- [Embedding Drift Killer of RAG](https://dev.to/dowhatmatters/embedding-drift-the-quiet-killer-of-retrieval-quality-in-rag-systems-4l5m)
- [Zilliz · AWS Outage Wake-Up Call](https://zilliz.com/blog/the-aws-outage-was-a-wake-up-call-for-vector-database-cross-region-disaster-recovery)
- [Vector DB Debugging](https://kitemetric.com/blogs/vector-database-debugging-fixing-subtle-failures-in-faiss-pgvector-qdrant-and-redis)
- [Chitika · Retrieval Inconsistency After Vector DB Updates](https://www.chitika.com/vector-db-retrieval-inconsistency-rag/)
- [Achilles Heel of Vector Search: Filters](https://yudhiesh.github.io/2025/05/09/the-achilles-heel-of-vector-search-filters/)

---

## 附：迭代记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-04-20 | v0.1 | 首版。基于模板 v0.2，合并 `assets/source-materials/.../KNN算法.md` + `demos/02-KNN-and-Vector-Search.md`（待删）+ 联网交叉验证。试行第一例。 |
