---
tags: [API/sklearn, 算法/随机森林, 超参数/n_estimators, 超参数/max_features, 超参数/max_depth]
---

# API

> 维度：**代码**
> 知识点级别：【知道】sklearn RandomForestClassifier API
> 章节底稿全文见 [`README.md`](./README.md)（PPT slide 20-21, 24）

## ━━━━━━━━ 底稿 ━━━━━━━━

### PPT

> Slide 20 · 随机森林 API

〔图：RandomForestClassifier API 参数说明截图〕

> Slide 21 · 随机森林 API（续）

〔图：API 示例代码截图〕

> **Notes**：基于前面的分类器进一步训练。前面的分类器，贡献就是不同样本权重。

> Slide 24 · 本章小结（节选）

**3 随机森林 API**

`sklearn.ensemble.RandomForestClassifier()`

### 笔记

> 【知道】随机森林 API

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(
    n_estimators=10,        # 树的数量
    criterion="gini",       # gini / entropy
    max_depth=None,         # 树最大深度
    max_features="sqrt",    # sqrt(n_features), log2, None；旧版 "auto" 已在 sklearn ≥1.3 移除
    bootstrap=True,         # 是否有放回抽样
    min_samples_split=2,    # 节点分裂最小样本数
    min_samples_leaf=1,     # 叶节点最小样本数
)
```

**调参直觉**：

- `n_estimators` ↑ → 稳但慢，到一定值后边际收益递减
- `max_depth` 不限制时树会长到底；样本量大时建议限制防 OOM
- `max_features` 越小树差异越大、方差越低、但偏差可能上升

---

## ━━━━━━━━ 讲解 ━━━━━━━━

### 直觉

**业务痛点**：[`01-算法思想.md`](./01-算法思想.md) 把"两层随机"讲清楚了 —— 但工程上落地时面对的是 **15+ 个参数**的 `RandomForestClassifier`，不知道哪个该动、哪个保持默认、哪个有版本坑。

**生活锚（接 5 位医生的类比）**：参数就是医院给会诊小组定的**操作手册**：

| 旋钮 | 翻译 |
|---|---|
| `n_estimators` | 请几位医生（5 位 / 100 位 / 500 位）|
| `max_features` | 每位医生每道题最多看几个指标 |
| `bootstrap` | 病例库是抽样发还是全量发 |
| `max_depth` / `min_samples_*` | 单个医生最多能多细致地分析（防钻牛角尖）|

一句话：**API 把两层随机翻译成 3 个森林旋钮 + 沿用决策树的刹车**。

### 【代码】参数地图：1 个规模 + 1 个开关 + N 把刹车 + 1 个种子

```
                ┌────────────────────────────────┐
                │  RandomForestClassifier        │
                └────────────────────────────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
   规模（森林大小）        刹车（防过拟合）        种子（复现）
   n_estimators            max_depth              random_state
                           max_features
                           min_samples_split
                           min_samples_leaf
                           bootstrap
       │                      │
   开关（分裂准则）
   criterion
```

> 与单棵 DecisionTreeClassifier 对比：RF 多了 `n_estimators`、`max_features`、`bootstrap` 三个控制"森林"的旋钮，其余刹车参数含义完全复用。参数全表见 [`../../04-DecisionTree/05-titanic/02-API.md`](../../04-DecisionTree/05-titanic/02-API.md)。

---

### 【代码】关键参数表

| 参数 | 默认值 | 作用 | 调参直觉 |
|---|---|---|---|
| `n_estimators` | 100 | 森林里树的棵数 | 越大越稳，≥100 后收益递减；起手 100，计算允许可试 200-500 |
| `max_features` | `"sqrt"` | 每节点候选特征数 | 分类用 `"sqrt"`（$\sqrt{p}$）；回归用 `1.0`；减小→树差异增大方差降低 |
| `max_depth` | `None` | 单棵树最大深度 | None=不限（随机森林本来就不剪枝）；数据量大/特征多时建议设 10-30 |
| `bootstrap` | `True` | 是否有放回抽样 | 几乎不改；`False` 退化为普通 Bagging（无放回），失去 OOB 估计 |
| `criterion` | `"gini"` | 分裂准则 | gini 默认；entropy 差异通常 < 0.5% acc，不值得换 |
| `min_samples_split` | 2 | 节点继续分裂的最小样本数 | 大数据集设 10-50，防止叶子过碎 |
| `min_samples_leaf` | 1 | 叶子最少样本数 | 大数据集设 5-20，设 `min_samples_split` 的一半左右 |
| `random_state` | `None` | 随机种子 | 实验/复现必填，生产固化后无所谓 |

---

### 【代码】`n_estimators`：多少棵树才够

N 棵树投票的误差随 N 递减，但**边际收益会饱和**：

```
错误率
  ↑
  │ ╲
  │  ╲
  │   ╲___
  │       ────────────── ← 收敛线（Bias 下限）
  │
  └──────────────────────→ n_estimators
       10  50  100  200  500

N < 50  → 误差下降明显，值得加
N > 200 → 收益递减，主要影响训练时间
```

**实战起手**：`n_estimators=100`（sklearn 默认值）；时间充裕再试 200/500。

> 旧版 sklearn（< 0.22）默认 `n_estimators=10`，现版本已改为 100。如果看到旧教材/博客用 10，是历史原因。

---

### 【代码】`max_features`：最重要的刹车

这是 RF 独有的参数，决定"树间相关性"：

```
max_features 效果对比：
────────────────────────────────────────────────────
大（接近 p）  →  树之间选同样的强特征  →  高相关  →  方差降低少
小（接近 1）  →  树之间特征差异大     →  低相关  →  方差降低多
                                                  但偏差上升
"sqrt"        →  经验甜点（分类任务默认）
────────────────────────────────────────────────────
```

**sklearn ≥1.3 的坑**：旧版 `max_features="auto"` 在分类任务里等价于 `"sqrt"`，在回归任务里等价于 `1.0`。**sklearn 1.1 开始警告，1.3 彻底移除**。如果你看到报错 `ValueError: Invalid value for max_features`，把 `"auto"` 改成 `"sqrt"`（分类）或 `1.0`（回归）。

---

### 【代码】`bootstrap=True`：为什么几乎不改

关闭 Bootstrap（`bootstrap=False`）意味着：
- 每棵树用**完整**训练集（无放回）
- 失去了样本随机性 → 各棵树差异只靠特征随机
- 无法计算 **OOB（Out-of-Bag）**误差估计

**OOB 是什么**：每棵树约有 36.8% 的样本未被抽到，可用这部分数据评估该树的泛化——这相当于"免费的交叉验证"，不需要额外的 validation set。开启方式：`oob_score=True`。

```python
clf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
clf.fit(X_train, y_train)
print(clf.oob_score_)   # 不切 validation set 也能知道泛化误差
```

---

### 【代码】调参优先级

```
第1优先：n_estimators  →  直接拉到计算允许的最大值（100-200 通常足够）
第2优先：max_features  →  分类默认 "sqrt"；若过拟合可适当减小
第3优先：max_depth     →  None（默认），过拟合时设 10-30
第4优先：min_samples_* →  大数据集才需要调
```

实战配方（起手）：

```python
RandomForestClassifier(
    n_estimators=100,
    max_features="sqrt",    # 分类任务默认，不用动
    max_depth=None,         # 随机森林不需要剪枝（靠集成降方差）
    random_state=42,
    n_jobs=-1,              # 并行用所有 CPU 核心
)
```

---

### 【代码】与 DecisionTreeClassifier 参数复用提示

RF 的 `criterion / max_depth / min_samples_split / min_samples_leaf / random_state` 含义与单棵 `DecisionTreeClassifier` 完全一致——详细调参直觉见 [`../../04-DecisionTree/05-titanic/02-API.md`](../../04-DecisionTree/05-titanic/02-API.md)（"1 开关 + 3 把刹车 + 1 个种子"部分）。

区别只有：RF 在外面套了一层"N 棵树 + 投票"，内部每棵树的参数语义不变。

---

### 不在本章范围

- **GridSearchCV 实战调参** → [`03-泰坦尼克实践.md`](./03-泰坦尼克实践.md)
- **决策树自身参数详解**（`criterion / max_depth / min_samples_*`）→ [`../../04-DecisionTree/05-titanic/02-API.md`](../../04-DecisionTree/05-titanic/02-API.md)

### 术语版

| 故事里的元素 | 术语名 | 主场 |
|---|---|---|
| 请几位医生 | `n_estimators` 树的棵数 | 本节 |
| 每位医生每题看几个指标 | `max_features` 每节点候选特征数 | 本节 |
| 病例库抽样 vs 全量 | `bootstrap` 是否自助采样 | 本节 |
| 没被抽到的样本天然当验证集 | OOB / 袋外样本 out-of-bag（`oob_score`）| 本节 |
| 单棵树深度 / 叶子刹车 | `max_depth` / `min_samples_split` / `min_samples_leaf` | [`../../04-DecisionTree/05-titanic/02-API.md`](../../04-DecisionTree/05-titanic/02-API.md) |
| 分裂准则 | `criterion`（gini / entropy）| [`../../04-DecisionTree/05-titanic/02-API.md`](../../04-DecisionTree/05-titanic/02-API.md) |
| 随机种子 | `random_state` | — |

**这一节的关键启示**：**先拉 `n_estimators`，再调 `max_features`，最后才动 `max_depth`**。

→ 下一步：[`03-泰坦尼克实践.md`](./03-泰坦尼克实践.md) —— 这些旋钮在真实数据上怎么转

> Sources：
> - PPT Slide 20-21, 24
> - 笔记段：API + 调参直觉
> - sklearn ≥1.3 `max_features="auto"` 移除：sklearn 1.1 CHANGELOG
> - OOB 估计：sklearn.ensemble.RandomForestClassifier 文档 `oob_score` 参数说明
