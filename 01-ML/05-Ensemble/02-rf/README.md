# 第 2 章 · 随机森林

> 维度：**算法 + 代码**
>
> **按知识点拆分的讲解版**：
>
> 1. [`01-算法思想.md`](./01-算法思想.md) — 【理解】构建方法（两层随机性 + 伪代码 + 与 Bagging 的关系）
> 2. [`02-API.md`](./02-API.md) — 【知道】sklearn API（参数表 + max_features 坑 + OOB）
> 3. [`03-泰坦尼克实践.md`](./03-泰坦尼克实践.md) — 【实践】生存预测（6 步 pipeline + GridSearchCV + 特征重要性）

## ━━━━━━━━ 底稿 ━━━━━━━━

### PPT

> 从 [`../06集成学习.pptx`](../06集成学习.pptx) 提取（主 PPT Slide 14-25）。图占位标 〔图〕；排版整理；文字保留 PPT 原话。

#### Slide 14 · 目录（章扉页）

- 集成学习思想
- 随机森林算法
- Adaboost 算法
- GBDT
- XGBoost

#### Slide 15 · 学习目标

- 理解随机森林的构建方法
- 知道随机森林的 API
- 能够使用随机森林完成分类任务

#### Slide 16 · 随机森林算法定义

随机森林算法 —— 树多了，即为森林

随机森林是基于 Bagging 思想实现的一种集成学习算法，采用决策树模型作为每一个弱学习器。

**训练**：
- 有放回的产生随机训练样本（什么是有放回抽样？）
- 随机挑选 n 个特征（n 小于总特征数量，随机森林的要求）

**预测**：平权投票，多数表决输出预测结果

```
平权投票
决策树 1 ─┐
决策树 2 ─┼──→ 平权投票 → 最终预测
决策树 N ─┘
```

> **Notes**：基于前面的分类器进一步训练。前面的分类器，贡献就是不同样本权重。

#### Slide 17 · 随机森林步骤

〔图：随机森林构建步骤流程图〕

```
随机森林构建步骤：
────────────────────────────────────────
Step 1  从原始训练集有放回地抽取子集
Step 2  从全部特征中随机选取 k 个特征
Step 3  用子集 + k 特征训练一棵决策树
Step 4  重复 Step 1-3，训练 N 棵树
Step 5  新样本输入所有树 → 平权投票
────────────────────────────────────────
```

#### Slide 18 · 随机森林 – 概念（思考题）

**思考题 1**：为什么要随机抽样训练集？

**思考题 2**：为什么要有放回地随机抽样？

#### Slide 19 · 随机森林 – 概念（解答）

**思考题 1**：为什么要随机抽样训练集？→ 确保各学习器训练集"有差异"

如果不进行随机抽样，每棵树的训练集都一样，那么最终训练出的树分类结果也是完全一样。

**思考题 2**：为什么要有放回地随机抽样？→ 确保各学习器训练集"有交集"

如果不是有放回的抽样，那么每棵树的训练样本都是不同的，都是没有交集的，这样每棵树都是"有偏的"，也就是说每棵树训练出来都是有很大的差异的；而随机森林最后分类取决于多棵树（弱分类器）的投票表决。

综上：弱学习器的训练样本既有交集也有差异数据，更容易发挥投票表决效果

```
有放回抽样类比：
10 个学生各自从整本教材中随机抄了 80% 的内容（有重复）来复习。
→ 每个人都学过大部分核心概念，只是例题略有不同。
→ 考试时大家水平差不多，取平均值有意义。

无放回抽样类比：
10 个学生各自从整本教材中随机拿走了 10% 的内容（无重复）来复习。
→ 每个人仅关注一小部分概念，互不覆盖。
→ 考试时，每人只答各自学过的一部分题，取均值无意义。
```

#### Slide 20 · 随机森林 API

〔图：RandomForestClassifier API 参数说明截图〕

#### Slide 21 · 随机森林 API（续）

〔图：API 示例代码截图〕

> **Notes**：基于前面的分类器进一步训练。前面的分类器，贡献就是不同样本权重。

#### Slide 22 · 泰坦尼克号案例（代码 1）

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def dm01_随机森林():
    # 1 获取数据集
    titan = pd.read_csv("./data/titanic/train.csv")
    # 2 确定特征值和目标值
    x = titan[["Pclass", "Age", "Sex"]].copy()
    y = titan["Survived"]
    # 3-1 处理数据-处理缺失值
    x['Age'].fillna(value=titan["Age"].mean(), inplace=True)
    print(x.head())
    # 3-2 one-hot 编码
    x = pd.get_dummies(x)
    # 4 数据集划分
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, random_state=22, test_size=0.2)
```

> **Notes**：模型权重，用于...；样本权重，用于...

#### Slide 23 · 泰坦尼克号案例（代码 2）

```python
    # 5-1 使用决策树进行模型训练和评估
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    dtc_y_pred = dtc.predict(x_test)
    accuracy = dtc.score(x_test, y_test)
    print('单一决策树 accuracy-->\n', accuracy)

    # 5-2 随机森林进行模型训练和评估
    rfc = RandomForestClassifier(max_depth=6, random_state=9)
    rfc.fit(x_train, y_train)
    rfc_y_pred = rfc.predict(x_test)
    accuracy = rfc.score(x_test, y_test)
    print('随机森林 accuracy-->\n', accuracy)

    # 5-3 随机森林 交叉验证网格搜索 进行模型训练和评估
    estimator = RandomForestClassifier()
    param = {"n_estimators": [40, 50, 60, 70], "max_depth": [2, 4, 6, 8, 10], "random_state": [9]}
    grid_search = GridSearchCV(estimator, param_grid=param, cv=2)
    grid_search.fit(x_train, y_train)
    accuracy = grid_search.score(x_test, y_test)
    print("随机森林网格搜索 accuracy:", accuracy)
    print(grid_search.best_estimator_)
```

#### Slide 24 · 本章小结

**1 随机森林概念**

bagging 思想的代表算法，bagging + 决策树

**2 随机森林构建过程**

① 随机选数据、② 随机选特征、③ 训练弱学习器、④ 重复 1-3 训练 n 个、⑤ 平权投票

**3 随机森林 API**

`sklearn.ensemble.RandomForestClassifier()`

#### Slide 25 · 自检题

**1、请对下列随机森林的构建方法进行排序：**

- A）重复采样，构建出多棵决策树
- B）随机选取部分样本，并随机选取部分特征交给其中一棵决策树训练
- C）如果是分类场景则采用平权投票的方式决定最终随机森林的预测结果，如果是回归场景则采用简单平均法获取最终结果
- D）将相同的测试数据交给所有构建出来的决策树进行结果预测

**正确答案：B → A → D → C**

### 笔记

> 【理解】算法思想

**Random Forest**：基于 Bagging 思想 + 决策树作为基学习器的集成算法。

**两层随机性**：

1. **样本随机**：有放回抽样产生每棵树的训练集（bootstrap）
2. **特征随机**：每个节点分裂时，从全部特征中随机选 $k$ 个再挑最优（一般 $k = \sqrt{n}$）

**预测**：分类用平权投票，回归用平均。

**为什么不剪枝过拟合风险也较低**：两层随机保证树之间充分独立，方差被投票平均掉。

**关键超参**：

- 森林中树的数量 `n_estimators`（一般取较大值）
- 每节点候选特征数 `max_features`
- 树最大深度 `max_depth`（样本量大时建议限制）

**思考题速答**：

- 为什么要随机抽样？→ 不抽样则所有树训练集相同，结果完全一致，集成没意义
- 为什么要**有放回**？→ 无放回则各树训练集互斥、各自"片面"，差异过大反而难以聚合

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

> 【实践】泰坦尼克号生存预测

```python
# 流程骨架
# 1. 读 titanic.csv → 选特征 [Pclass, Age, Sex] → onehot
# 2. train_test_split
# 3. RandomForestClassifier().fit(X_train, y_train)
# 4. score / GridSearchCV 调 n_estimators + max_depth
```

**超参搜索**：用 `GridSearchCV` 在 `n_estimators ∈ {10, 50, 100, 200}`、`max_depth ∈ {3, 5, 8, None}` 上做 5 折 CV。
