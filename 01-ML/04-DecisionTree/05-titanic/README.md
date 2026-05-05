# 第 5 章 · 泰坦尼克号案例

> 维度：**代码**
>
> **按知识点拆分的讲解版**：
>
> 1. [`01-案例背景.md`](./01-案例背景.md) — 【实践】数据与目标 + 字段速查
> 2. [`02-API.md`](./02-API.md) — 【知道】`DecisionTreeClassifier` 5 参数 + 三把刹车
> 3. [`03-实现.md`](./03-实现.md) — 【实践】6 步 pipeline + plot_tree 节点解读

## ━━━━━━━━ 底稿 ━━━━━━━━

### PPT

> 从 [`../05决策树.pptx`](../05决策树.pptx) 提取（Slide 59-65）。图占位标 〔图〕；排版整理；文字保留 PPT 原话。

#### Slide 59 · 章节索引

- 决策树简介
- ID3 决策树
- C4.5 决策树
- CART 决策树
- 案例：泰坦尼克号生存预测
- CART 回归树
- 决策树剪枝

#### Slide 60 · 学习目标

- 知道分类决策树 API 函数
- 完成泰坦尼克号生存预测的案例

#### Slide 61 · 决策树 API 介绍

```
┌─────────────────────────────────────────────────┐
│  sklearn.tree.DecisionTreeClassifier            │
├─────────────────────────────────────────────────┤
│  criterion          = 'gini'   分裂准则          │
│  max_depth          = None     最大深度          │
│  min_samples_split  = 2        节点分裂阈值      │
│  min_samples_leaf   = 1        叶子样本下限      │
│  random_state       = None     随机种子          │
└─────────────────────────────────────────────────┘
        ↑ 后三个参数 = 防过拟合的"刹车"
```

```python
class sklearn.tree.DecisionTreeClassifier(
    criterion='gini', max_depth=None, random_state=None
)
```

- **Criterion**：特征选择标准
  - "gini" 或 "entropy"，前者代表基尼系数，后者代表信息增益。默认 "gini"，即 CART 算法
- **min_samples_split**：父节点再划分所需最小样本数，样本数过少就不再分裂
- **min_samples_leaf**：叶子节点最少样本数，如果比此值小，则不进行分裂
- **max_depth**：决策树最大深度    根节点（原数据）设为 0

特征 a：书本颜色（3 本科技书 / 3 本言情小说）（蓝 = α / 红 = β / 2B / 3A 1B）

> **Notes**：
> - **min_samples_split**：类比 你要开分公司，但总部规定：至少有 50 个员工才能分家。如果只有 3 个人，就别折腾了！
> - **min_samples_leaf**：法庭判案不能只听一个人证词。每个判决（叶子）至少要有 5 个可靠证人（样本）支持！
> - **max_depth**：就像你做决定时，最多只问自己 3 个问题就得出结论。如果允许问 20 个问题，可能会钻牛角尖！

#### Slide 62 · 不限制 → 过拟合

如果没有这些参数的限制，决策树会：

- 无限分裂，让每个叶节点里面的样本都是纯净的单一样本
- → 模型太复杂，从而**过拟合**

因此这些参数需要对模型进行约束。

7 yes / 3 no  →  婚史 → 是 / 否  →  2 no 2 yes / 5 no 1 yes

```
不限制 max_depth / min_samples_* → 树会一直分到每片叶子只剩 1 个样本

         根：7 yes / 3 no
              │
       ┌──────┴──────┐
      婚史=是         婚史=否
      2y 2n          5n 1y
        │              │
     ┌──┴──┐        ┌──┴──┐
     学历?           收入?
     ...            ...
       │              │
   ┌───┼───┐      ┌───┼───┐
   年龄? 籍贯? ...  车? 房? ...
     │              │
    ...            ...
     │              │
  ┌──┴──┐        ┌──┴──┐
  [1 yes]  [1 no]   [1 yes]  [1 no]   ← 每片叶子只剩 1 个样本

  ✓ 训练集 100% 准确    ✗ 测试集泛化差 → 过拟合
```

上述 3 个参数限制树的深度。过深，容易过拟合。

> 就像你做决定时，最多只问自己 3 个问题就得出结论。如果允许问 20 个问题，可能会钻牛角尖！

#### Slide 63 · 案例背景

**1912 年 4 月 15 日**，在她的处女航中，泰坦尼克号在与冰山相撞后沉没，在 2224 名乘客和机组人员中造成 1502 人死亡。

造成海难失事的原因之一是乘客和机组人员没有足够的救生艇。尽管幸存生存有一些运气因素，但有些人比其他人更容易生存，例如妇女，儿童和上流社会。

有了遇难和幸运数据，运用机器学习工具来预测哪些乘客可幸免于悲剧。

**数据情况**：

- 数据集中的特征包括票的类别、是否存活、乘坐班次、年龄、登陆、home.dest、房间、船和性别等
- 乘坐班是指乘客班（1，2，3），是社会经济阶层的代表
- age 数据存在缺失

```
1912.04.15  RMS Titanic  ❄ 撞冰山沉没
─────────────────────────────────────────
  乘客 + 机组：2224       遇难：1502
  存活率：约 32%          女/童/上流：偏高

  数据片段（train.csv）：
  ┌─────────┬────────┬──────┬──────┬───────────┐
  │ Pclass  │ Sex    │ Age  │ ...  │ Survived  │
  ├─────────┼────────┼──────┼──────┼───────────┤
  │   3     │ male   │ 22.0 │ ...  │    0      │
  │   1     │ female │ 38.0 │ ...  │    1      │
  │   3     │ female │ 26.0 │ ...  │    1      │
  │   1     │ female │ 35.0 │ ...  │    1      │
  │   3     │ male   │  NaN │ ...  │    0      │  ← Age 有缺失
  │  ...    │  ...   │ ...  │ ...  │   ...     │
  └─────────┴────────┴──────┴──────┴───────────┘
        x（特征）                      y（标签）
```

#### Slide 64 · 代码

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def dm04_泰坦尼克():
    # 1 读数据到内存
    taitan_df = pd.read_csv("./data/titanic/train.csv")
    taitan_df.head()         # 查看前 5 条数据
    taitan_df.info()         # 查看特性信息

    # 2 数据基本处理
    # 2-1 确定 x y
    x = taitan_df[['Pclass', 'Age', 'Sex']]
    y = taitan_df['Survived']

    # 2-2 缺失值处理
    x['Age'].fillna(x['Age'].mean(), inplace=True)

    # 2-3 pclass 类别型数据,需要转数值 one-hot 编码
    print('x-->1\n', x)
    x.info()
    x = pd.get_dummies(x)
    print('x-->2\n', x)
    x.info()

    # 2-4 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=33)

    # 3 实例化决策树模型 训练模型
    estimator = DecisionTreeClassifier()
    estimator.fit(x_train, y_train)

    # 4 模型预测
    y_pred = estimator.predict(x_test)

    # 5 模型评估
    # 5-1 输出预测准确率
    myret = estimator.score(x_test, y_test)
    print('myret-->\n', myret)

    # 5-2 更加详细的分类性能
    myreport = classification_report(
        y_pred, y_test, target_names=['died', 'survived'])
    print('myreport-->\n', myreport)

    # 6 决策树可视化
    plt.figure(figsize=(30, 20))
    plot_tree(estimator,
              max_depth=10,
              filled=True,
              feature_names=['Pclass', 'Age', 'Sex_female', 'Sex_male'],
              class_names=['died', 'survived'])
    plt.show()
```

> **Notes**：看一下报告结果，说明什么问题。

#### Slide 65 · 树图节点含义

```
plot_tree() 输出的单个节点（4 行格式）：

      ┌──────────────────────────┐
      │  x[2] <= 0.5             │  ← 分裂条件（特征索引 + 阈值）
      │  gini  = 0.473           │  ← 节点纯度（0 = 纯，0.5 = 最杂）
      │  samples = 712           │  ← 落到这个节点的样本数
      │  value = [439, 273]      │  ← 各类样本数 [died, survived]
      └────────────┬─────────────┘
                   │
           ┌───────┴───────┐
        True (≤0.5)    False (>0.5)
           ↓               ↓
      ┌─────────┐     ┌─────────┐
      │ x[1]…   │     │ x[0]…   │
      │ gini=…  │     │ gini=…  │
      │ samples │     │ samples │
      │ value=… │     │ value=… │
      └─────────┘     └─────────┘

  特征索引（feature_names=['Pclass','Age','Sex_female','Sex_male']）：
    x[0]=Pclass  x[1]=Age  x[2]=Sex_female  x[3]=Sex_male
```

**图的节点含义**：

- **x[2]**：通过哪个特征进行分裂
- **gini**：是多少
- **sample**：总共多少个
- **value**：当前节点中，每个分类（幸存 / 罹难）的数目

### 笔记

> 05 · 泰坦尼克号生存预测

**学习目标**：

1. 能用 sklearn 决策树解决二分类问题
2. 掌握关键超参数的工程含义

> 【实践】案例背景

1912 年泰坦尼克号沉没，2224 人中 1502 人遇难。任务：根据**票类、班次、姓名、年龄、上船港口、房间、性别**等特征，预测乘客是否生还。

**业务侧观察**：妇女、儿童、社会地位高者存活率显著更高 → 决策树天然适合表达这类"分群规则"。

**特征工程要点**：

- 类别特征（Sex、Embarked）用 `pd.get_dummies` 或 `OrdinalEncoder`
- 缺失值（Age、Cabin）填充或丢弃
- 决策树**不需要**标准化

> 【知道】API 介绍

```python
sklearn.tree.DecisionTreeClassifier(
    criterion='gini',       # 'gini' (CART) 或 'entropy' (ID3 风格)
    max_depth=None,         # 树最大深度，控过拟合
    min_samples_split=2,    # 节点继续分裂的最小样本数
    min_samples_leaf=1,     # 叶子节点最少样本数
    random_state=None,
)
```

**关键参数工程含义**：

| 参数 | 作用 | 调参直觉 |
|---|---|---|
| `criterion` | 分裂准则 | gini 默认（更快）；entropy 在某些数据上略好 |
| `max_depth` | 限制深度 | 数据量大 / 特征多时设 10-100；过深易过拟合 |
| `min_samples_split` | 内部节点最小分裂样本 | 大数据集（10w+）可设 10 |
| `min_samples_leaf` | 叶子最少样本 | 大数据集设 5+，避免叶子过碎 |
| `random_state` | 随机种子 | 复现实验 |

> 【实践】案例实现

**Pipeline 骨架**：

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. 加载 + 清洗
df = pd.read_csv('titanic.csv')
df = df[['Pclass', 'Sex', 'Age', 'Survived']].dropna()
X = pd.get_dummies(df[['Pclass', 'Sex', 'Age']])
y = df['Survived']

# 2. 切分
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 训练
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_tr, y_tr)

# 4. 评估
print(accuracy_score(y_te, clf.predict(X_te)))
```

**决策树的产品化优势**：

- **可解释性**：可视化（`sklearn.tree.plot_tree`）后能给业务方看每条规则。每个节点显示 4 项信息（Slide 65）：`x[i]` 当前分裂特征、`gini` 节点纯度、`samples` 当前样本总数、`value` 各类样本数（如 `[died, survived]`）
- **无需标准化 / 量化**：开箱即用
- **混合类型友好**：数值 + 类别都能处理
- **快推理**：从根到叶 $O(\log n)$
