# 第 3 章 · AdaBoost

> 维度：**算法 + 数学**
>
> **按知识点拆分的讲解版**：
>
> 1. [`01-简介.md`](./01-简介.md) — 【知道】算法思想
> 2. [`02-推导.md`](./02-推导.md) — 【理解】公式推导
> 3. [`03-构建过程.md`](./03-构建过程.md) — 【理解】手算构建过程
> 4. [`04-葡萄酒实战.md`](./04-葡萄酒实战.md) — 【实践】葡萄酒分类

## ━━━━━━━━ 底稿 ━━━━━━━━

### PPT

> 从 [`../06集成学习.pptx`](../06集成学习.pptx) 提取（主 PPT Slide 26-47）。图占位标 〔图〕；排版整理；文字保留 PPT 原话。

#### Slide 26 · 目录（章扉页）

- 集成学习思想
- 随机森林算法——bagging
- Boosting 算法之 Adaboost 算法（分类问题）
- Boosting 算法之 GBDT（回归问题）
- Boosting 算法之 XGBoost（回归问题）

> **Notes**：注意，第一到第十的样本，编号 id 从 0-9

#### Slide 27 · 学习目标

- 理解 adaboost 算法的思想
- 知道 adaboost 的构建过程
- 实践泰坦尼克号生存预测案例

#### Slide 28 · Boosting 思想回顾

〔图：Boosting 流程图，三个学习器串行〕

```
全部样本 → 学习器1 ─通信→ 学习器2 ─通信→ 学习器3
              ↓               ↓               ↓
         初步结果        修正的结论      再次修正的结论
```

> **Notes**：注意，第一到第十的样本，编号 id 从 0-9

#### Slide 29 · Adaboost 类比：珠宝真伪鉴定

〔图：珠宝鉴定类比——三位专家依次鉴定，后一位重点关注前一位的错误〕

```
全部样本 → 专家1鉴定 ─通信→ 专家2鉴定 ─通信→ 专家3鉴定
               ↓               ↓               ↓
          初步结果         修正的结论       再次修正的结论
```

#### Slide 30 · Adaboost 算法（核心思想）

**Adaptive Boosting（自适应提升）**：基于 Boosting 思想实现的一种集成学习算法，核心思想是通过逐步提高那些被前一步分类错误的样本的权重来训练一个强分类器。

〔图：二维特征空间 身高/体重 上的样本分布，线性分类器依次修正〕

1. 训练第一个弱学习器（性别分类）
2. 调整数据分布

```
身高
  ↑  ○ × ○   线性分类器 Y = ...
  |  × ○ ×
  |  ○ × ×
  +───────→ 体重

让下一个分类在训练的时候多关注一下错样本
分类错误的样本 → 权重放大 ↑
分类正确的样本 → 权重缩小 ↓
```

> **Notes**：集成算法，若为 SAMME.r 表示输出软标签，也就是概率

#### Slide 31 · Adaboost 算法（第 2 个弱学习器）

〔图：在调整后的样本权重分布上训练第 2 个弱学习器〕

3. 训练第二个弱学习器
4. 再次调整数据分布

```
蓝色区域：圈圈    红色区域：叉叉
（第 2 个学习器重点关注第 1 轮分错的样本）
```

#### Slide 32 · Adaboost 算法（整体过程）

〔图：多轮迭代示意图〕

5. 依次训练学习器，调整数据分布
6. 整体过程实现

AdaBoost 通过迭代训练一系列弱分类器，每一轮根据前一轮的表现，动态调整样本权重——分错的样本被"加重"，分对的被"减轻"，让后续分类器更关注难例。

**每个模型的发言权**（模型权重 α）：误差小的模型发言权大，误差大的模型发言权小。

#### Slide 33 · Adaboost 算法推导（流程）

〔图：三个模型串行，各自有权重〕

**1 初始化训练数据权重相等，训练第 1 个学习器**

- 如果有 100 个样本，则每个样本的初始化权重为：1/100
- 根据预测结果找一个错误率最小的分裂点（定义分割线）
- 计算、更新：分类错误率 / 模型权重 / 样本权重

**2 根据新权重的样本集训练第 2 个学习器**

- 根据预测结果找一个错误率最小的分裂点
- 计算、更新：分类错误率 / 模型权重 / 样本权重

**3 迭代训练**：在前一个学习器的基础上，根据新的样本权重训练当前学习器，直到训练出 m 个弱学习器

**4 m 个弱学习器集成预测公式：**

$$H(x) = \text{sign}\left(\sum_{t=1}^{m} \alpha_t h_t(x)\right)$$

- $\alpha$ 为模型的权重
- $h_i(x)$ 是每个模型的预测结果
- 最终输出结果 $H(x)$ 大于 0 则归为正类，小于 0 则归为负类

#### Slide 34 · Adaboost 算法推导（公式）

**模型权重计算公式：**

$$\alpha_t = \frac{1}{2} \ln\frac{1 - \varepsilon_t}{\varepsilon_t}$$

- $\varepsilon_t$ 为模型权重，表示第 $t$ 个弱学习器的错误率（样本中测错的占比，0~0.5，超过 0.5 则模型比随机猜测还糟）
- $\varepsilon_t$ 越小，模型越好，模型权重越大。当 $\varepsilon_t = 0$ 则 $\alpha_t$ 最大

**样本权重计算公式：**

$$D_{t+1}(i) = \frac{D_t(i)}{Z_t} \times \begin{cases} e^{-\alpha_t} & \text{分类正确} \\ e^{\alpha_t} & \text{分类错误} \end{cases}$$

- $D_t(i)$ 为上一轮样本权重
- $Z_t$ 为归一化值（上一轮所有样本权重总和）
- $\alpha_t$ 为模型权重

> **Notes**：本页讲完，直接将补充材料（讲师备注，指补充材料 PPT）

#### Slide 35 · 同质要求说明

基分类器必须是同一种，比如都是逻辑回归或都是决策树。否则称为"异构集成"。

> **Notes**：本页讲完，直接将补充材料

#### Slide 36 · Adaboost 构建过程举例（题目）

已知训练数据见下面表格，假设弱分类器由 $x$ 产生，即用若干个决策树，预测结果使该分类器在训练数据集上的分类误差率最低，试用 Adaboost 算法学习一个强分类器。

〔表格：10 个样本，一维特征 x，标签为正例（+1）/ 负例（-1）〕

正例 / 负例

> **Notes**：本页讲完，直接将补充材料

#### Slide 37 · 构建第 1 个弱分类器（初始化）

〔表格：特征 x / 权重（初始均匀）/ 标签〕

标签约定：
- 左侧（$x \leq$ 分裂点）→ 预测 +1
- 右侧（$x >$ 分裂点）→ 预测 -1（依次类推各分裂点方向）

#### Slide 38 · 构建第 1 个弱分类器（权重更新）

〔表格：特征 x / 更新后权重 / 标签〕

```
该模型的发言权：α₁ = 0.5 × ln((1-ε₁)/ε₁)

权重更新：
  Zt 发生变更
  初始值 0.1 × e^alpha（分类错误样本）
         0.1 × e^(-alpha)（分类正确样本）
  重新归一化（发现总权重不再为 1）
```

> **Notes**：本页讲完，直接将补充材料

#### Slide 39 · 构建第 2 个弱学习器（权重）

〔表格：特征 x / 更新后权重（错误样本权重更大）/ 标签〕

> **Notes**：本页讲完，直接将补充材料

#### Slide 40 · 构建第 2 个弱学习器（计算）

〔图：第 2 轮加权错误率计算，样本 3、4、5 被重点关注〕

```
第 2 轮：
  ε₂ = Σ D₂(i) × 𝟙[h₂(xᵢ) ≠ yᵢ]
  在新权重下找最优分裂点
  计算 α₂
```

#### Slide 41 · 构建第 3 个弱学习器

〔图：第 3 轮构建，在第 2 轮更新权重后的样本分布上训练〕

#### Slide 42 · 最终强学习器

每个学习器都基于前一个学习器的权重信息。

$$H(x) = \text{sign}(0.4236 \cdot h_1(x) + 0.6496 \cdot h_2(x) + 0.7515 \cdot h_3(x))$$

**验证**：$X = 3.5$ 带入公式：

$$0.4236 \times (-1) + 0.6496 \times (+1) + 0.7514 \times (-1) = -0.5254 < 0 \Rightarrow \text{负类}$$

#### Slide 43 · 案例 AdaBoost 实战葡萄酒数据（需求 + API）

**需求**：已知葡萄酒数据，根据数据进行葡萄酒分类

**API**：

```python
myada = AdaBoostClassifier(
    base_estimator=mytree,   # 参1：弱分类器（事先定义好的决策树对象）
    n_estimators=500,         # 参2：弱分类器个数
    learning_rate=0.1,        # 参3：学习率，作用于模型权重，从而影响样本权重的更新快慢
)
```

- `learning_rate`：0 ~ +∞
- 用于增大或缩小每个权重的贡献，从而影响了样本权重的更新快慢

#### Slide 44 · AdaBoost 实战葡萄酒数据（代码 1）

```python
# AdaBoost 实战葡萄酒数据
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier  # 集成学习
from sklearn.metrics import accuracy_score

def dm01_adaboost():
    # 1 读数据到内存 df_wine
    df_wine = pd.read_csv('./data/wine0501.csv')

    # 2 特征处理
    # 2-2 Adaboost 一般做二分类，去掉一类 (1,2,3)
    df_wine = df_wine[df_wine['Class label'] != 1]

    # 2-3 准备特征值和目标值  Alcohol 酒精含量  Hue 颜色
    x = df_wine[['Alcohol', 'Hue']].values
    y = df_wine['Class label']

    # 2-4 类别转化 y (2,3) => (0,1)
    y = LabelEncoder().fit_transform(y)

    # 2-5 划分数据
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=22, test_size=0.2)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

**思路分析**：
```
# 1 读数据到内存
# 2 特征处理
#   2-1 修改列名
#   2-2 Adaboost 一般做二分类，去掉过多的类别（比如有类别 1,2,3）
#   2-4 类别转化 (2,3)=>(0,1)  注意 (0,1) 标签与 (-1,1) 标签均可接受
#   2-5 划分数据
# 3 实例化单决策树 实例化 Adaboost-由 500 颗树组成
# 4 单决策树训练和评估
# 5 AdaBoost 训练和评估
```

如果基学习器太强，则集成学习不一定能得到更好结果。

#### Slide 45 · AdaBoost 实战葡萄酒数据（代码 2）

```python
    # 3 实例化单决策树 实例化 Adaboost-由 500 颗树组成
    mytree = DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=0)
    myada = AdaBoostClassifier(base_estimator=mytree, n_estimators=500, learning_rate=0.1, random_state=0)

    # 4 单决策树训练和评估
    mytree.fit(X_train, y_train)
    myscore = mytree.score(X_test, y_test)
    print('myscore-->', myscore)

    # 5 AdaBoost 训练和评估
    myada.fit(X_train, y_train)
    myscore = myada.score(X_test, y_test)
    print('myscore-->', myscore)
```

#### Slide 46 · 本章小结

**1 Adaboost 概念**

通过逐步提高被分类错误的样本的权重来训练一个强分类器。提升的思想。

**2 Adaboost 构建过程**

1. 初始化数据权重，来训练第 1 个弱学习器。找最小的错误率计算模型权重，再更新数据权重。
2. 根据更新的数据集权重，来训练第 2 个弱学习器，再找最小的错误率计算模型权重，再更新数据权重。
3. 依次重复第 2 步，训练 n 个弱学习器。组合起来进行预测。结果大于 0 为正类、结果小于 0 为负类。

#### Slide 47 · 自检题

**1、下列关于 Adaboost 的说法正确的是？（多选）**

- A）Adaboost 算法一般用来做二分类，特别在视觉领域应用较多
- B）AdaBoost 算法不能提高精度
- C）AdaBoost 算法 API 函数可以配置学习率参数，学习率参数作用于每一颗树的数据权重更新上
- D）AdaBoost 算法使用的树深度不要过深，否则容易过拟合

**答案：AD**

### 笔记

> 【知道】AdaBoost 算法简介

**Adaptive Boosting**：基于 Boosting 思想，通过**逐步提高被前一轮分类错误样本的权重**来训练强分类器。

**两个动态权重**：

| 对象 | 调整方式 |
|---|---|
| **样本权重** $D_t$ | 错分样本权重 ↑，正确样本权重 ↓ |
| **学习器权重** $\alpha_t$ | 误差率小 → 权重大；误差率大 → 权重小 |

**核心步骤**：

- **权值调整**：错分样本权重提高，让后续学习器更关注难样本
- **加权多数表决**：好学习器权重大，差的小

**同质要求**：所有基学习器必须同类型（如都是决策桩 / 都是 LR）；混搭不同类型称为"异构集成（heterogeneous ensemble）"，不属于 AdaBoost。

> 【理解】AdaBoost 算法推导

**强学习器**（分类输出 $> 0$ 为正类，$< 0$ 为负类）：

$$H(x) = \mathrm{sign}\left(\sum_{t=1}^{m} \alpha_t h_t(x)\right)$$

其中 $\alpha_t$ 是第 $t$ 个弱学习器的权重，$h_t(x) \in \{-1, +1\}$。

**模型权重公式**：

$$\alpha_t = \frac{1}{2} \ln\frac{1 - \varepsilon_t}{\varepsilon_t}$$

$\varepsilon_t = \sum_{i=1}^{N} D_t(i) \cdot \mathbb{1}[h_t(x_i) \neq y_i]$ 是第 $t$ 个弱学习器在当前样本权重下的加权错误率。$\varepsilon_t < 0.5$ 时 $\alpha_t > 0$，越准 $\alpha_t$ 越大。

**样本权重更新公式**：

$$D_{t+1}(i) = \frac{D_t(i)}{Z_t} \cdot \begin{cases} e^{-\alpha_t} & h_t(x_i) = y_i \\ e^{\alpha_t} & h_t(x_i) \neq y_i \end{cases}$$

- $Z_t$：归一化因子（保证 $\sum_i D_{t+1}(i) = 1$）
- 正确样本权重乘 $e^{-\alpha_t} < 1$（衰减），错误样本乘 $e^{\alpha_t} > 1$（放大）

> 【理解】AdaBoost 构建过程

**示例**：10 个样本一维特征，弱分类器为单一阈值决策桩（decision stump）。

**第 1 轮**：

1. 初始权重均匀：$D_1(i) = 0.1$
2. 遍历分裂点 $\{0.5, 1.5, \ldots, 8.5\}$，找加权错误率最小者：以 2.5 为分裂点，3 个错样本，$\varepsilon_1 = 0.3$
3. $\alpha_1 = \frac{1}{2}\ln\frac{0.7}{0.3} \approx 0.4236$
4. 更新权重：正确样本 $\times e^{-0.4236} \approx 0.6547$；错误样本 $\times e^{0.4236} \approx 1.5275$
5. 归一化 $Z_1 = 0.06547 \times 7 + 0.15275 \times 3 \approx 0.9165$
6. 最终：正确样本权重 $\approx 0.0714$，错误样本 $\approx 0.1667$

**第 2 轮**：在新权重下找最优分裂点 → 8.5，$\varepsilon_2 \approx 0.214$，$\alpha_2 \approx 0.6496$ → 再更新权重。

**第 3 轮**：$\varepsilon_3 \approx 0.182$，$\alpha_3 \approx 0.7514$。

**强学习器**：$H(x) = \mathrm{sign}(0.4236 h_1 + 0.6496 h_2 + 0.7514 h_3)$。

**注意**：第 $t+1$ 轮选分裂点时，错误率是**加权**的（$\sum D_t(i) \cdot \mathbb{1}[h(x_i) \neq y_i]$），不是简单计数。

> 【实践】AdaBoost 葡萄酒数据

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # 决策桩；sklearn <1.2 用 base_estimator=
    n_estimators=500,
    learning_rate=0.1,
)
ada.fit(X_train, y_train)
```

**`learning_rate`**：缩放每轮模型权重 $\alpha_t$ 的贡献，间接影响样本权重更新速度。范围 $(0, +\infty)$，越小需要越多轮 `n_estimators` 配合。

**两条实战提醒**：
- 标签编码 (0,1) 或 (-1,1) sklearn 均可接受
- 基学习器若过强（深树），集成反而可能更差，决策桩 `max_depth=1` 是常用起点

**结论**：AdaBoost 比单决策树过拟合更轻、测试集略胜，与 bagging 准确率相近。
