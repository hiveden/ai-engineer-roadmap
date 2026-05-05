# 第 5 章 · XGBoost

> 维度：**算法 + 数学 + 代码**
>
> **按知识点拆分的讲解版**：
>
> 1. [`01-算法思想.md`](./01-算法思想.md) — 【知道】对 GBDT 的改进（4 大改进 + RF/GBDT/XGB 三足鼎立对比）
> 2. [`02-目标函数.md`](./02-目标函数.md) — 【理解】推导（二阶泰勒 → 叶子最优值 → 打分函数 → 分裂 Gain）
> 3. [`03-API.md`](./03-API.md) — 【了解】XGBClassifier（参数表 + 早停 + DMatrix 选型 + GPU）
> 4. [`04-红酒预测.md`](./04-红酒预测.md) — 【实践】完整 pipeline（不均衡处理 + GridSearch + 特征重要性 + 三方选型）

## ━━━━━━━━ 底稿 ━━━━━━━━

### PPT

> 从 [`../06集成学习.pptx`](../06集成学习.pptx) 提取（主 PPT Slide 67-92）+ 从 [`../06集成学习补充材料.pptx`](../06集成学习补充材料.pptx) 提取（补充 Slide 16-30）。图占位标 〔图〕；排版整理；文字保留 PPT 原话。

#### Slide 67 · 目录（章扉页）

- 集成学习思想
- 随机森林算法
- Adaboost 算法
- GBDT
- XGBoost

#### Slide 68 · 学习目标

- 知道 XGBoost 算法的思想
- 理解 XGBoost 目标函数
- 了解 XGBoost 的算法 API
- 实现红酒品质预测案例

#### Slide 69 · XGBoost 思想（引入）

**XGBoost（Extreme Gradient Boosting）**：极端梯度提升树，集成学习方法的王牌，在数据挖掘比赛中，大部分获胜者用了 XGBoost。

**Xgb 的构建思想**：

1. 构建模型的方法是最小化训练数据的损失函数：训练的模型复杂度较高，易过拟合。
2. 在损失函数中加入正则化项，提高对未知的测试数据的泛化性能。

#### Slide 70 · XGBoost 正则化项

**XGBoost（Extreme Gradient Boosting）**：是对 GBDT 的改进，并且在损失函数中加入了正则化项

正则化项用来降低模型的复杂度

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2$$

- $\gamma T$：$T$ 表示一棵树的叶子结点数量
- $\lambda$：$w$ 表示叶子结点输出值组成的向量，向量的模；$\lambda$ 对该项的调节系数

#### Slide 71 · XGBoost 直觉例子（电子游戏喜好）

假设我们要预测一家人对电子游戏的喜好程度，考虑到年轻和年老相比，年轻更可能喜欢电子游戏，以及男性和女性相比，男性更喜欢电子游戏，故先根据年龄大小区分小孩和大人，然后再通过性别区分开是男是女，逐一给各人在电子游戏喜好程度上打分：

〔图：多个特征（年龄/性别）构建树的示意〕

```
树 1（年龄特征）：
  小孩 → 得分 +2
  大人 → 得分 -1

树 2（性别特征）：
  男性 → 得分 +0.9
  女性 → 得分 -0.9

最终得分 = 树1得分 + 树2得分（累加）
```

#### Slide 72 · XGBoost 两棵树累加

训练出 tree1 和 tree2，类似之前 gbdt 的原理，两棵树的结论累加起来便是最终的结论。

〔图：tree1 + tree2 的结构，多个特征，复杂度表示为正则项〕

树 tree1 的复杂度表示为 $\Omega(f_1)$（叶子数惩罚 + L2 正则）

#### Slide 73 · XGBoost 目标函数

进行 $t$ 次迭代的学习模型的目标函数如下：

$$\text{Obj}^{(t)} = \sum_{i=1}^{n} L(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t) + \text{const}$$

直接对目标函数求解比较困难，通过泰勒展开将目标函数换一种近似的表示方式。

#### Slide 74 · XGBoost（复习）泰勒展开

**泰勒展开**：将一个函数在某一点处展开成无限项的多项式表达式（用一系列表达式，等价于 $f(x)$ 的值）

**一阶泰勒展开**：

$$f(x + \Delta x) \approx f(x) + f'(x) \Delta x$$

**二阶泰勒展开**：

$$f(x + \Delta x) \approx f(x) + f'(x) \Delta x + \frac{1}{2} f''(x) \Delta x^2$$

#### Slide 75 · XGBoost 目标函数推导 2——泰勒展开

目标函数对 $\hat{y}_i^{(t-1)}$ 进行泰勒二阶展开，得到如下近似表示的公式：

$$L(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) \approx L(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)$$

观察目标函数，发现以下两项表示 $t-1$ 个弱学习器构成学习器的目标函数，都是常数，我们可以将其去掉：

$$\text{Obj}^{(t)} \approx \sum_{i=1}^{n} \left[g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)\right] + \Omega(f_t)$$

其中 $g_i$ 和 $h_i$ 分别为损失函数的一阶导、二阶导：

$$g_i = \frac{\partial L(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}, \quad h_i = \frac{\partial^2 L(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2}$$

#### Slide 76 · XGBoost 从样本角度转为叶子节点角度

从样本角度转为按照叶子节点输出角度，优化损失函数：

举个栗子：$m = 10$ 个样本，落在 D 结点 3 个样本，落在 E 结点 2 个样本，落在 F 结点 2 个样本，落在 G 结点 3 个样本；计算 $f_t(x_i)$ 的表达形式如下：

```
样本 → 叶子节点映射：
  D 结点（3 个样本）: w_D
  E 结点（2 个样本）: w_E
  F 结点（2 个样本）: w_F
  G 结点（3 个样本）: w_G
```

上式中：
- $g_i$ 表示每个样本的一阶导，$h_i$ 表示每个样本的二阶导
- $f_t(x_i)$ 表示样本的预测值
- $T$ 表示叶子结点的数目
- $\|w\|^2$ 由叶子结点值组成向量的模

#### Slide 77 · XGBoost 目标函数转化

目标函数中的各项可以做以下转换：

$$g_i f_t(x_i) \Rightarrow \text{样本的预测值 } w_{q(x_i)}$$

$$h_i f_t^2(x_i) \Rightarrow \text{转换从叶子结点的角度看}$$

$$\lambda\|w\|^2 \Rightarrow \text{从叶子角度来看}$$

汇总得到：

$$\text{Obj}^{(t)} = \sum_{j=1}^{T} \left[G_j w_j + \frac{1}{2}(H_j + \lambda)w_j^2\right] + \gamma T$$

#### Slide 78 · XGBoost 目标函数推导 3——转化为叶子节点输出角度

令：

$$G_j = \sum_{i \in I_j} g_i \quad \text{（所有落在叶子 j 的样本一阶导之和）}$$

$$H_j = \sum_{i \in I_j} h_i \quad \text{（所有落在叶子 j 的样本二阶导之和）}$$

最终：

$$\text{Obj}^{(t)} = \sum_{j=1}^{T} \left[G_j w_j + \frac{1}{2}(H_j + \lambda)w_j^2\right] + \gamma T$$

#### Slide 79 · XGBoost 目标函数推导 4——最优解

**求损失函数最小值**

对 $w_j$ 求导并令其等于 0，可得到 $w$ 的最优值：

$$\frac{\partial}{\partial w_j}\left[G_j w_j + \frac{1}{2}(H_j + \lambda)w_j^2\right] = G_j + (H_j + \lambda)w_j = 0$$

$$w_j^* = -\frac{G_j}{H_j + \lambda}$$

最优 $w^*$ 带入公式可求目标函数的最小值：

$$\text{Obj}^* = -\frac{1}{2}\sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma T$$

#### Slide 80 · XGBoost 打分函数

目标函数最终为：

$$\text{Obj}^* = -\frac{1}{2}\sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma T$$

该公式也叫做**打分函数（scoring function）**，从损失函数、树的复杂度两个角度来衡量一棵树的优劣。当我们构建树时，可以用来选择树的划分点：

**分裂增益（Gain）**：

$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma$$

#### Slide 81 · XGBoost 分裂判断

根据上一页 PPT 中计算的 gain 值：

对树中的每个叶子结点尝试进行分裂，计算分裂前 - 分裂后的分数：

- 如果 $\text{Gain} > 0$，则分裂之后树的损失更小，会考虑此次分裂
- 如果 $\text{Gain} < 0$，说明分裂后的分数比分裂前的分数大，此时不建议分裂

**当触发以下条件时停止分裂**：
- 达到最大深度
- 叶子结点数量低于某个阈值
- 所有的结点在分裂不能降低损失
- 等等...

#### Slide 82 · XGBoost 算法 API

**XGB 的安装和使用**：

在 sklearn 机器学习库中没有集成 xgb。想要使用 xgb，需要手工安装：

```bash
pip3 install xgboost
# 可以在 xgb 的官网上查看最新版本：https://xgboost.readthedocs.io/en/latest/
```

**XGB 的编码风格**：
- 支持非 sklearn 方式，也即是自己的风格
- 支持 sklearn 方式，调用方式保持 sklearn 的形式

#### Slide 83 · xgb 案例：红酒品质分类（需求）

**已知**：数据集共包含 11 个特征，共计 3269 条数据。我们通过训练模型来预测红酒的品质，品质共有 6 个类别，分别使用数字：0、1、2、3、4、5 来表示

**需求**：对红酒品质进行多分类

**分析**：
1. 目标是多分类
2. 数据存在样本不均衡问题

#### Slide 84 · 案例：红酒品质分类（代码 1）

```python
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

# 基本数据处理
def dm01_realdata():
    # 1 加载训练集
    data = pd.read_csv('./data/红酒品质分类.csv')
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1] - 3  # 标签从 3-8 → 0-5

    # 2 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=22
    )

    # 3 数据存储
    pd.concat([x_train, y_train], axis=1).to_csv('data/红酒品质分类-train.csv')
    pd.concat([x_test, y_test], axis=1).to_csv('data/红酒品质分类-test.csv')
```

#### Slide 85 · xgb 案例：红酒品质分类（基本程序框架）

〔图：程序整体框架（dm01 / dm02 / dm03 / dm04 流程）〕

#### Slide 86 · xgb 案例：红酒品质分类（代码 2——基础训练）

```python
def dm02_训练模型():
    # 1 加载数据集
    train_data = pd.read_csv('./data/红酒品质分类-train.csv')
    test_data = pd.read_csv('./data/红酒品质分类-test.csv')

    # 2 准备数据
    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # 3 xgb 模型训练
    estimator = xgb.XGBClassifier(
        n_estimators=100, objective='multi:softmax',
        eval_metric='merror', eta=0.1,
        use_label_encoder=False, random_state=22
    )
    estimator.fit(x_train, y_train)

    # 4 xgb 模型评估
    y_pred = estimator.predict(x_test)
    print(classification_report(y_true=y_test, y_pred=y_pred))

    # 5 模型保存
    joblib.dump(estimator, './data/mymodelxgboost.pth')
```

#### Slide 87 · xgb 案例（代码 3——样本不均衡处理）

```python
from sklearn.utils import class_weight

def dm03_训练模型():
    # 1 加载数据集
    train_data = pd.read_csv('./data/红酒品质分类-train.csv')
    test_data = pd.read_csv('./data/红酒品质分类-test.csv')

    # 2 准备数据
    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    # 2-2 样本不均衡问题处理
    classes_weights = class_weight.compute_sample_weight(
        class_weight='balanced', y=y_train
    )

    # 3 xgb 模型训练
    estimator = xgb.XGBClassifier(
        n_estimators=100, objective='multi:softmax',
        eval_metric='merror', eta=0.1,
        use_label_encoder=False, random_state=22
    )
    # 训练的时候，指定样本的权重
    estimator.fit(x_train, y_train, sample_weight=classes_weights)

    # 4 xgb 模型评估
    y_pred = estimator.predict(x_test)
    print(classification_report(y_true=y_test, y_pred=y_pred))
```

#### Slide 88 · 本章小结

**1 xgboost 的目标函数**

$$\text{Obj}^* = -\frac{1}{2}\sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma T$$

**2 xgboost 的模型复杂度**

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2$$

**3 xgboost API**

```python
XGBClassifier(n_estimators, max_depth, learning_rate, objective)
```

#### Slide 89 · 自检题（1-2）

**1、下列关于 XGBoost 的描述错误的是？**

- A）它是极限梯度提升树（Extreme Gradient Boosting）的缩写
- B）它在数据挖掘方面拥有更好的性能
- C）Xgboost 使用了正则化
- D）xgboost 算法不可以使用线性模型进行集成

**答案：D**

**2、下列关于 XGBoost 损失函数的正则化项描述错误的是？**

- A）它使用的是 CART 回归树作为弱学习器
- B）它的正则化项只包含一棵树的结果
- C）它的正则化项由树的叶子节点的个数以及 L2 正则化项组成
- D）模型可以通过超参数来调整正则化项对模型的惩罚力度

**答案：B**

#### Slide 90 · 自检题（3-4）

**3、下列关于 XGBoost 损失函数的描述错误的是？**

- A）它的第 T 棵树的损失与第 T-1 棵树无关
- B）在求第 T 棵树的结构时可将前 T-1 棵树的结构作为常数
- C）它使用了二阶泰勒展开式去近似目标函数
- D）最终得出的损失函数值越小代表模型的效果越好

**答案：A**

**4、下列关于 XGBoost 树的描述错误的是？**

- A）它可以使用打分函数确定某个节点是否能够继续分裂
- B）它可以使用打分函数确定某个特征的最佳分割点
- C）最大树深度和最小叶子节点样本数可以用来调节树结构
- D）超参数 gamma 的大小对树结构没有影响

**答案：D**

#### Slide 91 · 总结图示

〔图：集成学习整体总结示意图〕

#### Slide 92 · 提升树 vs 权重树对比

```
提升树（残差树）举例：           权重树举例：
商品一共 82 元                  （AdaBoost 样本权重示意）
  50 元                             ○ ○ × ←错样本权重大
  20 元                             × ○ ○
  10 元                             ○ × ○
  ───                        ──────────────────
累加预测                       加权投票/模型权重
```

---

#### Slide 16（补充）· XGBoost 电子游戏例子（补充版）

假设我们要预测一家人对电子游戏的喜好程度，考虑到年轻和年老相比，年轻更可能喜欢电子游戏，以及男性和女性相比，男性更喜欢电子游戏，故先根据年龄大小区分小孩和大人，然后再通过性别区分开是男是女，逐一给各人在电子游戏喜好程度上打分：

〔图：XGBoost 特征分裂树示意〕

#### Slide 17（补充）· XGBoost 两棵树累加（补充版）

利用不同的特征训练出 tree1 和 tree2，两棵树的结论累加起来，从而更接近最终的结论。类似之前 gbdt 的原理（基础值 + 残差），两棵树的结论累加起来更接近最终的结论。

```
tree1（年龄特征） + tree2（性别特征） = 最终预测
```

#### Slide 18（补充）· XGBoost Loss 的复杂性

但是问题是 XGBoost 的 loss 更为复杂：

$$\text{XGBoost loss} = \text{损失} + \text{正则项}$$

（不一定是平方损失）+（限制模型复杂度）

也希望将该 loss 降到最低。但是存在正则项，难以求负梯度，从而优化困难。

**对策：化繁为简**——利用泰勒展开公式来近似该 loss 表达式。

#### Slide 19（补充）· 泰勒展开（补充版）

〔同主 PPT Slide 74 内容〕

**泰勒展开**：将一个函数在某一点处展开成无限项的多项式表达式

**一阶泰勒展开**：$f(x + \Delta x) \approx f(x) + f'(x)\Delta x$

**二阶泰勒展开**：$f(x + \Delta x) \approx f(x) + f'(x)\Delta x + \frac{1}{2}f''(x)\Delta x^2$

#### Slide 20（补充）· XGBoost 增益计算要素

```
预测值：Wᵢ（即叶子得分 / 权重）
f_t：当前树的预测值

增益越高，损失函数降低越多（可有可无的项可以去掉）
```

#### Slide 21（补充）· 计算一阶导 gᵢ 和二阶导 hᵢ

〔图：特征 x1 / x2 样本分布〕

根据目标函数（loss）简化后，计算 $g_i$（一阶导）和 $h_i$（二阶导）：

**当损失函数是平方损失的时候：**

$$L = \frac{1}{2}(y_i - \hat{y}_i)^2$$

$$g_i = \frac{\partial L}{\partial \hat{y}} = -(y_i - \hat{y}_i) \quad (\text{系数} \times 2 \text{ 抵消})$$

$$h_i = \frac{\partial^2 L}{\partial \hat{y}^2} = 1$$

标签与随机猜测的差异即为一阶导。

#### Slide 22（补充）· 第一棵决策树分裂判断

因此，计算出残差与一阶导 $g_i$ 和二阶导 $h_i$

第一棵决策树进行修正（分裂开始！！）

要找一个最佳特征以及分裂点（比如"天赋"特征）。需要一个评判标准。

复杂推导，可证：可以利用**增益 Gain** 来寻找合适的分割线，同时，损失函数实现最小化。

```
GBDT 用的是平方和损失！！
弱学习器之间的通信：通过 gᵢ 和 hᵢ 传递信息
```

⚠ **注意**：此处 $x$ 系数 2 是因为一阶导计算中已包含该系数，与平方损失的 $\frac{1}{2}$ 系数抵消，需代回验证。

#### Slide 23（补充）· 略——复杂推导 1

〔内容：XGBoost 目标函数完整推导（略）〕

#### Slide 24（补充）· 略——复杂推导 2

〔内容：XGBoost 分裂增益推导（略）〕

#### Slide 25（补充）· 利用 Gain 寻找分裂点

〔图：父节点、左节点、右节点的样本统计〕

```
左节点样本统计（GR, HR）
右节点样本统计（GL, HL）
分裂前父节点样本统计

先考察天赋特征：
  Gain = ½[GL²/(HL+λ) + GR²/(HR+λ) - (GL+GR)²/(HL+HR+λ)] - γ
```

#### Slide 26（补充）· 叶子节点权重计算

〔图：节点 w1 / w2 / w3 计算示意〕

```
给出每个节点的回归值（叶子权重）：
  wⱼ* = -Gⱼ/(Hⱼ + λ)  （分子无平方）

学习率设定了更新的步伐

之前 GBDT 是残差均值；
XGBoost 叶子值 = -Gⱼ/(Hⱼ+λ)，由一阶/二阶导和正则共同决定，更稳健。
```

#### Slide 27（补充）· 第二棵决策树修正

当前的局面（第一棵树修正后）：

第二棵决策树进行修正——换一个特征，分裂开始！！

〔图：换特征后的样本分布〕

#### Slide 28（补充）· 换特征

〔图：用另一个特征进行第二棵树的分裂〕

#### Slide 29（补充）· 第二棵树叶子权重

〔图：节点 w1 / w2 / w3 权重计算〕

```
计算权重（第二棵树）：
  wⱼ* = -Gⱼ/(Hⱼ + λ)

进一步逼近真实值

更多的决策树，可以重复利用特征构建，也可添加新特征。
```

#### Slide 30（补充）· 总结图示（补充末页）

〔图：集成学习补充材料总结图〕

### 笔记

> 【知道】XGBoost 算法思想

XGBoost 是对 GBDT 的三点改进：

1. **泰勒二阶展开**求解损失（GBDT 只用一阶）
2. **加正则项**控制树复杂度（防过拟合）
3. **自创分裂指标**（从损失函数推导，分裂时考虑复杂度增益）

> 【理解】XGBoost 目标函数

**整体目标**（前 $K$ 棵树）：

$$\mathrm{Obj} = \sum_{i=1}^{n} L(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$

第一项是样本损失，第二项是 $K$ 棵树的复杂度正则。

**树复杂度**：

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2$$

- $T$：叶子结点数；$\gamma$ 是叶子数惩罚系数
- $w$：叶子输出值向量；$\lambda$ 是 L2 正则系数

**第 $t$ 轮目标**（前 $t-1$ 棵已固定）：

$$\mathrm{Obj}^{(t)} = \sum_{i=1}^{n} L(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t) + \mathrm{const}$$

**为何要展开**：损失加正则后形式复杂，难以直接对 $f_t$ 求负梯度优化。泰勒二阶近似把目标变成关于 $f_t$ 的二次函数，可直接闭式求解。

**泰勒二阶展开**（对 $\hat{y}_i^{(t-1)}$ 处展开）：

$$L(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) \approx L(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)$$

其中：

$$g_i = \partial_{\hat{y}^{(t-1)}} L, \quad h_i = \partial^2_{\hat{y}^{(t-1)}} L$$

**去常数项**后的目标：

$$\mathrm{Obj}^{(t)} \approx \sum_{i=1}^{n} \left[g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)\right] + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

**视角变换**（按样本 → 按叶子）：设 $I_j = \{i \mid q(x_i) = j\}$ 为落到叶子 $j$ 的样本集合，$f_t(x_i) = w_{q(x_i)}$：

$$\mathrm{Obj}^{(t)} = \sum_{j=1}^{T} \left[\left(\sum_{i \in I_j} g_i\right) w_j + \frac{1}{2}\left(\sum_{i \in I_j} h_i + \lambda\right) w_j^2\right] + \gamma T$$

记 $G_j = \sum_{i \in I_j} g_i$，$H_j = \sum_{i \in I_j} h_i$：

$$\mathrm{Obj}^{(t)} = \sum_{j=1}^{T} \left[G_j w_j + \frac{1}{2}(H_j + \lambda) w_j^2\right] + \gamma T$$

**对 $w_j$ 求导得最优叶子值**：

$$w_j^* = -\frac{G_j}{H_j + \lambda}$$

**对比 GBDT**：GBDT 叶子值 = 该叶子内残差均值；XGBoost 叶子值 = $-G_j/(H_j+\lambda)$，由一阶/二阶导和正则共同决定，更稳健。

**代回得目标最小值**（打分函数 / scoring function）：

$$\mathrm{Obj}^* = -\frac{1}{2} \sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma T$$

**分裂增益**（树构建时选最佳分裂点）：

$$\mathrm{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma$$

- $\mathrm{Gain} > 0$ 才分裂（公式中已减去 $\gamma$，等价于增益 > $\gamma$ 阈值）
- 停止条件：达到最大深度 / 叶子样本数过低 / 增益不足

> 【了解】XGBoost API

**安装**：sklearn 不内置，需 `pip install xgboost`。提供两套 API：原生风格（`xgb.train` + `DMatrix`）与 sklearn 风格（`XGBClassifier`），下面用后者。

```python
from xgboost import XGBClassifier

bst = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.3,
    objective="binary:logistic",  # multi:softmax / reg:squarederror
    reg_lambda=1,    # L2
    gamma=0,         # 分裂阈值
    # eta=0.1,            # learning_rate 的原生别名
    # eval_metric="merror",  # 多分类错误率，二分类常用 "logloss" / "auc"
)
bst.fit(X_train, y_train)
```

> 【实践】红酒品质预测

**数据**：11 个特征，3269 条数据，品质 1–6。

**数据特点**：6 个品质类别样本数差异显著（典型不均衡），训练时需要 `sample_weight` 校正。

```python
# 流程骨架
# 1. 读 wine.csv → X, y
# 2. train_test_split
# 3. XGBClassifier(objective="multi:softmax").fit(X_train, y_train)  # 品质有序，也可用 reg:squarederror 当回归
# 4. GridSearchCV 调 n_estimators + max_depth + learning_rate
```

```python
from sklearn.utils import class_weight
w = class_weight.compute_sample_weight('balanced', y_train)
bst.fit(X_train, y_train, sample_weight=w)  # 处理类别不均衡
```

**分层 CV**：不均衡数据用 `StratifiedKFold(n_splits=5, shuffle=True)` 作为 `GridSearchCV(cv=...)`，保证每折类别比例稳定。
