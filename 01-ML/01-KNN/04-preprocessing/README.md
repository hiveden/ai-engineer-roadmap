# 第 4 章 · 特征预处理

> **按知识点拆分的讲解版**（本文件 = 章 PPT 完整底稿，作复习记忆页）：
>
> 1. [`01-为什么预处理.md`](./01-为什么预处理.md) — ★ 掌握：动机（KNN 量纲问题）
> 2. [`02-归一化.md`](./02-归一化.md) — ★ 掌握：MinMaxScaler + 异常值弊端
> 3. [`03-标准化.md`](./03-标准化.md) — ★ 掌握：StandardScaler + 应用场景
> 4. [`04-鸢尾花案例.md`](./04-鸢尾花案例.md) — ★ 掌握：KNN + 标准化完整 6 步实战

## 底稿

> 04 · 特征预处理

**学习目标**：

1. 知道为什么进行归一化、标准化
2. 能应用归一化 API 处理数据
3. 能应用标准化 API 处理数据
4. 使用 KNN 算法进行鸢尾花分类

> 【知道】为什么进行归一化、标准化

特征的单位或者大小相差较大，或者某特征的方差相比其他的特征要大出几个数量级，容易影响（支配）目标结果，使得一些模型（算法）无法学习到其它的特征。

比如 KNN 欧式距离的计算。

> 【掌握】归一化

通过对原始数据进行变换把数据映射到【 0~1 】之间。

数据归一化的 API 实现：

```python
sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)…)
```

```python
'''
演示  特征预处理之归一化
    特征预处理解释：
        背景：
            在实际开发中，如果多个特征列因为量纲（单位）的问题，导致数值的差距过大。
            则会导致模型预测值偏差。
            为了保证每个列对最终的预测结果的权重比都是相近的。
            所以我们需要对特征预处理操作。
        特征预处理实现方式：
            1：归一化（现在）
            2：标准化
        归一化介绍：
            概述：它是特征预处理一种方案。
                对原始数据进行处理，获取 1 个，默认 [mi, mx] 【0, 1】区间值
            公式：(min max 列的    mx mi 区间的)
                x' = (x - min) / (max - min)
                x'' = x' * (mx - mi) + mi
            公式解释：
                x   → 某一个特征列的值：原值
                min → 该特征列的最小值
                max → 该特征列的最大值
                mi  → 区间的最小值默认 0
                mx  → 区间的最大值默认 1
            弊端：
                强依赖于该列的特征 最大值和最小值，如果差值比较大的话，计算效果不明显
                归一化适用于小数据集 的特征预处理。
'''

# 导包
from sklearn.preprocessing import MinMaxScaler

# 准备数据
data = [[90, 2, 10, 40],
        [60, 4, 15, 45],
        [75, 3, 13, 46]]

# 初始化 归一化器
transform = MinMaxScaler()
# 开始转换
# 方法 fit + transform    fit: 计算每一列的最小值和最大值    transform: 对数据进行归一化
data = transform.fit_transform(data)

print(data)
```

- `feature_range`：缩放区间
- 调用 `fit_transform(X)` 将特征进行归一化缩放

归一化受到最大值与最小值的影响，这种方法容易受到异常数据的影响，鲁棒性较差，适合传统精确小数据场景。

> 【掌握】标准化

通过对原始数据进行标准化，转换为【 均值=0，标准差=1 】的标准正态分布的数据。

- mean → 【 均值 】
- σ → 为特征的【 标准差 】

数据标准化的 API 实现：

```python
sklearn.preprocessing.StandardScaler()
```

调用 `fit_transform(X)` 将特征进行归一化缩放。

```python
'''
标准化介绍
    概述：特征预处理一种方案
    公式：x' = (x - 该列的平均值) / 该列的标准差
    公式解释：
        x    → 某特征列的某个具体的值 即原值
        mean → 该列的平均值
    应用场景：比较合适大数据集的应用场景。当数据量比较大的时候
        受最大值和最小值的影响会微乎其微。
    总结：
        无论是归一化，还是标准化，目的都是避免因为特征列的量纲问题，导致权重不同
        从而影响预测结果

    与 MinMaxScaler 的关键区别：
        - MinMaxScaler：缩放到固定范围（如 [0, 1]），对异常值敏感
        - StandardScaler：基于统计分布（均值/标准差），对异常值相对鲁棒
'''

from sklearn.preprocessing import StandardScaler

# 准备数据
data = [[90, 2, 10, 40],
        [60, 4, 15, 45],
        [75, 3, 13, 46]]

# 初始化 归一化器
transform = StandardScaler()
# 开始转换
# 方法 fit + transform    fit: 计算每一列的最小值和最大值    transform: 对数据进行归一化
data = transform.fit_transform(data)

print(data)

print('均值', transform.mean_)
print('方差', transform.var_)
```

对于标准化来说，如果出现异常点，由于具有一定数据量，少量的异常点对于平均值的影响并不大。

#### 高斯分布

正态分布是一种概率分布，大自然很多数据或者特征符合正态分布，也叫高斯分布。

正态分布记作 N(μ, σ)：μ 决定了其位置，其标准差 σ 决定了分布的幅度。

当 μ=0、σ=1 时的正态分布是标准正态分布。

方差。

3σ 法则的实例。

> 【实操】利用 KNN 算法进行鸢尾花分类

鸢尾花 Iris Dataset 数据集是机器学习领域经典数据集，鸢尾花数据集包含了 150 条鸢尾花信息，每 50 条取自三个鸢尾花中之一：Versicolour、Setosa 和 Virginica。

每个花的特征用如下属性描述：花萼长度 / 花萼宽度 / 花瓣长度 / 花瓣宽度（共 4 个特征）。

代码实现：

```python
'''
鸢尾花数据集的 KNN 模型训练 以及 评估
步骤:
1 加载数据
2 数据预处理（清洗）
3 特征工程（提取，归一化标准化）
4 模型训练
5 模型评估
6 模型预测
'''

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score    # 模型评估工具

# todo 步骤:
# todo 1 加载数据
mydataset = load_iris()

# todo 2 数据预处理（清洗）
# 数据集分割
# 参数 1：数据集中的特征列（4 列）        参数 2：标签（150 列，1 行）
# 参 3：测试集所占比例 20%（随机抽）       参 4：随机数种子
# 150 样本：训练集为 120 个 测试集为 30 个
x_train, x_test, y_train, y_test = train_test_split(
    mydataset.data, mydataset.target,
    test_size=0.3, random_state=22, stratify=mydataset.target
)

# todo 3 特征工程（提取，归一化标准化）
# 实例化标准化
transfer = StandardScaler()
# 先对训练集计算均值方差（并且保存这俩值），再进行标准化
x_train = transfer.fit_transform(x_train)

# 对于测试集，使用训练集的均值方差进行标准化
x_test = transfer.transform(x_test)

# todo 4 模型训练
# 实例化 KNN 分类器模型
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

# todo 5 模型评估
y_predict = model.predict(x_test)
print('预测结果：', y_predict)
print('标签：', y_test)

# 给模型评估打分   输入测试集的标签 和 测试集的预测结果
myscore1 = accuracy_score(y_test, y_predict)
myscore2 = model.score(x_test, y_test)   # 模型评估方法 2，输入 测试集特征和标签

print('准确率 1：', myscore1)
print('准确率 2：', myscore2)

# todo 6 模型预测
x_ceshi = [[3, 5, 4, 2]]   # 创建预测数据
x_ceshi = transfer.transform(x_ceshi)   # 对测试数据进行标准化
y_predict = model.predict(x_ceshi)
print('预测结果：', y_predict)

y_predict_probability = model.predict_proba(x_ceshi)
print('预测结果的概率分布：', y_predict_probability)        # 90%      5%       5%
```

### 总结

**数据归一化**：如果出现异常点，影响了最大值和最小值，那么结果显然会发生改变。应用场景：最大值与最小值非常容易受异常点影响，鲁棒性较差，只适合传统精确小数据场景。

**数据标准化**：如果出现异常点，由于具有一定数据量，少量的异常点对于平均值的影响并不大。应用场景：适合现代嘈杂大数据场景。（以后就是用你了）

## 不在本章范围

- **正态分布 / 高斯分布的数学公式 / 3σ 法则** —— 统计学主场，KNN 站只用工具不展开
- **RobustScaler**（中位数 + IQR）—— 超出 PPT 范围
- **Pipeline 防 data leakage** —— 数据预处理工程化，超出本章
- **K 值调优**（如何选最优 K）—— 第 5 章 [`05-hyperparameter/`](../05-hyperparameter/) 主场
