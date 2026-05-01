# 分类 API：KNeighborsClassifier

## 底稿

> 【实操】分类 API

KNN 分类 API：

```python
sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
```

```python
"""
函数功能：创建一个 KNN 分类器，用简单数据训练，并对新样本进行预测。

KNN（K-近邻算法）
    概述：
        专业：找离测试集最近的哪 K 个样本，然后投票。哪个标签值多，
        就用它作为测试集的最终结果。
    KNN 算法实现思路：
        思路 1：分类思路    投票，选票数最多的
        思路 2：回归思路    求平均值

KNN（K-近邻算法）：分类（多数表决）
    1. 计算未知样本到每一个训练样本的距离
    2. 将训练样本根据距离大小升序排列
    3. 取出距离最近的 K 个训练样本
    4. 进行多数表决，统计 K 个样本中哪个类别的样本个数最多
    5. 将未知的样本归属到出现次数最多的类别

关键参数：n_neighbors=1 表示使用 1 个最近邻进行预测。
"""

# 导包
from sklearn.neighbors import KNeighborsClassifier

# 定义模型   参数 1 为 KNN 的 K 值（最近邻的 K 个样本作为调查对象）
model = KNeighborsClassifier(n_neighbors=1)

# 定义训练集
X = [[0], [1], [2], [3]]   # 特征必须是二维结构 4 行 1 列（4 个样本，1 个特征）
# y = [0, 1, 0, 1]         # 定义标签，4 个样本，有 4 个标签。必须是一维结构
y = ['dog', 'cat', 'duck', 'xx']

# 喂入数据，训练模型
# 将样本数据信息存储起来，用于在推理时刻计算距离
model.fit(X, y)

# 模型推理   参数为待预测的样本（必须也是二维结构，此刻是 1 行 1 列，代表 1 个样本，一个特征）
# 如果平票，哪个样本的索引较小，就用哪个样本的标签
result = model.predict([[2.5]])
print(result)
```

n_neighbors : int，可选（默认 = 5），k_neighbors 查询默认使用的邻居数。

---

## 关于 sklearn

**sklearn**（读 "S-K-learn"，全名 `scikit-learn`）是 Python 的 ML 标准库 —— 封装了主流算法（KNN / 决策树 / SVM / 逻辑回归 …），所有算法都遵循 **"构造 → fit → predict" 三件套**（下面就讲）。

> 安装 `pip install scikit-learn`；使用 `import sklearn`（包名 ≠ 导入名，常见坑）。

## 不在本章范围

本章只讲 **KNN sklearn API 的用法**，**不讲**：

- Python / pip / 虚拟环境配置
- IDE 选择（VS Code / PyCharm / Jupyter …）
- numpy / pandas 等配套基础库

假设你能装 Python 包并跑通 `import sklearn`。如果这些不熟，先去专题资料补。

## 直觉：sklearn 三件套

KNN 用代码跑只要 **3 步** —— sklearn 全家桶通用：

1. **构造**：`KNeighborsClassifier(n_neighbors=5)` —— 选 K
2. **训练**：`model.fit(X, y)` —— 喂数据
3. **预测**：`model.predict(X_new)` —— 出结果

[第 1 章五步法](../01-intro/05-工作流程.md) 的"算距离 / 排序 / 取 K / 投票 / 输出"全部被 sklearn 封装在 `fit` + `predict` 里。

## 最简分类 demo：豆瓣预测

延续 [01-intro 数据集](../01-intro/02-接近程度.md)：9 部已知电影 + 流浪地球 3，预测你会不会喜欢。

```python
from sklearn.neighbors import KNeighborsClassifier

# 9 部已知电影：特征 = (评分, 主演吸引度)
X = [[8.3, 9.0],   # 流浪地球 2
     [7.8, 8.5],   # 阿凡达
     [8.0, 9.5],   # 泰坦尼克
     [6.4, 7.0],   # 哥斯拉
     [5.5, 4.0],   # 喜羊羊
     [8.5, 8.0],   # 沙丘 2
     [8.4, 9.2],   # 复仇者联盟 4
     [7.2, 6.5],   # 寒战
     [7.4, 8.0]]   # 长津湖

y = ['喜欢', '喜欢', '喜欢',
     '不喜欢', '不喜欢',
     '喜欢', '喜欢',
     '不喜欢', '不喜欢']

# 1. 构造分类器，K=5
model = KNeighborsClassifier(n_neighbors=5)

# 2. 训练
model.fit(X, y)

# 3. 预测流浪地球 3 (8.5, 9.5)
prediction = model.predict([[8.5, 9.5]])
print(prediction)
# 输出: ['喜欢']
```

→ 第 1 章 [02 练习](../01-intro/02-接近程度.md) 手算的同一个例子，sklearn 用 3 行搞定。

## 三件套 = sklearn 全家桶通用模板

不只 KNN，sklearn 里所有监督学习算法都遵循这套：

```python
model = SomeAlgorithm(参数)         # 1. 构造
model.fit(X, y)                     # 2. 训练
predictions = model.predict(X_new)  # 3. 预测
```

后续要学的所有算法（决策树 / 随机森林 / SVM / 逻辑回归 …）**API 形态完全一样**。学会 KNN 三件套 = 学会 sklearn 入门。

## 术语版

| 故事里的元素 | 术语名 | 主场 |
|---|---|---|
| 训练数据（二维列表，每行一个样本）| **X**（大写，**特征矩阵** feature matrix）| — |
| 训练标签 | **y**（小写，**标签** labels）| — |
| 把数据喂给模型学习 | **fit / 训练** training | — |
| 用模型预测新样本 | **predict / 预测** prediction | — |

**关键启示**：KNN 分类在 sklearn 里就是 **3 行代码**（构造 + fit + predict）。

→ 回归版本（`KNeighborsRegressor`）只差最后一步，见 [`02-回归API`](./02-回归API.md)。
