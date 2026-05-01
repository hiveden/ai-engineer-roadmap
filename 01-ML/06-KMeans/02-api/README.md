# 第 2 章 · 聚类 API 初步使用

> 维度：**代码**

## 底稿

> 02 · 聚类 API 的初步使用

**学习目标**：

1. 了解 KMeans 算法的 API
2. 动手实践 KMeans 算法

> 【了解】API 介绍

```python
sklearn.cluster.KMeans(n_clusters=8)
```

**关键参数**：

- `n_clusters`：聚类中心数量（即 K），整型，默认 `8`，决定生成的簇数 / 质心（centroids）个数

**核心方法**：

| 方法 | 作用 |
|---|---|
| `estimator.fit(x)` | 训练：迭代求质心 |
| `estimator.predict(x)` | 推理：把样本分配到最近的质心 |
| `estimator.fit_predict(x)` | 等价于 `fit(x)` + `predict(x)`，返回每个样本的簇标签 |

**约定**：聚类没有"正确答案"，`predict` 返回的标签 `0, 1, 2, ...` 只是簇编号，与任何外部类别无对应关系——同一份数据多次跑可能簇编号互换。

> 【实践】案例

随机生成不同二维数据集作为训练集，用 K-Means 聚类，尝试不同 K 观察效果。

**1. 创建数据集**：

```python
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y_true = make_blobs(n_samples=1000, n_features=2,
                       centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                       cluster_std=[0.4, 0.2, 0.2, 0.2],
                       random_state=22)
```

**2. K-Means 聚类 + CH 系数评估**：

```python
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score

y_pred = KMeans(n_clusters=4, random_state=22).fit_predict(X)
print('CH score:', calinski_harabasz_score(X, y_pred))
```

CH 系数（Calinski-Harabasz）越大说明簇内越紧、簇间越分——下一章评价指标详述。
