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

> 【实践】案例

随机生成不同二维数据集作为训练集，用 K-Means 聚类，尝试不同 K 观察效果。

**5 步流程**（Slide 13）：
1 导包 sklearn.cluster.KMeans / sklearn.datasets.make_blobs
2 创建数据集
3 实例化 Kmeans 模型并预测
4 展示聚类效果
5 评估聚类效果好坏

```python
# 1.导入工具包
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score  # calinski_harabaz_score 废弃

# 2 创建数据集 1000个样本,每个样本2个特征 4个质心蔟数据标准差[0.4, 0.2, 0.2, 0.2]
x, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]],
                cluster_std = [0.4, 0.2, 0.2, 0.2], random_state=22)

plt.figure()
plt.scatter(x[:, 0], x[:, 1], marker='o')
plt.show()

# 3 使用k-means进行聚类, 并使用CH方法评估
y_pred = KMeans(n_clusters=3, random_state=22).fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.show()

# 4 模型评估
print(calinski_harabasz_score(x, y_pred))
```

> 【小结】

1 聚类算法 API
- sklearn.cluster.KMeans(n_clusters=8)
- 参数：n_clusters：开始的聚类中心数量
- 方法：estimator.fit_predict(x) — 计算聚类中心并预测每个样本属于哪个类别，相当于先调用 fit(x)，然后再调用 predict(x)
- calinski_harabasz_score(x, y_pred) 用来评估聚类效果，数值越大越好
