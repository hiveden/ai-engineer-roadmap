# 第 5 章 · 综合案例

> 维度：**代码**

## 底稿

> 05 · 综合案例

**学习目标**：

1. 综合运用 K-Means + 评价指标完成聚类任务

> 【了解】案例介绍

**已知**：客户性别、年龄、年收入、消费指数。

**需求**：对客户进行分析，找到业务突破口，寻找黄金客户。（没有使用标准化，因为量纲差别不大）

**数据**：200 条顾客数据，4 个特征。

> 【实践】案例实现

**步骤**：

```
1. 加载数据
2. 特征工程（标准化，避免年收入 vs 年龄量纲不一致主导距离）
3. 肘部法 + SC / CH 选 K
4. K-Means 聚类
5. 可视化 + 业务解读
```

**代码**（PPT Slide 59）：

```python
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# 聚类分析用户分群
def dm01_聚类分析用户群():
    dataset = pd.read_csv('data/customers.csv')
    dataset.info()
    print('dataset-->\n', dataset)
    X = dataset.iloc[:, [3, 4]]
    print('X-->\n', X)
    mysse = []
    mysscore = []
    # 评估聚类个数
    for i in range(2, 11):
        mykeans = KMeans(n_clusters=i)
        mykeans.fit(X)
        mysse.append(mykeans.inertia_)      # inertia 簇内误差平方和
        ret = mykeans.predict(X)
        mysscore.append(silhouette_score(X, ret))    # sc系数 聚类需要1个以上的类别

    plt.plot(range(2, 11), mysse)
    plt.title('the elbow method')
    plt.xlabel('number of clusters')
    plt.ylabel('mysse')
    plt.grid()
    plt.show()
    plt.title('sh')
    plt.plot(range(2, 11), mysscore)
    plt.grid(True)
    plt.show()
    pass
```

**效果分析**：通过肘方法、sc 系数都可以看出，聚成 5 类效果最好

```python
def dm02_聚类分析用户群():
    dataset = pd.read_csv('data/customers.csv')
    X = dataset.iloc[:, [3, 4]]
    mykeans = KMeans(n_clusters=5)
    mykeans.fit(X)
    y_kmeans = mykeans.predict(X)
    # 把类别是0的, 第0类数据,第1列数据, 作为x/y, 传给plt.scatter函数
    plt.scatter(X.values[y_kmeans == 0, 0], X.values[y_kmeans == 0, 1], s=100, c='red', label='Standard')
    # 把类别是1的, 第0类数据,第1列数据, 作为x/y, 传给plt.scatter函数
    plt.scatter(X.values[y_kmeans == 1, 0], X.values[y_kmeans == 1, 1], s=100, c='blue', label='Traditional')
    # 把类别是2的, 第0类数据,第1列数据, 作为x/y, 传给plt.scatter函数
    plt.scatter(X.values[y_kmeans == 2, 0], X.values[y_kmeans == 2, 1], s=100, c='green', label='Normal')
    plt.scatter(X.values[y_kmeans == 3, 0], X.values[y_kmeans == 3, 1], s=100, c='cyan', label='Youth')
    plt.scatter(X.values[y_kmeans == 4, 0], X.values[y_kmeans == 4, 1], s=100, c='magenta', label='TA')
    plt.scatter(mykeans.cluster_centers_[:, 0], mykeans.cluster_centers_[:, 1], s=300, c='black', label='Centroids')

    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()
```

**业务结论**：聚成 5 类，右上角属于挣的多，消费的也多黄金客户群
