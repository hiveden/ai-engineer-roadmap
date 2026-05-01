# 第 5 章 · 综合案例

> 维度：**代码**

## 底稿

> 05 · 综合案例

**学习目标**：

1. 综合运用 K-Means + 评价指标完成聚类任务

> 【了解】案例介绍

**已知**：客户性别、年龄、年收入、消费指数。

**需求**：对客户分群，找业务突破口、定位"黄金客户"。

**数据**：200 条顾客数据，4 个特征（Mall Customer Segmentation Dataset 风格）。

业务视角：聚类结果交给市场团队解读——比如某簇"高收入低消费"= 潜力未开发用户，针对性营销。

> 【实践】案例实现

**步骤**：

```
1. 加载数据
2. 特征工程（标准化，避免年收入 vs 年龄量纲不一致主导距离）
3. 肘部法 + SC / CH 选 K
4. K-Means 聚类
5. 可视化 + 业务解读
```

**代码骨架**：

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# 1. 加载数据
df = pd.read_csv('Mall_Customers.csv')
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# 2. 标准化
X_scaled = StandardScaler().fit_transform(X)

# 3. 肘部法选 K
sse = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=22, n_init=10).fit(X_scaled)
    sse.append(km.inertia_)
plt.plot(range(2, 11), sse, marker='o')
plt.xlabel('K'); plt.ylabel('SSE'); plt.show()

# 4. 选定 K 后聚类
km = KMeans(n_clusters=5, random_state=22, n_init=10).fit(X_scaled)
df['cluster'] = km.labels_

# 5. 评估
print('SC:', silhouette_score(X_scaled, km.labels_))
print('CH:', calinski_harabasz_score(X_scaled, km.labels_))

# 6. 业务解读：按簇看均值画像
print(df.groupby('cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean())
```

**坑点提醒**：

- 不标准化 → `Annual Income` 数值大主导欧氏距离，`Spending Score` 几乎不起作用
- `n_init=10`：随机初始化 10 次取 SSE 最优，缓解局部最优
- 性别是类别变量需先编码（如 `pd.get_dummies`）才能进 K-Means
