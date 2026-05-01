# 第 4 章 · 评价指标

> 维度：**概念 + 数学**

## 底稿

> 04 · 聚类评价指标

**学习目标**：

1. 了解 SSE 聚类评估指标
2. 了解 SC 聚类评估指标
3. 了解 CH 聚类评估指标
4. 了解肘方法的作用

> 【了解】SSE — 误差平方和

**SSE**（Sum of Squared Errors）：所有样本到其所属簇质心的距离平方和。

$$\mathrm{SSE} = \sum_{i=1}^{K} \sum_{p \in C_i} \lVert p - m_i \rVert^2$$

**符号**：$K$ = 簇数；$C_i$ = 第 $i$ 个簇；$p$ = 簇内样本；$m_i$ = 第 $i$ 簇质心。

**解读**：SSE 越小，样本越靠近所属质心，簇内越紧凑——但单看 SSE 会被 K 主导（K 越大 SSE 越小，K=N 时 SSE=0），所以不能直接比较不同 K 的 SSE 大小，要配合**肘部法**找拐点。

> 【了解】SC 系数（轮廓系数）

**SC**（Silhouette Coefficient）：结合**凝聚度**（Cohesion）和**分离度**（Separation）评估聚类效果。

**单样本 $i$ 的轮廓系数**：

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$

**计算步骤**：

1. $a(i)$：样本 $i$ 到**同簇内**其他样本的平均距离 → 越小说明簇内越紧
2. $b(i)$：样本 $i$ 到**最近其他簇**所有样本的平均距离 → 越大说明该样本越不像别的簇
3. 整体 SC = 所有样本 $s(i)$ 的均值

**取值范围**：$[-1, 1]$，越大越好。$\approx 1$ 簇分得清晰；$\approx 0$ 簇边界模糊；$< 0$ 样本被分到了错误的簇。

> 【了解】肘部法

**肘部法**（Elbow Method）用于**确定 K 值**。

**做法**：

1. 对 K 从 1 到 N 依次跑 K-Means，记录每次的 SSE
2. SSE 随 K 单调下降（K=N 时 SSE=0）
3. 画 SSE-K 曲线，找**拐点**（下降速率突然变缓的"肘部"）→ 该点对应的 K 即最佳簇数

**直觉**：拐点之前每加一簇 SSE 大幅下降（信息收益高）；拐点之后 SSE 下降变缓（边际收益小，开始过拟合噪声）。

工程经验：肘部不一定明显，配合 SC / CH 交叉验证。

> 【了解】CH 系数

**CH**（Calinski-Harabasz）：结合凝聚度、分离度、**质心个数**——倾向用最少的簇取得好聚类效果。

$$\mathrm{CH}(K) = \frac{\mathrm{SSB} / (K - 1)}{\mathrm{SSW} / (m - K)}$$

**SSW**（Within-cluster Sum of Squares，簇**内**离散度）：

$$\mathrm{SSW} = \sum_{i=1}^{m} \lVert x_i - C_{p_i} \rVert^2$$

每个样本 $x_i$ 到所属质心 $C_{p_i}$ 距离平方累加；越小簇内越紧。

**SSB**（Between-cluster Sum of Squares，簇**间**离散度）：

$$\mathrm{SSB} = \sum_{j=1}^{K} n_j \lVert C_j - X \rVert^2$$

$C_j$ = 第 $j$ 簇质心，$X$ = 全体质心的中心，$n_j$ = 簇 $j$ 样本数；越大簇间越散。

**符号**：$m$ = 样本数，$K$ = 簇数。**CH 越大越好**。分母里 $(K-1)/(m-K)$ 起到惩罚 K 过大的作用。

> 【实践】聚类评估的使用

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

km = KMeans(n_clusters=4, random_state=22).fit(X)
labels = km.labels_

print('SSE:', km.inertia_)                          # 簇内平方和
print('SC :', silhouette_score(X, labels))          # 轮廓系数
print('CH :', calinski_harabasz_score(X, labels))   # CH 系数
```

`km.inertia_` = SSE；轮廓与 CH 系数 sklearn 直接给函数。
