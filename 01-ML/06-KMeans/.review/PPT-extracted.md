# PPT 提取报告: 07聚类.pptx

总页数: 64

含代码: 是 / 含公式标记字符: 是


## Slide 1 · 聚类算法

---

## Slide 2 · (无标题)

**要点**：
- 聚类算法简介
- 聚类算法API的使用
- Kmeans实现流程
- 模型评估方法
- 案例顾客数据聚类分析法

---

## Slide 3 · (无标题)

**要点**：
- 知道什么是聚类
- 了解聚类算法的应用场景
- 知道聚类算法的分类

---

## Slide 4 · 聚类算法 – 概念

**要点**：
- 一种无监督学习算法  没有标签
- 根据样本特征之间的相似性，将样本划分到不同的类别中；不同的相似度计算方法，会得到不同的聚类结果，常用的相似度计算方法有欧式距离法。
- 聚类算法的目的是在没有先验知识的情况下，自动发现数据集中的内在结构和模式。
- 什么是聚类算法？
- 使用不同的聚类准则  ，产生的聚类结果不同
- 繁衍方式 (胎生、卵生)
- 生活环境(陆地、两栖、水中)
  - 食用 ？
- 呼吸方式 (肺、腮)

〔图：漏斗图〕
〔图：游戏机图片〕
〔图：形状〕

**备注**：
（无）

---

## Slide 5 · 聚类算法在现实生活中的应用

**要点**：
- 基于位置信息的商业推送
- 新闻聚类，筛选排序
- 用户画像，广告推荐
- Data Segmentation
- 搜索引擎的流量推荐
- 恶意流量识别
- 图像分割，降维，识别
- 离群点检测，信用卡异常消费
- 发掘相同功能的基因片段

**备注**：
（无）

---

## Slide 6 · 聚类算法分类

**要点**：
- 1.不同的聚类方法对颗粒度的控制程度不同 （即：样本被分为几个类别 ？）
- 体重
- 体重
- 身高
- 身高
- 2.根据实现方法分类
  - K-means：按照质心分类，用距离计算相似性，主要介绍K-means，通用、普遍。  如何决定K (分几个类？)
  - 层次聚类：对数据进行逐层划分，直到达到聚类的类别个数
  - DBSCAN聚类是一种基于密度的聚类算法 Density-Based Spatial Clustering of Applications with Noise
  - 谱聚类是一种基于图论的聚类算法

〔图：散点图（粗细聚类对比）〕
〔图：散点图（粗细聚类对比）〕

**备注**：
（无）

---

## Slide 7 · (无标题，小结)

**要点**：
- 1 聚类概念
  - 无监督学习算法，主要用于将相似的样本自动归到一个类别中；
  - 计算样本和样本之间的相似性，一般使用欧式距离
- 2 聚类分类
  - 颗粒度：粗聚类、细聚类。
  - 实现方法： K-means聚类、层次聚类、 DBSCAN聚类、谱聚类

---

## Slide 8 · (练习题)

**要点**：
- 1、下列关于聚类算法的描述错误的是？
  - A）聚类算法是一种无监督的机器学习算法
  - B）聚类算法通过计算样本之间的相似度来确定它是属于哪一个聚集类别
  - C）在聚类算法中样本之间的相似度只能通过欧式距离来衡量
  - D）不同的聚类准则产生的聚类效果也不同
- 答案解析：衡量样本间相似度的方法不止欧式距离一种，它只不过是常用的一种。
- 答案：C

---

## Slide 9 · (无标题，章节目录)

**要点**：
- 聚类算法简介
- 聚类算法API的使用
- Kmeans实现流程
- 模型评估方法
- 案例顾客数据聚类分析法

---

## Slide 10 · (无标题，学习目标)

**要点**：
- 了解Kmeans算法的API
- 动手实践Kmeans算法

---

## Slide 11 · 聚类算法API

**要点**：
- sklearn.cluster.KMeans(n_clusters=8)
- 参数: n_clusters:开始的聚类中心数量
  - 整型，缺省值=8，生成的聚类数，即产生的质心（centroids）数。
- 方法
  - estimator.fit(x)
  - estimator.predict(x)
  - estimator.fit_predict(x)
  - 计算聚类中心并预测每个样本属于哪个类别,相当于先调用fit(x),然后再调用predict(x)

**备注**：
（无）

---

## Slide 12 · (无标题，API 使用展示)

**要点**：
- 使用KMeans模型数据探索聚类
- 随机创建不同二维数据集作为训练集，并结合k-means算法将其聚类，尝试分别聚类不同数量的簇，并观察聚类效果：

〔图：聚类效果图〕
〔图：代码运行结果图〕

**备注**：
（无）

---

## Slide 13 · (无标题，API 使用步骤)

**要点**：
- 使用KMeans模型数据探索聚类
  - 1 导包 sklearn.cluster.KMeans
  - sklearn.datasets.make_blobs
  - 2 创建数据集
  - 3 实例化Kmeans模型并预测
  - 4 展示聚类效果
  - 5 评估聚类效果好坏

〔图：代码运行结果图〕

**备注**：
（无）

---

## Slide 14 · (无标题，API 使用代码)

**要点**：
- 使用KMeans模型数据探索聚类

**代码**：
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

〔图：代码运行结果图〕

**备注**：
（无）

---

## Slide 15 · (无标题，小结)

**要点**：
- 1 聚类算法API
  - sklearn.cluster.KMeans(n_clusters=8)
  - 参数：n_clusters:开 始的聚类中心数量
  - 方法：estimator.fit_predict(x)
    - 计算聚类中心并预测每个样本属于哪个类别,相当于先调用fit(x),
    - 然后再调用predict(x)
  - calinski_harabasz_score(x, y_pred) 用来评估聚类效果，数值越大越好

---

## Slide 16 · (练习题)

**要点**：
- 1、下列关于聚类算法API的描述正确的是？（多选）
  - A）它是通过sklearn.cluster.Kmeans来实现的
  - B）可以通过n_clusters参数指定样本最终被归为多少个聚类
  - C）右图中的样本一共被分为4个类
  - D）右图中的样本一共被分为2个类
- 答案解析：c错误，聚类成2个类别。
- 答案：ABD

〔图：聚类效果散点图〕

**备注**：
（无）

---

## Slide 17 · (无标题，章节目录)

**要点**：
- 聚类算法简介
- 聚类算法API的使用
- Kmeans实现流程
- 模型评估方法
- 案例顾客数据聚类分析法

---

## Slide 18 · (无标题，学习目标)

**要点**：
- 能掌握K-means聚类的实现步骤

---

## Slide 19 · KMeans算法实现流程

**要点**：
- 1 、事先确定常数K ，常数K意味着最终的聚类类别数
- 2、随机选择 K 个样本点作为初始聚类中心
- 3、计算每个样本到 K 个中心的距离，选择最近的聚类中心点作为标记类别
- 4、根据每个类别中的样本点，重新计算出新的聚类中心点（平均值），如果计算得出的新中心点与原中心点一样则停止聚类，否则重新进行第 3 步过程，直到聚类中心不再变化

**备注**：
（无）

---

## Slide 20 · (无标题，K-means 直觉图)

**要点**：
- 无监督学习举例
- 剧透： k-means聚类
- 特征空间
- 不断迭代
- 不需要标签

〔图：文本/邮件示意图〕
〔图：散点图（特征空间）〕

**备注**：
（无）

---

## Slide 21 · KMeans算法实现流程

**要点**：
- 预选中心
- 归类
- 重新计算中心

〔图：流程图（预选中心）〕
〔图：流程图（归类）〕
〔图：流程图（重新计算中心）〕

**备注**：
（无）

---

## Slide 22 · KMeans算法实现流程

〔图：KMeans流程动图/示意图〕

**备注**：
（无）

---

## Slide 23 · KMeans算法实现流程 – 举例

**要点**：
- 已知数据如下表所示，按照KMeans算法实现流程进行聚类：

〔图：数据表格〕

**备注**：
（无）

---

## Slide 24 · KMeans算法实现流程 – 举例

**要点**：
- 1、随机设置K个特征空间内的点作为初始的聚类中心（本案例中设置p1和p2）

〔图：散点图（初始聚类中心）〕

**备注**：
（无）

---

## Slide 25 · KMeans算法实现流程 – 举例

**要点**：
- 2、对于其他每个点计算到K个中心的距离，选择最近的一个聚类中心点作为标记类别

〔图：散点图（距离计算）〕
〔图：表格（距离计算结果）〕
〔图：散点图（归类结果）〕

**备注**：
（无）

---

## Slide 26 · KMeans算法实现流程 – 举例

**要点**：
- 3、接着对标记的聚类中心，重新计算每个聚类的新中心点（平均值）
- 更新

〔图：散点图（更新中心）〕

**备注**：
（无）

---

## Slide 27 · KMeans算法实现流程 – 举例

**要点**：
- 4、如果计算得出的新中心点与原中心点一样（质心不再移动），那么结束，否则重新进行第二步过程【经过判断，需要重复上述步骤，开始新一轮迭代】
- 更新

〔图：散点图（迭代）〕
〔图：表格（新一轮距离计算）〕
〔图：散点图（新归类结果）〕

**备注**：
（无）

---

## Slide 28 · KMeans算法实现流程 – 举例

**要点**：
- 5、当每次迭代结果不变时，认为算法收敛，聚类完成
- 请大家思考
- 数据是否需要进行标准化或归一化 ？？

〔图：散点图（最终聚类结果）〕

**备注**：
（无）

---

## Slide 29 · (无标题，聚类中心选择深入说明)

**要点**：
- 对于聚类中心的选择，更深入的说明
- 取最佳
- 如何衡量best output:
- 所有样本点到其所属簇中心的距离平方和。这个值越小，说明样本离簇中心越近，聚类效果理论上越好。

〔图：背景图案〕
〔图：散点图（多次初始化对比）〕
〔图：文字说明〕
〔图：图形用户界面文字〕
〔图：图形用户界面文字〕

**备注**：
点阵图   两个聚类中心初始化不同的位置，对聚类结果的影响很大。

---

## Slide 30 · (无标题，小结)

**要点**：
- 1 KMeans算法实现流程
  - 1 、事先确定常数K ，常数K意味着最终的聚类类别数
  - 2、随机选择 K 个样本点作为初始聚类中心
  - 3、计算每个样本到 K 个中心的距离，选择最近的聚类中心点作为标记类别
  - 4、根据每个类别中的样本点，重新计算出新的聚类中心点（平均值），如果计算得出的新中心点与原中心点一样则停止聚类，否则重新进行第 3步过程，直到聚类中心不再变化

---

## Slide 31 · (练习题)

**要点**：
- 1、下列是Kmeans算法的实现流程，请对它们进行排序：
  - A） 将该未知样本点归类为与D值最小时的中心点相同的类别；
  - B） 计算未知样本点分别到这K个中心点的距离D；
  - C） 重复上述过程，直至新的中心点与旧的中心点一致，则迭代停止，将最后这次的聚类作为最优聚类结果。
  - D） 随机初始化K个中心点；
  - E） 计算这K个分类簇的均值分别作为这K个簇新的中心点；
- 答案解析 D在开始  C在最后
- 答案：D→B→A→E→C

**备注**：
（无）

---

## Slide 32 · (无标题，章节目录)

**要点**：
- 聚类算法简介
- 聚类算法API的使用
- Kmeans实现流程
- 模型评估方法          如何决定最佳的超参 (聚类中心数)
- 案例顾客数据聚类分析法

---

## Slide 33 · (无标题，学习目标)

**要点**：
- 了解 SSE 聚类评估指标
- 了解 SC 聚类评估指标
- 了解 CH 聚类评估指标
- 了解肘方法的作用

---

## Slide 34 · 误差平方和SSE (Sum of Squared Errors)

**要点**：
- Ci 表示簇
- k 表示聚类中心的个数
- p 表示某个簇内的样本
- m 表示质心点
- SSE 越小，表示数据点越接近它们 各自的中心，聚类效果越好
- 此时  K 越多  聚类越多  从而每个数据点能找到更近的聚类中心。

〔图：公式图片（SSE 公式）〕
〔图：示意图〕

**备注**：
（无）

---

## Slide 35 · "肘"方法 (Elbow method) -  K值确定

**要点**：
- "肘" 方法通过 SSE 确定 n_clusters 的值
- 对于n个点的数据集，迭代计算 k from 1 to n，每次聚类完成后计算 SSE
- SSE 是会逐渐变小的。因为 最终，每个点都是它所在的簇中心本身。
- SSE 变化过程中会出现一个拐点，下降率突然变缓时即认为是最佳 n_clusters 值。
  - -----数据通常有更多的噪音，在增加分类无法带来更多回报时，我们停止增加类别  （早停）。
- 如果K过大，容易过拟合（每个样本独自成一类）。

〔图：肘部法曲线图〕

**备注**：
（无）

---

## Slide 36 · 聚类效果评估 – 代码效果展示SSE误差平方和

**代码**：
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score

def dm01_SSE误差平方和求模型参数():
    sse_list = []
    # 产生数据random_state=22固定好
    x, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                      cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=22)
    for clu_num in range(1, 100):
        my_kmeans = KMeans(n_clusters=clu_num, max_iter=100, random_state=0)
        my_kmeans.fit(x)
        sse_list.append(my_kmeans.inertia_ ) # 获取sse的值
    plt.figure(figsize=(18, 8), dpi=100)
    plt.xticks(range(0, 100, 3), labels=range(0, 100, 3))
    plt.grid()
    plt.title('sse')
    plt.plot(range(1, 100), sse_list, 'or-')
    plt.show()
    # 通过图像可观察到 n_clusters=4 sse开始下降趋缓, 最佳值4
```

**备注**：
（无）

---

## Slide 37 · 聚类效果评估 – 代码效果展示SSE误差平方和

**要点**：
- 通过图像可观察到 n_clusters=4 sse开始下降趋缓, 最佳值4

〔图：SSE 曲线图（肘部在 n_clusters=4 处）〕

**备注**：
（无）

---

## Slide 38 · (无标题，SC轮廓系数法)

**要点**：
- SC轮廓系数法（Silhouette Coefficient）
- 轮廓系数法考虑簇内的内聚程度(Cohesion)，簇外的分离程度(Separation)。其计算过程如下：
  - 对计算每一个样本 i 到同簇内其他样本的平均距离 ai，该值越小，说明簇内的相似程度越大
  - 计算每一个样本 i 到最近簇 j 内的所有样本的平均距离 bij，该值越大，说明该样本越不属于其他簇 j
  - 问题： 对任一个样本 如何判定最近的簇 ？
  - 答： 对其他团簇内所有样本也求平均距离。平均距离最小的就是最近团簇。

〔图：背景图案〕
〔图：散点图（簇内/簇间距离示意）〕
〔图：卡通人物〕

---

## Slide 39 · (无标题，SC公式)

**要点**：
- 模型整体评估：
- 让每个样本都求一次 轮廓系数Si
- 越大越好

〔图：背景图案〕
〔图：SC 公式图片〕
〔图：SC 整体评估公式图片〕

---

## Slide 40 · 聚类效果评估 – 代码效果展示 – SC系数

**代码**：
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

def dm02_轮廓系数SC():
    tmp_list = []
    # 产生数据random_state=22固定好
    x, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                      cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=22)
    for clu_num in range(2, 100):
        my_kmeans = KMeans(n_clusters=clu_num, max_iter=100, random_state=0)
        my_kmeans.fit(x)
        ret = my_kmeans.predict(x)
        tmp_list.append(silhouette_score(x, ret))  # sc
    plt.figure(figsize=(18, 8), dpi=100)
    plt.xticks(range(0, 100, 3), labels=range(0, 100, 3))
    plt.grid()
    plt.title('sc')
    plt.plot(range(2, 100), tmp_list, 'ob-')
    plt.show()
    # 通过图像可观察到 n_clusters=4 取到最大值
```

**备注**：
（无）

---

## Slide 41 · 聚类效果评估 – 代码效果展示 – SC系数

**要点**：
- 通过图像可观察到 n_clusters=4 取到最大值； 最佳值4

〔图：SC 曲线图（最大值在 n_clusters=4 处）〕

**备注**：
（无）

---

## Slide 42 · 聚类效果评估 – CH指数（Calinski-Harabasz Index）

**要点**：
- CH 指数考虑 每个簇内的内聚程度、每个簇之间离散程度。
- 宗旨：

〔图：CH 指数宗旨/公式图片〕

**备注**：
（无）

---

## Slide 43 · (无标题，CH 公式推导)

〔图：背景图案〕
〔图：CH 公式（簇内散度）〕
〔图：散点图〕
〔图：CH 公式（簇间散度）〕

**要点**：
- 由于总样本数n越多，这个数值越大，因此需要求平均，聚类结果之间才有可比性。
- 利用自由度进行求平均

〔图：徽标/公式〕

**备注**：
样本方差 、无偏估计

---

## Slide 44 · (无标题，自由度概念)

**要点**：
- 聚类：
- 自由度的概念：
  - 有 5 颗糖果要分给 3 个小朋友（A、B、C），
  - 但有一个规则：
  - 他们三个人分到的糖果总数必须是 5 颗。
  - 你任意给A 和 B 发糖，
  - 但是C的糖果数是确定的。
  - 因此自由度 = 3-1 =2
- 均值（质心）已确定
- 自由度 = n-1 = 3

〔图：CH 公式图片〕

---

## Slide 45 · (无标题，自由度应用)

**要点**：
- 在本案例中，从全局来看：
  - 数据集一种n个样本，而聚类中心有k个
  - 因此自由度 = n - k
- 由于总样本数n越多，这个数值越大，因此需要求平均，聚类结果之间才有可比性。
- 利用自由度进行求平均

〔图：散点图〕
〔图：公式图片〕
〔图：徽标/公式〕

---

## Slide 46 · 提问： 为什么用自由度做分母，而不是簇内样本数？

**要点**：
- 详细参见统计学基本概念： 样本方差：对总体方差的无偏估计。
- 大白话类比：射箭比赛              case  1                                         case 2 耍赖
  - 先射箭，后画靶心
  - 方差很大，评估下来箭术不佳
  - 衡量的是数据总体的弥散或波动程度
  - 箭术同样不佳，但靶心是数据自己决定的，因此方差被低估
  - 因此，需要修正方差，用自由度作为分母（= 样本方差）

〔图：靶心图（case 1）〕
〔图：靶心图（case 2 耍赖）〕
〔图：徽标〕
〔图：箭术图〕
〔图：箭术图〕

---

## Slide 47 · (无标题，完整证明)

**要点**：
- 完整证明（样本方差）
- 略

〔图：公式推导图片〕
〔图：公式推导图片〕

---

## Slide 48 · (无标题，簇间散度推导)

**要点**：
- 由于团簇越多，这个数值越大，因此需要 求平均
- 利用自由度求平均
- （K 个 团簇，  K-1 个自由度）
- 等价于对各个聚类中心的位置求平均。

〔图：散点图〕
〔图：公式图片（簇间散度）〕
〔图：公式图片〕
〔图：图形用户界面〕

**备注**：
（无）

---

## Slide 49 · CH指数（Calinski-Harabasz Index）

**要点**：
- 等价于对各个聚类中心的位置求平均。

〔图：CH 指数完整公式表格图片〕
〔图：公式图片〕

---

## Slide 50 · 聚类效果评估 – 代码效果展示 – CH指数

**代码**：
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score

def dm03_ch系数():
    tmp_list = []
    # 产生数据random_state=22固定好
    x, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                      cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=22)
    for clu_num in range(2, 100):
        my_kmeans = KMeans(n_clusters=clu_num, max_iter=100, random_state=0)
        my_kmeans.fit(x)
        ret = my_kmeans.predict(x)
        tmp_list.append(calinski_harabasz_score(x, ret))  # sc
    plt.figure(figsize=(18, 8), dpi=100)
    plt.xticks(range(0, 100, 3), labels=range(0, 100, 3))
    plt.grid()
    plt.title('ch')
    plt.plot(range(2, 100), tmp_list, 'og-')
    plt.show()
```

**备注**：
（无）

---

## Slide 51 · 聚类效果评估 – 代码效果展示 – CH指数

**要点**：
- 通过图像可观察到 n_clusters=4 取到最大值； 最佳值4

〔图：CH 曲线图（最大值在 n_clusters=4 处）〕

**备注**：
（无）

---

## Slide 52 · (无标题，层次聚类)

**要点**：
- 层次聚类  --  凝聚  与  分裂  2种模式
  - 分裂： 反其道而行之
  - 首先，所有元素归属同一个集群，然后分裂集群，直到所有元素都独立成群
- 聚类停止条件：
  - 提前设定K
  - 用轮廓系数等指标

〔图：背景图案〕
〔图：徽标〕
〔图：层次聚类图（凝聚/分裂）〕
〔图：层次聚类图示〕

---

## Slide 53 · (无标题，小结)

**要点**：
- 1 误差平方和SSE
  - 误差平方和的值越小越好
  - 主要考量：簇内聚程度
- 2 肘部法
  - 下降率突然变缓时即认为是最佳的k值
- 3 SC系数
  - 取值为[-1, 1]，其值越大越好
  - 主要考量：簇内聚程度、簇间分离程度
- 4 CH系数
  - 分数s高则聚类效果越好
  - CH达到的目的：用尽量少的类别聚类尽量多的样本，同时获得较好的聚类效果
  - 主要考量：簇内聚程度、簇间分离程度、质心个数

**备注**：
（无）

---

## Slide 54 · (练习题)

**要点**：
- 1、下列可用于评估聚类算法的方法或指标的是：（多选）
  - A） SSE：误差平方和
  - B） 肘部法
  - C） Silhouette Coefficient： SC系数
  - D） Calinski-Harabasz Index： CH指数
- 答案：ACD

〔图：公式图片〕

**备注**：
（无）

---

## Slide 55 · (无标题，章节目录)

**要点**：
- 聚类算法简介
- 聚类算法API的使用
- Kmeans实现流程
- 模型评估方法
- 案例顾客数据聚类分析法

---

## Slide 56 · (无标题，学习目标)

**要点**：
- 能使用聚类算法完成客户案例分析
- 知道怎么求最佳K值

---

## Slide 57 · 案例：顾客数据聚类分析

**要点**：
- 已知：客户性别、年龄、年收入、消费指数
- 需求：对客户进行分析，找到业务突破口，寻找黄金客户       （没有使用标准化，因为量纲差别不大）

〔图：数据集预览图〕

**备注**：
（无）

---

## Slide 58 · 案例：顾客数据聚类分析

**要点**：
- 客户分群效果展示：
- 从图中可以看出，聚成5类，右上角属于挣的多，消费的也多黄金客户群

〔图：客户分群散点图〕

**备注**：
（无）

---

## Slide 59 · 案例：顾客数据聚类分析

**代码**：
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

**备注**：
（无）

---

## Slide 60 · 案例：顾客数据聚类分析

**要点**：
- 效果分析：
- 通过肘方法、sc系数都可以看出，聚成5类效果最好

〔图：肘方法曲线图〕
〔图：SC曲线图〕

**备注**：
（无）

---

## Slide 61 · 案例：顾客数据聚类分析

**代码**：
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

**备注**：
（无）

---

## Slide 62 · (空白页)

---

## Slide 63 · (无标题，层次聚类宇宙类比)

**要点**：
- hiratical clustering in the Universe
- Cosmic Web

〔图：背景图案〕
〔图：日程表/树状图〕
〔图：层次聚类示意图〕
〔图：宇宙结构示意图〕
〔图：文字说明〕

---

## Slide 64 · (封面页/结尾页)

〔图：结尾图片〕

---
