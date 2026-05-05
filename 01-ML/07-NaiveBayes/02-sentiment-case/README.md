# 第 2 章 · 情感分析案例

> 维度：**代码**

## 底稿

> 02 · 朴素贝叶斯实战

**学习目标**：

1. 知道 `sklearn.naive_bayes` 提供的朴素贝叶斯 API
2. 能用朴素贝叶斯做商品评论情感分类（好评 / 差评）

> 【实算】一个例子（手算朴素贝叶斯）

我们有一个小型商品评论数据集（已人工标注）

| 评论内容（分词后） | 标签 |
| --- | --- |
| 好 书 | 好评 |
| 书 好 | 好评 |
| 垃圾 书 | 差评 |
| 太 垃圾 | 差评 |

共 4 条评论：2 条好评，2 条差评。

我们要预测新评论："书 垃圾" 是好评还是差评？

目标：

$$P(\text{好评} \mid \text{书}, \text{垃圾})$$

$$P(\text{差评} \mid \text{书}, \text{垃圾})$$

$$P(C \mid W) = \frac{P(W \mid C)\, P(C)}{P(W)}$$

$$P(\text{好评} \mid \text{书}, \text{垃圾}) = \frac{P(\text{书、垃圾} \mid \text{好评})\, P(\text{好评})}{P(\text{书、垃圾})}$$

$$P(\text{差评} \mid \text{书}, \text{垃圾}) = \frac{P(\text{书、垃圾} \mid \text{差评})\, P(\text{差评})}{P(\text{书、垃圾})}$$

第一步 计算先验概率（Prior）：

$$P(\text{好评}) = \tfrac{2}{4} = 0.5,\qquad P(\text{差评}) = \tfrac{2}{4} = 0.5$$

第二步 构建词频统计

我们统计每个类别中**每个词的出现次数**

**所有词的集合（词汇表）：** "好"、"书"、"垃圾"、"太" → 共 N = 4 个词

| 词 | 好评中出现次数 | 差评中出现次数 |
| --- | --- | --- |
| 好 | 2 | 0 |
| 书 | 2 | 1 |
| 垃圾 | 0 | 2 |
| 太 | 0 | 1 |

好评总词数 = 2 + 2 = 4
差评总词数 = 1 + 2 + 1 = 4

第三步 拉普拉斯平滑（α=1，N=4 为词汇表大小）：

$$P(\text{词} \mid \text{类别}) = \frac{\text{词在类别中出现次数} + 1}{\text{类别总词数} + N}$$

对于 好评：

$$P(\text{书} \mid \text{好评}) = \tfrac{2+1}{4+4} = \tfrac{3}{8}$$

$$P(\text{垃圾} \mid \text{好评}) = \tfrac{0+1}{4+4} = \tfrac{1}{8}$$

对于 差评：

$$P(\text{书} \mid \text{差评}) = \tfrac{1+1}{4+4} = \tfrac{2}{8} = \tfrac{1}{4}$$

$$P(\text{垃圾} \mid \text{差评}) = \tfrac{2+1}{4+4} = \tfrac{3}{8}$$

未平滑反例：

$$P(\text{垃圾} \mid \text{好评}) = \tfrac{0}{4} = 0 \;\Rightarrow\; P(\text{好评} \mid \text{书}, \text{垃圾}) = 0$$

第四步 应用朴素贝叶斯（假设词独立）：

在已知文档类别（比如"好评"或"差评"）的前提下，各个词的出现是相互独立的

$$P(\text{词}_1, \text{词}_2 \mid C) \approx P(\text{词}_1 \mid C) \cdot P(\text{词}_2 \mid C)$$

最终两个分子：

$$P(\text{书}, \text{垃圾} \mid \text{好评}) \cdot P(\text{好评}) = \left(\tfrac{3}{8} \times \tfrac{1}{8}\right) \times 0.5 = \tfrac{3}{64} \times 0.5 = \tfrac{3}{128} \approx 0.0234$$

$$P(\text{书}, \text{垃圾} \mid \text{差评}) \cdot P(\text{差评}) = \left(\tfrac{2}{8} \times \tfrac{3}{8}\right) \times 0.5 = \tfrac{6}{64} \times 0.5 = \tfrac{6}{128} \approx 0.0469$$

第五步 计算真实后验概率。

考虑到：一条评论要么是好评，要么是差评，没有第三种

全概率公式（如果所有可能的类别是 C₁,C₂,…,Cₖ，且它们**互斥且覆盖全部可能性**）：

$$P(E) = \sum_{i=1}^{k} P(E \mid C_i) \cdot P(C_i)$$

比如：
P（发烧）= P（普通感冒）*P(发烧| 普感)
+ P（病毒感冒）*P(发烧| 病毒感冒)
+ P（第三种）*P（发烧| 第三种）

代入本例求分母：

$$P(\text{书}, \text{垃圾}) = P(\text{书}, \text{垃圾} \mid \text{好评}) \cdot P(\text{好评}) + P(\text{书}, \text{垃圾} \mid \text{差评}) \cdot P(\text{差评})$$

$$\text{好评分子} + \text{差评分子} = \tfrac{3}{128} + \tfrac{6}{128} = \tfrac{9}{128}$$

所以最终后验：

$$P(\text{好评} \mid \text{书}, \text{垃圾}) = \frac{3/128}{9/128} = \tfrac{1}{3} \approx 33.3\%$$

$$P(\text{差评} \mid \text{书}, \text{垃圾}) = \frac{6/128}{9/128} = \tfrac{2}{3} \approx 66.7\%$$

> 备注：朴素贝叶斯假设 必须带有条件      P(程序员,超重|喜欢) = P(程序员|喜欢) * P(超重|喜欢)
> 所以注意  这里 P（书、垃圾） 不等于 P（书） * P（垃圾） 因为在自然语言处理（NLP）中，几乎从来不会假设"完全独立"。

> 【知道】API 介绍

**核心 API**：

```python
from sklearn.naive_bayes import MultinomialNB  # 多项分布朴素贝叶斯
clf = MultinomialNB(alpha=1.0)  # alpha = 拉普拉斯平滑系数
clf.fit(X_train, y_train)
clf.predict(X_test)
clf.predict_proba(X_test)
```

**家族成员**（按特征分布选）：

| 类 | 特征类型 | 典型场景 |
|---|---|---|
| `MultinomialNB` | 离散计数 | 文本词频 / TF-IDF |
| `BernoulliNB` | 0/1 二值 | 词是否出现（短文本） |
| `GaussianNB` | 连续高斯 | 数值特征（鸢尾花等） |

> 【实践】商品评论情感分析

**任务**：评论文本 → {好评 1, 差评 0}。

**需求**

已知商品评论数据，根据数据进行情感分类（好评、差评）

| Unnamed: 0 | 内容 | 评价 |
| --- | --- | --- |
| 0 | 从编程小白的角度看，入门级别的 | 好评 |
| 1 | 很好的入门书，简洁全面，适合小白 | 好评 |
| 2 | 讲解全面，许多小细节都有顾及，三个小例子很完整 | 好评 |
| 3 | 前半部分讲概念深入浅出，要言不烦，很赞 | 好评 |
| 4 | 看的过程，一遍感觉不错，但是仔细看一下，才发现整个一个的概念都不清晰，例子莫名其妙，整体看起来很乱 | 差评 |
| 5 | 中规中矩的教材，零基础的看了依旧看不懂 | 差评 |
| 6 | 内容太浅显，个人认为不适合任何有C语言编程基础的人 | 差评 |
| 7 | 破书一本 | 差评 |
| 8 | 适合完全没有C语言经验的小白看，有其他语言经验的可以去看其他的书 | 好评 |
| 9 | 基础知识写的挺好的 | 好评 |
| 10 | 太基础 | 差评 |
| 11 | 略，啰嗦。适合完全没有编程经验基础的小白 | 差评 |
| 12 | 真的很不建议买 | 差评 |

（注：第 4 行长文本在原图中被截断，此处按 PPT 截图可读部分尽量复原；以原 .csv 文件为准。）

**商品评论情感分析流程**

\# 1 获取数据\# 2 数据基本处理    \#2-1 处理数据y    \# 2-2 加载停用词    \# 2-3 处理数据x 把文档分词    \# 2-4 统计词频矩阵 作为句子特征\# 3 准备训练集测试集
\# 4 模型训练   \# 4-1 实例化贝叶斯 添加拉普拉斯平滑参数    \# 4-2 模型预测\# 5 模型评估

**步骤**：

1. **获取数据**：评论 CSV，列含 `内容` / `评价`
2. **数据处理**：
   - 取出 `内容` 列做语料
   - 标签映射（"好评"→1，"差评"→0）
   - 加载停用词表
   - jieba 分词 → 用空格拼接成"伪英文"格式
3. **特征工程**：`CountVectorizer`（或 `TfidfVectorizer`）把文本转词频矩阵，过滤停用词
4. **划分训练/测试集**
5. **模型训练**：`MultinomialNB().fit(...)`
6. **模型评估**：`score` / `classification_report`

**代码骨架**：

> PPT 原代码（保真照抄，含被 PPT 文本框压扁的换行 / 中文全角引号 / 已废弃 API）

```text
import numpy as npimport pandas as pdimport matplotlib.pyplot as pltimport jiebafrom sklearn.feature_extraction.text import CountVectorizerfrom sklearn.naive_bayes import MultinomialNB   # 多项分布朴素贝叶斯def dm02_模型训练():    # 1 获取数据    data_df = pd.read_csv('./data/书籍评价.csv', encoding='gbk')    print('data_df-->\n', data_df)    # 2 数据基本处理    # 2-1 处理数据y    data_df['评论标号'] = np.where(data_df['评价'] == '好评', 1, 0)    y = data_df['评论标号']    print('data_df-->\n', data_df)    # 2-2 加载停用词    stopwords = []    with open('./data/stopwords.txt', 'r', encoding='utf-8') as f:        lines = f.readlines()        stopwords = [line.strip() for line in lines]    stopwords = list(set(stopwords))  # 去重
```

```text
    # 2-3 处理数据x 把文档分词    comment_list = [','.join(jieba.lcut(line)) for line in data_df['内容']]    # print('comment_list-->\n', comment_list)    # 2-4 统计词频矩阵 作为句子特征    transfer = CountVectorizer(stop_words=stopwords)    x = transfer.fit_transform(comment_list)    mynames = transfer.get_feature_names()    x = x.toarray()    # 3 准备训练集测试集    x_train = x[:10, :]         # 准备训练集    y_train = y.values[0:10]    x_test = x[10:, :]          # 准备测试集    y_test = y.values[10:]    print('x_train.shape-->',x_train.shape)    print('y_train.shape-->',y_train.shape)
```

```text
    # 4 模型训练    # 4-1 实例化贝叶斯 # 添加拉普拉修正平滑参数    mymultinomialnb = MultinomialNB()    mymultinomialnb.fit(x_train, y_train)        # 4-2 模型预测    y_pred = mymultinomialnb.predict(x_test)    print('预测值-->', y_pred)    print('真实值-->', y_test)        # 5 模型评估    myscore = mymultinomialnb.score(x_test, y_test)    print('myscore-->', myscore)
```

**坑点**：

- 分词后必须用空格拼回字符串，`CountVectorizer` 默认按空格切词
- 中文停用词表要单独加载，sklearn 自带的是英文表
- 类别极不平衡时，`MultinomialNB` 会偏向多数类——先看 `value_counts()`
- 文本量大时换 `TfidfVectorizer`，对常见词做降权

## Sources

- [sklearn — Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [sklearn — `CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- [jieba 中文分词 README](https://github.com/fxsjy/jieba)
