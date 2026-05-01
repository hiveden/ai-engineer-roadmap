# 第 2 章 · 情感分析案例

> 维度：**代码**
>
> 本章按 H3 节摘要型填充，不做单点拆分。后续如需深讲，可拆为：
>
> 1. `01-api.md` — `MultinomialNB` 与变体
> 2. `02-pipeline.md` — 文本 → 向量 → 训练 → 评估全流程

## 底稿

> 02 · 朴素贝叶斯实战

**学习目标**：

1. 知道 `sklearn.naive_bayes` 提供的朴素贝叶斯 API
2. 能用朴素贝叶斯做商品评论情感分类（好评 / 差评）

> 【知道】API 介绍

**核心 API**：

```python
from sklearn.naive_bayes import MultinomialNB
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

```python
import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 1. 数据
df = pd.read_csv("reviews.csv")
df["label"] = (df["评价"] == "好评").astype(int)

# 2. 分词 + 停用词
stopwords = set(open("stopwords.txt", encoding="utf-8").read().split())
def cut(text):
    return " ".join(w for w in jieba.cut(text) if w not in stopwords)
df["tokens"] = df["内容"].map(cut)

# 3. 向量化
vec = CountVectorizer()
X = vec.fit_transform(df["tokens"])
y = df["label"].values

# 4. 划分
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 训练 + 评估
clf = MultinomialNB(alpha=1.0)
clf.fit(X_tr, y_tr)
print(clf.score(X_te, y_te))
```

**坑点**：

- 分词后必须用空格拼回字符串，`CountVectorizer` 默认按空格切词
- 中文停用词表要单独加载，sklearn 自带的是英文表
- 类别极不平衡时，`MultinomialNB` 会偏向多数类——先看 `value_counts()`
- 文本量大时换 `TfidfVectorizer`，对常见词做降权
