# 答案：鸢尾花数据集探索 + 可视化

> 配套题目：[`step3-iris-explore.md`](./step3-iris-explore.md)
> 对应代码：[`step3_iris_explore.py`](./step3_iris_explore.py)

---

## A. sklearn 数据集 API

### Q1 `load_iris()` 返回什么？

**答**：`sklearn.utils.Bunch` 对象——dict 的子类，**同时支持** `obj.key` 和 `obj['key']` 两种访问。

```python
type(mydataset)  # <class 'sklearn.utils.Bunch'>
mydataset.data == mydataset['data']  # True，等价
```

**为什么 sklearn 这么设计**：toy dataset 既要像 dict 一样灵活（支持迭代 keys、序列化），又要让用户用 `.data` 这种属性访问省去引号。Bunch = dict + AttrDict，两全。

**关键 keys**：

| key | 内容 |
|---|---|
| `data` | 特征矩阵 X，shape (n_samples, n_features) |
| `target` | 标签向量 y，shape (n_samples,) |
| `target_names` | 类别名字数组（按 target 整数索引） |
| `feature_names` | 特征列名 |
| `DESCR` | 数据集说明文档（长字符串） |
| `frame` | 整合好的 DataFrame（如果 `as_frame=True` 才有） |
| `filename` | 原始数据文件路径 |

---

### Q2 data / target 的 shape 和 dtype

```python
mydataset.data.shape    # (150, 4)，dtype = float64
mydataset.target.shape  # (150,)，  dtype = int64
mydataset.target_names  # array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
```

**为啥 target 是整数**：
- sklearn 的分类器（`KNeighborsClassifier` / `LogisticRegression` 等）内部用整数处理类别
- 字符串标签也支持，但会先做 LabelEncoder 转整数
- 整数表达紧凑、运算快、占内存少

**还原成字符串名字**：

```python
mydataset.target_names[mydataset.target]
# array(['setosa', 'setosa', ..., 'virginica'])  ← numpy 花式索引
```

`target_names[0]` = `'setosa'`，`target_names[1]` = `'versicolor'`，`target_names[2]` = `'virginica'`。本质就是 enum 的 ordinal → name 映射。

---

### Q3 DESCR 是什么？

**答**：长字符串，包含数据集的"说明书"。

```python
print(mydataset.DESCR)
```

输出包括：
- 样本数 / 特征数
- 每个特征的统计信息（min/max/mean/std）
- 类别分布
- 数据集来源、引用文献
- 缺失值情况
- 特征单位（cm）

**工程类比**：DESCR ≈ **OpenAPI Schema** 或 **数据库表的 COMMENT 字段**。拿到陌生数据先看 DESCR，等价于"调新接口先看文档"。

---

## B. numpy ↔ pandas 互转

### Q4 为啥要转 DataFrame？

**答**：numpy ndarray **没有列名**，只有位置索引（第 0 列 / 第 1 列…）。seaborn 是**列驱动**的——`x='petal length (cm)'` 这种字符串只能从 DataFrame 列名取，ndarray 不行。

类比：
- ndarray ≈ 没列名的二维数组（像没表头的 CSV）
- DataFrame ≈ 有列名的表（像数据库表）
- seaborn / pandas 这套是奔着"语义化列名"设计的，光位置索引信息密度太低

---

### Q5 `pd.DataFrame(data, columns=...)`

```python
data = pd.DataFrame(mydataset.data, columns=mydataset.feature_names)
```

| 参数 | 要求 |
|---|---|
| 数据 | shape 必须 2D `(n_rows, n_cols)`；可以是 ndarray / list of lists / dict |
| `columns` | 长度必须 = `n_cols`（4 列必须传 4 个名字） |

**不传 columns 时**：默认列名是 `0, 1, 2, 3, ...`（整数索引），可以用但不语义化。

```python
pd.DataFrame(mydataset.data)
#       0    1    2    3
# 0   5.1  3.5  1.4  0.2
# 1   4.9  3.0  1.4  0.2
# ...
```

---

### Q6 `data['label'] = mydataset.target` 列赋值

**答**：pandas 列赋值会按**行索引对齐 align**——

```python
data['label'] = mydataset.target  # numpy array 长度 150
```

只要长度跟 DataFrame 行数（150）一致，pandas 就**按位置**塞进去（numpy 没行索引，按位置 fallback）。

**长度不匹配会报错**：

```python
data['label'] = [1, 2, 3]
# ValueError: Length of values (3) does not match length of index (150)
```

**对比 `np.column_stack` / `np.hstack`**：

| | `data['label'] = ...` | `np.column_stack` |
|---|---|---|
| 操作对象 | DataFrame | ndarray |
| 列名 | 自动用赋值的 key | 没有，按位置索引 |
| 类型保留 | 每列独立 dtype | 整个矩阵统一 dtype（混合 int/float 会向上转 float） |
| 适用场景 | 表格数据加列 | 数值矩阵拼接 |

---

## C. seaborn 可视化

### Q7 `sns.lmplot` 参数

```python
sns.lmplot(data=data, x='petal length (cm)', y='petal width (cm)',
           hue='label', fit_reg=False)
```

| 参数 | 作用 |
|---|---|
| `data` | DataFrame |
| `x` / `y` | 列名（字符串） |
| `hue` | 按这列分组上色（categorical） |
| `fit_reg` | 是否画线性回归拟合线（默认 True） |

**为什么要 `fit_reg=False`**：lmplot 默认会画 `y ~ x` 的线性回归直线 + 95% 置信区间带。**对分类任务的探索没意义**——我们关心的是不同类别的点群分布，不是 x→y 的线性关系。关掉拟合线后变成纯散点图。

**等价写法**：直接用 `sns.scatterplot`，不用关 fit_reg。lmplot 是 scatterplot + 拟合线的组合，scatterplot 更轻量。

---

### Q8 为啥不画花萼，画花瓣？

**答**：观察散点图——

- **花瓣**（petal length × petal width）：三类**几乎线性可分**，setosa 完全独立成一团，versicolor 和 virginica 有少量重叠
- **花萼**（sepal length × sepal width）：三类**严重重叠**，肉眼几乎分不开

**这个观察的工程价值**：
1. **特征选择 feature selection**：花瓣两维已经把分类做了 80%，花萼两维其实只是噪声放大（可以做特征重要性排序证实）
2. **KNN 决策边界**：花瓣维度上 KNN 几乎闭眼都对，花萼维度上 KNN 容易在重叠区误判
3. **可视化即诊断**：训练前先肉眼看一眼 2D 投影，能预判模型上限。可视化也能做 PCA 降到 2D 后画——同样道理

**业务锚**：选特征 ≈ 写 SQL 的 WHERE 条件——选区分度高的列做过滤，选区分度低的列就是浪费。

---

## D. 工程小坑

### Q9 探索陌生数据集的标准动作

**5 步快速画像**：

```python
import pandas as pd
from collections import Counter

# 1. 维度
print(df.shape, df.dtypes)

# 2. 类别分布（标签）
print(Counter(y))
# 看是否平衡（不平衡需要 stratify / resample）

# 3. 数据范围（量纲）
print(df.describe())  # min/max/mean/std/quartile
# 看量纲差异（决定要不要 scaling）

# 4. 缺失值
print(df.isnull().sum())
# 决定填充策略

# 5. 可视化分布
import seaborn as sns
sns.pairplot(df, hue='label')  # 散点矩阵看特征间相关 + 类别可分性
```

**用 `df.head()` / `df.sample(5)` 看具体几行也是必备**——光看统计量看不出"原始数据长啥样"。

---

### Q10 100GB CSV 怎么探索？

**核心思路**：不全读，分块/抽样/lazy。

| 方法 | API | 何时用 |
|---|---|---|
| 抽样读 | `pd.read_csv(path, nrows=10000)` | 只看头部，了解 schema |
| 分块读 | `pd.read_csv(path, chunksize=100000)` | 流式累计统计量（mean / count） |
| 列裁剪 | `pd.read_csv(path, usecols=['x', 'y', 'label'])` | 只关心几列时省内存 |
| 类型优化 | `pd.read_csv(path, dtype={'id': 'int32', 'flag': 'bool'})` | 默认 int64 / float64 太奢侈 |
| 抽样后画图 | `df.sample(n=10000, random_state=42)` | 画图肉眼看够用 |
| Lazy 引擎 | `polars` / `dask.dataframe` / `pyarrow` | 数据真大、分布式 / 列存优化 |

**KNN 在 100GB 上根本跑不动**——回到我们 `00-basicapi` 答案 Q10 说的"训练集大小=模型大小"。100GB 数据训完模型就是 100GB。这种场景下应该换梯度模型（XGBoost）或深度学习。

---

## 一句话总结

> **sklearn 的 toy dataset = 训练数据 + 元信息打包好**。Bunch 对象既有 `.data`/`.target`/`.feature_names`，还有 `.DESCR` 说明书。探索阶段**先转 DataFrame 再喂可视化工具**，光 numpy 数组没法做语义化操作。
