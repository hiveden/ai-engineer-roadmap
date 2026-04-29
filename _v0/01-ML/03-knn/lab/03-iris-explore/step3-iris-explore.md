# 复习题：鸢尾花数据集探索 + 可视化

> 对应代码：[`step3_iris_explore.py`](./step3_iris_explore.py)
> 范围：sklearn 数据集 API / numpy 与 pandas 互转 / seaborn 散点图

---

## A. sklearn 数据集 API

### Q1 `load_iris()` 返回的是什么类型？怎么取数据？
- 返回类型？（提示：Bunch 对象）
- 既能 `mydataset.data` 也能 `mydataset['data']`，为什么？
- 关键 keys 都有哪几个？分别装什么？

### Q2 `mydataset.data` 和 `mydataset.target` 的 shape 分别是？dtype 分别是？
- data shape = ?
- target shape = ?
- 为啥 target 是整数 0/1/2 而不是字符串"setosa"？怎么从整数还原回字符串？

### Q3 `mydataset.DESCR` 是什么？为什么探索阶段必看？
- 是字符串还是结构化数据？
- 包含哪些信息？（样本数 / 特征含义 / 类别含义 / 数据来源）
- 工程类比：DESCR ≈ ?

---

## B. numpy ↔ pandas 互转

### Q4 为什么不能直接把 `mydataset.data` 喂给 seaborn？必须转 DataFrame？
- numpy ndarray 缺什么？（提示：列名 / 行索引）
- seaborn 的设计哲学：列驱动 column-oriented，必须有列名

### Q5 `pd.DataFrame(mydataset.data, columns=mydataset.feature_names)` 这行干了什么？
- 参数 1（数据）：shape 必须？
- 参数 2（columns）：长度必须等于？
- 不传 columns 默认列名是啥？

### Q6 `data['label'] = mydataset.target` 这行为啥能直接赋值新列？
- pandas 列赋值的对齐规则
- 如果 mydataset.target 长度 ≠ data 行数会发生什么？
- 这跟 numpy 的"列拼接"（`np.column_stack` / `np.hstack`）有什么区别？

---

## C. seaborn 可视化

### Q7 `sns.lmplot` 的几个关键参数都在做什么？
- `data` / `x` / `y`：基础三件套
- `hue='label'`：作用是？（按类别分组上色）
- `fit_reg=False`：默认是 True 会画啥？为啥探索分布时要关掉？

### Q8 这份代码画了花瓣长 vs 花瓣宽。为什么注释里的另一个组合（花萼长 vs 花萼宽）被注释掉了？
- 提示：不同特征对的"分类区分度" discriminative power 不一样
- 花瓣维度（petal length/width）对 iris 三类几乎线性可分
- 花萼维度（sepal length/width）三类有重叠
- 这个观察对后续选特征 / 评估 KNN 决策边界有什么启示？

---

## D. 工程小坑

### Q9 探索一份**陌生**数据集时，建议的标准动作清单是？
- 看 `keys()` / `shape` / `dtypes` / `DESCR`
- 看类别分布（`Counter(y)` 或 `pd.Series(y).value_counts()`）
- 画 pairplot 或散点矩阵观察特征相关性
- 检查缺失值 / 异常值

### Q10 这份代码用的是 sklearn toy dataset（150 行内存装得下）。如果数据集是 100GB CSV 该怎么探索？
- 不能 `pd.read_csv` 全读
- 用 `nrows=` / `chunksize=` 抽样读
- 用 `pyarrow` / `polars` / `dask` 这种 lazy 引擎
- `df.sample(n=10000)` 抽样后再画图

---

## 答题状态

- [ ] Q1 Bunch 对象
- [ ] Q2 shape / dtype
- [ ] Q3 DESCR 是啥
- [ ] Q4 必须 DataFrame
- [ ] Q5 columns 参数
- [ ] Q6 列赋值对齐
- [ ] Q7 lmplot 参数
- [ ] Q8 特征区分度
- [ ] Q9 探索标准动作
- [ ] Q10 大数据集探索
