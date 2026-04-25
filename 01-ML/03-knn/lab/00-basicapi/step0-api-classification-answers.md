# 答案：KNN API 基础（分类 + 回归）

> 配套题目：[`step0-api-classification.md`](./step0-api-classification.md)
> 对应代码：[`step0_api_classification.py`](./step0_api_classification.py)
> 用法：先去答题文件做完，卡住或答完后再翻这里对答案。

---

## A. 输入数据形状

### Q1 `X = [[0], [1], [2], [3]]` 为啥要套两层 `[]`？

**答**：sklearn 全家桶 fit/predict 对 X 的硬规矩——必须 2D，shape `(n_samples, n_features)`。

- `[[0],[1],[2],[3]]` shape = `(4, 1)`：4 个样本 × 1 个特征 ✅
- `[0,1,2,3]` shape = `(4,)`：1D 数组，sklearn **无法判断**这是"4 个 1D 样本"还是"1 个 4D 样本"——二义性，所以直接拒绝

**实际报错**：
```
ValueError: Expected 2D array, got 1D array instead.
Reshape your data either using array.reshape(-1, 1) if your data has a single feature
or array.reshape(1, -1) if it contains a single sample.
```

**业务锚**：X 是数据库表，行=样本，列=字段。一张表至少 2 维，不存在"一维表"。哪怕只有 1 列也得是 `n×1` 列向量，不是裸数组。

---

### Q2 `y = [0, 0, 1, 1]` 为什么是 1D？回归 y 也 1D，区别在哪？

**答**：
- y 是**标签向量** label vector，每个样本对应一个标签 → shape `(n_samples,)` 天然 1D
- 这是 sklearn 单输出任务的接口契约。X 是表（多列特征），y 是表里"那一列"label

**分类 vs 回归的 y 差异**（仅在数值类型）：

| | 分类 | 回归 |
|---|---|---|
| dtype | `int` / `str` / `object`（类别编号或名字） | `float` |
| 含义 | 离散标签（如 0=猫, 1=狗） | 连续值（如房价、温度） |
| 取值集合 | 有限 | 无限 / 实数 |

**多输出场景**（这份 demo 不涉及）：同时预测多个目标值时 y 才会变成 2D `(n_samples, n_targets)`，比如同时预测一个房子的"价格 + 面积估值"。

---

### Q3 `predict([[4]])` 为啥也要 2D？

**答**：predict 是**批量接口** batch API。sklearn 整套设计哲学是向量化——一次推理能处理 n 条样本，所以输入 shape 必须 `(n_samples, n_features)`，哪怕 n=1 也得包成 batch。

- `predict([4])` → 报 `ValueError: Expected 2D array...`（同 X 错误）
- `predict([[4]])` → 返回 shape `(1,)`，比如 `array([1])`
- `predict([[4],[5]])` → 返回 shape `(2,)`，比如 `array([1, 1])`

**REST API 类比**：成熟的预测服务接口都长这样——
```http
POST /predict
Content-Type: application/json
{ "samples": [[4], [5]] }   ← 永远是数组，单条/批量统一 schema
```
sklearn 把这个习惯下沉到了 Python 层。好处是单条/批量代码一份，性能也能利用 numpy 向量化。

---

## B. fit 和 predict 各自做了什么

### Q4 `model.fit(X, y)` 在 KNN 这里做了什么？

**答**：几乎啥都没做。具体步骤：

1. 校验 X/y 形状、dtype、是否含 NaN
2. 把训练集存到内部属性：`self._fit_X = X`, `self._y = y`
3. 根据 `algorithm` 参数：
   - `'brute'`（默认在某些场景）→ 啥树都不建，纯存数据
   - `'kd_tree'` / `'ball_tree'` → 建立空间索引树（这是 fit 唯一会"工作"的情况）
   - `'auto'` → 启发式选一个

**复杂度**：
- brute：O(1) 级别，几乎瞬间
- 建树：O(n · d · log n)

**对比逻辑回归的 fit**：
- 逻辑回归要做梯度下降，迭代算 `weights`、`bias`，可能跑成百上千轮 epoch → eager learner（早干活）
- KNN fit 不学任何参数，把"理解数据"全部延迟到 predict → **lazy learner**

**lazy 的本质**：决策边界不是 fit 时画好的（像逻辑回归画一条直线），而是 predict 时**在线**根据查询点周围的邻居动态生成。代价就是每次推理都要重新算一遍。

---

### Q5 `predict` 才是真正干活的地方，干了哪几步？

**答**：3 步流水线。

```
查询点 x_query
  ↓
Step 1: 算距离
  对训练集每个点算 d(x_query, x_train_i)，默认欧氏（Minkowski p=2）
  brute 模式：O(n_train · d)
  ↓
Step 2: 找邻居
  按距离排序，取最小的 k 个（argpartition，比全排序快）
  ↓
Step 3: 合并答案
  ┌── 分类：majority vote（k 个邻居标签里出现次数最多的）
  └── 回归：mean（k 个邻居 y 的算术平均；weights='distance' 时按 1/d 加权）
```

**分类 vs 回归的代码共享情况**：Step 1 和 Step 2 完全是同一套代码（在 sklearn 内部 `NeighborsBase` 父类里）。差异**只在 Step 3**——`KNeighborsClassifier` 子类实现 vote，`KNeighborsRegressor` 子类实现 mean。这就是为啥分类和回归性能特征几乎一样，调优旋钮也一样。

---

## C. 分类 vs 回归（demo 的核心对比）

### Q6 dm01 和 dm02 在 sklearn API 上差哪几个名字？

**答**：

| 维度 | dm01 分类 | dm02 回归 |
|---|---|---|
| 类名 | `KNeighborsClassifier` | `KNeighborsRegressor` |
| y 类型 | 离散标签 `[0,0,1,1]` | 连续值 `[0.1,0.2,0.3,0.4]` |
| `predict()` 返回 | 类别标签（dtype 跟 y 一致） | float |
| `predict_proba()` | ✅ 返回 `(n_samples, n_classes)` 概率矩阵 | ❌ 没这接口（连续值没"概率"一说） |
| `classes_` 属性 | ✅ 记录所有类别 | ❌ 没这属性 |
| `score()` 默认指标 | accuracy | R²（决定系数） |

`predict_proba` 在分类里很有用——可以拿来做**阈值调节**。比如风控场景，宁可错杀不可放过，可以把"判 1"的阈值从默认 0.5 调到 0.3。

---

### Q7 "投票" vs "平均"分别怎么算？逐字算给我看。

**dm01（分类，k=1，查 `[[4]]`）**：

| 训练样本 | 距离 \|4 - x\| |
|---|---|
| `[0]` (y=0) | 4 |
| `[1]` (y=0) | 3 |
| `[2]` (y=1) | 2 |
| `[3]` (y=1) | **1** ← 最近 |

k=1 → 直接抄最近邻 `[3]` 的标签 → **预测 `[1]`** ✅

**dm02（回归，k=2，查 `[[3,11,10]]`）**：

| 训练样本 | 欧氏距离 |
|---|---|
| `[0,0,1]` (y=0.1) | √(9+121+81) = √211 ≈ 14.53 |
| `[1,1,0]` (y=0.2) | √(4+100+100) = √204 ≈ 14.28 |
| `[3,10,10]` (y=0.3) | √(0+1+0) = **1.00** ← 最近 |
| `[4,11,12]` (y=0.4) | √(1+0+4) = √5 ≈ **2.24** ← 次近 |

k=2 → 取最近的 2 个 → 标签 `[0.3, 0.4]` → 默认 `weights='uniform'` 等权平均：

$$\hat{y} = \frac{0.3 + 0.4}{2} = 0.35$$

→ **预测 `[0.35]`** ✅

**`weights='uniform'` 的含义**：距离信息**只用于挑选邻居**（决定哪 2 个进入候选），不参与加权。哪怕一个距离 1.00、一个距离 2.24，也是各占 50%。

如果换成 `weights='distance'`，公式变成按 1/d 加权：

$$\hat{y} = \frac{0.3/1.00 + 0.4/2.24}{1/1.00 + 1/2.24} = \frac{0.3 + 0.179}{1 + 0.446} \approx 0.331$$

更靠近距离近那个邻居（0.3）。

---

## D. 超参 n_neighbors

### Q8 dm01 用 k=1，dm02 用 k=2，各自隐患？

**k=1（dm01）**：
- 决策边界完全贴合训练集每个点 → **过拟合 overfitting**，高 variance
- 噪声点直接当答案：训练集里有个错标的样本，落在它附近的查询都会被误导
- 课堂演示用 k=1 是为了让结果好算（手算"找最近那个"），生产慎用

**k=2 分类（如果 dm01 改成 k=2）**：
- **平票风险**：偶数邻居 + 投票 = 可能两类各占一半
- sklearn 默认按类别索引最小的赢（不稳定，依赖 label encoding 顺序）
- 所以分类 k 一律选**奇数**：3 / 5 / 7

**k=2 回归（dm02）**：
- 没有平票问题——数值平均永远有解（即使两个 y 完全相同，平均还是那个值）
- 但 k 太小依然意味着高 variance：邻居换一个，预测就跳一下
- 回归 k 选偶数没事，但小 k 仍不推荐（这份 demo 只有 4 个训练样本，所以才用 2）

---

### Q9 dm01 改成 k=4 会发生什么？

**答**：训练集只有 4 个样本，k=4 = 全员到齐参与投票。

```python
estimator = KNeighborsClassifier(n_neighbors=4)
estimator.fit([[0],[1],[2],[3]], [0, 0, 1, 1])
estimator.predict([[4]])
estimator.predict([[100]])  # 哪怕查个非常远的点
estimator.predict([[-50]])  # 同样
```

4 个邻居标签 = `[0, 0, 1, 1]`，**2:2 完美平票**。

**sklearn 平票破解规则**：按类别索引（`classes_` 数组里的下标）最小的赢。`classes_ = array([0, 1])`，`0` 索引更小 → 全部预测 `0`。

**副作用**：**所有**查询点都会被预测成 `0`，模型完全失去区分能力。这是 k=n 的极端退化——模型坍缩成"永远预测最常出现且索引最小的类"。

**两个教训**：
1. k 不能取到训练集大小附近，否则模型变成"猜全局多数类"
2. 即使 k 远小于 n，偶数 k 在分类里也容易平票——奇数才稳

可以把 k 改成 4 实际跑一下，验证是不是无论查啥都返回 0。

---

## 一句话总结

> **KNN 的 fit 只是 INSERT，predict 才是 SELECT + ORDER BY + LIMIT k + 聚合**。
> 分类和回归只差最后那个聚合是 `MODE()` 还是 `AVG()`。
