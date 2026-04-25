# 答案：数据集分割 + 分层抽样

> 配套题目：[`step4-split-stratify.md`](./step4-split-stratify.md)
> 对应代码：[`step4_split_stratify.py`](./step4_split_stratify.py)

---

## A. train_test_split 基础

### Q1 返回值

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
```

**返回顺序固定**：`X_train, X_test, y_train, y_test`——记顺序：先 X 后 y，先 train 后 test。

**为啥不返回 tuple**：sklearn 设计哲学是"扁平、可解构"，符合 numpy 习惯。但代价是顺序记错就翻车。

**写错的常见姿势**：

```python
X_train, X_test = train_test_split(X, y)   # ❌
# 实际返回 4 个值，前两个是 X_train 和 X_test，y 全丢了
# 不会报错（解构数量对不上才报），但后续训练报错"y 维度不对"
```

**安全写法**：永远把 4 个值都解构出来，或显式只传 X：

```python
X_train, X_test = train_test_split(X)  # 只传一个 X 时只返回 2 个值
```

---

### Q2 `test_size` 浮点 vs 整数

| 类型 | 含义 | 例子 |
|---|---|---|
| float (0, 1) | 比例 | `test_size=0.2` → 测试集占 20% |
| int | 绝对样本数 | `test_size=30` → 测试集 30 个样本 |

**`train_size` 参数也存在**：可以两个都设，但要保证 train_size + test_size ≤ 1.0（或样本数 ≤ 总数）。一般只设一个就够了。

```python
train_test_split(X, y, train_size=0.6, test_size=0.2)  # 剩下 20% 不进任何集合（不推荐）
```

---

### Q3 `random_state` 复现

**作用**：固定随机种子，使每次切分结果**完全一致**——可复现 reproducibility。

**底层机制**：sklearn 内部用 `np.random.RandomState(seed)` 生成"打乱顺序"的索引。同一个 seed → 同一个索引序列 → 同样的切分。

**不写 random_state 默认行为**：
- sklearn 0.22+ 默认是 `None`，每次跑用全局 numpy 随机数生成器
- 后果：每次切分结果都不同，调参时分数波动看不出是模型变好还是切分运气好

**学习/调参阶段必须固定**：否则你"调好了"k=7 比 k=5 高 2%，可能下次跑就反过来。

**生产场景**：
- **训练**：可以不固定（每次见到新数据切一次），让模型见到更多分布
- **A/B 实验**：必须固定（保证两组用户用同一份切分）
- **复现 bug**：必须固定（线上出错时能本地重现）

42 是 *Hitchhiker's Guide* 致敬数字，sklearn 教程默认值。

---

## B. stratify 分层抽样

### Q4 stratify 在做什么？

**答**：保证训练集和测试集的**类别比例**与原数据集一致。

iris 三类各 50 个（共 150）：

| | 不加 stratify（随机） | 加 stratify |
|---|---|---|
| 训练集（120）| 0/1/2 类各 ~40，但有方差（可能 38/42/40） | 严格 40/40/40 |
| 测试集（30）| 0/1/2 类各 ~10，方差更大（可能 7/12/11） | 严格 10/10/10 |

**iris 类别均衡 + 数据量大**，加不加 stratify 差异不大。**类别不均衡 + 数据量小**才能看出威力（见 Q5）。

**stratify 实现**：内部用 `StratifiedShuffleSplit`——按类别分组，**每组内部独立随机抽 20%**，最后合并。

---

### Q5 1% 正样本不加 stratify 的灾难

**场景**：n=1000 总样本，正样本 10 个（1%），负样本 990 个。

**不加 stratify 切 80/20**：
- 测试集大小：200
- 测试集正样本**期望数**：200 × 0.01 = 2
- 实际抽样方差大：可能 0 个正样本（运气坏），可能 5 个（运气好）

**抽到 0 个正样本的后果**：

| 指标 | 值 | 含义 |
|---|---|---|
| accuracy | 99% | "全猜负" 也有 99%（200/200 全对） |
| precision | 0/0 = NaN | 分母 = 模型预测的正样本数，可能是 0 |
| recall | 0/0 = NaN | 分母 = 真实正样本数，是 0 |
| F1 | NaN | 衍生指标也废 |

**等于评估完全失效**。

**加 stratify 后**：测试集正样本严格 = 2 个（200 × 0.01），评估指标稳定。

**类比**：你给负载均衡器配权重 99:1，结果某次 sample 里 100% 流量打到 1% 那台机器上——这就是"小概率事件被放大"。stratify 强制你"按比例分配"，避免方差炸裂。

---

### Q6 stratify 的限制

**能传什么**：任意 1D 数组（不限于 y），只要长度等于样本数。

**多分类多标签场景**：

```python
# 单标签多分类：直接传 y
stratify=y_multiclass  # ✅

# 多标签（每样本可能属于多类）：sklearn 原生不支持
# 解决方案：
# 1. 把多标签合并成"组合标签"字符串
y_combined = ['_'.join(map(str, row)) for row in y_multilabel]
# 2. 用 iterative-stratification 库（PyPI）
from skmultilearn.model_selection import IterativeStratification
```

**时间序列数据不能用 stratify**：

```python
# ❌ 错的姿势
train_test_split(X_timeseries, y, stratify=y)
# 问题：stratify 需要 shuffle，但时间序列必须保持顺序

# ✅ 正确姿势
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X_timeseries):
    ...
# 永远是"过去训练，未来测试"
```

---

## C. 切分前后的检验

### Q7 切完第一件事

**3 步快速 sanity check**：

```python
from collections import Counter
import numpy as np

# 1. 数量对不对
assert x_train.shape[0] + x_test.shape[0] == X.shape[0]
assert x_train.shape[1] == x_test.shape[1] == X.shape[1]  # 列数一致

# 2. 类别比例对不对
print('原始:', Counter(y))
print('训练:', Counter(y_train))
print('测试:', Counter(y_test))
# 比例应该接近原始（加了 stratify 会精确一致）

# 3. 特征分布对不对（可选）
print('训练 mean:', x_train.mean(axis=0))
print('测试 mean:', x_test.mean(axis=0))
# 接近 → OK，差很多 → covariate shift（见 Q8）
```

---

### Q8 covariate shift

**定义**：训练集和测试集的**特征分布** P(X) 不同（即使 P(y|X) 相同）。

**典型成因**：
- 数据采集时间跨度大（用户行为随时间漂移）
- 设备型号不同（手机摄像头 A 拍的图 vs 摄像头 B 的）
- 抽样方法有偏（训练集只采集了某地区，测试集是全国）

**这次切分发现 covariate shift 怎么办**：

| 场景 | 解决 |
|---|---|
| `random_state` 运气不好 | 换 seed 重切，或者用交叉验证消除单次切分方差 |
| 数据本身有时间漂移 | 用 `TimeSeriesSplit` 或加 timestamp 特征让模型显式学时间 |
| 数据量太小 | 收集更多数据，或者用 bootstrap 增强 |

**KNN 对 covariate shift 极度敏感**——训练集没见过的特征区域，预测就是瞎猜（找不到合理近邻）。所以 KNN 在分布漂移大的场景表现差。

---

## D. 工程坑

### Q9 shuffle=False

**默认 `shuffle=True`**：切分前先打乱样本顺序。

**关闭场景**：

```python
# 时间序列，前 80% 训练后 20% 测试
X_train, X_test, y_train, y_test = train_test_split(
    X_timeseries, y, test_size=0.2,
    shuffle=False,  # 关键！
)
```

**关闭后样本按原顺序切**——iris 数据集如果不打乱，前 50 行全是类 0，所以默认 `shuffle=True` 是必须的。

**`shuffle=False` + `stratify=y` 同时用会报错**：

```python
ValueError: Stratified train/test split is not implemented for shuffle=False
```

原因：分层抽样**必然涉及打乱**（要在每类内部随机抽 20%）。如果不能打乱，就只能"顺序切"，无法保证比例。两者互斥。

时间序列任务一般也不需要 stratify——按时间切就完了。

---

### Q10 三段式 train / val / test

**一次切不出三段**，需要连切两次：

```python
# 第一次：切出测试集（最终评估，全程不能碰）
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 此时 X_temp 是剩下的 80%

# 第二次：在 X_temp 内部切出训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)
# 0.8 × 0.25 = 0.2，最终 60/20/20
```

**第二次的 0.25 怎么算**：要从 80% 里再切出 20%（占总数），所以是 0.2 / 0.8 = **0.25**（占剩余的 25%）。

**三段式的工程理由**：

| 集合 | 用途 | 看几次 |
|---|---|---|
| **train** | 训练模型，调权重 | 每次训练 |
| **val** | 调超参（k / learning rate / 树深度...），做模型选择 | 每次调参 |
| **test** | **最终**评估，模拟"模型上线后真实表现" | **只在选定模型后跑一次** |

**为啥不能用 test 调参**：调参时反复看 test 分数，等于让 test 影响了模型选择——模型间接"记住"了 test，最终上线表现会比 test 分数差。这是变种 data leakage。

**业界惯例**：训练 / 验证 / 测试 = 60/20/20 或 70/15/15。数据少时可以用 5-fold CV 替代 val（见 `06-cv-gridsearch` 主题）。

---

## 一句话总结

> **`train_test_split` 是分类任务起手式——记住四件事**：
> 1. 返回 `(X_train, X_test, y_train, y_test)` 顺序不能错
> 2. `random_state` 学习阶段必固定，否则分数没法对比
> 3. `stratify=y` 分类任务**默认应该开**，不开就赌运气
> 4. 时间序列别用它，用 `TimeSeriesSplit`
