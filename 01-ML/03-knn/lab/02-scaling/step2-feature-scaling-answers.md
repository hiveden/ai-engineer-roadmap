# 答案：特征缩放（归一化 + 标准化）

> 配套题目：[`step2-feature-scaling.md`](./step2-feature-scaling.md)
> 对应代码：[`step2_feature_scaling.py`](./step2_feature_scaling.py)

---

## A. 为什么要做特征缩放

### Q1 量纲不统一，KNN 会发生什么？

**答**：**大量纲特征霸占距离贡献**。

dm01 的 4 列假设是 `[年龄, 工龄, 学历, 评分]`：
- 第 1 列范围 60~90，跨度 30
- 第 2 列范围 2~4，跨度 2

算两条样本的欧氏距离：
$$d = \sqrt{(\Delta_1)^2 + (\Delta_2)^2 + (\Delta_3)^2 + (\Delta_4)^2}$$

假设两人年龄差 10、工龄差 1，平方后变 `100 + 1 = 101`——**第 1 列贡献了 99% 的距离**。第 2 列工龄实际上是被忽略的（虽然差异比例 50% 比年龄差异比例 11% 还大）。

KNN 的预测完全靠距离排序找邻居，等于只用了"年龄"这一列，其他特征白给。

**为什么逻辑回归/树模型受影响小**：
- 逻辑回归虽然也线性算特征，但每列有独立的权重 w，可以学一个小 w 来"压制"大量纲列。但梯度下降收敛会变慢（病态优化），所以**还是建议缩放**。
- 树模型（决策树/RF/XGBoost）按"分裂点"决策——`x_1 > 75` 这种判断，跟 x_1 是 75 还是 0.5 数学上等价。**完全不受量纲影响**。

---

### Q2 哪些算法依赖特征缩放？

| 类型 | 算法 | 为什么 |
|---|---|---|
| **依赖** | KNN | 距离敏感 |
| **依赖** | SVM（带核） | 核函数算距离 / 内积 |
| **依赖** | 神经网络 | 梯度下降，量纲不齐病态收敛 |
| **依赖** | K-means | 距离聚类 |
| **依赖** | PCA / LDA | 协方差矩阵特征值，被大量纲列主导 |
| **依赖** | 逻辑回归 / 线性回归（带正则） | L1/L2 正则会"惩罚"大权重，量纲不齐惩罚不公平 |
| **不依赖** | 决策树 | 单维度分裂点，与量纲无关 |
| **不依赖** | 随机森林 | 同上 |
| **不依赖** | XGBoost / LightGBM | 同上 |
| **不依赖** | 朴素贝叶斯（高斯版） | 各特征独立估计，量纲在自己列里 |

**一句话规律**：算"距离"或"梯度"的怕量纲，算"分裂点"或"独立分布"的不怕。

---

## B. MinMaxScaler 的计算细节

### Q3 dm01 第 1 列手算

第 1 列原值：`[90, 60, 75]`
- min = 60
- max = 90
- 跨度 = 30

套公式 `(x - 60) / 30`：

| 原值 | 计算 | 结果 |
|---|---|---|
| 90 | (90-60)/30 = 30/30 | **1.0** |
| 60 | (60-60)/30 = 0/30 | **0.0** |
| 75 | (75-60)/30 = 15/30 | **0.5** |

第 1 列归一化结果：`[1.0, 0.0, 0.5]` ✅

完整 4 列输出（你可以跑代码验证）：
```
[[1.   0.   0.   0. ]
 [0.   1.   1.   0.83333333]
 [0.5  0.5  0.6  1. ]]
```

**洞察**：每列都被独立压到 [0, 1]，**最大值变 1，最小值变 0**，中间值按比例线性映射。原始数据的相对大小关系完全保留（90>75>60 → 1.0>0.5>0.0）。

---

### Q4 按行还是按列？

**按列（column-wise）**。

每列代表一个**特征 feature**（年龄、工龄……），每行代表一个**样本 sample**（一个人）。

**按行归一化的反例**：把一个人的"年龄 90、工龄 2、学历 10、评分 40"算 min/max → min=2, max=90 → 90 变 1.0, 2 变 0.0。这等于把**完全不同物理含义**的特征混在一起算尺度，毫无意义。

**为什么按列对**：归一化的目的是"让每个特征的尺度一致"，所以每列独立处理才合理。年龄列内部找它自己的 min/max，工龄列找工龄列的 min/max，互不干扰。

sklearn 所有 scaler（`MinMaxScaler` / `StandardScaler` / `RobustScaler` ...）默认都按列。这跟它对 X 的硬规矩 shape `(n_samples, n_features)` 是配套的——列就是 feature。

---

### Q5 feature_range 参数

```python
MinMaxScaler(feature_range=(-1, 1))
```

公式变成：`x_scaled = (x - min) / (max - min) * (b - a) + a`，其中 `(a, b)` 是目标范围。

**`[-1, 1]` 的常见场景**：
- **神经网络输入**：很多激活函数（tanh、leaky relu）以 0 为中心，输入也居中能加速收敛
- **图像处理**：从 `[0, 255]` 像素值转 `[-1, 1]` 是 GAN/扩散模型的标准前处理
- **某些距离度量**：余弦相似度场景下输入对称分布更直观

KNN 一般用默认 `[0, 1]` 就够。

---

## C. StandardScaler 的计算细节

### Q6 dm03 第 1 列手算

第 1 列原值：`[90, 60, 75]`

**Step 1：mean**
$$\bar{x} = \frac{90 + 60 + 75}{3} = \frac{225}{3} = 75$$

**Step 2：var（注意是总体方差，分母 n）**
$$\sigma^2 = \frac{(90-75)^2 + (60-75)^2 + (75-75)^2}{3} = \frac{225 + 225 + 0}{3} = 150$$

**Step 3：std**
$$\sigma = \sqrt{150} \approx 12.247$$

**Step 4：套公式 `(x - 75) / 12.247`**

| 原值 | 计算 | 结果 |
|---|---|---|
| 90 | (90-75)/12.247 = 15/12.247 | **≈ 1.2247** |
| 60 | (60-75)/12.247 = -15/12.247 | **≈ -1.2247** |
| 75 | 0/12.247 | **0** |

输出：
```
transformer.mean_ -->  [75., 3., 12.667, 43.667]
transformer.var_  -->  [150., 0.667, 4.222, 6.889]
```
（其他列你可以照葫芦画瓢）

**关键陷阱**：
- sklearn 的 `var_` 用的是**总体方差**（分母 n）
- numpy 的 `np.var(x)` 默认也是 n
- 但 numpy 的 `np.std(x)` 默认 n，pandas 的 `.std()` 默认 n-1（样本标准差）
- 不同库默认值不一样，算细节时要核对

---

### Q7 `var_` 命名坑

**为啥不叫 `std_`**：sklearn 设计时存了**方差**而不是标准差。可能是历史原因（早期版本就这样），改名会破坏兼容性。

**工程影响**：
- 你想用标准差：`np.sqrt(transformer.var_)`
- 直接用 `transformer.var_` 当 std → 公式错，结果差一个开方，bug 难查

**真正用来 transform 的属性是 `transformer.scale_`**：
```python
transformer.scale_  # 等于 sqrt(transformer.var_)，是 sklearn 内部 transform 真正用的除数
```

为什么要单独存 `scale_`：处理边界情况（如方差为 0 的常数列），sklearn 会把 `scale_` 设为 1 避免除零，而 `var_` 仍保留原始 0。两者解耦。

**记忆法**：
- `mean_` 是均值
- `var_` 是方差（注意 var 不是 std）
- `scale_` 是 sklearn 内部用来除的值（≈ std）

---

### Q8 sanity check

标准化后每列：
- mean = 0
- std = 1（var = 1）

dm03 第 1 列结果 `[1.2247, -1.2247, 0]`：
- 求和：1.2247 + (-1.2247) + 0 = **0** ✅
- 平方和：1.5 + 1.5 + 0 = **3** = n（列数）✅

**用途**：写代码时如果 `fit_transform` 后某列均值不是 0，说明哪步搞错了（比如不小心按行做了，或者 dtype 是 int 触发整数除法）。

---

## D. fit / transform / fit_transform 的区别

### Q9 三个方法各自在做什么？

```python
transformer = MinMaxScaler()

transformer.fit(X_train)              # 只算 min/max，存到 transformer.data_min_ / data_max_
X_train_scaled = transformer.transform(X_train)  # 套公式

# 等价于：
X_train_scaled = transformer.fit_transform(X_train)  # 一步到位
```

| 方法 | 做什么 | 修改对象状态？ | 返回 |
|---|---|---|---|
| `fit(X)` | 学习统计量（min/max 或 mean/var） | ✅ | self（链式调用用） |
| `transform(X)` | 套公式变换 | ❌ | 变换后的数组 |
| `fit_transform(X)` | 上面两件事一起做 | ✅ | 变换后的数组 |

**为什么拆成两步**——因为有了状态分离，可以做这种事：

```python
transformer.fit(X_train)              # 训练集学统计量
transformer.transform(X_train)        # 变换训练集
transformer.transform(X_test)         # 测试集复用同一组统计量 ✅
transformer.transform(new_request)    # 在线推理也复用 ✅
```

如果 `fit` 和 `transform` 揉在一起，无法做"训练时学一次，推理时复用"。

---

### Q10 测试集只能 transform，不能 fit_transform

**正确写法**：**C**

```python
X_test_scaled = transformer.transform(X_test)   # ✅ 复用训练集学到的 min/max
```

**写法 A 错在哪**（`fit_transform(X_test)`）：
```python
X_test_scaled = transformer.fit_transform(X_test)  # ❌ data leakage
```
重新拟合 = 覆盖之前训练集学到的 min/max，用了**测试集自己的 min/max**。这有两个问题：
1. **数据泄漏 data leakage**：相当于"提前看了测试集分布"
2. **训练/测试尺度对不上**：模型在"训练集尺度"下学到的决策边界，套到"测试集尺度"上的样本就乱了

**写法 B 错在哪**（新建一个 transformer）：
```python
new_t = MinMaxScaler()
X_test_scaled = new_t.fit_transform(X_test)  # ❌ 量纲对不上
```
比 A 更明显——训练集和测试集用了**两套不同**的 min/max。同样的原始值 75，在训练集可能映射成 0.5，在测试集可能映射成 0.7，模型完全无法对齐。

**生产防坑**：用 sklearn Pipeline。

```python
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('scaler', MinMaxScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
])
pipe.fit(X_train, y_train)            # Pipeline 自动 fit_transform 训练集
pipe.predict(X_test)                  # 自动只 transform 测试集
```

Pipeline 强制了"训练集 fit，其他只 transform"的契约，**无法写错**。

---

## E. MinMax vs Standardize 怎么选

### Q11 选型对比

| | MinMaxScaler | StandardScaler |
|---|---|---|
| 公式 | `(x-min) / (max-min)` | `(x-mean) / std` |
| 输出范围 | `[0, 1]`（确定） | 不确定（理论无界，实际 ≈ ±3） |
| 假设 | 数据无极端异常值 | 数据近似正态分布 |
| 异常值敏感度 | **极敏感** | 较鲁棒 |
| 保留分布形状 | 是（线性压缩） | 是（线性平移+缩放） |

**异常值场景对比**：

假设第 1 列原值变成 `[90, 60, 75, 9999]`（最后一个是脏数据）：

- **MinMax**：min=60, max=9999, 跨度 9939。正常值 90 变 `(90-60)/9939 ≈ 0.003`，60 变 0，75 变 `0.0015`——**正常值全部被挤到接近 0**，区分度消失
- **Standardize**：mean ≈ 2556，std ≈ 4288。正常值 90 变 `(90-2556)/4288 ≈ -0.575`，9999 变 `1.74`——异常值依然显著，但正常值之间还**保有区分度**

**选型清单**：

| 场景 | 选 | 理由 |
|---|---|---|
| 神经网络输入 | MinMax `[0,1]` 或 `[-1,1]` | 激活函数喜欢有界输入 |
| 图像像素 | MinMax `[0,255]→[0,1]` | 范围天然有界 |
| KNN / SVM（无明显异常值） | Standardize | 默认首选，对异常值有一定鲁棒 |
| KNN / SVM（数据有异常值） | RobustScaler（用中位数/IQR） | 异常值最鲁棒 |
| 分布近似正态 | Standardize | 公式假设匹配 |
| 不知道选啥 | Standardize | KNN 课程中默认推荐 |

---

## 一句话总结

> **特征缩放是 KNN 的入场券**。MinMax 把特征压到固定范围，Standardize 把特征对齐到 N(0,1) 分布，但**两者都必须只在训练集上 fit**——这是 sklearn 设计 fit/transform 分离的根本原因。
