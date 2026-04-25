# 答案：鸢尾花 KNN 端到端流程

> 配套题目：[`step5-iris-pipeline.md`](./step5-iris-pipeline.md)
> 对应代码：[`step5_iris_pipeline.py`](./step5_iris_pipeline.py)

---

## A. 6 步流程的"为什么"

### Q1 标准 6 步流程

```
1. 加载数据    load_iris() / pd.read_csv() / 自建数据库连接
2. 数据切分    train_test_split (含 stratify)
3. 特征工程    StandardScaler / MinMaxScaler / OneHotEncoder ...
4. 模型训练    KNeighborsClassifier().fit(...)
5. 模型评估    accuracy_score / model.score / classification_report
6. 模型推理    transfer.transform(new_data) → model.predict(...)
```

**哪步可省**：
- step 3（特征工程）：树模型可以跳过（不依赖距离/梯度），KNN/SVM/NN 不能省
- step 5（评估）：上线前必须做，POC 可省
- 其他都不能省

**业务锚**：
- 1 = 取数据（SELECT）
- 2 = 拆样本（数据集合分组）
- 3 = ETL（清洗、归一）
- 4 = 训练（学）
- 5 = QA（验收）
- 6 = 上线服务（推理）

---

### Q2 为什么切分必须在特征工程之前？

**反着做的反例**：

```python
# ❌ 错误顺序
x_scaled = StandardScaler().fit_transform(X)  # 用了全部数据算 mean/std
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, ...)
```

**问题**：scaler 的 `fit` 阶段算 mean/std 时**看到了测试集**——测试集的统计量泄漏给了训练流程。这叫 **data leakage 数据泄漏**。

**后果**：
- 模型在"含 leakage 的训练集"上学到的边界，套到测试集上**异常准确**（因为测试集已经被偷看过）
- 上线后真实新数据进来 → 性能崩盘
- 评估指标过于乐观，欺骗自己也欺骗 review 你的人

**正确顺序的逻辑**：
- 测试集 = "未来的、未知的数据"模拟
- 训练阶段所有动作（fit / fit_transform）**绝不能**碰测试集
- 测试集只在最后接受 `transform` + `predict`

---

### Q3 测试集为啥只 transform

**答**：`fit_transform` = `fit`（学统计量）+ `transform`（套公式），训练集执行后 scaler 已经"装好弹药"。

```python
x_train = transfer.fit_transform(x_train)   # 学到 mean/std，存在 transfer.mean_ / scale_
x_test = transfer.transform(x_test)         # 复用同一组 mean/std
```

**如果测试集也 fit_transform**：

```python
# ❌
x_test = transfer.fit_transform(x_test)
```

后果：
1. `transfer.mean_` 被覆盖为测试集的 mean → 训练集 scaler 状态丢失
2. 测试集用了**自己的** mean/std，跟训练集对不上 → 同样的原始值在两套尺度下被映射到不同位置
3. 模型学到的决策边界（基于训练集尺度）作用到测试集（不同尺度）→ 预测结果不可靠

**口诀**：训练集 fit_transform，**其他全都只 transform**（测试集、验证集、新数据、线上请求）。

---

## B. 模型评估两种打分

### Q4 两种 API 等价吗？

**功能完全等价**，差在调用形式：

```python
# 函数式
y_pred = model.predict(x_test)
score1 = accuracy_score(y_test, y_pred)

# 方法式
score2 = model.score(x_test, y_test)
# 内部：return accuracy_score(y_test, self.predict(x_test))
```

**输入差异**：

| | 输入 |
|---|---|
| `accuracy_score(y_true, y_pred)` | 已经预测好的标签 |
| `model.score(X_test, y_test)` | 原始测试特征 + 真实标签（内部自己 predict） |

**为什么有两种**：
- `model.score` 是 sklearn estimator 接口的统一约定（所有 estimator 都有），方便统一调用
- `accuracy_score` 是 metrics 模块独立函数，可以拿任意 (y_true, y_pred) 算，不依赖 estimator

**什么时候用哪个**：
- 已有 y_pred（比如想算多个指标）→ 用 `accuracy_score` 等函数
- 一行评估 → `model.score`
- GridSearchCV / cross_val_score → 内部调 `model.score`

---

### Q5 `accuracy_score` 参数顺序

**正确顺序**：`accuracy_score(y_true, y_pred)`

**记忆法**：sklearn metrics 全家桶**永远 `y_true` 在前**，`y_pred` 在后。

**accuracy 对称所以倒过来无害**：
```python
accuracy_score(y_true, y_pred) == accuracy_score(y_pred, y_true)  # True
```

**但其他指标顺序错就翻车**：

```python
# precision / recall / confusion_matrix 不对称
precision_score(y_true, y_pred)  # ✅ 正确
precision_score(y_pred, y_true)  # ❌ 数学上是错的，结果意义完全不同
```

`precision_score` 衡量"模型说是正的，里面真有多少是正"。倒过来变成"真实是正的，里面有多少被模型说是正"——这其实是 recall 的定义。**接口允许你写错，结果也给你算出来一个数，但意义南辕北辙**。

**坑爹度排名**（按可见度从低到高）：
1. accuracy：错了没差，最坑（错而不知）
2. precision/recall：数值变化大，容易被人 review 出来
3. confusion_matrix：矩阵转置了，肉眼能看出

**铁律**：永远 `y_true` 在前，写完 review 一遍。

---

### Q6 默认指标

| 任务 | `model.score` 默认 |
|---|---|
| 分类（KNeighborsClassifier / LogisticRegression / ...） | accuracy |
| 回归（KNeighborsRegressor / LinearRegression / ...） | **R² 决定系数**，不是 MSE |

**R² 的取值**：
- 1.0 = 完美预测
- 0.0 = 跟"永远预测均值"一样烂
- 负数 = 比"永远预测均值"还烂（模型有问题）

**换其他指标**：

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
)

# 一份完整报告
print(classification_report(y_test, y_pred))
# 输出 precision / recall / f1 / support 一应俱全

# 混淆矩阵
print(confusion_matrix(y_test, y_pred))
```

不平衡分类用 accuracy 会被欺骗（参考 `04-split-stratify` Q5），实战首选 `f1` / `precision-recall AUC`。

---

## C. 推理阶段细节

### Q7 新数据为啥也要 transform

**答**：模型在"标准化后的尺度"上学到的决策边界，新数据必须**用同一把尺**才能比较。

```python
# 训练集尺度：StandardScaler 让 mean=0, std=1
# 模型学到的决策边界 ≈ "petal_length > 0.3 时偏向类 2"（这里的 0.3 是标准化后的值）

# 新数据原始值 [3, 5, 4, 2]
x_new = [[3, 5, 4, 2]]

# ❌ 不 transform 直接预测
model.predict(x_new)
# 把原始值 4（cm）当标准化后的尺度去比，但模型边界是 0.3
# 4 >> 0.3，被强行判到某一类，预测完全错乱
```

**契约保证**：训练时怎么处理特征（fit_transform），推理时**用同一个 transformer 对象** transform 新数据。要保证这个契约，工程上有两种姿势：

1. **手动**：把 `transfer` 对象 joblib.dump 持久化，推理时 load 出来
2. **自动**：用 `Pipeline` 把 scaler 和 model 串起来，`pipe.predict(new_data)` 内部自动按训练时的步骤处理（见 `06-cv-gridsearch`）

第二种是**生产推荐**，第一种容易写错（特别是多个 transformer 时）。

---

### Q8 predict_proba 在 KNN 的含义

**输出**：shape `(n_samples, n_classes)` = `(1, 3)` for iris。

```python
y_new_proba = model.predict_proba(x_new)
# 例：array([[0.0, 0.667, 0.333]])
```

**每行加起来 = 1**（概率归一化）。

**KNN 的"概率"是怎么算的**：

k=3 时，找到 3 个最近邻，看它们的标签分布：

| 邻居 1 标签 | 邻居 2 标签 | 邻居 3 标签 | proba |
|---|---|---|---|
| 0 | 0 | 0 | [1.0, 0.0, 0.0] |
| 0 | 0 | 1 | [0.667, 0.333, 0.0] |
| 0 | 1 | 2 | [0.333, 0.333, 0.333] |
| 1 | 1 | 2 | [0.0, 0.667, 0.333] |
| 2 | 2 | 2 | [0.0, 0.0, 1.0] |

**k=3 时所有可能概率值**：`{0/3, 1/3, 2/3, 3/3}` = `{0.0, 0.333, 0.667, 1.0}`。

**这不是真概率**——只是邻居中各类的占比。逻辑回归的 `predict_proba` 是真概率（基于 sigmoid 输出）。KNN 的"概率"颗粒度由 k 决定（k 越大概率越细腻）。

**`weights='distance'` 时**：按 1/距离 加权，颗粒度变细，可以是任意 0~1 之间的值。

---

### Q9 predict 决策规则

`predict_proba = [[0.0, 0.667, 0.333]]` → `predict` 返回 `[1]`（取最大概率的类索引）。

**实现**：

```python
y_pred = np.argmax(y_proba, axis=1)
# proba [[0.0, 0.667, 0.333]] → argmax = 1
```

**两类并列最大**（如 `[0.5, 0.5, 0]`，k=2 时常见）：
- `argmax` 返回**第一个**最大值的索引（即类别索引最小的）
- 这就是为啥 KNN 分类 k 用偶数会偏向小索引类（参考 `00-basicapi` Q9）

**predict_proba + 自定义阈值的场景**：

```python
# 风控：宁可错杀不可放过
y_proba = model.predict_proba(x_new)
y_pred = (y_proba[:, 1] > 0.3).astype(int)  # 阈值从 0.5 调到 0.3
# 让"判正"更容易触发
```

**适用场景**：
- 不平衡分类（疾病诊断、风控、反欺诈）
- 业务对 precision/recall 有偏好（医疗：高 recall 漏诊代价大；广告：高 precision 误投代价大）
- 多模型集成（ensemble）：用概率平均/投票而非硬标签

KNN 的概率粒度粗，做阈值调节不如逻辑回归丝滑——但思路通用。

---

## D. 工程坑

### Q10 transformer 持久化

**StandardScaler fit 之后会保存**：

```python
transfer = StandardScaler()
transfer.fit_transform(x_train)

# fit 之后这些属性才有值：
transfer.mean_       # 每列均值
transfer.var_        # 每列方差
transfer.scale_      # 每列标准差（= sqrt(var_)）
transfer.n_features_in_  # 特征数
```

这就是 **stateful transformer** 设计——对象既是变换器又是状态容器。

**生产持久化**：

```python
import joblib
joblib.dump(transfer, 'scaler.pkl')
joblib.dump(model, 'model.pkl')

# 推理服务启动时加载
transfer = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')
new_pred = model.predict(transfer.transform(new_data))
```

**Pipeline 更优雅**（见 `06-cv-gridsearch`）：

```python
joblib.dump(pipe, 'pipeline.pkl')   # 一个文件搞定 scaler + model
```

---

### Q11 换数据集要改哪几行？

**最小改动**：只改 step 1。

```python
# step 1：换数据集
mydataset = load_breast_cancer()   # 或 load_wine()、load_digits() 等
# 其他 5 步代码完全不变

# 唯一可能要调的：step 4 的 k 值
# 不同数据集最优 k 不同，需重新 GridSearchCV
```

**这就是 sklearn API 一致性的威力**：

| 算法 | API 模板 |
|---|---|
| KNeighborsClassifier | `.fit(X, y).predict(X) / .score(X, y) / .predict_proba(X)` |
| LogisticRegression | `.fit(X, y).predict(X) / .score(X, y) / .predict_proba(X)` |
| RandomForestClassifier | `.fit(X, y).predict(X) / .score(X, y) / .predict_proba(X)` |
| SVC | `.fit(X, y).predict(X) / .score(X, y) / .predict_proba(X)`（要 probability=True） |

**所有 estimator 都有 fit / predict / score**——这是 sklearn 的"鸭子类型"接口契约。背一个模板，整个生态都能用。

**为什么能做到**：
- BaseEstimator 抽象基类定义接口
- 每个具体算法（KNN / LR / RF）继承并实现核心逻辑
- 所有"组件"（estimator / transformer / pipeline / cross-val）按这个契约互操作

工程师视角：sklearn = 教科书级的"接口和实现分离"+"组合优于继承"。

---

## 一句话总结

> **sklearn 监督学习 6 步流程是模板，换数据集只改 step 1，换算法只改 step 4**。
> 关键纪律：**切分必须在 scaler 之前**，**测试集和新数据只能 transform**，**新数据推理记得也要 transform**。
> KNN 的 `predict_proba` 不是真概率，是 k 邻居的类别占比——但能做阈值调节就够了。
