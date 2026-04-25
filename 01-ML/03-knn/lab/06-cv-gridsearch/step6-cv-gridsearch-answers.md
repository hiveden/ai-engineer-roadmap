# 答案：交叉验证 + 网格搜索 + Pipeline

> 配套题目：[`step6-cv-gridsearch.md`](./step6-cv-gridsearch.md)
> 对应代码：朴素版 [`step6_cv_gridsearch_naive.py`](./step6_cv_gridsearch_naive.py) / 正确版 [`step6_cv_gridsearch_pipeline.py`](./step6_cv_gridsearch_pipeline.py)

---

## A. 交叉验证

### Q1 为啥要交叉验证

**单次 train/test split 的问题**：

```python
# 同一份数据，不同 random_state
for seed in [10, 22, 42, 100, 7, 99]:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=seed)
    model.fit(X_tr, y_tr)
    print(model.score(X_te, y_te))
# 输出：0.85, 0.92, 0.88, 0.95, 0.83, 0.90 ← 跳来跳去
```

**单次评估方差大的根本原因**：测试集太小（30 个样本）+ 切分纯靠运气。某次切分如果"困难样本"都进了测试集，分数就低；都进了训练集，分数就高。

**CV 的本质**：每个样本**轮流当 val**，最终取均值——把"运气波动"平均掉。

**类比**：
- 单次 split = "一次考试定生死"
- CV = "连续考 5 次取平均，更靠谱"

---

### Q2 K-fold 流程

**4 折交叉验证图示**：

```
原始训练集（120 样本）→ 切成 4 等份 [A, B, C, D]，每份 30

第 1 折：train = [B, C, D] (90)  val = [A] (30)  → score_1 = 0.93
第 2 折：train = [A, C, D] (90)  val = [B] (30)  → score_2 = 0.91
第 3 折：train = [A, B, D] (90)  val = [C] (30)  → score_3 = 0.95
第 4 折:  train = [A, B, C] (90)  val = [D] (30)  → score_4 = 0.92

CV 评分 = mean([0.93, 0.91, 0.95, 0.92]) = 0.9275
CV 标准差 = std([...]) = 0.015
```

**每个样本在过程中**：
- 当 val：恰好 1 次
- 当 train：k-1 次（这里是 3 次）
- 这就是为啥 CV 比单次切分稳——所有样本都对评估贡献了一次

**4 vs 5 vs 10 折的稳健度**：
- 折数越多 → 每次 train 用的数据越多 → 模型越接近"用全部数据训练"的真实能力
- 但折数越多 → 总训练次数越多（开销 ∝ K）
- **业界默认 5 / 10 是经验甜蜜点**：方差够低 + 开销可接受

---

### Q3 K 怎么选

| K | 适用 | 缺点 |
|---|---|---|
| **2** | 极简调试 | 每折只用一半数据训练，模型偏弱 |
| **5** | **业界默认**，最常用 | 平衡的甜蜜点 |
| **10** | 数据中等（< 10000） | 比 5 折慢一倍 |
| **N（LOO）** | 数据极少（< 50） | 开销爆炸 O(N) |
| **n_splits=N** | 同 LOO | 几乎只在统计学论文用 |

**业务锚**：K = 你愿意付 K 倍训练时间换更稳的评估。

**特殊情况**：
- 不平衡数据 → 用 `StratifiedKFold`（按类别比例分层切折）
- 时间序列 → 用 `TimeSeriesSplit`（永远过去训练未来验证）
- 分组数据（同一用户多条记录）→ 用 `GroupKFold`（同组样本不能分到不同折）

---

## B. GridSearchCV API

### Q4 GridSearchCV 在做什么

**核心动作**：穷举搜索 + 交叉验证。

```python
GridSearchCV(model, {'n_neighbors': [1, 3, 5, 7, 9, 11]}, cv=4)
```

**展开成伪代码**：

```python
results = {}
for k in [1, 3, 5, 7, 9, 11]:        # 6 个超参组合
    fold_scores = []
    for fold in range(4):              # 4 折 CV
        train_fold, val_fold = split_kfold(X_train, y_train, fold)
        m = KNeighborsClassifier(n_neighbors=k)
        m.fit(train_fold)
        fold_scores.append(m.score(val_fold))
    results[k] = mean(fold_scores)

best_k = argmax(results)               # 选 mean 最高的
```

**总训练次数**：6 (params) × 4 (folds) = **24 次** fit + score。

**param_grid 格式**：dict，key = 参数名，value = 候选列表。

```python
{'n_neighbors': [3, 5, 7]}                    # 单超参
{'n_neighbors': [3, 5], 'weights': ['uniform', 'distance']}   # 多超参，笛卡尔积
[{'n_neighbors': [3,5]}, {'metric': ['cosine']}]              # 列表的字典 = 多套独立搜索空间
```

---

### Q5 best_* 属性

`fit` 完成后能查这些：

```python
estimator.best_score_      # 最优组合的 CV 平均分
# 例：0.9333

estimator.best_params_     # 最优超参字典
# 例：{'n_neighbors': 7}（朴素版）或 {'knn2__n_neighbors': 7}（Pipeline 版）

estimator.best_estimator_  # 已 refit 在全部 x_train 上的最终模型
# 直接可以 .predict(x_test)

estimator.cv_results_      # 详细每个组合 × 每折的所有结果
# dict 格式，转 DataFrame 看更清晰
```

`cv_results_` 里能看到的列：

| 列 | 含义 |
|---|---|
| `params` | 每个超参组合 |
| `mean_test_score` | 该组合的 CV 平均分 |
| `std_test_score` | CV 标准差（看模型对训练集的稳健性） |
| `rank_test_score` | 排名 |
| `split0_test_score` ~ `split3_test_score` | 每折单独分数 |
| `mean_fit_time` | 平均训练时间 |
| `mean_score_time` | 平均评估时间 |

---

### Q6 自动 refit

**默认 `refit=True`**：选出 `best_params_` 后，**用全部 x_train**（不再切折）重训一次最终模型。

**为什么要 refit**：
- CV 时每折只用了 (k-1)/k 数据训练
- 最终上线模型应该用**全部可用数据**训练，性能更好
- refit 保证 `best_estimator_` 是"用全部 x_train 训练"的版本

**所以可以直接**：

```python
estimator.predict(x_test)
# 等价于：
estimator.best_estimator_.predict(x_test)
```

**关掉 refit 的场景**（`refit=False`）：
- 只想知道最优参数，不想花时间训练（极少见）
- 多目标搜索（同时看 multiple metrics），需要指定 refit 哪个指标

---

### Q7 多超参组合爆炸

```python
{'n_neighbors': [3,5,7],         # 3 个
 'weights': ['uniform','distance'],  # 2 个
 'p': [1, 2]}                     # 2 个（曼哈顿/欧氏）
# 笛卡尔积：3 × 2 × 2 = 12 个组合
# CV K=4 → 48 次训练
```

**组合爆炸的解药**：

| 方法 | API | 适用 |
|---|---|---|
| 随机采样 | `RandomizedSearchCV(model, params, n_iter=20)` | 超参连续型多 / 候选多 |
| 减半搜索 | `HalvingGridSearchCV` | 先小子集筛掉差的，再大子集精调（sklearn 0.24+ 实验性） |
| 贝叶斯优化 | `optuna` / `hyperopt` / `scikit-optimize` | 超参高维 / 评估代价大 |
| 早停 | 树模型 / NN 用 early stopping | 训练时间长 |

**实战经验**：先 `RandomizedSearchCV` 粗筛 50 个随机点，找到大致范围；再 `GridSearchCV` 在范围内精调。

---

## C. Pipeline + GridSearch（核心金矿）

### Q8 朴素版 leakage 在哪

**朴素版流程**：

```
┌─ x_train (120 样本) ────────────────────────────────────┐
│                                                         │
│  Step 3: scaler.fit_transform(x_train)                 │
│           ↑                                             │
│  scaler 学到的 mean/std 来自所有 120 样本（含未来的 val）│
│                                                         │
│  Step 4: GridSearchCV(cv=4) 在 x_train 上切 4 折        │
│  ┌──────┬──────┬──────┬──────┐                          │
│  │ 30 A │ 30 B │ 30 C │ 30 D │                          │
│  └──────┴──────┴──────┴──────┘                          │
│  第 1 折 val = A，但 scaler 已经看过 A 的统计量了！     │
│  → A 的标准化"完美贴合"，模型评估虚高                   │
└─────────────────────────────────────────────────────────┘
```

**关键洞察**：scaler 的 mean/std 是基于"所有 120 样本"算的——这本身**包含了**之后会被切到 val 折的那 30 个样本的信息。模型在 val 折上看到的特征已经是"用全集统计量标准化的"，等于变相偷看了答案。

**类比**：考试前老师把整套试卷的答案分布告诉你"这次考试 80% 选 B"——你知道这个分布后猜 B 的命中率自然高，但实际能力没增加。

---

### Q9 Pipeline 怎么救

**正确版流程**：

```
┌─ x_train (120 样本) ─────────────────────────────────────┐
│                                                          │
│  ❌ 不在外面 fit_transform                              │
│                                                          │
│  GridSearchCV(pipe, cv=4) 切 4 折                        │
│  ┌──────┬──────┬──────┬──────┐                           │
│  │ 30 A │ 30 B │ 30 C │ 30 D │                           │
│  └──────┴──────┴──────┴──────┘                           │
│                                                          │
│  第 1 折：                                               │
│    train_fold = B+C+D (90)                              │
│    val_fold   = A     (30)                              │
│                                                          │
│    pipe.fit(train_fold):                                 │
│      └─ scaler.fit_transform(train_fold)  ← 只看 90      │
│      └─ knn.fit(scaled_train)                            │
│                                                          │
│    pipe.predict(val_fold):                               │
│      └─ scaler.transform(val_fold)        ← A 只 transform│
│      └─ knn.predict(scaled_val)                          │
│                                                          │
│  ✅ scaler 从未看过 val 折的统计量                       │
└──────────────────────────────────────────────────────────┘
```

**核心机制**：Pipeline 把"标准化 + 模型"打包成一个整体 estimator，GridSearchCV 在每折内部独立调 `pipe.fit(train_fold)`——scaler 也只在 train_fold 上 fit。

**契约由代码保证**：写错 leakage 在 Pipeline 这种姿势下**几乎不可能**——除非你手动绕开 Pipeline，不然每折必然独立 fit。

---

### Q10 4% 差距的来源

| 版本 | best_score_ | 含义 |
|---|---|---|
| 朴素版 | 0.971 | scaler 偷看了 val 统计量后的虚高分数 |
| 正确版 | 0.933 | scaler 老老实实只在 train 折 fit 的真实分数 |

**这 4% 不是"模型变好了"**，是"评估方法被污染让分数虚高"。

**上线后会怎样**：
- 真实新数据 → scaler（基于全部 x_train fit 的）transform → 模型预测
- 这个流程**接近**朴素版的"模型已经看过统计量"——但**新数据本身**是 scaler 真正没见过的
- 所以上线表现介于朴素版（0.971）和正确版（0.933）之间，**更接近 0.933**

**反过来理解**：正确版的 0.933 是**对未来表现的可靠估计**；朴素版的 0.971 是"自欺欺人"的乐观偏差。

**史观**：data leakage 是**机器学习项目最容易翻车的姿势之一**。Kaggle 比赛公开数据里的 leakage 有时候让冠军方案在 private leaderboard 暴跌——就是这种乐观偏差被戳穿。

---

### Q11 Pipeline 语法

```python
pipe = Pipeline([
    ('std', StandardScaler()),
    ('knn2', KNeighborsClassifier()),
])
```

**`'std'` / `'knn2'` 是 step 名**：自定义字符串，唯一即可。常见命名：`'scaler'`, `'pca'`, `'clf'`, `'reg'`。

**双下划线参数路径**：

```python
{'knn2__n_neighbors': [1, 3, 5, 7]}
#    ^^                ^^^^^^^^^^^^^
#  step 名            该 step 的参数名
```

**为啥是双下划线**：sklearn 全局约定，避免和参数名本身的下划线混淆（`n_neighbors` 自己就有单下划线）。

**单下划线写法不识别**：

```python
{'knn2_n_neighbors': [3, 5]}    # ❌ sklearn 不识别
# ValueError: Invalid parameter knn2_n_neighbors for estimator Pipeline
```

**多层嵌套**：

```python
pipe = Pipeline([
    ('preproc', Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
    ])),
    ('clf', KNeighborsClassifier()),
])

param_grid = {
    'preproc__scaler__with_mean': [True, False],   # 三层路径
    'preproc__pca__n_components': [2, 3, 4],
    'clf__n_neighbors': [3, 5, 7],
}
```

每层用 `__` 隔开。

---

### Q12 Pipeline 还能用在哪

**1. 多 transformer 串联**：

```python
Pipeline([
    ('encode', OneHotEncoder()),       # 类别特征 → 数值
    ('scale', StandardScaler()),       # 标准化
    ('reduce', PCA(n_components=10)),  # 降维
    ('clf', LogisticRegression()),
])
```

**2. 持久化整个流程**：

```python
joblib.dump(pipe, 'pipeline.pkl')  # 一个文件 = scaler + model
# 推理时：
pipe = joblib.load('pipeline.pkl')
pipe.predict(raw_new_data)         # 自动走完所有变换
```

**3. 上线推理服务**：

```python
@app.post('/predict')
def predict(request):
    raw = parse(request)
    return pipe.predict(raw)       # 一行搞定，scaler 状态自动管
```

不用 Pipeline 时要手动管 scaler / encoder / pca 多个对象，**忘 transform 一个就翻车**。Pipeline 把这个错误防御内置了。

**4. 复杂场景：ColumnTransformer**：

```python
from sklearn.compose import ColumnTransformer

preproc = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'income']),       # 数值列标准化
    ('cat', OneHotEncoder(), ['city', 'gender']),       # 类别列 one-hot
])
pipe = Pipeline([('pre', preproc), ('clf', KNN())])
```

不同列走不同变换，依然单一 Pipeline。生产标配。

---

## D. 工程坑 / 进阶

### Q13 双重 leakage 警告

朴素版（原始课程代码，本节简化版已删）有这种段：

```python
# 已经做过的：
x_train = transfer.fit_transform(x_train)  # leakage 1
estimator = GridSearchCV(model, ..., cv=4).fit(x_train, y_train)
best_k = estimator.best_params_['n_neighbors']

# 又来一遍：
x_train_ = transfer.fit_transform(x_train_)   # leakage 2，重新 fit_transform 又一次
final = KNeighborsClassifier(n_neighbors=best_k).fit(x_train_, y_train)
```

**问题**：手动重训那段又 `fit_transform` 了一次，scaler 状态被覆盖。本质和 leakage 1 一样，但多了一层。

**Pipeline 版彻底避免**：`estimator.predict(x_test)` 一行解决，不需要手动重训。

---

### Q14 cv_results_ 怎么用

```python
import pandas as pd
df = pd.DataFrame(estimator.cv_results_)
print(df[['param_knn2__n_neighbors', 'mean_test_score', 'std_test_score', 'rank_test_score']])
```

**画 score vs k 曲线**：

```python
import matplotlib.pyplot as plt
plt.errorbar(df['param_knn2__n_neighbors'],
             df['mean_test_score'],
             yerr=df['std_test_score'],
             fmt='-o')
plt.xlabel('k')
plt.ylabel('CV accuracy')
plt.title('KNN performance vs k')
plt.show()
```

**肉眼看拐点**——如果 k=3 到 k=7 分数稳定上升，k=9 突然掉，说明 k=7 是甜蜜点。

---

### Q15 GridSearchCV 提速

```python
GridSearchCV(model, params, cv=4, n_jobs=-1)  # -1 = 用所有 CPU 核
```

**并行原理**：每个 (params, fold) 组合独立训练，互不依赖 → 完美并行。M4 Pro 12 核 → 理论上 12 倍加速。

**其他提速手段**：

| 方法 | 提速倍数 | 代价 |
|---|---|---|
| `n_jobs=-1` | 跟核数线性 | 内存占用增加 |
| `RandomizedSearchCV(n_iter=20)` | 跟 n_iter/总组合 成比例 | 可能错过最优 |
| `HalvingGridSearchCV` | 通常 3-10 倍 | 需要数据集足够大 |
| `cv=3` 替代 `cv=10` | 3.3 倍 | 评估方差稍大 |
| 提前用小数据集筛 | 看情况 | 手动两阶段 |

**KNN 特别慢的原因**：predict 阶段 O(n_train · n_test · d)，每折 CV 都重跑。`n_jobs=-1` 几乎必加。

---

### Q16 测试集隔离

**铁律**：测试集**全程不参与 GridSearchCV**。

```python
# ✅ 正确流程
X_tr, X_te, y_tr, y_te = train_test_split(...)

# CV 只在 X_tr 上做
gs = GridSearchCV(pipe, params, cv=5).fit(X_tr, y_tr)

# 测试集只在最后用一次
final_score = gs.score(X_te, y_te)
print('上线预期表现:', final_score)
```

**反复用测试集调参的危害**：

```python
# ❌ 间接 leakage
for k in [3, 5, 7, 9]:
    model = KNN(n_neighbors=k).fit(X_tr, y_tr)
    print(model.score(X_te, y_te))   # 反复看测试集分数
# 你"挑"了那个测试集分数最高的 k —— 测试集已经间接影响了模型选择
# 上线后真实表现会比测试分数差
```

**为什么不能信"测试集多看几次"**：你的眼睛和大脑就是模型的一部分。看到分数后做决策（"换 k", "改特征"），**间接训练**了你 → 你训练的模型 → 测试集污染。

**正确姿势**：测试集**只看一次**，记下分数，写报告。要再调，回到 train + CV 流程，不准动测试集。这就是为啥要 train / val / test 三段（CV 替代了 val 角色）。

---

## 一句话总结

> **CV + GridSearch + Pipeline 是 sklearn 监督学习"严肃做"的标配三件套**。
>
> - **CV**：消除单次切分的运气方差
> - **GridSearch**：穷举超参找最优
> - **Pipeline**：把预处理和模型打包，**强制保证 fit 只在 train 折发生**——这是 data leakage 的最后一道防线
>
> 朴素版（在 GridSearchCV 外面 fit_transform）看起来对，跑出来分数还高，但**那 4% 是偷的**。上线就还。
