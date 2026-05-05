---
tags: [API/XGBoost, 工程/sklearn兼容, 超参数/调参, 工程/早停]
---

# XGBClassifier API

> 维度：**代码**
> 知识点级别：【了解】XGBClassifier API
> 章节底稿全文见 [`README.md`](./README.md)（PPT Slide 82 + 笔记 API 段）

## ━━━━━━━━ 底稿 ━━━━━━━━

### PPT

> Slide 82 · XGBoost 算法 API

**XGB 的安装和使用**：

在 sklearn 机器学习库中没有集成 xgb。想要使用 xgb，需要手工安装：

```bash
pip3 install xgboost
# 可以在 xgb 的官网上查看最新版本：https://xgboost.readthedocs.io/en/latest/
```

**XGB 的编码风格**：
- 支持非 sklearn 方式，也即是自己的风格
- 支持 sklearn 方式，调用方式保持 sklearn 的形式

> Slide 88 · 本章小结（API 部分）

**xgboost API**

```python
XGBClassifier(n_estimators, max_depth, learning_rate, objective)
```

### 笔记

> 【了解】XGBoost API

**安装**：sklearn 不内置，需 `pip install xgboost`。提供两套 API：原生风格（`xgb.train` + `DMatrix`）与 sklearn 风格（`XGBClassifier`），下面用后者。

```python
from xgboost import XGBClassifier

bst = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.3,
    objective="binary:logistic",  # multi:softmax / reg:squarederror
    reg_lambda=1,    # L2
    gamma=0,         # 分裂阈值
    # eta=0.1,            # learning_rate 的原生别名
    # eval_metric="merror",  # 多分类错误率，二分类常用 "logloss" / "auc"
)
bst.fit(X_train, y_train)
```

---

## ━━━━━━━━ 讲解 ━━━━━━━━

### 安装：与 sklearn 解耦

XGBoost 是独立开源库，**不打包在 sklearn 里**（GBDT 打包在 sklearn，XGBoost 独立发布）：

```bash
pip install xgboost
```

安装后有两套 API，选哪个：

```
两套 API
──────────────────────────────────────────────────────────
原生 API（xgb.train + DMatrix）
  优点：性能更好，原生支持早停、watch list
  缺点：不能直接插 sklearn Pipeline / GridSearchCV
  适合：生产训练、大数据量

sklearn 兼容 API（XGBClassifier / XGBRegressor）
  优点：fit/predict/score 接口统一，直接进 Pipeline
  缺点：参数名有些和原生不同（learning_rate vs eta）
  适合：快速实验、和 sklearn 生态配合
```

本讲解用 sklearn 兼容 API。

### XGBClassifier 关键参数

```python
from xgboost import XGBClassifier

estimator = XGBClassifier(
    # ── 树的数量 ──────────────────────────────────────
    n_estimators=100,         # 建多少棵树（越多越慢，early_stopping 可自动停）

    # ── 树的结构 ──────────────────────────────────────
    max_depth=6,              # 每棵树最大深度（防过拟合核心杠杆）

    # ── 学习节奏 ──────────────────────────────────────
    learning_rate=0.3,        # 每棵树的缩放系数（原生别名 eta）
                              # 越小 → 需要更多树，但泛化更好

    # ── 正则化（与目标函数对应）──────────────────────
    reg_lambda=1,             # λ：叶子值 L2 正则系数（对应 Obj 里的 λ‖w‖²）
    reg_alpha=0,              # α：叶子值 L1 正则（稀疏场景用）
    gamma=0,                  # γ：叶子数惩罚系数（Gain 里减去的那个 γ）
                              # gamma > 0 → 分裂必须有足够 Gain 才能发生

    # ── 随机采样（防过拟合 + 提速）──────────────────
    subsample=1.0,            # 每棵树随机取 x% 样本（行采样）
    colsample_bytree=1.0,     # 每棵树随机取 x% 特征（列采样）

    # ── 任务类型 ──────────────────────────────────────
    objective='binary:logistic',  # 二分类
    # objective='multi:softmax',  # 多分类（需要 num_class 参数）
    # objective='reg:squarederror', # 回归

    # ── 其他 ──────────────────────────────────────────
    eval_metric='logloss',    # 验证集评估指标
    random_state=42,
    n_jobs=-1,                # 并行线程数（-1 = 所有核）
)
```

参数速查表：

| 参数 | 对应理论 | 调参直觉 |
|---|---|---|
| `n_estimators` | 树的数量 | 配合 `early_stopping_rounds` 自动决定，不要死设 |
| `learning_rate` | 缩放步长 | 0.01-0.1（精调）/ 0.1-0.3（快速实验） |
| `max_depth` | 树深度 | 通常 3-8，过拟合时减小 |
| `gamma` | 分裂 Gain 阈值 $\gamma$ | 0 = 无惩罚；0.1-1 = 中等约束 |
| `reg_lambda` | L2 正则 $\lambda$ | 默认 1；过拟合时增大 |
| `reg_alpha` | L1 正则 $\alpha$ | 高维稀疏时用 |
| `subsample` | 行采样比例 | 0.6-0.9（降过拟合 + 提速） |
| `colsample_bytree` | 列采样比例 | 0.6-0.9（类似随机森林效果） |

### sklearn 兼容接口

fit / predict / score 和 sklearn 一致，可以直接插 Pipeline：

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

pipe = Pipeline([
    ('scaler', StandardScaler()),          # XGBoost 不要求缩放，但接上没坏处
    ('xgb', XGBClassifier(n_estimators=100, random_state=42)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

### 早停机制

`early_stopping_rounds` 是 XGBoost 原生功能，**比死设 `n_estimators` 聪明**：

```python
estimator = XGBClassifier(
    n_estimators=1000,           # 设大一点，让早停来决定
    learning_rate=0.05,
    early_stopping_rounds=50,    # 连续 50 轮验证集没有提升 → 停止
    eval_metric='logloss',
)

estimator.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],   # 必须传验证集
    verbose=100,                  # 每 100 轮打印一次
)

print(f"最优树数量: {estimator.best_iteration}")
```

```
早停逻辑：
  第 1 轮   val_logloss = 0.61
  第 10 轮  val_logloss = 0.45   ← 新最优
  ...
  第 60 轮  val_logloss = 0.42   ← 新最优
  第 110 轮 val_logloss = 0.43   ← 连续 50 轮无提升
  → 停止，返回第 60 轮的模型
```

**工程建议**：学习率调小（0.01-0.05）+ 大 `n_estimators` + 早停，是调 XGBoost 的标准姿势。

### DMatrix vs sklearn 接口选型

```
场景                            推荐接口
─────────────────────────────────────────────────────
快速实验 / GridSearchCV         XGBClassifier（sklearn 接口）
生产训练 / 超大数据集           xgb.train + DMatrix（原生接口）
早停（需要 eval_set）           两者都支持，但原生更灵活
保存模型（跨语言部署）          原生接口 → save_model → JSON/UBJ
                                （可在 Java / Go / C++ 加载）
```

`DMatrix` 是 XGBoost 原生数据格式，内存压缩更高效：

```python
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)

params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'eta': 0.1,         # learning_rate 的原生名
    'lambda': 1,        # reg_lambda 的原生名
    'gamma': 0,
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dval, 'val')],
    early_stopping_rounds=50,
    verbose_eval=100,
)
```

### GPU 加速

有 NVIDIA GPU + 安装了 CUDA 版 XGBoost 时：

```python
estimator = XGBClassifier(
    tree_method='gpu_hist',   # 用 GPU 构建直方图（最快）
    device='cuda',            # XGBoost ≥ 2.0 的写法
    n_estimators=500,
)
```

提速幅度：CPU 单核 vs GPU，数据量 > 10 万时通常 5-20x。没 GPU 时 `tree_method='hist'`（CPU 直方图，仍比默认快）。

### 一句话钉板

**XGBClassifier 是 sklearn 兼容的薄壳，核心参数三类：树结构（`max_depth`）、正则（`gamma / reg_lambda`）、采样（`subsample / colsample_bytree`）；加上早停（`early_stopping_rounds`），就是工业级调参的标准姿势。**

> Sources：
> - PPT Slide 82；Slide 88（小结 API 段）；笔记 API 段
> - XGBoost 官方文档 https://xgboost.readthedocs.io/（参数表 + DMatrix 用法）
