# LESSON-PLAN · 第 1 节 · 线性回归（demo-01 加州房价）

> **受众**：未来某次 session 开始时需要带 Alex 过线性回归 / demo-01 的 Claude。
>
> **这是备课资料库，不是教案。** 讲解方式（苏格拉底 / 直讲）由当时的 Claude 判断；这份文档确保你不会遇到"学生突然问了个合理问题，你没备到"。
>
> **10 节课里的第 1 节**。算法清单见 [`demos/README.md`](../README.md)。

---

## 0. 课前须知（防复读失败）

### 0.1 这节课的位置

| 维度 | 值 |
|---|---|
| 算法 | 线性回归（LinearRegression） |
| 任务类型 | 回归（输出连续数字） |
| 数据集 | 加州房价（20640, 8） |
| roadmap 阶段 | 阶段一·传统 ML 工程化·第 2 层·接口层 |
| 目标深度（来自 demos/README.md） | **L3** — 能讲清楚每步动机 + 能做选型判断，不要求手写代码 |
| 在 10 节课里 | 第 1 节（第一个 demo，骨架建立） |

### 0.2 当前学习状态（2026-04-19 快照）

- A 组 10 概念 + B 组 5 工程坑 Level 2 通关于 session-04
- demo-01 在 session-06 跑通，用户能用自己的话说 ML 4 步流程
- 过拟合 / 泛化 Level 2 于 session-07（原话："记住了答案没记住规律，不能泛化"）
- 置信度 Level 2 于 session-07（属 demo-02 范围）

**如果又回炉重讲已通关概念 = 浪费时间**。先 attune。

---

## 1. 讲义集成（迁移自原 `01-Machine-Learning-Foundation/03-Linear-Regression-Engineering.md`）

> **迁移策略**：原文档内容已全部迁入本节，原文件已删除（2026-04-19 重构）。原文里的数学推导（正规方程矩阵公式、梯度下降公式）**已过滤**——学生 L4 追问时备到延展层（§3.5）。

### 1.1 核心隐喻 · 模型就是"加权累加器"

**业务逻辑从硬编码到权重化**：

```
Software 1.0 写法：
double predictPrice(House h) {
    if (h.area > 100 && h.isSchoolDistrict) return 500.0;
    return 300.0;
}

Software 2.0 写法（线性回归）：
Price = w1 * Area + w2 * RoomCount + w3 * SchoolScore + b
```

术语映射：
- **权重 w**：每个特征的"话语权"（对结果的影响力）
- **偏置 b**：起步价 / 底噪
- **学习目标**：找出使预测总误差最小的那组 w 和 b

**demo-01 step3 的输出直接对应这件事**：`model.coef_` 是 8 个权重数组，`model.intercept_` 是那个 b。

### 1.2 求解策略（工程师视角的两条路）

> ⚠️ **只讲概念对比，不讲公式**。如果学生追问"怎么具体算出来"，下限答案："模型试出来的，具体怎么试是 sklearn 内部的事"。

| 策略 | 一句话定位 | 适用场景 | 工程类比 |
|---|---|---|---|
| 正规方程（Normal Equation） | 一步到位算精确解 | 特征少、数据 < 10 万 | SQL 全量 join + 矩阵求逆 |
| 梯度下降（Gradient Descent） | 迭代往最优解走 | 数据大、分布式训练 | 盲人下山：环顾四周找最陡方向小步踩 |

**demo-01 用的 `LinearRegression` 走的是正规方程路径**（OLS，最小二乘）。工业里大数据场景会用 `SGDRegressor`（随机梯度下降）。

### 1.3 评估指标 · 别被平均值骗了

| 指标 | 公式文字版 | 特点 | 什么时候用 |
|---|---|---|---|
| MAE | 每条 abs(预测 - 真实) 取平均 | 单位和 y 一致，最符合人类直觉 | 向业务方汇报（老板一听就懂） |
| MSE | 每条 (预测 - 真实)² 取平均 | 严惩离群点（大误差平方会爆炸） | 模型训练时内部优化用 |
| RMSE | √MSE | MSE 单位修复，但仍严惩离群点 | 折中选项 |
| R²（决定系数） | 1 - (残差平方和 / 总平方和) | 无量纲 0-1（可能为负）；1=完美、0=相当于预测均值、<0=比预测均值还差 | 模型间对比 / score() 返回 |

**demo-01 step4 只用了 MAE**（直观易讲）。学生 L3 应知道 R² 存在，但不要求能手算公式。

### 1.4 实战要点（工程师视角的必看清单）

1. **OLS（LinearRegression）不需要特征缩放**——不受尺度影响
2. **Ridge / Lasso 需要特征缩放**——正则项对尺度敏感
3. **特征之间高度相关（多重共线性）会让权重不稳定**——demo-01 数据没这个问题，但真实项目要警觉
4. **数据集版本要对齐**：波士顿房价已在 sklearn ≥1.2 被弃用（包含有争议的特征），仓库改用加州房价

### 1.5 正则化 · 解决过拟合的"紧箍咒"（选讲）

> **学生不主动问就别讲**。但你必须备到——原培训班深讲这块，学生翻原材料会问。

- **L1（Lasso）**：权重稀疏化，很多权重被压成 0 → 自动特征选择（"暴力裁员"）
- **L2（Ridge）**：所有权重都变小但不为 0 → 防止模型偏执（"集体降薪"），工业默认选项
- **超参数**：Lasso/Ridge 用 `alpha`（越大正则越强）；LogisticRegression 用 `C`（C = 1/alpha，**倒数**，易混淆）
- 讲法：L1 像裁员、L2 像全员降薪。不写公式

### 1.6 面试题档案（备用）

**Q：梯度下降为什么一定要先做特征标准化？**

> 学生可能在你讲求解策略时追问。下限答案："特征尺度不一致（一个范围 1-10，另一个 1-1,000,000），梯度走起来会像在很扁的椭圆里来回震荡，收敛很慢"。
> 不要画等高线图 / 不要讲凸优化。

---

## 2. Demo 代码逐步拆解

> 这一节是**直接资料对照**——每个 step 学生可能问什么、权威答案在哪。

### 2.1 step1_data.py · 加载数据、看结构

**代码做的事**：
```python
raw = fetch_california_housing()       # 返回 Bunch
df = pd.DataFrame(raw.data, columns=raw.feature_names)
df["Price"] = raw.target               # 加一列标签
print(df.head())
print(f"{len(df)} 行，{len(df.columns)} 列")
```

**Bunch 对象字段**（sklearn 数据集通用）：
- `.data` → 二维 ndarray（特征矩阵）
- `.target` → 一维 ndarray（标签/目标）
- `.feature_names` → 特征名列表
- `.target_names` → 类名列表（分类数据集才有，demo-01 回归没有）
- `.DESCR` → 数据集描述字符串
- `.filename` → 本地缓存路径

**数据集元数据（sklearn 官方）**：
- 形状：(20640, 8)
- target：房价中位数，单位 10 万美元，范围 0.15 ~ 5.0
- 8 个特征：MedInc / HouseAge / AveRooms / AveBedrms / Population / AveOccup / Latitude / Longitude

**学生可能问**：
- "load_ 和 fetch_ 有什么区别" → `load_*` = 小数据内置、`fetch_*` = 大数据首次下载缓存
- "Bunch 是什么类型" → sklearn 自定义的类字典对象，既能 `raw["data"]` 也能 `raw.data`
- "为什么要转 DataFrame" → 原始 ndarray 没列名，转 DataFrame 能带上 feature_names，后面操作按列名索引
- "df['Price'] = raw.target 是什么操作" → SQL 的 `ALTER TABLE ADD COLUMN`

### 2.2 step2_split.py · 切 X/y + 切 train/test

**代码做的事**：
```python
X = df.drop("Price", axis=1)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**train_test_split 完整签名**：
```
train_test_split(X, y,
                 test_size=0.25,      # 比例或样本数
                 train_size=None,
                 random_state=None,   # 随机种子
                 shuffle=True,         # 切前打乱
                 stratify=None)        # 分层抽样
```

**每个参数的权威含义**：
- `test_size=0.2` → 20% 做测试，80% 做训练
- `random_state=42` → 随机种子。固定后任何人/机器跑出同样切分。工程类比：单测 `seed(42)`（42 是 Kaggle 文化梗，源自《银河系漫游指南》）
- `shuffle=True` → 切前先随机打乱（默认开）
- `stratify=y` → **本代码没用**。作用是保持训练/测试集类别比例和全集一致。回归任务基本用不上（stratify 主要用于分类）

**大写 X 小写 y**：sklearn / 论文全部遵循的惯例。X 是矩阵（大写），y 是向量（小写）。

**学生可能问**：
- "为什么要切分" → 防止"背答案"。测试集是模型没见过的数据，才能反映真实能力
- "20%/80% 可以改吗" → 可以。但数据少于 1 万条时测试集通常 20-30%，更大可以降到 10%
- "axis=1 和 axis=0 怎么记" → axis=0 沿行向下（操作行），axis=1 沿列向右（操作列）。或用关键字参数 `columns=["Price"]` 避开歧义
- "random_state 不设会怎样" → 每次运行切出不同的训练/测试集，模型表现也会波动。**调试阶段必须固定，线上随机切分可以不固定**

### 2.3 step3_train.py · 创建模型 + 训练

**代码做的事**：
```python
model = LinearRegression()
model.fit(X_train, y_train)

for name, weight in zip(X.columns, model.coef_):
    print(f"{name:>12s}  {weight:+.4f}")
print(f"基础价：{model.intercept_:+.4f}")
```

**LinearRegression 权威签名**：
```
LinearRegression(fit_intercept=True,   # 是否学截距
                 copy_X=True,
                 n_jobs=None,
                 positive=False)       # 是否强制系数≥0
```

**方法 / 属性清单**：

| 方法 | 返回 | 说明 |
|---|---|---|
| `fit(X, y)` | self | 拟合 |
| `predict(X)` | ndarray (n_samples,) | 预测连续数字 |
| `score(X, y)` | float | R² 分数，(-∞, 1.0] |

| 属性 | shape | 含义 |
|---|---|---|
| `coef_` | (n_features,) | 8 个权重 |
| `intercept_` | float | 偏置 b |
| `n_features_in_` | int | 训练时的特征数 |

**权重怎么读**：
- 正数：特征越大，房价越高
- 负数：特征越大，房价越低
- 绝对值大小 ≠ 特征重要性（量纲不一样不能直接比）——**除非特征先做了标准化**，这是常见误解

**学生可能问**：
- "模型学到的到底是什么" → 就是这一组 coef_ 和 intercept_。模型文件里存的也就是这些数字 + 元数据
- "权重为什么有负数" → 某些特征和房价负相关（比如 AveBedrms 太高反而房价低，可能因为卧室占比高意味着房子总空间被分散）
- "为什么不做标准化" → LinearRegression 的 OLS 求解对尺度不敏感。如果换成 Ridge/Lasso 就必须做

### 2.4 step4_predict.py · 预测 + 评估

**代码做的事**：
```python
y_pred = model.predict(X_test)

for i in range(10):
    real = y_test.values[i]
    pred = y_pred[i]
    err = pred - real
    bar = "██" * int(abs(err) * 5)
    print(f"真实 {real:.2f}  预测 {pred:.2f}  误差 {err:+.2f}  {bar}")

mae = mean_absolute_error(y_test, y_pred)
print(f"平均误差：{mae:.3f}（× 10万美元 = {mae*10:.1f} 万美元）")
print(f"平均房价：{y_test.mean():.3f}")
print(f"误差占比：{mae / y_test.mean() * 100:.1f}%")
```

**关键 API**：
- `model.predict(X_test)` → 返回 (n_samples,) 的数字数组
- `mean_absolute_error(y_true, y_pred)` → 返回浮点数，越小越好
- `y_test.values` → 把 pandas Series 转成 numpy 数组

**demo-01 的真实表现**：MAE ≈ 0.53（5.3 万美元），平均房价 2.06（20.6 万美元），误差占比 25.9%。这个数字说明什么？

- 模型能力有限，线性假设太简单（加州房价有地理、时段非线性因素）
- 这不是代码 bug，是线性回归在这个数据上的上限
- 学生看到这个输出**不能误以为代码有问题**——session-06 已确认他理解这点

**学生可能问**：
- "为什么不是 0% 误差" → 连人都做不到 0% 误差。模型能学到的只有数据里的信号，剩下的全是噪声
- "25.9% 算好还是差" → 对线性回归入门 demo 来说正常。工业级房价预测会用 XGBoost + 特征工程，能压到 10% 以下
- "怎么改进" → 换算法（XGBoost）/ 加特征（地段、学区、交通）/ 正则化。**到 demo-04/05 决策树那节会碰**
- "能不能画预测 vs 真实的散点图" → 可以（matplotlib），但超出 demo-01 范围。学生要画就给一段代码，不展开

### 2.5 run.py · 一体版

一个脚本从头到尾跑完整流程。用途：建立"整个 pipeline 就 50 行代码"的体感。**不是主教学路径**——主路径还是 4 个 step 拆开跑（用户明确要求分步）。

---

## 3. 知识点清单（按层，资料必须全）

### 3.1 概念层（苏格拉底为主）

| 知识点 | 已通关？ | 讲法建议 |
|---|---|---|
| 模型是函数 | 3（session-01） | 可做 spot-check，别重讲 |
| 回归任务输出连续数字 | 2（session-04） | 可 spot-check |
| 特征 / 标签（X / y） | 2（session-04） | 用 step2 的 `X = df.drop("Price")` 验证手感 |
| 训练集 / 测试集切分动机 | 2（session-04） | step2 跑完让他说"为什么切" |
| 过拟合 / 泛化 | 2（session-07） | step4 评估时可 spot-check："怎么看出模型过拟合" |
| 加权累加器心智 | 待建立 | step3 看 coef_ 后的关键讲法（见 §1.1） |
| MAE vs MSE vs R² | 待建立 | step4 评估时直讲对比（见 §1.3） |
| 线性回归的局限 | 待建立 | demo-01 误差 25.9% 是最好的活教材 |

### 3.2 API 层（直讲）

- `pd.DataFrame(data, columns=...)` · `.head()` · `.shape` · `.drop(col, axis=1)` · `df["col"]` · `.iloc[行, 列]`
- `fetch_california_housing()` → Bunch 对象
- `train_test_split(X, y, test_size=, random_state=, shuffle=, stratify=)`
- `LinearRegression()` + `.fit()` + `.predict()` + `.score()` + `.coef_` + `.intercept_`
- `mean_absolute_error(y_true, y_pred)` · `mean_squared_error(..., squared=False)` → RMSE · `r2_score`

### 3.3 规约层（直讲）

- X 必须是 2D (n_samples, n_features)；单样本 `X.reshape(1, -1)` 或 `[[f1, f2, ...]]`
- y 回归是 1D 数字数组
- predict 返回保持和训练时一致的列顺序/数量
- 大写 X / 小写 y 的 ML 惯例（不是语法，是行内规矩）
- `random_state=42` 的文化梗
- `axis=0` 沿行下、`axis=1` 沿列右（pandas / numpy 通用）

### 3.4 工程层（L3 目标）

- 模型文件存的是什么 → coef_ + intercept_ + feature schema
- 训推一致性 → 线上推理时列顺序必须和训练一致
- 模型部署形态 → `.pkl` / `.onnx` / `.pmml`，加载后 `model.predict(features)` 像调普通函数
- 跨语言落地路径 → Python 训练 → ONNX 导出 → Java/Go/Node 用 onnxruntime 加载
- 这是用户护城河所在（CLAUDE.md 明示）

### 3.5 延展层（学生问到再讲）

| 话题 | 下限答案 | 权威出处 |
|---|---|---|
| 权重怎么算出来的 | "模型试出来的" → 追问给两条路径（正规方程 / 梯度下降） | 本文 §1.2 |
| 正则化 L1 / L2 | L1=裁员、L2=降薪；`alpha` 控制力度 | 本文 §1.5 |
| 梯度下降变体 BGD/SGD/mini-batch | "一次用多少数据算梯度的区别" | 原培训班 `assets/.../线性回归.md` |
| SGDRegressor 和 LinearRegression 区别 | "求解策略不同：前者迭代，后者一步到位" | sklearn 官方 |
| 多重共线性 | "特征之间高相关导致权重不稳定" | 不主动展开 |
| 波士顿数据集为什么弃用 | "含有争议的特征，sklearn ≥1.2 已移除" | sklearn 官方 |

---

## 4. 四个 step 的推进建议

| step | 建议时长 | 讲法分工 | 关键动作 |
|---|---|---|---|
| step1 | 10-15 min | 直讲（Bunch 字段规约）+ 苏格拉底（行列代表什么） | 让他自己看输出，说"一行 = 一个街区" |
| step2 | 5-10 min | 直讲（train_test_split 参数）+ 苏格拉底（为什么要切） | 确认 "切分防过拟合" 心智 |
| step3 | 15-25 min | 直讲（fit/coef_ API）+ 苏格拉底（加权累加器类比） | 看 coef_ 输出，让他说"哪些特征正向、哪些负向" |
| step4 | 15-20 min | 直讲（MAE API）+ 苏格拉底（25.9% 误差能接受吗） | 带出"线性模型有上限"的直觉，埋决策树章伏笔 |

---

## 5. 检测方案

> 评分标准：0 没听过 / 1 听过讲不清 / 2 能一句话讲清 / 3 能用工程类比向同事解释 + 知道场景

### 5.1 Level 2 检测题（demo-01 跑完该过）

- "用自己的话说 ML 的 4 步流程" → 已在 session-06 通过
- "为什么要切训练集和测试集" → 防背答案
- "coef_ 是什么，长度是多少" → 8 个数，每个是对应特征的权重
- "MAE 是什么含义，单位和谁一样" → 平均绝对误差，单位和 y（房价）一样
- "模型给的权重有正有负，正负代表什么" → 特征和房价的相关方向

### 5.2 Level 3 检测题（线性回归这节课的终点）

选型判断：
- "给你一个任务：预测用户下周登录次数。选线性回归还是逻辑回归？" → Linear，连续数字
- "给你一个任务：预测用户会不会流失。选哪个？" → Logistic，二分类
- "什么场景下线性回归是坏选择？" → 数据关系非线性（S 型、阶跃、周期）/ 特征维度极高 / 要预测的是类别

架构类（转型护城河）：
- "LinearRegression 训练完，推理服务要存什么能在 Java 里跑" → coef_ + intercept_ + feature schema。跨语言通过 ONNX / PMML
- "线上数据分布和训练时不一样了会怎样" → 模型衰退 / 状态漂移（B 组已通关）
- "怎么在代码里判断过拟合发生了" → 对比 `model.score(X_train, y_train)` 和 `model.score(X_test, y_test)`

---

## 6. 附录

### 6.1 数据集对照表

| 维度 | 仓库 demo-01（加州房价） | 原培训班 03 章（波士顿房价） |
|---|---|---|
| 形状 | (20640, 8) | (506, 13) |
| sklearn API | `fetch_california_housing()` | `load_boston()` **已在 1.2 弃用** |
| 加载方式 | 直接调 API | 从 CMU 在线 URL 手动读 CSV |
| target | 房价中位数 × 10 万美元 | 房价中位数（单位千美元） |
| 原因 | 波士顿数据集含有争议特征 | 历史遗留 |

学生如果翻原培训班 `assets/source-materials/第4阶段-机器学习/线性回归.md` 会看到波士顿代码。直接告诉他弃用了，不用纠结。

### 6.2 sklearn 官方文档链接

- LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
- SGDRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html
- Ridge: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- Lasso: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
- train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- mean_absolute_error: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
- mean_squared_error: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
- r2_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
- fetch_california_housing: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

### 6.3 项目内相关文档

- **已删除的原讲义**：`01-Machine-Learning-Foundation/03-Linear-Regression-Engineering.md` 的全部内容已迁入本文 §1（2026-04-19 重构）
- 原培训班对照：`assets/source-materials/第4阶段-机器学习/线性回归.md`（gitignored，本地可访问）
- Orientation（学习方法总纲）：[`01-LESSON-PLAN.md`](../00-mental-model/01-LESSON-PLAN.md)
- 开发者视角总览：[`01-LESSON-PLAN.md`](../00-mental-model/01-LESSON-PLAN.md)
- demos 总清单：[`demos/README.md`](../README.md)
- 下一节（逻辑回归）：[`demos/demo-02-breast-cancer/LESSON-PLAN.md`](../demo-02-breast-cancer/LESSON-PLAN.md)

### 6.4 相关 session 日志

- session-01 · 2026-04-07 · "模型是函数" 心智建立
- session-03 · 2026-04-07 · 特征/标签 Level 2→3、curve-fit 震撼
- session-04 · 2026-04-13 · A+B 组 Level 2 全通关（含过拟合、偏差方差、训推一致性）
- session-06 · 2026-04-14 · demo-01 跑通，用户能说 4 步流程
- session-07 · 2026-04-19 · demo-02 step 1/2/3 跑通；本备课文档在此会话产出

### 6.5 迁移记录

本文档从以下源迁移/整合而来：

1. 讲义 `01-Machine-Learning-Foundation/03-Linear-Regression-Engineering.md`（**已删除**，2026-04-19 重构）
   - §1 加权累加器 → 本文 §1.1
   - §2 求解策略 → 本文 §1.2（去数学）
   - §3 评估指标 → 本文 §1.3
   - §4 实战代码 → 融入本文 §2（按 demo-01 step 重组）
   - §5 正则化 → 本文 §1.5（标记"选讲"）
   - §6 面试题 → 本文 §1.6

2. `demos/LESSON-PLAN.md`（双 demo 合并版，已删除）
   - 线性回归相关部分全部迁入本文
   - 分类相关部分迁入 demo-02 的 LESSON-PLAN.md

3. sklearn 官方文档
   - API 签名、参数含义、形状规约 → 本文 §2 和 §3

4. 原培训班 `assets/source-materials/第4阶段-机器学习/线性回归.md`（gitignored）
   - 用作对照（§6.1）——原文含数学推导，不迁入

**重构记录**：
- 2026-04-19：完成原讲义 03 → 本备课的迁移 + 删除原讲义
- 相关重构动作：删除 `04-Logistic-Regression-and-Classification-Metrics.md`（内容迁到 demo-02 备课）、删除 `demos/LESSON-PLAN.md` 双 demo 合并版、其他算法讲义（02/05/06/07/08）移入 `demos/` 根作为待做 demo 的资料库
