---
tags: [算法/逻辑回归, API/sklearn, 正则化/solver]
---

# API 介绍

> 维度：**代码**
> 知识点级别：【知道】LogisticRegression 三个核心参数
> 章节底稿全文见 [`README.md`](./README.md)（PPT slide 25）

## ━━━━━━━━ 底稿 ━━━━━━━━

### PPT

> Slide 25 · 逻辑回归API函数和案例 – API介绍

```
sklearn.linear_model.LogisticRegression(solver='liblinear', penalty='l2', C = 1.0)
```

solver（计算引擎）:
1. liblinear 对小数据集场景训练速度更快，sag 和 saga 对大数据集更快一些。
2. 正则化：
   1. sag、saga 支持 L2 正则化或者没有正则化
   2. liblinear 和 saga 支持 L1 正则化

penalty：正则化的种类，l1 或者 l2

C：正则化力度

默认将类别数量少的当做正例

〔图：API 参数界面截图〕

### 笔记

> 【知道】API 介绍

```python
sklearn.linear_model.LogisticRegression(solver='liblinear', penalty='l2', C=1.0)
```

> 旁注：上述 `solver='liblinear'` 是旧版 sklearn 默认；sklearn ≥1.0 实际默认为 `'lbfgs'`。

**关键参数**：

- `solver`：损失函数优化方法
  - `liblinear`：小数据集更快（默认）
  - `sag` / `saga`：大数据集更快
  - `newton-cg` / `lbfgs` / `sag` / `saga`：仅支持 L2 / 无正则
  - `liblinear` / `saga`：支持 L1 正则
- `penalty`：正则化种类，`l1` 或 `l2`
- `C`：正则化力度（**越小正则越强**，与岭回归 `alpha` 相反）

**约定**：sklearn 默认把**类别数量少**的当作正例。

---

## ━━━━━━━━ 讲解 ━━━━━━━━

### 生活锚 · 微波炉预设按钮为什么只给三个旋钮

家里那台微波炉面板上密密麻麻一堆按钮，但日常你只会拧三个：

- **食物类型**：解冻 / 加热 / 烧烤——选错了肉烤焦、汤溢出
- **份量档位**：小 / 中 / 大——一人份按"大"会糊
- **时长 / 火力**：30 秒 vs 3 分钟——一秒之差冷热两重天

剩下十几个按钮（童锁 / 时钟 / 自定义记忆位）平时根本不碰。微波炉厂商把"99% 场景下用户该调的旋钮"挑出来摆在最显眼位置，错配会立刻报警（"E3"闪烁），别的细节交给默认。

调 sklearn 的 `LogisticRegression` 是同一个手感：十几个参数里只有三个常碰，本节把这三个旋钮的位置摸熟。

### 业务问题

工程师拿到任务："上线一个二分类模型，跑通就行"——这是 LR 在工业界最常见的接活姿势。打开 sklearn 文档一看，`LogisticRegression` 类列了十几个参数，文档密密麻麻像汽车说明书。

业务侧不关心这些，只想要三件事：

- **跑得动**：数据规模 1 万还是 100 万，能不能在合理时间出结果
- **不过拟合**：训练集 99%、测试集 70% 的尴尬别再来
- **改对方向**：调一档参数模型变好还是变坏，得心里有数

但参数之间还互相打架：选错了组合直接报错，选对了组合也可能悄悄退化成另一个模型。十几个里到底哪三个最常被改、改错会怎样？下面把这三个旋钮钉死。

### 【代码】调用骨架

```
导入  →  实例化（设三参数） →  fit(X, y)  →  predict / predict_proba / score
```

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver='lbfgs', penalty='l2', C=1.0, max_iter=1000)
clf.fit(X_train, y_train)                  # 训练：内部跑 solver 优化损失
y_pred  = clf.predict(X_test)              # 输出类别（0/1）
y_proba = clf.predict_proba(X_test)        # 输出概率，shape = (n, 2)
acc     = clf.score(X_test, y_test)        # 默认是准确率 accuracy
print(clf.coef_, clf.intercept_)           # 训练好的 w 和 b
```

### 【代码】关键 API 解读

#### `solver` 求解器（optimizer 求解算法）

sklearn 把"用什么数值方法去最小化交叉熵损失 cross-entropy loss"封装成 `solver` 字符串。**不影响模型形式，只影响怎么找到最优 w / b**。

| 取值 | 适用规模 | 支持的 `penalty` |
|---|---|---|
| `liblinear` | 小数据（< 10⁴ 样本） | `l1` / `l2` |
| `lbfgs`（≥1.0 默认）| 中小数据 | `l2` / `none` |
| `newton-cg` | 中小数据 | `l2` / `none` |
| `sag` | 大数据 | `l2` / `none` |
| `saga` | 大数据 | `l1` / `l2` / `elasticnet` / `none` |

> 数学推导（为什么 lbfgs 适合中小、saga 适合大数据）→ 02-原理 章主场，本节只看接口。

#### `penalty` 正则化（regularization）

往损失函数后面加一项惩罚，避免 `w` 过大导致过拟合。

- `'l2'`（默认）：惩罚 $\sum w_i^2$，**收缩 shrinkage** —— 所有 `w` 按比例缩小
- `'l1'`：惩罚 $\sum \lvert w_i \rvert$，**稀疏 sparsity** —— 部分 `w` 直接归零，自动做特征选择
- `'none'`（sklearn ≥1.2 写作 `None`）：不加正则
- `'elasticnet'`：L1 + L2 混合，仅 `saga` 支持，需配合 `l1_ratio`

> L1 / L2 的几何对比 / 为什么 L1 会稀疏 → 后续正则化章节主场。

#### `C` 正则化强度倒数（inverse of regularization strength）

注意：**`C` 越小，正则越强**，和岭回归 `Ridge(alpha=...)` 的方向相反。

| `C` 值 | 含义 | 行为 |
|---|---|---|
| `C = 100`  | 正则弱 | 几乎不约束 `w`，逼近无正则 LR，可能过拟合 |
| `C = 1.0`（默认）| 中等 | 通用起点 |
| `C = 0.01` | 正则强 | `w` 被压得很小，可能欠拟合 |

调参时通常对数尺度网格搜索：`C ∈ {0.001, 0.01, 0.1, 1, 10, 100}`。

#### 其他常用参数（一行带过）

- `max_iter`：求解器最大迭代次数。默认 100，`lbfgs` 在未标准化数据上常不收敛 → 报 `ConvergenceWarning`，加大到 1000 或先做标准化
- `class_weight='balanced'`：类别不均衡时按频率反比加权
- `random_state`：`sag` / `saga` / `liblinear` 涉及随机时固定种子

### 常见坑

- **solver vs penalty 不兼容**：`LogisticRegression(solver='lbfgs', penalty='l1')` 直接报错 `Solver lbfgs supports only 'l2' or 'none' penalties`。改 `solver='liblinear'` 或 `'saga'`
- **`C` 方向反直觉**：从 `Ridge` / `Lasso` 迁移过来的人惯性认为"系数越大正则越强"，LR 是反的
- **未收敛警告**：`ConvergenceWarning: lbfgs failed to converge` —— 99% 的原因是没标准化或 `max_iter` 太小，不是模型有问题
- **默认正例方向**：sklearn 把**类别数量少**的当正例（`classes_` 排序后的第二类）。如果你的标签是 `{2: 良性, 4: 恶性}`，恶性少 → 恶性是正例，`predict_proba` 第二列才是"恶性概率"。生产代码里别靠默认，显式查 `clf.classes_`
- **`predict_proba` 列序**：列顺序按 `clf.classes_` 排，不是按你的标签出现顺序

### 【代码】跨语言落地视角

LR 训练完只需保存两个数组：`coef_`（shape `(1, n_features)`，二分类）和 `intercept_`（shape `(1,)`）。Go / Java / Node 推理 5 行搞定，不依赖 sklearn：

```python
# Python 端导出
import json
json.dump({'w': clf.coef_.tolist(), 'b': clf.intercept_.tolist()}, open('lr.json', 'w'))
```

```go
// Go 端推理（伪代码骨架，编译需自补 import）
func predict(x []float64, w []float64, b float64) int {
    z := b
    for i := range x { z += w[i] * x[i] }
    if 1.0/(1.0+math.Exp(-z)) >= 0.5 { return 1 }
    return 0
}
```

这是 LR 在生产环境受欢迎的核心原因之一：**模型即两个数字数组**，跨语言部署成本接近零。树模型 / 神经网络都做不到。

**这一节的关键启示**：solver 选规模，penalty 选形态，C 反着调——三参数定 LR。

### 术语

| 故事里的元素 | 术语名 | 主场 |
|---|---|---|
| 求最小损失的数值方法 | 求解器 solver / 优化算法 optimizer | 本节 + 02 原理 |
| 防过拟合的惩罚项 | 正则化 regularization（L1 / L2） | 后续正则化章 |
| 正则化强度的倒数 | `C` = inverse regularization strength | 本节 |
| 训练好的系数 / 截距 | 权重 weight = `coef_` / 偏置 bias = `intercept_` | 02 原理 |
| 输出概率而非类别 | `predict_proba` | 04 评估 |
| 训练时损失没降到底 | 收敛 convergence / `ConvergenceWarning` | 本节 |

→ 下一节：[`02-癌症分类案例.md`](./02-癌症分类案例.md) 把这三个参数用到真实数据上。

> Sources：
> - PPT Slide 25
> - sklearn.linear_model.LogisticRegression
