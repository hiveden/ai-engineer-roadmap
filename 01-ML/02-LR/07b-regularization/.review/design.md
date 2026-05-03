# Marimo Demo 设计文档：L1 / L2 正则化的收缩与稀疏

**demo 文件名**：`l1-l2-shrinkage.py`
**端口**：2734
**学习层级**：【理解】L1 稀疏化 vs L2 等比衰减；【实践】λ 调节与权重观测
**对应 PPT**：slide 109 ~ 120（07b-regularization README）
**前置 demo**：07a-overfit/`poly-degree-overfit.py`（端口 2733）— 同款 10 次多项式过拟合数据

---

## 1. 教学目标（5 条）

1. **看见过拟合的解药**：用与 07a 相同的数据 + 10 次多项式，复现红色"疯狂振荡"的过拟合曲线，再用 L1/L2 把它"压平"。
2. **L1 = 砍特征**（稀疏解）：观察 Lasso 把高次项系数**直接归零**，权重柱图出现一排"光秃秃"的零柱，并用红框标记。
3. **L2 = 压特征**（等比收缩）：观察 Ridge 把所有系数按比例缩小，**没有归零**，权重柱呈现"整体矮化"。
4. **λ 是惩罚力度旋钮**：对数滑块 1e-4 → 1e2 扫一遍，看 train MSE 单调上升、test MSE 先降后升的 **U 形偏差-方差权衡**。
5. **理解几何直觉**：L1 等量收缩 vs L2 按比例收缩，归零的是哪些维度（高次方项），保留的是哪些（线性 + 二次项）。

---

## 2. 知识点映射（PPT slide → demo 视图）

| PPT slide | 知识点 | demo 视图 |
|---|---|---|
| 109 思想 | 加惩罚项 → 抑制权重 | 顶部 mo.md 公式 + 模式徽章 |
| 110 正则项作用 | 第一项拟合 + 第二项压权重 | B 权重柱图（拟合 vs 收缩两种力的妥协结果） |
| 111 L1 公式 + 稀疏 | $\lambda \sum \lvert w_i \rvert$；w 趋于 0 甚至等于 0 | B 视图 L1 列 **红框标 weight=0** + 切换 toggle |
| 112 L2 公式 + 橡皮筋 | $\lambda \sum w_i^2$；趋向 0 但不等于 0 | B 视图 L2 列（无红框，全部非零但矮）|
| 113~115 API（Lasso / Ridge）| sklearn 用法 | demo 内部直接 `Lasso(alpha=λ)` / `Ridge(alpha=λ)` |
| 116~117 总结 | L1 砍 / L2 压 | 底部说明 callout |
| 120 填空 | α 越大正则力度越大、权重越小 | C 视图 U 形折线 + λ 对数滑块手感验证 |

---

## 3. 视图布局（A + B + C 三视图）

### A · 散点 + 三条拟合曲线对比（左）

**内容**：
- 训练样本：蓝色圆点（70 个），透明度 0.7，stroke=white
- 测试样本：浅蓝空心圆（30 个），可选半透明
- 三条拟合曲线（均为当前 λ 下的拟合，用同一 λ 重训三个模型对比）：
  - **无正则**（LinearRegression）：红色实线 `#ef4444`，strokeWidth=2.5 — 当前 λ 下作为基线，不变
  - **L1 (Lasso)**：橙色实线 `#f97316`，strokeWidth=2.5
  - **L2 (Ridge)**：绿色实线 `#10b981`，strokeWidth=2.5
- 真实曲线（灰色虚线 dashArray=[4,3]）：$y = 0.5x^2 + x + 2$ 作为参考锚点

**坐标轴**：x ∈ [-3.2, 3.2]，y ∈ [-6, 12]

**标题**：`A · 三种拟合对比 · λ={λ:.4f}`

**实现要点**：
- 在 [-3, 3] 上密采样 200 点，分别用三个 pipeline 预测
- pipeline = `PolynomialFeatures(degree=10) → StandardScaler → {LinearRegression | Lasso(α=λ) | Ridge(α=λ)}`
- λ 只影响 L1/L2 两条曲线，无正则线作为不变锚点

**工具**：Altair（mark_circle + mark_line + mark_rule for true curve）

**尺寸**：width=440, height=360

---

### B · 权重柱状图（中）—— 主舞台

**内容**：10 个权重 $w_1, w_2, \ldots, w_{10}$（对应 $x^1, x^2, \ldots, x^{10}$）按 degree 从左到右排列，三组并排：

```
  w₁  w₂  w₃  w₄  w₅  w₆  w₇  w₈  w₉  w₁₀
┌─────────────────────────────────────────┐
│ ▇   ▆        ▇▇▇      ▇▇            ▇   │  无正则（红） — 高次项疯涨
├─────────────────────────────────────────┤
│ ▇   ▆   ░   ░   ░   ░   ░   ░   ░   ░   │  L1（橙）— 后 8 个被红框框住的 0
├─────────────────────────────────────────┤
│ ▆   ▅   ▃   ▂   ▁   ▁   ▁   ▁   ▁   ▁   │  L2（绿）— 全部小但非零
└─────────────────────────────────────────┘
```

**关键视觉编码**：
- 三行并排，每行 10 根柱子，颜色与 A 视图一致（红 / 橙 / 绿）
- y 轴：权重值（可正可负），加 `mark_rule(y=0)` 黑色基线
- **L1 行的零柱**：当 `|w_i| < 1e-6` 时，柱位置画一个 **红色虚线框**（`mark_rect` + `strokeDash=[3,2]` + `fillOpacity=0` + `stroke='#dc2626'`），并加 "0" 文字标签，强调"被砍掉"
- 每根柱子顶部显示数值（小字号 9px，浮点保留 2 位）
- 标题：`B · 权重对比 · L1 红框 = 被剔除的特征`

**实现要点**：
- 三个模型权重展平为 long-format DataFrame：`{degree: 1~10, weight: float, type: 'none'|'L1'|'L2', is_zero: bool}`
- Altair facet 或 layer 实现三行
- 红框层只保留 `type='L1' & is_zero=True` 的子集

**工具**：Altair `mark_bar` + `mark_rule` + `mark_rect`（零标记）+ `mark_text`（数值）

**尺寸**：width=520, height=360

---

### C · 训练/测试 MSE 关于 λ 的双折线（右）

**内容**：
- 横轴：λ，**对数刻度** `scale=alt.Scale(type='log')`，范围 [1e-4, 1e2]
- 两条折线（取决于当前 toggle 是 L1 还是 L2，画选中那种的曲线）：
  - 训练 MSE：蓝色 `#3b82f6` `mark_line + mark_point`
  - 测试 MSE：橙色 `#f97316` `mark_line + mark_point`
- 当前 λ 标记：红色大圆点 size=250 在两条线上各一个
- 垂直参考线：当前 λ 的 `mark_rule`（灰色虚线）

**预期形状**（U 形）：
- λ 极小（1e-4）：几乎等于无正则 → 训练 MSE 极低，测试 MSE 偏高（过拟合）
- λ 适中（1e-2 ~ 1）：训练略升、测试到达谷底（甜蜜点）
- λ 极大（1e2）：所有权重被压趋零 → 退化为常数预测，训练/测试都飙升（欠拟合）

**标题**：`C · {L1|L2} · λ - MSE U 形 · 当前 λ={λ:.4f}`

**数字反馈**：图右上角文字 `train MSE = X.XXX | test MSE = Y.YYY | gap = Z.ZZZ`

**实现要点**：
- 预计算所有 λ 网格（30 个对数采样点）的 train/test MSE，缓存为 DataFrame
- 切换 L1/L2 toggle 时切换数据源
- 当前 λ 不一定落在网格点上 → 单独算一次当前 λ 的 train/test MSE 作为红点

**工具**：Altair（log scale + 双线）

**尺寸**：width=440, height=360

---

## 4. 控件设计

### 4.1 λ 对数滑块（核心）

```python
log_lambda = mo.ui.slider(
    start=-4.0, stop=2.0, step=0.05, value=-1.0,
    label="log₁₀(λ)", show_value=True,
)
# 实际 λ = 10 ** log_lambda.value
```

- **范围**：log₁₀(λ) ∈ [-4, 2] → λ ∈ [1e-4, 1e2]
- **默认**：log₁₀(λ) = -1 → λ = 0.1（适中正则）
- **步长**：0.05（足够丝滑）
- **为什么对数**：λ 的影响是数量级的，线性滑块在 [1e-4, 1e2] 区间几乎全部聚在 0 附近不可调

### 4.2 L1 / L2 toggle（Radio）

```python
reg_type = mo.ui.radio(
    options=["L1 (Lasso)", "L2 (Ridge)"],
    value="L1 (Lasso)",
    label="正则化类型",
    inline=True,
)
```

- 影响 C 视图的双折线数据源（切换显示 L1 或 L2 的 U 形）
- 不影响 A / B 视图（A/B 始终同时显示三种以便对比）
- **设计取舍**：A/B 始终显示三种是为了"对比"教学目标；C 切换是因为两条 U 形曲线叠在一起会乱

### 4.3 4 个预设（Dropdown）

```python
preset = mo.ui.dropdown(
    options={
        "✋ 手动 (滑块控制)": None,
        "0️⃣ 无正则 (λ ≈ 0)": -4.0,
        "🌱 弱 (λ = 0.001)": -3.0,
        "✓ 适中 (λ = 0.1)": -1.0,
        "💪 强 (λ = 10)": 1.0,
    },
    value="✋ 手动 (滑块控制)",
    label="预设",
)
```

- 选择预设 → 锁定 log₁₀(λ) 值（覆盖滑块），徽章显示"预设模式"
- 选 "手动" → 释放滑块控制权
- 单一数据源原则：UI 上滑块仍可见，但徽章提示当前生效值来源

### 4.4 模式徽章 + 实时数字

```
┌────────────────────────────────────────────────────────────────────┐
│ 🟡 预设模式 · λ=0.1 · L1 (Lasso)                                  │
│ 当前 λ = 0.1000                                                   │
│ L1 train MSE = 1.05  test MSE = 1.12  gap = 0.07  zero_count = 7  │
│ L2 train MSE = 0.98  test MSE = 1.18  gap = 0.20                  │
│ 无正则 train MSE = 0.82  test MSE = 1.55  gap = 0.73 (严重过拟合)│
└────────────────────────────────────────────────────────────────────┘
```

- L1 行特别显示 `zero_count`（多少个权重 < 1e-6）= 稀疏度量
- gap 颜色：< 0.15 绿 / < 0.40 黄 / ≥ 0.40 红

---

## 5. 数据生成

### 合成数据（与 07a-overfit 完全一致）

```python
np.random.seed(666)
N = 100
x = np.random.uniform(-3, 3, size=N)
y = 0.5 * x**2 + x + 2 + np.random.normal(0, 1, size=N)

X = x.reshape(-1, 1)
# 70/30 train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
```

### 多项式 + 标准化 pipeline（关键：标准化是必须的）

```python
def make_pipeline(model):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=10, include_bias=False)),
        ("scaler", StandardScaler()),
        ("reg", model),
    ])

pipe_none = make_pipeline(LinearRegression())
pipe_l1   = make_pipeline(Lasso(alpha=cur_lambda, max_iter=20000))
pipe_l2   = make_pipeline(Ridge(alpha=cur_lambda))
```

**为什么 StandardScaler 必须**：不同次数特征 $x, x^2, \ldots, x^{10}$ 量级差好几个数量级（$x \in [-3,3]$ → $x^{10} \in [0, 59049]$），不标准化的话 L1/L2 惩罚被高次项独占，结果失真。这个细节在 README slide 114 的 `normalize=True` 已弃用之后必须显式做。

### 权重读取

```python
weights_none = pipe_none.named_steps["reg"].coef_   # shape (10,)
weights_l1   = pipe_l1.named_steps["reg"].coef_
weights_l2   = pipe_l2.named_steps["reg"].coef_
```

注意：因为做了 StandardScaler，权重是**标准化空间下**的，柱图直接展示这个就行（教学用，不需要 inverse-transform 回原空间）。

### λ 网格预计算（C 视图）

```python
lambda_grid = np.logspace(-4, 2, 30)  # 30 个对数采样
records = []
for lam in lambda_grid:
    for name, model_cls in [("L1", Lasso), ("L2", Ridge)]:
        pipe = make_pipeline(model_cls(alpha=lam, max_iter=20000))
        pipe.fit(X_train, y_train)
        records.append({
            "lambda": lam, "type": name,
            "train_mse": mean_squared_error(y_train, pipe.predict(X_train)),
            "test_mse":  mean_squared_error(y_test,  pipe.predict(X_test)),
        })
df_curve = pd.DataFrame(records)
```

用 `@mo.cache` 缓存（数据是 deterministic 的）。

---

## 6. 预设方案表

| 预设 | log₁₀(λ) | λ | 预期 L1 zero_count | 预期 L2 权重幅度 | 教学要点 |
|---|---|---|---|---|---|
| **0️⃣ 无正则** | -4.0 | 1e-4 | 0~1（几乎全保留）| 全部 ≈ 无正则 | 复现 07a 过拟合曲线，作为基线 |
| **🌱 弱** | -3.0 | 1e-3 | 2~4 | 略小 | 刚开始有效果，多数权重还在 |
| **✓ 适中** | -1.0 | 0.1 | 6~8 | 显著缩小 | 甜蜜点：测试 MSE 接近最低 |
| **💪 强** | 1.0 | 10 | 9~10（几乎全砍）| 趋近 0 | 欠拟合：曲线被压成水平线，训练/测试都飙升 |

> 数值是预期值，实际跑一次记录用于文档和测试断言。

---

## 7. 5 秒测试预演（3 个判断点）

### 判断点 1（第 2 秒）：A 视图能不能"一眼区分三条曲线的形态"
- **预期**：无正则线在数据点间疯狂上下抖（07a 同款过拟合）；L1 和 L2 都贴近真实抛物线
- **不及格信号**：三条线视觉上重合 → 说明 λ 默认值太小，没拉开差距 → 调整默认 λ=0.1

### 判断点 2（第 3 秒）：B 视图能不能"一眼看出 L1 的零柱"
- **预期**：L1 行后 6~8 根柱子是"红框 + 标 0"的空位，与 L2 行的"全部矮但非零"形成视觉对比
- **不及格信号**：L1 行所有柱都还有高度 → λ 太小，提升默认或加 normalize 标志位

### 判断点 3（第 4~5 秒）：C 视图拖滑块能不能看到 U 形测试曲线
- **预期**：滑块从 -4 → 2，红点先沿测试线（橙）下行到谷底（log₁₀(λ) ≈ -1），再上升；训练线（蓝）单调上行
- **不及格信号**：测试线没有 U 形 → 数据集太小或 split 不合理，用 random_state=5 应稳定（与 07a 一致）

---

## 8. UI 草图（ASCII）

```
╔════════════════════════════════════════════════════════════════════════════╗
║  L1 / L2 正则化 · 收缩与稀疏 · 10 次多项式过拟合数据                       ║
║                                                                            ║
║  L = MSE(W) + λ · Σ|wᵢ|     (L1, Lasso)                                  ║
║  L = MSE(W) + λ · Σwᵢ²      (L2, Ridge)                                  ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  [预设▼ 适中 (λ=0.1)]  [○ L1 (Lasso) ● L2 (Ridge)]                       ║
║  [log₁₀(λ) ═══════●══════════] = -1.00  (λ = 0.1000)                     ║
║                                                                            ║
║  ┌──────────────────────────────────────────────────────────────────┐    ║
║  │  🟡 预设模式 · λ=0.1 · L1                                          │    ║
║  │  L1: train=1.05 test=1.12 gap=0.07 🟢  zero=7/10                  │    ║
║  │  L2: train=0.98 test=1.18 gap=0.20 🟡                             │    ║
║  │  无: train=0.82 test=1.55 gap=0.73 🔴 (过拟合基线)               │    ║
║  └──────────────────────────────────────────────────────────────────┘    ║
║                                                                            ║
║  ╔══════════════════╦════════════════════╦═════════════════════════╗    ║
║  ║ A · 散点+三条线   ║ B · 权重柱状图     ║ C · λ-MSE U 形 (L1)    ║    ║
║  ║                  ║ 无: ▇▆ ▇▇▇  ▇▇  ▇  ║   train╲      ╱test    ║    ║
║  ║   红线疯狂抖     ║ L1: ▇▆[0][0][0]... ║         ╲    ╱          ║    ║
║  ║   橙绿线贴真值   ║ L2: ▆▅▃▂▁▁▁▁▁▁    ║          ╲● ╱           ║    ║
║  ║                  ║                    ║   ●=当前 λ              ║    ║
║  ║   440×360        ║   520×360          ║   440×360               ║    ║
║  ╚══════════════════╩════════════════════╩═════════════════════════╝    ║
║                                                                            ║
║  ## 玩法                                                                   ║
║  1. 拖 log λ 滑块从 -4 → 2，看三条线如何从"过拟合 → 刚好 → 欠拟合"        ║
║  2. 切 L1/L2 toggle，看 C 图 U 形位置不同                                ║
║  3. 重点看 B 图：L1 行红框越来越多 = 越来越多特征被砍                     ║
║  4. 4 个预设一键对比                                                      ║
╚════════════════════════════════════════════════════════════════════════════╝
```

---

## 9. 端口 / 文件路径

| 项目 | 值 |
|---|---|
| 端口 | 2734 |
| 主文件 | `01-ML/02-LR/07b-regularization/demos/l1-l2-shrinkage.py` |
| 启动 | `.venv/bin/marimo edit --port 2734 --headless --no-token 01-ML/02-LR/07b-regularization/demos/l1-l2-shrinkage.py` |
| 自检 | `marimo export script <path> -o /tmp/_lint.py && python /tmp/_lint.py` |
| 端口邻居 | 2731=04b-gd / 2732=05-metrics / 2733=07a-overfit / **2734=07b-regularization** |

---

## 10. Cell 组织（实现骨架）

| # | Cell 内容 | 关键导出变量 |
|---|---|---|
| 1 | 导入 + matplotlib 字体 | mo, np, pd, alt, sklearn 全套 |
| 2 | 标题 + 公式 (mo.md) | — |
| 3 | UI 控件：preset / reg_type / log_lambda | preset, reg_type, log_lambda |
| 4 | 解析当前 λ（preset 优先） | cur_lambda, mode_tag_html |
| 5 | 数据生成 + split | X_train, X_test, y_train, y_test |
| 6 | 三个 pipeline 拟合（@mo.cache 按 λ） | pipe_none, pipe_l1, pipe_l2 |
| 7 | 权重提取 + zero_count | w_none, w_l1, w_l2, zero_count |
| 8 | MSE 计算 + 颜色判断 | metrics dict |
| 9 | λ 网格预计算（@mo.cache，全局一次） | df_curve |
| 10 | A 图：散点 + 三条曲线 | chart_A |
| 11 | B 图：权重柱状图 + 红框 | chart_B |
| 12 | C 图：λ-MSE U 形 | chart_C |
| 13 | 状态徽章 mo.md | badge |
| 14 | 布局合成 hstack | final |
| 15 | 玩法说明 mo.md | — |

---

## 11. 防坑清单

- ✅ **PolynomialFeatures(include_bias=False)**：避免与 LinearRegression 内部 intercept 重复，否则 L1 会把 intercept 也砍掉
- ✅ **StandardScaler 必加**：否则 L1/L2 惩罚被高次项绑架（量级 1e4 vs 1e0）
- ✅ **Lasso(max_iter=20000)**：低 λ + 高维特征下默认 1000 次迭代不收敛会 ConvergenceWarning
- ✅ **zero 阈值 1e-6**：浮点 Lasso 的"零"不是严格 0，需阈值判定
- ✅ **λ=0 不传给 Lasso/Ridge**：用 LinearRegression 替代（log₁₀(λ)=-4 时 λ=1e-4 仍可，无需特判）
- ✅ **C 图 log scale**：λ 范围跨 6 个数量级，线性轴会让低 λ 区域全挤一起
- ✅ **df_curve 缓存**：30 个 λ × 2 种正则 = 60 次 fit，用 `@mo.cache` 否则每次滑块滑动都重跑
- ✅ **B 图三行用 facet 还是 layer**：推荐 `alt.vconcat` 三个独立 chart（每行一个 type），便于精确控制红框
- ✅ **红框对零柱**：用 `mark_rect` 替代 `mark_bar`，独立编码 stroke 而不是 fill
- ✅ **真实曲线参考线**：A 图加灰色虚线 $y=0.5x^2+x+2$，让用户分辨"L1 + L2 都贴真值，无正则远离"
- ✅ **gap 颜色阈值**：复用 07a 的 0.15/0.40 区间，保持心智一致

---

## 12. 验收清单

- [ ] **预设 0️⃣ 无正则**：A 图无正则线疯狂抖动，三条线分得开
- [ ] **预设 ✓ 适中**：B 图 L1 后 7+ 根零柱（红框），L2 全部柱子矮但非零
- [ ] **预设 💪 强**：A 图 L1/L2 线退化为接近水平直线，C 图红点位于 U 形右臂
- [ ] **滑块联动**：拖 log λ → A/B/C 三视图同步刷新，无延迟
- [ ] **toggle 切换**：L1/L2 切换 → C 图 U 形数据源切换，A/B 不变
- [ ] **手动模式徽章**：选 "✋ 手动" → 徽章变蓝，滑块生效

---

## 13. 不做的（scope 外）

- ❌ Elastic Net（L1+L2 组合）→ 增加复杂度，本 demo 聚焦 L1 vs L2 二分对比
- ❌ Cross-Validation（LassoCV / RidgeCV）→ 自动选 λ 偏离"看见 λ 影响"的教学目标
- ❌ 几何等高线（菱形 vs 圆形约束区）→ 是 README 的"几何直觉"补充，但需要 2D 参数空间，本 demo 是 10 维特征，做不动；留待后续概念补充章
- ❌ Loss landscape 3D → 04b-gd 已经做过，不重复

---

## 14. 总结

本 demo 是 02-LR 第 7b 章的**核心可视化**，用与 07a 共享的"10 次多项式过拟合"数据集，把 L1（Lasso 稀疏化）和 L2（Ridge 等比衰减）两种正则化的**视觉差异**钉死在三个视图上：

- **A 图**——回到业务，看三条拟合曲线如何从"疯狂振荡"被拉回"贴近真值"
- **B 图（主舞台）**——权重柱状图三行并排，**L1 红框标零柱**是整个 demo 最强的"恍然大悟"瞬间，对应 PPT slide 114 "Lasso 会将高次方项系数变为 0" 的核心论点
- **C 图**——λ 对数滑块下的 U 形 MSE 曲线，量化"过拟合 → 甜蜜点 → 欠拟合"的连续过渡

通过 4 个预设 + 一根 λ 滑块 + 一个 L1/L2 toggle，用户在 30 秒内可建立完整心智："正则化 = 用 λ 这个旋钮，在 L1 的'砍特征'和 L2 的'压特征'之间二选一，把过拟合拉回来"。

**与 07a 的衔接**：07a 演示"过拟合是个问题"，07b 演示"L1/L2 是解药"。两者数据完全相同，端口相邻（2733/2734），可在两个浏览器 tab 并排对照学习。
