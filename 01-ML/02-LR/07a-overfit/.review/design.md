# Marimo Demo 设计文档：多项式次数与过拟合

**demo 文件名**：`poly-degree-overfit.py`  
**端口**：2733  
**学习层级**：【理解】概念 + 【实践】实时拟合观测  

---

## 1. 设计目标

通过**多项式次数滑块**控制模型复杂度，让用户实时看到：
- **左图**：散点 + 当前 degree 拟合曲线（红线）的拟合效果
- **右图**：训练 MSE / 测试 MSE 关于 degree 的双折线（U 形）
- **动态标记**：当前 degree 在右图用红点标记，实时显示训练/测试 MSE gap

**核心洞察**：U 形曲线 = 模型复杂度的"甜蜜点"的可视化锚点  
→ 低 degree = 欠拟合（训练 MSE 都很大）  
→ 中等 degree = 刚好（两条线接近）  
→ 高 degree = 过拟合（测试 MSE 急剧上升）

---

## 2. 数据

**合成数据**：$y = 0.5x^2 + x + 2 + N(0, 1)$

- 样本数：$N=100$
- 自变量：$x \sim \text{Uniform}(-3, 3)$
- 噪声：$\epsilon \sim N(0, 1)$（标准正态）
- **自动 70/30 拆分**：70 个训练样本，30 个测试样本
- **固定随机种子**（e.g. `np.random.seed(666)`）确保可重现

---

## 3. 交互设计

### 3.1 主控件：多项式次数滑块

```
degree_slider = mo.ui.slider(1, 15, step=1, value=2, label="多项式次数", show_value=True)
```

- **范围**：1 ～ 15（足够展示 U 形全程）
- **默认值**：2（true underlying model 正好是 2 次，刚好拟合）
- **步长**：1（整数 degree）

### 3.2 预设档位（4 个快捷按钮）

虽然有滑块，但提供 4 个预设快速对比：

| 档位 | degree | 说明 | 预期效果 |
|---|---|---|---|
| **欠拟合** | 1 | 直线 | 训练/测试 MSE 都大，拟合曲线明显偏离数据 |
| **刚好** | 2 | 抛物线 | 两条线最接近，拟合曲线贴近数据 |
| **微过拟合** | 5 | 5 次多项式 | 测试 MSE 开始显著上升，拟合曲线有轻微抖动 |
| **严重过拟合** | 12 | 12 次多项式 | 测试 MSE 急升，拟合曲线在数据点间剧烈振荡 |

```python
# 可选实现：dropdown 或 4 个独立按钮
presets = {
    "1️⃣ 欠拟合 (degree=1)": 1,
    "2️⃣ 刚好 (degree=2)": 2,
    "5️⃣ 微过拟合 (degree=5)": 5,
    "🚀 严重过拟合 (degree=12)": 12,
}
preset_dropdown = mo.ui.dropdown(options=presets, value=2, label="快速档位")
```

当用户选择 preset，滑块同步更新；当用户拖动滑块，preset 重置为"自由探索"。

---

## 4. 双视图联动

### 4.1 左图：散点 + 拟合曲线（Altair）

**组成**：
1. **散点**：训练样本（蓝点，透明度 0.7）
2. **拟合曲线**：当前 degree 的多项式回归拟合线（**红色，lineWidth=2.5**）
3. **参考线**（虚线，淡绿，可选）：true underlying $y=0.5x^2+x+2$ 的真实曲线
4. **标题**：`"散点 + 拟合曲线 · degree={current_degree}"`

**坐标轴**：
- x: [-3.2, 3.2]
- y: [-6, 12]（根据数据范围）

**实现要点**：
- 用 `numpy.polyfit()` + `numpy.polyval()` 拟合 degree 次多项式
- 在 [-3, 3] 范围内密集采样（e.g. 200 点）生成光滑曲线
- **不绘制测试样本**（测试集只用于右图 MSE 计算）

```python
# 伪代码
X_train, X_test, y_train, y_test = train_test_split(...)
coeffs = np.polyfit(X_train, y_train, degree_slider.value)
x_dense = np.linspace(-3, 3, 200)
y_fitted = np.polyval(coeffs, x_dense)
# → Altair chart: 散点(X_train, y_train) + line(x_dense, y_fitted)
```

### 4.2 右图：训练/测试 MSE 双折线（Altair）

**组成**：
1. **度数轴**：x 轴 = degree (1 ～ 15)
2. **两条折线**：
   - **训练 MSE**：蓝色线，mark_line + mark_point
   - **测试 MSE**：橙色线，mark_line + mark_point
3. **当前度数标记**：**大红圆点**（size=250），双重圆点表示训练/测试在当前 degree 的位置
4. **数字提示**：在右上或右下显示 `train_mse = X.XXX, test_mse = Y.YYY, gap = Z.ZZZ`

**预计形状**：
- degree=1 时：两条线都高（欠拟合）
- degree=2 时：两条线都低且接近（最低点）
- degree=3~5：测试线开始上升，训练线继续下降
- degree=15：测试线远高于训练线（典型过拟合）

**实现要点**：
```python
# 预计算所有 degree=1~15 的 train/test MSE
degrees = np.arange(1, 16)
train_mses = []
test_mses = []
for d in degrees:
    coeffs = np.polyfit(X_train, y_train, d)
    y_pred_train = np.polyval(coeffs, X_train)
    y_pred_test = np.polyval(coeffs, X_test)
    train_mses.append(mean_squared_error(y_train, y_pred_train))
    test_mses.append(mean_squared_error(y_test, y_pred_test))

# 准备 DataFrame
df_mse = pd.DataFrame({
    "degree": np.repeat(degrees, 2),
    "mse": train_mses + test_mses,
    "type": ["train"] * 15 + ["test"] * 15,
})

# Altair：双线 + 当前点标记
```

---

## 5. 关键交互逻辑

### 5.1 状态变量

```python
degree = degree_slider.value  # 响应式：用户每拖动滑块即更新
```

### 5.2 联动计算

每当 `degree` 改变时：

1. **拟合系数**：`coeffs = polyfit(X_train, y_train, degree)`
2. **训练 MSE**：`np.polyval(coeffs, X_train)` → MSE
3. **测试 MSE**：`np.polyval(coeffs, X_test)` → MSE
4. **Gap**：`test_mse - train_mse`
5. **更新左图**：红线 = `polyval(coeffs, x_dense)`
6. **更新右图**：红点位置 = (degree, train_mse 和 test_mse)

### 5.3 文字反馈

在两图上方显示：

```
当前 degree={n} | 训练 MSE={train:.4f} | 测试 MSE={test:.4f} | gap={gap:.4f}
```

**Gap 配色**：
- gap < 0.2：🟢 绿色背景（拟合良好）
- 0.2 ≤ gap < 0.5：🟡 黄色背景（微过拟合）
- gap ≥ 0.5：🔴 红色背景（严重过拟合）

---

## 6. 页面布局

```
┌─────────────────────────────────────────┐
│  标题 + 说明                             │
├─────────────────────────────────────────┤
│ [Preset Dropdown]  [Slider: degree]     │
│ Current: train_mse=X, test_mse=Y, gap=Z │
├──────────────────┬──────────────────────┤
│                  │                      │
│   左图：散点      │    右图：U 形折线    │
│   + 拟合曲线     │   train & test MSE   │
│                  │   + 红点标记        │
│   600×400px      │    600×400px        │
│                  │                      │
└──────────────────┴──────────────────────┘
│  说明 & 玩法                             │
└─────────────────────────────────────────┘
```

**顶部说明**（mo.md）：
```
# 多项式次数与过拟合

用 2 次多项式（$y = 0.5x^2 + x + 2 + N(0,1)$）生成数据。
**目标**：通过调整拟合多项式的次数，观察训练/测试 MSE 如何变化。

- **左图**：散点 + 当前拟合曲线（红）
- **右图**：训练 MSE（蓝）vs 测试 MSE（橙）—— **U 形是过拟合的标志**
- **red dot**：当前 degree 的 MSE 位置
- **gap**：test_mse - train_mse，gap 越大 → 过拟合越严重
```

**底部玩法**（mo.md）：
```
## 玩法

1. 拖滑块从 degree=1 到 15，**看左图拟合曲线如何变化**
2. 观察右图红点如何沿着两条折线移动
   - 当红点落在"两线接近"的位置 → 是最优复杂度
   - 当红点在右侧（orange 线远高 blue 线）→ 过拟合开始
3. 尝试 4 个预设档位快速对比
4. 特别注意：**多项式 degree 太高时，拟合曲线会在数据点间"疯狂抖动"**
```

---

## 7. 代码结构（伪代码）

```python
@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

@app.cell
def _(mo):
    # UI 控件
    degree_slider = mo.ui.slider(1, 15, step=1, value=2, label="多项式次数")
    preset_dropdown = mo.ui.dropdown(
        {"欠拟合 (1)": 1, "刚好 (2)": 2, "微过 (5)": 5, "严重过 (12)": 12},
        value=2
    )
    mo.vstack([preset_dropdown, degree_slider])

@app.cell
def _(mo, degree_slider, preset_dropdown):
    # 同步逻辑（可选：拖滑块 → preset 重置）
    ...

@app.cell
def _():
    # 生成合成数据
    np.random.seed(666)
    X = np.random.uniform(-3, 3, 100)
    y = 0.5 * X**2 + X + 2 + np.random.normal(0, 1, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

@app.cell
def _(degree_slider, X_train, X_test, y_train, y_test):
    # 计算当前 degree 的拟合和 MSE
    degree = degree_slider.value
    coeffs = np.polyfit(X_train, y_train, degree)
    
    y_pred_train = np.polyval(coeffs, X_train)
    y_pred_test = np.polyval(coeffs, X_test)
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    gap = test_mse - train_mse
    
    # for left chart
    x_dense = np.linspace(-3, 3, 200)
    y_fitted = np.polyval(coeffs, x_dense)

@app.cell
def _(gap):
    # 配色
    color = "#10b981" if gap < 0.2 else ("#fbbf24" if gap < 0.5 else "#ef4444")
    # 显示文字反馈
    ...

@app.cell
def _(X_train, y_train, X_test, y_test):
    # 预计算所有 degree 的 MSE（缓存）
    all_degrees = np.arange(1, 16)
    train_mses = []
    test_mses = []
    for d in all_degrees:
        coeffs = np.polyfit(X_train, y_train, d)
        yp_train = np.polyval(coeffs, X_train)
        yp_test = np.polyval(coeffs, X_test)
        train_mses.append(mean_squared_error(y_train, yp_train))
        test_mses.append(mean_squared_error(y_test, yp_test))
    
    df_mse = pd.DataFrame({
        "degree": list(all_degrees) * 2,
        "mse": train_mses + test_mses,
        "type": ["train"]*15 + ["test"]*15
    })

@app.cell
def _(alt, X_train, y_train, x_dense, y_fitted):
    # 左图：散点 + 拟合曲线
    scatter = alt.Chart(...).mark_circle(...)...
    line = alt.Chart(...).mark_line(...)...
    chart_left = (scatter + line).properties(width=600, height=400)

@app.cell
def _(alt, df_mse, degree, train_mse, test_mse):
    # 右图：U 形折线 + 红点
    lines = alt.Chart(df_mse).mark_line().encode(...)
    points = alt.Chart(df_mse).mark_point().encode(...)
    cur_point = alt.Chart(...).mark_circle(size=250, color="#ef4444").encode(...)
    chart_right = (lines + points + cur_point).properties(width=600, height=400)

@app.cell
def _(mo, chart_left, chart_right):
    # 并排显示
    mo.hstack([mo.ui.altair_chart(chart_left), mo.ui.altair_chart(chart_right)])
```

---

## 8. 样式指南

### 颜色方案

| 元素 | 颜色 | 用途 |
|---|---|---|
| 散点（训练数据） | #1f77b4 (蓝) | 区分训练集 |
| 拟合曲线 | #ef4444 (红) | 当前模型预测 |
| 训练 MSE 线 | #3b82f6 (蓝) | 对比：训练表现 |
| 测试 MSE 线 | #f97316 (橙) | 对比：泛化能力 |
| 当前点标记 | #ef4444 (红) | 突出当前状态 |
| gap 提示（好） | #10b981 (绿) | 拟合良好 |
| gap 提示（中） | #fbbf24 (黄) | 开始过拟合 |
| gap 提示（差） | #ef4444 (红) | 严重过拟合 |

### 字体与大小

- 标题：16px, 加粗
- 图表标题：12px
- 文字反馈：13px, monospace（数字对齐）
- 轴标签：11px

### 中文适配

```python
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
```

（Altair 中文不需要配置）

---

## 9. 关键数值验证

按 README 中的底稿示例，用 `random_state=5` 或 `random_state=666` 时：

| degree | 训练 MSE | 测试 MSE | 说明 |
|---|---|---|---|
| 1 | ~3.08 | ~3.15 | 欠拟合：都很大 |
| 2 | ~1.10 | ~1.11 | 最优：接近真实分布 |
| 5 | ~0.95 | ~1.41 | 开始过拟合 |
| 10+ | ~0.85 | ~1.8+ | 严重过拟合：gap 急升 |

**目标：确保数据能清晰展示 U 形双曲线**

---

## 10. 不做的（scope 外）

- ❌ 正则化（L1/L2）→ 07b 章节的内容
- ❌ 交叉验证（K-fold）→ 增加复杂性，本 demo 只演示 train/test split
- ❌ 其他损失函数 → 只用 MSE
- ❌ 3D 曲面（Loss landscape）→ 本 demo 重点是"1D 折线"，不是"地形图"

---

## 11. 总结

这个 demo 的核心设计是**多项式次数滑块驱动双视图联动**：
- 左图让用户**看到**拟合曲线如何变化（从光滑到疯狂振荡）
- 右图让用户**量化**过拟合（训练/测试 MSE gap）
- 红点标记 + gap 数字提示让"甜蜜点"变成清晰可见的视觉锚点

**教学目标达成**：学生通过 5 分钟的交互，能直观理解"过拟合 = 模型太复杂"这一概念，且能指出"怎样的复杂度刚好"。
