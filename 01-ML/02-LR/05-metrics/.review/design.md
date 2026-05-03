# Demo 设计文档：MAE / MSE / RMSE 对异常值的敏感度
**文件名**：`metric-vs-outlier.py` | **端口**：2732 | **学习层级**：【知道】指标公式，【了解】敏感度对比

---

## 核心设计理念

**本质目标**：把 PPT slide 86-87「MSE 对异常点敏感 → 爆炸」这个抽象概念变成"拖拽就看得见"的手感演示。

### 交互核心
用户拖拽**一个浮动的红色异常点**，实时看三个指标 (MAE / MSE / RMSE) 如何变化：
- **常态**：三个指标相近，拟合线正常
- **异常点轻微偏离**：MSE 已经明显放大
- **异常点极端偏离**：MSE 爆炸式增长，拟合线被"拽歪"

---

## 界面布局（三行结构）

### 第 1 行：控制面板
```
[预设下拉框: 轻微/中等/极端/双异常]  [X 坐标输入框]  [Y 坐标输入框]
```

**组件**：
- `preset` dropdown：4 个预设位置，一键跳转（快速对比）
- `x_outlier` / `y_outlier` slider（或 number input）：精细调整异常点坐标

### 第 2 行：四列可视化

```
┌─────────────────────────────────────────────────────────────────┐
│  [左] 散点 + 拟合线   │  [中-左] MAE 柱     │  [中-右] MSE 柱    │  [右] RMSE 柱
│  异常点高亮 红星      │  + 指标数字        │   + 指标数字       │  + 指标数字
│                      │  + baseline 灰线   │   + baseline 灰线  │  + baseline 灰线
└─────────────────────────────────────────────────────────────────┘
```

**子组件说明**：

#### 左：数据散点 + 拟合线（Altair）
- 蓝色圆点：N=15 个正常数据点 `(x, y) = 2x + 5 + N(0,1)`
- 红色五角星：可拖拽异常点（或用 slider 改坐标）
- 蓝色实线：当前拟合线 `y = LinearRegression(data+异常点)`
- 灰色虚线：baseline 拟合线（无异常点时）
- **hover 提示**：显示 (x, y, 误差)

#### 中-左/中-右/右：三个指标面板（纵向堆叠）
每个柱图：
1. **柱状图**（高度 = 指标值）
   - 异常点时：蓝色柱
   - baseline：灰色虚柱（参考线）
2. **数字面板**（下方）
   ```
   MAE:  2.45
   基线: 1.82 (1.3x)
   ```
   - 当前值
   - baseline 值
   - **放大倍数**（关键！）

### 第 3 行：信息面板（callout）

```
🔍 观察点：
  • MSE 从 baseline 的 X.XX → Y.YY，放大 {mag_mse:.1f}x！
  • MAE 从 baseline 的 A.AA → B.BB，放大 {mag_mae:.1f}x
  • RMSE 从 baseline 的 R.RR → S.SS，放大 {mag_rmse:.1f}x

💡 为什么 MSE 爆炸？因为异常点的误差被平方，占据了整个 loss 的 90%+。
   使用 MAE 时，异常点的贡献仅为线性增长，模型不被"绑架"。
```

---

## 数据生成

### 基础数据（N=15）
```python
np.random.seed(42)
x_base = np.linspace(0, 10, 15)
y_base = 2 * x_base + 5 + np.random.normal(0, 1, 15)
```

### 异常点
- 初始位置：`x_outlier=5, y_outlier=20`（明显偏离拟合线）
- 可调范围：`x ∈ [0,10], y ∈ [-5,30]`
- 用户可通过 **slider / dropdown 预设** 改位置

### 拟合计算
```python
from sklearn.linear_model import LinearRegression

# 当前数据 = 正常数据 + 异常点
X_current = np.vstack([x_base, [x_outlier]])
y_current = np.vstack([y_base, [y_outlier]])
model = LinearRegression(fit_intercept=True)
model.fit(X_current, y_current)

# baseline 模型（仅正常数据）
model_baseline = LinearRegression(fit_intercept=True)
model_baseline.fit(x_base.reshape(-1,1), y_base)
```

---

## 计算逻辑

### 三种指标

```python
def calc_metrics(y_true, y_pred):
    """返回 (mae, mse, rmse)"""
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# 当前指标（含异常点）
y_pred_current = model.predict(X_current)
mae_cur, mse_cur, rmse_cur = calc_metrics(y_current, y_pred_current)

# baseline（无异常点）
y_pred_base = model_baseline.predict(x_base.reshape(-1,1))
mae_base, mse_base, rmse_base = calc_metrics(y_base, y_pred_base)

# 放大倍数
mag_mae = mae_cur / mae_base if mae_base > 0.01 else 1.0
mag_mse = mse_cur / mse_base if mse_base > 0.01 else 1.0
mag_rmse = rmse_cur / rmse_base if rmse_base > 0.01 else 1.0
```

---

## 预设场景（preset dropdown）

4 个快速对比模式：

| 预设名 | x_out | y_out | 场景说明 |
|---|---|---|---|
| 无异常 | — | — | 仅显示 15 个正常点（baseline） |
| 轻微 | 5.5 | 14 | 离拟合线 ±2 范围，MSE ~2x |
| 中等 | 7.0 | 22 | 离拟合线 ±8 范围，MSE ~5x |
| 极端 | 5.0 | 28 | 离拟合线 ±15 范围，MSE ~20x+ |
| 双异常 | [3,8] | [12,25] | 两个异常点同时存在（高级） |

用户选择后，异常点立即跳转，三个柱图实时更新。

---

## 交互流程

### Flow 1：预设选择
```
用户点 dropdown → preset 变化
→ x_outlier, y_outlier 变化（或被锁定显示）
→ 散点图重绘（异常点位置更新）
→ 拟合线重算（LinearRegression）
→ 三个指标重算 + 柱图高度变化
→ 放大倍数更新
```

### Flow 2：手动 slider 调整
```
用户拖 x_outlier slider
→ 异常点 x 坐标变化（实时）
→ 散点图中红星位置变化
→ 拟合线立刻重算
→ 三个指标同步更新
→ 柱图高度动画过渡
```

### Flow 3：对比学习
```
用户先选 [无异常] 看 baseline
→ 记住 MAE/MSE/RMSE 的 baseline 值
→ 再选 [极端] 看爆炸
→ 信息面板自动强调"MSE 放大 20x！"
```

---

## 可视化细节

### 色彩方案
- **正常点**：蓝色 `#1f77b4`，opacity=0.8
- **异常点**：红色 `#ef4444`，marker=star，size=200
- **拟合线（当前）**：蓝色 `#3b82f6`，linewidth=2.5
- **拟合线（baseline）**：灰色 `#d1d5db`，linewidth=2，stroke dash=[3,3]
- **柱图（当前）**：蓝色 `#3b82f6`，opacity=0.85
- **柱图（baseline）**：灰色 `#e5e7eb`，opacity=0.5
- **放大倍数标签**：红色 `#dc2626`（> 5x 时），橙色 `#f59e0b`（2-5x），绿色 `#10b981`（≈ 1x）

### 柱图设计
- 纵轴：指标值（log scale 可选，避免极端异常时 MSE 爆表）
- 横轴：三个类别 [MAE, MSE, RMSE]，各占 1 柱 + baseline 灰柱
- **上方标签**：显示当前值 + baseline 值 + 倍数

示例：
```
    |
20x |
    |  ██ (blue)
    |  ██
5x  |  ██  ▓▓
    |  ██  ▓▓  ▓▓
    |  ██  ▓▓  ▓▓
1x  |  ██  ▓▓  ▓▓
    |__██__▓▓__▓▓__
      MAE MSE RMSE
      2.5 45  6.7
      1.8 3.2 1.8  <- baseline
      1.4x 14x 3.7x <- 倍数
```

---

## 技术实现清单

### Cell 架构（marimo）

| Cell | 内容 | 返回值 |
|---|---|---|
| 导入 | `mo`, `np`, `pd`, `alt`, `LinearRegression`, `plt` | — |
| 控制面板 | `preset` dropdown + `x_outlier` / `y_outlier` slider | UI 组件 |
| 数据生成 | 基础 15 点 + 异常点 | `x_base, y_base, x_outlier, y_outlier` |
| 拟合计算 | 训练 LinearRegression（当前 + baseline） | `model, model_baseline, y_pred_cur, y_pred_base` |
| 指标计算 | MAE/MSE/RMSE 三种 + 放大倍数 | `mae_cur, mse_cur, rmse_cur, mag_*` |
| 散点图（Altair） | 蓝点 + 红星 + 双拟合线 | `chart_scatter` |
| 三个柱图（Altair） | MAE / MSE / RMSE 柱 + baseline | `chart_mae, chart_mse, chart_rmse` |
| 信息面板 | callout 展示放大倍数 + 解释 | `callout_info` |
| 布局组合 | `mo.hstack` / `mo.vstack` 拼接上述图表 | 最终 demo |

### 关键技术点

1. **反应式更新**：slider 变化 → 下游所有图表自动重算（marimo 依赖追踪）
2. **实时拟合**：每帧用 `LinearRegression.fit()` 重训，确保拟合线同步
3. **Altair 交互**：hover 显示 (x, y, err)；可选：点击选择以突显
4. **动画过渡**（可选）：用 Altair 的 `transition` 让柱图高度平滑变化
5. **中文字体**：`plt.rcParams["font.sans-serif"] = ["PingFang SC"]`（若有 matplotlib）

---

## 启动命令

```bash
marimo edit --port 2732 metric-vs-outlier.py
```

---

## 玩法说明（Slide + Narrative）

### 用户视角的 5 秒教学

> 看见 15 个蓝点拟合成一条线。现在**拖红星**（异常点）从正常位置偏离。
>
> **观察**：
> - 📊 柱图：MSE 柱子疯狂涨高（20x!），MAE 只涨 3x
> - 📈 拟合线：被红星"拽弯"了！原本过蓝点中心的线，现在偏向了红星
> - 💡 **为什么**？因为 MSE 里有**平方**。一个误差 10 的点，贡献 100 到 MSE，但只贡献 10 到 MAE。
>
> **结论**：异常点多时，**改用 MAE**（不被异常值绑架），或提前清洗数据！

---

## 预期效果

- ✅ 直观看到：拖拽 → 数字变 → 线弯曲（三个维度同时更新）
- ✅ 强化记忆：「MSE 对异常敏感」从抽象公式 → 眼前的 20x 倍数
- ✅ 对比学习：baseline vs 异常，差异清晰
- ✅ 可复用：改参数快速看其他场景（预设），不需要写代码

---

## 进阶扩展（可选，P2）

1. **第二个异常点**：同时拖拽 2 个异常点，看合并效应
2. **Loss 曲线**：下方加 1D 抛物线，显示 MAE/MSE/RMSE 的形状差异
3. **离群值检测**：高亮标记"这 5 个点可能异常"（基于 3σ 规则）
4. **模型选择建议**：根据当前数据自动说"推荐用 MAE" or "RMSE 也行"

---

## 成功指标

| 指标 | 目标 | 验证方式 |
|---|---|---|
| 交互流畅度 | 拖拽响应 < 100ms | 启动 demo，拖 slider 无卡顿 |
| 指标准确度 | MAE/MSE/RMSE 与 sklearn 一致 | 手算 3 个点验证 |
| 视觉清晰度 | 三个指标柱一眼对比出倍数差 | 截图检查柱高度比例 |
| 学生反馈 | 用户能说出"MSE 对异常敏感"原因 | 看完 demo 后提问 |

---

**总结**：一个拖拽就能感受"MSE 被异常点平方放大"的交互式 demo，用数字 + 图表 + 线条同步变化，把抽象评估指标变成具体的"手感"。
