# Review · poly-degree-overfit demo 设计

> 审被：`design.md`（P1 输出） · 模式：4 维评审 · 结论分层：必修 vs 可选

---

## 总评

**PASS（带 4 项必修）**。整体方向正确：单滑块驱动 + 双视图联动 + 4 档预设，完全踩中 `_marimo-math-guide.md` 的"单通道输入 / 多视图破抽象 / 预设锚点"三原则。数据生成沿用 README 底稿（`seed=666` + `random_state=5`），数值落点已被验证（degree=2 时 train/test 都到 ~1.10，差距 < 0.02；degree=5 拉开到 ~0.5；degree=10+ gap > 1）—— **U 形会很明显**。

但有几个会影响"5 秒看懂"的细节没明确：U 形最低点没有专门标注、双视图 degree 同步性靠口头描述没固化、preset↔slider 的状态同步留了 TODO、degree=15 在 N=70 训练样本下数值稳定性可能有坑。

下面分维度给出。

---

## 维度 1 · 5 秒测试

**判定**：通过，但缺一个视觉锚点。

正面：
- 顶部一行 `degree={n} | train={x} | test={y} | gap={z}` + 三色背景（绿/黄/红）—— 这是好的"快读"信号。
- 双视图左右并排，标题里嵌 `degree={current_degree}`，用户扫一眼就知道当前在哪。

**必修 1**：右图 U 形曲线**必须**显式标注"测试 MSE 最低点"（即 bias-variance 甜蜜点）。当前设计只画了"当前 degree 的红点"，但当红点不在最低点时，用户没有参照系判断"应该往哪移动"。

建议方案：
- 在右图加一个**绿色钻石** mark（`mark_point(shape="diamond", size=200, color="#10b981")`）固定在 `argmin(test_mses)` 位置
- 配文字 annotation："最优 degree = {best_d}"

这样"红点 vs 绿钻石"的相对位置变成天然的进度条 —— 5 秒内能判断"过/欠/刚好"。

---

## 维度 2 · 单通道输入

**判定**：方向对，但 preset↔slider 同步留了悬念。

设计里 §3.2 说：
> "当用户选择 preset，滑块同步更新；当用户拖动滑块，preset 重置为'自由探索'"

§5.1 又只声明了 `degree = degree_slider.value`。`_marimo-math-guide.md` 反模式明确点名："滑块语义双重 → 必有一处违反直觉"。

**必修 2**：明确"唯一来源"。两条路二选一：

| 方案 A（推荐） | 方案 B |
|---|---|
| `degree = degree_slider.value`，preset 是"快捷按钮"组（4 个 `mo.ui.button`），点击时调用 `degree_slider.set_value(...)` | preset 用 `mo.state` + dropdown，slider 是 derived（每次 preset 变了重置 slider）|
| 简单，符合 guide 推荐的"reactive，不用 state" | 多了一个 state cell，但 dropdown 比 4 个按钮节省版面 |

推荐 A：guide 里 `_marimo-math-guide.md` §交互状态决策树明确说"教学要所见即所得，每个滑块改动应立刻体现在所有视图"，A 方案最直接。

伪代码（推荐 A）：
```python
preset_buttons = mo.hstack([
    mo.ui.button(label="欠拟合 (1)", on_click=lambda _: degree_slider.set_value(1)),
    mo.ui.button(label="刚好 (2)",   on_click=lambda _: degree_slider.set_value(2)),
    mo.ui.button(label="微过 (5)",   on_click=lambda _: degree_slider.set_value(5)),
    mo.ui.button(label="严重过 (12)", on_click=lambda _: degree_slider.set_value(12)),
])
```

---

## 维度 3 · 多视图

**判定**：双视图设置合理，但联动一致性需要再加一道"硬约束"。

正面：
- 左图（散点 + 拟合曲线）= 看"曲线长什么样"
- 右图（U 形折线 + 红点）= 看"误差量化"
- 两个视图都被同一个 `degree` 驱动 → 联动天然成立

**必修 3**：左图标题 `散点 + 拟合曲线 · degree={current_degree}` 和右图红点都依赖 `degree_slider.value`。需要保证它们**渲染在同一个 reactive 周期里**，否则会出现"左图 degree=5 但右图红点还停在 4"的撕裂。

实现要点：
- 左右图的 `mo.ui.altair_chart()` 要在**同一个 cell** 里 hstack 输出（设计 §7 末尾已经这样做了，但代码注释里没有强调原因——加一行注释说明）
- `train_mse / test_mse / gap` 的计算放在第 5 个 cell（设计里 `degree_slider, X_train, ...` 那个 cell），所有下游视图都从这里 fan-out —— 这就是 guide 推荐的"单一来源"

可选优化：
- 左图加**真实曲线**虚线（`y = 0.5x² + x + 2`，淡绿，dashed）。设计里 §4.1 提到了"参考线（可选）"——**强烈建议保留**。这是过/欠拟合的"真理基线"，用户能直接看到"红线偏离绿线多远"。

---

## 维度 4 · 反模式审查

逐条对照 `_marimo-math-guide.md` 反模式清单：

| 反模式 | 设计是否触犯 |
|---|---|
| 一上来 3D | ✅ 不触犯（双 2D 已足够） |
| 滑块语义双重 | ⚠️ 见必修 2 |
| 只有自由探索无 preset | ✅ 不触犯（4 档预设） |
| 一图塞 8 种 mark | ✅ 不触犯（左 2 种、右 3 种） |
| 中文 matplotlib 默认字体 | ✅ Altair 不需要配置 |
| `mo.state` + `run_button` 实时反馈 | ⚠️ 取决于必修 2 怎么实现 |

---

## 必修项汇总（4 条）

| # | 必修 | 影响维度 |
|---|---|---|
| 1 | 右图加"测试 MSE 最低点"绿钻石标记 + annotation | 5 秒测试 |
| 2 | 明确 preset↔slider 单一来源（推荐方案 A：preset = 按钮组调用 `set_value`） | 单通道输入 |
| 3 | degree 范围降到 **1-12**（见下方"特别关注"） | 数值稳定 |
| 4 | 左右图 hstack 在同一 cell 输出，避免渲染撕裂 | 多视图联动 |

---

## 特别关注（评审清单回应）

**Q1：滑块跨度 1-15 合理吗？**
不完全合理。N=70 训练样本下，`np.polyfit(degree=14)` 会出 RankWarning（条件数极差），degree=15 数值上可能完全失控（系数 1e10 量级）—— 视觉上变成"竖直毛刺"，**反而盖过 U 形的教学信号**。建议改为 **1-12**（degree=12 已经足够展示"剧烈抖动"，README 底稿用的也是 X^1~X^10）。如果坚持 15，必须在代码里 try/except RankWarning 并截断。

**Q2：双视图联动？**
设计里靠"同一个 `degree` 变量"实现，原理上没问题。落地见必修 4。

**Q3：U 形最低点如何标注？**
设计里**没有专门标注**。见必修 1，强烈建议加绿钻石。

**Q4：数据生成能否制造明显 U 形？**
能。`y = 0.5x² + x + 2 + N(0,1)`，true degree=2，noise 标准差 1.0 在 y 量级 ~10 上信噪比合适。README 底稿数值（degree=1: ~3.08, degree=2: ~1.10, degree=5: 1.41, degree=10+: ~1.8+）已经是 U 形且 gap 拉开。

**Q5：4 档预设覆盖完整 bias-variance 谱？**
`{1, 2, 5, 12}` 覆盖：高 bias（1）/ 平衡（2）/ 微 high variance（5）/ 极端 high variance（12）。覆盖完整。**唯一建议**：把"刚好 (2)"做成默认 preset，让用户打开页面**第一眼就看到"理想状态"**作为锚点。

---

## 可选建议（不阻塞实现）

1. **真实曲线虚线**：左图叠淡绿 dashed `y = 0.5x²+x+2`。提供"真理基线"，强化"红线偏离 = 拟合误差"的直觉。
2. **degree=1 时的轴范围**：直线在 x∈[-3,3] 下 y 值可能跑到 [-2, 5]，但散点 y 范围可达 [-6, 12]，注意 y 轴 fix 到 `[-6, 12]`（设计 §4.1 已写明，确认实现时不被 Altair auto-scale 覆盖）。
3. **数值精度**：`mse` 显示用 `.3f` 而非 `.4f`（4 位小数对教学没增量信息，反而增加噪声）。
4. **gap 阈值的合理性**：设计里 `gap < 0.2 / 0.5` 阈值是 README 底稿数值线性外推的，落地后用 degree=1~12 的实际 gap 分布校准一下（可能要调到 `0.15 / 0.4`）。
5. **左图 x 轴**：散点用 `X_train` 不显示 `X_test` 是合理的（避免视觉混乱），但可以在标题或 caption 注明 "（仅显示训练集 70 样本）"。
6. **Preset"刚好"做默认值**：`degree_slider` 默认 `value=2`，第一眼看到的就是 train/test 几乎重合的"理想状态"。设计里已经默认 2，确认保留。

---

## 实现风险预警

- `np.polyfit` 在 degree ≥ 10 + N=70 训练样本时会触发 `np.RankWarning: Polyfit may be poorly conditioned`。建议用 `warnings.filterwarnings("ignore", category=np.RankWarning)` 屏蔽（教学 demo 不需要警告噪声），或者直接换 `sklearn.preprocessing.PolynomialFeatures + LinearRegression` 流水线（数值更稳）。
- 设计里两次出现 `random_state=5`（§7 数据生成）和未明确（§2 只说"固定随机种子"）—— 落地时确认 train_test_split 用 `random_state=5`（与 README 底稿数值对齐），不要随手改。

---

**结论**：PASS · 必修 4 项 · 可选 6 项

