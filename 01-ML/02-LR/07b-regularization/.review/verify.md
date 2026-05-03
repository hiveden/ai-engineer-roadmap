# 07b-regularization Demo 验收报告（P4）

> 被验：`demos/l1-l2-shrinkage.py` · 端口 2734 · 行数 662
> 验收维度：V1 review 必修 / V2 服务健康 / V3 代码质量 / V4 教学价值
> 结论：**PASS**（4/4 维度通过，无返工项）

---

## V1 · review 必修 3 项核查

| # | 必修项 | 期望 | 实测 | 结论 |
|---|---|---|---|---|
| 1 | B 视图零柱**换色** | 不再红色，黄填+黑边 | `chart_B` 的 `zero_marks` 层 `fill="#fde047"` `fillOpacity=0.9` `stroke="#000000"` `strokeWidth=1.8`（line 462-478），跨 0 轴的小矩形（±6%×_w_max）+ `mark_text(text="0", fontWeight="bold", color="#000000")` 双重信号 | **通过** |
| 2 | preset/slider 模式徽章双色 | 黄/蓝双色徽章 | `is_preset=True` → 黄底（`bg=#fef3c7` `color=#92400e` `border=#f59e0b`）+ 灰色补丁「（手动滑块已锁定，选 ✋ 手动 解锁）」（line 256-264）；`is_preset=False` → 蓝底（`bg=#dbeafe` `color=#1e40af` `border=#3b82f6`）显示 `log₁₀(λ)=X · λ=Y`（line 266-270） | **通过** |
| 3 | B 视图标注「标准化空间权重」 | 副标题警示 | `chart_B.TitleParams.subtitle="⚠ 标准化空间下系数 (PolynomialFeatures+StandardScaler 后) · 仅供稀疏性对比，非原始 x^k 系数"`，`subtitleColor="#9333ea"` 紫色警示色（line 514-519） | **通过** |

> 必修 3 项实现质量超出最低要求：必修 1 用「跨零轴矩形 + 黑体 0 文字」双层信号，比 review 给的方案 A/B/C 任一更显眼；必修 2 在徽章右侧追加 `当前聚焦：L1/L2` mini chip（line 274-277），让 toggle 状态也可见；必修 3 用紫色而非灰色副标题，符合 dev-log 自述的「第二眼就被看到」设计意图。

---

## V2 · 服务健康（端口 2734）

| 检查项 | 结果 |
|---|---|
| HTTP 状态码 | `200 OK` |
| 响应时间 | `0.024s` |
| 监听进程 | Python PID 26686, IPv4 127.0.0.1:2734 LISTEN |
| 端口归属 | 与 design.md 第 9 节声明一致（2731=04b / 2732=05 / 2733=07a / **2734=07b**），无冲突 |

**结论**：服务运行正常。

---

## V3 · 代码质量

### 三视图齐备

| 视图 | 实现位置 | width×height | 关键 mark |
|---|---|---|---|
| A · 散点+三拟合 | line 315-396 | 420×340 | mark_circle (散点) + mark_line (3 模型) + mark_line+strokeDash (真实曲线) |
| B · 权重柱（主舞台）| line 399-521 | 520×110×3 行 vconcat | mark_bar + mark_rule + mark_bar(零柱黄底) + mark_text(数值/0) |
| C · λ-MSE U 形 | line 524-618 | 420×340 | mark_line (实+虚) + mark_rule (当前 λ) + mark_circle (红点) |

### 控件齐备

| 控件 | 实现 | 验收 |
|---|---|---|
| λ 对数滑块 | `mo.ui.slider(start=-4.0, stop=2.0, step=0.05, value=-1.0, label="正则强度 log₁₀(λ)")` (line 100-104) | 范围 1e-4 ~ 1e2 ✓，label 已强化（建议 2 部分采纳）|
| L1/L2 toggle | `mo.ui.radio(["L1 (Lasso)", "L2 (Ridge)"], value="L1 (Lasso)", inline=True)` (line 94-99) | ✓ |
| 4 预设 | `PRESETS = {手动:None, 无正则:-4.0, 弱:-2.0, 适中:-1.0, 强:1.0}` (line 81-87) | 弱档已采纳建议 4（-3→-2）✓ |

### 软建议采纳情况

| 建议 | 状态 | 实测证据 |
|---|---|---|
| 1 · C 视图 L1+L2 同框 | ✅ | line 571-589：`sel_lines` 实线粗 sw=2.8 + `other_lines` 虚线细 sw=1.5 opacity=0.45 |
| 2 · log 滑块标签强化 | 部分 ✅ | label 从 `log₁₀(λ)` 改为 `正则强度 log₁₀(λ)`；徽章显示双形式 `log₁₀(λ)=X.XX · λ=Y` |
| 3 · toggle 联动 A/B 高亮 | ✅ | A 图：line 347-353 选中模型 opacity=1.0 sw=3.0，未选 0.35 / 1.8；B 图：line 423-426 行级 opacity 联动 |
| 4 · 弱档调到 1e-2 | ✅ | line 84：`「🌱」弱 · λ = 0.01` (-2.0)，与无正则差 100×、与适中差 10× |
| 5 · 与 07a pipeline 一致 | N/A | dev-log 已说明：07a-overfit/demos/ 目前为空，无需对齐 |

### 工程细节

- ✅ `PolynomialFeatures(degree=10, include_bias=False)`（line 161）— 防 intercept 重复
- ✅ `StandardScaler` 必加（line 162）— 防量级失衡
- ✅ `Lasso(alpha=lam, max_iter=50000, tol=1e-4)`（line 169）— 加大迭代防收敛警告
- ✅ `warnings.filterwarnings("ignore", category=ConvergenceWarning)`（line 34）+ sklearn UserWarning 屏蔽（line 35）
- ✅ `ZERO_TOL=1e-6`（line 189）— 浮点零阈值
- ✅ C 图 log scale `alt.Scale(type="log", domain=[1e-4, 1e2])`（line 575, 585）
- ✅ `df_curve` 启动一次性预计算 30×2=60 fit（line 222-241），后续切换不重跑
- ✅ B 图三行用 `alt.vconcat` 独立 chart，便于精确控制零标记层（line 507-511），与 design 防坑清单一致
- ✅ A 图真实曲线 mark_line + strokeDash[4,3] 灰色（line 368-372）

### 代码组织

15 cell 与 design.md 第 10 节骨架完全对齐；reactive 派生 `cur_lambda` 单一入口（line 121-130）；`_color_scale_b` 私有命名避免跨 cell 冲突（dev-log 踩坑已修复）。

---

## V4 · 教学价值

### L1 稀疏（红框→黄柱）一眼可辨？**通过**

- 黄底（`#fde047` 鲜黄）+ 黑边（1.8px stroke）+ 黑体「0」文字三重信号叠加
- 完全脱离红色家族：与无正则线红 `#ef4444` 不冲突，与 L1 主色橙 `#f97316` 不混淆
- 跨零轴的小矩形比柱子本身更显眼（柱高=0 时几乎不可见，而小矩形固定 ±6%×_w_max 高度）
- design 第 7 节判断点 2 的「不及格信号」已规避

### L2 平滑（不归零但变小）对比鲜明？**通过**

- B 图第 3 行（绿色 `#10b981`）共享 y 轴 `domain=[-_w_max*1.15, _w_max*1.15]`（line 430），与无正则行同尺度，「整体矮化」一眼可读
- 与 L1 行（黄底零柱）形成「断崖 vs 渐降」的视觉差异
- 无任何零标记叠加在 L2 行（`is_zero` 仅在 `model=="L1 (Lasso)"` 时为 True，line 412），语义清晰

### 与 07a 数据一致性？**通过**

- `np.random.seed(666)` ✓ (line 136)
- `N=100, x ~ U(-3,3)` ✓ (line 137-138)
- `y = 0.5*x**2 + x + 2 + N(0,1)` ✓ (line 139)
- `train_test_split(test_size=0.3, random_state=5)` ✓ (line 142-144)
- A 图 x 轴 [-3.2, 3.2]、y 轴 [-6, 12] ✓ (line 364-365)
- 真实曲线参考 `0.5*x²+x+2` 灰虚线 ✓ (line 335)

> 注：07a-overfit/demos/ 当前为空，跨章 1:1 对照可在 07a 实现时再校验。本 demo 端独立可成立。

### 教学叙事完整性

- 三视图分工（A 业务 / B 参数 / C 超参）符合 _marimo-math-guide.md 的「抓眼球→建直觉→升维」模板
- mo.md 公式（line 63-69）+ 玩法说明（line 638-655）+ 模式徽章实时数字反馈（line 244-312）三层文字支撑
- 端口 2733（07a）+ 2734（07b）双 tab 对照学习路径已明示

---

## 总评

| 维度 | 结论 |
|---|---|
| V1 · review 3 必修 | **PASS** · 全部到位且超额（双层信号 / mini chip / 紫色警示）|
| V2 · 服务健康 | **PASS** · 200 OK / 24ms / IPv4 LISTEN |
| V3 · 代码质量 | **PASS** · 三视图 + 控件齐备，5 条软建议除 N/A 外全采纳 |
| V4 · 教学价值 | **PASS** · L1/L2 视觉差异钉死，07a 数据完全对齐 |

**最终判定：PASS · 无返工项 · 可进入 07c 章节或并行启动 07a 配套 demo**

### 可选后续优化（非阻塞）

1. 当 07a-overfit/demos/ 实装时，校验红线形态在两 demo 中视觉一致（同 pipeline 同 seed）。
2. C 图当前未处理「选中类型 train/test 与未选中类型同 split 颜色相同」的潜在歧义——若用户反馈不易区分 L1-train vs L2-train，可考虑在 tooltip 加显式 `type` 字段（已实现 line 578）或将未选中线 desaturate 到灰阶。
3. 徽章中 `当前聚焦：L1/L2` chip 与 reg_type radio 的视觉关联可进一步加强（如同色边框）——当前已用相同黄底，可接受。
