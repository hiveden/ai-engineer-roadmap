# Demo 制作指南（skill）

> **Step 1** · marimo demo 设计 + grid 布局 + 视频录屏友好

---

## 🚨 必读 · ML 章节所有 demo = 录屏 demo

**ML 教程的所有 demo 都进 B 站视频管线**（KNN / LR / LogReg / DT / KMeans / NB 全部 12 期视频章）。**没有"非录屏 demo"选项**——除非整章不出视频（极少）。

**开工前必先读 [`_5-layout-guide.md`](./_5-layout-guide.md)**，特别 §10 顶部"🚨 强制流程"段：

1. **抄骨架**：参考已改造好的 demo（如 `02-LR/01-intro/demos/k-to-w-migration.py` + `custom.css` + `layouts/*.grid.json`），不要重新发明
2. **打印 §10 11 项 closing gate** 到工作记忆顶
3. **每改一处 grid.json / chart properties / label** → playwright 截图自检
4. **不让 user 当 QA**——交付前自己确保 11 项全 ✓

**典型违规模式（avoid）**：
- shot dropdown / narration / 提示卡 塞进录屏区（违反 _5 §3）
- sidebar 5+ col 因为"中文 dropdown 选项长"（违反 _5 §4，应改选项不改 sidebar）
- label 带中文括号 "切分特征（X1 还是 X2）"（违反 _5 §4 极简 label）
- css_file 用 `marimo.css`（旧）而非 `custom.css`（_5 标准）

**本指南（`_2`）的 §0-§9 是 demo 设计层面的基础知识**（5 原则 / cell 拆分 / mo.md 陷阱 / grid 配置基础），适用于所有 demo；**录屏专属规则全在 `_5`**（录屏分辨率 / Strategy A vs B / sidebar 极简 / 字体覆盖 / 失败模式表 / 11 项 closing gate）。

> **不要只看本文写完 demo 就交付**——必须用 `_5 §10` 的 closing gate 做最终检查。

---

## 何时用

- 写新 marimo demo（控件 + chart + 数据卡）
- 调 grid layout（拆 cell / 写 `layouts/XX.grid.json`）
- 录屏前最后清单（§9）
- mo.md 不显示 / chart 缩豆 / 控件溢出 等问题排查

## 索引（按需取读）

| § | 内容 | 何时读 |
|---|---|---|
| **0** | **设计五原则**（单核心隐喻 / 渐进披露 / linked views / 5 秒测试 / 几何>颜色块） | **设计新 demo 前必读** |
| 1 | Cell 拆分原则（grid 友好）+ chart 自适应坐标系 | 写新 demo |
| 2 | mo.md 渲染陷阱（if/else 不显示）| mo.md 没出现 |
| 3 | mo.ui.table vs mo.md table | 决定表格样式 |
| 4 | Grid Layout 配置（含 ASCII → JSON）| 写 layout 文件 |
| 5 | ASCII 布局参考 cell | 末尾占位 cell |
| 6 | 录屏端口约定（2718 edit / 2750+ run）| 启 marimo 前 |
| 7 | Demo 制作 8 步 | 流程速查 |
| 8 | 失败模式 | 遇到问题时 |
| 9 | 验证清单 | demo 完工自检 |

> 录屏自动化细节（Playwright / ffmpeg / highlight）→ 见 [`_6-recording-guide.md`](./_6-recording-guide.md)

---

## 0. 设计五原则（设计新 demo 前必读）

> 来源：3blue1brown / Distill / Bret Victor / setosa.io / immersivemath / R2D3 调研（详见 `02-LR/03b-math/.review/demo-design-research.md`）。
> Cell 拆分（§1）和 Grid 布局（§4）解决"怎么摆"，本节解决"摆什么 / 为什么这样摆"。

### 原则 1 · 单核心隐喻

一个数学概念**只用一种主隐喻**讲透，其它视图都服务这个隐喻。

数学概念 → 主隐喻候选（任选其一为主）：
- 偏导 / 梯度 → **几何**（切线斜率 / 切片曲线）
- 矩阵乘 → **几何变换**（向量空间被拉伸/旋转）/ **数据流**（行=样本 列=特征）/ **代数**（行点列）—— 三选一，不混
- 概率分布 → **采样直方图** / **密度曲线**
- 损失函数 → **景观**（landscape）/ **残差几何**

> ❌ 反模式：5 视图轰炸 = 5 个独立隐喻（曲线 + 等高线 + 切片 + 3D 表面 + 颜色块），用户每切一视图就要重建心智模型，认知负载×5。
> ✅ 正模式：3b1b 偏导只用「2D 等高线 + 切线」一种几何隐喻，多视图都是它的镜像。

### 原则 2 · 渐进披露（progressive disclosure）

默认露出**最少视图 + 最少控件**。"想看更多"由用户主动展开（accordion / tab）。

| 展开方式 | 适合场景 |
|---|---|
| 默认主视图 + accordion 折叠次视图 | 主+辅 关系明确 |
| Tab 切换平行视图 | 多个等价视角 |
| 滑块的"显示高级参数"开关 | 控件超过 5 个 |

> ❌ 反模式：偏导 demo 默认同屏摆 5 视图（1D + 2D + 切片×2 + 3D），每个都很小，每个都看不清。
> ✅ 正模式：默认 2D 等高线 + 切线（两图 linked），accordion 提供 3D 验证视图（用户点开才渲染）。

### 原则 3 · Linked views 服从主叙事

多视图必须**互为镜像**——同一份数据、同一组参数的两种坐标表达。视图间状态联动（拖动一个，另一个同步）。

> ❌ 反模式：partial-derivative 视图 1 用 `g(x) = (x-2)² + 1`，视图 2 用 `f(x,y)`——两个不同函数对象，"同源"破缺，用户无法把视图 1 的切线对应到视图 2 的等高线。
> ✅ 正模式：视图 1 = `f(x, y_cur)` 沿 x 切片（一维曲线 + 切线），视图 2 = `f(x,y)` 等高线（同函数二维），切线斜率 = ∂f/∂x。拖 y_cur 滑块两图同步动。

### 原则 4 · 5 秒测试单焦点

**每次交互回答一个具体问题**。控件越少越好；多个控件 → 拆成多个 demo（不是塞一个 demo 里）。

5 秒测试：陌生用户进入 demo，5 秒内能否说出"这个 demo 在演示什么"？

> ❌ 反模式：matrix-mul-viz 9 个控件混在一起（A矩阵 4 数 + 输入向量 2 数 + 切换器 3 个），用户不知道先动哪个。
> ✅ 正模式：拆 3 子 demo——「线性变换」（拖 A 列向量看网格变形）/「点积几何」（两向量看投影）/「数据视角」（X·w 行点列）。每个 ≤ 4 控件。

### 原则 5 · 几何 > 颜色块（矩阵乘专项）

教矩阵乘时**几何变换隐喻 > 颜色块行点列**——但教 LR 数据视角（X·w）时颜色块 OK，需声明"现在切换到数据视角"。

| 阶段 | 推荐隐喻 | 理由 |
|---|---|---|
| 矩阵乘**入门** | 几何变换（A 把单位圆变椭圆） | 直觉强，3b1b/immersivemath 范式 |
| 点积**入门** | 投影几何（cosθ + 模长） | 看得见"对齐程度" |
| LR **数据视角** | 颜色块行点列（X·w） | 行=样本 列=特征 是 LR 自然语言 |
| 神经网层**计算** | 颜色块（W·x + b） | 逐元素跟踪计算 OK |

> ❌ 反模式：第一个矩阵乘 demo 直接上颜色块行×列点积（脱离几何意义）。
> ✅ 正模式：先几何变换 demo（A 拉伸网格）→ 点积投影 demo → 再过渡到「数据视角的颜色块」（明确切换隐喻）。

### 自检：开做 demo 前对照

- [ ] 这个 demo 的**主隐喻**是什么？（一句话能说清）
- [ ] 默认状态露出几个视图？> 3 个 → 砍或折叠
- [ ] 多视图是否同源？（同一函数 / 同一组参数 / 同一份数据）
- [ ] 控件数 > 5？→ 考虑拆 demo
- [ ] 5 秒内能说出"这是演示 XX"吗？
- [ ] 矩阵乘 demo：第一视图是几何变换吗？

---

## 1. Cell 拆分原则（grid 友好）

**一个 cell = grid 最小可摆放单元**。组合 cell 锁死布局灵活性。

| 原则 | ❌ 反例 | ✅ 正例 |
|---|---|---|
| 图表独占 cell | `mo.vstack([mo.md("标题"), chart])` | chart 单独 cell + 标题独立 mo.md cell |
| 标题/状态卡独占 mo.md cell | 把 `LOOCV %` 拼进表格段 markdown 标题 | 独立 mo.md cell（突出关键状态）|
| chart title 抽出 | `chart.properties(title="k=11 · ...")` | chart 不带 title + 独立 mo.md cell 写图说明 |
| 控件可合并 | 4 滑块拆 4 cell（无意义） | 一行 hstack widths=[1,1,1,1] 合一 cell |
| 不要嵌套 vstack/hstack | `hstack([vstack([标题, k]), vstack([标题, rating])])` | 一行 4 个组件直接 hstack |
| **多 tab 内容拆多 demo** | `mo.ui.tabs({1:tab1, 2:tab2, 3:tab3})` 让整个 tabs 在 grid 视图下变一块拆不开 | 每 tab 独立 demo + 独立 layout（如 `01-cv-folds.py / 02-gridsearch.py / 03-digits.py`） |
| **chart 自适应坐标系**（防小尺度缩豆）| 固定 `alt.Scale(domain=[0,10])` 在交互滑动到小尺度时形状缩成 5% | `_W = max(d_*) · 1.25` + `domain=[a−W, a+W]` + 所有 layer `clip=True`（多 chart 协同时统一算法保视觉对齐）|

## 2. mo.md 渲染陷阱

`mo.md(...)` 在 if/else 分支里直接调用**不显示**——必须赋给变量后单独写表达式作为 cell 末行：

```python
# ❌ 不显示
if cond:
    mo.md("...")
else:
    mo.md(...)

# ✅ 显示
if cond:
    _result = mo.md("...")
else:
    _result = mo.md(...)
_result   # cell 末行表达式 = cell 输出
```

marimo cell 输出 = 最后一个表达式（或赋值的值）。

## 3. mo.ui.table vs mo.md table

| 维度 | `mo.ui.table` | `mo.md` table |
|---|---|---|
| 互动 | Search/Explore/Export/翻页 | 无 |
| 行高 | ~40px（含装饰） | ~30px |
| 视频友好度 | 低（多余 UI 干扰） | **高**（紧凑） |
| 适合场景 | 互动 demo / 大数据集 | 视频录屏 / 小固定数据 |

**双 cell 共存策略**：互动表 + 紧凑表两个 cell 都写，grid 里录屏时只显示紧凑版（互动版 position=null）。

减小 `mo.ui.table` 视觉占用：
- `page_size=5`（默认 10）
- `show_column_summaries=False`（关 type 提示行）
- 删冗余列（如长 string 列）

## 4. Grid Layout 配置

> **录屏类 demo（含 shot 切换 / 双槽对照）**：直接看 [`_5-layout-guide.md`](./_5-layout-guide.md)，本节是基础 demo 的 grid 通用知识。

### 4.1 绑定到 demo

```python
app = marimo.App(
    width="medium",
    layout_file="layouts/XX-name.grid.json",
    css_file="marimo.css",
)
```

### 4.2 maxWidth 选择

| 比例 | maxWidth | 总高（行 × 20px） | 推荐场景 |
|---|---|---|---|
| 2:1 | 1200 | 任意 | 决策边界类 demo（02-proximity 风格） |
| **严格 16:9** | **1280** | **36 = 720px** | **录屏一屏满 · 2×2 主体网格**（e09）|
| 16:9（宽松）| 1280 | >36 | 内容多需滚动，但视频窗口仍 16:9 |
| 1.4:1 | 1400 | 任意 | 信息密度高的 demo（03-k-tuning 风格） |

`rowHeight` 默认 20px；`columns` 默认 24。**严格 16:9 = 36 行**，超过则失去原生匹配（视频上下留黑或需 crop）。

#### 4.2.1 横屏 2×2 主体网格（e09 案例）

8+ 内容 cell 想塞进 36 行：左右两列上下两行 = 4 个等大 panel + 顶部标题 / 控件 / 预测卡：

```
0  ┌── 标题 h=2 全宽 ─────────────┐
2  ├ 控件1 h=4 ┬ 控件2 h=4 ───────┤
6  ├── 预测卡 h=2 全宽 ───────────┤
8  ├ 视觉 (左 12) ┬ 数字 (右 12) ─┤  ← 第 1 行 panel · h=14
22 ├ 参数 (左 12) ┬ 对比 (右 12) ─┤  ← 第 2 行 panel · h=14
36
```

cells 数组 y 顺序保持单调（8→8→22→22）→ 调布局心智零负担。叙事配对：每行左视觉 / 右数字。

### 4.3 直接生成 layout JSON（不用全靠手拖）

按 ASCII 布局先写 `layouts/XX-name.grid.json`：

```json
{
  "type": "grid",
  "data": {
    "columns": 24,
    "rowHeight": 20,
    "maxWidth": 1280,
    "bordered": true,
    "cells": [
      {"position": null},                  // 隐藏（imports / 计算 cell / accordion）
      {"position": [0, 0, 24, 3]},         // 全宽 60px 高
      {"position": [0, 9, 17, 3]},         // 左 ~70%
      {"position": [17, 9, 7, 3]}          // 右 ~30%
    ]
  }
}
```

cells 数组顺序 = demo 文件中 `@app.cell` 出现顺序。`position: null` = 不显示。

格式：`[x, y, width, height]` 单位是 grid units。

## 5. ASCII 布局参考 cell（开发用）

> **新录屏类 demo** 推荐用 [`_5-layout-guide.md`](./_5-layout-guide.md) §6 的 CSS 红虚线边框替代 ASCII 参考 cell——前者所见即所得，后者还要切回去看。本节适合非录屏 demo / 早期 demo（e01-e11）。

每个 demo 末尾加一个 mo.md cell 内嵌 ASCII 布局图，作为开发参考（grid position=null 录屏时隐藏）：

```python
@app.cell
def _(mo):
    # 📐 Grid 布局参考（开发用 · 录屏隐藏）
    mo.md("""
## 📐 Grid 布局（16:9 · 录屏推荐）

```
   0           14          24
0  ┌────── 标题（h=3）────────┐
3  ├────── 数据信息卡（h=3）──┤
6  ├────── 4 控件（h=3）──────┤
9  ├──── 预测卡 ────┬─状态─┤
...
```
    """)
    return
```

收益：
- 写 layout JSON 时有视觉参考
- 后续调整布局时不用反复看 grid 视图

## 6. 录屏端口约定

| 端口 | 用途 |
|---|---|
| 2718 | `marimo edit`（人工调布局，**勿动**） |
| 2750+ | `marimo run`（录屏专用，每个 demo 独立端口） |
|       | 2750 = 02-proximity / 2751 = 03-k-tuning / 2752 = 04-regression / 27XX 递增 |

录屏 Agent 用 `marimo run` 不用 `marimo edit`——edit 有侧边栏污染画面。

## 7. Demo 制作 8 步

1. **写代码逻辑**（数据 + 计算 + UI 组件）
2. **按 §1 拆 cell**（每图表/标题独占）
3. **加 §5 ASCII 布局参考 cell**（末尾 · 录屏 null）
4. **生成 `layouts/XX.grid.json`**（按 ASCII 写或 marimo edit 拖）
5. **demo 顶部绑定 `layout_file=`**
6. **`marimo run XX.py --headless --port 27XX --no-token`** 启动
7. **Playwright 截图验证**（颜色 / 布局 / 比例）
8. **micro-tweak**：拖 grid 微调，自动写回 JSON

## 8. 失败模式

| 失败 | 教训 |
|---|---|
| Altair 多层 chart 颜色冲突（绿/红变橙/灰） | 加 `.resolve_scale(color='independent')` |
| if/else 里 `mo.md(...)` 不显示 | 赋值给 `_var` + cell 末行表达式 |
| `chart.properties(title=...)` 锁死布局 | title 抽出独立 mo.md cell |
| `mo.ui.table` 视频里太占空间 | 用 mo.md 紧凑版（双 cell 共存）|
| grid 布局调好但 demo 没绑定 | demo 顶部 `app = marimo.App(...layout_file=...)` |
| 录屏看到 marimo edit 侧边栏 | 用 marimo run 不用 edit |
| 嵌套 vstack 让 grid 拆不开 | 一层组合最大（hstack 一行 4 控件 OK；hstack 内套 vstack 不 OK） |
| 4 个滑块在 2×2 网格里 | 改一行 hstack widths=[1,1,1,1] |
| 录屏脚本 switch 索引错位 | `marimo-slider` 和 `marimo-switch` 是两个独立集合，索引互不重叠（不要把 switch 当 slider 数） |
| 录屏比 TTS 短 ~300ms | ffmpeg `-sseof -{int(T_END)}` 截掉小数，用 `-sseof -<float>` 或 ffprobe + `-ss / -t` |
| 交互到小尺度 chart 形状缩豆 | 固定 domain=[0,10] 不适配 → 自适应 `domain=[a±W, W=max(d_*)·1.25]` |
| state 1 静帧形状被 [0,10]² 裁切 | 改 demo 默认值（如 A=(2,3) → A=(5,5)）让初始状态形状完整可见，比改渲染逻辑成本更低 |
| 控件嵌套 vstack 撑高 cell → 录屏溢出顶部 | 平铺单层 `hstack([s1,s2,s3,s4,s5])`；中文长 label 用 `widths=[1,1,1,1,1.4]` 防换行 |
| Altair Y categorical 默认字典序，渲染顺序与口播假设不一致（e11a 口播写"上方红色"但实际红在下方，因 `"CV-N 平均"(C) < "单次..."(中文)`） | chart 写完先 Playwright 截图确认上下顺序再写"上方/下方"文案；或 `alt.Y(sort=[...])` 显式指定 |
| `mo.ui.slider` 默认 `show_value=False` → 录屏看不到 k=3 / 身高=175 等具体值 | 所有 slider 显式 `show_value=True`（哪怕 demo 演示也用得上）|
| 控件密集行（cell h=4 槽位）默认 marimo widget 太大 → label / track / thumb 占视觉权重过多 | marimo.css 加 selector：`label{font-size:11px}` / `[data-orientation="horizontal"].relative.flex{width:96px}` / `[role="slider"]{height:12px;width:12px}` / `[role="switch"]{height:18px;width:32px}` |
| label 太长撑高 cell（"注入异常值 (体重 250kg)"）→ widget 换行 | 短语 + 单位放 label 末尾："异常值 250kg" / "身高" / "体重" |
| 多视图同屏轰炸（5 视图 / 8 控件）认知载荷爆 → 用户每切一视图重建心智 | §0 原则 2 渐进披露：默认 2 视图 + accordion 折叠次视图；超 5 控件拆 demo |
| 视图间用不同函数对象（视图 1 `g(x)` / 视图 2 `f(x,y)`），切线对应不上等高线 | §0 原则 3 linked views 同源：所有视图共享同一函数 + 同一参数滑块 |
| 矩阵乘 demo 第一视图直接上颜色块行点列，脱离几何意义 | §0 原则 5：先几何变换（A 拉伸网格）→ 点积投影 → 再过渡数据视角颜色块（声明"切换隐喻"） |

## 9. 验证清单

新 demo 录屏前过一遍：

**设计层（§0 五原则）**：
- [ ] 一句话能说出主隐喻（几何 / 代数 / 数据流任一）
- [ ] 默认视图 ≤ 3，次要视图放 accordion
- [ ] 多视图同源（同函数同参数同数据）
- [ ] 控件 ≤ 5（超过则拆 demo）
- [ ] 5 秒测试：陌生用户能说出"这是演示什么"
- [ ] 矩阵乘 demo：第一视图是几何变换不是颜色块

**实现层（§1-6）**：
- [ ] 每个 chart 在独立 cell（无 vstack 包裹）
- [ ] chart 不含 `properties(title=...)`（title 抽到独立 mo.md cell）
- [ ] 状态卡 / 预测卡 各独立 mo.md cell
- [ ] 控件最多一层 hstack（无嵌套）
- [ ] Altair 多层加 `.resolve_scale(color='independent')`
- [ ] mo.md 在 if/else 里赋给 `_var` + cell 末行表达式
- [ ] demo 顶部绑定 `layout_file`
- [ ] `layouts/XX.grid.json` 存在
- [ ] `marimo run XX.py --headless --port 27XX --no-token` 跑得通
- [ ] Playwright 截图看颜色 / 布局正常
