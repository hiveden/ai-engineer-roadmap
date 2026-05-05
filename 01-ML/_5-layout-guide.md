# Step 5 · 调 demo 录屏布局

> 把 demo 调成 16:9 录屏可用的样子。**人工 gate**：每期开拍前必跑。

## 何时用

- 新 demo 写完，进入 Step 5（调比例）
- 已有 demo 但镜头脚本变了（如新增双图对比镜头）
- 录屏发现尺寸/字号不合适

如已有 layouts.json 且 demo 未变 → **跳过本节**。

## 索引（按需取读）

| 想做的事 | 跳到 |
|---|---|
| 选录屏分辨率 | §1 |
| 设计镜头切换（单图 vs 双图） | §2 |
| 摆放控件 / 文字 / 提示 | §3 |
| 缩 sidebar / 极简 label | §4 |
| 改字体（盖宋体）| §5 |
| 标录屏边界（开发期）| §6 |
| 排查布局 bug | §7 |
| 录屏 ffmpeg 命令 | §9 |
| 调完前自检 | §10 |
| edit / run 模式 DOM 差异（CSS selector 失效）| §11 |

---

## 1. 录屏分辨率（B 站 16:9）

**录屏 1280×720** + ffmpeg 升采样 1.2× 到 1920×1080。

```bash
ffmpeg -i raw.mov \
  -vf "crop=1280:720:0:0,scale=1920:1080:flags=lanczos" \
  out.mp4
```

为什么不直接录 1920×1080：
- marimo 默认组件按 ~1280px 设计，1920 容器下控件被拉变形（slider track 太长拖动失精度，font 比例失调）
- 1.2× 升采样几乎无糊（lanczos 算法）

**总画布**：1280 wide × ~960-1000 high。
- 顶部 1280×720 = 录屏区
- 720 之下 = **录屏外提示区**（主播看，ffmpeg crop 不抓）

---

## 2. 镜头策略：Strategy A vs B

### Strategy A · 单图镜头（chart 数量随 shot 变）

每个 shot 主区图数量不一样：
- 单图镜头：chart 占满 ~1100×600
- 双图镜头：每图缩到 ~545×600

**适用**：镜头之间画面变化大、想用尺寸跳变制造视觉转场（电影感强）。

### Strategy B · 永远双槽（推荐 LR 类对照教学）

主区始终 2 个视觉槽，按 shot 切换内容：

| 镜头 | slot 1 | slot 2 |
|---|---|---|
| A 一元起步 | chart_1d 拟合直线 | 残差表（数据 + MSE）|
| B 升级二元 | chart_2d 决策面 | 公式动态展开（代值算 ŷ）|
| C 立体视角 | fig_3d 平面 | chart_2d 同模型俯视 |
| D 同构对比 | chart_1d | chart_2d |

**好处**：
- 画面稳定（镜头切换不变骨架，观众认知负荷低）
- 教学密度翻倍（每屏双视图对照）

**反例 / 何时不用**：
- 镜头讲单一概念，加副视图反而干扰（如纯讲数学推导，slot 2 留空更好）
- 副视图凑不出有意义的对照（强行塞 = 视觉噪音）

### 实现

```python
@app.cell
def stage(chart_1d, chart_2d, fig_3d, mo, shot, ...):
    if shot.value.startswith("A"):
        slot1 = mo.ui.altair_chart(chart_1d)
        slot2 = mo.md(...)  # 残差表
    elif shot.value.startswith("B"):
        slot1 = mo.ui.altair_chart(chart_2d)
        slot2 = mo.md(f"...{w1.value}...")  # 公式代值
    elif shot.value.startswith("C"):
        slot1 = mo.ui.plotly(fig_3d)
        slot2 = mo.ui.altair_chart(chart_2d)
    else:  # D
        slot1 = mo.ui.altair_chart(chart_1d.properties(width=400, height=440))
        slot2 = mo.ui.altair_chart(chart_2d.properties(width=400, height=440))
    mo.hstack([slot1, slot2], gap=0.5, widths="equal")
```

D 镜头 chart properties 显式覆盖 width — 因为 chart_2d 默认带 color legend 占额外 80px，hstack widths='equal' 会 X scroll。

### shot 联动 effective_X 模式（推荐 · 单一控件驱动一切）

避免 shot dropdown + button 同时切相同语义内容（重复设计 → 主播心智负担）。**单一 shot dropdown 同时驱动 chart / panel / title / narration**：

```python
# 1. shot dropdown（含"E · 自定义"打开 sidebar slider）
shot = mo.ui.dropdown(
    options=["A · 欠拟合", "B · 刚好", "C · 微过", "D · 严重", "E · 自定义"],
    value="A · 欠拟合", label="🎬 镜头",
)

# 2. effective_degree cell：A/B/C/D 锁定，E 用 slider
@app.cell
def _(degree_slider, shot):
    _PRESET = {"A · 欠拟合": 1, "B · 刚好": 2, "C · 微过": 5, "D · 严重": 12}
    effective_degree = _PRESET.get(shot.value, int(degree_slider.value))
    return (effective_degree,)

# 3. 下游全部用 effective_degree 而非 slider.value
```

### Dynamic title（标题随 shot 联动）

让 title cell 依赖 shot + 关键参数，标题里直接显示当前镜头 + 数值。观众扫一眼就知道在讲哪一档：

```python
@app.cell(hide_code=True)
def title(effective_degree, mo, shot):
    _label = shot.value.split(" · ")[1] if " · " in shot.value else shot.value
    mo.md(
        f"### 多项式过拟合 · 🎬 {shot.value.split(' · ')[0]} · {_label} "
        f"（degree = {effective_degree}）"
    ).style(margin="0", padding="4px 12px", font_size="15px")
    return
```

切 shot=B → title 显示 "🎬 B · 刚好（degree = 2）"。无声视觉反馈，让 shot 切换在录屏区**有可见效果**（弥补 shot 在录屏外的问题）。

---

## 3. 上下布局（提示组件下移）

录屏区**只放视觉演示**：标题 + 控件 + chart + 数字面板。

**文字 / 操作器**全部录屏区下方：
- shot dropdown（Playwright selector 操控，观众不用看到）
- narration 口播稿（按 shot 动态切换内容，主播看）
- 真实参数 callout（如和 narration 重复就删）

```
画布 1280×~960
┌──────────── header 60 ─────────────┐  ┐
├ side ─┬── stage 双槽 1100×560 ───┤  │
│ 160   │ slot1 + slot2 各 ~545     │  ├─ 录屏 1280×720
├──────┴── panel 60 ──────────────┤  │
├ shot 60 │ ────────────────────── ┤  ┘
├──── narration 整宽 1280×200 ─── ┤  ← 录屏外
└─────────────────────────────────┘
```

**绝不**把 shot dropdown / narration 放进 1280×720 内——观众会困惑"镜头 ABCD"是什么 meta 控件。

### 录屏外组件的铁律

1. **y 坐标 ≥ 36**（= 720/20 = 36 行边界）。`y=32, h=3` 看似在 720 内但底部到 700px **贴边判定模糊**——直接用 y ≥ 36 避免歧义
2. **truth_hint / scenario_card 等提示组件用纯 HTML div**，不要 `mo.callout(mo.md(...).style(...))`。后者在受限 cell 高度（h ≤ 5）下 box 渲染但 textContent 空白（_5 §7 第 4 条失败模式）。推荐模板：

   ```python
   @app.cell
   def truth_hint(mo, ...):
       mo.md(f"""<div style="background:#dbeafe;color:#1e40af;
           padding:8px 12px;border-radius:6px;font-size:13px;line-height:1.5;">
       🎯 真值：w₁=0.5, w₂=0.3, b=50  ·  调到这组值 → MSE→0
       </div>""")
       return
   ```

### chart height ↔ cell h 换算（避免试错 2-4 轮）

altair chart `properties(height=N)` 是**plot 区高度**，cell 实际占用还要加 title (~30px) + 上下 axis (~50px)：

| chart properties | 实际占用 | cell h（rowHeight=20）|
|---|---|---|
| `height=300` | ~380px | **h ≥ 19** |
| `height=380` | ~460px | **h ≥ 24** |
| `height=440` | ~520px | **h ≥ 26** |
| `height=520` | ~600px | **h ≥ 30** |

公式：`cell_h ≥ ceil((chart_height + 80) / 20)`，含 title + axis。**少一行 cell h** = altair x 轴 label 被截。Plotly fig 类似，但其 `update_layout(height=N)` 已含 title margin，公式收窄到 +50：`cell_h ≥ ceil((fig_height + 50) / 20)`。

---

## 4. sidebar 极简

教学视频里 chart 是主，控件是辅。

| 维度 | 推荐 | 理由 |
|---|---|---|
| sidebar 占总宽 | **12-15%**（160-200px / 1280）| 让位主区 |
| slider label | **1-2 字符**（`k` / `w₁` / `🔒 w₂=0`），或 sklearn API 名（如 `fit_intercept` / `x_new`）。**绝不**带中文说明括号（"x_new （新样本特征值）" ❌）| 录屏口播是主信息渠道，label 是次要锚点；中文说明放 narration / scenario_card 而非 sidebar 内 |
| `mo.ui.slider(full_width=True)` | **必加** | sidebar 缩窄后 slider 铺满容器 |
| `transform: scale(...)` | ⚠️ **仅 edit 模式生效**（见 §11）| 录屏用 `marimo run`，selector 失效。**首选 grid cell 直接缩**（如 5col→4col） |

注：marimo UI **没有 size 参数**（small/medium/large），唯一尺寸控制是 `full_width`。压缩首选 grid cell 列数，CSS hack 是后备。

### grid.json position 数组的对齐铁律

**`grid.json` `cells` 数组的 index = demo 中 `@app.cell` 出现顺序（自顶向下）**。每加 / 删 / 改顺序一个 cell 都要：

1. 重数 demo 中所有 `@app.cell`（用 `grep -n "^@app.cell" file.py`）
2. 对照 grid.json `cells` 数组逐位重映射
3. **没有自动校验工具**——肉眼数 + diff 检查，错位置一个会导致 cell 内容被分配到错误位置

防范：cell 顺序确定后**不要轻易插入新 cell**，要插就在文件末尾追加。

### cell 命名（CSS selector 前提）

把 `@app.cell` 装饰器下的 `def _():` 改成具名 `def stage(...):`，marimo 会用函数名作为 `data-cell-name`：

```python
@app.cell
def stage(chart_1d, mo, shot):  # 函数名 stage → data-cell-name='stage'
    ...
```

⚠️ **`[data-cell-name='X']` selector 仅 marimo edit 模式生效**，run 模式 DOM 没这个 attr（详见 §11）。录屏用的是 run 模式。

如果 sidebar 视觉真的需要再压缩，建议 edit 模式预览时 hack 看效果：

```css
/* 仅 edit 模式有效；run 模式无效 */
[data-cell-name='controls'] {
  transform: scale(0.92);
  transform-origin: top left;
  width: 109% !important;
}
```

### sidebar 内排版

vstack `gap=0` + 极简组别 header（11px / uppercase / 灰色），sliders 紧贴。`---` 改 1px 实线（border-top）省高度。

---

## 5. 字体（盖默认宋体）

marimo 仅暴露 **3 个公开字体 CSS 变量**，其他字号字重无 API：

```css
:root {
  --marimo-heading-font: "PingFang SC", "Hiragino Sans GB", system-ui, sans-serif;
  --marimo-text-font:    "PingFang SC", "Hiragino Sans GB", system-ui, sans-serif;
  /* --marimo-monospace-font 保持默认 ui-monospace */
}
```

PingFang SC = 苹方（macOS 系统现代中文无衬线）。fallback 链覆盖 Win / Linux。

---

## 6. 录屏区可视化边框（开发期）

custom.css 加红虚线伪元素标 1280×720 边界，所见即所得：

```css
.react-grid-layout::before {
  content: "📹 录屏区 1280×720";
  position: absolute;
  top: 0; left: 0;
  width: 1280px; height: 720px;
  border: 2px dashed #ef4444;
  pointer-events: none;
  z-index: 9999;
  font: 600 12px ui-monospace, monospace;
  color: #ef4444;
  padding: 4px 8px;
  background: rgba(239, 68, 68, 0.04);
  box-sizing: border-box;
}
```

**录屏前注释掉这段**——边框会出现在录屏画面里。

---

## 7. 失败模式（专杀此前踩过的坑）

| 现象 | 根因 | 修法 |
|---|---|---|
| cell 被推到画面外 | react-grid-layout **隐式 compaction**：长 cell 推开邻居 | 邻位 cell h 对齐（如 controls h = stage h） |
| 双图 hstack 出 X scroll | altair color legend 占额外 80px | 双图模式 `chart.properties(width=N)` 显式覆盖 + cell `overflow-x: hidden` 兜底 |
| accordion 内容盖邻居 | 主动展开后 1000+px 撑爆 cell（无论原 h 多大）| **grid 里不放 accordion**；详情用纯 HTML div |
| callout 显示空白 | marimo-callout-output web component 在受限 cell 高度下渲染异常（box 在但内容不渲染）| 改纯 HTML div 包 mo.md，或加大 cell h |
| 画布跟浏览器宽度变 | grid 容器默认 `w-full` | `custom.css` 锁 `.react-grid-layout { width: Npx !important }` + `#App { overflow-x: auto }` |
| chart `width="container"` 不稳 | altair container 模式渲染时机和容器尺寸耦合 | 回固定数字 + 给 cell 足够 padding |
| 标题宋体 | marimo 默认中文字体 fallback | `--marimo-heading-font` 覆盖（§5）|
| sm:pt-8 顶部留白 32px | marimo 默认 Tailwind 顶 padding | custom.css `[class*="sm:pt-8"] { padding-top: 0 !important }` |
| **cell 内容大幅下沉**（顶部空白 60px+）| callout / mo.md 默认 padding (24px) + react-grid-item 内 `p-2` (8px) **叠加** | custom.css `.react-grid-item > div { padding: 4px 6px !important }` 全局收紧 |
| **chart x/y 轴 label 被 cell 截掉** | cell h 不够装 chart `properties(height=N)` + axis 装饰 | 按 §3 公式 `cell_h ≥ ceil((height + 80) / 20)` 重算 |
| **CSS selector 在 run 模式失效** | `[data-cell-name]` 仅 edit 模式有 | 见 §11；run 模式只能用全局 selector（class / id）|
| **playwright 切 marimo dropdown 无效但脚本不报错** | shadow DOM + React state 不监听 native change 事件 | 见 §11；脚本必须加文本断言 |
| **mark_text 推 label 出 chart 边界**（如 dy=-6）| altair y 轴默认 max=数据最大值，label 浮在数据点上方导致溢出 cell | y 加 `alt.Scale(domainMax=max(data) * 1.15)` 留 15% 顶部留白 |
| **mo.state 改变让同 cell 多 widget 全重建**（不相关 widget 也 reset）| marimo cell 重跑会重新执行所有 ui 创建语句 | **拆 widget 到独立 cell**——只让需要的 cell 依赖 state，shot/dropdown 等独立 widget 自己一个 cell |
| **mo.ui.button `on_click=lambda _: slider.set_value(N)` 不工作** | marimo slider **没有 set_value 方法**（幻觉 API），on_click 是 `Callable[[T], T]` 返回新 value 而非 setter | 用 `mo.state` 共享状态：`get_x, set_x = mo.state(2); on_click=lambda _: set_x(1)`，slider value=get_x() |
| **`mo.ui.altair_chart()` 包同构 chart 报 `Duplicate signal name`** | 同一 chart 被两次包装时 Vega-Lite 注入同名 selection signal | 直接传 chart 对象到 hstack，**不包** mo.ui.altair_chart()；代价：失去 chart 选择交互（录屏通常不需要）|
| **`mo.md()` 内嵌 HTML block 内部 markdown 表格不渲染** | CommonMark spec：HTML block 内不解析 markdown | 要么全 md（不包 HTML wrapper），要么全 HTML `<table>` 标签 |
| **altair 多层 chart y 轴标题被隐式拼接** | 多层叠加每层不同 y 字段 → axis title 自动拼成 "y, y_pred, y_cur..." | 让所有层共用相同列名 "y"；rules 改 `mark_line + detail` 分组（避开 y/y2 双通道）|
| **plotly 3D 在双槽 hstack 右侧 legend 溢出** | equal-width 双槽宽度 ~540px，3D 带 legend 实际占用 ~620px | 双槽含 plotly 一律 `fig.update_layout(showlegend=False)` |
| **chart 双图临界宽度 ~1000px**（27col stage） | stage 27col=1080px 内宽 ~1050px；双 chart × (width=440 + axis 80) ≈ 1040 紧贴边界 | chart `width ≤ 400` 留 50px buffer |

---

## 8. 方法论（避免低效返工）

| 反例 | 正解 |
|---|---|
| 硬编码 grid.json 数字（每改重启 marimo + 截图验证 30s/轮）| **marimo edit 拖拽**，所见即所得，避开 compaction 雷 |
| 改一处验一次 | 一次性测多组数字，或先纸上画清楚 ASCII |
| cell 尺寸不看内容需求 | 先估每个组件最小可用尺寸（label 字符 × 字号 + padding），再分 grid 单元 |
| 依赖 marimo 内部 className | 文档明说跨版本不稳，仅 3 个字体变量公开，hack 自负风险 |
| 在 grid 里塞 accordion / 长 callout | 改用纯 HTML div + 固定 h，避免 web component 在受限容器下的渲染异常 |

---

## 9. ffmpeg 录屏 + 升采样

```bash
# 1. macOS 录屏（QuickTime / OBS / shortcuts）原始 .mov → raw.mov
# 2. crop 1280×720 + scale 1920×1080
ffmpeg -i raw.mov \
  -vf "crop=1280:720:0:0,scale=1920:1080:flags=lanczos" \
  -c:v libx264 -preset slow -crf 18 \
  -pix_fmt yuv420p \
  out.mp4
```

`flags=lanczos`：升采样高质量算法（默认 bilinear 会糊）。

`crf 18`：B 站推荐 18-23，越小越清晰文件越大。

`pix_fmt yuv420p`：兼容性（不加部分播放器无法播）。

---

## 10. 调完前自检

> **🚨 强制流程**（防止"陈述规则正确但应用规则失败"——本节不是"参考"，是 **closing gate**）：
>
> **新 demo 调布局开工前**：
> 1. **抄骨架**：先打开 `01-ML/02-LR/01-intro/demos/k-to-w-migration.py` + `custom.css` + `layouts/*.grid.json`，按这个模板 4 件套（title / sidebar / stage / panel）+ 提示组件外置 + cell 命名规范，**不要重新发明**
> 2. **打印本节 11 项到工作记忆顶**：每改一处 grid.json / chart properties / label / cell 内容 → 自问"本改动影响哪一项？"
>
> **调布局过程中**：
> 3. **每改一处** → playwright 截图比对（不是肉眼脑补）→ 看 11 项里有没有违规
> 4. 不要凭"理由"放过违规：每条违规都能解释（如"dataset 中文长所以 sidebar 宽"），但理由本身指向**根因路径**（改 dataset 选项），而不是简单路径（改 sidebar 宽度）
>
> **交付前**：
> 5. **不让 user 当 QA**：交付截图前自己确保 11 项全 ✓。让 user 反馈布局违规 = 流程失败
> 6. 如果 11 项有 ✗，回 §3 / §4 / §7 修正后重检，直到全 ✓
>
> **典型违规模式（avoid）**：
> - 把 dropdown / 口播稿 / 真实参数 callout 塞进录屏区（违反 §3 提示组件外置）
> - sidebar 给 5+ col 因为"中文 dropdown 选项长"（违反 §4，应改选项不改 sidebar）
> - label 带中文括号 "fit_intercept（学截距 b）"（违反 §4 极简 label）
> - 选 Strategy A 没读 demo README（违反 §2"问眼睛在看什么"）

---

| 项 | 验证 |
|---|---|
| 1280×720 内含 4 件套（标题/控件/chart/数字面板）| 截图比对 |
| shot dropdown / narration 在 720 之外 | 截图 1280×720 不含它们 |
| sidebar < 主区 1/4 视觉 | 主观评估 |
| slider/switch label ≤ 5 字符（sklearn API 名 OK，**禁带中文说明括号**）| 数字数 + grep 中文括号 |
| 标题不是宋体 | 视觉确认 |
| 字体在 Win / macOS / Linux 都有 fallback | 检查 fallback 链 |
| 切 A/B/C/D 镜头 stage 内容正确 | **edit 模式手动拖** dropdown，或脚本切后**读 callout text 断言**（不要靠"看截图"——切失败时 4 张截图二进制相同，肉眼难辨；详见 §11） |
| 双图镜头无 X scroll | dev tools `scrollWidth > clientWidth` 测 |
| 浏览器宽度变化时画布锁定 | 拖窗 |
| custom.css 录屏区红框已注释 | grep `react-grid-layout::before`（**仅 Step 6 录屏前需注释**；Step 5 调布局阶段保留 active 是 visual aid）|
| ffmpeg crop 命令验证 | 跑一遍输出 mp4 |

全 ✓ 才能开 Step 6 录屏。

---

## 11. edit 模式 vs run 模式的 DOM 差异（专栏）

> **录屏用 `marimo run`**，不是 `marimo edit`。两个模式的 DOM 结构不一样，很多 CSS selector / Playwright 操作只在 edit 模式生效。这是 §4 / §7 多条规则的潜在失效原因。

### 关键差异表

| 维度 | `marimo edit` | `marimo run`（录屏用）|
|---|---|---|
| `[data-cell-name='X']` attr | ✓ 存在 | ✗ **不存在** |
| `[data-cell-role='output']` attr | ✓ 存在 | ✗ **不存在** |
| cell 编辑器 / 拖拽 grid handle | ✓ | ✗ |
| `.react-grid-layout` / `.react-grid-item` | ✓ 存在（grid 视图）| ✓ 存在（layout_file 模式）|
| `marimo-callout-output` web component | ✓ | ✓ |
| dropdown native `<select>` | 隐藏在 shadow DOM | 隐藏在 shadow DOM |

### 影响：哪些规则在 run 模式失效

| 规则 | 失效原因 | run 模式替代 |
|---|---|---|
| §4 `[data-cell-name='controls'] { transform: scale(...) }` | data-cell-name 不存在 | grid cell 直接缩列数（5col → 4col）|
| §7 `[data-cell-name='stage'] { overflow-x: hidden }` | 同上 | chart `.properties(width=N)` 显式覆盖（双图避免溢出的唯一稳妥手段）|
| 任何按 cell name 精准选样式 | data-cell-name 不存在 | 改全局 selector：`.react-grid-item` / `.react-grid-layout` |

### Playwright 操作 marimo dropdown 的限制

`marimo run` 下的 dropdown 是 React 组件 + 自定义 web component，**不监听 native `change` event**：

- `page.locator('select').select_option(...)` 报 "option not visible"（在 shadow DOM 内）
- `page.evaluate("...el.value=X; el.dispatchEvent(new Event('change'))")` 静默失败——event 触发了但 marimo reactivity 不响应
- **结果**：4 张"切场景"截图二进制完全相同（md5 一致），但脚本不报错

**绕开方案**：

```python
# 方案 A：临时改 demo 默认值后重启 server 截图
# (改 mo.ui.dropdown(value="A·一元") → "B·二元"，重启 marimo run，截图)

# 方案 B：脚本截图后读 DOM 文本断言已切
text = page.locator('[class*="callout"]').inner_text()
assert "镜头 B" in text, f"切场景失败，仍是: {text[:50]}"
```

**铁律**：自检脚本必须**文本断言** + 截图，单纯比图片肉眼无法区分静默失败。

### 方案 A 实操流程（4 镜头切镜头每镜头独立截）

`marimo run --watch` **不会**响应 demo 文件 `default value` 修改。每个镜头要走完整流程：

```bash
# 单镜头流程（每个 shot 重复一次）
lsof -ti:2757 | xargs kill -9        # kill 旧 marimo
# 改文件中 mo.ui.dropdown(value="B · L(k) 切片") 默认值
marimo run --port 2757 demo.py &     # 后台重启
sleep 4                              # 等启动
python playwright-screenshot.py      # 截图（含 DOM textContent 探针）
```

单镜头 ~15 秒，4 镜头切完 ~1 分钟。**不要试图用 marimo `--watch` 一次截多张**——文件改了 server 不会切 dropdown 状态。

### 调试自查工具（粘贴即用）

不确定某个 selector 在 run 模式有没有效时，跑这段 playwright 探针：

```python
attrs = page.evaluate("""
() => [...document.querySelectorAll('.react-grid-item')].map(el => ({
  attrs: [...el.attributes].map(a => `${a.name}=${a.value.slice(0,40)}`),
  text: el.textContent.slice(0, 40),
}))
""")
import json
print(json.dumps(attrs, ensure_ascii=False, indent=2))
```

看输出有没有 `data-cell-name` 出现——没有就改全局 selector。

---

## 文档链接

- [`_0-workflow.md`](./_0-workflow.md) · 总入口
- [`_2-demo-guide.md`](./_2-demo-guide.md) · demo 设计原则（5 原则 + cell 拆分 + maxWidth 选择）
- [`_marimo-guide.md`](./_marimo-guide.md) · marimo 平台知识（CSS 变量 / 模式 / 陷阱）
- [`_6-recording-guide.md`](./_6-recording-guide.md) · Step 6 录屏（接 §9 ffmpeg 输出 mp4）
