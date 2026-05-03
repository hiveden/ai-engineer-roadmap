# Demo 制作指南（skill）

> Step 1 细则。marimo demo 设计 + grid 布局 + 视频录屏友好。

## 1. Cell 拆分原则（grid 友好）

**一个 cell = grid 最小可摆放单元**。组合 cell 锁死布局灵活性。

| 原则 | ❌ 反例 | ✅ 正例 |
|---|---|---|
| 图表独占 cell | `mo.vstack([mo.md("标题"), chart])` | chart 单独 cell + 标题独立 mo.md cell |
| 标题/状态卡独占 mo.md cell | 把 `LOOCV %` 拼进表格段 markdown 标题 | 独立 mo.md cell（突出关键状态）|
| chart title 抽出 | `chart.properties(title="k=11 · ...")` | chart 不带 title + 独立 mo.md cell 写图说明 |
| 控件可合并 | 4 滑块拆 4 cell（无意义） | 一行 hstack widths=[1,1,1,1] 合一 cell |
| 不要嵌套 vstack/hstack | `hstack([vstack([标题, k]), vstack([标题, rating])])` | 一行 4 个组件直接 hstack |

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

### 4.1 绑定到 demo

```python
app = marimo.App(
    width="medium",
    layout_file="layouts/XX-name.grid.json",
    css_file="marimo.css",
)
```

### 4.2 maxWidth 选择

| 比例 | maxWidth | 推荐场景 |
|---|---|---|
| 2:1 | 1200 | 决策边界类 demo（02-proximity 风格） |
| **16:9** | **1280** | **标准视频录屏（推荐）** |
| 1.4:1 | 1400 | 信息密度高的 demo（03-k-tuning 风格） |

`rowHeight` 默认 20px；`columns` 默认 24。

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

## 9. 验证清单

新 demo 录屏前过一遍：

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
