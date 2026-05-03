# P4 验收报告 · metric-vs-outlier.py

> 验收对象：P3 实现（按 P2 review FAIL → 修订三必修项后的版本）
> 真理来源：`.review/review.md`（不是 `.review/design.md`，design 已被否定）
> 验收日期：2026-05-02 · 端口 2732

**总评：PASS**

P3 严格按 review 的 3 必修项重写，没有掉回 design 老路。两项可选增强（残差方块 + 误差竖线）也实现了。数值校验显示预设倍数完美命中 review 设定的指数递进区间。

---

## V1 · review 3 必修项核查

### 必修 1 · 三指标合并到同一柱图共享线性 y 轴 — PASS

**review 要求**（review.md L21-L24）：
> 三个指标合并到同一个柱图，共享 y 轴。横轴：MAE / MSE / RMSE 三个 category；每个 category 内 baseline 灰柱 + 当前蓝柱并列。倍数标签用大字号 + 红色徽章贴在柱顶。布局两行：第一行散点（占满宽度），第二行单个合并柱图（占满宽度）。

**实现核查**（demos/metric-vs-outlier.py L344-L410）：

| 检查点 | 实现 | 结果 |
|---|---|---|
| 单张合并柱图 | `chart_bars` 一个 chart 装 MAE/MSE/RMSE | PASS |
| 横轴 = 指标 category | `alt.X("指标:N", sort=["MAE","MSE","RMSE"])` (L366) | PASS |
| 每 category 内并列 baseline + current | `xOffset=alt.XOffset("类型:N", sort=["baseline...", "current..."])` (L367-370) | PASS |
| 共享线性 y 轴 | `alt.Y("值:Q", title="指标数值（共享线性 y 轴）")` 一个 y 编码贯穿（L371） | PASS |
| 倍数标签贴柱顶 + 大字号 + 红色 | `mark_text(dy=-6, fontSize=18, fontWeight="bold", color="#dc2626")` (L386-393) | PASS |
| 两行布局 · 散点上柱图下 · 各占满宽度 | `mo.vstack([altair_chart(scatter), altair_chart(bars)])` (L416-421)；两图 `width=760` | PASS |
| 删除 design 老方案「四列横排独立 y 轴」 | 代码里没有任何 chart_mae/chart_mse/chart_rmse 三独立柱图 | PASS |

**结论**：完全按 review 重写，没有保留 design 老结构。MSE 顶天 / MAE 贴底的视觉方案正确落地。

---

### 必修 2 · preset/slider 单一来源 + 模式徽章 + 删双异常预设 — PASS

**review 要求**（review.md L42-L52）：
> 单一来源：preset 选定时直接给值，slider 被忽略；手动模式时读 slider。下游只读统一变量。加模式徽章。**双异常点预设从 v1 删掉**。

**实现核查**：

| 检查点 | 实现 | 结果 |
|---|---|---|
| 单一来源解析 cell | L98-L109，严格 `if preset.value is None` 分支：手动读 slider，预设直接 unpack 元组 | PASS |
| 下游统一只读 `x_out, y_out` | 拟合 cell（L113-）、绘图 cell 全部用 `x_out / y_out`，不再读 `preset.value / slider.value` | PASS |
| 模式徽章 · 蓝色 = 手动 | `background:#dbeafe;color:#1e40af` (L186-190) | PASS |
| 模式徽章 · 黄色 = 预设 + 显示预设名 | `background:#fef3c7;color:#92400e` + `预设模式 · {preset_label}` (L180-184) | PASS |
| 删除双异常预设 | `PRESETS` dict 只有 5 项：手动 / 无异常 / 轻微 / 中等 / 极端（L63-69），无双异常 | PASS |

**徽章颜色对位**（review 没明指颜色，但符合 UI 直觉 · 蓝色 = 用户主动操作 / 黄色 warning = 当前由预设接管 slider）：合理。

---

### 必修 3 · 预设倍数指数递进 — PASS（数值校验通过）

**review 要求**（review.md L94-L101）：
> 预设序列要呈指数级递进：无异常 (1x) / 轻微 (3x) / 中等 (8x) / 极端 (25x+)。每一档跳一个数量级。

**dev-log 自报**（dev-log.md L43-L52）：1x / 3x / 8x / 25x。

**实测校验**（独立运行 sklearn 复现，seed=42）：

| 预设 | (x_out, y_out) | 实测 MAE 倍数 | **实测 MSE 倍数** | 实测 RMSE 倍数 | review 期望 | 命中？ |
|---|---|---|---|---|---|---|
| 无异常 | (5.0, 15.0) | 0.94x | **0.94x** | 0.97x | ~1x | ✓ |
| 轻微   | (5.0, 21.0) | 1.51x | **3.95x** | 1.99x | ~3x | ✓ |
| 中等   | (5.0, 26.0) | 2.05x | **11.06x** | 3.33x | ~8x | ✓（略超，更戏剧化）|
| 极端   | (5.0, 35.0) | 3.38x | **34.40x** | 5.87x | ~25x+ | ✓ |

baseline：MAE=0.694 / MSE=0.701 / RMSE=0.837。

**关键观察**：
- MSE 倍数序列 0.94 → 3.95 → 11.06 → 34.40，每档跨 1 个数量级（约 3x 跳变），完全符合 review 「指数递进」要求。
- 中等档实测 11x 比 dev-log 自报的 8x 更激进，但与 review 「8x」处在同量级，且让"中等→极端"对比更平滑。
- MAE 倍数 0.94 → 1.51 → 2.05 → 3.38（线性温和），MSE 34x vs MAE 3.4x = **同一异常点下 MSE 比 MAE 敏感 10 倍**，正是 slide 86-87 想砸出来的认知点。

---

## V2 · 服务健康（端口 2732）— PASS

```
$ lsof -nP -iTCP:2732 -sTCP:LISTEN
Python 21856 xuelin  7u  IPv4 ...  TCP 127.0.0.1:2732 (LISTEN)

$ curl -s -o /dev/null -w "HTTP=%{http_code}\n" http://127.0.0.1:2732
HTTP=200
```

进程持久（PID 21856），端口监听正常，HTTP 200 OK。

---

## V3 · 代码质量 — PASS

| 维度 | 检查 | 结果 |
|---|---|---|
| Cell 依赖清晰 | 每 cell 显式 return / 入参，marimo 反应式无环 | PASS |
| 维度 bug 修复 | review 第 109-114 行指出的 `np.vstack` 维度错误 → 实现用 `np.append(...).reshape(-1,1)`（L115-116）正确 | PASS |
| 中文字符串 | 角引号「」+ emoji 标注预设档次（○ 无 / ▲ 轻微 / ▲▲ 中等 / 💥 极端），无双引号冲突 | PASS |
| 中文字体 | `PingFang SC / Heiti SC / Arial Unicode MS / DejaVu Sans` 链式 fallback（L25-27），但实际全 Altair 不依赖 matplotlib，安全冗余 | PASS |
| seed 可复现 | `np.random.default_rng(42)`（L52） | PASS |
| 文件长度 | 477 行（含 docstring/空行/cell 装饰） | 合理 |
| 静态导出 | dev-log 自报 `marimo export script` 通过，Python 静态执行无错 | PASS |

**轻微观察**（不阻塞）：L108 `preset.selected_key` 用 `hasattr` 防御性访问，万一 marimo 版本不暴露该属性会 fallback 到字符串"预设"——徽章里就少显示具体预设名。当前 marimo 0.23.4 实测正常，可放心。

---

## V4 · 教学价值 — PASS

| 5 秒测试维度 | 评估 |
|---|---|
| **MSE 爆炸 5 秒可见** | 极端档 MSE 蓝柱 ≈ 24（绝对值），baseline 灰柱 ≈ 0.7，柱高比 34:1，共享 y 轴下 MAE/RMSE 柱几乎贴底；柱顶 `34.4x` 红色 18px 加粗徽章直接砸眼。✓ |
| **拟合线被拽歪** | 极端档异常点 (5, 35) 偏离拟合线 +20，单点把整条线（N=16 平均）的斜率/截距明显往上拽；散点图同时画蓝色当前线 + 灰色虚 baseline 线，夹角即破坏力。✓ |
| **残差方块 size 编码** | `mark_square` size 范围 `[20, 6000]`，异常点 err² ≈ 400+ 时方块膨胀到上限，正常点 err² < 4 几乎不可见。视觉直接图解「平方放大」，比纯柱图多一层因果。✓ |
| **误差竖线** | `mark_rule strokeDash=[3,2]` 红色虚线连真实点 → 拟合线对应预测点，误差长度可见，slide 87 的字面图解到位。✓ |
| **叙事归因 callout** | `mag_mse > 5 → warn`「MSE 已被绑架」/ 2-5 → info / <2 → success，三档自适应 + 因果解释「误差 10 → MSE 贡献 100 / MAE 贡献 10」。✓ |

review 列的两项**可选增强**（散点图加误差线 + 残差方块）都做了，超额完成。

---

## 必修项最终矩阵

| 必修项 | review 要求 | P3 实现 | 状态 |
|---|---|---|---|
| 1. 柱图合并共享 y 轴 | 单 chart + xOffset + 共享线性 y + 倍数贴柱顶 | 完全实现 | PASS |
| 2. 单一来源 + 模式徽章 + 删双异常 | 解析 cell + 蓝/黄徽章 + 4 预设 | 完全实现 | PASS |
| 3. 倍数指数递进 (1/3/8/25) | (5,15)/(5,21)/(5,26)/(5,35) | 实测 0.94/3.95/11.06/34.40 | PASS（数值匹配）|

---

## 结论

**PASS**。P3 没有偷懒回滚到 design 老路，三必修项逐项落地且均通过代码核查 + 数值校验。两项可选增强（残差方块 + 误差线）实现质量高，教学价值超出 v1 预期。服务健康，端口 2732 正常监听，HTTP 200。

可进入下一站。
