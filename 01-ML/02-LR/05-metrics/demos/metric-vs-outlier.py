"""
MAE / MSE / RMSE 对异常点敏感度 · 拖红星看爆炸

核心隐喻：拖一个红色异常点 → 同一张柱图上 MSE 顶天 / MAE 贴底
启动：marimo run metric-vs-outlier.py --port 2732 --headless --no-token

5 秒测试目标：用户不看任何文字，5 秒内能说出「MSE 对异常点最敏感」
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/metric-vs-outlier.grid.json",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import altair as alt
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return LinearRegression, alt, mo, np, pd


@app.cell
def _(mo):
    mo.md(
        r"""
        # 指标 vs 异常点 · MAE / MSE / RMSE

        15 个蓝点服从 `y = 2x + 5 + N(0,1)`，外加 1 个**红色异常点**。
        拖动它，看三个指标如何被拽到天上去。

        - **MAE** = 平均 |error| ──── 误差线性放大
        - **MSE** = 平均 error² ──── 误差**平方放大**（敏感的元凶）
        - **RMSE** = √MSE ──────── 介于两者之间，量纲与 y 一致
        """
    )
    return


@app.cell
def _(np):
    # 基础数据 · 固定 seed 保证可复现
    rng = np.random.default_rng(42)
    x_base = np.linspace(0, 10, 15)
    y_base = 2.0 * x_base + 5.0 + rng.normal(0, 1.0, size=15)
    return x_base, y_base


@app.cell
def _(mo):
    # ============ 控制面板 · 单一来源原则 ============
    # 预设直接出 (x_out, y_out)；slider 仅在「手动」模式生效
    # MSE 倍数指数递进：1x → 3x → 8x → 25x（拟合线 y=2x+5，x=5 处 y≈15）
    PRESETS = {
        "✋ 手动 (滑块控制)": None,
        "○ 无异常 · 1x baseline": (5.0, 15.0),       # 几乎贴拟合线
        "▲ 轻微 · MSE ~3x": (5.0, 21.0),             # 偏离 +6
        "▲▲ 中等 · MSE ~8x": (5.0, 26.0),            # 偏离 +11
        "💥 极端 · MSE ~25x": (5.0, 35.0),           # 偏离 +20
    }
    preset = mo.ui.dropdown(
        options=PRESETS,
        value="✋ 手动 (滑块控制)",
        label="预设场景",
    )
    x_out_slider = mo.ui.slider(
        0.0, 10.0, step=0.1, value=5.0,
        label="x_outlier", show_value=True,
    )
    y_out_slider = mo.ui.slider(
        -5.0, 40.0, step=0.5, value=28.0,
        label="y_outlier", show_value=True,
    )
    return PRESETS, preset, x_out_slider, y_out_slider


@app.cell
def _(mo, preset, x_out_slider, y_out_slider):
    mo.vstack(
        [
            preset,
            mo.hstack([x_out_slider, y_out_slider], justify="start", gap=2),
        ],
        gap=1,
    )
    return


@app.cell
def _(preset, x_out_slider, y_out_slider):
    # ============ 单一来源 · 解析当前异常点坐标 ============
    if preset.value is None:
        x_out, y_out = x_out_slider.value, y_out_slider.value
        is_preset = False
        preset_label = "手动"
    else:
        x_out, y_out = preset.value
        is_preset = True
        preset_label = preset.selected_key if hasattr(preset, "selected_key") else "预设"
    return is_preset, preset_label, x_out, y_out


@app.cell
def _(LinearRegression, np, x_base, x_out, y_base, y_out):
    # ============ 拟合：当前（含异常点） + baseline（仅正常点） ============
    X_cur = np.append(x_base, x_out).reshape(-1, 1)
    y_cur = np.append(y_base, y_out)

    model_cur = LinearRegression()
    model_cur.fit(X_cur, y_cur)
    y_pred_cur_full = model_cur.predict(X_cur)

    model_base = LinearRegression()
    X_base_2d = x_base.reshape(-1, 1)
    model_base.fit(X_base_2d, y_base)
    y_pred_base_full = model_base.predict(X_base_2d)

    # 拟合线端点（用于绘制直线）
    x_line = np.array([0.0, 10.0])
    y_line_cur = model_cur.predict(x_line.reshape(-1, 1))
    y_line_base = model_base.predict(x_line.reshape(-1, 1))
    return (
        X_cur,
        X_base_2d,
        model_cur,
        x_line,
        y_cur,
        y_line_base,
        y_line_cur,
        y_pred_base_full,
        y_pred_cur_full,
    )


@app.cell
def _(np, y_base, y_cur, y_pred_base_full, y_pred_cur_full):
    # ============ 三指标计算 ============
    def _metrics(y_true, y_pred):
        err = y_true - y_pred
        mae = float(np.mean(np.abs(err)))
        mse = float(np.mean(err ** 2))
        rmse = float(np.sqrt(mse))
        return mae, mse, rmse

    mae_cur, mse_cur, rmse_cur = _metrics(y_cur, y_pred_cur_full)
    mae_base, mse_base, rmse_base = _metrics(y_base, y_pred_base_full)

    def _mag(cur, base):
        return cur / base if base > 1e-6 else 1.0

    mag_mae = _mag(mae_cur, mae_base)
    mag_mse = _mag(mse_cur, mse_base)
    mag_rmse = _mag(rmse_cur, rmse_base)
    return (
        mae_base,
        mae_cur,
        mag_mae,
        mag_mse,
        mag_rmse,
        mse_base,
        mse_cur,
        rmse_base,
        rmse_cur,
    )


@app.cell
def _(is_preset, mag_mae, mag_mse, mag_rmse, mo, preset_label, x_out, y_out):
    # ============ 模式徽章 + 当前数值 ============
    if is_preset:
        badge = (
            f'<span style="background:#fef3c7;color:#92400e;padding:3px 10px;'
            f'border-radius:6px;font-size:13px;font-weight:600;">'
            f'预设模式 · {preset_label}</span>'
        )
    else:
        badge = (
            f'<span style="background:#dbeafe;color:#1e40af;padding:3px 10px;'
            f'border-radius:6px;font-size:13px;font-weight:600;">'
            f'手动模式</span>'
        )

    color_for = lambda m: "#10b981" if m < 1.5 else ("#f59e0b" if m < 5 else "#dc2626")

    mo.md(
        f"""
<div style="display:flex; gap:20px; align-items:center; flex-wrap:wrap;
            font-family:ui-monospace,monospace; font-size:14px; padding:8px 0;">
  <div>{badge}</div>
  <div>异常点 <b>(x={x_out:.1f}, y={y_out:.1f})</b></div>
  <div>MAE 放大 <b style="color:{color_for(mag_mae)}; font-size:16px">{mag_mae:.2f}x</b></div>
  <div>MSE 放大 <b style="color:{color_for(mag_mse)}; font-size:16px">{mag_mse:.2f}x</b></div>
  <div>RMSE 放大 <b style="color:{color_for(mag_rmse)}; font-size:16px">{mag_rmse:.2f}x</b></div>
</div>
        """
    )
    return


@app.cell
def _(
    alt,
    pd,
    x_base,
    x_line,
    x_out,
    y_base,
    y_line_base,
    y_line_cur,
    y_out,
    y_pred_cur_full,
):
    # ============ 第一行：散点 + 拟合线 + 残差方块（平方误差可视化） ============
    df_pts = pd.DataFrame(
        {
            "x": x_base,
            "y": y_base,
            "y_pred": y_pred_cur_full[:-1],          # 最后一个是异常点
            "err_sq": (y_base - y_pred_cur_full[:-1]) ** 2,
            "kind": ["正常点"] * len(x_base),
        }
    )
    # 异常点单独一行
    df_outlier = pd.DataFrame(
        {
            "x": [x_out],
            "y": [y_out],
            "y_pred": [y_pred_cur_full[-1]],
            "err_sq": [(y_out - y_pred_cur_full[-1]) ** 2],
            "kind": ["异常点"],
        }
    )
    df_all = pd.concat([df_pts, df_outlier], ignore_index=True)

    df_lines = pd.DataFrame(
        {
            "x": list(x_line) + list(x_line),
            "y": list(y_line_cur) + list(y_line_base),
            "kind": ["当前拟合线"] * 2 + ["baseline (无异常)"] * 2,
        }
    )

    # 残差方块（size 编码 err²，让平方放大效应可视化）
    squares = (
        alt.Chart(df_all)
        .mark_square(opacity=0.30, color="#ef4444", stroke="#ef4444", strokeWidth=1)
        .encode(
            x="x:Q",
            y="y:Q",
            size=alt.Size(
                "err_sq:Q",
                scale=alt.Scale(range=[20, 6000]),
                legend=None,
            ),
        )
    )
    # 残差竖线（连接真实点 → 拟合线对应预测点）
    rules = (
        alt.Chart(df_all)
        .mark_rule(color="#ef4444", opacity=0.55, strokeDash=[3, 2])
        .encode(x="x:Q", y="y:Q", y2="y_pred:Q")
    )
    # 正常点 · 蓝圆
    pts_normal = (
        alt.Chart(df_pts)
        .mark_circle(size=120, color="#1f77b4", stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-0.5, 10.5]), title="x"),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[-8, 42]), title="y"),
            tooltip=["x", "y", "y_pred", "err_sq"],
        )
    )
    # 异常点 · 红五角星
    pt_outlier = (
        alt.Chart(df_outlier)
        .mark_point(
            shape="M0,-1L0.225,-0.309L0.951,-0.309L0.363,0.118L0.588,0.809L0,0.382L-0.588,0.809L-0.363,0.118L-0.951,-0.309L-0.225,-0.309Z",
            size=600,
            color="#ef4444",
            fill="#ef4444",
            stroke="white",
            strokeWidth=2,
            opacity=1,
        )
        .encode(x="x:Q", y="y:Q", tooltip=["x", "y", "y_pred", "err_sq"])
    )
    # 拟合线（当前蓝 + baseline 灰虚）
    line_layer = (
        alt.Chart(df_lines)
        .mark_line(strokeWidth=2.5)
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["当前拟合线", "baseline (无异常)"],
                    range=["#3b82f6", "#9ca3af"],
                ),
                legend=alt.Legend(title=None, orient="top-right"),
            ),
            strokeDash=alt.StrokeDash(
                "kind:N",
                scale=alt.Scale(
                    domain=["当前拟合线", "baseline (无异常)"],
                    range=[[1, 0], [6, 4]],
                ),
                legend=None,
            ),
        )
    )

    chart_scatter = (rules + squares + line_layer + pts_normal + pt_outlier).properties(
        width=760,
        height=320,
        title="散点 + 拟合线 · 红方块大小 = 误差² · 异常点处冒巨大红方块就是 MSE 爆炸的元凶",
    )
    return (chart_scatter,)


@app.cell
def _(
    alt,
    mae_base,
    mae_cur,
    mag_mae,
    mag_mse,
    mag_rmse,
    mse_base,
    mse_cur,
    pd,
    rmse_base,
    rmse_cur,
):
    # ============ 第二行：合并柱图 · 共享线性 y 轴 · 5 秒测试核心 ============
    # 横轴 = MAE / MSE / RMSE 三个 category
    # 每个 category 内并列：baseline 灰柱 + current 蓝柱
    # 共享 y 轴 → MSE 顶天，MAE/RMSE 贴底，倍数差用柱高直接砸脸
    rows = []
    for name, base_v, cur_v, mag in [
        ("MAE", mae_base, mae_cur, mag_mae),
        ("MSE", mse_base, mse_cur, mag_mse),
        ("RMSE", rmse_base, rmse_cur, mag_rmse),
    ]:
        rows.append({"指标": name, "类型": "baseline (无异常)", "值": base_v, "倍数": 1.0})
        rows.append({"指标": name, "类型": "current (含异常)", "值": cur_v, "倍数": mag})
    df_bar = pd.DataFrame(rows)

    # 倍数标签数据（只标 current 柱顶）
    df_lab = df_bar[df_bar["类型"] == "current (含异常)"].copy()
    df_lab["label"] = df_lab["倍数"].apply(lambda m: f"{m:.1f}x")

    bars = (
        alt.Chart(df_bar)
        .mark_bar()
        .encode(
            x=alt.X("指标:N", title=None, sort=["MAE", "MSE", "RMSE"]),
            xOffset=alt.XOffset(
                "类型:N",
                sort=["baseline (无异常)", "current (含异常)"],
            ),
            y=alt.Y("值:Q", title="指标数值（共享线性 y 轴）"),
            color=alt.Color(
                "类型:N",
                scale=alt.Scale(
                    domain=["baseline (无异常)", "current (含异常)"],
                    range=["#d1d5db", "#3b82f6"],
                ),
                legend=alt.Legend(title=None, orient="top-right"),
            ),
            tooltip=["指标", "类型", alt.Tooltip("值:Q", format=".3f"), alt.Tooltip("倍数:Q", format=".2f")],
        )
    )
    # 倍数徽章（红色大字贴柱顶）
    labels = (
        alt.Chart(df_lab)
        .mark_text(
            align="center",
            baseline="bottom",
            dy=-6,
            fontSize=18,
            fontWeight="bold",
            color="#dc2626",
        )
        .encode(
            x=alt.X("指标:N", sort=["MAE", "MSE", "RMSE"]),
            xOffset=alt.XOffset(
                "类型:N",
                sort=["baseline (无异常)", "current (含异常)"],
            ),
            y=alt.Y("值:Q"),
            text="label:N",
        )
    )

    chart_bars = (bars + labels).properties(
        width=760,
        height=300,
        title="三指标合并柱图 · 共享 y 轴 · MSE 顶天 / MAE 贴底 = 5 秒看出谁最敏感",
    )
    return (chart_bars,)


@app.cell
def _(chart_scatter, mo):
    # 散点图 · 拟合线 + 残差方块（独占 cell · grid 友好）
    mo.ui.altair_chart(chart_scatter)
    return


@app.cell
def _(chart_bars, mo):
    # 柱图 · MAE/MSE/RMSE 共享 y 轴对比
    mo.ui.altair_chart(chart_bars)
    return


@app.cell
def _(mag_mae, mag_mse, mag_rmse, mo):
    # ============ 信息面板 · 叙事归因 ============
    if mag_mse > 5:
        kind = "warn"
        verdict = f"💥 **MSE 已被异常点绑架**（放大 {mag_mse:.1f}x）。如果数据里有异常点又来不及清洗，**改用 MAE**。"
    elif mag_mse > 2:
        kind = "info"
        verdict = f"⚠️ MSE 已经明显放大（{mag_mse:.1f}x），但还在可接受范围。继续拖远异常点会进一步爆炸。"
    else:
        kind = "success"
        verdict = "✅ 当前异常点不严重，三个指标差异不大。试试「极端」预设感受爆炸。"

    mo.callout(
        mo.md(
            f"""
**为什么 MSE 爆炸？**
异常点的误差被**平方**：误差 10 → MSE 贡献 100；MAE 贡献只 10。一颗老鼠屎坏一锅汤。

| 指标 | 当前 / baseline | 性质 |
|---|---|---|
| MAE  | {mag_mae:.2f}x | 线性放大，对异常点温和 |
| MSE  | {mag_mse:.2f}x | 平方放大，**对异常点剧烈敏感** |
| RMSE | {mag_rmse:.2f}x | √MSE，量纲与 y 一致，敏感度介于两者 |

{verdict}
            """
        ),
        kind=kind,
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ### 玩法

        1. **预设序列**：依次选 `无异常 → 轻微 → 中等 → 极端`，看下方柱图 MSE 蓝柱**指数级跳高**，MAE/RMSE 几乎不动
        2. **手动模式**：切回手动，拖 `y_outlier` 从 15 → 40，盯住散点图——异常点处的红方块膨胀成巨人
        3. **拟合线被拽**：散点图里蓝色实线（当前）和灰色虚线（baseline）的夹角 = 异常点对模型的破坏力
        4. **结论**：「数据有异常点 + 来不及清洗」→ MAE 比 MSE 鲁棒；其余情况 MSE 仍是默认选择（可微 + 凸优化友好）
        """
    )
    return


@app.cell
def _(mo):
    # ===== 📐 录屏 grid 布局参考（开发用 · 录屏 position=null 隐藏）=====
    mo.accordion(
        {
            "📐 录屏布局参考（grid 设计意图）": mo.md(
                r"""
**目标 viewport**：1280 宽 · 16:9 录屏 · 单屏不滚动。
四段式：标题 / 控件 / 状态徽章 / 散点 + 柱图 双图 / 信息面板 callout。

### Grid 骨架（24 列 × rowHeight 20px）

```
   0           14          24
0  ┌────────── 标题 + 引导（h=4）────────────┐
4  ├────── 控件区：preset + x_out + y_out ───┤  h=4
8  ├────── 状态徽章 + 三个放大倍数 ─────────┤  h=3
11 ├──── 散点 + 拟合线 ────┬── 三柱图 ─────┤
   │  chart_scatter        │  chart_bars   │
   │  width≈14col          │  width≈10col  │  h=19
30 ├──────── 信息面板（叙事归因 callout）───┤  h=9
39 └─────────────────────────────────────────┘

   ←─── 14col ───→← ──── 10col ────→
              maxWidth 1280px
```

### 镜头脚本（5 秒测试 → 叙事下钻）

| # | 时长 | 焦点 | 教学动作 |
|---|---|---|---|
| **A** | 0-5s | 首屏柱图 MSE 顶天 | 不看文字也能秒懂 |
| **B** | 5-30s | 散点图红方块膨胀 | 拖 y_outlier 看平方放大可视化 |
| **C** | 30-60s | 预设序列依次切换 | 1x → 3x → 8x → 25x 指数感 |
| **D** | 60-90s | 信息面板归因 | 叙事 + 何时改用 MAE |

### 关键录屏提示

1. **散点 + 柱图左右并排**（grid 14:10）→ 视线扫一眼即覆盖两个核心证据
2. **状态徽章窄条**贴在控件下方，三个放大倍数实时跳动（颜色编码：绿/橙/红）
3. **信息面板用 callout** 而非纯 md → kind 切换给视觉信号（success/info/warn）
4. **玩法 mo.md** 录屏 null（位于 callout 之后用作教师参考，非镜头内容）
                """
            )
        },
        multiple=False,
    )
    return


if __name__ == "__main__":
    app.run()
