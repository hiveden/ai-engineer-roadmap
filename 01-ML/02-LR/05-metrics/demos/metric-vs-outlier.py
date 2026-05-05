"""
MAE / MSE / RMSE 对异常点敏感度 · Strategy B 双槽

教学核心：拖异常点 → MSE 指数爆炸 / MAE 线性温和 / RMSE 居中。
5 秒测试：不看文字，柱图 MSE 顶天 = 秒懂谁最敏感。

录屏布局（_5-layout-guide Strategy B）：
- A · 散点+柱图：slot1 = scatter(fit+squares), slot2 = bar chart
- B · 残差表：slot1 = scatter, slot2 = residual table(x/y/ŷ/|err|/err²)
- C · 倍数面板：slot1 = bar chart, slot2 = magnification comparison
- D · 归因：slot1 = bar chart, slot2 = verdict text

跑：
  cd 01-ML/02-LR/05-metrics/demos && marimo edit --port 2759 metric-vs-outlier.py
  cd 01-ML/02-LR/05-metrics/demos && marimo run --port 2759 metric-vs-outlier.py
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/metric-vs-outlier.grid.json",
    css_file="custom.css",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import altair as alt
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    return LinearRegression, alt, mo, np, pd


@app.cell(hide_code=True)
def title(mo):
    mo.md(
        r"### 指标 vs 异常点 · MAE / MSE / RMSE · 拖红星看 MSE 爆炸"
    ).style(margin="0", padding="4px 12px", font_size="15px", line_height="1.3")
    return


@app.cell
def _(np):
    # 基础数据 · 固定 seed
    rng = np.random.default_rng(42)
    x_base = np.linspace(0, 10, 15)
    y_base = 2.0 * x_base + 5.0 + rng.normal(0, 1.0, size=15)
    return x_base, y_base


@app.cell
def _(mo):
    # ===== 滑块 + shot（极简 label）=====
    _S = dict(show_value=True, full_width=True)
    x_out_slider = mo.ui.slider(0.0, 10.0, step=0.1, value=5.0, label="x_out", **_S)
    y_out_slider = mo.ui.slider(-5.0, 40.0, step=0.5, value=28.0, label="y_out", **_S)
    preset = mo.ui.dropdown(
        options=["手动", "○ 无异常", "▲ 轻微", "▲▲ 中等", "💥 极端"],
        value="💥 极端",
        label="预设",
    )
    shot = mo.ui.dropdown(
        options=["A · 散点+柱图", "B · 残差表", "C · 倍数面板", "D · 归因"],
        value="A · 散点+柱图",
        label="🎬 镜头",
    )
    return preset, shot, x_out_slider, y_out_slider


@app.cell
def controls(mo, preset, x_out_slider, y_out_slider):
    # 控件组合（sidebar 4col）
    _h = lambda s: mo.md(s).style(
        margin="0", padding="0", font_size="11px",
        font_weight="700", color="#6b7280", letter_spacing="0.05em",
    )
    mo.vstack(
        [_h("异常点"), preset, x_out_slider, y_out_slider],
        gap=0,
        align="stretch",
    )
    return


@app.cell
def _(preset, x_out_slider, y_out_slider):
    # 单一来源 · 异常点坐标
    _PRESETS = {
        "○ 无异常": (5.0, 15.0),
        "▲ 轻微": (5.0, 21.0),
        "▲▲ 中等": (5.0, 26.0),
        "💥 极端": (5.0, 35.0),
    }
    if preset.value == "手动":
        x_out, y_out = float(x_out_slider.value), float(y_out_slider.value)
    else:
        x_out, y_out = _PRESETS.get(preset.value, (5.0, 35.0))
    return x_out, y_out


@app.cell
def _(LinearRegression, np, x_base, x_out, y_base, y_out):
    # 拟合：当前（含异常点）+ baseline（仅正常点）
    X_cur = np.append(x_base, x_out).reshape(-1, 1)
    y_cur = np.append(y_base, y_out)

    model_cur = LinearRegression().fit(X_cur, y_cur)
    y_pred_cur = model_cur.predict(X_cur)

    model_base = LinearRegression().fit(x_base.reshape(-1, 1), y_base)
    y_pred_base = model_base.predict(x_base.reshape(-1, 1))

    # 拟合线端点
    x_line = np.array([0.0, 10.0])
    y_line_cur = model_cur.predict(x_line.reshape(-1, 1))
    y_line_base = model_base.predict(x_line.reshape(-1, 1))
    return x_line, y_cur, y_line_base, y_line_cur, y_pred_base, y_pred_cur


@app.cell
def _(np, y_base, y_cur, y_pred_base, y_pred_cur):
    # 三指标
    def _metrics(y_true, y_pred):
        err = y_true - y_pred
        mae = float(np.mean(np.abs(err)))
        mse = float(np.mean(err ** 2))
        rmse = float(np.sqrt(mse))
        return mae, mse, rmse

    mae_cur, mse_cur, rmse_cur = _metrics(y_cur, y_pred_cur)
    mae_base, mse_base, rmse_base = _metrics(y_base, y_pred_base)

    mag_mae = mae_cur / mae_base if mae_base > 1e-6 else 1.0
    mag_mse = mse_cur / mse_base if mse_base > 1e-6 else 1.0
    mag_rmse = rmse_cur / rmse_base if rmse_base > 1e-6 else 1.0
    return (
        mae_base, mae_cur, mag_mae, mag_mse, mag_rmse,
        mse_base, mse_cur, rmse_base, rmse_cur,
    )


@app.cell
def _(alt, pd, x_base, x_line, x_out, y_base, y_line_base, y_line_cur, y_out, y_pred_cur):
    # ===== chart_scatter =====
    _df_pts = pd.DataFrame({
        "x": x_base, "y": y_base,
        "y_pred": y_pred_cur[:-1],
        "err_sq": (y_base - y_pred_cur[:-1]) ** 2,
    })
    _df_out = pd.DataFrame({
        "x": [x_out], "y": [y_out],
        "y_pred": [y_pred_cur[-1]],
        "err_sq": [(y_out - y_pred_cur[-1]) ** 2],
    })
    _df_all = pd.concat([_df_pts, _df_out], ignore_index=True)
    _df_lines = pd.DataFrame({
        "x": list(x_line) * 2,
        "y": list(y_line_cur) + list(y_line_base),
        "kind": ["当前拟合"] * 2 + ["baseline"] * 2,
    })

    _squares = alt.Chart(_df_all).mark_square(
        opacity=0.30, color="#ef4444", stroke="#ef4444", strokeWidth=1
    ).encode(x="x:Q", y="y:Q", size=alt.Size("err_sq:Q", scale=alt.Scale(range=[20, 5000]), legend=None))

    _rules = alt.Chart(_df_all).mark_rule(
        color="#ef4444", opacity=0.55, strokeDash=[3, 2]
    ).encode(x="x:Q", y="y:Q", y2="y_pred:Q")

    _pts = alt.Chart(_df_pts).mark_circle(
        size=100, color="#1f77b4", stroke="white", strokeWidth=1.5
    ).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[-0.5, 10.5]), title="x"),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[-8, 42]), title="y"),
    )
    _star = alt.Chart(_df_out).mark_point(
        shape="M0,-1L0.225,-0.309L0.951,-0.309L0.363,0.118L0.588,0.809L0,0.382L-0.588,0.809L-0.363,0.118L-0.951,-0.309L-0.225,-0.309Z",
        size=500, color="#ef4444", fill="#ef4444", stroke="white", strokeWidth=2,
    ).encode(x="x:Q", y="y:Q")

    _lines = alt.Chart(_df_lines).mark_line(strokeWidth=2.5).encode(
        x="x:Q", y="y:Q",
        color=alt.Color("kind:N", scale=alt.Scale(
            domain=["当前拟合", "baseline"], range=["#3b82f6", "#9ca3af"]
        ), legend=alt.Legend(title=None, orient="top-right")),
        strokeDash=alt.StrokeDash("kind:N", scale=alt.Scale(
            domain=["当前拟合", "baseline"], range=[[1, 0], [6, 4]]
        ), legend=None),
    )

    chart_scatter = (_rules + _squares + _lines + _pts + _star).properties(
        width=460, height=400, title="散点 + 拟合线 · 红方块 = 误差²",
    )
    return (chart_scatter,)


@app.cell
def _(alt, mae_base, mae_cur, mag_mae, mag_mse, mag_rmse, mse_base, mse_cur, pd, rmse_base, rmse_cur):
    # ===== chart_bars =====
    _rows = []
    for name, base_v, cur_v in [
        ("MAE", mae_base, mae_cur),
        ("MSE", mse_base, mse_cur),
        ("RMSE", rmse_base, rmse_cur),
    ]:
        _rows.append({"指标": name, "类型": "baseline", "值": base_v})
        _rows.append({"指标": name, "类型": "current", "值": cur_v})
    _df_bar = pd.DataFrame(_rows)

    _df_lab = pd.DataFrame([
        {"指标": "MAE", "类型": "current", "值": mae_cur, "label": f"{mag_mae:.1f}x"},
        {"指标": "MSE", "类型": "current", "值": mse_cur, "label": f"{mag_mse:.1f}x"},
        {"指标": "RMSE", "类型": "current", "值": rmse_cur, "label": f"{mag_rmse:.1f}x"},
    ])

    # y 轴顶部留 15% 给 mark_text label（防 dy=-6 推出 chart 边界）
    _y_max = max(mae_cur, mse_cur, rmse_cur, mae_base, mse_base, rmse_base) * 1.15

    _bars = alt.Chart(_df_bar).mark_bar().encode(
        x=alt.X("指标:N", title=None, sort=["MAE", "MSE", "RMSE"]),
        xOffset=alt.XOffset("类型:N", sort=["baseline", "current"]),
        y=alt.Y("值:Q", title="指标数值", scale=alt.Scale(domain=[0, _y_max])),
        color=alt.Color("类型:N", scale=alt.Scale(
            domain=["baseline", "current"], range=["#d1d5db", "#3b82f6"]
        ), legend=alt.Legend(title=None, orient="top-right")),
        tooltip=["指标", "类型", alt.Tooltip("值:Q", format=".3f")],
    )
    _labels = alt.Chart(_df_lab).mark_text(
        align="center", baseline="bottom", dy=-6,
        fontSize=18, fontWeight="bold", color="#dc2626",
    ).encode(
        x=alt.X("指标:N", sort=["MAE", "MSE", "RMSE"]),
        xOffset=alt.XOffset("类型:N", sort=["baseline", "current"]),
        y="值:Q", text="label:N",
    )

    chart_bars = (_bars + _labels).properties(
        width=460, height=400, title="三指标柱图 · MSE 顶天 = 对异常点最敏感",
    )
    return (chart_bars,)


@app.cell
def stage(
    chart_bars, chart_scatter, mae_cur, mag_mae, mag_mse, mag_rmse, mo,
    mse_cur, np, pd, rmse_cur, shot, x_base, x_out, y_base, y_out, y_pred_cur,
):
    # 🎬 中央舞台 · Strategy B 双槽
    if shot.value.startswith("A"):
        _slot1 = mo.ui.altair_chart(chart_scatter)
        _slot2 = mo.ui.altair_chart(chart_bars)

    elif shot.value.startswith("B"):
        _slot1 = mo.ui.altair_chart(chart_scatter)
        _err = y_base - y_pred_cur[:-1]
        _rows_html = ""
        for i in range(min(5, len(x_base))):
            _rows_html += f"<tr><td>{x_base[i]:.2f}</td><td>{y_base[i]:.2f}</td><td>{y_pred_cur[i]:.2f}</td><td>{abs(_err[i]):.3f}</td><td>{_err[i]**2:.3f}</td></tr>"
        _out_err = y_out - y_pred_cur[-1]
        _slot2 = mo.md(f"""<div style="font-size:13px;line-height:1.5;padding:10px 14px;">
<b>残差表</b> · 15 正常 + 1 异常
<table style="width:100%;border-collapse:collapse;margin:8px 0;font-size:13px;">
<tr style="border-bottom:2px solid #e5e7eb;"><th>x</th><th>y</th><th>ŷ</th><th>|err|</th><th>err²</th></tr>
{_rows_html}
<tr style="color:#9ca3af;"><td colspan="5">... 共 15 行</td></tr>
</table>
<hr style="margin:8px 0;border:none;border-top:1px solid #e5e7eb;">
<b style="color:#ef4444;">异常点</b> x={x_out:.1f}, y={y_out:.1f}, ŷ={y_pred_cur[-1]:.1f}<br>
|err| = {abs(_out_err):.2f}, err² = <b style="color:#dc2626;">{_out_err**2:.1f}</b> ← 平方放大！<br><br>
<b>MSE</b> = 16 个 err² 平均 = {mse_cur:.2f}
</div>""")

    elif shot.value.startswith("C"):
        _slot1 = mo.ui.altair_chart(chart_bars)
        _c = lambda m: "#10b981" if m < 1.5 else ("#f59e0b" if m < 5 else "#dc2626")
        _slot2 = mo.md(f"""<div style="font-size:14px;line-height:1.8;padding:8px 12px;">
<b>倍数对照</b>（current / baseline）
<table style="width:100%;border-collapse:collapse;margin:8px 0;font-size:14px;">
<tr style="border-bottom:2px solid #e5e7eb;"><th style="text-align:left;">指标</th><th style="text-align:left;">放大倍数</th><th style="text-align:left;">性质</th></tr>
<tr><td>MAE</td><td><b style="color:{_c(mag_mae)}">{mag_mae:.2f}x</b></td><td>线性放大</td></tr>
<tr><td>MSE</td><td><b style="color:{_c(mag_mse)}">{mag_mse:.2f}x</b></td><td>平方放大</td></tr>
<tr><td>RMSE</td><td><b style="color:{_c(mag_rmse)}">{mag_rmse:.2f}x</b></td><td>√MSE</td></tr>
</table>
<hr style="margin:8px 0;border:none;border-top:1px solid #e5e7eb;">
<b>差距来源</b>：误差 e → MAE 贡献 |e|，MSE 贡献 e²。<br>
异常点 e 大 → e² 远大于 |e| → MSE 被一颗老鼠屎拉爆。
</div>""")

    else:  # D · 归因
        _slot1 = mo.ui.altair_chart(chart_bars.properties(width=420, height=380))
        if mag_mse > 5:
            _verdict = f'💥 MSE 被异常点绑架（{mag_mse:.1f}x）。数据有异常又来不及清洗 → <b>改用 MAE</b>。'
        elif mag_mse > 2:
            _verdict = f'⚠️ MSE 已明显放大（{mag_mse:.1f}x），继续拖远会更爆炸。'
        else:
            _verdict = '✅ 三指标差异不大。试试「极端」预设感受爆炸。'
        _slot2 = mo.md(f"""<div style="font-size:14px;line-height:1.7;padding:8px 12px;">
<b>何时选哪个指标？</b>
<table style="width:100%;border-collapse:collapse;margin:8px 0;font-size:14px;">
<tr style="border-bottom:2px solid #e5e7eb;"><th style="text-align:left;">场景</th><th style="text-align:left;">推荐</th><th style="text-align:left;">理由</th></tr>
<tr><td>数据干净</td><td>MSE</td><td>可微+凸，优化友好</td></tr>
<tr><td>有异常点</td><td>MAE</td><td>对 outlier 鲁棒</td></tr>
<tr><td>需要量纲一致</td><td>RMSE</td><td>√MSE，单位与 y 同</td></tr>
</table>
<hr style="margin:8px 0;border:none;border-top:1px solid #e5e7eb;">
<b>当前判定</b>：{_verdict}
</div>""")

    mo.hstack([_slot1, _slot2], gap=0.5, widths="equal", align="start")
    return


@app.cell
def panel_view(mag_mae, mag_mse, mag_rmse, mo, preset, x_out, y_out):
    # 数字面板（录屏内 · 紧凑一行）
    _c = lambda m: "#10b981" if m < 1.5 else ("#f59e0b" if m < 5 else "#dc2626")
    mo.md(f"""<div style="font-family:ui-monospace,monospace;font-size:13px;line-height:1.4;
        background:#f9fafb;border:1px solid #e5e7eb;border-radius:6px;padding:6px 12px;margin:0;">
    <b>预设</b> {preset.value} · 异常点 ({x_out:.1f}, {y_out:.1f}) &nbsp;|&nbsp;
    MAE <b style="color:{_c(mag_mae)}">{mag_mae:.2f}x</b> &nbsp;
    MSE <b style="color:{_c(mag_mse)}">{mag_mse:.2f}x</b> &nbsp;
    RMSE <b style="color:{_c(mag_rmse)}">{mag_rmse:.2f}x</b> &nbsp;|&nbsp;
    <span style="color:#6b7280">MSE 平方放大 → 对异常点最敏感</span>
    </div>""")
    return


@app.cell
def truth_hint(mo):
    # 真实参数（录屏外）
    mo.md("""<div style="background:#dbeafe;color:#1e40af;border-left:4px solid #3b82f6;
        padding:6px 14px;border-radius:6px;font-size:13px;line-height:1.4;margin:0;">
    🎯 <b>数据生成</b> y = 2x + 5 + N(0,1)（15 点）
    &nbsp;·&nbsp;
    <b>预设序列</b> 无异常 1x → 轻微 3x → 中等 8x → 极端 25x（MSE 放大倍数）
    </div>""")
    return


@app.cell
def shot_picker(shot):
    # 镜头切换器（录屏外）
    shot
    return


@app.cell
def narration(mo, shot):
    # 口播稿：按 shot 切换（录屏外）
    _scripts = {
        "A": """
**🎬 A · 散点+柱图 · 5 秒测试**（30 秒）

> "左边散点图：蓝点正常数据，红星异常点，红方块大小 = 误差²。
>  右边柱图：三个指标的 baseline（灰）vs current（蓝）。
>  一眼看出：MSE 蓝柱顶天，MAE 几乎不动。
>  这就是 MSE 对异常点最敏感的直觉证据。"
""",
        "B": """
**🎬 B · 残差表 · 看数字**（30 秒）

> "15 个正常点的 err² 都很小（~1-3），
>  异常点的 err² 直接飙到几百。
>  MSE = 平均所有 err² → 一个巨大的 err² 拉爆整体平均。
>  这就是「一颗老鼠屎坏一锅汤」。"
""",
        "C": """
**🎬 C · 倍数面板 · 定量对比**（30 秒）

> "MAE 放大 ~2x，MSE 放大 ~25x，RMSE 放大 ~5x。
>  差距来源：误差 e → MAE 贡献 |e|（线性），MSE 贡献 e²（平方）。
>  异常点 e=20 → |e|=20，e²=400。
>  平方放大是 MSE 敏感的数学根因。"
""",
        "D": """
**🎬 D · 归因 · 何时选谁**（30 秒）

> "结论：数据干净用 MSE（可微+凸优化友好）；
>  有异常点又来不及清洗 → 改用 MAE。
>  RMSE 是折中：保留 MSE 的平方敏感但量纲与 y 一致，报告常用。"

🔑 **关键**：选指标 = 选对异常点的容忍度
""",
    }
    _key = shot.value[0] if shot.value else "A"
    mo.md(_scripts.get(_key, "")).style(
        font_size="15px", line_height="1.6", margin="0", padding="14px 24px",
        background="#fffbeb", border_radius="8px", border_left="4px solid #fbbf24",
    )
    return


@app.cell
def layout_doc(mo):
    # 📐 录屏 grid 布局参考（position=null 隐藏）
    mo.accordion(
        {
            "📐 录屏布局参考（grid 设计意图）": mo.md(
                r"""
**目标 viewport**：1280×720（16:9 单屏不滚动）。
左 sidebar 4col 控件 + 中右 stage 双槽 + 底 panel；shot/narration/truth_hint 全部录屏外。

### 横屏骨架

```
   0           4                              32
y=0   ┌────── 标题 (h=3) ─────────────────────────┐
y=3   ├ controls ┬──── stage 双槽 (h=26) ────────┤
      │ 4col     │   slot1 + slot2 widths=equal  │
      │ preset   │   A: scatter | bar chart      │
      │ x_out    │   B: scatter | 残差表          │
      │ y_out    │   C: bar chart | 倍数面板      │
      │          │   D: bar chart | 归因文本      │
y=29  ├──────────┴──── panel (h=3) ──────────────┤  ← 720 内
y=32  ├──── 录屏外提示区 ──────────────────────────┤
y=36  ├──── shot_picker (h=3) ───────────────────┤
y=39  ├──── truth_hint (h=3) ────────────────────┤
y=42  ├──── narration 口播稿 (h=12) ─────────────┤
```

### 镜头脚本（Strategy B 双槽）

| # | slot1 | slot2 | 教学焦点 |
|---|---|---|---|
| **A** | scatter+squares | bar chart | 5 秒测试 MSE 顶天 |
| **B** | scatter | 残差表 | 数字钻入 err² 放大 |
| **C** | bar chart | 倍数面板 | 定量 MAE/MSE/RMSE 差距 |
| **D** | bar chart | 归因文本 | 何时选谁 |
                """
            )
        },
        multiple=False,
    )
    return


if __name__ == "__main__":
    app.run()
