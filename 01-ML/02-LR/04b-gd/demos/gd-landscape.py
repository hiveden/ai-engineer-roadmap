"""
梯度下降 Loss Landscape · 五视图交互可视化（02-LR 04b 核心 demo）

演示概念：梯度下降思想、学习率三态（小/合适/大）、参数优化几何直觉

启动：
  marimo edit --port 2758 gd-landscape.py
  marimo run  --port 2758 gd-landscape.py

视图（Strategy B 双槽 · 4 镜头）：
  A · 业务+收敛: chart_data | chart_loss_curve
  B · 滚球+地形: chart_w_slice | chart_terrain
  C · 地形+收敛: chart_terrain | chart_loss_curve
  D · 3D全景: fig3d | chart_terrain

玩法：
  1. 拖预设 dropdown 对比 6 种学习率场景
  2. 手动模式：拖 w/b/lr 滑块 → 轨迹实时重算
  3. 拖帧滑块 0→30 慢动作回放 GD 每一步
  4. 看徽章：发散/收敛/未收敛 三态实时反馈
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/gd-landscape.grid.json",
    css_file="custom.css",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import altair as alt
    import pandas as pd
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return alt, mo, np, pd, plt


@app.cell(hide_code=True)
def title(mo):
    mo.md(
        r"### 梯度下降 Loss Landscape · 拖帧看 GD 每一步 · 预设对比 lr 三态"
    ).style(margin="0", padding="4px 12px", font_size="15px", line_height="1.3")
    return


@app.cell
def _(np):
    rng = np.random.default_rng(42)
    N = 20
    x_data = np.linspace(-3, 3, N)
    w_true, b_true = 2.0, 1.0
    y_data = w_true * x_data + b_true + rng.normal(0, 1.2, size=N)
    return x_data, y_data


@app.cell
def _(mo):
    _S = dict(show_value=True, full_width=True)

    PRESETS = {
        "✋ 手动": None,
        "✓ 完美": (4.0, 4.0, 0.08),
        "🎯 长征": (5.5, -3.5, 0.05),
        "🐢 龟速": (4.0, 4.0, 0.005),
        "🌊 振荡": (-1.0, -3.0, 0.27),
        "💥 发散": (-1.0, -3.0, 0.4),
        "⛰️ 鞍点": (2.0, -3.5, 0.03),
    }
    preset = mo.ui.dropdown(options=PRESETS, value="✋ 手动", label="", full_width=True)
    frame = mo.ui.slider(0, 30, step=1, value=0, label="", **_S)
    w = mo.ui.slider(-2.0, 6.0, step=0.1, value=0.5, label="w₀", **_S)
    b = mo.ui.slider(-4.0, 6.0, step=0.1, value=-2.0, label="b₀", **_S)
    lr = mo.ui.slider(0.001, 0.3, step=0.001, value=0.05, label="lr", **_S)

    shot = mo.ui.dropdown(
        options=["A · 业务+收敛", "B · 滚球+地形", "C · 地形+收敛", "D · 3D全景"],
        value="A · 业务+收敛",
        label="🎬 镜头",
    )
    return b, frame, lr, preset, shot, w


@app.cell
def controls(b, frame, lr, mo, preset, w):
    _h = lambda s: mo.md(s).style(
        margin="0", padding="0", font_size="11px",
        font_weight="700", color="#6b7280", letter_spacing="0.05em",
    )
    _div = mo.md("").style(
        border_top="1px solid #e5e7eb", margin="4px 0", padding="0", height="1px",
    )
    mo.vstack(
        [_h("预设"), preset, _div, _h("参数"), w, b, lr, _div, _h("帧"), frame],
        gap=0,
        align="stretch",
    )
    return


@app.cell
def _(b, frame, lr, np, preset, w, x_data, y_data):
    # GD 计算：从起点 (w0, b0) 用 lr_use 跑 30 步
    if preset.value is None:
        _w0, _b0, _lr_use = w.value, b.value, lr.value
        is_preset = False
    else:
        _w0, _b0, _lr_use = preset.value
        is_preset = True

    _traj = [(_w0, _b0)]
    _cw, _cb = _w0, _b0
    for _ in range(30):
        _yp = _cw * x_data + _cb
        _err = _yp - y_data
        _gw = 2 * np.mean(_err * x_data)
        _gb = 2 * np.mean(_err)
        _cw -= _lr_use * _gw
        _cb -= _lr_use * _gb
        _cw = float(np.clip(_cw, -1e6, 1e6)) if np.isfinite(_cw) else 1e6
        _cb = float(np.clip(_cb, -1e6, 1e6)) if np.isfinite(_cb) else 1e6
        _traj.append((_cw, _cb))

    _f = min(frame.value, len(_traj) - 1)
    cur_w, cur_b = _traj[_f]

    _traj_arr = np.array(_traj)
    is_diverged = bool(np.any(np.abs(_traj_arr) > 1e3))
    _last_step = np.linalg.norm(_traj_arr[-1] - _traj_arr[-2])
    is_converged = bool(_last_step < 1e-4) and not is_diverged

    state = {
        "w": cur_w,
        "b": cur_b,
        "history": _traj[: _f + 1],
        "full_traj": _traj,
        "start": (_w0, _b0),
    }
    active_lr = _lr_use
    return active_lr, cur_b, cur_w, is_converged, is_diverged, is_preset, state


@app.cell
def _(cur_b, cur_w, np, x_data, y_data):
    y_pred_cur = cur_w * x_data + cur_b
    cur_loss = float(np.mean((y_pred_cur - y_data) ** 2))

    x_mean, y_mean = x_data.mean(), y_data.mean()
    w_opt = float(np.sum((x_data - x_mean) * (y_data - y_mean)) / np.sum((x_data - x_mean) ** 2))
    b_opt = float(y_mean - w_opt * x_mean)
    loss_opt = float(np.mean((w_opt * x_data + b_opt - y_data) ** 2))

    _err = y_pred_cur - y_data
    grad_w = float(2 * np.mean(_err * x_data))
    grad_b = float(2 * np.mean(_err))
    return b_opt, cur_loss, grad_b, grad_w, loss_opt, w_opt, y_pred_cur


@app.cell
def _(
    active_lr,
    b_opt,
    cur_b,
    cur_loss,
    cur_w,
    is_converged,
    is_diverged,
    is_preset,
    loss_opt,
    mo,
    w_opt,
):
    # 状态徽章面板
    if is_diverged:
        _color = "#ef4444"
    elif cur_loss - loss_opt < 0.05:
        _color = "#10b981"
    elif cur_loss - loss_opt < 0.5:
        _color = "#f59e0b"
    else:
        _color = "#ef4444"

    _mode_tag = (
        f'<span style="background:#fef3c7;color:#92400e;padding:2px 6px;border-radius:4px;font-size:11px;">预设·lr={active_lr}</span>'
        if is_preset
        else f'<span style="background:#dbeafe;color:#1e40af;padding:2px 6px;border-radius:4px;font-size:11px;">手动·lr={active_lr}</span>'
    )

    if is_diverged:
        _conv_tag = '<span style="background:#fee2e2;color:#991b1b;padding:2px 6px;border-radius:4px;font-size:11px;font-weight:bold;">⚠ 发散</span>'
    elif is_converged:
        _conv_tag = '<span style="background:#d1fae5;color:#065f46;padding:2px 6px;border-radius:4px;font-size:11px;">✓ 收敛</span>'
    else:
        _conv_tag = '<span style="background:#e5e7eb;color:#374151;padding:2px 6px;border-radius:4px;font-size:11px;">○ 未收敛</span>'

    _loss_str = f"{cur_loss:.3f}" if cur_loss < 1e6 else "∞"

    panel = mo.md(
        f"""<div style="font-family:ui-monospace,monospace; font-size:12px; line-height:1.4;
            background:#f9fafb; border:1px solid #e5e7eb; border-radius:6px;
            padding:4px 10px; margin:0; display:flex; gap:12px; align-items:center; flex-wrap:wrap;">
        {_mode_tag} {_conv_tag}
        <span><b>w={cur_w:.2f} b={cur_b:.2f}</b></span>
        <span style="color:{_color};font-weight:700">Loss={_loss_str}</span>
        <span style="color:#6b7280">最优 w*={w_opt:.2f} b*={b_opt:.2f} loss*={loss_opt:.3f}</span>
        </div>"""
    )
    return (panel,)


@app.cell
def _(alt, b_opt, cur_b, cur_w, pd, w_opt, x_data, y_data, y_pred_cur):
    # A · 业务图：数据 + 残差方块
    df = pd.DataFrame(
        {"x": x_data, "y": y_data, "y_pred": y_pred_cur, "err_sq": (y_data - y_pred_cur) ** 2}
    )
    _xr = [-3.5, 3.5]
    xs_cur = pd.DataFrame({"x": _xr, "y": [cur_w * v + cur_b for v in _xr]})
    xs_opt = pd.DataFrame({"x": _xr, "y": [w_opt * v + b_opt for v in _xr]})

    pts = (
        alt.Chart(df)
        .mark_circle(size=120, color="#1f77b4", stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-3.8, 3.8])),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[-9, 11])),
            tooltip=["x", "y", "y_pred", "err_sq"],
        )
    )
    squares = (
        alt.Chart(df)
        .mark_square(opacity=0.3, color="#ef4444", stroke="#ef4444", strokeWidth=1)
        .encode(x="x:Q", y="y:Q", size=alt.Size("err_sq:Q", scale=alt.Scale(range=[20, 3500]), legend=None))
    )
    _seg = []
    for _i in range(len(x_data)):
        _seg.append({"x": float(x_data[_i]), "y": float(y_data[_i]), "_p": _i})
        _seg.append({"x": float(x_data[_i]), "y": float(y_pred_cur[_i]), "_p": _i})
    rules = (
        alt.Chart(pd.DataFrame(_seg))
        .mark_line(color="#ef4444", opacity=0.5, strokeDash=[3, 2])
        .encode(x="x:Q", y="y:Q", detail="_p:N")
    )
    line_cur = alt.Chart(xs_cur).mark_line(color="#ef4444", strokeWidth=3).encode(x="x:Q", y="y:Q")
    line_opt = alt.Chart(xs_opt).mark_line(color="#10b981", strokeWidth=2, strokeDash=[6, 4]).encode(x="x:Q", y="y:Q")

    chart_data = (rules + squares + pts + line_cur + line_opt).properties(
        width=440, height=380, title="A · 红方块=误差² · 红线当前/绿虚最优"
    )
    return (chart_data,)


@app.cell
def _(
    b_opt,
    cur_b,
    cur_w,
    grad_b,
    grad_w,
    np,
    plt,
    state,
    w_opt,
    x_data,
    y_data,
):
    # C · 等高线地形图（matplotlib）
    ws_grid = np.linspace(-2, 6, 80)
    bs_grid = np.linspace(-4, 6, 80)
    W, B = np.meshgrid(ws_grid, bs_grid)
    Y_pred = W[..., None] * x_data + B[..., None]
    L = np.mean((Y_pred - y_data) ** 2, axis=-1)

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    levels = np.exp(np.linspace(np.log(L.min() + 0.1), np.log(L.max()), 18))
    ax.contourf(W, B, L, levels=levels, cmap="viridis_r", alpha=0.85)
    ax.contour(W, B, L, levels=levels, colors="white", linewidths=0.5, alpha=0.4)

    if len(state["history"]) > 1:
        hist = np.array(state["history"])
        hist_disp = np.clip(hist, [-2, -4], [6, 6])
        ax.plot(hist_disp[:, 0], hist_disp[:, 1], "o-", color="white", markersize=3, linewidth=1, alpha=0.85)

    if -2 <= cur_w <= 6 and -4 <= cur_b <= 6 and np.isfinite(grad_w) and np.isfinite(grad_b):
        _gnorm = np.hypot(grad_w, grad_b)
        if _gnorm > 1e-6:
            _ax = -grad_w / _gnorm * 0.6
            _ay = -grad_b / _gnorm * 0.6
            ax.annotate(
                "", xy=(cur_w + _ax, cur_b + _ay), xytext=(cur_w, cur_b),
                arrowprops={"arrowstyle": "->", "color": "#fbbf24", "lw": 2.2}, zorder=6,
            )

    _cw_disp = float(np.clip(cur_w, -2, 6))
    _cb_disp = float(np.clip(cur_b, -4, 6))
    ax.scatter([_cw_disp], [_cb_disp], s=200, c="#ef4444", edgecolor="white", linewidth=2, zorder=5, label="当前")
    ax.scatter([w_opt], [b_opt], s=160, c="#10b981", marker="D", edgecolor="white", linewidth=2, zorder=5, label="最优")

    ax.set_xlabel("w", fontsize=10)
    ax.set_ylabel("b", fontsize=10)
    ax.set_title("C · 地形图 · 白线=GD脚印 · 黄箭头=负梯度", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    chart_terrain = fig
    return (chart_terrain,)


@app.cell
def _(alt, cur_b, cur_w, np, pd, w_opt, x_data, y_data):
    # B1 · 1D 抛物线（固定 b，扫 w）
    ws_slice = np.linspace(-2, 6, 100)
    losses_w = np.array([np.mean((wv * x_data + cur_b - y_data) ** 2) for wv in ws_slice])
    df_w = pd.DataFrame({"w": ws_slice, "loss": losses_w})

    cur_loss_w = float(np.mean((cur_w * x_data + cur_b - y_data) ** 2))
    opt_loss_w = float(np.mean((w_opt * x_data + cur_b - y_data) ** 2))

    _parabola = alt.Chart(df_w).mark_line(color="#3b82f6", strokeWidth=2.5).encode(x="w", y="loss")
    _cur_dot = (
        alt.Chart(pd.DataFrame({"w": [cur_w], "loss": [cur_loss_w]}))
        .mark_circle(size=280, color="#ef4444", stroke="white", strokeWidth=2)
        .encode(x="w", y="loss")
    )
    _opt_dot = (
        alt.Chart(pd.DataFrame({"w": [w_opt], "loss": [opt_loss_w]}))
        .mark_point(shape="diamond", size=200, filled=True, color="#10b981", stroke="white", strokeWidth=2)
        .encode(x="w", y="loss")
    )
    chart_w_slice = (_parabola + _cur_dot + _opt_dot).properties(
        width=440, height=380, title=f"B1 · 固定 b={cur_b:.1f} 扫 w · 红球滚谷底"
    )
    return (chart_w_slice,)


@app.cell
def _(alt, frame, np, pd, state, x_data, y_data):
    # B2 · loss vs step 折线
    _full = state["full_traj"]
    _losses = []
    for hw, hb in _full:
        _l = float(np.mean((hw * x_data + hb - y_data) ** 2))
        _losses.append(min(_l, 1e6))
    df_loss = pd.DataFrame({"step": range(len(_full)), "loss": _losses})

    _line = (
        alt.Chart(df_loss)
        .mark_line(color="#8b5cf6", strokeWidth=2.5, point=alt.OverlayMarkDef(size=35))
        .encode(
            x=alt.X("step:Q", scale=alt.Scale(domain=[0, 30])),
            y=alt.Y("loss:Q", scale=alt.Scale(type="log", clamp=True), title="loss (log)"),
            tooltip=["step", "loss"],
        )
    )
    _f_idx = min(frame.value, len(_full) - 1)
    _cur_marker = (
        alt.Chart(pd.DataFrame({"step": [_f_idx], "loss": [max(_losses[_f_idx], 1e-9)]}))
        .mark_circle(size=280, color="#ef4444", stroke="white", strokeWidth=2)
        .encode(x="step:Q", y="loss:Q")
    )
    chart_loss_curve = (_line + _cur_marker).properties(
        width=440, height=380, title="B2 · loss vs step · 收敛曲线"
    )
    return (chart_loss_curve,)


@app.cell
def _(
    b_opt,
    cur_b,
    cur_loss,
    cur_w,
    loss_opt,
    np,
    state,
    w_opt,
    x_data,
    y_data,
):
    # D · 3D 曲面（plotly）
    import plotly.graph_objects as go

    ws_3d = np.linspace(-2, 6, 50)
    bs_3d = np.linspace(-4, 6, 50)
    W3, B3 = np.meshgrid(ws_3d, bs_3d)
    L3 = np.mean((W3[..., None] * x_data + B3[..., None] - y_data) ** 2, axis=-1)

    fig3d = go.Figure()
    fig3d.add_trace(
        go.Surface(
            x=ws_3d, y=bs_3d, z=L3, colorscale="Viridis", reversescale=True,
            opacity=0.85, showscale=False,
            contours={"z": {"show": True, "usecolormap": True, "project_z": True, "width": 1}},
            name="loss",
        )
    )

    if len(state["history"]) > 1:
        _hist = np.array(state["history"])
        _hist = np.clip(_hist, [-2, -4], [6, 6])
        _hist_loss = np.array([np.mean((hw * x_data + hb - y_data) ** 2) for hw, hb in _hist])
        fig3d.add_trace(
            go.Scatter3d(
                x=_hist[:, 0], y=_hist[:, 1], z=_hist_loss,
                mode="lines+markers", line={"color": "white", "width": 5},
                marker={"color": "white", "size": 3}, name="GD 轨迹",
            )
        )

    _z_cur = cur_loss if cur_loss < 1e6 else float(L3.max())
    fig3d.add_trace(
        go.Scatter3d(
            x=[float(np.clip(cur_w, -2, 6))], y=[float(np.clip(cur_b, -4, 6))], z=[_z_cur],
            mode="markers", marker={"color": "#ef4444", "size": 8, "line": {"color": "white", "width": 2}},
            name="当前",
        )
    )
    fig3d.add_trace(
        go.Scatter3d(
            x=[w_opt], y=[b_opt], z=[loss_opt], mode="markers",
            marker={"color": "#10b981", "size": 8, "symbol": "diamond", "line": {"color": "white", "width": 2}},
            name="最优",
        )
    )

    fig3d.update_layout(
        title=dict(text="D · 3D loss 曲面（拖拽旋转）", font=dict(size=12)),
        scene={"xaxis_title": "w", "yaxis_title": "b", "zaxis_title": "loss",
                "camera": {"eye": {"x": 1.6, "y": -1.6, "z": 1.0}}},
        height=380, margin={"l": 0, "r": 0, "t": 30, "b": 0},
    )
    return (fig3d,)


@app.cell
def stage(
    chart_data,
    chart_loss_curve,
    chart_terrain,
    chart_w_slice,
    fig3d,
    mo,
    shot,
):
    # 🎬 中央舞台 · Strategy B 双槽
    if shot.value.startswith("A"):
        _slot1 = mo.ui.altair_chart(chart_data)
        _slot2 = mo.ui.altair_chart(chart_loss_curve)
    elif shot.value.startswith("B"):
        _slot1 = mo.ui.altair_chart(chart_w_slice)
        _slot2 = chart_terrain
    elif shot.value.startswith("C"):
        _slot1 = chart_terrain
        _slot2 = mo.ui.altair_chart(chart_loss_curve)
    else:  # D
        _slot1 = mo.ui.plotly(fig3d)
        _slot2 = chart_terrain
    mo.hstack([_slot1, _slot2], gap=0.5, widths="equal", align="start")
    return


@app.cell
def panel_view(panel):
    panel
    return


@app.cell
def truth_hint(mo):
    mo.md("""
    <div style="background:#dbeafe;color:#1e40af;border-left:4px solid #3b82f6;
    padding:6px 14px;border-radius:6px;font-size:13px;line-height:1.4;margin:0;">
    🎯 <b>真值</b> w=2.0, b=1.0（数据生成）
    &nbsp;·&nbsp; GD 收敛后应逼近此值
    </div>
    """)
    return


@app.cell
def shot_picker(shot):
    shot
    return


@app.cell
def narration(mo, shot):
    _scripts = {
        "A": """
    **🎬 A · 业务+收敛**（45 秒）

    > "左图是真实数据 + 当前拟合线。红方块面积 = 误差²，MSE = 方块平均。
    >  右图是 loss vs step：每走一步 GD，loss 应该下降。
    >  选预设「✓ 完美」拖帧 0→30：看白线从远处滚到绿钻石，右图单调下降。
    >  选「💥 发散」：右图冲天柱，loss 爆炸。"

    🎯 对照：左图方块缩小 ⇔ 右图 loss 下降
    """,
        "B": """
    **🎬 B · 滚球+地形**（60 秒）

    > "左图把 b 固定，只让 w 变，画出 loss 抛物线 —— 红球沿曲线滚到谷底。
    >  右图是完整 2D 等高线：白线 = GD 脚印，黄箭头 = 负梯度（下一步方向）。
    >  1D 抛物线是 2D 等高线的一个「切片」。
    >  拖帧看：红球滚向谷底 ⇔ 白线走向绿钻石。"

    🔑 1D 切片是 2D 的投影直觉
    """,
        "C": """
    **🎬 C · 地形+收敛**（45 秒）

    > "左图 2D 等高线：看 GD 白线的路径形状。
    >  选「🌊 振荡」：白线锯齿摇摆 → 右图 loss 锯齿。
    >  选「🐢 龟速」：白线短短一段 → 右图下降缓慢。
    >  黄色箭头 = 负梯度方向 = GD 下一步要走的方向。"

    🔑 路径形状直接反映 lr 选择好坏
    """,
        "D": """
    **🎬 D · 3D全景**（60 秒）

    > "左图 3D 曲面：拖拽旋转看 loss 碗的立体形状。
    >  白线 = GD 轨迹从高处滚到碗底。
    >  右图 2D 等高线是 3D 的俯视投影。
    >  对照看：3D 高度 = 2D 颜色深浅。"

    🔑 3D 建立「碗底 = 最优」的立体直觉
    """,
    }
    _key = shot.value[0] if shot.value else "A"
    mo.md(_scripts.get(_key, "")).style(
        font_size="15px", line_height="1.6", margin="0", padding="14px 24px",
        background="#fffbeb", border_radius="8px",
        border_left="4px solid #fbbf24",
    )
    return


@app.cell
def layout_doc(mo):
    mo.accordion(
        {
            "📐 录屏布局参考（gd-landscape · 32col · maxWidth 1280）": mo.md(
                r"""
    ## Grid 布局（Strategy B 双槽 · 4 镜头）

    ```
       0     5                              32
    y=0  ┌────── 标题 (h=3) ─────────────────────┐
    y=3  ├ controls ┬──── stage 双槽 (h=26) ─────┤
     │ 5col     │   slot1 + slot2 equal      │
     │ 预设     │   A: chart_data | loss_curve│
     │ w₀/b₀/lr│   B: w_slice | terrain     │
     │ 帧       │   C: terrain | loss_curve  │
     │          │   D: fig3d | terrain       │
    y=29 ├──────────┴──── panel (h=3) ───────────┤  ← 录屏内
    y=32 └───── 录屏区 720px = y=36 ──────────────┘
    y=36 ├──── shot_picker (h=3) ────────────────┤  ← 录屏外
    y=39 ├──── truth_hint (h=3) ─────────────────┤
    y=42 ├──── narration 口播稿 (h=12) ──────────┤
    ```

    ### Cell → grid 映射（19 cell）

    | Idx | 内容 | position |
    |---|---|---|
    | 0  | imports | null |
    | 1  | title | [0, 0, 32, 3] |
    | 2  | data gen | null |
    | 3  | slider defs | null |
    | 4  | controls | [0, 3, 5, 26] |
    | 5  | GD computation | null |
    | 6  | loss/optimal | null |
    | 7  | panel build | null |
    | 8  | chart_data (A) | null |
    | 9  | chart_terrain (C) | null |
    | 10 | chart_w_slice (B1) | null |
    | 11 | chart_loss_curve (B2) | null |
    | 12 | fig3d (D) | null |
    | 13 | stage | [5, 3, 27, 26] |
    | 14 | panel_view | [0, 29, 32, 3] |
    | 15 | truth_hint | [0, 39, 32, 3] |
    | 16 | shot_picker | [0, 36, 8, 3] |
    | 17 | narration | [0, 42, 32, 12] |
    | 18 | layout_doc | null |
                """
            )
        },
        multiple=False,
    )
    return


if __name__ == "__main__":
    app.run()
