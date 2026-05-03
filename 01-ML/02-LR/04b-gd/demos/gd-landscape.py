"""
梯度下降 Loss Landscape · 五视图交互可视化（02-LR 04b 核心 demo）

演示概念：梯度下降思想、学习率三态（小/合适/大）、参数优化几何直觉

启动：marimo run --port 2731 gd-landscape.py

视图（A+B1+B2+C+D）：
  A · 业务图（数据点 + 残差方块 + 当前/最优拟合线）
  B1 · 1D 抛物线（固定 b 扫 w，红球往谷底滚）
  B2 · loss vs step 折线（GD 收敛轨迹 - 区分龟速/振荡/飞出）
  C · 2D 等高线 + GD 白线轨迹 + 负梯度箭头
  D · 3D 曲面 + GD 轨迹（拖拽旋转）

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


@app.cell
def _(mo):
    mo.md(
        """
    # 梯度下降 Loss Landscape · 五视图

    用直线 `y = w·x + b` 拟合 20 个点。**目标**：让 GD 自动找到 MSE 最小的 (w, b)。

    - **A 业务图**：红方块大小 = 该点误差²，MSE = 方块平均
    - **B1 1D 切片**：固定 b 扫 w，看「滚球」直觉
    - **B2 loss vs step**：GD 收敛曲线（龟速/振荡/飞出 一眼看出）
    - **C 等高线**：白线 = GD 脚印，红箭头 = 负梯度方向（下一步走向）
    - **D 3D 曲面**：拖拽旋转看立体感
    """
    )
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
    # 预设：(w0, b0, lr) · None 表示手动（用滑块当起点）
    # 注：lr=0.4 已可见振荡发散过程（review M1：原 0.5 直接 clip 到边界，看不到动）
    PRESETS = {
        "✋ 手动 (滑块=起点)": None,
        "✓ 完美 · 合适 lr+起点": (4.0, 4.0, 0.08),
        "🎯 远距离 · 长征": (5.5, -3.5, 0.05),
        "🐢 龟速 · lr 太小": (4.0, 4.0, 0.005),
        "🌊 振荡 · lr 太大": (-1.0, -3.0, 0.27),
        "💥 飞出去 · lr=0.4 发散": (-1.0, -3.0, 0.4),
        "⛰️ 鞍点状起步": (2.0, -3.5, 0.03),
    }
    preset = mo.ui.dropdown(
        options=PRESETS,
        value="✋ 手动 (滑块=起点)",
        label="预设场景",
    )
    # review S3：默认从第 0 帧开始，让用户拖滑块看动画感
    frame = mo.ui.slider(0, 30, step=1, value=0, label="帧 (0=起点, 30=终点)", show_value=True)

    w = mo.ui.slider(-2.0, 6.0, step=0.1, value=0.5, label="w 起点 (斜率)", show_value=True)
    b = mo.ui.slider(-4.0, 6.0, step=0.1, value=-2.0, label="b 起点 (截距)", show_value=True)
    lr = mo.ui.slider(0.001, 0.3, step=0.001, value=0.05, label="学习率", show_value=True)
    return b, frame, lr, preset, w


@app.cell
def _(b, frame, lr, mo, preset, w):
    mo.vstack(
        [
            mo.hstack([preset, frame], justify="start", gap=2),
            mo.hstack([mo.vstack([w, b]), lr], justify="start", gap=2),
        ],
        gap=1,
    )
    return


@app.cell
def _(b, frame, lr, np, preset, w, x_data, y_data):
    # 统一 GD 计算：从起点 (w0, b0) 用 lr_use 跑 30 步，frame 选哪一帧展示
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
        # clip 防 NaN/inf（飞出场景）
        _cw = float(np.clip(_cw, -1e6, 1e6)) if np.isfinite(_cw) else 1e6
        _cb = float(np.clip(_cb, -1e6, 1e6)) if np.isfinite(_cb) else 1e6
        _traj.append((_cw, _cb))

    _f = min(frame.value, len(_traj) - 1)
    cur_w, cur_b = _traj[_f]

    # 发散判据（review M1）：轨迹任意点超出 |1e3| → 已发散
    _traj_arr = np.array(_traj)
    is_diverged = bool(np.any(np.abs(_traj_arr) > 1e3))

    # 收敛判据（review S5）：最后两步差距 < 1e-4
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
    # 当前位置 loss + 闭式最优解
    y_pred_cur = cur_w * x_data + cur_b
    cur_loss = float(np.mean((y_pred_cur - y_data) ** 2))

    x_mean, y_mean = x_data.mean(), y_data.mean()
    w_opt = float(np.sum((x_data - x_mean) * (y_data - y_mean)) / np.sum((x_data - x_mean) ** 2))
    b_opt = float(y_mean - w_opt * x_mean)
    loss_opt = float(np.mean((w_opt * x_data + b_opt - y_data) ** 2))

    # 当前点的负梯度（C 图箭头用）
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
    # ============ 状态徽章 ============
    _gap = cur_loss - loss_opt
    if is_diverged:
        _color = "#ef4444"
    elif _gap < 0.05:
        _color = "#10b981"
    elif _gap < 0.5:
        _color = "#f59e0b"
    else:
        _color = "#ef4444"

    _mode_tag = (
        f'<span style="background:#fef3c7;color:#92400e;padding:2px 8px;border-radius:4px;font-size:12px;">预设模式 · lr={active_lr}</span>'
        if is_preset
        else f'<span style="background:#dbeafe;color:#1e40af;padding:2px 8px;border-radius:4px;font-size:12px;">手动模式 · lr={active_lr}</span>'
    )

    # review M1 必修：发散徽章
    if is_diverged:
        _conv_tag = '<span style="background:#fee2e2;color:#991b1b;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:bold;">⚠ 已发散</span>'
    elif is_converged:
        _conv_tag = '<span style="background:#d1fae5;color:#065f46;padding:2px 8px;border-radius:4px;font-size:12px;">✓ 已收敛</span>'
    else:
        _conv_tag = '<span style="background:#e5e7eb;color:#374151;padding:2px 8px;border-radius:4px;font-size:12px;">○ 未收敛</span>'

    # 安全显示 cur_loss（发散后可能爆炸）
    _loss_str = f"{cur_loss:.3f}" if cur_loss < 1e6 else "∞ (发散)"

    mo.md(
        f"""
    <div style="display:flex; gap:18px; align-items:center; font-family:ui-monospace,monospace; font-size:15px; flex-wrap:wrap;">
      <div>{_mode_tag}</div>
      <div>{_conv_tag}</div>
      <div>当前 <b>w={cur_w:.2f}, b={cur_b:.2f}</b></div>
      <div>Loss = <b style="color:{_color}; font-size:18px">{_loss_str}</b></div>
      <div style="color:#6b7280">最优 w*={w_opt:.2f}, b*={b_opt:.2f}, loss*={loss_opt:.3f}</div>
    </div>
    """
    )
    return


@app.cell
def _(alt, b_opt, cur_b, cur_w, pd, w_opt, x_data, y_data, y_pred_cur):
    # ============ A · 业务图：数据 + 残差方块 ============
    df = pd.DataFrame(
        {
            "x": x_data,
            "y": y_data,
            "y_pred": y_pred_cur,
            "err_sq": (y_data - y_pred_cur) ** 2,
        }
    )
    xs = pd.DataFrame({"x": [-3.5, 3.5]})
    xs["y_cur"] = cur_w * xs["x"] + cur_b
    xs["y_opt"] = w_opt * xs["x"] + b_opt

    pts = (
        alt.Chart(df)
        .mark_circle(size=140, color="#1f77b4", stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X("x", scale=alt.Scale(domain=[-3.8, 3.8])),
            y=alt.Y("y", scale=alt.Scale(domain=[-9, 11])),
            tooltip=["x", "y", "y_pred", "err_sq"],
        )
    )
    squares = (
        alt.Chart(df)
        .mark_square(opacity=0.35, color="#ef4444", stroke="#ef4444", strokeWidth=1)
        .encode(
            x="x",
            y="y",
            size=alt.Size("err_sq:Q", scale=alt.Scale(range=[20, 4000]), legend=None),
        )
    )
    rules = (
        alt.Chart(df)
        .mark_rule(color="#ef4444", opacity=0.5, strokeDash=[3, 2])
        .encode(x="x", y="y", y2="y_pred")
    )
    line_cur = alt.Chart(xs).mark_line(color="#ef4444", strokeWidth=3).encode(x="x", y="y_cur")
    line_opt = (
        alt.Chart(xs)
        .mark_line(color="#10b981", strokeWidth=2, strokeDash=[6, 4])
        .encode(x="x", y="y_opt")
    )

    chart_data = (rules + squares + pts + line_cur + line_opt).properties(
        width=380, height=320, title="A · 红方块大小 = 误差² · MSE = 方块平均"
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
    # ============ C · 等高线地形图（matplotlib） ============
    ws_grid = np.linspace(-2, 6, 80)
    bs_grid = np.linspace(-4, 6, 80)
    W, B = np.meshgrid(ws_grid, bs_grid)
    Y_pred = W[..., None] * x_data + B[..., None]
    L = np.mean((Y_pred - y_data) ** 2, axis=-1)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    levels = np.exp(np.linspace(np.log(L.min() + 0.1), np.log(L.max()), 18))
    ax.contourf(W, B, L, levels=levels, cmap="viridis_r", alpha=0.85)
    ax.contour(W, B, L, levels=levels, colors="white", linewidths=0.5, alpha=0.4)

    # GD 轨迹（白线）
    if len(state["history"]) > 1:
        hist = np.array(state["history"])
        # clip 到网格内显示，否则发散点会拉飞坐标轴
        hist_disp = np.clip(hist, [-2, -4], [6, 6])
        ax.plot(
            hist_disp[:, 0],
            hist_disp[:, 1],
            "o-",
            color="white",
            markersize=3,
            linewidth=1,
            alpha=0.85,
        )

    # 负梯度箭头（review S1：把「负梯度方向」显性化）
    # 仅在当前点在网格范围内、梯度有限时画
    if -2 <= cur_w <= 6 and -4 <= cur_b <= 6 and np.isfinite(grad_w) and np.isfinite(grad_b):
        # 单位化箭头（视觉长度固定 0.6），避免梯度过大溢出
        _gnorm = np.hypot(grad_w, grad_b)
        if _gnorm > 1e-6:
            _ax = -grad_w / _gnorm * 0.6
            _ay = -grad_b / _gnorm * 0.6
            ax.annotate(
                "",
                xy=(cur_w + _ax, cur_b + _ay),
                xytext=(cur_w, cur_b),
                arrowprops={"arrowstyle": "->", "color": "#fbbf24", "lw": 2.2},
                zorder=6,
            )

    # 当前点 + 最优点
    _cw_disp = float(np.clip(cur_w, -2, 6))
    _cb_disp = float(np.clip(cur_b, -4, 6))
    ax.scatter(
        [_cw_disp],
        [_cb_disp],
        s=250,
        c="#ef4444",
        edgecolor="white",
        linewidth=2,
        zorder=5,
        label="当前",
    )
    ax.scatter(
        [w_opt],
        [b_opt],
        s=200,
        c="#10b981",
        marker="D",
        edgecolor="white",
        linewidth=2,
        zorder=5,
        label="最优",
    )

    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.set_title("C · 地形图 · 白线=GD脚印 · 黄箭头=负梯度方向", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    chart_terrain = fig
    return (chart_terrain,)


@app.cell
def _(alt, cur_b, cur_w, np, pd, w_opt, x_data, y_data):
    # ============ B1 · 1D 抛物线（固定 b，扫 w） ============
    ws_slice = np.linspace(-2, 6, 100)
    losses_w = np.array([np.mean((wv * x_data + cur_b - y_data) ** 2) for wv in ws_slice])
    df_w = pd.DataFrame({"w": ws_slice, "loss": losses_w})

    cur_loss_w = float(np.mean((cur_w * x_data + cur_b - y_data) ** 2))
    opt_loss_w = float(np.mean((w_opt * x_data + cur_b - y_data) ** 2))

    _parabola = alt.Chart(df_w).mark_line(color="#3b82f6", strokeWidth=2.5).encode(x="w", y="loss")
    _cur_dot = (
        alt.Chart(pd.DataFrame({"w": [cur_w], "loss": [cur_loss_w]}))
        .mark_circle(size=300, color="#ef4444", stroke="white", strokeWidth=2)
        .encode(x="w", y="loss")
    )
    _opt_dot = (
        alt.Chart(pd.DataFrame({"w": [w_opt], "loss": [opt_loss_w]}))
        .mark_point(
            shape="diamond",
            size=200,
            filled=True,
            color="#10b981",
            stroke="white",
            strokeWidth=2,
        )
        .encode(x="w", y="loss")
    )
    chart_w_slice = (_parabola + _cur_dot + _opt_dot).properties(
        width=380,
        height=240,
        title=f"B1 · 固定 b={cur_b:.2f}，只动 w · 红球往谷底滚",
    )
    return (chart_w_slice,)


@app.cell
def _(alt, frame, np, pd, state, x_data, y_data):
    # ============ B2 · loss vs step 折线（review S2：替代「扫 b」抛物线） ============
    # 全 30 步的 loss 序列，让 4 种典型场景立刻可辨：
    #   完美=单调下降快收敛 / 龟速=单调下降但慢 / 振荡=锯齿 / 飞出=冲天柱
    _full = state["full_traj"]
    _losses = []
    for hw, hb in _full:
        _l = float(np.mean((hw * x_data + hb - y_data) ** 2))
        # 显示用 cap 到 1e6，否则发散后图被拉飞
        _losses.append(min(_l, 1e6))
    df_loss = pd.DataFrame({"step": range(len(_full)), "loss": _losses})

    # 用 log scale Y，让 0.001 ~ 1e6 都能看到
    _line = (
        alt.Chart(df_loss)
        .mark_line(color="#8b5cf6", strokeWidth=2.5, point=alt.OverlayMarkDef(size=40))
        .encode(
            x=alt.X("step:Q", scale=alt.Scale(domain=[0, 30])),
            y=alt.Y(
                "loss:Q",
                scale=alt.Scale(type="log", clamp=True),
                title="loss (log)",
            ),
            tooltip=["step", "loss"],
        )
    )

    # 当前帧高亮（红圆）
    _f_idx = min(frame.value, len(_full) - 1)
    _cur_marker = (
        alt.Chart(pd.DataFrame({"step": [_f_idx], "loss": [max(_losses[_f_idx], 1e-9)]}))
        .mark_circle(size=300, color="#ef4444", stroke="white", strokeWidth=2)
        .encode(x="step:Q", y="loss:Q")
    )

    chart_loss_curve = (_line + _cur_marker).properties(
        width=380,
        height=240,
        title="B2 · loss vs step · 收敛曲线（红=当前帧）",
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
    # ============ D · 3D 曲面（plotly · 拖拽旋转） ============
    import plotly.graph_objects as go

    ws_3d = np.linspace(-2, 6, 50)
    bs_3d = np.linspace(-4, 6, 50)
    W3, B3 = np.meshgrid(ws_3d, bs_3d)
    L3 = np.mean((W3[..., None] * x_data + B3[..., None] - y_data) ** 2, axis=-1)

    fig3d = go.Figure()
    fig3d.add_trace(
        go.Surface(
            x=ws_3d,
            y=bs_3d,
            z=L3,
            colorscale="Viridis",
            reversescale=True,
            opacity=0.85,
            showscale=False,
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
                x=_hist[:, 0],
                y=_hist[:, 1],
                z=_hist_loss,
                mode="lines+markers",
                line={"color": "white", "width": 5},
                marker={"color": "white", "size": 3},
                name="GD 轨迹",
            )
        )

    # 当前点（clip 到网格 + 安全 z）
    _z_cur = cur_loss if cur_loss < 1e6 else float(L3.max())
    fig3d.add_trace(
        go.Scatter3d(
            x=[float(np.clip(cur_w, -2, 6))],
            y=[float(np.clip(cur_b, -4, 6))],
            z=[_z_cur],
            mode="markers",
            marker={"color": "#ef4444", "size": 10, "line": {"color": "white", "width": 2}},
            name="当前",
        )
    )
    fig3d.add_trace(
        go.Scatter3d(
            x=[w_opt],
            y=[b_opt],
            z=[loss_opt],
            mode="markers",
            marker={
                "color": "#10b981",
                "size": 10,
                "symbol": "diamond",
                "line": {"color": "white", "width": 2},
            },
            name="最优",
        )
    )

    fig3d.update_layout(
        title="D · 3D loss 曲面（拖拽旋转 / 滚轮缩放 / 双击复位）",
        scene={
            "xaxis_title": "w",
            "yaxis_title": "b",
            "zaxis_title": "loss",
            "camera": {"eye": {"x": 1.6, "y": -1.6, "z": 1.0}},
        },
        height=520,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return (fig3d,)


@app.cell
def _(chart_data, mo):
    mo.ui.altair_chart(chart_data)
    return


@app.cell
def _(chart_terrain):
    chart_terrain
    return


@app.cell
def _(chart_w_slice, mo):
    mo.ui.altair_chart(chart_w_slice)
    return


@app.cell
def _(chart_loss_curve, mo):
    mo.ui.altair_chart(chart_loss_curve)
    return


@app.cell
def _(fig3d):
    fig3d
    return


@app.cell
def _(mo):
    mo.md(
        """
    ---
    ### 玩法 · 5 步看懂梯度下降

    1. **先选预设「✓ 完美」**：拖帧滑块 0→30，看白线从起点滚到绿钻石，B2 单调下降
    2. **切「🐢 龟速」**：30 步还远未到底，B2 缓慢下降未触底 → lr 太小的代价
    3. **切「🌊 振荡」**：B2 出现锯齿，C 白线左右摇晃 → lr 过大但还没飞
    4. **切「💥 飞出去 lr=0.4」**：徽章变红「⚠ 已发散」，B2 冲天柱 → lr 太大彻底失控
    5. **切回「✋ 手动」**：自己拖 w/b/lr 滑块找出"刚好"的学习率

    **看 C 图黄色箭头**：那是「负梯度方向」 = GD 下一步要走的方向。在最优点附近箭头会变短、消失。
    """
    )
    return


@app.cell
def _(mo):
    # ===== 📐 录屏 grid 布局参考（开发用 · 录屏隐藏）=====
    mo.accordion(
        {
            "📐 Grid 布局参考（gd-landscape · 24×20 · maxWidth 1400）": mo.md(
                r"""
## 📐 Grid 布局（5 视图 · 16:9 友好）

```
        0                  12                  24
y=0   ┌──────── 标题 mo.md（h=3）──────────────────┐
y=3   ├──────── 控件区 hstack（h=4）───────────────┤
      │  preset · frame · w · b · lr               │
y=7   ├──────── 状态徽章 mo.md（h=2）──────────────┤
      │  模式徽章 · 收敛徽章 · w/b · loss · 最优    │
y=9   ├──────── A 业务图 ─────┬──── C 等高线 ──────┤
      │  数据点 + 残差方块     │  matplotlib 地形图 │
      │  当前/最优拟合线       │  白线 GD 脚印       │
      │  (h=18, w=12)          │  黄箭头 负梯度      │
y=27  ├──────── B1 抛物线 ────┬──── B2 loss/step ──┤
      │  固定 b 扫 w           │  loss vs step       │
      │  红球往谷底滚          │  log scale + 红点   │
      │  (h=14, w=12)          │  (h=14, w=12)       │
y=41  ├──────── D · 3D loss 曲面（plotly） ─────────┤
      │  拖拽旋转 / 滚轮缩放 / 双击复位             │
      │  (h=24 ≈ 480px, w=24 整行)                  │
y=65  ├──────── 玩法 5 步 mo.md（h=8）─────────────┤
y=73  └────────────────────────────────────────────┘
```

### Cell → grid 映射（19 业务 cell + 1 本参考 cell）

| Idx | 内容 | position |
|---|---|---|
| 0  | imports                  | null（隐藏）|
| 1  | 标题 mo.md               | [0, 0, 24, 3] |
| 2  | 数据生成                 | null |
| 3  | 滑块定义                 | null |
| 4  | 控件 vstack/hstack       | [0, 3, 24, 4] |
| 5  | GD 计算                  | null |
| 6  | loss / 最优解            | null |
| 7  | 状态徽章 mo.md           | [0, 7, 24, 2] |
| 8  | chart_data 定义          | null |
| 9  | chart_terrain 定义       | null |
| 10 | chart_w_slice 定义       | null |
| 11 | chart_loss_curve 定义    | null |
| 12 | fig3d 定义               | null |
| 13 | A · chart_data 渲染      | [0, 9, 12, 18] |
| 14 | C · chart_terrain 渲染   | [12, 9, 12, 18] |
| 15 | B1 · chart_w_slice 渲染  | [0, 27, 12, 14] |
| 16 | B2 · chart_loss_curve 渲染 | [12, 27, 12, 14] |
| 17 | D · fig3d 渲染           | [0, 41, 24, 24] |
| 18 | 玩法 5 步 mo.md          | [0, 65, 24, 8] |
| 19 | 本 ASCII 参考 cell       | null（隐藏）|

### 设计意图

- **第一行 A + C**：业务图（直观看拟合）+ 地形图（看 GD 路径）= 「现实↔参数空间」对照
- **第二行 B1 + B2**：抛物线切片 + loss 曲线 = 「单维直觉」+「时间维收敛」
- **第三行 D 整行**：3D 曲面留足旋转空间（480px ≈ rowHeight 20 × 24）
- **maxWidth 1400**：5 视图信息密度高，比 1280 多 9% 横向空间
- **录屏切镜头**：A→C→B1→B2→D 五段，每段聚焦一个视图
                """
            )
        },
        multiple=False,
    )
    return


if __name__ == "__main__":
    app.run()
