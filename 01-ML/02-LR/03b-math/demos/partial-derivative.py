"""
偏导 + 极值 · 纯数学交互 demo · 03b-math 配套（不带 LR 数据）

【设计核心 · 单核心隐喻 = 几何切片】
所有视图共用同一函数对象 f(x,y) + 同一组 (x_cur, y_cur)：
  - 视图 1 = f(x, y_cur) 沿 x 切片的曲线 + 切线（斜率 = ∂f/∂x）
  - 视图 2 = f(x,y) 等高线 + 当前点 + 梯度箭头（俯视）
  - 视图 2-y = f(x_cur, y) 沿 y 切片（对偶，斜率 = ∂f/∂y）
  - 视图 3 = f(x,y) 3D 曲面（立体直觉）
所有视图都是同一函数同一参数的不同坐标镜像 → 拖动滑块四视图同步。

录屏布局（_5-layout-guide Strategy B 双槽）：
- A · ∂f/∂x 几何定义：slot1 = x-slice + tangent，slot2 = 几何定义文字
- B · 对偶切片：slot1 = x-slice，slot2 = y-slice（对比两个方向的偏导）
- C · 俯视+立体：slot1 = 等高线+梯度箭头，slot2 = 3D 曲面
- D · 切片↔全景：slot1 = x-slice，slot2 = 等高线（切片对应全景中的位置）

跑：
  cd 01-ML/02-LR/03b-math/demos && marimo run --port 2762 partial-derivative.py
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/partial-derivative.grid.json",
    css_file="custom.css",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import altair as alt
    import pandas as pd
    import plotly.graph_objects as go

    return alt, go, mo, np, pd


@app.cell(hide_code=True)
def title(mo):
    mo.md(
        r"### 偏导 + 极值 · 切一刀看斜率 · 所有方向导数=0 ⇒ 极值点"
    ).style(margin="0", padding="4px 12px", font_size="15px", line_height="1.3")
    return


@app.cell
def _():
    # ===== 函数定义 + 解析偏导 + 极值点 =====
    def _bowl(x, y):
        return x ** 2 + y ** 2

    def _bowl_dx(x, y):
        return 2 * x

    def _bowl_dy(x, y):
        return 2 * y

    def _shifted(x, y):
        return (x - 1) ** 2 + (y - 2) ** 2

    def _shifted_dx(x, y):
        return 2 * (x - 1)

    def _shifted_dy(x, y):
        return 2 * (y - 2)

    def _saddle(x, y):
        return x ** 2 - y ** 2

    def _saddle_dx(x, y):
        return 2 * x

    def _saddle_dy(x, y):
        return -2 * y

    func_table = {
        "碗": (_bowl, _bowl_dx, _bowl_dy, (0.0, 0.0), "极小（碗底）", "x²+y²"),
        "偏移碗": (_shifted, _shifted_dx, _shifted_dy, (1.0, 2.0), "极小（偏移）", "(x-1)²+(y-2)²"),
        "鞍点": (_saddle, _saddle_dx, _saddle_dy, (0.0, 0.0), "鞍点（非极值）", "x²-y²"),
    }
    return (func_table,)


@app.cell
def _(mo):
    # ===== 滑块 + shot dropdown =====
    _S = dict(show_value=True, full_width=True)
    func_choice = mo.ui.dropdown(
        options=["碗", "偏移碗", "鞍点"],
        value="鞍点",
        label="f",
    )
    x_cur = mo.ui.slider(-3.0, 3.0, step=0.1, value=1.5, label="x", **_S)
    y_cur = mo.ui.slider(-3.0, 3.0, step=0.1, value=1.5, label="y", **_S)

    shot = mo.ui.dropdown(
        options=["A · ∂f/∂x", "B · 对偶", "C · 全景", "D · 切片↔全景"],
        value="A · ∂f/∂x",
        label="🎬 镜头",
    )
    return func_choice, shot, x_cur, y_cur


@app.cell
def controls(func_choice, mo, x_cur, y_cur):
    # 控件组合（sidebar 4col）
    _h = lambda s: mo.md(s).style(
        margin="0", padding="0", font_size="11px",
        font_weight="700", color="#6b7280", letter_spacing="0.05em",
    )
    _div = mo.md("").style(
        border_top="1px solid #e5e7eb", margin="4px 0", padding="0", height="1px",
    )
    mo.vstack(
        [_h("函数"), func_choice, _div, _h("位置"), x_cur, y_cur],
        gap=0,
        align="stretch",
    )
    return


@app.cell
def _(func_choice, func_table, x_cur, y_cur):
    # 计算当前值
    import math as _math

    _key = func_choice.value
    f, fx, fy, opt_pt, opt_label, f_latex = func_table[_key]
    cur_x = float(x_cur.value)
    cur_y = float(y_cur.value)
    cur_fv = float(f(cur_x, cur_y))
    cur_dx = float(fx(cur_x, cur_y))
    cur_dy = float(fy(cur_x, cur_y))
    cur_gnorm = _math.sqrt(cur_dx ** 2 + cur_dy ** 2)
    return (
        cur_dx,
        cur_dy,
        cur_fv,
        cur_gnorm,
        cur_x,
        cur_y,
        f,
        f_latex,
        opt_label,
        opt_pt,
    )


@app.cell
def _(
    cur_dx,
    cur_dy,
    cur_fv,
    cur_gnorm,
    cur_x,
    cur_y,
    f_latex,
    mo,
    opt_label,
    opt_pt,
):
    # ===== 数字面板（录屏内）=====
    if cur_gnorm < 0.05:
        _badge = '<span style="background:#d1fae5;color:#065f46;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600">极值点</span>'
        _color = "#10b981"
    elif cur_gnorm < 0.5:
        _badge = '<span style="background:#fef3c7;color:#92400e;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600">靠近</span>'
        _color = "#f59e0b"
    else:
        _badge = '<span style="background:#fee2e2;color:#991b1b;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600">远离</span>'
        _color = "#ef4444"

    _xo, _yo = opt_pt
    _panel_md = f"""
    <div style="font-family:ui-monospace,monospace; font-size:13px; line-height:1.45;
        background:#f9fafb; border:1px solid #e5e7eb; border-radius:6px;
        padding:6px 12px; margin:0;">
    {_badge} &nbsp;
    <b>f={f_latex}</b> &nbsp;
    (x,y)=({cur_x:.2f},{cur_y:.2f}) &nbsp;
    f=<b style="color:{_color}">{cur_fv:.3f}</b> &nbsp;|&nbsp;
    ∂f/∂x=<b>{cur_dx:+.3f}</b> &nbsp;
    ∂f/∂y=<b>{cur_dy:+.3f}</b> &nbsp;
    ‖∇f‖=<b>{cur_gnorm:.3f}</b> &nbsp;|&nbsp;
    <span style="color:#6b7280">{opt_label} 在({_xo:.1f},{_yo:.1f})</span>
    </div>
    """
    panel = mo.md(_panel_md)
    return (panel,)


@app.cell
def _(alt, cur_dx, cur_x, cur_y, f, np, opt_pt, pd):
    # ============ chart_1d · 沿 x 切片 + 切线 ============
    _xs = np.linspace(-3.2, 3.2, 200)
    _zs = f(_xs, cur_y)
    _df = pd.DataFrame({"x": _xs, "z": _zs})

    _z_cur = float(f(cur_x, cur_y))
    _slope = cur_dx

    _tan_xs = np.linspace(cur_x - 1.5, cur_x + 1.5, 30)
    _tan_zs = _slope * (_tan_xs - cur_x) + _z_cur
    _df_tan = pd.DataFrame({"x": _tan_xs, "z": _tan_zs})

    _df_red = pd.DataFrame({"x": [cur_x], "z": [_z_cur]})
    _xo, _yo = opt_pt
    _df_opt = pd.DataFrame({"x": [_xo], "z": [float(f(_xo, cur_y))]})

    _curve = (
        alt.Chart(_df)
        .mark_line(color="#3b82f6", strokeWidth=2.8)
        .encode(
            x=alt.X("x", scale=alt.Scale(domain=[-3.2, 3.2]), title="x"),
            y=alt.Y("z", title="f(x, y_cur)"),
        )
    )
    _tan_line = (
        alt.Chart(_df_tan)
        .mark_line(color="#f97316", strokeWidth=2.8, strokeDash=[5, 3])
        .encode(x="x", y="z")
    )
    _red_dot = (
        alt.Chart(_df_red)
        .mark_circle(size=320, color="#ef4444", stroke="white", strokeWidth=2)
        .encode(x="x", y="z")
    )
    _opt_dot = (
        alt.Chart(_df_opt)
        .mark_point(shape="diamond", size=260, filled=True, color="#10b981", stroke="black", strokeWidth=1.5)
        .encode(x="x", y="z")
    )

    chart_1d = (
        (_curve + _tan_line + _red_dot + _opt_dot)
        .resolve_scale(color="independent")
        .properties(
            width=480,
            height=400,
            title=f"沿 x 切片 · ∂f/∂x={_slope:+.2f}（橙虚=切线）",
        )
    )
    return (chart_1d,)


@app.cell
def _(alt, cur_dy, cur_x, cur_y, f, np, opt_pt, pd):
    # ============ chart_slice_y · 沿 y 切片 ============
    _ys = np.linspace(-3.2, 3.2, 200)
    _zs = f(cur_x, _ys)
    _df = pd.DataFrame({"y": _ys, "z": _zs})

    _z_cur = float(f(cur_x, cur_y))
    _slope_y = cur_dy

    _tan_ys = np.linspace(cur_y - 1.2, cur_y + 1.2, 30)
    _tan_zs = _slope_y * (_tan_ys - cur_y) + _z_cur
    _df_tan = pd.DataFrame({"y": _tan_ys, "z": _tan_zs})

    _, _yo = opt_pt
    _df_red = pd.DataFrame({"y": [cur_y], "z": [_z_cur]})
    _df_opt = pd.DataFrame({"y": [_yo], "z": [float(f(cur_x, _yo))]})

    _curve = (
        alt.Chart(_df)
        .mark_line(color="#8b5cf6", strokeWidth=2.5)
        .encode(
            x=alt.X("y", scale=alt.Scale(domain=[-3.2, 3.2]), title="y"),
            y=alt.Y("z", title="f(x_cur, y)"),
        )
    )
    _tan_line = (
        alt.Chart(_df_tan)
        .mark_line(color="#f97316", strokeWidth=2.5, strokeDash=[5, 3])
        .encode(x="y", y="z")
    )
    _red_dot = (
        alt.Chart(_df_red)
        .mark_circle(size=260, color="#ef4444", stroke="white", strokeWidth=2)
        .encode(x="y", y="z")
    )
    _opt_dot = (
        alt.Chart(_df_opt)
        .mark_point(shape="diamond", size=200, filled=True, color="#10b981", stroke="black", strokeWidth=1.5)
        .encode(x="y", y="z")
    )

    chart_slice_y = (
        (_curve + _tan_line + _red_dot + _opt_dot)
        .resolve_scale(color="independent")
        .properties(
            width=480,
            height=400,
            title=f"沿 y 切片 · ∂f/∂y={_slope_y:+.2f}（橙虚=切线）",
        )
    )
    return (chart_slice_y,)


@app.cell
def _(alt, cur_dx, cur_dy, cur_x, cur_y, f, np, opt_pt, pd):
    # ============ chart_contour · 等高线 + 梯度箭头 ============
    _xs = np.linspace(-3.2, 3.2, 60)
    _ys = np.linspace(-3.2, 3.2, 60)
    _X, _Y = np.meshgrid(_xs, _ys)
    _Z = f(_X, _Y)

    _df_grid = pd.DataFrame({"x": _X.ravel(), "y": _Y.ravel(), "z": _Z.ravel()})

    _xo, _yo = opt_pt

    _heat = (
        alt.Chart(_df_grid)
        .mark_rect()
        .encode(
            x=alt.X("x:Q", bin=alt.Bin(maxbins=60), title="x"),
            y=alt.Y("y:Q", bin=alt.Bin(maxbins=60), title="y"),
            color=alt.Color("z:Q", scale=alt.Scale(scheme="viridis"), legend=None),
        )
    )

    _df_red = pd.DataFrame({"x": [cur_x], "y": [cur_y]})
    _red_cross = (
        alt.Chart(_df_red)
        .mark_point(shape="cross", size=400, color="#ef4444", filled=True, strokeWidth=3)
        .encode(x="x", y="y")
    )

    # 梯度箭头
    _scale = 0.15
    _df_arrow = pd.DataFrame({
        "x": [cur_x, cur_x + cur_dx * _scale],
        "y": [cur_y, cur_y + cur_dy * _scale],
    })
    _arrow = (
        alt.Chart(_df_arrow)
        .mark_line(color="#fbbf24", strokeWidth=3)
        .encode(x="x", y="y")
    )
    _df_arrow_head = pd.DataFrame({
        "x": [cur_x + cur_dx * _scale],
        "y": [cur_y + cur_dy * _scale],
    })
    _arrow_head = (
        alt.Chart(_df_arrow_head)
        .mark_point(shape="triangle", size=160, color="#fbbf24", filled=True)
        .encode(x="x", y="y")
    )

    _df_opt = pd.DataFrame({"x": [_xo], "y": [_yo]})
    _opt_diamond = (
        alt.Chart(_df_opt)
        .mark_point(shape="diamond", size=240, filled=True, color="#10b981", stroke="black", strokeWidth=1.5)
        .encode(x="x", y="y")
    )

    chart_contour = (
        (_heat + _red_cross + _arrow + _arrow_head + _opt_diamond)
        .resolve_scale(color="independent")
        .properties(
            width=480,
            height=400,
            title="等高线 · 黄箭头=∇f · 绿钻=极值",
        )
    )
    return (chart_contour,)


@app.cell
def _(cur_x, cur_y, f, go, np, opt_pt):
    # ============ fig_3d · 3D Surface ============
    _xs = np.linspace(-3.0, 3.0, 50)
    _ys = np.linspace(-3.0, 3.0, 50)
    _X, _Y = np.meshgrid(_xs, _ys)
    _Z = f(_X, _Y)

    _zc = float(f(cur_x, cur_y))
    _xo, _yo = opt_pt
    _zo = float(f(_xo, _yo))

    fig_3d = go.Figure()
    fig_3d.add_trace(
        go.Surface(
            x=_xs, y=_ys, z=_Z,
            colorscale="Viridis", opacity=0.85, showscale=False,
            contours={"z": {"show": True, "usecolormap": True, "project_z": True}},
        )
    )
    fig_3d.add_trace(
        go.Scatter3d(
            x=[cur_x], y=[cur_y], z=[_zc],
            mode="markers",
            marker={"size": 8, "color": "#ef4444", "line": {"color": "white", "width": 2}},
            name="当前点",
        )
    )
    fig_3d.add_trace(
        go.Scatter3d(
            x=[_xo], y=[_yo], z=[_zo],
            mode="markers",
            marker={"size": 9, "color": "#10b981", "symbol": "diamond", "line": {"color": "black", "width": 1.5}},
            name="极值点",
        )
    )
    fig_3d.update_layout(
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        height=450,
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "f(x,y)",
            "camera": {"eye": {"x": 1.6, "y": -1.6, "z": 1.0}},
        },
        title="3D 曲面 · 红=当前 绿钻=极值/鞍点",
        font={"family": "PingFang SC, Arial Unicode MS"},
        showlegend=False,
    )
    return (fig_3d,)


@app.cell
def truth_hint(f_latex, mo, opt_label, opt_pt):
    _xo, _yo = opt_pt
    mo.md(
        f"""<div style="background:#dbeafe;color:#1e40af;border-left:4px solid #3b82f6;
        padding:6px 14px;border-radius:6px;font-size:13px;line-height:1.4;margin:0;">
        🎯 <b>当前函数</b> f={f_latex} &nbsp;·&nbsp;
        {opt_label} 在 ({_xo:.1f},{_yo:.1f}) &nbsp;·&nbsp;
        拖 x,y 到该点 → ‖∇f‖→0
        </div>"""
    )
    return


@app.cell
def shot_picker(shot):
    shot
    return


@app.cell
def stage(
    chart_1d,
    chart_contour,
    chart_slice_y,
    cur_dx,
    cur_x,
    cur_y,
    fig_3d,
    mo,
    shot,
):
    # 🎬 中央舞台 · Strategy B 双槽
    if shot.value.startswith("A"):
        # A: x-slice + 几何定义文字
        _slot1 = mo.ui.altair_chart(chart_1d)
        _slot2 = mo.md(
            f"""
    **偏导 = 切片的切线斜率**

    把 f(x,y) 在 **y={cur_y:.2f}** 处沿 x 切一刀
    → 1D 曲线 f(x, {cur_y:.2f})

    这条曲线在 **x={cur_x:.2f}** 处的切线斜率：

    $$\\frac{{\\partial f}}{{\\partial x}} = {cur_dx:+.3f}$$

    - 橙虚线 = 切线（斜率即偏导）
    - 红点 = 当前位置
    - 绿钻 = 极值在该切片投影

    **拖 y** → 整条曲线形状变
    **拖 x** → 红点沿曲线滑
    """
        ).style(font_size="14px", line_height="1.55", padding="12px 16px")

    elif shot.value.startswith("B"):
        # B: x-slice vs y-slice（对偶对照 · 双图 width 显式覆盖避免溢出）
        _slot1 = mo.ui.altair_chart(chart_1d.properties(width=440, height=400))
        _slot2 = mo.ui.altair_chart(chart_slice_y.properties(width=440, height=400))

    elif shot.value.startswith("C"):
        # C: 等高线 + 3D
        _slot1 = mo.ui.altair_chart(chart_contour)
        _slot2 = mo.ui.plotly(fig_3d)

    else:  # D: x-slice vs 等高线
        _slot1 = mo.ui.altair_chart(chart_1d.properties(width=420, height=380))
        _slot2 = mo.ui.altair_chart(chart_contour.properties(width=420, height=380))

    mo.hstack([_slot1, _slot2], gap=0.5, widths="equal", align="start")
    return


@app.cell
def panel_view(panel):
    panel
    return


@app.cell
def narration(mo, shot):
    _scripts = {
        "A": """
    **🎬 A · ∂f/∂x 几何定义**（45 秒）

    > "偏导是什么？把多元函数沿一个方向切一刀 → 得到 1D 曲线。
    >  这条曲线在当前点的切线斜率 = 那个方向的偏导。
    >  蓝色曲线 = f(x, y_cur)（固定 y，只看 x 变化）。
    >  橙色虚线 = 切线，斜率 = ∂f/∂x。
    >  拖 y 滑块 → 切片形状变（换了一刀的位置）。
    >  拖 x 滑块 → 红点沿着当前切片滑，切线斜率实时变。"

    🎯 拖到 x=0, y=0 → 鞍点处 ∂f/∂x=0（切线水平）
    """,
        "B": """
    **🎬 B · 对偶切片 · 两个方向对比**（60 秒）

    > "左图沿 x 切，右图沿 y 切。
    >  鞍点的关键：沿 x 切 → 开口向上（极小方向）；
    >  沿 y 切 → 开口向下（极大方向）。
    >  两个切线斜率 = 两个偏导 = 梯度的两个分量。
    >  同一点两个方向曲率相反 → 鞍点（非极值）。"

    🎯 对比：碗函数两个方向都凸（都开口向上）→ 真极小
    """,
        "C": """
    **🎬 C · 全景 · 等高线+3D**（60 秒）

    > "左：俯视等高线 → 黄箭头 = 梯度向量 ∇f。
    >  梯度方向 = 等高线的法线 = 最陡上升方向。
    >  右：3D 曲面 → 红球/绿钻的相对位置。
    >  鞍点：3D 看像马鞍，等高线看像十字交叉。
    >  碗：3D 看像碗底，等高线看像同心圆。"

    🎯 拖到绿钻位置 → 黄箭头缩到 0（梯度=0）
    """,
        "D": """
    **🎬 D · 切片↔全景对应**（45 秒）

    > "左：1D 切片（沿 x）→ 蓝色曲线上的红点。
    >  右：等高线全景 → 红十字在同一位置。
    >  拖 x → 左图红点滑，右图红十字水平移动。
    >  拖 y → 左图整条曲线变形，右图红十字垂直移动。
    >  切片 ↔ 全景 一一对应 → 偏导的几何意义完整了。"

    🔑 **关键**：偏导=切片斜率；梯度=所有偏导组成的向量；梯度=0⇒极值候选
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
            "📐 录屏布局参考（grid 设计意图）": mo.md(
                r"""
    **目标 viewport**：1280×720（16:9 单屏不滚动）。
    左 sidebar 4col 控件 + 中右 stage 双槽 + 底 panel；narration / shot dropdown / truth_hint 全部录屏外。

    ### 横屏骨架

    ```
       0           4                              32
    y=0   ┌────── 标题 (h=3) ─────────────────────────┐
    y=3   ├ controls ┬──── stage 双槽 (h=26) ────────┤
      │ 4col     │   slot1 + slot2 widths=equal  │
      │ f choice │   A: x-slice | 几何定义文字    │
      │ x slider │   B: x-slice | y-slice 对偶   │
      │ y slider │   C: 等高线  | 3D 曲面        │
      │          │   D: x-slice | 等高线         │
    y=29  ├──────────┴──── panel (h=3) ──────────────┤  ← 720 内 = y<36
    y=32  ├──── 录屏外提示区 ───────────────────────┤
    y=36  ├──── truth_hint (h=3) ───────────────────┤
    y=39  ├──── shot dropdown (h=3) ────────────────┤
    y=42  ├──── narration 口播稿 (h=12) ────────────┤
    ```

    ### 镜头脚本（Strategy B 双槽）

    | # | 时长 | slot1 | slot2 | 教学焦点 |
    |---|---|---|---|---|
    | **A** | 0-45s | x-slice + 切线 | 几何定义文字 | ∂f/∂x = 切片的切线斜率 |
    | **B** | 45-105s | x-slice | y-slice | 两个方向对比（鞍点核心） |
    | **C** | 105-165s | 等高线+梯度 | 3D 曲面 | 全景：梯度方向 + 立体直觉 |
    | **D** | 165-210s | x-slice | 等高线 | 切片↔全景一一对应 |
                """
            )
        },
        multiple=False,
    )
    return


if __name__ == "__main__":
    app.run()
