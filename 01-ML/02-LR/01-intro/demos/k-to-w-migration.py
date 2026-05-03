"""
k → w 迁移 · 一元 LR ↔ 二元 LR · 直觉对照

把初中 y = kx + b 的"1 个旋钮"直觉，迁移到 ML 多元 y = w₁x₁ + w₂x₂ + b 的"向量旋钮"。

视图：
- 左：一元 1D 散点 + 拟合直线 + 残差方块（让 MSE 看得见）
- 右：二元 2D 散点 + 预测平面（mark_rect 网格背景色）
- toggle「退化模式 w₂=0」：右图退化为左图同构

跑：cd 01-ML/02-LR/01-intro/demos && marimo edit --port 2735 k-to-w-migration.py
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import altair as alt
    import pandas as pd

    return alt, mo, np, pd


@app.cell
def _(mo):
    mo.md(
        r"""
        # k → w 迁移 · 一元 LR ↔ 二元 LR

        **目标**：从初中 $y = kx + b$ 的「1 个旋钮 + 1 个偏置」，过渡到 ML 多元 $y = w_1 x_1 + w_2 x_2 + b$ 的「向量旋钮 + 1 个偏置」。

        - 左图 · 一元：身高（相对 160 的差值）→ 体重，1 个特征 → 1 根斜率 k
        - 右图 · 二元：身高 + 工龄 → 体重，2 个特征 → 2 根斜率 w₁, w₂（拟合的是平面）
        - 勾选 **「锁定 w₂=0」** → 右图退化为只有 w₁·x₁ + b，**和左图同构**
        """
    )
    return


@app.cell
def _(np):
    # ===== 真实参数 + 数据生成 =====
    # 用相对身高 x₁_raw - 160（范围 0-25），让 b 范围直观（不必负到 -80）
    W_TRUE = np.array([0.5, 0.3])  # w₁=0.5（每多 1cm 身高，体重 +0.5kg）；w₂=0.3（每多 1 年工龄，体重 +0.3kg）
    B_TRUE = 50.0  # 基线体重（身高 160 + 工龄 0 时 ≈ 50kg）

    _rng = np.random.default_rng(42)

    # 一元数据：5 个点，只用身高
    _x1 = np.array([0.0, 6.0, 12.0, 14.0, 20.0])  # 相对身高（身高 - 160）
    _y1 = W_TRUE[0] * _x1 + B_TRUE + _rng.normal(0, 1.5, 5)
    x_1d = _x1
    y_1d = _y1

    # 二元数据：15 个点，身高 + 工龄
    _x_h = _rng.uniform(0, 25, 15)  # 相对身高 0-25
    _x_w = _rng.uniform(0, 20, 15)  # 工龄 0-20
    X_2d = np.column_stack([_x_h, _x_w])
    y_2d = X_2d @ W_TRUE + B_TRUE + _rng.normal(0, 2, 15)

    return B_TRUE, W_TRUE, X_2d, x_1d, y_1d, y_2d


@app.cell
def _(mo):
    # ===== 滑块配置 =====
    k_1d = mo.ui.slider(0.0, 1.5, step=0.05, value=0.5, label="k （一元斜率）", show_value=True)
    b_1d = mo.ui.slider(30.0, 70.0, step=0.5, value=50.0, label="b （一元截距）", show_value=True)

    w1_2d = mo.ui.slider(0.0, 1.5, step=0.05, value=0.5, label="w₁ （身高权重）", show_value=True)
    w2_2d = mo.ui.slider(-1.0, 1.5, step=0.05, value=0.3, label="w₂ （工龄权重）", show_value=True)
    b_2d = mo.ui.slider(30.0, 70.0, step=0.5, value=50.0, label="b （偏置）", show_value=True)

    lock_w2 = mo.ui.switch(value=False, label="锁定 w₂ = 0（退化为一元）")
    return b_1d, b_2d, k_1d, lock_w2, w1_2d, w2_2d


@app.cell
def _(b_1d, b_2d, k_1d, lock_w2, mo, w1_2d, w2_2d):
    mo.vstack(
        [
            mo.md("## 控件"),
            mo.hstack(
                [
                    mo.vstack([mo.md("**左 · 一元**"), k_1d, b_1d]),
                    mo.vstack([mo.md("**右 · 二元**"), w1_2d, w2_2d, b_2d, lock_w2]),
                ],
                gap=2,
                justify="start",
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(lock_w2, w2_2d):
    # 退化模式：w₂=0 强制为 0（视觉等同于一元）
    w2_eff = 0.0 if lock_w2.value else float(w2_2d.value)
    return (w2_eff,)


@app.cell
def _(alt, b_1d, k_1d, np, pd, x_1d, y_1d):
    # ============ 左图 · 一元 LR：散点 + 直线 + 残差方块 ============
    _k = float(k_1d.value)
    _b = float(b_1d.value)
    _y_pred = _k * x_1d + _b
    _err_sq = (y_1d - _y_pred) ** 2
    mse_1d = float(np.mean(_err_sq))

    _df = pd.DataFrame(
        {
            "x": x_1d,
            "y": y_1d,
            "y_pred": _y_pred,
            "err_sq": _err_sq,
        }
    )

    _xs_line = pd.DataFrame({"x": np.linspace(-2, 27, 30)})
    _xs_line["y"] = _k * _xs_line["x"] + _b

    _x_dom = [-2, 27]
    _y_dom = [40, 75]

    _pts = (
        alt.Chart(_df)
        .mark_circle(size=160, color="#1f77b4", stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=_x_dom), title="x = 身高 - 160 (cm)"),
            y=alt.Y("y:Q", scale=alt.Scale(domain=_y_dom), title="y = 体重 (kg)"),
            tooltip=["x", "y", "y_pred", "err_sq"],
        )
    )
    _squares = (
        alt.Chart(_df)
        .mark_square(opacity=0.30, color="#ef4444")
        .encode(
            x="x:Q",
            y="y:Q",
            size=alt.Size("err_sq:Q", scale=alt.Scale(range=[20, 2500]), legend=None),
        )
    )
    _rules = (
        alt.Chart(_df)
        .mark_rule(color="#ef4444", opacity=0.55, strokeDash=[3, 2])
        .encode(x="x:Q", y="y:Q", y2="y_pred:Q")
    )
    _line = (
        alt.Chart(_xs_line)
        .mark_line(color="#ef4444", strokeWidth=3)
        .encode(x="x:Q", y="y:Q")
    )

    chart_1d = (_rules + _squares + _pts + _line).properties(
        width=380,
        height=340,
        title="一元 LR · y = k·x + b · 1 个特征 → 1 根斜率",
    )
    return chart_1d, mse_1d


@app.cell
def _(X_2d, alt, b_2d, np, pd, w1_2d, w2_eff, y_2d):
    # ============ 右图 · 二元 LR：散点 + 平面预测背景（mark_rect 网格） ============
    _w1 = float(w1_2d.value)
    _w2 = w2_eff
    _b = float(b_2d.value)

    _y_pred_2d = X_2d @ np.array([_w1, _w2]) + _b
    mse_2d = float(np.mean((_y_pred_2d - y_2d) ** 2))

    # 50×50 背景网格 → mark_rect 显示预测 ŷ
    _gx = np.linspace(0, 25, 50)
    _gy = np.linspace(0, 20, 50)
    _GX, _GY = np.meshgrid(_gx, _gy)
    _GZ = _w1 * _GX + _w2 * _GY + _b

    _grid_df = pd.DataFrame(
        {
            "x1": _GX.ravel(),
            "x2": _GY.ravel(),
            "y_hat": _GZ.ravel(),
        }
    )

    _pts_df = pd.DataFrame(
        {
            "x1": X_2d[:, 0],
            "x2": X_2d[:, 1],
            "y": y_2d,
            "y_pred": _y_pred_2d,
            "err": y_2d - _y_pred_2d,
        }
    )

    # 与散点共享同一色阶（viridis），让"背景色 vs 散点色"差距 = 残差直觉
    _y_min = float(min(y_2d.min(), _GZ.min()))
    _y_max = float(max(y_2d.max(), _GZ.max()))
    _color_scale = alt.Scale(scheme="viridis", domain=[_y_min, _y_max])

    _bg = (
        alt.Chart(_grid_df)
        .mark_rect(opacity=0.65)
        .encode(
            x=alt.X("x1:Q", bin=alt.Bin(maxbins=50), title="x₁ = 身高 - 160 (cm)"),
            y=alt.Y("x2:Q", bin=alt.Bin(maxbins=50), title="x₂ = 工龄 (年)"),
            color=alt.Color("mean(y_hat):Q", scale=_color_scale, title="ŷ"),
        )
    )

    _pts2 = (
        alt.Chart(_pts_df)
        .mark_circle(size=180, stroke="white", strokeWidth=2)
        .encode(
            x=alt.X("x1:Q", scale=alt.Scale(domain=[0, 25])),
            y=alt.Y("x2:Q", scale=alt.Scale(domain=[0, 20])),
            color=alt.Color("y:Q", scale=_color_scale, legend=None),
            tooltip=["x1", "x2", "y", "y_pred", "err"],
        )
    )

    chart_2d = (_bg + _pts2).properties(
        width=380,
        height=340,
        title="二元 LR · y = w₁·x₁ + w₂·x₂ + b · 2 个特征 → 2 根斜率（平面）",
    )
    return chart_2d, mse_2d


@app.cell
def _(b_1d, b_2d, k_1d, lock_w2, mo, mse_1d, mse_2d, w1_2d, w2_eff):
    # ============ 退化徽章 + 参数对照面板 ============
    if lock_w2.value:
        _badge = (
            '<span style="background:#dcfce7;color:#065f46;padding:3px 10px;'
            'border-radius:6px;font-size:13px;font-weight:600;">'
            "✓ 退化为一元（w₂ = 0）"
            "</span>"
        )
    else:
        _badge = (
            '<span style="background:#dbeafe;color:#1e40af;padding:3px 10px;'
            'border-radius:6px;font-size:13px;font-weight:600;">'
            "完整二元模式"
            "</span>"
        )

    _panel_md = f"""
<div style="font-family:ui-monospace,monospace; font-size:14px; line-height:1.7;
            background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px;
            padding:14px 18px; margin-top:8px;">

<div style="margin-bottom:8px;">{_badge}</div>

<b>左图 · 一元</b>：y = k·x + b<br>
&nbsp;&nbsp;当前 <b>k = {k_1d.value:.2f}</b>,&nbsp; <b>b = {b_1d.value:.2f}</b>
&nbsp;&nbsp;<span style="color:#6b7280">MSE = {mse_1d:.3f}</span><br>
&nbsp;&nbsp;→ <b>1 个权重 (k)</b> + <b>1 个偏置 (b)</b> = <b>2 个参数</b>

<div style="height:8px;"></div>

<b>右图 · 二元</b>：y = w₁·x₁ + w₂·x₂ + b<br>
&nbsp;&nbsp;当前 <b>w₁ = {w1_2d.value:.2f}</b>,&nbsp; <b>w₂ = {w2_eff:.2f}</b>{"&nbsp;<i style='color:#16a34a'>(已锁定)</i>" if lock_w2.value else ""},&nbsp;
<b>b = {b_2d.value:.2f}</b>
&nbsp;&nbsp;<span style="color:#6b7280">MSE = {mse_2d:.3f}</span><br>
&nbsp;&nbsp;→ <b>2 个权重 (w₁, w₂)</b> + <b>1 个偏置 (b)</b> = <b>3 个参数</b>

<div style="height:10px;border-top:1px dashed #d1d5db;margin:10px 0 8px 0;"></div>

<b>关键观察</b>：
<ul style="margin:4px 0 0 18px; padding:0;">
  <li><b>k 是单个数</b> → <b>w 是向量 [w₁, w₂]</b>（特征加 1 个，w 就多一维）</li>
  <li><b>b 永远只有 1 个</b>，无论几个特征——它和具体 x 无关，所有样本共享</li>
  <li><b>w₂ = 0 时右图退化为左图</b>（试试勾选「锁定 w₂=0」感受同构）</li>
</ul>

</div>
"""
    panel = mo.md(_panel_md)
    return (panel,)


@app.cell
def _(B_TRUE, W_TRUE, mo):
    # 真实参数提示（让用户知道往哪调）
    mo.callout(
        mo.md(
            f"""
**真实参数（数据生成时用的）**：w₁ = {W_TRUE[0]}, w₂ = {W_TRUE[1]}, b = {B_TRUE}

把滑块调到这组值附近，应该能看到：左图 k = {W_TRUE[0]} 时直线最贴；右图 w₁={W_TRUE[0]}, w₂={W_TRUE[1]} 时背景色和散点色最融合。
"""
        ),
        kind="info",
    )
    return


@app.cell
def _(chart_1d, chart_2d, mo, panel):
    mo.vstack(
        [
            mo.hstack(
                [mo.ui.altair_chart(chart_1d), mo.ui.altair_chart(chart_2d)],
                gap=1,
                widths=[1, 1],
            ),
            panel,
        ],
        gap=1,
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ### 玩法

        1. **左图独立玩**：拖 k、b 让红线穿过 5 个蓝点，红方块越小越好（方块大小 = 误差²）
        2. **右图独立玩**：拖 w₁、w₂、b 让背景色阶和散点颜色"融合不见"（说明 ŷ ≈ y）
        3. **同构验证（核心）**：勾选「锁定 w₂=0」→ 右图就只剩 x₁ 在影响 y，背景色变成「只沿 x₁ 方向渐变的竖条纹」——**这就是左图的样子**
        4. **真实参数对照**：把所有权重都调到 0.5 / 0.3 / 50，看左右两图的 MSE 都接近 0

        ### 一句话总结

        > **一元 → 多元的唯一变化**：k（标量）变成 w（向量），每多一个特征就多一根独立的"斜率"。b 始终是 1 个标量，含义不变。
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 立体视角：把二元的「色阶平面」撑起来看

        2D 俯视只能看「色阶」，看不到平面的"陡度"。3D 视图能让你拖拽旋转：

        - 平面的 **x₁ 方向坡度** = w₁（拖滑块看平面绕 x₁ 轴倾斜）
        - 平面的 **x₂ 方向坡度** = w₂（拖滑块看平面绕 x₂ 轴倾斜）
        - **整体上下平移** = b（抬高/压低整个平面）

        勾选「锁定 w₂=0」时，平面变成"沿 x₁ 方向倾斜、沿 x₂ 方向完全平"——这就是退化为一元的几何意义（一根斜率方向 + 一根方向永远水平）。

        红色虚线是每个真实点到平面的**垂直距离**（残差）。
        """
    )
    return


@app.cell
def _(X_2d, b_2d, mo, np, w1_2d, w2_eff, y_2d):
    # ============ 3D Surface · 二元拟合平面（plotly）============
    import plotly.graph_objects as go

    _w1 = float(w1_2d.value)
    _w2 = w2_eff
    _b = float(b_2d.value)

    # 30×30 网格 → 拟合平面
    _gx = np.linspace(0, 25, 30)
    _gy = np.linspace(0, 20, 30)
    _GX, _GY = np.meshgrid(_gx, _gy)
    _GZ = _w1 * _GX + _w2 * _GY + _b
    _GZ = np.clip(_GZ, 20, 90)  # 防极端值飞出

    # 共享色阶域：包含真实 y 和预测面
    _z_min = float(min(y_2d.min(), _GZ.min()))
    _z_max = float(max(y_2d.max(), _GZ.max()))

    fig_3d = go.Figure()

    # 拟合平面
    fig_3d.add_trace(
        go.Surface(
            x=_GX,
            y=_GY,
            z=_GZ,
            colorscale="Viridis",
            cmin=_z_min,
            cmax=_z_max,
            opacity=0.78,
            showscale=False,
            contours={"z": {"show": True, "usecolormap": True, "project_z": True}},
            name="拟合平面",
        )
    )

    # 真实数据散点
    fig_3d.add_trace(
        go.Scatter3d(
            x=X_2d[:, 0],
            y=X_2d[:, 1],
            z=y_2d,
            mode="markers",
            marker=dict(
                size=6,
                color=y_2d,
                colorscale="Viridis",
                cmin=_z_min,
                cmax=_z_max,
                line=dict(color="white", width=1.5),
            ),
            name="真实数据",
        )
    )

    # 残差线（每个点垂直连到平面）
    _y_pred_pts = X_2d @ np.array([_w1, _w2]) + _b
    for _i in range(len(X_2d)):
        fig_3d.add_trace(
            go.Scatter3d(
                x=[X_2d[_i, 0], X_2d[_i, 0]],
                y=[X_2d[_i, 1], X_2d[_i, 1]],
                z=[y_2d[_i], _y_pred_pts[_i]],
                mode="lines",
                line=dict(color="#ef4444", width=2, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig_3d.update_layout(
        title=f"3D 视角 · y = w₁·x₁ + w₂·x₂ + b（w₁={_w1:.2f}, w₂={_w2:.2f}, b={_b:.1f}）",
        scene=dict(
            xaxis_title="x₁ = 身高 − 160 (cm)",
            yaxis_title="x₂ = 工龄 (年)",
            zaxis_title="y = 体重 (kg)",
            camera={"eye": {"x": 1.6, "y": -1.6, "z": 1.0}},
        ),
        font=dict(family="PingFang SC, Heiti SC, Arial Unicode MS, DejaVu Sans"),
        height=520,
        margin=dict(l=0, r=0, b=0, t=50),
        showlegend=False,
    )
    mo.ui.plotly(fig_3d)
    return


if __name__ == "__main__":
    app.run()
