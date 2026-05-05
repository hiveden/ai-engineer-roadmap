"""
多项式 LR · 特征数 = 弯曲能力 · 油耗 vs 车速 U 形数据

主隐喻：「特征数 = 弯曲能力」—— 给 LR 不同特征 [v] / [v, v²] / [v, v², v³]，
       看拟合曲线从平直变弯。

数据：7 个真实油耗-车速点（U 形），与教材 03-线性回归分类.md 一致。

跑：cd 01-ML/02-LR/01-intro/demos && marimo run poly-feature-effect.py --port 2764
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/poly-feature-effect.grid.json",
    css_file="custom.css",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import altair as alt
    import pandas as pd
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline

    return (
        LinearRegression,
        PolynomialFeatures,
        alt,
        make_pipeline,
        mo,
        np,
        pd,
    )


@app.cell(hide_code=True)
def title(degree, mo):
    # Dynamic title：degree 切换时录屏区有可见反馈（_5 §2）
    _verdict = {0: "常数 ✗", 1: "直线 ✗", 2: "抛物线 ✓", 3: "三次曲线"}.get(
        int(degree.value), "?"
    )
    mo.md(
        f"### 多项式 LR · 油耗 vs 车速 U 形 · "
        f"degree = {int(degree.value)} ({_verdict})"
    ).style(margin="0", padding="4px 12px", font_size="15px", line_height="1.3")
    return


@app.cell
def _(np):
    # ===== 真实业务数据：7 个车速-油耗点（U 形）=====
    # 与教材 03-线性回归分类.md ASCII 图一致；80km/h 附近最省油
    v_data = np.array([30, 45, 60, 75, 90, 105, 120], dtype=float)  # 车速 km/h
    y_data = np.array([8.6, 6.5, 5.7, 5.6, 6.0, 7.1, 8.5], dtype=float)  # 油耗 L/100km
    return v_data, y_data


@app.cell
def _(mo):
    # ===== 控件：degree 滑块（0/1/2/3）=====
    # _5 §4：label ≤ 5 字符（禁中文说明括号）
    degree = mo.ui.slider(
        0, 3, step=1, value=2, label="degree",
        show_value=True, full_width=True,
    )
    return (degree,)


@app.cell
def controls(degree, mo):
    # 单控件 + 解读小字
    _hint = mo.md(
        "0 = 常数（水平线）&nbsp;|&nbsp; "
        "1 = 直线 LR &nbsp;|&nbsp; "
        "2 = 抛物线（贴合 U 形 ✓）&nbsp;|&nbsp; "
        "3 = 三次曲线"
    ).style(font_size="11px", color="#6b7280", margin="0 12px")
    mo.vstack([degree, _hint], gap=4, align="stretch")
    return


@app.cell
def _(LinearRegression, PolynomialFeatures, degree, make_pipeline, np, v_data, y_data):
    # ===== 拟合 + 公式字符串 =====
    _d = int(degree.value)

    if _d == 0:
        # 常数模型：y = mean(y)
        _y_mean = float(y_data.mean())
        coef = np.array([])
        intercept = _y_mean
        _formula = f"y = {_y_mean:.2f} （常数 = 平均油耗）"
    else:
        _model = make_pipeline(
            PolynomialFeatures(degree=_d, include_bias=False),
            LinearRegression(),
        )
        _model.fit(v_data.reshape(-1, 1), y_data)
        coef = _model.named_steps["linearregression"].coef_
        intercept = float(_model.named_steps["linearregression"].intercept_)

        # 反向格式化（高次项在前）
        _terms = []
        for i in range(_d, 0, -1):
            _c = coef[i - 1]
            if i == 1:
                _terms.append(f"{_c:+.4f}·v")
            else:
                _terms.append(f"{_c:+.6f}·v^{i}")
        _terms.append(f"{intercept:+.2f}")
        _formula = "y = " + " ".join(_terms)

    # 预测 + MSE
    if _d == 0:
        _y_pred_at_data = np.full_like(y_data, intercept)
        _v_smooth = np.linspace(20, 130, 200)
        _y_pred_smooth = np.full_like(_v_smooth, intercept)
    else:
        _y_pred_at_data = _model.predict(v_data.reshape(-1, 1))
        _v_smooth = np.linspace(20, 130, 200)
        _y_pred_smooth = _model.predict(_v_smooth.reshape(-1, 1))

    mse = float(np.mean((_y_pred_at_data - y_data) ** 2))
    formula_str = _formula
    v_smooth = _v_smooth
    y_pred_smooth = _y_pred_smooth
    return formula_str, mse, v_smooth, y_pred_smooth


@app.cell
def _(alt, degree, pd, v_data, v_smooth, y_data, y_pred_smooth):
    # ===== 主图：散点 + 拟合曲线 =====
    _df_pts = pd.DataFrame({"v": v_data, "y": y_data})
    _df_curve = pd.DataFrame({"v": v_smooth, "y": y_pred_smooth})

    _x_dom = [20, 130]
    _y_dom = [4.5, 10.0]

    _color_by_degree = {
        0: "#9ca3af",  # 灰（常数）
        1: "#f97316",  # 橙（直线，警告色：失败）
        2: "#10b981",  # 绿（抛物线，成功色 ✓）
        3: "#3b82f6",  # 蓝（三次）
    }
    _line_color = _color_by_degree[int(degree.value)]

    _pts = (
        alt.Chart(_df_pts)
        .mark_circle(size=200, color="#1f2937", stroke="white", strokeWidth=2)
        .encode(
            x=alt.X("v:Q", scale=alt.Scale(domain=_x_dom), title="车速 v (km/h)"),
            y=alt.Y("y:Q", scale=alt.Scale(domain=_y_dom), title="油耗 (L/100km)"),
            tooltip=["v", "y"],
        )
    )

    _curve = (
        alt.Chart(_df_curve)
        .mark_line(color=_line_color, strokeWidth=4)
        .encode(x="v:Q", y="y:Q")
    )

    _zero_line = (
        alt.Chart(pd.DataFrame({"v": [80], "y": [_y_dom[0]]}))
        .mark_rule(color="#10b981", opacity=0.25, strokeDash=[4, 4])
        .encode(x="v:Q")
    )

    # _5 §7 双图临界 1000px：width ≤ 500 留 80px buffer 防 X scroll
    chart = (_zero_line + _curve + _pts).properties(width=500, height=420)
    return (chart,)


@app.cell
def _(chart):
    chart
    return


@app.cell
def _(alt, np, pd, v_data):
    # ===== 升维视图：(v, v²) 演示 "v² 是 v 的影子" =====
    # 教学价值：把 v 升级为 [v, v²] 这一步在做什么——
    #   v² 不是任意值，它是 v 的"影子"（z=v² 这条二次曲线）
    #   特征工程师不创造信息，只是把 v 的非线性形式"显式化"
    _df_z_pts = pd.DataFrame({"v": v_data, "z": v_data ** 2})
    _v_smooth_z = np.linspace(20, 130, 100)
    _df_z_curve = pd.DataFrame(
        {"v": _v_smooth_z, "z": _v_smooth_z ** 2}
    )

    _x_dom = [20, 130]
    _z_dom = [0, 18000]

    _curve_z = (
        alt.Chart(_df_z_curve)
        .mark_line(color="#8b5cf6", strokeWidth=3, strokeDash=[6, 3])
        .encode(
            x=alt.X("v:Q", scale=alt.Scale(domain=_x_dom), title="v 车速 (km/h)"),
            y=alt.Y("z:Q", scale=alt.Scale(domain=_z_dom), title="z = v² (升维特征)"),
        )
    )
    _pts_z = (
        alt.Chart(_df_z_pts)
        .mark_circle(size=200, color="#1f2937", stroke="white", strokeWidth=2)
        .encode(x="v:Q", y="z:Q", tooltip=["v", "z"])
    )
    _labels_z = (
        alt.Chart(_df_z_pts)
        .mark_text(dy=-12, fontSize=11, color="#6b7280")
        .encode(x="v:Q", y="z:Q", text="z:Q")
    )

    chart_z = (_curve_z + _pts_z + _labels_z).properties(width=500, height=420)
    return (chart_z,)


@app.cell
def _(chart_z):
    chart_z
    return


@app.cell
def _(degree, formula_str, mo, mse):
    # ===== 信息卡：当前 degree / 公式 / MSE =====
    _verdict_map = {
        0: ("✗ 失败", "#9ca3af", "常数线穿不过任何弯曲数据"),
        1: ("✗ 失败", "#f97316", "直线 LR 平均掉 U 形 → 哪儿都不准"),
        2: ("✓ 成功", "#10b981", "抛物线贴合 U 形 → MSE 大幅降低"),
        3: ("≈ 类似 2", "#3b82f6", "再加 v³ 不显著改善（数据本质是二次）"),
    }
    _v, _color, _comment = _verdict_map[int(degree.value)]

    _md = f"""
<div style="font-family:ui-monospace,monospace; font-size:14px; line-height:1.5;
        background:#f9fafb; border:1px solid #e5e7eb; border-radius:6px;
        padding:8px 14px; margin:0;">
<span style="background:{_color}22; color:{_color}; padding:3px 10px;
       border-radius:6px; font-weight:600; font-size:13px;">
  degree = {int(degree.value)} · {_v}
</span>
&nbsp;<span style="color:#374151">{_comment}</span>
<br>
<b>当前模型</b>：<code style="color:{_color}">{formula_str}</code>
&nbsp;|&nbsp; <b>MSE</b> = <span style="color:{_color}; font-weight:600">{mse:.4f}</span>
</div>
"""
    info_card = mo.md(_md)
    return (info_card,)


@app.cell
def _(info_card):
    info_card
    return


@app.cell
def hint(mo):
    # 录屏外提示卡（grid position=null 录屏隐藏）
    mo.md(
        """
**双视图主隐喻**：
- **左图 (v, y)**：原始空间——拖滑块看曲线弯曲能力随 degree 变化
  - degree 0→1 没本质改善（直线穿不过 U 形）
  - degree 1→2 是**质变**（MSE 骤降 40 倍，绿抛物线贴合 ✓）
  - degree 2→3 边际改善（数据本质是二次，再加 v³ 拟合差不多）
- **右图 (v, z=v²)**：升维空间——演示"v² 是 v 的影子"
  - 紫色虚线 z = v² 上 7 个数据点必然落在这条曲线上
  - 这就是 `PolynomialFeatures(degree=2)` 自动算出来的"新特征"
  - LR 不创造信息，是特征工程师把 v 的非线性形式（v²）**显式化**喂给 LR

**配套教材**：[`../03-线性回归分类.md`](../03-线性回归分类.md) §多项式线性回归
"""
    )
    return


if __name__ == "__main__":
    app.run()
