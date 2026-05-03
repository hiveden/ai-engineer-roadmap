"""
L1 / L2 正则化 · 收缩与稀疏 · 10 次多项式过拟合数据

A. 散点 + 三条拟合曲线（无正则 / L1 / L2）  → 业务对比
B. 权重柱图（三组并排，L1 零柱「黄填黑边」标记）→ 主舞台 · 「砍 vs 压」
C. λ-MSE U 形 · L1+L2 同框（toggle 高亮当前选择）→ 偏差-方差权衡

数据：与 07a-overfit 共享（seed=666, N=100, split random_state=5），10 次多项式 + StandardScaler
跑：marimo edit --port 2734 --headless --no-token l1-l2-shrinkage.py
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/l1-l2-shrinkage.grid.json",
)


@app.cell
def _():
    import warnings
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    import matplotlib.pyplot as plt

    from sklearn.linear_model import LinearRegression, Lasso, Ridge
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return (
        Lasso,
        LinearRegression,
        Pipeline,
        PolynomialFeatures,
        Ridge,
        StandardScaler,
        alt,
        mean_squared_error,
        mo,
        np,
        pd,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # L1 / L2 正则化 · 收缩与稀疏

        用与 07a 同款的「10 次多项式过拟合」数据，看正则化怎么把红色「疯狂振荡」拉回来。

        $$
        \mathcal{L}_{\text{L1}}(W) = \text{MSE}(W) + \lambda \sum_i |w_i| \quad\text{（Lasso · 砍特征）}
        $$

        $$
        \mathcal{L}_{\text{L2}}(W) = \text{MSE}(W) + \lambda \sum_i w_i^2 \quad\text{（Ridge · 压特征）}
        $$

        - **A 图**：三条拟合曲线对比（红=无正则，橙=L1，绿=L2）
        - **B 图（主舞台）**：权重柱图，L1「黄底黑边」零柱 = 被砍掉的特征
        - **C 图**：λ-MSE U 形（L1+L2 同框对比甜蜜点）
        """
    )
    return


@app.cell
def _(mo):
    PRESETS = {
        "「✋」手动 (滑块控制)": None,
        "「0」无正则 · λ ≈ 1e-4": -4.0,
        "「🌱」弱 · λ = 0.01": -2.0,    # review 建议：从 -3 调到 -2，跨度更明显
        "「✓」适中 · λ = 0.1": -1.0,
        "「💪」强 · λ = 10": 1.0,
    }

    preset = mo.ui.dropdown(
        options=PRESETS,
        value="「✓」适中 · λ = 0.1",
        label="预设场景",
    )
    reg_type = mo.ui.radio(
        options=["L1 (Lasso)", "L2 (Ridge)"],
        value="L1 (Lasso)",
        label="正则类型 (影响 C 图高亮 + A/B 视觉聚焦)",
        inline=True,
    )
    log_lambda = mo.ui.slider(
        start=-4.0, stop=2.0, step=0.05, value=-1.0,
        label="正则强度 log₁₀(λ)",
        show_value=True,
    )
    return PRESETS, log_lambda, preset, reg_type


@app.cell
def _(log_lambda, mo, preset, reg_type):
    mo.vstack(
        [
            mo.hstack([preset, reg_type], justify="start", gap=2),
            log_lambda,
        ],
        gap=1,
    )
    return


@app.cell
def _(log_lambda, preset):
    # 派生量：preset 优先，否则用滑块
    if preset.value is None:
        cur_log_lambda = float(log_lambda.value)
        is_preset = False
    else:
        cur_log_lambda = float(preset.value)
        is_preset = True
    cur_lambda = float(10.0 ** cur_log_lambda)
    return cur_lambda, cur_log_lambda, is_preset


@app.cell
def _(np, train_test_split):
    # 与 07a-overfit 完全一致的合成数据
    np.random.seed(666)
    N = 100
    x_all = np.random.uniform(-3, 3, size=N)
    y_all = 0.5 * x_all**2 + x_all + 2 + np.random.normal(0, 1, size=N)
    X_all = x_all.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.3, random_state=5
    )
    return X_test, X_train, x_all, y_all, y_test, y_train


@app.cell
def _(
    Lasso,
    LinearRegression,
    Pipeline,
    PolynomialFeatures,
    Ridge,
    StandardScaler,
):
    # pipeline 工厂：10 次多项式 + 标准化 + 模型
    def make_pipe(model):
        return Pipeline(
            [
                ("poly", PolynomialFeatures(degree=10, include_bias=False)),
                ("scaler", StandardScaler()),
                ("reg", model),
            ]
        )

    def fit_three(lam, X_train, y_train):
        pn = make_pipe(LinearRegression())
        pl1 = make_pipe(Lasso(alpha=lam, max_iter=50000, tol=1e-4))
        pl2 = make_pipe(Ridge(alpha=lam))
        pn.fit(X_train, y_train)
        pl1.fit(X_train, y_train)
        pl2.fit(X_train, y_train)
        return pn, pl1, pl2
    return (fit_three,)


@app.cell
def _(X_train, cur_lambda, fit_three, y_train):
    pipe_none, pipe_l1, pipe_l2 = fit_three(cur_lambda, X_train, y_train)
    return pipe_l1, pipe_l2, pipe_none


@app.cell
def _(np, pipe_l1, pipe_l2, pipe_none):
    w_none = np.asarray(pipe_none.named_steps["reg"].coef_).ravel()
    w_l1 = np.asarray(pipe_l1.named_steps["reg"].coef_).ravel()
    w_l2 = np.asarray(pipe_l2.named_steps["reg"].coef_).ravel()
    ZERO_TOL = 1e-6
    zero_count_l1 = int(np.sum(np.abs(w_l1) < ZERO_TOL))
    zero_count_l2 = int(np.sum(np.abs(w_l2) < ZERO_TOL))
    return ZERO_TOL, w_l1, w_l2, w_none, zero_count_l1


@app.cell
def _(
    X_test,
    X_train,
    mean_squared_error,
    pipe_l1,
    pipe_l2,
    pipe_none,
    y_test,
    y_train,
):
    def _mse_pair(pipe):
        tr = float(mean_squared_error(y_train, pipe.predict(X_train)))
        te = float(mean_squared_error(y_test, pipe.predict(X_test)))
        return tr, te

    tr_n, te_n = _mse_pair(pipe_none)
    tr_l1, te_l1 = _mse_pair(pipe_l1)
    tr_l2, te_l2 = _mse_pair(pipe_l2)
    metrics = {
        "none": {"train": tr_n, "test": te_n, "gap": te_n - tr_n},
        "L1":   {"train": tr_l1, "test": te_l1, "gap": te_l1 - tr_l1},
        "L2":   {"train": tr_l2, "test": te_l2, "gap": te_l2 - tr_l2},
    }
    return (metrics,)


@app.cell
def _(X_test, X_train, fit_three, mean_squared_error, np, pd, y_test, y_train):
    # 全局预计算：30 个 λ × 2 种正则 = 60 次 fit
    def _build_curve():
        _grid = np.logspace(-4, 2, 30)
        _rows = []
        for _lam in _grid:
            _pn, _pl1, _pl2 = fit_three(float(_lam), X_train, y_train)
            for _name, _pipe in [("L1", _pl1), ("L2", _pl2)]:
                _rows.append({
                    "lambda": float(_lam),
                    "log_lambda": float(np.log10(_lam)),
                    "type": _name,
                    "train_mse": float(mean_squared_error(y_train, _pipe.predict(X_train))),
                    "test_mse":  float(mean_squared_error(y_test, _pipe.predict(X_test))),
                })
        return _rows

    df_curve = pd.DataFrame(_build_curve())
    return (df_curve,)


@app.cell
def _(
    cur_lambda,
    cur_log_lambda,
    is_preset,
    metrics,
    mo,
    preset,
    reg_type,
    zero_count_l1,
):
    # 模式徽章 · 必修 2：preset 锁定时附加「滑块已锁定」提示
    if is_preset:
        _badge_label = preset.value
        _mode_html = (
            f'<span style="background:#fef3c7;color:#92400e;padding:3px 10px;'
            f'border-radius:4px;font-size:12px;font-weight:600;border:1px solid #f59e0b;">'
            f'🟡 预设模式 · {_badge_label} · λ={cur_lambda:.4g}</span>'
            f' <span style="color:#92400e;font-size:11px;margin-left:6px;">'
            f'（手动滑块已锁定，选 ✋ 手动 解锁）</span>'
        )
    else:
        _mode_html = (
            f'<span style="background:#dbeafe;color:#1e40af;padding:3px 10px;'
            f'border-radius:4px;font-size:12px;font-weight:600;border:1px solid #3b82f6;">'
            f'🔵 手动模式 · log₁₀(λ)={cur_log_lambda:.2f} · λ={cur_lambda:.4g}</span>'
        )

    # 当前选择高亮
    _sel = "L1" if reg_type.value.startswith("L1") else "L2"
    _sel_html = (
        f'<span style="background:#fef3c7;color:#92400e;padding:2px 8px;border-radius:4px;'
        f'font-size:11px;margin-left:8px;">当前聚焦：{_sel}</span>'
    )

    def _gap_color(gap):
        if gap < 0.15:
            return "#10b981"
        if gap < 0.40:
            return "#f59e0b"
        return "#ef4444"

    def _gap_dot(gap):
        return f'<span style="color:{_gap_color(gap)}; font-weight:700;">●</span>'

    m_n, m_l1, m_l2 = metrics["none"], metrics["L1"], metrics["L2"]

    mo.md(
        f"""
        <div style="font-family:ui-monospace,monospace; font-size:13px; line-height:1.7;">
          <div>{_mode_html}{_sel_html}</div>
          <div style="margin-top:6px;">
            <span style="color:#dc2626;font-weight:700;">无正则</span> ·
            train={m_n['train']:.3f}　test={m_n['test']:.3f}　gap={m_n['gap']:.3f} {_gap_dot(m_n['gap'])}
            <span style="color:#9ca3af;">（过拟合基线）</span>
          </div>
          <div>
            <span style="color:#f97316;font-weight:700;">L1 (Lasso)</span> ·
            train={m_l1['train']:.3f}　test={m_l1['test']:.3f}　gap={m_l1['gap']:.3f} {_gap_dot(m_l1['gap'])}
            　<b>zero_count = {zero_count_l1}/10</b>
          </div>
          <div>
            <span style="color:#10b981;font-weight:700;">L2 (Ridge)</span> ·
            train={m_l2['train']:.3f}　test={m_l2['test']:.3f}　gap={m_l2['gap']:.3f} {_gap_dot(m_l2['gap'])}
          </div>
        </div>
        """
    )
    return


@app.cell
def _(
    alt,
    cur_lambda,
    np,
    pd,
    pipe_l1,
    pipe_l2,
    pipe_none,
    reg_type,
    x_all,
    y_all,
):
    # ========== A · 散点 + 三条拟合曲线 ==========
    df_scatter = pd.DataFrame({"x": x_all, "y": y_all})

    xs = np.linspace(-3, 3, 200).reshape(-1, 1)
    yp_none = pipe_none.predict(xs).ravel()
    yp_l1 = pipe_l1.predict(xs).ravel()
    yp_l2 = pipe_l2.predict(xs).ravel()
    y_true_curve = 0.5 * xs.ravel() ** 2 + xs.ravel() + 2

    df_lines = pd.concat(
        [
            pd.DataFrame({"x": xs.ravel(), "y": yp_none, "model": "无正则"}),
            pd.DataFrame({"x": xs.ravel(), "y": yp_l1,   "model": "L1 (Lasso)"}),
            pd.DataFrame({"x": xs.ravel(), "y": yp_l2,   "model": "L2 (Ridge)"}),
        ]
    )
    df_truth = pd.DataFrame({"x": xs.ravel(), "y": y_true_curve})

    # 建议 3：toggle 联动 — 当前选中模型 100% 不透明，另一者降到 0.35
    _sel = "L1 (Lasso)" if reg_type.value.startswith("L1") else "L2 (Ridge)"
    df_lines["opacity"] = df_lines["model"].apply(
        lambda m: 1.0 if (m == "无正则" or m == _sel) else 0.35
    )
    df_lines["sw"] = df_lines["model"].apply(
        lambda m: 3.0 if m == _sel else (2.5 if m == "无正则" else 1.8)
    )

    color_scale = alt.Scale(
        domain=["无正则", "L1 (Lasso)", "L2 (Ridge)"],
        range=["#ef4444", "#f97316", "#10b981"],
    )

    pts = (
        alt.Chart(df_scatter)
        .mark_circle(size=55, color="#1f77b4", opacity=0.55, stroke="white", strokeWidth=1)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-3.2, 3.2]), title="x"),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[-6, 12]), title="y"),
        )
    )
    truth = (
        alt.Chart(df_truth)
        .mark_line(color="#6b7280", strokeDash=[4, 3], strokeWidth=1.5)
        .encode(x="x:Q", y="y:Q")
    )
    lines = (
        alt.Chart(df_lines)
        .mark_line()
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color("model:N", scale=color_scale, legend=alt.Legend(title="拟合曲线")),
            opacity=alt.Opacity("opacity:Q", legend=None, scale=alt.Scale(domain=[0, 1], range=[0, 1])),
            size=alt.Size("sw:Q", legend=None, scale=alt.Scale(domain=[1, 3], range=[1.8, 3.0])),
        )
    )

    chart_A = (truth + pts + lines).properties(
        width=420,
        height=340,
        title=alt.TitleParams(
            text=f"A · 三种拟合对比 · λ={cur_lambda:.4g}",
            subtitle="灰虚线=真实曲线 y=0.5x²+x+2 · 加粗=当前选中正则",
            fontSize=13,
            subtitleFontSize=10,
            subtitleColor="#6b7280",
        ),
    )
    return (chart_A,)


@app.cell
def _(ZERO_TOL, alt, np, pd, reg_type, w_l1, w_l2, w_none):
    # ========== B · 权重柱图（主舞台） ==========
    # 必修 1：零柱用「黄色实心填充 + 黑边」(避免与无正则红撞色)
    # 必修 3：副标题标注「标准化空间下系数」
    degrees = np.arange(1, 11)
    rows = []
    for name, w in [("无正则", w_none), ("L1 (Lasso)", w_l1), ("L2 (Ridge)", w_l2)]:
        for d, wv in zip(degrees, w):
            rows.append({
                "degree": int(d),
                "weight": float(wv),
                "model": name,
                "is_zero": (name == "L1 (Lasso)") and (abs(float(wv)) < ZERO_TOL),
                "abs_w": abs(float(wv)),
            })
    df_w = pd.DataFrame(rows)

    _color_scale_b = alt.Scale(
        domain=["无正则", "L1 (Lasso)", "L2 (Ridge)"],
        range=["#ef4444", "#f97316", "#10b981"],
    )

    # 联动：选 L1 时 L2 行变淡，反之亦然（无正则始终满）
    _sel = "L1 (Lasso)" if reg_type.value.startswith("L1") else "L2 (Ridge)"
    df_w["row_opacity"] = df_w["model"].apply(
        lambda m: 1.0 if (m == "无正则" or m == _sel) else 0.4
    )

    # 共享 y 轴范围（让三行可比）
    _w_max = max(float(np.max(np.abs(w_none))), 0.1)
    y_dom = [-_w_max * 1.15, _w_max * 1.15]

    def _row_chart(model_name, title_color):
        sub = df_w[df_w["model"] == model_name].copy()

        bars = (
            alt.Chart(sub)
            .mark_bar(stroke="white", strokeWidth=0.5)
            .encode(
                x=alt.X(
                    "degree:O",
                    title=None,
                    axis=alt.Axis(labelExpr="'w' + datum.value", labelFontSize=10),
                ),
                y=alt.Y(
                    "weight:Q",
                    scale=alt.Scale(domain=y_dom),
                    title=None,
                    axis=alt.Axis(labelFontSize=9),
                ),
                color=alt.Color("model:N", scale=_color_scale_b, legend=None),
                opacity=alt.Opacity("row_opacity:Q", legend=None, scale=alt.Scale(domain=[0, 1], range=[0, 1])),
            )
        )
        zero_rule = (
            alt.Chart(pd.DataFrame({"y": [0]}))
            .mark_rule(color="#1f2937", strokeWidth=1)
            .encode(y="y:Q")
        )

        # 零柱标记（仅 L1 行有效）—— 必修 1：黄色实心填充 + 黑边
        zero_marks = (
            alt.Chart(sub[sub["is_zero"]])
            .mark_bar(
                fill="#fde047",         # 鲜黄实心
                fillOpacity=0.9,
                stroke="#000000",       # 黑边
                strokeWidth=1.8,
            )
            .encode(
                x=alt.X("degree:O"),
                y=alt.Y("y_lo:Q"),
                y2="y_hi:Q",
            )
            .transform_calculate(
                y_lo=f"-{_w_max * 0.06}",
                y_hi=f"{_w_max * 0.06}",
            )
        )
        zero_text = (
            alt.Chart(sub[sub["is_zero"]])
            .mark_text(
                text="0",
                fontSize=11,
                fontWeight="bold",
                color="#000000",
                dy=0,
            )
            .encode(x="degree:O", y=alt.value(180))
        )

        # 数值小标签
        val_text = (
            alt.Chart(sub[~sub["is_zero"]])
            .mark_text(fontSize=8, color="#374151", baseline="bottom", dy=-2)
            .encode(
                x="degree:O",
                y="weight:Q",
                text=alt.Text("weight:Q", format=".2f"),
            )
        )

        return (zero_rule + bars + zero_marks + val_text + zero_text).properties(
            width=520, height=110,
            title=alt.TitleParams(text=model_name, color=title_color, fontSize=11, anchor="start"),
        )

    chart_B = alt.vconcat(
        _row_chart("无正则", "#dc2626"),
        _row_chart("L1 (Lasso)", "#ea580c"),
        _row_chart("L2 (Ridge)", "#059669"),
        spacing=4,
    ).properties(
        title=alt.TitleParams(
            text="B · 权重柱图 · L1「黄底黑边」= 被砍掉 (zero)",
            subtitle="⚠ 标准化空间下系数 (PolynomialFeatures+StandardScaler 后) · 仅供稀疏性对比，非原始 x^k 系数",
            fontSize=13,
            subtitleFontSize=10,
            subtitleColor="#9333ea",
        ),
    )
    return (chart_B,)


@app.cell
def _(
    X_test,
    X_train,
    alt,
    cur_lambda,
    cur_log_lambda,
    df_curve,
    fit_three,
    mean_squared_error,
    np,
    pd,
    reg_type,
    y_test,
    y_train,
):
    # ========== C · λ-MSE U 形（建议 1：L1 + L2 同框对比） ==========
    sel_type = "L1" if reg_type.value.startswith("L1") else "L2"

    # 长格式数据（每行：lambda, type, split=train/test, mse）
    df_long = pd.melt(
        df_curve,
        id_vars=["lambda", "log_lambda", "type"],
        value_vars=["train_mse", "test_mse"],
        var_name="split",
        value_name="mse",
    )
    df_long["split"] = df_long["split"].map({"train_mse": "train", "test_mse": "test"})
    df_long["selected"] = df_long["type"] == sel_type

    # 当前 λ 红点（只对选中类型计算）
    _pn, _pl1, _pl2 = fit_three(cur_lambda, X_train, y_train)
    _pipe_sel = _pl1 if sel_type == "L1" else _pl2
    _cur_train = float(mean_squared_error(y_train, _pipe_sel.predict(X_train)))
    _cur_test = float(mean_squared_error(y_test, _pipe_sel.predict(X_test)))
    df_cur = pd.DataFrame({
        "lambda": [cur_lambda, cur_lambda],
        "split": ["train", "test"],
        "mse": [_cur_train, _cur_test],
    })

    split_color = alt.Scale(
        domain=["train", "test"],
        range=["#3b82f6", "#f97316"],
    )

    # 选中类型：实线粗，未选中：虚线细
    sel_lines = (
        alt.Chart(df_long[df_long["selected"]])
        .mark_line(strokeWidth=2.8, point=True)
        .encode(
            x=alt.X("lambda:Q", scale=alt.Scale(type="log", domain=[1e-4, 1e2]), title="λ (log scale)"),
            y=alt.Y("mse:Q", title="MSE", scale=alt.Scale(zero=False)),
            color=alt.Color("split:N", scale=split_color, legend=alt.Legend(title="集合 (选中)")),
            tooltip=["type", "split", alt.Tooltip("lambda:Q", format=".4g"), alt.Tooltip("mse:Q", format=".4f")],
        )
    )
    other_lines = (
        alt.Chart(df_long[~df_long["selected"]])
        .mark_line(strokeWidth=1.5, strokeDash=[4, 3], opacity=0.45, point=False)
        .encode(
            x=alt.X("lambda:Q", scale=alt.Scale(type="log", domain=[1e-4, 1e2])),
            y="mse:Q",
            color=alt.Color("split:N", scale=split_color, legend=None),
        )
    )

    # 当前 λ 垂直参考线
    rule_v = (
        alt.Chart(pd.DataFrame({"lambda": [cur_lambda]}))
        .mark_rule(color="#9ca3af", strokeDash=[3, 3])
        .encode(x="lambda:Q")
    )
    cur_dots = (
        alt.Chart(df_cur)
        .mark_circle(size=260, stroke="white", strokeWidth=2.5)
        .encode(
            x="lambda:Q",
            y="mse:Q",
            color=alt.Color("split:N", scale=split_color, legend=None),
        )
    )

    chart_C = (other_lines + sel_lines + rule_v + cur_dots).properties(
        width=420,
        height=340,
        title=alt.TitleParams(
            text=f"C · λ-MSE U 形 · 当前 {sel_type} · λ={cur_lambda:.4g} (log₁₀={cur_log_lambda:.2f})",
            subtitle="实线=选中正则，虚线=另一种 (对比甜蜜点位置) · 红点=当前 λ",
            fontSize=13,
            subtitleFontSize=10,
            subtitleColor="#6b7280",
        ),
    )
    return (chart_C,)


@app.cell
def _(chart_A, mo):
    # A 视图：散点 + 三条拟合曲线（独占 cell · grid 友好）
    mo.ui.altair_chart(chart_A)
    return


@app.cell
def _(chart_B, mo):
    # B 视图：权重柱图（主舞台 · L1 黄柱稀疏）
    mo.ui.altair_chart(chart_B)
    return


@app.cell
def _(chart_C, mo):
    # C 视图：训练/测试 MSE vs λ
    mo.ui.altair_chart(chart_C)
    return


@app.cell
def _(mo):
    mo.md(
        """
        ---
        ### 玩法

        1. **拖 log₁₀(λ) 滑块** 从 -4 → 2：A 图三条线从「红色疯狂抖 / 橙绿贴真值」 → 「全部压成水平线」
        2. **看 B 图（主舞台）**：
           - 「无正则」行高次项系数巨大（过拟合的根源）
           - **「L1」行黄底黑边「0」标记** = 被砍掉的特征 → λ 越大零柱越多
           - 「L2」行所有柱「整体矮化」但**无零柱** = 等比收缩
        3. **切 L1/L2 toggle**：A/B 视图对应行加粗高亮，C 图实线在选中类型上
        4. **C 图（U 形）**：实线 = 选中正则，虚线 = 另一种；
           注意 L1 和 L2 的**测试 MSE 谷底位置**通常不在同一 λ
        5. **4 档预设**：一键对比「无正则 / 弱 / 适中 / 强」
           - ⚠️ 选预设后再拖滑块**无效**（已锁定）→ 切回「✋ 手动」解锁

        ### 与 07a-overfit 衔接
        端口 2733（07a 过拟合演示）+ 2734（本 demo · 解药）共享同款数据，可两 tab 并排对照。
        """
    )
    return


@app.cell
def _(mo):
    # ===== 📐 录屏 grid 布局参考（开发用 · 录屏隐藏 · position=null）=====
    mo.accordion(
        {
            "📐 录屏布局参考（grid 设计意图）": mo.md(
                r"""
**目标 viewport**：1400 宽 · 24 列 · rowHeight 20 · 信息密度高
B 视图（权重柱图）= 主舞台占右侧大块；A/C 左侧上下叠放（散点 / U 形）

### Grid 骨架（24 列 · 行单位 = 20px）

```
   0                       9                                24
0  ┌──────────────── 标题（h=3）─────────────────────────────┐
3  ├──────────────── 控件 preset/reg/slider（h=4）──────────┤
7  ├──────────────── 模式徽章 + 三组 MSE 指标（h=5）────────┤
12 ├─ A 散点+三拟合 ──┬──────────────────────────────────────┤
   │  (9w × 17h)      │                                      │
   │                  │   B · 权重柱图（主舞台）             │
29 ├─ C λ-MSE U 形 ──┤   (15w × 34h · 三行 vconcat)         │
   │  (9w × 17h)      │   L1 黄底黑边零柱 = 砍特征证据       │
   │                  │                                      │
46 ├──────────────── 玩法说明（h=10）────────────────────────┤
56 └────────────────────────────────────────────────────────┘
```

### 镜头脚本

| # | 时长 | 焦点 | 操作 |
|---|---|---|---|
| **A** | 0-30s | 标题 + 控件 + 模式徽章 | 介绍三种正则 + λ 概念 |
| **B** | 30-90s | B 主舞台（权重柱图） | 拖滑块 -4→2，看 L1 黄柱出现/L2 整体压扁 |
| **C** | 90-150s | A 散点（左上） | 看红线疯狂抖 → λ 增大变平 |
| **D** | 150-200s | C U 形（左下） | 切 L1/L2 toggle，看实虚线甜蜜点错位 |
| **E** | 200-240s | 4 档预设巡游 | 「无 / 弱 / 适中 / 强」一键演示 |

### 关键提示

1. **B 占大头是主舞台**：右侧 15 列 × 34 行（680px 高）容纳三行权重对比
2. **A/C 左侧叠放**：相同宽度（9 列）让两种视图垂直对齐
3. **控件区一行 hstack**：preset + reg_type 同行，log_λ 滑块单独一行
4. **指标卡 h=5**：三种正则 + zero_count 信息密度需要 100px
                """
            )
        },
        multiple=False,
    )
    return


if __name__ == "__main__":
    app.run()
