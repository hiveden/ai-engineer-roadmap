"""
L1 / L2 正则化 · 收缩与稀疏 · Strategy B 双槽（slot1=L1 / slot2=L2）

教学核心：10 次多项式过拟合数据上看正则化怎么把系数拉回来。
- L1 (Lasso)：菱形约束 → 砍特征（产生零系数）
- L2 (Ridge)：圆形约束 → 压特征（等比收缩）

镜头脚本（Strategy B · 始终左 L1 右 L2）：
- A · 拟合曲线：slot1=L1 散点+拟合, slot2=L2 散点+拟合（看输出差异）
- B · 权重对比（主舞台）：slot1=L1 权重柱(零柱黄标), slot2=L2 权重柱(压缩)
- C · λ-MSE：slot1=L1 U 形, slot2=L2 U 形（看最优 λ 位置差异）

数据：与 07a-overfit 共享（seed=666, N=100, split random_state=5），10 次多项式 + StandardScaler
跑：cd 01-ML/02-LR/07b-regularization/demos && marimo run --port 2761 l1-l2-shrinkage.py
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/l1-l2-shrinkage.grid.json",
    css_file="custom.css",
)


@app.cell
def _():
    import warnings
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt

    from sklearn.linear_model import LinearRegression, Lasso, Ridge
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
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


@app.cell(hide_code=True)
def title(mo):
    mo.md(
        r"### L1 / L2 正则化 · 砍特征 vs 压特征 · 拖 $\lambda$ 看系数收缩"
    ).style(margin="0", padding="4px 12px", font_size="15px", line_height="1.3")
    return


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
def _(mo):
    # ===== 控件定义（极简 label）=====
    _S = dict(show_value=True, full_width=True)
    log_lambda = mo.ui.slider(
        -4.0, 2.0, step=0.1, value=-1.0, label="log₁₀λ", **_S
    )
    preset = mo.ui.dropdown(
        options={
            "✋ 手动": None,
            "≈0 (-4)": -4.0,
            "弱 (-2)": -2.0,
            "中 (-1)": -1.0,
            "强 (+1)": 1.0,
        },
        value="✋ 手动",
        label="preset",
    )
    shot = mo.ui.dropdown(
        options=["A · 拟合曲线", "B · 权重对比", "C · λ-MSE"],
        value="B · 权重对比",
        label="🎬 镜头",
    )
    return log_lambda, preset, shot


@app.cell
def controls(log_lambda, mo, preset):
    # sidebar 控件组合（4col = 160px · 极致紧凑）
    _h = lambda s: mo.md(s).style(
        margin="0", padding="0", font_size="11px",
        font_weight="700", color="#6b7280", letter_spacing="0.05em",
    )
    _div = mo.md("").style(
        border_top="1px solid #e5e7eb", margin="4px 0", padding="0", height="1px",
    )
    mo.vstack(
        [_h("正则强度"), preset, log_lambda, _div],
        gap=0,
        align="stretch",
    )
    return


@app.cell
def _(log_lambda, preset):
    # 派生量：preset 优先，否则用滑块
    if preset.value is not None:
        cur_log_lambda = float(preset.value)
    else:
        cur_log_lambda = float(log_lambda.value)
    cur_lambda = float(10.0 ** cur_log_lambda)
    return cur_lambda, cur_log_lambda


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
def _(
    X_test,
    X_train,
    mean_squared_error,
    np,
    pipe_l1,
    pipe_l2,
    pipe_none,
    y_test,
    y_train,
):
    # 提取权重 + 计算 metrics
    w_none = np.asarray(pipe_none.named_steps["reg"].coef_).ravel()
    w_l1 = np.asarray(pipe_l1.named_steps["reg"].coef_).ravel()
    w_l2 = np.asarray(pipe_l2.named_steps["reg"].coef_).ravel()
    ZERO_TOL = 1e-6
    zero_count_l1 = int(np.sum(np.abs(w_l1) < ZERO_TOL))
    zero_count_l2 = int(np.sum(np.abs(w_l2) < ZERO_TOL))

    def _mse_pair(pipe):
        tr = float(mean_squared_error(y_train, pipe.predict(X_train)))
        te = float(mean_squared_error(y_test, pipe.predict(X_test)))
        return tr, te

    tr_n, te_n = _mse_pair(pipe_none)
    tr_l1, te_l1 = _mse_pair(pipe_l1)
    tr_l2, te_l2 = _mse_pair(pipe_l2)
    return (
        ZERO_TOL,
        te_l1,
        te_l2,
        te_n,
        tr_l1,
        tr_l2,
        tr_n,
        w_l1,
        w_l2,
        w_none,
        zero_count_l1,
        zero_count_l2,
    )


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
                    "test_mse": float(mean_squared_error(y_test, _pipe.predict(X_test))),
                })
        return _rows

    df_curve = pd.DataFrame(_build_curve())
    return (df_curve,)


@app.cell
def _(alt, cur_lambda, np, pd, pipe_l1, pipe_l2, pipe_none, x_all, y_all):
    # ========== 拟合曲线图 · L1 / L2 各一张 ==========
    df_scatter = pd.DataFrame({"x": x_all, "y": y_all})
    xs = np.linspace(-3, 3, 200).reshape(-1, 1)
    yp_none = pipe_none.predict(xs).ravel()
    yp_l1 = pipe_l1.predict(xs).ravel()
    yp_l2 = pipe_l2.predict(xs).ravel()
    y_true = 0.5 * xs.ravel() ** 2 + xs.ravel() + 2

    _x_dom = [-3.2, 3.2]
    _y_dom = [-6, 12]

    def _make_fit_chart(yp_reg, model_name, color):
        df_none_line = pd.DataFrame({"x": xs.ravel(), "y": yp_none})
        df_reg_line = pd.DataFrame({"x": xs.ravel(), "y": yp_reg})
        df_true_line = pd.DataFrame({"x": xs.ravel(), "y": y_true})

        pts = (
            alt.Chart(df_scatter)
            .mark_circle(size=40, color="#64748b", opacity=0.45)
            .encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=_x_dom), title="x"),
                y=alt.Y("y:Q", scale=alt.Scale(domain=_y_dom), title="y"),
            )
        )
        truth = (
            alt.Chart(df_true_line)
            .mark_line(color="#6b7280", strokeDash=[4, 3], strokeWidth=1.5)
            .encode(x="x:Q", y="y:Q")
        )
        none_line = (
            alt.Chart(df_none_line)
            .mark_line(color="#ef4444", strokeWidth=1.8, opacity=0.4, strokeDash=[3, 2])
            .encode(x="x:Q", y="y:Q")
        )
        reg_line = (
            alt.Chart(df_reg_line)
            .mark_line(color=color, strokeWidth=3)
            .encode(x="x:Q", y="y:Q")
        )
        return (truth + pts + none_line + reg_line).properties(
            width=460, height=340,
            title=alt.TitleParams(
                text=f"{model_name} · λ={cur_lambda:.4g}",
                subtitle="灰虚=真实曲线 · 红淡=无正则 · 粗线=正则化后",
                fontSize=12, subtitleFontSize=10, subtitleColor="#6b7280",
            ),
        )

    chart_fit_l1 = _make_fit_chart(yp_l1, "L1 (Lasso)", "#f97316")
    chart_fit_l2 = _make_fit_chart(yp_l2, "L2 (Ridge)", "#10b981")
    return chart_fit_l1, chart_fit_l2


@app.cell
def _(ZERO_TOL, alt, np, pd, w_l1, w_l2, w_none):
    # ========== 权重柱图 · L1 / L2 各一张 ==========
    degrees = np.arange(1, 11)
    _w_max = max(float(np.max(np.abs(w_none))), 0.1)
    y_dom = [-_w_max * 1.15, _w_max * 1.15]

    def _make_weight_chart(w_reg, model_name, color, is_l1=False):
        # 上行：无正则（参考基线）
        df_none = pd.DataFrame({
            "degree": degrees, "weight": w_none, "model": "无正则",
        })
        row_none = (
            alt.Chart(df_none)
            .mark_bar(color="#ef4444", opacity=0.5)
            .encode(
                x=alt.X("degree:O", title=None,
                         axis=alt.Axis(labelExpr="'w' + datum.value", labelFontSize=9)),
                y=alt.Y("weight:Q", scale=alt.Scale(domain=y_dom), title=None,
                         axis=alt.Axis(labelFontSize=9)),
            )
        ).properties(width=460, height=100,
                     title=alt.TitleParams(text="无正则(基线)", fontSize=10,
                                           color="#dc2626", anchor="start"))

        # 下行：正则化后
        df_reg = pd.DataFrame({
            "degree": degrees, "weight": w_reg,
            "is_zero": [abs(float(w)) < ZERO_TOL for w in w_reg],
        })

        bars = (
            alt.Chart(df_reg)
            .mark_bar(color=color)
            .encode(
                x=alt.X("degree:O", title=None,
                         axis=alt.Axis(labelExpr="'w' + datum.value", labelFontSize=9)),
                y=alt.Y("weight:Q", scale=alt.Scale(domain=y_dom), title=None,
                         axis=alt.Axis(labelFontSize=9)),
            )
        )
        zero_rule = (
            alt.Chart(pd.DataFrame({"y": [0]}))
            .mark_rule(color="#1f2937", strokeWidth=0.8)
            .encode(y="y:Q")
        )

        layers = [zero_rule, bars]

        # L1 零柱标记：黄色实心 + 黑边
        if is_l1:
            df_zero = df_reg[df_reg["is_zero"]].copy()
            if len(df_zero) > 0:
                zero_marks = (
                    alt.Chart(df_zero)
                    .mark_bar(fill="#fde047", fillOpacity=0.9,
                              stroke="#000000", strokeWidth=1.8)
                    .encode(x="degree:O", y=alt.value(55), y2=alt.value(45))
                )
                zero_text = (
                    alt.Chart(df_zero)
                    .mark_text(text="0", fontSize=10, fontWeight="bold", color="#000")
                    .encode(x="degree:O", y=alt.value(50))
                )
                layers += [zero_marks, zero_text]

        row_reg = alt.layer(*layers).properties(
            width=460, height=160,
            title=alt.TitleParams(text=model_name, fontSize=10,
                                  color=color, anchor="start"),
        )

        return alt.vconcat(row_none, row_reg, spacing=4).properties(
            title=alt.TitleParams(
                text=f"{model_name} · 权重柱图",
                subtitle="标准化空间系数 · 上=无正则基线 下=正则化后",
                fontSize=12, subtitleFontSize=9, subtitleColor="#6b7280",
            ),
        )

    chart_w_l1 = _make_weight_chart(w_l1, "L1 (Lasso)", "#f97316", is_l1=True)
    chart_w_l2 = _make_weight_chart(w_l2, "L2 (Ridge)", "#10b981", is_l1=False)
    return chart_w_l1, chart_w_l2


@app.cell
def _(alt, cur_lambda, cur_log_lambda, df_curve, np, pd):
    # ========== λ-MSE U 形 · L1 / L2 各一张 ==========
    def _make_mse_chart(reg_type, color):
        df_sel = df_curve[df_curve["type"] == reg_type].copy()
        df_long = pd.melt(
            df_sel, id_vars=["lambda", "log_lambda", "type"],
            value_vars=["train_mse", "test_mse"],
            var_name="split", value_name="mse",
        )
        df_long["split"] = df_long["split"].map(
            {"train_mse": "train", "test_mse": "test"}
        )

        split_color = alt.Scale(
            domain=["train", "test"], range=["#3b82f6", "#f97316"]
        )

        lines = (
            alt.Chart(df_long)
            .mark_line(strokeWidth=2.5, point=True)
            .encode(
                x=alt.X("lambda:Q", scale=alt.Scale(type="log", domain=[1e-4, 1e2]),
                         title="λ (log)"),
                y=alt.Y("mse:Q", title="MSE", scale=alt.Scale(zero=False)),
                color=alt.Color("split:N", scale=split_color,
                                legend=alt.Legend(title="集合")),
                tooltip=[alt.Tooltip("lambda:Q", format=".4g"),
                         alt.Tooltip("mse:Q", format=".4f"), "split"],
            )
        )

        # 当前 λ 垂直线
        rule_v = (
            alt.Chart(pd.DataFrame({"lambda": [cur_lambda]}))
            .mark_rule(color="#9ca3af", strokeDash=[3, 3], strokeWidth=1.5)
            .encode(x="lambda:Q")
        )

        # 当前 λ 红点（内插 train/test）
        _idx = int(np.argmin(np.abs(df_sel["lambda"].values - cur_lambda)))
        _row = df_sel.iloc[_idx]
        df_cur = pd.DataFrame({
            "lambda": [cur_lambda, cur_lambda],
            "split": ["train", "test"],
            "mse": [float(_row["train_mse"]), float(_row["test_mse"])],
        })
        cur_dots = (
            alt.Chart(df_cur)
            .mark_circle(size=200, stroke="white", strokeWidth=2)
            .encode(
                x="lambda:Q", y="mse:Q",
                color=alt.Color("split:N", scale=split_color, legend=None),
            )
        )

        return (lines + rule_v + cur_dots).properties(
            width=460, height=340,
            title=alt.TitleParams(
                text=f"{reg_type} · λ-MSE · log₁₀λ={cur_log_lambda:.1f}",
                subtitle="蓝=train 橙=test · 竖线=当前λ · U形谷底=最优",
                fontSize=12, subtitleFontSize=10, subtitleColor="#6b7280",
            ),
        )

    chart_mse_l1 = _make_mse_chart("L1", "#f97316")
    chart_mse_l2 = _make_mse_chart("L2", "#10b981")
    return chart_mse_l1, chart_mse_l2


@app.cell
def shot_picker(shot):
    # 镜头切换器（提示区，录屏 crop 掉）
    shot
    return


@app.cell
def truth_hint(cur_lambda, cur_log_lambda, mo, zero_count_l1, zero_count_l2):
    # 真实参数提示（录屏外）
    mo.md(
        f"""<div style="background:#dbeafe;color:#1e40af;border-left:4px solid #3b82f6;
        padding:6px 14px;border-radius:6px;font-size:13px;line-height:1.4;margin:0;">
        🎯 当前 λ={cur_lambda:.4g} (log₁₀={cur_log_lambda:.2f})
        &nbsp;·&nbsp; L1 零系数={zero_count_l1}/10
        &nbsp;·&nbsp; L2 零系数={zero_count_l2}/10
        &nbsp;·&nbsp; 真实函数 y=0.5x²+x+2
        </div>"""
    )
    return


@app.cell
def stage(
    chart_fit_l1,
    chart_fit_l2,
    chart_mse_l1,
    chart_mse_l2,
    chart_w_l1,
    chart_w_l2,
    mo,
    shot,
):
    # 🎬 中央舞台 · Strategy B 双槽（slot1=L1 / slot2=L2）
    # 直接传 altair chart 对象（不用 mo.ui.altair_chart 避免 duplicate signal 冲突）
    if shot.value.startswith("A"):
        _slot1, _slot2 = chart_fit_l1, chart_fit_l2
    elif shot.value.startswith("B"):
        _slot1, _slot2 = chart_w_l1, chart_w_l2
    else:  # C
        _slot1, _slot2 = chart_mse_l1, chart_mse_l2

    mo.hstack([_slot1, _slot2], gap=0.5, widths="equal", align="start")
    return


@app.cell
def panel(
    cur_lambda,
    cur_log_lambda,
    mo,
    te_l1,
    te_l2,
    te_n,
    tr_l1,
    tr_l2,
    tr_n,
    zero_count_l1,
):
    # 数字面板（录屏内 · 紧凑单行）
    def _gc(gap):
        if gap < 0.15:
            return "#10b981"
        return "#f59e0b" if gap < 0.4 else "#ef4444"

    _gap_n = te_n - tr_n
    _gap_l1 = te_l1 - tr_l1
    _gap_l2 = te_l2 - tr_l2

    mo.md(f"""
    <div style="font-family:ui-monospace,monospace; font-size:12px; line-height:1.4;
            background:#f9fafb; border:1px solid #e5e7eb; border-radius:6px;
            padding:5px 12px; margin:0;">
    <b>λ={cur_lambda:.3g}</b> (log₁₀={cur_log_lambda:.1f}) &nbsp;|&nbsp;
    <span style="color:#dc2626">NoReg</span> test={te_n:.2f}
    <span style="color:{_gc(_gap_n)}">gap={_gap_n:.2f}</span> &nbsp;|&nbsp;
    <span style="color:#f97316">L1</span> test={te_l1:.2f}
    <span style="color:{_gc(_gap_l1)}">gap={_gap_l1:.2f}</span>
    zero={zero_count_l1}/10 &nbsp;|&nbsp;
    <span style="color:#10b981">L2</span> test={te_l2:.2f}
    <span style="color:{_gc(_gap_l2)}">gap={_gap_l2:.2f}</span>
    </div>
    """)
    return


@app.cell
def narration(mo, shot):
    # 口播稿：按 shot 切换（录屏外）
    _scripts = {
        "A": """
    **🎬 A · 拟合曲线**（40 秒）

    > "左 L1 右 L2，同一份过拟合数据。
    >  拖 λ 从 -4 → +2：
    >  - 红淡虚线 = 无正则（始终疯狂振荡）
    >  - 粗线 = 加正则后 → λ 越大越平滑
    >  - L1 先把高频部分砍平（突然变直），L2 是渐进变平。"

    🎯 对照：相同 λ 下两边曲线的平滑程度差异
    """,
        "B": """
    **🎬 B · 权重对比（主舞台）**（60 秒）

    > "左 L1 右 L2，上行灰色 = 无正则基线（系数爆炸）。
    >  拖 λ 增大：
    >  - **L1 行出现黄底黑边「0」** = 被彻底砍掉的特征
    >  - **L2 行所有柱等比缩矮** 但没有零柱
    >  核心区别：L1 砍 (sparsity) vs L2 压 (shrinkage)。"

    🎯 λ=0.1 时 L1 约 4-6 个零柱；L2 零柱 = 0
    """,
        "C": """
    **🎬 C · λ-MSE U 形**（40 秒）

    > "左 L1 右 L2，都是 train(蓝) / test(橙) MSE vs λ。
    >  注意两边**谷底位置不同**：
    >  - L1 最优 λ ≈ 0.01-0.1
    >  - L2 最优 λ 可能偏更大
    >  竖线 = 当前 λ，拖滑块看红点爬 U 形。"

    🎯 test MSE 谷底 = 最佳偏差-方差权衡点
    """,
    }
    _key = shot.value[0] if shot.value else "B"
    mo.md(_scripts.get(_key, "")).style(
        font_size="15px", line_height="1.6", margin="0", padding="14px 24px",
        background="#fffbeb", border_radius="8px",
        border_left="4px solid #fbbf24",
    )
    return


@app.cell
def layout_doc(mo):
    # ===== 📐 录屏 grid 布局参考（开发用）=====
    mo.accordion(
        {
            "📐 录屏布局参考（grid 设计意图）": mo.md(
                r"""
    **目标 viewport**：1280 宽 · 32 列 · rowHeight 20

    ### Grid 骨架（32 列 · 行 = 20px）

    ```
       0    4                              32
    0  ┌──────── title (32w × 3h) ─────────┐
    3  ├─ ctrl ─┬── stage (28w × 26h) ────┤  ┐
       │ (4w)   │  slot1(L1) | slot2(L2)  │  │
       │ preset │                          │  ├ 录屏 1280×720
       │ log₁₀λ│                          │  │
    29 ├────────┴── panel (32w × 3h) ─────┤  │
    32 ├───────────────────────────────────┤  ┘ ← y=32 (640px)
       │        (gap 36-32=4 rows)         │
    36 ├─ shot_picker (5w × 2h) ──────────┤  ← y=36 (720px)
    38 ├─ truth_hint (10w × 3h) ──────────┤
    41 ├─ narration (32w × 10h) ──────────┤
    51 └───────────────────────────────────┘
    ```

    ### 镜头脚本（Strategy B · 左 L1 右 L2 永远对照）

    | # | 时长 | slot1 (L1) | slot2 (L2) | 教学焦点 |
    |---|---|---|---|---|
    | **A** | 0-40s | L1 拟合曲线 | L2 拟合曲线 | λ 对输出平滑的影响差异 |
    | **B** | 40-100s | L1 权重柱(黄零柱) | L2 权重柱(等比压) | 砍 vs 压 核心视觉 |
    | **C** | 100-140s | L1 U 形 | L2 U 形 | 最优 λ 位置不同 |
                """
            )
        },
        multiple=False,
    )
    return


if __name__ == "__main__":
    app.run()
