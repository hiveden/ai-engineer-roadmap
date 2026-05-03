"""
多项式次数与过拟合 · 双视图联动 demo

左图：散点（训练集）+ 当前 degree 拟合曲线（红）+ 真实曲线（淡绿虚线）
右图：训练 MSE / 测试 MSE 随 degree 变化的双折线 + 当前 degree 红点 + 最优 degree 绿钻石

跑：marimo run --port 2733 --headless --no-token poly-degree-overfit.py
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/poly-degree-overfit.grid.json",
)


@app.cell
def _():
    import warnings

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    # 中文字体（Altair 不需要，matplotlib 备用）
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Heiti SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    # 关闭 polyfit 在高 degree 下的 RankWarning（教学 demo 不需要警告噪声）
    warnings.filterwarnings("ignore", category=np.exceptions.RankWarning) if hasattr(
        np, "exceptions"
    ) else warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")
    return alt, mean_squared_error, mo, np, pd, train_test_split


@app.cell
def _(mo):
    mo.md(
        r"""
        # 多项式次数与过拟合

        用 2 次多项式 $y = 0.5x^2 + x + 2 + N(0, 1)$ 生成 100 个点，70/30 拆分训练/测试。
        通过调整**拟合多项式的次数**，观察训练 / 测试 MSE 如何变化。

        - **左图**：散点（训练集 70 样本）+ 当前 degree 拟合曲线（红）+ 真实曲线（淡绿虚线）
        - **右图**：训练 MSE（蓝）vs 测试 MSE（橙）—— **U 形是过拟合的标志**
        - 红点 = 当前 degree 位置；绿钻石 = test MSE 最低点（甜蜜点）
        - **gap = test_mse − train_mse**，gap 越大 → 过拟合越严重
        """
    )
    return


@app.cell
def _(np, train_test_split):
    # ====== 数据：固定 seed=666 + random_state=5（与 README 底稿一致） ======
    np.random.seed(666)
    X_all = np.random.uniform(-3, 3, 100)
    y_all = 0.5 * X_all**2 + X_all + 2 + np.random.normal(0, 1, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.3, random_state=5
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(mo):
    # ====== 主控件：单一来源 = degree_slider ======
    # 范围 1-12（review 必修 3：避免 polyfit 数值崩溃；底稿用 X^1~X^10 已足够）
    degree_slider = mo.ui.slider(
        1, 12, step=1, value=2, label="多项式次数 degree", show_value=True
    )
    return (degree_slider,)


@app.cell
def _(degree_slider, mo):
    # ====== 预设按钮组（review 必修 2：preset = 按钮组调 set_value，唯一来源仍是 slider）======
    btn_under = mo.ui.button(
        label="1 · 欠拟合",
        on_click=lambda _: degree_slider.set_value(1),
    )
    btn_just = mo.ui.button(
        label="2 · 刚好",
        on_click=lambda _: degree_slider.set_value(2),
    )
    btn_mild = mo.ui.button(
        label="5 · 微过拟合",
        on_click=lambda _: degree_slider.set_value(5),
    )
    btn_severe = mo.ui.button(
        label="12 · 严重过拟合",
        on_click=lambda _: degree_slider.set_value(12),
    )
    return btn_just, btn_mild, btn_severe, btn_under


@app.cell
def _(btn_just, btn_mild, btn_severe, btn_under, degree_slider, mo):
    mo.vstack(
        [
            mo.md("**快速档位**（点击按钮 → 滑块同步更新）"),
            mo.hstack(
                [btn_under, btn_just, btn_mild, btn_severe],
                justify="start",
                gap=1,
            ),
            degree_slider,
        ],
        gap=0.5,
    )
    return


@app.cell
def _(X_test, X_train, degree_slider, mean_squared_error, np, y_test, y_train):
    # ====== 当前 degree 的拟合 + MSE（单一计算来源，下游所有视图从这里 fan-out）======
    degree = int(degree_slider.value)
    coeffs = np.polyfit(X_train, y_train, degree)

    y_pred_train = np.polyval(coeffs, X_train)
    y_pred_test = np.polyval(coeffs, X_test)

    train_mse = float(mean_squared_error(y_train, y_pred_train))
    test_mse = float(mean_squared_error(y_test, y_pred_test))
    gap = test_mse - train_mse

    # 左图用的密集采样
    x_dense = np.linspace(-3, 3, 200)
    y_fitted = np.polyval(coeffs, x_dense)
    # clip 防止 high degree 下数值飞出图框（视觉上仍能看到「抖动」）
    y_fitted = np.clip(y_fitted, -20, 25)
    return degree, gap, test_mse, train_mse, x_dense, y_fitted


@app.cell
def _(X_test, X_train, mean_squared_error, np, y_test, y_train):
    # ====== 预计算所有 degree=1..12 的 train/test MSE（U 形折线数据源）======
    all_degrees = np.arange(1, 13)
    train_mses = []
    test_mses = []
    for _d in all_degrees:
        _coeffs = np.polyfit(X_train, y_train, _d)
        _yp_tr = np.polyval(_coeffs, X_train)
        _yp_te = np.polyval(_coeffs, X_test)
        train_mses.append(float(mean_squared_error(y_train, _yp_tr)))
        test_mses.append(float(mean_squared_error(y_test, _yp_te)))

    train_mses = np.array(train_mses)
    test_mses = np.array(test_mses)
    # 最优 degree = test MSE 最低点
    best_degree = int(all_degrees[int(np.argmin(test_mses))])
    best_test_mse = float(test_mses[int(np.argmin(test_mses))])
    return all_degrees, best_degree, best_test_mse, test_mses, train_mses


@app.cell
def _(degree, gap, mo, test_mse, train_mse):
    # ====== 状态徽章 + 数字反馈 ======
    if gap < 0.15:
        _color, _label = "#10b981", "拟合良好"
    elif gap < 0.4:
        _color, _label = "#fbbf24", "微过拟合"
    else:
        _color, _label = "#ef4444", "严重过拟合"

    mo.md(
        f"""
<div style="display:flex; gap:24px; align-items:center; font-family:ui-monospace,monospace; font-size:15px; padding:8px 12px; background:#f9fafb; border-radius:6px;">
  <div>当前 <b>degree = {degree}</b></div>
  <div>训练 MSE = <b>{train_mse:.3f}</b></div>
  <div>测试 MSE = <b>{test_mse:.3f}</b></div>
  <div>gap = <b style="color:{_color}; font-size:18px">{gap:.3f}</b></div>
  <div><span style="background:{_color}; color:white; padding:2px 10px; border-radius:4px; font-size:13px;">{_label}</span></div>
</div>
"""
    )
    return


@app.cell
def _(
    X_train,
    alt,
    degree,
    np,
    pd,
    x_dense,
    y_fitted,
    y_train,
):
    # ====== 左图：散点 + 当前拟合曲线（红）+ 真实曲线（淡绿虚线）======
    df_scatter = pd.DataFrame({"x": X_train, "y": y_train})
    df_fit = pd.DataFrame({"x": x_dense, "y": y_fitted})

    # 真实曲线 y = 0.5x² + x + 2（参考基线）
    _x_true = np.linspace(-3, 3, 200)
    _y_true = 0.5 * _x_true**2 + _x_true + 2
    df_true = pd.DataFrame({"x": _x_true, "y": _y_true})

    scatter = (
        alt.Chart(df_scatter)
        .mark_circle(size=70, color="#1f77b4", opacity=0.7, stroke="white", strokeWidth=1)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-3.2, 3.2]), title="x"),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[-6, 12]), title="y"),
            tooltip=["x", "y"],
        )
    )
    line_true = (
        alt.Chart(df_true)
        .mark_line(color="#10b981", strokeWidth=2, strokeDash=[6, 4], opacity=0.7)
        .encode(x="x:Q", y="y:Q")
    )
    line_fit = (
        alt.Chart(df_fit)
        .mark_line(color="#ef4444", strokeWidth=2.5)
        .encode(x="x:Q", y="y:Q")
    )

    chart_left = (line_true + scatter + line_fit).properties(
        width=420,
        height=340,
        title=f"散点 + 拟合曲线 · degree = {degree}（红=拟合 / 淡绿虚线=真实）",
    )
    return (chart_left,)


@app.cell
def _(
    all_degrees,
    alt,
    best_degree,
    best_test_mse,
    degree,
    pd,
    test_mse,
    test_mses,
    train_mse,
    train_mses,
):
    # ====== 右图：训练/测试 MSE 双折线 + 当前 degree 红点 + 最优 degree 绿钻石 ======
    df_mse = pd.DataFrame(
        {
            "degree": list(all_degrees) * 2,
            "MSE": list(train_mses) + list(test_mses),
            "type": ["训练 MSE"] * len(all_degrees) + ["测试 MSE"] * len(all_degrees),
        }
    )

    color_scale = alt.Scale(
        domain=["训练 MSE", "测试 MSE"], range=["#3b82f6", "#f97316"]
    )

    lines = (
        alt.Chart(df_mse)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X(
                "degree:Q",
                scale=alt.Scale(domain=[0.5, 12.5]),
                axis=alt.Axis(values=list(range(1, 13)), title="多项式次数 degree"),
            ),
            y=alt.Y("MSE:Q", title="MSE"),
            color=alt.Color("type:N", scale=color_scale, title=None),
        )
    )
    points = (
        alt.Chart(df_mse)
        .mark_point(size=60, filled=True)
        .encode(
            x="degree:Q",
            y="MSE:Q",
            color=alt.Color("type:N", scale=color_scale, title=None),
        )
    )

    # 当前 degree 的两个红点（训练 + 测试）
    df_cur = pd.DataFrame(
        {
            "degree": [degree, degree],
            "MSE": [train_mse, test_mse],
            "type": ["当前", "当前"],
        }
    )
    cur_marker = (
        alt.Chart(df_cur)
        .mark_circle(size=280, color="#ef4444", stroke="white", strokeWidth=2.5)
        .encode(x="degree:Q", y="MSE:Q", tooltip=["degree", "MSE"])
    )

    # review 必修 1：测试 MSE 最低点绿钻石
    df_best = pd.DataFrame({"degree": [best_degree], "MSE": [best_test_mse]})
    best_marker = (
        alt.Chart(df_best)
        .mark_point(
            shape="diamond",
            size=300,
            filled=True,
            color="#10b981",
            stroke="white",
            strokeWidth=2,
        )
        .encode(x="degree:Q", y="MSE:Q", tooltip=["degree", "MSE"])
    )
    # 最优点上方的文字 annotation
    best_text = (
        alt.Chart(pd.DataFrame({"degree": [best_degree], "MSE": [best_test_mse], "label": [f"最优 degree={best_degree}"]}))
        .mark_text(
            color="#10b981",
            fontSize=11,
            fontWeight="bold",
            dy=-18,
            align="center",
        )
        .encode(x="degree:Q", y="MSE:Q", text="label:N")
    )

    chart_right = (lines + points + best_marker + cur_marker + best_text).properties(
        width=420,
        height=340,
        title="训练/测试 MSE vs degree · U 形 = 过拟合的标志",
    )
    return (chart_right,)


@app.cell
def _(chart_left, mo):
    # 左图：散点 + 当前 degree 拟合曲线（独占 cell · grid 友好）
    mo.ui.altair_chart(chart_left)
    return


@app.cell
def _(chart_right, mo):
    # 右图：训练/测试 MSE vs degree 双折线 + U 形
    mo.ui.altair_chart(chart_right)
    return


@app.cell
def _(mo):
    mo.md(
        """
        ---
        ### 玩法

        1. 拖滑块从 **degree=1 → 12**，看左图拟合曲线如何从「直线」变成「疯狂抖动」
        2. 看右图红点的位置：
           - 落在两线接近处 → 拟合刚好（红点贴近绿钻石）
           - 滑到右侧（橙线远高蓝线）→ 过拟合开始
        3. 点击 4 个预设按钮快速对比典型档位
        4. 注意 **degree ≥ 8** 时拟合曲线在数据点之间剧烈振荡 —— 这就是 high variance 的视觉证据

        **教学锚点**：红点和绿钻石的相对位置就是「调参方向」—— 红点在绿钻石左边 = 复杂度不够；右边 = 复杂度过头。
        """
    )
    return


@app.cell
def _(mo):
    # ===== Grid 布局参考（开发用 · 录屏隐藏 position=null）=====
    mo.accordion(
        {
            "Grid 布局参考（24 列 × rowHeight 20 · maxWidth 1280）": mo.md(
                r"""
**目标 viewport**：16:9 横屏，单屏不滚动。三段式：标题 / 控件 / 双图并排 / 状态徽章。

```
   0                          12                          24    列
0  ┌──────────────── 标题 mo.md（h=3）──────────────────────┐
3  │                                                        │
   ├──────── 控件区 vstack：4 预设按钮 + degree 滑块（h=4）──┤
7  │                                                        │
   │                                                        │
   │                                                        │
   │       chart_left          │       chart_right          │
   │   散点 + 拟合曲线         │   train/test MSE U 形     │
   │   （红=拟合 绿=真实）     │   红点=当前 绿钻=甜蜜点    │
   │       (12 × 19)           │       (12 × 19)            │
   │                                                        │
   │                                                        │
26 ├──────── 状态徽章：degree / train MSE / test MSE / gap ─┤
30 └────────────────────────────────────────────────────────┘
```

### Cell 索引 → grid 映射

| idx | cell 内容 | position |
|---|---|---|
| 0 | imports | null |
| 1 | 标题 md | [0, 0, 24, 3] |
| 2 | 数据生成 | null |
| 3 | degree_slider 定义 | null |
| 4 | 4 个预设按钮定义 | null |
| 5 | 控件 vstack（按钮+滑块） | [0, 3, 24, 4] |
| 6 | 当前 degree 拟合 + MSE 计算 | null |
| 7 | 全 degree=1..12 MSE 预计算 | null |
| 8 | 状态徽章 md | [0, 26, 24, 4] |
| 9 | chart_left 定义 | null |
| 10 | chart_right 定义 | null |
| 11 | 左图渲染 | [0, 7, 12, 19] |
| 12 | 右图渲染 | [12, 7, 12, 19] |
| 13 | 玩法 md | null |
| 14 | 本 accordion | null |

### 录屏要点

1. 拖滑块从 1 → 12，红线在左图变抖；右图红点沿 U 形右移
2. gap 三色：绿（<0.15 拟合良好）/ 黄（<0.4 微过拟合）/ 红（≥0.4 严重过拟合）
3. 点 4 预设按钮 → 滑块同步 → 双图同步刷新
"""
            )
        },
        multiple=False,
    )
    return


if __name__ == "__main__":
    app.run()
