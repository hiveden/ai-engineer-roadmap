"""
多项式次数与过拟合 · 双视图联动 demo

左图：散点（训练集）+ 当前 degree 拟合曲线（红）+ 真实曲线（淡绿虚线）
右图：训练 MSE / 测试 MSE 随 degree 变化的双折线 + 当前 degree 红点 + 最优 degree 绿钻石

录屏布局（_5-layout-guide Strategy B 双槽）：
- 左右同时显示 fit curve + U-shape loss
- degree slider + 4 预设按钮提供快速跳转

跑：cd 01-ML/02-LR/07a-overfit/demos && marimo run --port 2760 poly-degree-overfit.py
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/poly-degree-overfit.grid.json",
    css_file="custom.css",
)


@app.cell
def _():
    import warnings

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    warnings.filterwarnings("ignore", category=np.exceptions.RankWarning) if hasattr(
        np, "exceptions"
    ) else warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")
    return alt, mean_squared_error, mo, np, pd, train_test_split


@app.cell(hide_code=True)
def title(effective_degree, mo, shot):
    # 动态标题：跟随 shot + degree 联动
    _shot_label = shot.value.split(" · ")[1] if " · " in shot.value else shot.value
    mo.md(
        f"### 多项式过拟合 · 🎬 {shot.value.split(' · ')[0]} · {_shot_label} "
        f"（degree = {effective_degree}）"
    ).style(margin="0", padding="4px 12px", font_size="15px", line_height="1.3")
    return


@app.cell
def _(np, train_test_split):
    # 数据：y = 0.5x² + x + 2 + noise，100 点 70/30 拆分
    np.random.seed(666)
    X_all = np.random.uniform(-3, 3, 100)
    y_all = 0.5 * X_all**2 + X_all + 2 + np.random.normal(0, 1, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.3, random_state=5
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(mo):
    # degree_slider：仅"E · 自定义"模式下用户拖动调精细
    _S = dict(show_value=True, full_width=True)
    degree_slider = mo.ui.slider(
        1, 12, step=1, value=2, label="degree", **_S
    )
    return (degree_slider,)


@app.cell
def _(mo):
    # shot dropdown：4 关键档位 + 自定义模式
    shot = mo.ui.dropdown(
        options=[
            "A · 欠拟合",
            "B · 刚好",
            "C · 微过拟合",
            "D · 严重过拟合",
            "E · 自定义",
        ],
        value="A · 欠拟合",
        label="🎬 镜头",
    )
    return (shot,)


@app.cell
def _(degree_slider, shot):
    # effective_degree：A/B/C/D 锁定档位，E 用 slider 值
    _PRESET = {
        "A · 欠拟合": 1,
        "B · 刚好": 2,
        "C · 微过拟合": 5,
        "D · 严重过拟合": 12,
    }
    effective_degree = _PRESET.get(shot.value, int(degree_slider.value))
    return (effective_degree,)


@app.cell
def controls(degree_slider, mo, shot):
    # sidebar：仅 degree slider（自定义模式用），加提示当前是否锁定
    _h = lambda s: mo.md(s).style(
        margin="0", padding="0", font_size="11px",
        font_weight="700", color="#6b7280", letter_spacing="0.05em",
    )
    _is_custom = shot.value.startswith("E")
    _hint = mo.md(
        "<div style='font-size:11px;color:#6b7280;line-height:1.4;'>"
        + ("✋ 自定义：拖滑块调 degree" if _is_custom
           else "🔒 已锁定（切到 E 解锁滑块）")
        + "</div>"
    ).style(margin="0", padding="2px 0")

    mo.vstack(
        [_h("DEGREE"), degree_slider, _hint],
        gap=0.2,
        align="stretch",
    )
    return


@app.cell
def _(X_test, X_train, effective_degree, mean_squared_error, np, y_test, y_train):
    # 当前 degree 拟合 + MSE（effective_degree = shot 锁定值或 slider 自定义值）
    degree = effective_degree
    coeffs = np.polyfit(X_train, y_train, degree)

    y_pred_train = np.polyval(coeffs, X_train)
    y_pred_test = np.polyval(coeffs, X_test)

    train_mse = float(mean_squared_error(y_train, y_pred_train))
    test_mse = float(mean_squared_error(y_test, y_pred_test))
    gap = test_mse - train_mse

    # 左图密集采样
    x_dense = np.linspace(-3, 3, 200)
    y_fitted = np.polyval(coeffs, x_dense)
    y_fitted = np.clip(y_fitted, -20, 25)
    return degree, gap, test_mse, train_mse, x_dense, y_fitted


@app.cell
def _(X_test, X_train, mean_squared_error, np, y_test, y_train):
    # 全 degree=1..12 的 train/test MSE（U 形数据源）
    all_degrees = np.arange(1, 13)
    train_mses = []
    test_mses = []
    for _d in all_degrees:
        _coeffs = np.polyfit(X_train, y_train, _d)
        train_mses.append(float(mean_squared_error(y_train, np.polyval(_coeffs, X_train))))
        test_mses.append(float(mean_squared_error(y_test, np.polyval(_coeffs, X_test))))

    train_mses = np.array(train_mses)
    test_mses = np.array(test_mses)
    best_degree = int(all_degrees[int(np.argmin(test_mses))])
    best_test_mse = float(test_mses[int(np.argmin(test_mses))])
    return all_degrees, best_degree, best_test_mse, test_mses, train_mses


@app.cell
def _(best_degree, degree, gap, mo, test_mse, train_mse):
    # 数字面板
    if gap < 0.15:
        _color, _label = "#10b981", "拟合良好"
    elif gap < 0.4:
        _color, _label = "#fbbf24", "微过拟合"
    else:
        _color, _label = "#ef4444", "严重过拟合"

    panel = mo.md(
        f"""<div style="font-family:ui-monospace,monospace; font-size:13px; line-height:1.45;
        background:#f9fafb; border:1px solid #e5e7eb; border-radius:6px;
        padding:6px 12px; margin:0;">
<b>degree={degree}</b> &nbsp;|&nbsp;
train MSE=<b>{train_mse:.3f}</b> &nbsp;|&nbsp;
test MSE=<b>{test_mse:.3f}</b> &nbsp;|&nbsp;
gap=<b style="color:{_color}; font-size:16px">{gap:.3f}</b>
<span style="background:{_color}; color:white; padding:2px 8px; border-radius:4px; font-size:12px; margin-left:8px;">{_label}</span>
&nbsp;|&nbsp; 最优 degree={best_degree}（绿钻石）
</div>"""
    )
    return (panel,)


@app.cell
def _(X_train, alt, degree, np, pd, x_dense, y_fitted, y_train):
    # 左图：散点 + 拟合曲线（红）+ 真实曲线（淡绿虚线）
    _df_scatter = pd.DataFrame({"x": X_train, "y": y_train})
    _df_fit = pd.DataFrame({"x": x_dense, "y": y_fitted})
    _x_true = np.linspace(-3, 3, 200)
    _df_true = pd.DataFrame({"x": _x_true, "y": 0.5 * _x_true**2 + _x_true + 2})

    _scatter = (
        alt.Chart(_df_scatter)
        .mark_circle(size=60, color="#1f77b4", opacity=0.7, stroke="white", strokeWidth=1)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-3.2, 3.2]), title="x"),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[-6, 12]), title="y"),
            tooltip=["x", "y"],
        )
    )
    _line_true = (
        alt.Chart(_df_true)
        .mark_line(color="#10b981", strokeWidth=2, strokeDash=[6, 4], opacity=0.7)
        .encode(x="x:Q", y="y:Q")
    )
    _line_fit = (
        alt.Chart(_df_fit)
        .mark_line(color="#ef4444", strokeWidth=2.5)
        .encode(x="x:Q", y="y:Q")
    )

    chart_left = (_line_true + _scatter + _line_fit).properties(
        width=400,
        height=440,
        title=f"拟合曲线 · degree={degree}（红=拟合 / 绿虚=真实）",
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
    # 右图：train/test MSE U 形双折线
    _df_mse = pd.DataFrame(
        {
            "degree": list(all_degrees) * 2,
            "MSE": list(train_mses) + list(test_mses),
            "type": ["train"] * len(all_degrees) + ["test"] * len(all_degrees),
        }
    )
    _color_scale = alt.Scale(domain=["train", "test"], range=["#3b82f6", "#f97316"])

    _lines = (
        alt.Chart(_df_mse)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X(
                "degree:Q",
                scale=alt.Scale(domain=[0.5, 12.5]),
                axis=alt.Axis(values=list(range(1, 13)), title="degree"),
            ),
            y=alt.Y("MSE:Q", title="MSE"),
            color=alt.Color("type:N", scale=_color_scale, title=None),
        )
    )
    _points = (
        alt.Chart(_df_mse)
        .mark_point(size=50, filled=True)
        .encode(x="degree:Q", y="MSE:Q", color=alt.Color("type:N", scale=_color_scale, title=None))
    )

    # 当前 degree 红圆
    _df_cur = pd.DataFrame({"degree": [degree, degree], "MSE": [train_mse, test_mse]})
    _cur_marker = (
        alt.Chart(_df_cur)
        .mark_circle(size=260, color="#ef4444", stroke="white", strokeWidth=2.5)
        .encode(x="degree:Q", y="MSE:Q")
    )

    # 最优 degree 绿钻石
    _df_best = pd.DataFrame({"degree": [best_degree], "MSE": [best_test_mse]})
    _best_marker = (
        alt.Chart(_df_best)
        .mark_point(shape="diamond", size=280, filled=True, color="#10b981", stroke="white", strokeWidth=2)
        .encode(x="degree:Q", y="MSE:Q")
    )
    _best_text = (
        alt.Chart(pd.DataFrame({"degree": [best_degree], "MSE": [best_test_mse], "label": [f"best={best_degree}"]}))
        .mark_text(color="#10b981", fontSize=11, fontWeight="bold", dy=-16)
        .encode(x="degree:Q", y="MSE:Q", text="label:N")
    )

    chart_right = (_lines + _points + _best_marker + _cur_marker + _best_text).properties(
        width=400,
        height=440,
        title="train/test MSE vs degree · U 形 = 过拟合",
    )
    return (chart_right,)


@app.cell
def stage(chart_left, chart_right, mo):
    # 🎬 中央舞台 · Strategy B 双槽（始终显示 fit + loss）
    _slot1 = mo.ui.altair_chart(chart_left)
    _slot2 = mo.ui.altair_chart(chart_right)
    mo.hstack([_slot1, _slot2], gap=0.5, widths="equal", align="start")
    return


@app.cell
def panel_view(panel):
    panel
    return


@app.cell
def truth_hint(mo):
    # 真实参数提示（录屏外）
    mo.md(
        """<div style="background:#dbeafe;color:#1e40af;border-left:4px solid #3b82f6;
        padding:6px 14px;border-radius:6px;font-size:13px;line-height:1.4;margin:0;">
        🎯 <b>真实函数</b> y = 0.5x² + x + 2 + N(0,1)
        &nbsp;·&nbsp; 最优 degree=2（和数据生成一致）
        </div>"""
    )
    return


@app.cell
def shot_picker(shot):
    # 镜头切换器（录屏外）
    shot
    return


@app.cell
def narration(mo, shot):
    # 口播稿（录屏外）
    _scripts = {
        "A": """
**🎬 A · 欠拟合**（30 秒）

> "degree=1 就是直线。左图红线是直线，明显拟合不到曲线数据。
>  右图看：红点在最左边，train 和 test MSE 都高。
>  这就是 underfitting —— 模型太简单。"

🎯 **操作**：确认 degree=1，指出红线太直
""",
        "B": """
**🎬 B · 刚好（甜蜜点）**（30 秒）

> "degree=2 是二次曲线。左图红线完美贴合绿虚线（真实函数）。
>  右图看：红点落在绿钻石位置，test MSE 最低。
>  这就是 sweet spot —— 复杂度刚好。"

🎯 **操作**：拖 slider 到 2，指出红点和绿钻石重合
""",
        "C": """
**🎬 C · 微过拟合**（30 秒）

> "degree=5。左图红线开始在数据点间微微抖动。
>  右图：test MSE 开始升高（gap 变大），train MSE 继续降。
>  两条线分叉 = 过拟合的信号。"

🎯 **操作**：拖到 5，指出 gap 颜色变黄
""",
        "D": """
**🎬 D · 严重过拟合**（45 秒）

> "degree=12。左图红线疯狂振荡 —— 数据点之间剧烈抖动。
>  train MSE ≈ 0（几乎穿过每个训练点），但 test MSE 飙升。
>  这就是 high variance 的视觉证据。
>  教学锚点：红点相对绿钻石的位置 = 调参方向。"

🎯 **操作**：拖到 12，指出振荡 + gap 变红
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
左 sidebar 5col 控件 + 中右 stage 双槽 + 底 panel；narration / shot / truth_hint 全部录屏外。

### 横屏骨架

```
   0        5                              32
y=0   ┌────── 标题 (h=3) ─────────────────────────┐
y=3   ├ controls ┬──── stage 双槽 (h=26) ────────┤
      │ 5col     │  slot1: 拟合曲线（红线+绿虚线）│
      │ degree   │  slot2: U 形 MSE 双折线        │
      │ presets  │  始终并排显示（无切换）          │
      │          │                                 │
y=29  ├──────────┴──── panel (h=3) ──────────────┤  ← 640px < 720
y=36  ├──── 录屏外提示区 ───────────────────────┤
y=36  ├──── truth_hint (h=3) ───────────────────┤
y=39  ├──── shot dropdown (h=3) ────────────────┤
y=42  ├──── narration 口播稿 (h=12) ────────────┤
```

### Cell 索引 → grid 映射

| idx | cell 内容 | position |
|---|---|---|
| 0 | imports | null |
| 1 | 标题 | [0, 0, 32, 3] |
| 2 | 数据生成 | null |
| 3 | slider + shot 定义 | null |
| 4 | controls vstack | [0, 3, 5, 26] |
| 5 | 当前 degree 计算 | null |
| 6 | 全 degree MSE 预计算 | null |
| 7 | panel 计算 | null |
| 8 | chart_left 定义 | null |
| 9 | chart_right 定义 | null |
| 10 | stage 双槽 | [5, 3, 27, 26] |
| 11 | panel_view | [0, 29, 32, 3] |
| 12 | truth_hint | [0, 36, 10, 3] |
| 13 | shot_picker | [0, 39, 8, 3] |
| 14 | narration | [0, 42, 32, 12] |
| 15 | layout_doc | null |
                """
            )
        },
        multiple=False,
    )
    return


if __name__ == "__main__":
    app.run()
