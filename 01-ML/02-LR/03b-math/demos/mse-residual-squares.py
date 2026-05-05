"""
MSE 残差方块 · 红方块面积 = 误差² · 03b-math · e03 损失函数

教学核心：MSE = 红方块面积平均 → 拖 k/b 让方块缩小到最小 ⇔ 找最优拟合。

录屏布局（_5-layout-guide Strategy B 双槽）：
- A · 看方块：slot1 = chart_data 主图，slot2 = 残差表（5 行 x/y/ŷ/err²）
- B · 看 L(k) 切片：slot1 = chart_data，slot2 = chart_k_slice 抛物线
- C · 看 L(b) 切片：slot1 = chart_data，slot2 = chart_b_slice 抛物线
- D · 双抛物线对照：slot1 = chart_k_slice，slot2 = chart_b_slice

跑：
  cd 01-ML/02-LR/03b-math/demos && marimo edit --port 2757 mse-residual-squares.py
  cd 01-ML/02-LR/03b-math/demos && marimo run --port 2757 mse-residual-squares.py
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/mse-residual-squares.grid.json",
    css_file="custom.css",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import altair as alt
    import pandas as pd

    return alt, mo, np, pd


@app.cell(hide_code=True)
def title(mo):
    # 录屏内紧凑标题
    mo.md(
        r"### MSE 损失 · 红方块面积 = 误差² · 拖 $k,b$ 让方块缩到最小"
    ).style(margin="0", padding="4px 12px", font_size="15px", line_height="1.3")
    return


@app.cell
def _(np):
    # 数据：5 个身高-体重样本（中心化让 b_opt 在 1 附近，可视化范围友好）
    _rng = np.random.default_rng(42)
    x_data = np.linspace(-3, 3, 5)
    K_TRUE, B_TRUE = 0.97, 1.0
    y_data = K_TRUE * x_data + B_TRUE + _rng.normal(0, 1.0, size=5)
    return B_TRUE, K_TRUE, x_data, y_data


@app.cell
def _(np, x_data, y_data):
    # 闭式解
    _x_mean, _y_mean = x_data.mean(), y_data.mean()
    k_opt = float(
        np.sum((x_data - _x_mean) * (y_data - _y_mean))
        / np.sum((x_data - _x_mean) ** 2)
    )
    b_opt = float(_y_mean - k_opt * _x_mean)
    loss_opt = float(np.mean((k_opt * x_data + b_opt - y_data) ** 2))
    return b_opt, k_opt, loss_opt


@app.cell
def _(mo):
    # ===== 滑块（极简 label · sklearn / 数学符号）=====
    _S = dict(show_value=True, full_width=True)
    k_slider = mo.ui.slider(-2.0, 6.0, step=0.1, value=0.5, label="k", **_S)
    b_slider = mo.ui.slider(-4.0, 6.0, step=0.1, value=-2.0, label="b", **_S)

    shot = mo.ui.dropdown(
        options=["A · 残差表", "B · L(k) 切片", "C · L(b) 切片", "D · 双抛物线"],
        value="A · 残差表",
        label="🎬 镜头",
    )
    return b_slider, k_slider, shot


@app.cell
def controls(b_slider, k_slider, mo):
    # 控件组合（极致紧凑 vstack · sidebar 4col=207px）
    _h = lambda s: mo.md(s).style(
        margin="0", padding="0", font_size="11px",
        font_weight="700", color="#6b7280", letter_spacing="0.05em",
    )
    _div = mo.md("").style(
        border_top="1px solid #e5e7eb", margin="4px 0", padding="0", height="1px",
    )
    mo.vstack(
        [_h("参数"), k_slider, b_slider],
        gap=0,
        align="stretch",
    )
    return


@app.cell
def _(b_slider, k_slider):
    # 单一来源
    cur_k = float(k_slider.value)
    cur_b = float(b_slider.value)
    return cur_b, cur_k


@app.cell
def _(cur_b, cur_k, np, x_data, y_data):
    y_pred_cur = cur_k * x_data + cur_b
    cur_loss = float(np.mean((y_pred_cur - y_data) ** 2))
    cur_loss = float(np.clip(cur_loss, 0.0, 1e8))
    return cur_loss, y_pred_cur


@app.cell
def _(b_opt, cur_b, cur_k, cur_loss, k_opt, loss_opt, mo):
    # ===== 数字面板（录屏内）=====
    _ratio = cur_loss / loss_opt if loss_opt > 1e-9 else 0.0
    if cur_loss - loss_opt < 0.05:
        _color = "#10b981"
    elif cur_loss - loss_opt < 0.5:
        _color = "#f59e0b"
    else:
        _color = "#ef4444"

    _panel_md = f"""
    <div style="font-family:ui-monospace,monospace; font-size:13px; line-height:1.45;
            background:#f9fafb; border:1px solid #e5e7eb; border-radius:6px;
            padding:6px 12px; margin:0;">
    <b>当前</b> k={cur_k:.2f} b={cur_b:.2f}
    <span style="color:{_color}; font-weight:700">MSE={cur_loss:.3f}</span> &nbsp;|&nbsp;
    <b>最优</b> k*={k_opt:.2f} b*={b_opt:.2f}
    <span style="color:#6b7280">MSE*={loss_opt:.3f}</span> &nbsp;|&nbsp;
    <span style="color:#6b7280">当前 / 最优 = <b>{_ratio:.2f}×</b></span><br>
    <span style="color:#374151">→ <b>红方块面积 = 误差²</b>；MSE = 5 个方块面积的平均；缩到最小 ⇔ 拟合最好</span>
    </div>
    """
    panel = mo.md(_panel_md)
    return (panel,)


@app.cell
def _(alt, b_opt, cur_b, cur_k, k_opt, pd, x_data, y_data, y_pred_cur):
    # ============ chart_data · 散点 + 拟合线 + 残差方块 ============
    _df = pd.DataFrame(
        {
            "x": x_data,
            "y": y_data,
            "y_pred": y_pred_cur,
            "err_sq": (y_data - y_pred_cur) ** 2,
        }
    )

    _xs = pd.DataFrame({"x": [-4.0, 4.0]})
    _xs["y_cur"] = cur_k * _xs["x"] + cur_b
    _xs["y_opt"] = k_opt * _xs["x"] + b_opt

    _x_dom = [-4.2, 4.2]
    _y_dom = [-8, 10]

    _pts = (
        alt.Chart(_df)
        .mark_circle(size=160, color="#1f77b4", stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=_x_dom), title="x"),
            y=alt.Y("y:Q", scale=alt.Scale(domain=_y_dom), title="y"),
            tooltip=[
                alt.Tooltip("x:Q", format=".2f"),
                alt.Tooltip("y:Q", format=".2f"),
                alt.Tooltip("y_pred:Q", format=".2f", title="ŷ"),
                alt.Tooltip("err_sq:Q", format=".3f", title="err²"),
            ],
        )
    )
    _squares = (
        alt.Chart(_df)
        .mark_square(opacity=0.32, color="#ef4444", stroke="#ef4444", strokeWidth=1)
        .encode(
            x="x:Q",
            y="y:Q",
            size=alt.Size(
                "err_sq:Q",
                scale=alt.Scale(range=[50, 3000]),
                legend=None,
            ),
        )
    )
    _rules = (
        alt.Chart(_df)
        .mark_rule(color="#ef4444", opacity=0.55, strokeDash=[3, 2])
        .encode(x="x:Q", y="y:Q", y2="y_pred:Q")
    )
    _line_cur = (
        alt.Chart(_xs)
        .mark_line(color="#ef4444", strokeWidth=3)
        .encode(x="x:Q", y="y_cur:Q")
    )
    _line_opt = (
        alt.Chart(_xs)
        .mark_line(color="#10b981", strokeWidth=2.2, strokeDash=[6, 4])
        .encode(x="x:Q", y="y_opt:Q")
    )

    chart_data = (_rules + _squares + _pts + _line_cur + _line_opt).properties(
        width=460,
        height=460,
        title="数据 + 红方块（误差²） + 红线当前 / 绿虚最优",
    )
    return (chart_data,)


@app.cell
def _(alt, cur_b, cur_k, k_opt, np, pd, x_data, y_data):
    # ============ chart_k_slice · 固定 b=cur_b，loss 随 k 变化 ============
    _ks = np.linspace(-2.0, 6.0, 120)
    _losses = np.array([np.mean((kv * x_data + cur_b - y_data) ** 2) for kv in _ks])
    _losses = np.clip(_losses, 0.0, 1e8)
    _df_k = pd.DataFrame({"k": _ks, "loss": _losses})

    _cur_loss_k = float(np.mean((cur_k * x_data + cur_b - y_data) ** 2))
    _opt_loss_at_kopt = float(np.mean((k_opt * x_data + cur_b - y_data) ** 2))

    _parabola = (
        alt.Chart(_df_k)
        .mark_line(color="#3b82f6", strokeWidth=2.5)
        .encode(
            x=alt.X("k:Q", title="k"),
            y=alt.Y("loss:Q", title="L"),
        )
    )
    _cur_dot = (
        alt.Chart(pd.DataFrame({"k": [cur_k], "loss": [_cur_loss_k]}))
        .mark_circle(size=320, color="#ef4444", stroke="white", strokeWidth=2)
        .encode(x="k:Q", y="loss:Q")
    )
    _opt_dot = (
        alt.Chart(pd.DataFrame({"k": [k_opt], "loss": [_opt_loss_at_kopt]}))
        .mark_point(
            shape="diamond",
            size=240,
            filled=True,
            color="#10b981",
            stroke="black",
            strokeWidth=1.5,
        )
        .encode(x="k:Q", y="loss:Q")
    )

    chart_k_slice = (_parabola + _cur_dot + _opt_dot).properties(
        width=460,
        height=460,
        title=f"L(k) · 固定 b={cur_b:.2f} · 红=当前 绿钻=k*={k_opt:.2f}",
    )
    return (chart_k_slice,)


@app.cell
def _(alt, b_opt, cur_b, cur_k, np, pd, x_data, y_data):
    # ============ chart_b_slice · 固定 k=cur_k，loss 随 b 变化 ============
    _bs = np.linspace(-4.0, 6.0, 120)
    _losses = np.array([np.mean((cur_k * x_data + bv - y_data) ** 2) for bv in _bs])
    _losses = np.clip(_losses, 0.0, 1e8)
    _df_b = pd.DataFrame({"b": _bs, "loss": _losses})

    _cur_loss_b = float(np.mean((cur_k * x_data + cur_b - y_data) ** 2))
    _opt_loss_at_bopt = float(np.mean((cur_k * x_data + b_opt - y_data) ** 2))

    _parabola = (
        alt.Chart(_df_b)
        .mark_line(color="#8b5cf6", strokeWidth=2.5)
        .encode(
            x=alt.X("b:Q", title="b"),
            y=alt.Y("loss:Q", title="L"),
        )
    )
    _cur_dot = (
        alt.Chart(pd.DataFrame({"b": [cur_b], "loss": [_cur_loss_b]}))
        .mark_circle(size=320, color="#ef4444", stroke="white", strokeWidth=2)
        .encode(x="b:Q", y="loss:Q")
    )
    _opt_dot = (
        alt.Chart(pd.DataFrame({"b": [b_opt], "loss": [_opt_loss_at_bopt]}))
        .mark_point(
            shape="diamond",
            size=240,
            filled=True,
            color="#10b981",
            stroke="black",
            strokeWidth=1.5,
        )
        .encode(x="b:Q", y="loss:Q")
    )

    chart_b_slice = (_parabola + _cur_dot + _opt_dot).properties(
        width=460,
        height=460,
        title=f"L(b) · 固定 k={cur_k:.2f} · 红=当前 绿钻=b*={b_opt:.2f}",
    )
    return (chart_b_slice,)


@app.cell
def truth_hint(B_TRUE, K_TRUE, b_opt, k_opt, mo):
    # 真实参数 + 闭式解最优值（提示区，录屏 crop 掉）
    # 用纯 HTML div（避开 marimo-callout-output 在 h=3 受限高度下空白渲染）
    mo.md(
        f"""<div style="background:#dbeafe;color:#1e40af;border-left:4px solid #3b82f6;
        padding:6px 14px;border-radius:6px;font-size:13px;line-height:1.4;margin:0;">
        🎯 <b>真值</b> k={K_TRUE}, b={B_TRUE}（数据生成）
        &nbsp;·&nbsp;
        <b>闭式解最优</b> k*={k_opt:.3f}, b*={b_opt:.3f}（5 点拟合）
        </div>"""
    )
    return


@app.cell
def shot_picker(shot):
    # 镜头切换器（提示区，录屏 crop 掉）
    shot
    return


@app.cell
def stage(
    chart_b_slice,
    chart_data,
    chart_k_slice,
    cur_b,
    cur_k,
    mo,
    np,
    pd,
    shot,
    x_data,
    y_data,
    y_pred_cur,
):
    # 🎬 中央舞台 · Strategy B 双槽
    if shot.value.startswith("A"):
        # A: chart_data + 残差表（5 行）
        _slot1 = mo.ui.altair_chart(chart_data)
        _err_sq = (y_data - y_pred_cur) ** 2
        _df = pd.DataFrame(
            {
                "x": x_data.round(2),
                "y": y_data.round(2),
                "ŷ": y_pred_cur.round(2),
                "err²": _err_sq.round(3),
            }
        )
        _slot2 = mo.md(
            f"""
**📊 残差表 · 5 个数据点**

{_df.to_markdown(index=False)}

---

**MSE = 5 个 err² 的平均 = {float(np.mean(_err_sq)):.3f}**

口播：每个红方块面积 = 这一行的 err²；
拖 k/b 让 5 个方块全部缩小 ⇔ MSE 变小。
"""
        ).style(font_size="13px", line_height="1.55", padding="12px 16px")

    elif shot.value.startswith("B"):
        # B: chart_data + chart_k_slice
        _slot1 = mo.ui.altair_chart(chart_data)
        _slot2 = mo.ui.altair_chart(chart_k_slice)

    elif shot.value.startswith("C"):
        # C: chart_data + chart_b_slice
        _slot1 = mo.ui.altair_chart(chart_data)
        _slot2 = mo.ui.altair_chart(chart_b_slice)

    else:  # D · 双抛物线对照（双图模式 width 显式覆盖避免 X scroll）
        _slot1 = mo.ui.altair_chart(chart_k_slice.properties(width=400, height=440))
        _slot2 = mo.ui.altair_chart(chart_b_slice.properties(width=400, height=440))

    mo.hstack([_slot1, _slot2], gap=0.5, widths="equal", align="start")
    return


@app.cell
def panel_view(panel):
    # 数字面板（录屏内）
    panel
    return


@app.cell
def narration(mo, shot):
    # 口播稿：按 shot 切换（录屏外）
    _scripts = {
        "A": """
**🎬 A · 残差表 · 看红方块**（45 秒）

> "5 个数据点，每个点都有真实 y 和预测 ŷ。
>  误差 = y - ŷ；误差² = 红方块面积。
>  MSE = 5 个方块面积的平均。
>  拖 k 看红线斜率变 → 5 个方块大小同步变；
>  拖 b 看红线整体平移 → 同样让方块变。
>  目标：让 5 个方块全部缩到最小。"

🎯 **目标**：调到 k*=0.80, b*=0.80（闭式解最优值）→ MSE→1.118
""",
        "B": """
**🎬 B · L(k) 切片 · 抛物线视角**（60 秒）

> "把所有 k 都试一遍，每个 k 对应一个 MSE，画出来就是右边的蓝色抛物线。
>  红圆点 = 当前 k 在抛物线上的位置。
>  绿钻石 = 最优 k* 的位置。
>  拖 k 滑块 → 红点沿着抛物线左右滚 → 滚到谷底就是 k*。
>  抛物线 = 凸函数 → 谷底唯一 → 这就是为什么 LR 有解析解。"
""",
        "C": """
**🎬 C · L(b) 切片 · 同理换 b**（45 秒）

> "把 k 固定，让 b 变化，又是一条抛物线（紫色）。
>  这次拖 b 滑块 → 红点沿紫色抛物线滚。
>  谷底 = b*（在 k 固定的前提下）。
>  k、b 各自都是凸的 → 整体也是凸的（2D 碗）。"
""",
        "D": """
**🎬 D · 双抛物线对照 · 两个方向都凸**（30 秒）

> "L(k) 蓝色 + L(b) 紫色，并排看。
>  两个方向各自的最低点 → 两个偏导 = 0
>  → 联立解出 (k*, b*)。
>  下期讲偏导 + 矩阵 → 推正规方程。"

🔑 **关键**：MSE 关于 (k, b) 是凸函数 → 最优解唯一 → 闭式可解
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
    # 📐 录屏 grid 布局参考（录屏 position=null 隐藏）
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
      │ k slider │   A: chart_data | 残差表       │
      │ b slider │   B: chart_data | L(k) 抛物线 │
      │          │   C: chart_data | L(b) 抛物线 │
      │          │   D: L(k) | L(b) 双抛物线     │
y=29  ├──────────┴──── panel (h=3) ──────────────┤  ← 720 内 = y=36
y=32  ├──── 录屏外提示区 (y > 36 = 720 外)──────┤
y=32  ├──── shot dropdown (h=3) ─────────────────┤
y=35  ├──── truth_hint callout (h=3) ────────────┤
y=38  ├──── narration 口播稿 (h=12) ─────────────┤
```

### 镜头脚本（Strategy B 双槽）

| # | 时长 | slot1 | slot2 | 教学焦点 |
|---|---|---|---|---|
| **A** | 0-45s | chart_data | 残差表 | 红方块 = err²，MSE = 平均 |
| **B** | 45-105s | chart_data | L(k) 抛物线 | 拖 k 看红点滚向谷底 |
| **C** | 105-150s | chart_data | L(b) 抛物线 | 同理换 b 方向 |
| **D** | 150-180s | L(k) | L(b) | 两个方向都凸 → 偏导=0 |
                """
            )
        },
        multiple=False,
    )
    return


if __name__ == "__main__":
    app.run()
