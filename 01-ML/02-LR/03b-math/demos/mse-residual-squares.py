"""
MSE 残差方块 · k/b 怎么影响拟合 · 03b-math 配套 demo

本 demo 演示概念：
  1) MSE 损失函数 = 所有样本误差²（红色方块）的平均
  2) 拖 k、b 滑块 → 拟合直线和方块实时变 → loss 数字实时变
  3) 1D 抛物线切片：固定一个参数扫另一个 → 红点滚向绿钻石（最优）

启动命令（端口 2730）:
  cd /Users/xuelin/projects/ai-engineer-roadmap
  .venv/bin/marimo edit --port 2730 --headless --no-token \
      01-ML/02-LR/03b-math/demos/mse-residual-squares.py

  # 部署只读运行：
  .venv/bin/marimo run --port 2730 --headless --no-token \
      01-ML/02-LR/03b-math/demos/mse-residual-squares.py

互动玩法（≤ 5 条）:
  - 选预设 → 看典型场景的红方块和 loss
  - 切回 ✋ 手动 → 滑块解锁
  - 拖 k 滑块 → A 视图直线斜率变 + B1 抛物线红点左右移
  - 拖 b 滑块 → A 视图直线上下移 + B2 抛物线红点左右移
  - 看绿色虚线/绿钻石 = 最优解（闭式解）位置
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
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return alt, mo, np, pd


@app.cell
def _(mo):
    mo.md(
        r"""
        # MSE 损失函数最小化 · k、b 怎么影响拟合效果

        用直线 $\hat{y} = k x + b$ 拟合 5 个身高-体重样本。**目标**：让红方块越小越好。

        - **A · 散点 + 拟合 + 残差方块**：红方块大小 ∝ 误差²，MSE = 方块平均面积
        - **B1 · loss 随 k 变化**（固定 b 当前值）：1D 抛物线，红点滚向绿钻石
        - **B2 · loss 随 b 变化**（固定 k 当前值）：同 B1，颜色紫色

        损失公式（评审建议 3）：
        $$L(k, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - k x_i - b)^2$$
        """
    )
    return


@app.cell
def _(np):
    # 数据：5 个身高-体重样本（对应 03b README slide 24 算例）
    # 中心化到 0 附近，避免 b_opt = -95 这种数值大幅震荡使可视化范围难看
    _rng = np.random.default_rng(42)
    x_data = np.linspace(-3, 3, 5)
    k_true, b_true = 0.97, 1.0
    y_data = k_true * x_data + b_true + _rng.normal(0, 1.0, size=5)
    return x_data, y_data


@app.cell
def _(np, x_data, y_data):
    # 闭式解（最优 k*, b*）—— design 第 91 行公式
    _x_mean, _y_mean = x_data.mean(), y_data.mean()
    k_opt = float(np.sum((x_data - _x_mean) * (y_data - _y_mean)) / np.sum((x_data - _x_mean) ** 2))
    b_opt = float(_y_mean - k_opt * _x_mean)
    loss_opt = float(np.mean((k_opt * x_data + b_opt - y_data) ** 2))
    return b_opt, k_opt, loss_opt


@app.cell
def _(mo):
    # 预设场景（review 软建议 5：标签精简）
    # value 是 (k0, b0)；None 代表手动模式
    PRESETS = {
        "✋ 手动": None,
        "✓ 完美": (0.97, 1.0),
        "📉 k 偏小": (0.3, 1.0),
        "📈 b 偏大": (0.97, 4.0),
        "🚀 远距": (-1.5, 4.5),
        "🔄 反向": (-0.8, 2.0),
    }
    preset = mo.ui.dropdown(options=PRESETS, value="✋ 手动", label="预设场景")

    # 手动滑块（仅手动模式生效）—— review 必修 1：删除 frame 滑块
    k_slider = mo.ui.slider(-2.0, 6.0, step=0.1, value=0.5, label="k 斜率", show_value=True)
    b_slider = mo.ui.slider(-4.0, 6.0, step=0.1, value=-2.0, label="b 截距", show_value=True)
    return PRESETS, b_slider, k_slider, preset


@app.cell
def _(b_slider, k_slider, mo, preset):
    # review 必修 2：明确预设/手动状态转移
    #   - 选预设：滑块隐藏，改成只读展示预设值（用户看到"预设把参数定到这里"）
    #   - 切回 ✋ 手动：滑块出现并解锁，保留当前值
    # marimo 的 slider 没有原生 disabled，所以用"显示/隐藏切换"达成等效效果
    if preset.value is None:
        _control_block = mo.vstack(
            [
                mo.md(
                    '<span style="background:#dbeafe;color:#1e40af;padding:2px 10px;'
                    'border-radius:4px;font-size:13px;font-weight:600">'
                    "🔵 手动模式 · 拖滑块改 k、b</span>"
                ),
                k_slider,
                b_slider,
            ]
        )
    else:
        _k0, _b0 = preset.value
        _control_block = mo.vstack(
            [
                mo.md(
                    '<span style="background:#fef3c7;color:#92400e;padding:2px 10px;'
                    'border-radius:4px;font-size:13px;font-weight:600">'
                    f"🟡 预设模式 · 切回 ✋ 手动 解锁滑块</span>"
                ),
                mo.md(
                    f'<div style="font-family:ui-monospace,monospace;color:#6b7280;font-size:14px">'
                    f"  k = <b>{_k0:.2f}</b>（已锁定） · b = <b>{_b0:.2f}</b>（已锁定）</div>"
                ),
            ]
        )

    mo.vstack([preset, _control_block], gap=1)
    return


@app.cell
def _(b_slider, k_slider, preset):
    # 单一来源：解析当前 k、b（手动 ⇒ 滑块；预设 ⇒ 字典值）
    if preset.value is None:
        cur_k = float(k_slider.value)
        cur_b = float(b_slider.value)
        is_preset = False
    else:
        _k0, _b0 = preset.value
        cur_k = float(_k0)
        cur_b = float(_b0)
        is_preset = True
    return cur_b, cur_k, is_preset


@app.cell
def _(cur_b, cur_k, np, x_data, y_data):
    # 当前预测和当前 loss
    y_pred_cur = cur_k * x_data + cur_b
    cur_loss = float(np.mean((y_pred_cur - y_data) ** 2))
    # clip 防 NaN/inf（极端预设也不会爆）
    cur_loss = float(np.clip(cur_loss, 0.0, 1e8))
    return cur_loss, y_pred_cur


@app.cell
def _(b_opt, cur_b, cur_k, cur_loss, is_preset, k_opt, loss_opt, mo):
    # review 必修 4：完整指标面板（当前 k/b/loss + 最优 k*/b*/loss* + 比值）
    _ratio = cur_loss / loss_opt if loss_opt > 1e-9 else 0.0
    if cur_loss - loss_opt < 0.05:
        _color = "#10b981"
    elif cur_loss - loss_opt < 0.5:
        _color = "#f59e0b"
    else:
        _color = "#ef4444"

    _mode_badge = (
        '<span style="background:#fef3c7;color:#92400e;padding:3px 10px;'
        'border-radius:4px;font-size:13px;font-weight:600">🟡 预设模式</span>'
        if is_preset
        else '<span style="background:#dbeafe;color:#1e40af;padding:3px 10px;'
        'border-radius:4px;font-size:13px;font-weight:600">🔵 手动模式</span>'
    )

    mo.md(
        f"""
<div style="display:flex; gap:20px; align-items:center; flex-wrap:wrap;
            font-family:ui-monospace,monospace; font-size:14px;
            background:#f9fafb; padding:12px 16px; border-radius:8px;
            border-left:4px solid {_color};">
  <div>{_mode_badge}</div>
  <div>当前 <b>k={cur_k:.3f}</b>, <b>b={cur_b:.3f}</b></div>
  <div>当前 Loss = <b style="color:{_color}; font-size:18px">{cur_loss:.3f}</b></div>
  <div style="color:#6b7280">最优 k*={k_opt:.3f}, b*={b_opt:.3f}, Loss*={loss_opt:.3f}</div>
  <div style="color:#6b7280">当前/最优 = <b>{_ratio:.2f}×</b></div>
</div>
"""
    )
    return


@app.cell
def _(alt, b_opt, cur_b, cur_k, k_opt, pd, x_data, y_data, y_pred_cur):
    # ============ A 视图 · 数据点 + 拟合 + 残差方块 ============
    # review 必修 3：自解释标题
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

    _pts = (
        alt.Chart(_df)
        .mark_circle(size=160, color="#1f77b4", stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X("x", scale=alt.Scale(domain=[-4.2, 4.2]), title="x（中心化身高）"),
            y=alt.Y("y", scale=alt.Scale(domain=[-8, 10]), title="y（中心化体重）"),
            tooltip=[
                alt.Tooltip("x", format=".2f"),
                alt.Tooltip("y", format=".2f"),
                alt.Tooltip("y_pred", format=".2f", title="预测"),
                alt.Tooltip("err_sq", format=".3f", title="误差²"),
            ],
        )
    )
    # 残差方块：size 编码 err²（屏幕像素，非数据空间），review 必修 5：altair 中文友好
    _squares = (
        alt.Chart(_df)
        .mark_square(opacity=0.32, color="#ef4444", stroke="#ef4444", strokeWidth=1)
        .encode(
            x="x",
            y="y",
            size=alt.Size(
                "err_sq:Q",
                scale=alt.Scale(range=[50, 3000]),
                legend=None,
            ),
            tooltip=[alt.Tooltip("err_sq", format=".3f", title="误差²")],
        )
    )
    # 残差竖线：连接真实点和预测点
    _rules = (
        alt.Chart(_df)
        .mark_rule(color="#ef4444", opacity=0.55, strokeDash=[3, 2])
        .encode(x="x", y="y", y2="y_pred")
    )
    _line_cur = (
        alt.Chart(_xs)
        .mark_line(color="#ef4444", strokeWidth=3)
        .encode(x="x", y="y_cur")
    )
    _line_opt = (
        alt.Chart(_xs)
        .mark_line(color="#10b981", strokeWidth=2.2, strokeDash=[6, 4])
        .encode(
            x="x",
            y="y_opt",
            tooltip=[
                alt.Tooltip("y_opt", format=".2f", title="最优拟合"),
            ],
        )
    )

    chart_data = (_rules + _squares + _pts + _line_cur + _line_opt).properties(
        width=440,
        height=340,
        title="A · 数据点 + 当前拟合（红线）+ 最优拟合（绿虚）+ 残差平方（红方块越大误差越大）",
    )
    return (chart_data,)


@app.cell
def _(alt, cur_b, cur_k, k_opt, np, pd, x_data, y_data):
    # ============ B1 视图 · 固定 b=cur_b，loss 随 k 变化 ============
    _ks = np.linspace(-2.0, 6.0, 120)
    _losses = np.array([np.mean((kv * x_data + cur_b - y_data) ** 2) for kv in _ks])
    _losses = np.clip(_losses, 0.0, 1e8)
    _df_k = pd.DataFrame({"k": _ks, "loss": _losses})

    _cur_loss_k = float(np.mean((cur_k * x_data + cur_b - y_data) ** 2))
    # 该切片上 k 的最优（b 固定时） — 解析为 k_slice* = sum(x*(y-b))/sum(x²)
    _k_slice_opt = float(
        np.sum(x_data * (y_data - cur_b)) / np.sum(x_data ** 2)
    )
    _opt_loss_k = float(np.mean((_k_slice_opt * x_data + cur_b - y_data) ** 2))

    _parabola = (
        alt.Chart(_df_k)
        .mark_line(color="#3b82f6", strokeWidth=2.5)
        .encode(
            x=alt.X("k", title="k（斜率）"),
            y=alt.Y("loss", title="Loss"),
        )
    )
    _cur_dot = (
        alt.Chart(pd.DataFrame({"k": [cur_k], "loss": [_cur_loss_k]}))
        .mark_circle(size=320, color="#ef4444", stroke="white", strokeWidth=2)
        .encode(
            x="k",
            y="loss",
            tooltip=[
                alt.Tooltip("k", format=".3f", title="当前 k"),
                alt.Tooltip("loss", format=".3f", title="当前 loss"),
            ],
        )
    )
    # 绿钻石：标记 k_opt（全局最优 k*），即使在 b≠b_opt 切片上也展示位置
    _opt_loss_at_kopt = float(np.mean((k_opt * x_data + cur_b - y_data) ** 2))
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
        .encode(
            x="k",
            y="loss",
            tooltip=[
                alt.Tooltip("k", format=".3f", title="最优 k*"),
                alt.Tooltip("loss", format=".3f", title="此切片 loss"),
            ],
        )
    )

    chart_k_slice = (_parabola + _cur_dot + _opt_dot).properties(
        width=440,
        height=260,
        title=f"B1 · 固定 b={cur_b:.2f}，loss 随 k 变化（红=当前 k，绿钻=最优 k*={k_opt:.2f}）",
    )
    return (chart_k_slice,)


@app.cell
def _(alt, b_opt, cur_b, cur_k, np, pd, x_data, y_data):
    # ============ B2 视图 · 固定 k=cur_k，loss 随 b 变化 ============
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
            x=alt.X("b", title="b（截距）"),
            y=alt.Y("loss", title="Loss"),
        )
    )
    _cur_dot = (
        alt.Chart(pd.DataFrame({"b": [cur_b], "loss": [_cur_loss_b]}))
        .mark_circle(size=320, color="#ef4444", stroke="white", strokeWidth=2)
        .encode(
            x="b",
            y="loss",
            tooltip=[
                alt.Tooltip("b", format=".3f", title="当前 b"),
                alt.Tooltip("loss", format=".3f", title="当前 loss"),
            ],
        )
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
        .encode(
            x="b",
            y="loss",
            tooltip=[
                alt.Tooltip("b", format=".3f", title="最优 b*"),
                alt.Tooltip("loss", format=".3f", title="此切片 loss"),
            ],
        )
    )

    chart_b_slice = (_parabola + _cur_dot + _opt_dot).properties(
        width=440,
        height=260,
        title=f"B2 · 固定 k={cur_k:.2f}，loss 随 b 变化（红=当前 b，绿钻=最优 b*={b_opt:.2f}）",
    )
    return (chart_b_slice,)


@app.cell
def _(chart_b_slice, chart_data, chart_k_slice, mo):
    # 主面板：A（上）+ B1/B2（下并排）
    mo.vstack(
        [
            mo.ui.altair_chart(chart_data),
            mo.hstack(
                [
                    mo.ui.altair_chart(chart_k_slice),
                    mo.ui.altair_chart(chart_b_slice),
                ],
                gap=1,
                widths=[1, 1],
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ### 教学要点回顾

        1. **MSE = 平方误差的平均**：A 视图的红方块面积之和 / 样本数 = Loss 数字
        2. **最小 loss ⇔ 最优模型**：绿色虚线（A）和绿钻石（B1/B2）指向同一个最优解 $(k^*, b^*)$
        3. **导数为 0 是极值点**：B1/B2 的抛物线谷底切线水平，对应 $\partial L / \partial k = 0$、$\partial L / \partial b = 0$
        4. **闭式解公式**（slide 24 展开 + 偏导后求零点）：
           $$k^* = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}, \quad b^* = \bar{y} - k^* \bar{x}$$
        5. **下一站**（04a/04b）：把 1D 抛物线升维到 2D 等高线/3D 曲面，再讲梯度下降迭代逼近
        """
    )
    return


if __name__ == "__main__":
    app.run()
