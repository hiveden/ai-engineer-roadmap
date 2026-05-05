"""
LR 决策边界 · 几何隐喻：直线 w₁x₁ + w₂x₂ + b = 0 切两类

互动：拖 w₁ / w₂ / b / 阈值 τ 滑块 → 决策边界实时移动 + 概率热图重染色
跑：marimo edit 01-decision-boundary.py --port 2733 --headless --no-token
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/01-decision-boundary.grid.json",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    return alt, mo, np, pd


@app.cell
def _(mo):
    mo.md(r"""
    # LR 决策边界 · 直线切两类

    $$z = w_1 x_1 + w_2 x_2 + b, \quad p = \sigma(z), \quad \hat{y} = \mathbb{1}[p \geq \tau]$$

    LR 的决策面**只能是线性的** —— 二维下是直线 $w_1 x_1 + w_2 x_2 + b = 0$。
    拖滑块看直线怎么转 / 平移；调阈值 τ 看判定边界如何整体平移。
    """)
    return


@app.cell
def _(mo):
    w1 = mo.ui.slider(-3.0, 3.0, value=1.5, step=0.1, label="w₁", show_value=True)
    w2 = mo.ui.slider(-3.0, 3.0, value=1.0, step=0.1, label="w₂", show_value=True)
    b = mo.ui.slider(-3.0, 3.0, value=-0.5, step=0.1, label="b（偏置）", show_value=True)
    tau = mo.ui.slider(0.1, 0.9, value=0.5, step=0.05, label="阈值 τ", show_value=True)
    mo.hstack([w1, w2, b, tau], widths=[1, 1, 1, 1], justify="space-around")
    return b, tau, w1, w2


@app.cell
def _(np):
    # 单一来源：合成两类样本（线性可分 + 少量噪声）
    rng = np.random.default_rng(7)
    _n = 60
    _pos = rng.normal(loc=[1.2, 1.0], scale=0.7, size=(_n, 2))
    _neg = rng.normal(loc=[-1.2, -0.8], scale=0.7, size=(_n, 2))
    X_data = np.vstack([_pos, _neg])
    y_data = np.array([1] * _n + [0] * _n)
    return X_data, y_data


@app.cell
def _(X_data, b, np, tau, w1, w2, y_data):
    # 衍生：当前 (w, b) 对每个样本的 logits / 概率 / 预测
    _z = X_data[:, 0] * w1.value + X_data[:, 1] * w2.value + b.value
    p_data = 1.0 / (1.0 + np.exp(-_z))
    y_pred = (p_data >= tau.value).astype(int)
    acc = float((y_pred == y_data).mean())
    # logit 阈值（决策边界右移 / 左移 由 τ 控制）
    z_thresh = float(np.log(tau.value / (1.0 - tau.value)))
    return acc, p_data, y_pred, z_thresh


@app.cell
def _(X_data, alt, b, np, p_data, pd, w1, w2, y_data, y_pred, z_thresh):
    # 视图 1：决策边界 + 概率热图
    _xs = np.linspace(-4, 4, 80)
    _ys = np.linspace(-4, 4, 80)
    _xx, _yy = np.meshgrid(_xs, _ys)
    _grid_z = _xx * w1.value + _yy * w2.value + b.value
    _grid_p = 1.0 / (1.0 + np.exp(-_grid_z))
    _heat_df = pd.DataFrame({
        "x1": _xx.ravel(), "x2": _yy.ravel(), "p": _grid_p.ravel()
    })
    heat = alt.Chart(_heat_df).mark_rect(opacity=0.55).encode(
        x=alt.X("x1:Q", bin=alt.Bin(maxbins=80), scale=alt.Scale(domain=[-4, 4])),
        y=alt.Y("x2:Q", bin=alt.Bin(maxbins=80), scale=alt.Scale(domain=[-4, 4])),
        color=alt.Color("p:Q",
            scale=alt.Scale(domain=[0, 0.5, 1], range=["#dc2626", "#f8fafc", "#16a34a"]),
            legend=alt.Legend(title="P(y=1)", orient="right")),
    )

    # 决策边界直线：w1*x1 + w2*x2 + b = z_thresh → x2 = (z_thresh - b - w1*x1)/w2
    if abs(w2.value) > 1e-3:
        _line_x1 = np.linspace(-4, 4, 50)
        _line_x2 = (z_thresh - b.value - w1.value * _line_x1) / w2.value
    else:
        _line_x1 = np.full(50, (z_thresh - b.value) / max(w1.value, 1e-3))
        _line_x2 = np.linspace(-4, 4, 50)
    _line_df = pd.DataFrame({"x1": _line_x1, "x2": _line_x2})
    line = alt.Chart(_line_df).mark_line(stroke="#0f172a", strokeWidth=2.5).encode(
        x=alt.X("x1:Q", scale=alt.Scale(domain=[-4, 4])),
        y=alt.Y("x2:Q", scale=alt.Scale(domain=[-4, 4])),
    )

    # 样本点（按真实标签 + 是否预测对）
    _pts_df = pd.DataFrame({
        "x1": X_data[:, 0], "x2": X_data[:, 1],
        "真实": ["正" if t == 1 else "负" for t in y_data],
        "预测": ["正" if t == 1 else "负" for t in y_pred],
        "对错": ["✓" if y_data[i] == y_pred[i] else "✗" for i in range(len(y_data))],
        "p": p_data,
    })
    pts = alt.Chart(_pts_df).mark_circle(size=90, stroke="#0f172a", strokeWidth=0.8).encode(
        x="x1:Q", y="x2:Q",
        color=alt.Color("真实:N",
            scale=alt.Scale(domain=["正", "负"], range=["#16a34a", "#dc2626"]),
            legend=alt.Legend(title="真实标签", orient="top-right")),
        shape=alt.Shape("对错:N",
            scale=alt.Scale(domain=["✓", "✗"], range=["circle", "cross"]),
            legend=alt.Legend(title="预测", orient="top-right")),
        tooltip=["真实:N", "预测:N", "对错:N", alt.Tooltip("p:Q", format=".3f")],
    )

    chart_boundary = (heat + line + pts).resolve_scale(color="independent").properties(
        width=420, height=380)
    return (chart_boundary,)


@app.cell
def _(alt, np, pd, tau, z_thresh):
    # 视图 2：sigmoid 曲线 + τ 切点（揭示阈值 → logits 阈值的映射）
    _zs = np.linspace(-6, 6, 200)
    _ps = 1.0 / (1.0 + np.exp(-_zs))
    curve = alt.Chart(pd.DataFrame({"z": _zs, "p": _ps})).mark_line(
        stroke="#2563eb", strokeWidth=2.5).encode(
        x=alt.X("z:Q", scale=alt.Scale(domain=[-6, 6]), title="z = w·x + b"),
        y=alt.Y("p:Q", scale=alt.Scale(domain=[0, 1]), title="P(y=1) = σ(z)"),
    )
    hl = alt.Chart(pd.DataFrame({"y": [tau.value]})).mark_rule(
        stroke="#9a3412", strokeDash=[5, 4], strokeWidth=1.5).encode(y="y:Q")
    vl = alt.Chart(pd.DataFrame({"x": [z_thresh]})).mark_rule(
        stroke="#9a3412", strokeDash=[5, 4], strokeWidth=1.5).encode(x="x:Q")
    pt = alt.Chart(pd.DataFrame({"z": [z_thresh], "p": [tau.value]})).mark_circle(
        color="#9a3412", size=180).encode(x="z:Q", y="p:Q")
    chart_sigmoid = (curve + hl + vl + pt).properties(width=420, height=260)
    return (chart_sigmoid,)


@app.cell
def _(chart_boundary, chart_sigmoid, mo):
    mo.hstack([chart_boundary, chart_sigmoid],
              widths=[1.1, 1], justify="space-around")
    return


@app.cell
def _(acc, b, mo, tau, w1, w2, y_data, y_pred, z_thresh):
    # 状态卡：当前参数 / 准确率 / 错分统计
    n_pos = int((y_data == 1).sum())
    n_neg = int((y_data == 0).sum())
    fn = int(((y_data == 1) & (y_pred == 0)).sum())  # 漏报
    fp = int(((y_data == 0) & (y_pred == 1)).sum())  # 误报
    acc_color = "#16a34a" if acc >= 0.85 else ("#eab308" if acc >= 0.7 else "#dc2626")
    if abs(w1.value) < 0.3 and abs(w2.value) < 0.3:
        tip = "权重几乎归零 → 模型只看 b，所有点同概率，决策边界悬空"
        tip_c = "#dc2626"
    elif tau.value > 0.7:
        tip = "阈值高 → 偏精确率（少误报多漏报，FN ↑）"
        tip_c = "#9a3412"
    elif tau.value < 0.3:
        tip = "阈值低 → 偏召回率（少漏报多误报，FP ↑）"
        tip_c = "#9a3412"
    else:
        tip = "阈值在均衡区，FN/FP 大致对称"
        tip_c = "#0f172a"
    _box = "padding:10px 14px;border-radius:6px;font-size:13px;line-height:1.7;"
    mo.md(f"""
<div style="display:flex;gap:10px;margin:6px 0;">
  <div style="flex:1;background:#f8fafc;border:1px solid #e2e8f0;{_box}">
    <strong>模型参数</strong><br>
    w = [<code>{w1.value:.2f}</code>, <code>{w2.value:.2f}</code>]<br>
    b = <code>{b.value:.2f}</code><br>
    决策边界：<code>{w1.value:.2f}·x₁ + {w2.value:.2f}·x₂ + {b.value:.2f} = {z_thresh:.2f}</code>
  </div>
  <div style="flex:1;background:#f0f9ff;border-left:4px solid {acc_color};{_box}">
    <strong style="color:{acc_color};">准确率 = {acc:.1%}</strong><br>
    样本：{n_pos} 正 / {n_neg} 负<br>
    漏报 FN = <strong style="color:#dc2626;">{fn}</strong> · 误报 FP = <strong style="color:#dc2626;">{fp}</strong>
  </div>
  <div style="flex:1.2;background:#fff7ed;border-left:4px solid {tip_c};{_box}">
    <strong style="color:#9a3412;">阈值 τ = {tau.value:.2f}</strong>
    （等价 logits 阈 z* = {z_thresh:.2f}）<br>
    {tip}
  </div>
</div>
""")
    return


@app.cell
def _(mo):
    mo.accordion({
        "为什么 LR 决策边界是直线？": mo.md(r"""
$\hat{y} = 1 \Leftrightarrow \sigma(z) \geq \tau \Leftrightarrow z \geq \log\dfrac{\tau}{1-\tau}$
（sigmoid 单调）。

带回 $z = w^\top x + b$ 得 $w^\top x + b \geq z^*$，**等号即决策边界**——
$x$ 是变量，$w$/$b$/$z^*$ 是常数，方程线性，所以边界是**直线 / 平面 / 超平面**。

阈值 τ 改变只是把这条直线平行平移（截距变），不会让它变弯。
"""),
        "反例：同心圆 LR 怎么转都画不出圆": mo.md(r"""
内圈 0、外圈 1 的同心圆数据，**任何直线都无法切开**。LR 的训练
准确率会停在 ~50%（随机水平）。

判断信号：
- 训练集 acc 怎么调超参都上不去 → LR 表达力不够，不是优化问题
- 看决策边界图：边界永远穿过密集点 → 数据非线性可分

**解法**：
- 特征工程加多项式 $(x_1^2, x_2^2, x_1 x_2)$ 把数据映射到高维（kernel trick 雏形）
- 换决策树 / SVM 核方法 / 神经网络
"""),
        "阈值 τ 调整 vs 偏置 b 调整": mo.md(r"""
两者**几何上等价**——都是平移决策边界的方式。

- 调 b：平移整条直线，所有概率重新计算
- 调 τ：保持 sigmoid 不变，只移决策门槛

工程上倾向**先训出最优 (w, b) 后调 τ**：
- 训练目标是凸损失最小（固定 τ=0.5 求 w, b）
- 部署时按业务调 τ（不重训）
- `predict_proba()[:, 1] >= τ` 自定义阈值
"""),
        "权重 w 的几何意义": mo.md(r"""
$w$ **垂直于决策边界**，指向"正类方向"。

- $|w|$ 越大 → 直线两侧 $\sigma$ 变化越快 → 决策越自信（概率两极化）
- $|w|$ 接近 0 → 整片区域 $\sigma \approx 0.5$ → 模型基本不区分

直觉验证：把 w₁、w₂ 都拖到 0.1 附近，热图整片变灰（接近 0.5），
所有点的预测概率都在 ~0.5。
"""),
    }, multiple=False)
    return


@app.cell
def _(mo):
    mo.md("""
## Grid 布局参考（16:9 · 36 行）

```
   0           12          24
0  ┌───── 标题 h=4 全宽 ─────┐
4  ├ w₁ ┬ w₂ ┬ b ┬ τ  h=4 ──┤
8  ├──── 状态卡 h=5 全宽 ────┤
13 ├ 决策边界(13) ┬ sigmoid(11) ─┤  ← h=18
31 ├──── accordion h=5 全宽 ─────┤
36
```
""")
    return


if __name__ == "__main__":
    app.run()
