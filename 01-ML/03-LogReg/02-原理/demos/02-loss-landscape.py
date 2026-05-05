"""
LR 损失景观 · MSE 非凸 vs log-loss 凸（梯度下降为什么必须换损失）

互动：
- 切换 MSE / log-loss 看 1D w 下的曲面
- 拖单样本 (y, p) 看 -log(p) / -log(1-p) 惩罚曲线
跑：marimo edit 02-loss-landscape.py --port 2734 --headless --no-token
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/02-loss-landscape.grid.json",
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
    # 损失景观 · 凸 vs 非凸

    **MSE + sigmoid** 在分类上 → 多个局部最小（梯度卡住）
    **log-loss + sigmoid** → 凸函数，梯度下降必到全局最优

    左图扫一维参数 $w$ 看曲面；右图固定 $y$ 拖预测概率 $p$ 看单样本惩罚。
    """)
    return


@app.cell
def _(mo):
    loss_kind = mo.ui.dropdown(
        options={"对数似然损失 (log-loss)": "log",
                 "均方误差 (MSE)": "mse",
                 "并排对比": "both"},
        value="并排对比", label="损失函数")
    y_true = mo.ui.dropdown(
        options={"y = 1（正例）": 1, "y = 0（负例）": 0},
        value="y = 1（正例）", label="单样本真实标签")
    p_pred = mo.ui.slider(0.01, 0.99, value=0.5, step=0.01,
                          label="单样本预测概率 p", show_value=True)
    mo.hstack([loss_kind, y_true, p_pred],
              widths=[1.3, 1, 1.5], justify="space-around")
    return loss_kind, p_pred, y_true


@app.cell
def _(np):
    # 单一来源：合成 1D 数据 (x, y) 用来扫 w 计算 J(w)
    rng = np.random.default_rng(11)
    _n = 30
    _x_pos = rng.normal(loc=1.5, scale=0.8, size=_n // 2)
    _x_neg = rng.normal(loc=-1.5, scale=0.8, size=_n // 2)
    X1 = np.concatenate([_x_pos, _x_neg])
    Y1 = np.array([1.0] * (_n // 2) + [0.0] * (_n // 2))
    return X1, Y1


@app.cell
def _(X1, Y1, np):
    # 扫 w ∈ [-6, 6]，分别算 MSE 和 log-loss（固定 b=0 简化）
    _ws = np.linspace(-6, 6, 240)
    _eps = 1e-12
    mse_curve = []
    log_curve = []
    for _w in _ws:
        _z = _w * X1
        _p = 1.0 / (1.0 + np.exp(-_z))
        mse_curve.append(float(np.mean((_p - Y1) ** 2)))
        log_curve.append(float(-np.mean(
            Y1 * np.log(_p + _eps) + (1 - Y1) * np.log(1 - _p + _eps))))
    ws = _ws
    mse_curve = np.array(mse_curve)
    log_curve = np.array(log_curve)
    # 找局部极小（朴素：相邻三点 < 比较）
    def _local_mins(arr):
        idx = []
        for i in range(1, len(arr) - 1):
            if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
                idx.append(i)
        return idx
    mse_minima = _local_mins(mse_curve)
    log_minima = _local_mins(log_curve)
    return log_curve, log_minima, mse_curve, mse_minima, ws


@app.cell
def _(alt, log_curve, log_minima, loss_kind, mse_curve, mse_minima, pd, ws):
    # 视图 1：损失曲面（按选择渲染 1 / 2 条）
    _frames = []
    if loss_kind.value in ("log", "both"):
        _frames.append(pd.DataFrame({
            "w": ws, "loss": log_curve, "kind": "log-loss（凸）"}))
    if loss_kind.value in ("mse", "both"):
        _frames.append(pd.DataFrame({
            "w": ws, "loss": mse_curve, "kind": "MSE（非凸）"}))
    _df = pd.concat(_frames, ignore_index=True)
    _ymax = float(_df["loss"].max() * 1.1)
    curves = alt.Chart(_df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("w:Q", scale=alt.Scale(domain=[-6, 6]), title="权重 w（b 固定为 0）"),
        y=alt.Y("loss:Q", scale=alt.Scale(domain=[0, _ymax]), title="J(w)"),
        color=alt.Color("kind:N",
            scale=alt.Scale(
                domain=["log-loss（凸）", "MSE（非凸）"],
                range=["#2563eb", "#dc2626"]),
            legend=alt.Legend(orient="top")),
    )
    # 标记局部极小
    _minima_rows = []
    if loss_kind.value in ("log", "both"):
        for i in log_minima:
            _minima_rows.append({"w": ws[i], "loss": log_curve[i], "kind": "log-loss（凸）"})
    if loss_kind.value in ("mse", "both"):
        for i in mse_minima:
            _minima_rows.append({"w": ws[i], "loss": mse_curve[i], "kind": "MSE（非凸）"})
    _minima_df = pd.DataFrame(_minima_rows) if _minima_rows else pd.DataFrame(
        {"w": [], "loss": [], "kind": []})
    minima_pts = alt.Chart(_minima_df).mark_point(
        size=180, shape="triangle-down", filled=True).encode(
        x="w:Q", y="loss:Q",
        color=alt.Color("kind:N", scale=alt.Scale(
            domain=["log-loss（凸）", "MSE（非凸）"],
            range=["#2563eb", "#dc2626"]), legend=None),
    )
    chart_landscape = (curves + minima_pts).properties(width=440, height=320)
    return (chart_landscape,)


@app.cell
def _(alt, np, p_pred, pd, y_true):
    # 视图 2：单样本惩罚曲线（揭示为什么 -log(p) 在 p→0 时趋向 ∞）
    _ps = np.linspace(0.001, 0.999, 200)
    if y_true.value == 1:
        _losses = -np.log(_ps)
        title_label = "y = 1：损失 = −log(p)"
        peak_color = "#dc2626"
        ok_color = "#16a34a"
    else:
        _losses = -np.log(1 - _ps)
        title_label = "y = 0：损失 = −log(1−p)"
        peak_color = "#dc2626"
        ok_color = "#16a34a"
    _df = pd.DataFrame({"p": _ps, "loss": _losses})
    line = alt.Chart(_df).mark_line(stroke="#7c3aed", strokeWidth=2.5).encode(
        x=alt.X("p:Q", scale=alt.Scale(domain=[0, 1]), title="预测概率 p = h_w(x)"),
        y=alt.Y("loss:Q", scale=alt.Scale(domain=[0, 5]),
                title="单样本 loss = −log(对的那一边)"),
    )
    # 当前点
    if y_true.value == 1:
        cur_loss = float(-np.log(p_pred.value))
    else:
        cur_loss = float(-np.log(1 - p_pred.value))
    cur = alt.Chart(pd.DataFrame({"p": [p_pred.value], "loss": [cur_loss]})).mark_circle(
        color="#9a3412", size=200).encode(
        x="p:Q", y="loss:Q",
        tooltip=[alt.Tooltip("p:Q", format=".2f"),
                 alt.Tooltip("loss:Q", format=".3f")])
    # 参考线：p=0.5 时 loss = ln(2) ≈ 0.693
    half = alt.Chart(pd.DataFrame({"y": [0.693]})).mark_rule(
        stroke="#94a3b8", strokeDash=[3, 3]).encode(y="y:Q")
    chart_penalty = (line + half + cur).properties(width=440, height=320)
    return chart_penalty, cur_loss, ok_color, peak_color, title_label


@app.cell
def _(chart_landscape, chart_penalty, mo):
    mo.hstack([chart_landscape, chart_penalty],
              widths=[1, 1], justify="space-around")
    return


@app.cell
def _(cur_loss, log_minima, mo, mse_minima, ok_color, p_pred,
      peak_color, title_label, y_true):
    # 状态卡：曲面统计 + 单样本解读
    _box = "padding:10px 14px;border-radius:6px;font-size:13px;line-height:1.7;"
    if y_true.value == 1:
        verdict = ("命中（loss 小）"
                   if p_pred.value >= 0.5 else "漏报（loss 大）")
        verdict_c = ok_color if p_pred.value >= 0.7 else peak_color
        intuition = (f"y=1，p={p_pred.value:.2f}：预测越靠 1 损失越小，"
                     f"靠 0 时 −log(p) 趋向 +∞")
    else:
        verdict = ("正确放过（loss 小）"
                   if p_pred.value < 0.5 else "误报（loss 大）")
        verdict_c = ok_color if p_pred.value <= 0.3 else peak_color
        intuition = (f"y=0，p={p_pred.value:.2f}：预测越靠 0 损失越小，"
                     f"靠 1 时 −log(1−p) 趋向 +∞")
    mo.md(f"""
<div style="display:flex;gap:10px;margin:6px 0;">
  <div style="flex:1;background:#f8fafc;border:1px solid #e2e8f0;{_box}">
    <strong>曲面统计（沿 w 扫描）</strong><br>
    log-loss 局部极小数：<strong style="color:#2563eb;">{len(log_minima)}</strong>（凸函数应为 1）<br>
    MSE 局部极小数：<strong style="color:#dc2626;">{len(mse_minima)}</strong>（&gt;1 = 非凸，梯度可能卡）
  </div>
  <div style="flex:1.2;background:#fff7ed;border-left:4px solid {verdict_c};{_box}">
    <strong>{title_label}</strong><br>
    当前损失 = <strong style="color:{verdict_c};">{cur_loss:.3f}</strong> · {verdict}<br>
    {intuition}
  </div>
</div>
""")
    return


@app.cell
def _(mo):
    mo.accordion({
        "为什么 MSE + sigmoid 非凸": mo.md(r"""
$J_\text{MSE}(w) = \frac{1}{m} \sum (\sigma(w x_i) - y_i)^2$

把 sigmoid 套进平方，二阶导数 $\partial^2 J / \partial w^2$ 在某些 $w$ 区间为负
（曲率反向），曲面像"两座山中间一个洼"。从不同初值出发可能停在不同的洼地。

**致命叠加**：sigmoid 在 $|z|>5$ 饱和区导数 $\sigma'(z) \to 0$，
预测严重错时（真 y=1, p=0.01）梯度也接近 0，参数走不动 → "梯度饱和 + 多局部最优"双杀。
"""),
        "为什么 log-loss + sigmoid 凸": mo.md(r"""
带入 $h_w(x) = \sigma(w^\top x + b)$ 化简：

$$J(w) = \frac{1}{m} \sum \left[\log(1 + e^{-y_i' z_i})\right]
\quad (y_i' = 2y_i - 1 \in \{-1, +1\})$$

每一项 $\log(1 + e^{-y' z})$ 是 $z$ 的凸函数（softplus），凸函数关于 $w$ 的线性组合
仍凸，求和后整体凸 → **全局只有一个最低点**。

工程后果：梯度下降从任意初值出发都收敛到同一个最优解 → 训练稳定。
"""),
        "−log(p) 为什么在 p→0 时趋向 ∞": mo.md(r"""
$\log p$ 是单调函数，$p \to 0^+$ 时 $\log p \to -\infty$，加负号变 $+\infty$。

**业务直觉**：模型说"我 99.99% 确定这是负例"（p=0.0001），结果它真是正例 → 损失爆炸。
模型对错误**极度自信**时被惩罚得最重——这正是 MLE 的核心精神：
**最离谱的预测 = 联合概率最小 = 负对数似然最大**。

工程坑：直接 `np.log(0)` 报警告 → 加 `1e-12` clip 或用 `scipy.special.xlogy`。
"""),
        "log-loss 是 MLE 的负对数化": mo.md(r"""
单样本的"分类正确概率" $P(y \mid x; w) = p^y (1-p)^{1-y}$（$y \in \{0,1\}$）。
n 样本似然 $L(w) = \prod p_i^{y_i}(1-p_i)^{1-y_i}$。

四步走：
1. **取 log**：连乘转连加，数值稳定（防 underflow）
2. **加负号**：max → min（"倒反天罡"）
3. **除 m**：批次归一化，与样本数无关
4. **得 log-loss**：$J(w) = -\frac{1}{m}\sum [y\log p + (1-y)\log(1-p)]$

所以 log-loss 不是拍脑袋设计的——它是 MLE 在伯努利分布下的等价形式。
"""),
    }, multiple=False)
    return


@app.cell
def _(mo):
    mo.md("""
## Grid 布局参考（16:9 · 36 行）

```
   0          12          24
0  ┌───── 标题 h=4 全宽 ─────┐
4  ├ kind ┬ y ┬ p   h=4 ──┤
8  ├──── 状态卡 h=5 全宽 ────┤
13 ├ 损失曲面(12) ┬ 惩罚曲线(12) ─┤  ← h=18
31 ├──── accordion h=5 全宽 ────┤
36
```
""")
    return


if __name__ == "__main__":
    app.run()
