"""
sigmoid 函数交互 demo · 把任意实数压到 (0, 1) 当概率

互动：拖 z 滑块 / 选预设 → 实时看 S 形曲线 + 概率条 + 导数曲线
跑：marimo edit 01-sigmoid.py --port 2731 --headless --no-token
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")

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
    # Sigmoid 函数 · 把任意实数压到 (0, 1) 当概率

    $$\sigma(z) = \frac{1}{1 + e^{-z}}$$

    线性回归输出 $z = w^\top x + b \in (-\infty, +\infty)$，分类需要概率 $\in (0, 1)$。
    **Sigmoid 就是这道桥**。拖 z 滑块或选预设 → 三视图实时联动。
    """)
    return

@app.cell
def _(mo):
    z = mo.ui.slider(-10, 10, value=0, step=0.1, label="z（线性输出）")
    preset = mo.ui.dropdown(
        options={"自由（用滑块）": "free", "中点 z=0": "0",
                 "z=2 → ≈0.88": "2", "z=-2 → ≈0.12": "-2",
                 "饱和 z=10": "10", "饱和 z=-10": "-10"},
        value="自由（用滑块）", label="预设")
    mo.hstack([z, preset], widths=[2, 1], justify="space-around")
    return preset, z

@app.cell
def _(np, preset, z):
    # 单一来源：z_use 由后续所有视图复用
    if preset.value == "free" or preset.value is None:
        z_use = float(z.value)
        mode = "手动"
    else:
        z_use = float(preset.value)
        mode = "预设"
    sigma_z = float(1.0 / (1.0 + np.exp(-z_use)))
    dsigma_z = float(sigma_z * (1.0 - sigma_z))
    return dsigma_z, mode, sigma_z, z_use

@app.cell
def _(alt, np, pd, sigma_z, z_use):
    # 视图 1：S 形主曲线（200 网格点 + 当前红点 + 0.5/0 参考线）
    _xs = np.linspace(-10, 10, 200)
    _df = pd.DataFrame({"z": _xs, "sigma": 1.0 / (1.0 + np.exp(-_xs))})
    _curve = alt.Chart(_df).mark_line(stroke="#2563eb", strokeWidth=2.5).encode(
        x=alt.X("z:Q", scale=alt.Scale(domain=[-10, 10]), title="z"),
        y=alt.Y("sigma:Q", scale=alt.Scale(domain=[-0.05, 1.05]), title="σ(z)"),
    )
    _hl = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
        stroke="#94a3b8", strokeDash=[4, 3]).encode(y="y:Q")
    _vl = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(
        stroke="#94a3b8", strokeDash=[4, 3]).encode(x="x:Q")
    _pt = alt.Chart(pd.DataFrame({"z": [z_use], "sigma": [sigma_z]})).mark_circle(
        color="red", size=200, opacity=0.95).encode(
        x="z:Q", y="sigma:Q",
        tooltip=[alt.Tooltip("z:Q", format=".2f"), alt.Tooltip("sigma:Q", format=".4f")])
    chart_sigmoid = (_curve + _hl + _vl + _pt).properties(
        width=320, height=260, title="σ(z) = 1 / (1 + exp(-z))")
    return (chart_sigmoid,)

@app.cell
def _(alt, pd, sigma_z, z_use):
    # 视图 2：概率条（按值梯度上色 · 红→灰→绿）
    _bg = alt.Chart(pd.DataFrame({"label": ["σ(z)"], "value": [1.0]})).mark_bar(
        color="#e5e7eb").encode(
        x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), title="概率 σ(z) ∈ [0, 1]"),
        y=alt.Y("label:N", title=None))
    _bar = alt.Chart(pd.DataFrame({"label": ["σ(z)"], "value": [sigma_z]})).mark_bar().encode(
        x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("label:N"),
        color=alt.Color("value:Q",
            scale=alt.Scale(domain=[0, 0.5, 1], range=["#dc2626", "#9ca3af", "#16a34a"]),
            legend=None))
    _txt = alt.Chart(pd.DataFrame({
        "label": ["σ(z)"], "text": [f"σ({z_use:.2f}) = {sigma_z:.4f}"]
    })).mark_text(align="left", dx=8, fontSize=14, fontWeight="bold",
                  color="#0f172a").encode(x=alt.value(8), y="label:N", text="text:N")
    chart_bar = (_bg + _bar + _txt).properties(
        width=320, height=120, title="当前概率（红=负类强 / 绿=正类强）")
    return (chart_bar,)

@app.cell
def _(alt, dsigma_z, np, pd, z_use):
    # 视图 3：导数曲线（峰值 0.25 + 当前红点）
    _xs = np.linspace(-10, 10, 200)
    _s = 1.0 / (1.0 + np.exp(-_xs))
    _df = pd.DataFrame({"z": _xs, "deriv": _s * (1.0 - _s)})
    _curve = alt.Chart(_df).mark_line(stroke="#7c3aed", strokeWidth=2.5).encode(
        x=alt.X("z:Q", scale=alt.Scale(domain=[-10, 10]), title="z"),
        y=alt.Y("deriv:Q", scale=alt.Scale(domain=[0, 0.3]), title="σ'(z)"))
    _ml = alt.Chart(pd.DataFrame({"y": [0.25]})).mark_rule(
        stroke="#94a3b8", strokeDash=[4, 3]).encode(y="y:Q")
    _pt = alt.Chart(pd.DataFrame({"z": [z_use], "deriv": [dsigma_z]})).mark_circle(
        color="red", size=200, opacity=0.95).encode(
        x="z:Q", y="deriv:Q",
        tooltip=[alt.Tooltip("z:Q", format=".2f"), alt.Tooltip("deriv:Q", format=".5f")])
    chart_deriv = (_curve + _ml + _pt).properties(
        width=320, height=260, title="σ'(z) = σ(z)(1 - σ(z))")
    return (chart_deriv,)

@app.cell
def _(chart_bar, chart_deriv, chart_sigmoid, mo):
    # 三视图横向布局
    mo.hstack([chart_sigmoid, chart_bar, chart_deriv],
              widths=[1, 1, 1], justify="space-around")
    return

@app.cell
def _(dsigma_z, mo, mode, sigma_z, z_use):
    # 数值面板 + 解读三联
    pred = "正类（喜欢/点击/患病…）" if sigma_z >= 0.5 else "负类"
    pred_c, pred_bg = (("#16a34a", "#e8f5e9") if sigma_z >= 0.5
                       else ("#dc2626", "#ffebee"))
    if abs(z_use) < 1.5:
        gn = f"z 接近 0 → 导数 <code>{dsigma_z:.5f}</code> **接近峰值 0.25**，梯度大、学习快"
        gc = "#16a34a"
    elif abs(z_use) > 5:
        gn = (f"|z| > 5 → 导数 <code>{dsigma_z:.5f}</code> **趋近 0**，"
              "**梯度消失**（深网络叠多层 sigmoid 的元凶）")
        gc = "#dc2626"
    else:
        gn = (f"|z|=<code>{abs(z_use):.2f}</code> 中间区间 → 导数 "
              f"<code>{dsigma_z:.5f}</code>，可用但不在峰值")
        gc = "#eab308"
    _box = "padding:10px 14px;border-radius:6px;font-size:14px;line-height:1.8;"
    mo.md(f"""
<div style="display:flex;gap:12px;margin:8px 0;">
  <div style="flex:1;background:#f8fafc;border:1px solid #e2e8f0;{_box}">
    <strong>当前数值</strong>（{mode}）<br>z = <code>{z_use:.2f}</code><br>
    σ(z) = <strong style="color:#2563eb;">{sigma_z:.4f}</strong><br>
    σ'(z) = <strong style="color:#7c3aed;">{dsigma_z:.5f}</strong></div>
  <div style="flex:1;background:{pred_bg};border-left:4px solid {pred_c};{_box}">
    <strong style="color:{pred_c};">分类决策</strong><br>
    阈值 0.5：σ = {sigma_z:.4f} {'≥' if sigma_z >= 0.5 else '<'} 0.5<br>
    → 预测 <strong style="color:{pred_c};">{pred}</strong></div>
  <div style="flex:1.3;background:#fff7ed;border-left:4px solid {gc};{_box}">
    <strong style="color:#9a3412;">梯度解读</strong><br>{gn}</div>
</div>
""")
    return

@app.cell
def _(mo):
    mo.accordion({
        "为什么是 sigmoid？(对数几率推导)": mo.md(r"""
**几率 (odds)** = $\dfrac{p}{1-p} \in (0, +\infty)$；
**对数几率 (log-odds)** = $\log \dfrac{p}{1-p} \in (-\infty, +\infty)$。

逻辑回归假设 log-odds 线性：$\log \dfrac{p}{1-p} = w^\top x + b = z$，
反解 $p = \dfrac{1}{1 + e^{-z}} = \sigma(z)$。**sigmoid 不是拍脑袋，是数学必然**。
"""),
        "导数 σ'=σ(1-σ) 为什么这么漂亮": mo.md(r"""
$\sigma'(z) = \dfrac{e^{-z}}{(1+e^{-z})^2} = \sigma(z)(1-\sigma(z))$。

**工程价值**：反向传播时不用重算 exp，前向已得 $\sigma(z)$ 直接乘 $(1-\sigma(z))$。
"""),
        "梯度消失：深网络弃用 sigmoid": mo.md(r"""
导数最大值 **0.25**（z=0），每过一层至少缩 1/4，10 层后 $0.25^{10} \approx 10^{-6}$，**底层学不到**。
|z|>5 饱和区导数 < 0.007，输入再变化输出几乎不动。

**现代解法**：ReLU / BatchNorm / 残差连接。**逻辑回归只 1 层**，sigmoid 没问题，仍是分类基石。
"""),
        "数值稳定：避免 exp 溢出": mo.md(r"""
朴素 $1/(1+e^{-z})$ 在 z=-1000 时 $e^{1000}$ **溢出**。稳定写法分两支：

```python
def sigmoid_stable(z):
    if z >= 0: return 1.0 / (1.0 + np.exp(-z))   # z 大：exp(-z) 小，安全
    ez = np.exp(z); return ez / (1.0 + ez)        # z 小：exp(z)  小，安全
```

PyTorch / sklearn 内部都做了类似处理，**自己实现时记得**。
"""),
    }, multiple=False)
    return

if __name__ == "__main__":
    app.run()
