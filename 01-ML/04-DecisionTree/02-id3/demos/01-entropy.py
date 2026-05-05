"""
信息熵交互 demo · H = -Σ p log₂ p

主隐喻：分布越均匀，熵越大；越纯净，熵 → 0。
拖 p₁ 滑块 / 选预设 → 柱状图 + 钟形曲线 + H 数值实时联动。
末尾教材三例（k=3 类）静态对比，照应 ID3 学习目标。

跑：
  marimo edit 01-entropy.py --port 2718                              # 调布局
  marimo run  01-entropy.py --port 2760 --headless --no-token        # 录屏
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/01-entropy.grid.json",
    css_file="marimo.css",
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
    # 信息熵 · 不确定度的度量

    $$H = -\sum_{i=1}^{k} p_i \log_2 p_i$$

    **直觉**：所有概率均匀 → 最混乱、$H$ 最大；某类占比 100% → 完全确定、$H = 0$。
    拖 $p_1$ 滑块或选预设，看二分类下分布与熵的关系。
    """)
    return


@app.cell
def _(mo):
    p1 = mo.ui.slider(0.0, 1.0, value=0.5, step=0.01,
                      label="p₁（正类占比）", show_value=True)
    preset = mo.ui.dropdown(
        options={
            "自由（用滑块）": "free",
            "均匀 p=0.50 → H=1.00": "0.5",
            "略偏 p=0.30 → H≈0.88": "0.3",
            "重偏 p=0.10 → H≈0.47": "0.1",
            "纯净 p=0.00 → H=0": "0.0",
        },
        value="自由（用滑块）", label="预设",
    )
    mo.hstack([p1, preset], widths=[2, 1], justify="space-around")
    return p1, preset


@app.cell
def _(np, p1, preset):
    # 单一来源：所有视图复用 p1_use / p2_use / H_use
    if preset.value == "free" or preset.value is None:
        p1_use = float(p1.value)
        mode = "手动"
    else:
        p1_use = float(preset.value)
        mode = "预设"
    p2_use = 1.0 - p1_use

    def h_binary(p):
        # 0·log0 = 0 约定；clip 到 (eps, 1-eps) 避免 log(0) 警告
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    H_use = float(h_binary(p1_use)) if 0.0 < p1_use < 1.0 else 0.0
    return H_use, h_binary, mode, p1_use, p2_use


@app.cell
def _(H_use, mo, mode, p1_use, p2_use):
    # 状态卡：H 数值大字 + 极端解读
    if H_use < 0.05:
        verdict = "<strong>完全纯净</strong> · 一类占满，毫无不确定性"
        vc, vbg = "#16a34a", "#e8f5e9"
    elif H_use > 0.97:
        verdict = "<strong>最混乱</strong> · 两类对半，最难预测"
        vc, vbg = "#dc2626", "#ffebee"
    else:
        verdict = (f"<strong>部分混乱</strong> · "
                   f"p₁={p1_use:.2f} / p₂={p2_use:.2f}，介于纯净与均匀之间")
        vc, vbg = "#ca8a04", "#fef9c3"

    mo.md(f"""
<div style="display:flex;gap:16px;align-items:center;
            padding:14px 20px;border-radius:8px;
            background:{vbg};border-left:6px solid {vc};">
  <div style="flex:0 0 auto;">
    <div style="font-size:11px;color:#64748b;letter-spacing:1px;">
      H（{mode}）
    </div>
    <div style="font-size:42px;font-weight:700;line-height:1;color:{vc};
                font-family:var(--marimo-monospace-font);">
      {H_use:.3f}
    </div>
    <div style="font-size:11px;color:#64748b;">bits</div>
  </div>
  <div style="flex:1;font-size:14px;line-height:1.6;color:#0f172a;">
    {verdict}<br>
    <span style="color:#64748b;font-size:12px;">
      二分类下 H ∈ [0, 1]；最大值 1 在 p=0.5 处取到
    </span>
  </div>
</div>
""")
    return


@app.cell
def _(alt, p1_use, p2_use, pd):
    # 视图 1：当前分布柱状图（两类）
    _df = pd.DataFrame({
        "class": ["类 1", "类 2"],
        "p": [p1_use, p2_use],
    })
    _bars = alt.Chart(_df).mark_bar().encode(
        x=alt.X("class:N", title=None,
                axis=alt.Axis(labelFontSize=13, labelFontWeight="bold",
                              labelAngle=0)),
        y=alt.Y("p:Q", title="占比",
                scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("class:N",
                scale=alt.Scale(range=["#2563eb", "#7c3aed"]),
                legend=None),
    )
    _txt = alt.Chart(_df).mark_text(
        dy=-8, fontSize=14, fontWeight="bold", color="#0f172a",
    ).encode(
        x="class:N", y="p:Q",
        text=alt.Text("p:Q", format=".2f"),
    )
    chart_dist = (_bars + _txt).properties(
        width="container", height=160,
    )
    chart_dist
    return (chart_dist,)


@app.cell
def _(H_use, alt, h_binary, np, p1_use, pd):
    # 视图 2：H(p) 钟形曲线 + 当前红点（与视图 1 同源 p1_use）
    _xs = np.linspace(0.001, 0.999, 200)
    _df = pd.DataFrame({"p": _xs, "H": h_binary(_xs)})
    _curve = alt.Chart(_df).mark_line(
        stroke="#0ea5e9", strokeWidth=2.5,
    ).encode(
        x=alt.X("p:Q", title="p₁",
                scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("H:Q", title="H（bits）",
                scale=alt.Scale(domain=[0, 1.05])),
    )
    _max_rule = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(
        stroke="#94a3b8", strokeDash=[4, 3],
    ).encode(y="y:Q")
    _pt = alt.Chart(pd.DataFrame({"p": [p1_use], "H": [H_use]})).mark_circle(
        color="#dc2626", size=240, opacity=0.95,
    ).encode(
        x="p:Q", y="H:Q",
        tooltip=[alt.Tooltip("p:Q", format=".2f"),
                 alt.Tooltip("H:Q", format=".4f")],
    )
    chart_curve = (_curve + _max_rule + _pt).properties(
        width="container", height=160,
    )
    chart_curve
    return (chart_curve,)


@app.cell
def _(alt, np, pd):
    # 视图 3：教材三例（3 类静态对比 · k=3）
    eps = 1e-12

    def _h_multi(probs):
        probs = np.array(probs, dtype=float)
        probs = np.clip(probs, eps, 1.0)
        return float(-np.sum(probs * np.log2(probs)))

    _examples = [
        ("均匀 (1/3,1/3,1/3)", [1/3, 1/3, 1/3]),
        ("偏斜 (1/10,2/10,7/10)", [0.1, 0.2, 0.7]),
        ("纯净 (1,0,0)", [1.0, 0.0, 0.0]),
    ]
    _rows = []
    for name, ps in _examples:
        h = _h_multi(ps)
        for i, p in enumerate(ps):
            _rows.append({"case": f"{name}\nH={h:.3f}",
                          "class": f"c{i+1}", "p": p})
    _df = pd.DataFrame(_rows)
    _bars = alt.Chart(_df).mark_bar().encode(
        x=alt.X("class:N", title=None,
                axis=alt.Axis(labelFontSize=11, labelAngle=0)),
        y=alt.Y("p:Q", title="占比", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("class:N",
                scale=alt.Scale(range=["#2563eb", "#7c3aed", "#0ea5e9"]),
                legend=None),
        column=alt.Column("case:N", title=None,
                header=alt.Header(labelFontSize=12, labelFontWeight="bold")),
    ).properties(width=140, height=130)
    chart_examples = _bars
    return (chart_examples,)


@app.cell
def _(chart_examples):
    # 教材 3 类例子（独立 cell · grid 单独占一行）
    chart_examples
    return


@app.cell
def _(mo):
    mo.accordion({
        "为什么用 log₂？为什么前面要带负号？": mo.md(r"""
**单事件信息量** $h(x) = -\log_2 p(x)$：概率越小、越"惊讶"、信息量越大。
- $\log_2$ 用 2 为底 → 单位是 **bit**（一次二选一所需的信息量）
- 概率 $p \in (0,1]$ 时 $\log_2 p \le 0$，加负号让信息量为正
- 用 $\ln$ 单位变 nat、$\log_{10}$ 单位变 dit；本质同一个量

**信息熵 = 信息量的期望**：$H = E[-\log_2 p] = -\sum p_i \log_2 p_i$。
"""),
        "为什么 p=0.5 是最大值？": mo.md(r"""
对 $H(p) = -p\log_2 p - (1-p)\log_2(1-p)$ 求导：
$$\frac{dH}{dp} = -\log_2 p + \log_2(1-p) = \log_2 \frac{1-p}{p}$$

令 $H' = 0$ → $\frac{1-p}{p} = 1$ → $p = 0.5$，二阶导小于 0 故为极大。

**直觉**：硬币最公平时（50/50）猜结果最难，平均要 1 bit 信息才能确定；若 $p=0.99$，几乎必正面，信息量已经接近 0 bit。
"""),
        "0·log₂(0) 怎么处理？": mo.md(r"""
数学上 $0 \cdot \log_2 0$ 是 $0 \cdot (-\infty)$，**约定为 0**。
理由：用洛必达 $\lim_{p\to 0^+} p\log_2 p = 0$。

**代码实现**：直接计算会触发 `RuntimeWarning: divide by zero`。两种处理：
```python
# 法 1：clip 到极小值
p = np.clip(p, 1e-12, 1.0)
# 法 2：np.where 替换
H = -np.where(p > 0, p * np.log2(p), 0).sum()
```
sklearn / scipy 内部都做了类似处理。
"""),
        "k 类时熵的最大值是多少？": mo.md(r"""
拉格朗日乘子法可证：$k$ 类下 $H$ 在均匀分布 $p_i = 1/k$ 时取最大：
$$H_{\max} = \log_2 k$$

- 2 类 → $\log_2 2 = 1$ bit
- 3 类 → $\log_2 3 \approx 1.585$ bits（即教材首例）
- 10 类 → $\log_2 10 \approx 3.32$ bits

**ID3 后续用法**：每个特征切分后子集熵的加权平均叫"经验条件熵"，原熵减去它叫"信息增益"——选增益最大的特征当分裂点。
"""),
    }, multiple=False)
    return


@app.cell
def _(mo):
    # 📐 Grid 布局参考（开发用 · 录屏隐藏）
    mo.md("""
## 📐 Grid 布局（16:9 · 1280×720 · 36 行）

```
   0           12          24
 0 ┌── 标题+公式+直觉 h=8 全宽 ─────┐
 8 ├── 控件 hstack h=3 ─────────────┤
11 ├── H 数值卡 h=5 全宽 ───────────┤
16 ├ 柱状图(12) ┬ 钟形曲线(12) h=12 ┤
28 ├── 教材 3 例对比 h=8 全宽 ──────┤
36
```

cell 顺序对应（含 null 位）：
imports / 标题 / 控件 / 计算 / H 卡 / chart_dist / chart_curve /
chart_examples / 教材渲染 cell / accordion / 本 cell。
计算 / chart_examples / accordion / ASCII 在 grid 中 position=null。
""")
    return


if __name__ == "__main__":
    app.run()
