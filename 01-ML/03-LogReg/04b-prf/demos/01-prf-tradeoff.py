"""
P/R/F1/Fβ 阈值 tradeoff · 拖阈值看精确率召回率怎么互相推

互动：
- 拖阈值 τ → P/R/F1 实时变 + 当前点在曲线上滑
- 调 β → 看 Fβ 偏向调节
跑：marimo edit 01-prf-tradeoff.py --port 2741 --headless --no-token
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/01-prf-tradeoff.grid.json",
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
    # P / R / F1 / Fβ · 阈值跷跷板

    $P = \dfrac{TP}{TP+FP}$（预测正里有多少真正） · $R = \dfrac{TP}{TP+FN}$（真正例抓回多少）

    $F_1 = \dfrac{2PR}{P+R}$（调和平均，短板裁判） · $F_\beta$（β 调偏向：β&gt;1 偏 R，β&lt;1 偏 P）

    拖 τ 看 P/R/F1 怎么互相推；β 滑块改变 Fβ 倾向。
    """)
    return


@app.cell
def _(mo):
    tau = mo.ui.slider(0.05, 0.95, value=0.5, step=0.02,
                       label="阈值 τ", show_value=True)
    beta = mo.ui.slider(0.25, 4.0, value=1.0, step=0.25,
                        label="Fβ 的 β（&gt;1 偏 R，&lt;1 偏 P）", show_value=True)
    show_avg_compare = mo.ui.switch(value=True, label="显示算术 vs 调和平均")
    mo.hstack([tau, beta, show_avg_compare],
              widths=[2, 2, 1.5], justify="space-around")
    return beta, show_avg_compare, tau


@app.cell
def _(np):
    # 单一来源：合成正/负预测概率
    rng = np.random.default_rng(17)
    _n_pos, _n_neg = 60, 80
    p_pos = np.clip(rng.beta(5, 2, _n_pos), 0.02, 0.98)
    p_neg = np.clip(rng.beta(2, 5, _n_neg), 0.02, 0.98)
    p_all = np.concatenate([p_pos, p_neg])
    y_all = np.array([1] * _n_pos + [0] * _n_neg)
    return p_all, y_all


@app.cell
def _(np, p_all, y_all):
    # 扫所有阈值算 P/R/F1 曲线
    _grid = np.linspace(0.02, 0.98, 97)
    _rows = []
    for _t in _grid:
        _yp = (p_all >= _t).astype(int)
        _tp = int(((y_all == 1) & (_yp == 1)).sum())
        _fp = int(((y_all == 0) & (_yp == 1)).sum())
        _fn = int(((y_all == 1) & (_yp == 0)).sum())
        _P = _tp / (_tp + _fp) if _tp + _fp > 0 else 1.0
        _R = _tp / (_tp + _fn) if _tp + _fn > 0 else 0.0
        _F1 = 2 * _P * _R / (_P + _R) if _P + _R > 0 else 0.0
        _rows.append({"τ": _t, "P": _P, "R": _R, "F1": _F1})
    sweep_df = __import__("pandas").DataFrame(_rows)
    return (sweep_df,)


@app.cell
def _(beta, p_all, tau, y_all):
    # 当前阈值的指标
    _yp = (p_all >= tau.value).astype(int)
    tp = int(((y_all == 1) & (_yp == 1)).sum())
    fp = int(((y_all == 0) & (_yp == 1)).sum())
    fn = int(((y_all == 1) & (_yp == 0)).sum())
    tn = int(((y_all == 0) & (_yp == 0)).sum())
    P_cur = tp / (tp + fp) if tp + fp > 0 else 1.0
    R_cur = tp / (tp + fn) if tp + fn > 0 else 0.0
    F1_cur = 2 * P_cur * R_cur / (P_cur + R_cur) if P_cur + R_cur > 0 else 0.0
    AM_cur = (P_cur + R_cur) / 2
    _b2 = beta.value ** 2
    Fb_cur = ((1 + _b2) * P_cur * R_cur / (_b2 * P_cur + R_cur)
              if _b2 * P_cur + R_cur > 0 else 0.0)
    return AM_cur, F1_cur, Fb_cur, P_cur, R_cur, fn, fp, tn, tp


@app.cell
def _(P_cur, R_cur, alt, pd, show_avg_compare, sweep_df, tau):
    # 视图 1：P/R/F1 vs τ 三曲线 + 当前点 + 阈值竖线
    _long = []
    for _, _r in sweep_df.iterrows():
        _long.append({"τ": _r["τ"], "metric": "P", "value": _r["P"]})
        _long.append({"τ": _r["τ"], "metric": "R", "value": _r["R"]})
        _long.append({"τ": _r["τ"], "metric": "F1", "value": _r["F1"]})
    _df = pd.DataFrame(_long)
    _palette = ["#2563eb", "#dc2626", "#7c3aed"]
    _lines = alt.Chart(_df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("τ:Q", scale=alt.Scale(domain=[0, 1]), title="阈值 τ"),
        y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1.05]), title="指标"),
        color=alt.Color("metric:N",
            scale=alt.Scale(domain=["P", "R", "F1"], range=_palette),
            legend=alt.Legend(title="指标", orient="top")),
    )
    _vl = alt.Chart(pd.DataFrame({"x": [tau.value]})).mark_rule(
        stroke="#0f172a", strokeWidth=2, strokeDash=[5, 4]).encode(x="x:Q")
    _cur = alt.Chart(pd.DataFrame({
        "τ": [tau.value, tau.value], "value": [P_cur, R_cur],
        "metric": ["P", "R"]
    })).mark_circle(size=180, stroke="white", strokeWidth=1.5).encode(
        x="τ:Q", y="value:Q",
        color=alt.Color("metric:N",
            scale=alt.Scale(domain=["P", "R", "F1"], range=_palette), legend=None))
    _layers = [_lines, _vl, _cur]
    if show_avg_compare.value:
        _am = sweep_df.assign(AM=(sweep_df["P"] + sweep_df["R"]) / 2)
        _am_line = alt.Chart(_am).mark_line(
            stroke="#94a3b8", strokeWidth=1.5, strokeDash=[3, 3]).encode(
            x="τ:Q", y="AM:Q")
        _layers.append(_am_line)
    chart_curves = alt.layer(*_layers).properties(width=420, height=320)
    return (chart_curves,)


@app.cell
def _(alt, beta, np, pd):
    # 视图 2：F1 vs Fβ 等高线对比（直觉：β 把"等 Fβ"曲线拉向 R 或 P）
    _grid_p = np.linspace(0.05, 1.0, 60)
    _grid_r = np.linspace(0.05, 1.0, 60)
    _xx, _yy = np.meshgrid(_grid_p, _grid_r)
    _f1 = 2 * _xx * _yy / (_xx + _yy + 1e-9)
    _b2 = beta.value ** 2
    _fb = (1 + _b2) * _xx * _yy / (_b2 * _xx + _yy + 1e-9)
    _df_fb = pd.DataFrame({
        "P": _xx.ravel(), "R": _yy.ravel(), "Fβ": _fb.ravel()
    })
    _heat = alt.Chart(_df_fb).mark_rect(opacity=0.85).encode(
        x=alt.X("P:Q", bin=alt.Bin(maxbins=60),
                scale=alt.Scale(domain=[0, 1]), title="Precision"),
        y=alt.Y("R:Q", bin=alt.Bin(maxbins=60),
                scale=alt.Scale(domain=[0, 1]), title="Recall"),
        color=alt.Color("Fβ:Q",
            scale=alt.Scale(scheme="viridis", domain=[0, 1]),
            legend=alt.Legend(title=f"Fβ (β={beta.value:.2f})")),
    )
    chart_heat = _heat.properties(width=320, height=320)
    return (chart_heat,)


@app.cell
def _(chart_curves, chart_heat, mo):
    mo.hstack([chart_curves, chart_heat],
              widths=[1.3, 1], justify="space-around")
    return


@app.cell
def _(AM_cur, F1_cur, Fb_cur, P_cur, R_cur, beta, fn, fp, mo, tau, tn, tp):
    # 状态卡：当前 P/R/F1 + 算术 vs 调和差距 + 业务建议
    _box = "padding:10px 14px;border-radius:6px;font-size:13px;line-height:1.7;"
    if beta.value > 1.5:
        bias_text = f"β={beta.value:.2f} 偏召回率（医疗筛查 / 反欺诈）"
        bias_c = "#dc2626"
    elif beta.value < 0.7:
        bias_text = f"β={beta.value:.2f} 偏精确率（推送 / 垃圾邮件）"
        bias_c = "#2563eb"
    else:
        bias_text = f"β={beta.value:.2f} ≈ F1 平衡"
        bias_c = "#7c3aed"
    diff = AM_cur - F1_cur
    if diff > 0.1:
        gap_note = ("⚠️ 算术 - 调和 = "
                    f"<strong>{diff:.2%}</strong> 大 → P 和 R 不平衡（短板被调和惩罚）")
        gap_c = "#dc2626"
    else:
        gap_note = ("算术 - 调和 = "
                    f"<strong>{diff:.2%}</strong>（P 和 R 接近）")
        gap_c = "#16a34a"
    mo.md(f"""
<div style="display:flex;gap:10px;margin:6px 0;">
  <div style="flex:1;background:#eff6ff;border-left:4px solid #2563eb;{_box}">
    <strong>τ = {tau.value:.2f} 时</strong><br>
    P = <strong style="color:#2563eb;">{P_cur:.1%}</strong>
    （{tp}/{tp+fp}）<br>
    R = <strong style="color:#dc2626;">{R_cur:.1%}</strong>
    （{tp}/{tp+fn}）<br>
    F1 = <strong style="color:#7c3aed;">{F1_cur:.1%}</strong>
  </div>
  <div style="flex:1;background:#f8fafc;border:1px solid #e2e8f0;{_box}">
    <strong>调和 vs 算术</strong><br>
    F1（调和） = <strong>{F1_cur:.2%}</strong><br>
    (P+R)/2（算术） = <strong>{AM_cur:.2%}</strong><br>
    <span style="color:{gap_c};">{gap_note}</span>
  </div>
  <div style="flex:1;background:#fff7ed;border-left:4px solid {bias_c};{_box}">
    <strong>Fβ = {Fb_cur:.1%}</strong><br>
    {bias_text}<br>
    错误：FN={fn} · FP={fp} · TN={tn}
  </div>
</div>
""")
    return


@app.cell
def _(mo):
    mo.accordion({
        "调和 vs 算术：为什么 F1 是短板裁判": mo.md(r"""
| 模型 | P | R | 算术 (P+R)/2 | 调和 F1 |
|---|---|---|---|---|
| 全抓 baseline | 60% | 100% | **80%** | 75% |
| 平衡 | 75% | 75% | 75% | 75% |
| 极端 | 1% | 99% | **50%** | ~2% |

算术平均把"全抓"打到 80% —— 但这模型烂得没法用。
**调和平均对小值敏感**：P 或 R 任何一个塌了，F1 跟着塌。

公式直觉：$F_1 = \dfrac{2PR}{P+R}$，分子是乘积，
任一项接近 0 整体接近 0；调和平均 = "倒数取平均再倒回去"。

口诀：**算术奖励均值，调和惩罚短板**。
"""),
        "Fβ 公式与 β 的几何意义": mo.md(r"""
$$F_\beta = (1 + \beta^2) \cdot \dfrac{PR}{\beta^2 P + R}$$

| β | 倾向 | 适用 |
|---|---|---|
| 0.5 | 偏 P | 推荐 / 垃圾邮件（误报代价大）|
| 1.0 | 平衡 | 默认 F1 |
| 2.0 | 偏 R | 癌症筛查 / 反欺诈（漏报代价大）|

记忆：**β 在分母 P 项上**（$\beta^2 P$），β 大 → P 弱时整体被拉得更狠 → 模型必须**保 R**。
所以 β > 1 偏召回。

直觉验证：把右图 β=4 → 等高线明显扁向"高 R"区域；β=0.25 → 等高线偏向"高 P"区域。
"""),
        "F1 不看 TN（公式里没有它）": mo.md(r"""
$F_1 = \dfrac{2 \cdot TP}{2 \cdot TP + FP + FN}$ —— **TN 完全不进公式**。

工程后果：
- 99% 负例的极不平衡场景，准确率被 TN 撑到 99%（看着没问题）
- F1 还是诚实反映正例那一侧的表现
- **不平衡数据偏爱 F1** 的根本原因

但 F1 也有盲区：用比例混合掩盖了 FP 和 FN 绝对量级的差距。
真要算"业务总损失"，建议用 cost-weighted error：
`cost = FN * w_fn + FP * w_fp`（见 04a 状态卡的"业务代价"列）。
"""),
        "P/R 曲线随 τ 变化的方向": mo.md(r"""
**τ ↑（更严，少判正例）**：
- TP ↓, FP ↓ 更多 → P 通常 ↑
- TP ↓, FN ↑ → R ↓

**τ ↓（更松，多判正例）**：
- TP ↑, FP ↑ → P 通常 ↓
- TP ↑ → R ↑

把上面 demo τ 拖到 0.95：R 接近 0（几乎不抓正例），P 接近 1（少数判正例都对）→ F1 接近 0。
拖到 0.05：R 接近 1（全抓），P ≈ 正例占比（瞎抓），F1 一般偏低。
**P 和 R 是天然跷跷板**——这正是 ROC/AUC 要在所有阈值下度量"曲线整体能力"的动机（见 04c）。
"""),
    }, multiple=False)
    return


@app.cell
def _(mo):
    mo.md("""
## Grid 布局参考（16:9 · 36 行）

```
   0           14          24
0  ┌───── 标题 h=4 全宽 ─────┐
4  ├ τ ┬ β ┬ avg-cmp h=4 ──┤
8  ├──── 状态卡 h=5 全宽 ────┤
13 ├ P/R/F1 曲线(14) ┬ Fβ 等高(10) ─┤  ← h=18
31 ├──── accordion h=5 全宽 ────┤
36
```
""")
    return


if __name__ == "__main__":
    app.run()
