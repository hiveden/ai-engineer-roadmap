"""
ROC + AUC · 阈值扫过的曲线 + 曲线下面积

互动：
- 拖阈值 τ → ROC 上当前点滑动 + 混淆矩阵实时更新
- 切模型质量（完美 / 强 / 弱 / 随机）看 AUC 怎么变
跑：marimo edit 01-roc-auc.py --port 2742 --headless --no-token
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/01-roc-auc.grid.json",
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
    # ROC 曲线 + AUC · 所有阈值下的整体能力

    **ROC** = 阈值 τ 从 1 滑到 0，每个 τ 算 (FPR, TPR) → 画轨迹

    **TPR** = $\frac{TP}{TP+FN}$ = R 召回率（真正例抓回多少）

    **FPR** = $\frac{FP}{FP+TN}$ = 假正例率（真负例里被误报多少）

    **AUC** = ROC 曲线下面积 ∈ [0.5, 1]，**1 = 完美 / 0.5 = 随机**
    """)
    return


@app.cell
def _(mo):
    quality = mo.ui.dropdown(
        options={"完美分类（AUC≈1）": "perfect",
                 "强模型（AUC≈0.9）": "strong",
                 "弱模型（AUC≈0.7）": "weak",
                 "随机猜（AUC≈0.5）": "random",
                 "比随机更差（AUC<0.5）": "worse"},
        value="强模型（AUC≈0.9）", label="模型质量")
    tau = mo.ui.slider(0.02, 0.98, value=0.5, step=0.02,
                       label="阈值 τ", show_value=True)
    show_grid = mo.ui.switch(value=True, label="显示 AUC 阴影")
    mo.hstack([quality, tau, show_grid],
              widths=[2, 2, 1.2], justify="space-around")
    return quality, show_grid, tau


@app.cell
def _(np, quality):
    # 单一来源：根据 quality 合成预测概率
    quality_kind = quality.value  # dropdown.value 返回 dict 的 value
    rng = np.random.default_rng(23)
    _n_pos, _n_neg = 80, 80
    if quality_kind == "perfect":
        p_pos = np.clip(rng.beta(15, 1, _n_pos), 0.5, 0.99)
        p_neg = np.clip(rng.beta(1, 15, _n_neg), 0.01, 0.5)
    elif quality_kind == "strong":
        p_pos = np.clip(rng.beta(6, 2, _n_pos), 0.05, 0.98)
        p_neg = np.clip(rng.beta(2, 6, _n_neg), 0.02, 0.95)
    elif quality_kind == "weak":
        p_pos = np.clip(rng.beta(3, 2.5, _n_pos), 0.05, 0.95)
        p_neg = np.clip(rng.beta(2.5, 3, _n_neg), 0.05, 0.95)
    elif quality_kind == "random":
        p_pos = rng.uniform(0.05, 0.95, _n_pos)
        p_neg = rng.uniform(0.05, 0.95, _n_neg)
    else:  # worse
        p_pos = np.clip(rng.beta(2, 6, _n_pos), 0.02, 0.95)  # 反过来
        p_neg = np.clip(rng.beta(6, 2, _n_neg), 0.05, 0.98)
    p_all = np.concatenate([p_pos, p_neg])
    y_all = np.array([1] * _n_pos + [0] * _n_neg)
    return p_all, quality_kind, y_all


@app.cell
def _(np, p_all, y_all):
    # 扫所有阈值生成 ROC 点（含端点 (0,0) 和 (1,1)）
    _ts = np.concatenate([[1.01], np.unique(p_all)[::-1], [-0.01]])
    _rows = []
    for _t in _ts:
        _yp = (p_all >= _t).astype(int)
        _tp = ((y_all == 1) & (_yp == 1)).sum()
        _fp = ((y_all == 0) & (_yp == 1)).sum()
        _fn = ((y_all == 1) & (_yp == 0)).sum()
        _tn = ((y_all == 0) & (_yp == 0)).sum()
        _tpr = _tp / (_tp + _fn) if _tp + _fn > 0 else 0.0
        _fpr = _fp / (_fp + _tn) if _fp + _tn > 0 else 0.0
        _rows.append({"τ": float(_t), "FPR": _fpr, "TPR": _tpr,
                      "TP": int(_tp), "FP": int(_fp),
                      "FN": int(_fn), "TN": int(_tn)})
    _roc_df = __import__("pandas").DataFrame(_rows)
    sorted_df = _roc_df.sort_values(["FPR", "TPR"]).reset_index(drop=True)
    auc = float(np.trapezoid(sorted_df["TPR"].values, sorted_df["FPR"].values))
    return auc, sorted_df


@app.cell
def _(p_all, tau, y_all):
    # 当前阈值的混淆 4 格 + (FPR, TPR)
    _yp = (p_all >= tau.value).astype(int)
    tp = int(((y_all == 1) & (_yp == 1)).sum())
    fp = int(((y_all == 0) & (_yp == 1)).sum())
    fn = int(((y_all == 1) & (_yp == 0)).sum())
    tn = int(((y_all == 0) & (_yp == 0)).sum())
    TPR_cur = tp / (tp + fn) if tp + fn > 0 else 0.0
    FPR_cur = fp / (fp + tn) if fp + tn > 0 else 0.0
    return FPR_cur, TPR_cur, fn, fp, tn, tp


@app.cell
def _(FPR_cur, TPR_cur, alt, pd, show_grid, sorted_df):
    # 视图 1：ROC 曲线 + 当前点 + 对角线 + AUC 阴影
    _layers = []
    if show_grid.value:
        _area = alt.Chart(sorted_df.copy()).mark_area(
            color="#3b82f6", opacity=0.18).encode(
            x=alt.X("FPR:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("TPR:Q", scale=alt.Scale(domain=[0, 1])),
        )
        _layers.append(_area)
    _diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        stroke="#94a3b8", strokeDash=[5, 5]).encode(x="x:Q", y="y:Q")
    _layers.append(_diag)
    _line = alt.Chart(sorted_df).mark_line(
        stroke="#2563eb", strokeWidth=2.5, interpolate="step-after").encode(
        x=alt.X("FPR:Q", scale=alt.Scale(domain=[0, 1]),
                title="FPR = FP / (FP+TN)"),
        y=alt.Y("TPR:Q", scale=alt.Scale(domain=[0, 1]),
                title="TPR = TP / (TP+FN) = Recall"),
    )
    _layers.append(_line)
    _cur = alt.Chart(pd.DataFrame({
        "FPR": [FPR_cur], "TPR": [TPR_cur]
    })).mark_circle(color="#dc2626", size=240, stroke="white",
                    strokeWidth=2).encode(x="FPR:Q", y="TPR:Q")
    _layers.append(_cur)
    chart_roc = alt.layer(*_layers).properties(width=380, height=380)
    return (chart_roc,)


@app.cell
def _(alt, np, p_all, pd, tau, y_all):
    # 视图 2：预测概率分布（正负叠加直方图）+ 阈值竖线
    _df = pd.DataFrame({
        "p": p_all,
        "类": ["真实正例" if y == 1 else "真实负例" for y in y_all]
    })
    _hist = alt.Chart(_df).mark_bar(opacity=0.65).encode(
        x=alt.X("p:Q", bin=alt.Bin(maxbins=30),
                scale=alt.Scale(domain=[0, 1]),
                title="预测概率 P(y=1)"),
        y=alt.Y("count():Q", stack=None, title="样本数"),
        color=alt.Color("类:N",
            scale=alt.Scale(domain=["真实正例", "真实负例"],
                            range=["#16a34a", "#dc2626"]),
            legend=alt.Legend(orient="top")),
    )
    _vl = alt.Chart(pd.DataFrame({"x": [tau.value]})).mark_rule(
        stroke="#0f172a", strokeWidth=2.5, strokeDash=[6, 4]).encode(x="x:Q")
    chart_dist = (_hist + _vl).properties(width=420, height=380)
    return (chart_dist,)


@app.cell
def _(chart_dist, chart_roc, mo):
    mo.hstack([chart_roc, chart_dist],
              widths=[1, 1.1], justify="space-around")
    return


@app.cell
def _(FPR_cur, TPR_cur, auc, fn, fp, mo, quality_kind, tau, tn, tp):
    # 状态卡：AUC + 当前点 + 模型质量评级
    _box = "padding:10px 14px;border-radius:6px;font-size:13px;line-height:1.7;"
    if auc >= 0.95:
        rating = "卓越"
        rate_c = "#16a34a"
    elif auc >= 0.85:
        rating = "强"
        rate_c = "#16a34a"
    elif auc >= 0.7:
        rating = "可用"
        rate_c = "#eab308"
    elif auc >= 0.55:
        rating = "弱"
        rate_c = "#ea580c"
    elif auc >= 0.45:
        rating = "随机水平（无信息）"
        rate_c = "#dc2626"
    else:
        rating = "比随机还差（标签可能颠倒）"
        rate_c = "#dc2626"
    quality_label = {"perfect": "完美", "strong": "强模型",
                     "weak": "弱模型", "random": "随机猜",
                     "worse": "比随机更差"}[quality_kind]
    mo.md(f"""
<div style="display:flex;gap:10px;margin:6px 0;">
  <div style="flex:1.3;background:#eff6ff;border-left:4px solid {rate_c};{_box}">
    <strong>AUC = <span style="color:{rate_c};font-size:18px;">{auc:.3f}</span></strong>
    （{rating}）<br>
    场景：<code>{quality_label}</code><br>
    <em>AUC 不依赖阈值——它度量"任取一对正负样本，模型给正例打分更高的概率"</em>
  </div>
  <div style="flex:1;background:#f8fafc;border:1px solid #e2e8f0;{_box}">
    <strong>τ = {tau.value:.2f} 时的 ROC 点</strong><br>
    TPR = <strong style="color:#2563eb;">{TPR_cur:.2%}</strong>
    （召回 = TP/(TP+FN)）<br>
    FPR = <strong style="color:#dc2626;">{FPR_cur:.2%}</strong>
    （误报率 = FP/(FP+TN)）
  </div>
  <div style="flex:1;background:#fef2f2;border-left:4px solid #dc2626;{_box}">
    <strong>当前混淆</strong><br>
    TP={tp} · FN={fn}<br>
    FP={fp} · TN={tn}
  </div>
</div>
""")
    return


@app.cell
def _(mo):
    mo.accordion({
        "AUC 的概率解释（最重要的一句话）": mo.md(r"""
**AUC = 任取一个正例和一个负例，模型给正例打的分**
（`predict_proba(...)[:, 1]`）**比给负例高的概率**。

数学：$\text{AUC} = P(\text{score}(x_+) > \text{score}(x_-))$

| AUC | 含义 |
|---|---|
| 1.0 | 所有正例分数 > 所有负例（完全分开）|
| 0.9 | 90% 的对子里正例分数更高 |
| 0.5 | 一半概率 → 模型相当于扔硬币 |
| < 0.5 | 比随机还差 → 标签可能反了，把分数取反就好 |

**性质**：AUC **不依赖阈值**——单一阈值只是 ROC 上一个点，AUC 度量整条曲线。
"""),
        "为什么 ROC 用 FPR 而不是 P (precision)": mo.md(r"""
**FPR = FP / (FP+TN)** 分母是真负例总数；**P = TP / (TP+FP)** 分母是预测正例总数。

类别不平衡时（比如正例只占 1%）：
- **ROC** 横纵都按"真实类总数"归一 → 曲线**不随类别比例变形**
- **P-R 曲线** 横纵分母不同 → 同样模型在不同正负比上的 P-R 曲线**形状会变**

工程实践：
- **样本平衡**或对负例敏感 → 用 ROC + AUC
- **极不平衡 + 关心正例**（欺诈、罕见病）→ 用 PR 曲线 + AP（average precision）
- 二者经常一起报，sklearn 都有：`roc_auc_score` / `average_precision_score`
"""),
        "ROC 怎么画的（阶梯过程）": mo.md(r"""
1. 把所有样本按预测概率**降序**排
2. 从 (0,0) 出发，τ = +∞（所有都判负）
3. 阈值降低 → 每碰到一个真正例 → **TPR ↑**（向上一格）
4. 每碰到一个真负例 → **FPR ↑**（向右一格）
5. 终点 (1,1) → τ = 0（所有都判正）

形状直觉：
- **完美分类**：先全部向上（正例排前），再全部向右 → "靠左上角"的折线
- **随机模型**：上下交错 → 接近对角线
- **比随机差**：先向右（负例排前）→ 曲线在对角线下方

**右图直方图就是排序的可视化**：正负分布越分开，ROC 越靠左上角。
"""),
        "比随机差怎么办（AUC < 0.5）": mo.md(r"""
**别重训**——把分数取反就行：

```python
proba = clf.predict_proba(X)[:, 1]
# 如果 AUC < 0.5
proba_flipped = 1 - proba   # 或者 -proba
roc_auc_score(y, proba_flipped)   # 现在 > 0.5
```

数学等价：把 ROC 曲线沿 y=x 翻转，AUC 从 a 变 1-a。

**根因排查**：99% 的"AUC<0.5" 是**标签反了**或 `pos_label` 设错了。

```python
# sklearn 默认正例 = classes_[1]
# 如果你的 y = ["恶性", "良性"]，按字母序 "良性" < "恶性"
# 那 classes_ = ["良性", "恶性"]，正例是恶性 ✓
# 但如果 y = [4, 2]，正例是 4 ✓
# 显式查 clf.classes_ 而不是猜
```
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
4  ├ quality ┬ τ ┬ shade h=4 ─┤
8  ├──── 状态卡 h=5 全宽 ────┤
13 ├ ROC(11) ┬ 概率分布(13) ─┤  ← h=18
31 ├──── accordion h=5 全宽 ────┤
36
```
""")
    return


if __name__ == "__main__":
    app.run()
