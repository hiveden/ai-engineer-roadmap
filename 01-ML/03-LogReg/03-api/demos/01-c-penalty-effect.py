"""
LR 三参数 · solver / penalty / C 怎么影响决策边界和系数

互动：
- 拖 C 滑块（log 尺度）→ 决策边界 + coef_ 条形图同步变
- 切 L1 / L2 → 看稀疏 vs 收缩
跑：marimo edit 01-c-penalty-effect.py --port 2735 --headless --no-token
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/01-c-penalty-effect.grid.json",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    return LogisticRegression, alt, make_classification, mo, np, pd


@app.cell
def _(mo):
    mo.md(r"""
    # LR 三参数 · solver / penalty / C

    `LogisticRegression(solver=..., penalty=..., C=...)`

    - **solver** = 优化方法（liblinear / lbfgs / saga …）—— 不影响形式，只影响怎么找到最优 w
    - **penalty** = `l1`（稀疏 / 自动特征选择）or `l2`（收缩 / 比例缩小）
    - **C** = 正则强度倒数（**越小正则越强**，反直觉）

    左图决策边界 + 概率热图；右图 6 维 coef_ 条形图（看哪些特征被压平 / 归零）。
    """)
    return


@app.cell
def _(mo):
    log_C = mo.ui.slider(-3.0, 3.0, value=0.0, step=0.25,
                         label="log₁₀(C)（C=10^x）", show_value=True)
    penalty = mo.ui.dropdown(
        options={"L2 收缩（默认）": "l2", "L1 稀疏": "l1", "无正则": "none"},
        value="L2 收缩（默认）", label="penalty")
    n_noise = mo.ui.slider(0, 4, value=2, step=1,
                           label="无关噪声特征数", show_value=True)
    mo.hstack([log_C, penalty, n_noise],
              widths=[2.2, 1.5, 1.5], justify="space-around")
    return log_C, n_noise, penalty


@app.cell
def _(make_classification, n_noise, np):
    # 单一来源：2 信号特征 + N 噪声特征（用来观察 L1/L2 怎么处理无关特征）
    n_total = 2 + int(n_noise.value)
    X, y = make_classification(
        n_samples=200, n_features=n_total,
        n_informative=2, n_redundant=0, n_repeated=0,
        n_clusters_per_class=1, class_sep=1.6,
        flip_y=0.05, random_state=7,
    )
    # 标准化（LR 默认对 scale 敏感）
    X = (X - X.mean(0)) / X.std(0)
    feature_names = [f"x{i+1}" + ("（信号）" if i < 2 else "（噪声）")
                     for i in range(n_total)]
    return X, feature_names, n_total, y


@app.cell
def _(LogisticRegression, X, log_C, n_total, np, penalty, y):
    C = float(10 ** log_C.value)
    # solver 自动适配 penalty
    if penalty.value == "l1":
        solver = "liblinear"
        kw = {"penalty": "l1", "C": C}
    elif penalty.value == "none":
        solver = "lbfgs"
        kw = {"penalty": None, "C": C}
    else:
        solver = "lbfgs"
        kw = {"penalty": "l2", "C": C}
    clf = LogisticRegression(solver=solver, max_iter=2000, **kw)
    clf.fit(X, y)
    coef = clf.coef_.ravel()
    intercept = float(clf.intercept_[0])
    train_acc = float(clf.score(X, y))
    # 取前 2 维做决策边界（其它维度固定为 0）
    w_plot = coef[:2]
    b_plot = intercept  # 因为其余 X[:, 2:] = 0 时贡献 0（已标准化均值为 0）
    n_zero = int((np.abs(coef) < 1e-4).sum())
    return C, b_plot, coef, intercept, n_total, n_zero, solver, train_acc, w_plot


@app.cell
def _(X, alt, b_plot, np, pd, w_plot, y):
    # 视图 1：决策边界（前 2 维）+ 样本散点
    _xs = np.linspace(-3, 3, 60)
    _ys = np.linspace(-3, 3, 60)
    _xx, _yy = np.meshgrid(_xs, _ys)
    _z = _xx * w_plot[0] + _yy * w_plot[1] + b_plot
    _p = 1.0 / (1.0 + np.exp(-_z))
    _heat_df = pd.DataFrame({
        "x1": _xx.ravel(), "x2": _yy.ravel(), "p": _p.ravel()
    })
    heat = alt.Chart(_heat_df).mark_rect(opacity=0.55).encode(
        x=alt.X("x1:Q", bin=alt.Bin(maxbins=60), scale=alt.Scale(domain=[-3, 3])),
        y=alt.Y("x2:Q", bin=alt.Bin(maxbins=60), scale=alt.Scale(domain=[-3, 3])),
        color=alt.Color("p:Q",
            scale=alt.Scale(domain=[0, 0.5, 1],
                            range=["#dc2626", "#f8fafc", "#16a34a"]),
            legend=alt.Legend(title="P(y=1)")),
    )
    # 决策边界 p=0.5 ⇔ z=0：x2 = -(w1*x1 + b)/w2
    if abs(w_plot[1]) > 1e-3:
        _line_x1 = np.linspace(-3, 3, 50)
        _line_x2 = -(w_plot[0] * _line_x1 + b_plot) / w_plot[1]
    else:
        _line_x1 = np.full(50, -b_plot / max(w_plot[0], 1e-3))
        _line_x2 = np.linspace(-3, 3, 50)
    _line_df = pd.DataFrame({"x1": _line_x1, "x2": _line_x2})
    line = alt.Chart(_line_df).mark_line(stroke="#0f172a", strokeWidth=2.5).encode(
        x=alt.X("x1:Q", scale=alt.Scale(domain=[-3, 3])),
        y=alt.Y("x2:Q", scale=alt.Scale(domain=[-3, 3])),
    )
    _pts_df = pd.DataFrame({
        "x1": X[:, 0], "x2": X[:, 1],
        "类": ["正" if t == 1 else "负" for t in y]
    })
    pts = alt.Chart(_pts_df).mark_circle(size=55, opacity=0.85,
                                          stroke="white", strokeWidth=0.6).encode(
        x="x1:Q", y="x2:Q",
        color=alt.Color("类:N",
            scale=alt.Scale(domain=["正", "负"], range=["#16a34a", "#dc2626"]),
            legend=None),
    )
    chart_boundary = (heat + line + pts).resolve_scale(color="independent").properties(
        width=380, height=380)
    return (chart_boundary,)


@app.cell
def _(alt, coef, feature_names, np, pd):
    # 视图 2：coef_ 条形图（按绝对值排序，零附近用阴影标记）
    _df = pd.DataFrame({
        "特征": feature_names,
        "w": coef,
        "abs": np.abs(coef),
        "归零": ["是" if abs(c) < 1e-4 else "否" for c in coef]
    })
    _df = _df.sort_values("abs", ascending=False)
    bars = alt.Chart(_df).mark_bar().encode(
        x=alt.X("w:Q", title="权重 w（已归一特征下）",
                scale=alt.Scale(domain=[-3, 3])),
        y=alt.Y("特征:N", sort=_df["特征"].tolist(), title=None),
        color=alt.Color("归零:N",
            scale=alt.Scale(domain=["否", "是"], range=["#2563eb", "#94a3b8"]),
            legend=alt.Legend(orient="top")),
        tooltip=["特征:N", alt.Tooltip("w:Q", format=".4f")],
    )
    zero_line = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        stroke="#0f172a", strokeWidth=1).encode(x="x:Q")
    chart_coef = (bars + zero_line).properties(width=380, height=380)
    return (chart_coef,)


@app.cell
def _(chart_boundary, chart_coef, mo):
    mo.hstack([chart_boundary, chart_coef],
              widths=[1, 1], justify="space-around")
    return


@app.cell
def _(C, coef, mo, n_total, n_zero, np, penalty, solver, train_acc):
    # 状态卡：参数总结 + 收缩 / 稀疏 度量
    _box = "padding:10px 14px;border-radius:6px;font-size:13px;line-height:1.7;"
    norm_l1 = float(np.abs(coef).sum())
    norm_l2 = float(np.sqrt((coef ** 2).sum()))
    if C < 0.05:
        c_note = "C 极小 → 正则极强 → 系数被压扁，可能欠拟合"
        c_color = "#dc2626"
    elif C < 0.5:
        c_note = "C 较小 → 正则强 → 系数收缩明显"
        c_color = "#ea580c"
    elif C < 5:
        c_note = "C 适中 → 通用起点"
        c_color = "#16a34a"
    else:
        c_note = "C 大 → 正则弱 → 接近无正则 LR，可能过拟合"
        c_color = "#eab308"
    if penalty.value == "l1" and n_zero > 0:
        sparse_note = (f"<strong style=\"color:#16a34a;\">L1 让 {n_zero}/{n_total} "
                       f"个特征系数归零</strong> · 自动特征选择")
    elif penalty.value == "l1":
        sparse_note = "L1 但当前 C 不够强，没有特征被归零"
    elif penalty.value == "l2":
        sparse_note = "L2 不会让系数恰好等于 0，只按比例缩小（收缩）"
    else:
        sparse_note = "无正则 → 系数仅由数据驱动"
    mo.md(f"""
<div style="display:flex;gap:10px;margin:6px 0;">
  <div style="flex:1.2;background:#eff6ff;border-left:4px solid #2563eb;{_box}">
    <strong>当前模型</strong><br>
    <code>LogisticRegression(solver='{solver}', penalty='{penalty.value}', C={C:.3g})</code><br>
    训练准确率 = <strong style="color:#2563eb;">{train_acc:.1%}</strong>
  </div>
  <div style="flex:1;background:#fff7ed;border-left:4px solid {c_color};{_box}">
    <strong>C = {C:.3g}</strong>（log₁₀ = {np.log10(C):.2f}）<br>
    {c_note}<br>
    ‖w‖₁ = {norm_l1:.3f} · ‖w‖₂ = {norm_l2:.3f}
  </div>
  <div style="flex:1;background:#f0fdf4;border-left:4px solid #16a34a;{_box}">
    <strong>正则形态</strong><br>
    {sparse_note}
  </div>
</div>
""")
    return


@app.cell
def _(mo):
    mo.accordion({
        "为什么 C 越小正则越强（反 Ridge 直觉）": mo.md(r"""
LR 损失 + 正则项的写法：

$$J(w) = \frac{1}{C} \cdot \text{Reg}(w) + \sum_i \mathcal{L}(y_i, h_w(x_i))$$

注意正则放在 **1/C** 系数上：
- C 大 → 1/C 小 → 正则项权重小 → 系数自由
- C 小 → 1/C 大 → 正则项权重大 → 系数被压

而 `Ridge(alpha=...)` 写的是 $J = \alpha \cdot \text{Reg}(w) + \text{Loss}$，
α 直接是正则权重 → α 大正则强。**两个反方向**。

**惯性陷阱**：从 Ridge / Lasso 迁移过来的人常以为"系数大正则强"，
LR 是反的。`C ∈ {0.001, 0.01, 0.1, 1, 10, 100}` 网格搜索时记得方向。
"""),
        "L1 vs L2 几何直觉（为什么 L1 稀疏）": mo.md(r"""
约束图直觉：
- **L2 约束** $\sum w_i^2 \leq r^2$：圆形 / 球形（光滑）
- **L1 约束** $\sum |w_i| \leq r$：菱形 / 多面体（带尖角）

损失等高线和约束区域第一次接触的点 = 解。
- L2 圆形 → 接触点几乎不会落在轴上 → 所有 w 都非零（**收缩**）
- L1 菱形 → **尖角恰好在轴上** → 接触点经常落在轴上（部分 w = 0，**稀疏**）

工程价值：
- 高维数据（很多无关特征）→ L1 自动筛掉无关特征
- 想保留所有特征但避免过拟合 → L2
- L1 + L2 混合 = `elasticnet`（仅 saga 支持）
"""),
        "solver 怎么选": mo.md(r"""
| 数据规模 | penalty | 推荐 solver |
|---|---|---|
| 小（< 10⁴ 样本）| l1 / l2 | `liblinear` |
| 中小 | l2 / none | `lbfgs`（≥1.0 默认）/ `newton-cg` |
| 大 | l2 / none | `sag` |
| 大 | l1 / l2 / elasticnet | `saga` |

**坑**：`solver=lbfgs, penalty=l1` → 直接报错
（`Solver lbfgs supports only 'l2' or 'none' penalties`）。
sklearn ≥1.5 会建议替代 solver。

**未收敛警告**：`ConvergenceWarning: lbfgs failed to converge` —— 99% 的原因是
没标准化或 `max_iter` 太小，不是模型问题。本 demo `max_iter=2000` + 已标准化。
"""),
        "predict_proba 输出方向 + classes_ 排序": mo.md(r"""
```python
clf.classes_           # array([0, 1]) 或 ["良性", "恶性"]
clf.predict_proba(X)   # 每行 [P(y=classes_[0]), P(y=classes_[1])]
```

sklearn 把 **`classes_[1]`（数量少 / 字母靠后）当正例**：
- y = [0, 1] → 正例 = 1，proba[:, 1] 是 P(y=1)
- y = ["良性", "恶性"] → 字母序"恶性" > "良性" → 正例 = "恶性"
- y = [4, 2]（恶性 4 良性 2）→ 正例 = 4

**生产代码不要靠默认**：
```python
pos_idx = list(clf.classes_).index("恶性")
proba_pos = clf.predict_proba(X)[:, pos_idx]
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
4  ├ logC ┬ penalty ┬ noise h=4 ─┤
8  ├──── 状态卡 h=5 全宽 ────┤
13 ├ 决策边界(12) ┬ coef 条形(12) ─┤  ← h=18
31 ├──── accordion h=5 全宽 ────┤
36
```
""")
    return


if __name__ == "__main__":
    app.run()
