"""
sklearn LinearRegression API · 5 步流程教学走查

把 README 的 5 步「导入 → 准备数据 → 实例化 → fit → predict」拆成可拖参数交互：
- dropdown 切 3 个数据集（5 点 PPT 原版 / 30 点加噪 / 加州房价单特征）
- switch 控 fit_intercept（开/关 → 直观看截距的作用）
- slider 拖 x_new → 同时显示 model.predict([[x]]) 和手动 coef_*x+intercept_，验"sklearn 不是黑盒"
- 6 个预设场景 + 教学卡片：先理解再操作

跑：cd 01-ML/02-LR/02-api/demos && marimo edit --port 2736 api-walkthrough.py
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/api-walkthrough.grid.json",
    css_file="marimo.css",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import altair as alt
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import fetch_california_housing

    return LinearRegression, alt, fetch_california_housing, mo, np, pd


@app.cell
def _(mo):
    mo.md(
        r"""
        # sklearn `LinearRegression` · 5 步流程走查

        ## 🎬 怎么用这个 demo

        1. **先选一个场景**（顶部 dropdown），读绿色卡片里的「关键发现」——理解这个场景演示什么
        2. **看右图 + 参数面板**：场景的预期结果是不是真的发生了？
        3. **想自己玩**：选「✋ 自定义」，下方 3 个滑块解锁

        每个场景 30 秒 - 2 分钟，6 个跑完 5-10 分钟覆盖 sklearn LR API 全部要点。
        """
    )
    return


@app.cell
def _():
    # ===== 6 个预设场景 + 教学说明 =====
    SCENARIOS = {
        "0 · ✋ 自定义（用下方滑块手动调）": None,
        "1 · 🎯 PPT 经典：4 行代码跑通": {
            "dataset": "身高体重 · 5 点（PPT 原版）",
            "fit_intercept": True,
            "x_new": 176.0,
            "教学": (
                "**预期结果**：coef_ ≈ 0.93、intercept_ ≈ -93.45、predict([[176]]) ≈ 70.21。\n\n"
                "**怎么看**：右图红线穿过 5 个蓝点，新样本（蓝菱形）落在红线上。\n\n"
                "**关键发现**：sklearn 4 行代码（导入 + 实例化 + fit + predict）就能用，"
                "不用关心怎么算的。"
            ),
        },
        "2 · 🚫 关掉 fit_intercept：看 b 被强制 0": {
            "dataset": "身高体重 · 5 点（PPT 原版）",
            "fit_intercept": False,
            "x_new": 176.0,
            "教学": (
                "**预期结果**：intercept_ = 0（强制），coef_ 跳到 ~0.4，"
                "预测严重偏低（约 70 → 不对）。\n\n"
                "**怎么看**：红线被强迫穿过原点 (0, 0)，5 个蓝点都在红线上方很远——"
                "拟合明显失败。\n\n"
                "**关键发现**：fit_intercept 默认 True 是有道理的——"
                "除非你确定数据已经中心化（X 和 y 都减过均值），不要乱关。"
            ),
        },
        "3 · 📈 数据从 5 点 → 30 点：参数稳吗？": {
            "dataset": "身高体重 · 加噪声 30 点",
            "fit_intercept": True,
            "x_new": 176.0,
            "教学": (
                "**预期结果**：coef_ 仍 ≈ 0.93，但残差总和变大（数据更多 + 加了噪声）。\n\n"
                "**怎么看**：散点变成一团，红线还在中间穿过——参数没变，但单点拟合不再完美。\n\n"
                "**关键发现**：参数（coef、intercept）由**数据分布**决定，不由数据量决定。"
                "多点反而让 LR 更稳，对噪声更鲁棒。"
            ),
        },
        "4 · 🌍 切到加州房价真实数据": {
            "dataset": "加州房价 · 单特征版（MedInc → target）",
            "fit_intercept": True,
            "x_new": 5.0,
            "教学": (
                "**预期结果**：coef_ ≈ 0.42、intercept_ ≈ 0.4、"
                "predict([[5.0]]) ≈ 2.5（10 万美元 ≈ 25 万）。\n\n"
                "**怎么看**：散点从「身高-体重」变成「收入-房价」，单位完全不同。\n\n"
                "**关键发现**：**API 一行不变**，sklearn 自动算出新的 coef 和 intercept——"
                "这就是参数化模型的好处：换数据不换代码。"
            ),
        },
        "5 · ⚠️ 加州数据 + 关 fit_intercept": {
            "dataset": "加州房价 · 单特征版（MedInc → target）",
            "fit_intercept": False,
            "x_new": 5.0,
            "教学": (
                "**预期结果**：intercept_ = 0，coef_ 被迫调高来弥补，R² 显著下降。\n\n"
                "**怎么看**：直线还是穿过原点，但加州房价数据中心点不在 (0, 0)，"
                "强制穿原点的拟合明显歪。\n\n"
                "**关键发现**：fit_intercept=False **危害比 5 点数据更明显**——"
                "大数据 + 没中心化 = 永远别关。"
            ),
        },
        "6 · 🎚️ 拖动新样本：看 predict 范围": {
            "dataset": "身高体重 · 5 点（PPT 原版）",
            "fit_intercept": True,
            "x_new": 165.0,
            "教学": (
                "**预期结果**：把 x_new 拖到 [160, 180] 范围内（数据范围内 = 内插）predict 准；"
                "拖出范围（如 200 或 140）predict 仍能算但**外推风险高**。\n\n"
                "**怎么看**：蓝菱形随滑块移动，始终在红线上"
                "（因为 LR 严格按 y = coef·x + intercept 算）。\n\n"
                "**关键发现**：LR 的 predict 没有「我不知道」机制——给任何 x 都返回一个数。"
                "**外推（extrapolation）是 LR 的硬限制**。"
            ),
        },
    }
    return (SCENARIOS,)


@app.cell
def _(SCENARIOS, mo):
    preset = mo.ui.dropdown(
        options=list(SCENARIOS.keys()),
        value="0 · ✋ 自定义（用下方滑块手动调）",
        label="🎬 选场景（先选一个再看下方说明）",
    )
    return (preset,)


@app.cell
def _(SCENARIOS, mo, preset):
    # ===== 场景说明卡片（先理解再操作）=====
    _p = SCENARIOS.get(preset.value)
    if _p is None:
        mo.callout(
            mo.md("**自定义模式** —— 用下方 3 个滑块手动调，自由探索 sklearn LR API。"),
            kind="info",
        )
    else:
        mo.callout(
            mo.md(f"### {preset.value}\n\n{_p['教学']}"),
            kind="success",
        )
    return


@app.cell
def _(mo):
    # ===== 控件区（自定义模式时用） =====
    dataset = mo.ui.dropdown(
        options=[
            "身高体重 · 5 点（PPT 原版）",
            "身高体重 · 加噪声 30 点",
            "加州房价 · 单特征版（MedInc → target）",
        ],
        value="身高体重 · 5 点（PPT 原版）",
        label="数据集",
    )
    fit_intercept = mo.ui.switch(value=True, label="fit_intercept（学截距 b）")
    return dataset, fit_intercept


@app.cell
def _(SCENARIOS, dataset, fit_intercept, preset):
    # ===== 解析 effective 值（场景预设 vs 自定义滑块二选一） =====
    _p = SCENARIOS.get(preset.value)
    if _p is None:
        eff_dataset = dataset.value
        eff_fit_intercept = fit_intercept.value
    else:
        eff_dataset = _p["dataset"]
        eff_fit_intercept = _p["fit_intercept"]
    return eff_dataset, eff_fit_intercept


@app.cell
def _(fetch_california_housing, np):
    # ===== 数据集准备（缓存：避免每次重算） =====
    def _make_5pt():
        x = np.array([[160.0], [166.0], [172.0], [174.0], [180.0]])
        y = np.array([56.3, 60.6, 65.1, 68.5, 75.0])
        return x, y, "身高 (cm)", "体重 (kg)"

    def _make_30pt():
        rng = np.random.default_rng(7)
        _x = rng.uniform(150, 190, 30)
        _y = 0.93 * _x - 93.45 + rng.normal(0, 2.5, 30)
        return _x.reshape(-1, 1), _y, "身高 (cm)", "体重 (kg)"

    def _make_california():
        _ds = fetch_california_housing()
        _x = _ds.data[:100, 0:1]
        _y = _ds.target[:100]
        return _x, _y, "MedInc (万美元)", "房价中位数 (10 万美元)"

    DATA_FACTORY = {
        "身高体重 · 5 点（PPT 原版）": _make_5pt,
        "身高体重 · 加噪声 30 点": _make_30pt,
        "加州房价 · 单特征版（MedInc → target）": _make_california,
    }
    return (DATA_FACTORY,)


@app.cell
def _(DATA_FACTORY, eff_dataset):
    # 当前数据集（用 effective 值）
    X, y, x_label, y_label = DATA_FACTORY[eff_dataset]()
    return X, x_label, y, y_label


@app.cell
def _(X, mo):
    # 数据集小标题（独占 cell · grid 友好）
    mo.md(f"### 当前数据集（{len(X)} 行 × 1 特征 + 1 标签）")
    return


@app.cell
def _(X, pd, x_label, y, y_label):
    # 数据集表格（独占 cell）
    _df = pd.DataFrame({
        "#": range(1, len(X) + 1),
        x_label: X.ravel().round(3),
        y_label: y.round(3),
    })
    table_view = _df  # 先转出来给下面 cell 用 mo.ui.table 渲染
    return (table_view,)


@app.cell
def _(mo, table_view):
    mo.ui.table(table_view, page_size=10, selection=None)
    return


@app.cell
def _(X, mo, x_label, y_label):
    # 数据集 shape 提示（独占 cell）
    mo.md(
        f"<div style='font-size:12px;color:#6b7280;'>"
        f"特征 X：<code>{x_label}</code>（shape <code>({len(X)}, 1)</code>）&nbsp;·&nbsp;"
        f"标签 y：<code>{y_label}</code>（shape <code>(len({X.shape[0]}),)</code>）"
        f"</div>"
    )
    return


@app.cell
def _(LinearRegression, X, eff_fit_intercept, y):
    # ===== 5 步流程的「步 3 + 步 4」实时跑 =====
    model = LinearRegression(fit_intercept=eff_fit_intercept)
    model.fit(X, y)
    coef = float(model.coef_[0])
    intercept = float(model.intercept_)
    n_features = int(model.n_features_in_)
    return coef, intercept, model, n_features


@app.cell
def _(X, mo):
    # ===== 步 5 的输入控件 x_new =====
    _xmin = float(X.min())
    _xmax = float(X.max())
    _span = _xmax - _xmin
    _step = max(round(_span / 100, 2), 0.01)
    _default = 176.0 if 150 <= _xmin and _xmax <= 200 else (_xmin + _xmax) / 2

    x_new = mo.ui.slider(
        start=round(_xmin, 2),
        stop=round(_xmax, 2),
        step=_step,
        value=round(_default, 2),
        label="x_new （新样本特征值）",
        show_value=True,
    )
    return (x_new,)


@app.cell
def _(SCENARIOS, preset, x_new):
    # eff_x_new：场景预设 vs 自定义 slider 二选一
    _p = SCENARIOS.get(preset.value)
    if _p is None:
        eff_x_new = x_new.value
    else:
        eff_x_new = _p["x_new"]
    return (eff_x_new,)


@app.cell
def _(mo):
    mo.md("## 1️⃣ 控件")
    return


@app.cell
def _(preset):
    preset
    return


@app.cell
def _(SCENARIOS, mo, preset):
    _is_preset = SCENARIOS.get(preset.value) is not None
    if _is_preset:
        mo.md(
            "<div style='font-size:12px;color:#92400e;background:#fef3c7;"
            "padding:6px 10px;border-radius:4px;'>"
            "📌 已锁定为预设值，下方滑块本次不生效。选「✋ 自定义」解锁。"
            "</div>"
        )
    else:
        mo.md(
            "<div style='font-size:12px;color:#1e40af;background:#dbeafe;"
            "padding:6px 10px;border-radius:4px;'>"
            "✋ 自定义模式：下方滑块控制实际值。"
            "</div>"
        )
    return


@app.cell
def _(dataset, fit_intercept, mo, x_new):
    # 控件组合排版（一行 hstack 是合理颗粒度）
    mo.hstack([dataset, fit_intercept, x_new], gap=2, justify="start")
    return


@app.cell
def _(coef, eff_fit_intercept, intercept, mo):
    # ===== 5 步代码块 =====
    _fi = "True" if eff_fit_intercept else "False"
    code_md = mo.md(
        rf"""
## 2️⃣ 5 步代码 · 实时跑

```python
# 1. 导入
from sklearn.linear_model import LinearRegression

# 2. 准备数据：X 必须 2D（n_samples × n_features），y 1D
#    （上方 dropdown 切换不同数据集 → X, y 联动）

# 3. 实例化（内部走 SVD 解最小二乘）
model = LinearRegression(fit_intercept={_fi})

# 4. 训练 fit() —— sklearn 算 w 和 b
model.fit(X, y)
print(model.coef_, model.intercept_)
# →  coef_ = [{coef:.4f}],  intercept_ = {intercept:.4f}

# 5. 预测 predict() —— 套公式 ŷ = w·x + b
y_hat = model.predict([[x_new]])
```

> 切场景或滑块 → 上面的数字实时变。
"""
    )
    code_md
    return


@app.cell
def _(X, alt, coef, eff_x_new, intercept, model, np, pd, x_label, y, y_label):
    # ===== 主图 =====
    _x_flat = X[:, 0]
    _y_pred_train = model.predict(X)

    _scatter_df = pd.DataFrame(
        {
            "x": _x_flat,
            "y": y,
            "y_pred": _y_pred_train,
            "residual": y - _y_pred_train,
        }
    )

    _xmin = float(min(_x_flat.min(), eff_x_new))
    _xmax = float(max(_x_flat.max(), eff_x_new))
    _pad = (_xmax - _xmin) * 0.1
    _x_line = np.linspace(_xmin - _pad, _xmax + _pad, 50)
    _line_df = pd.DataFrame({"x": _x_line, "y": coef * _x_line + intercept})

    _xn = float(eff_x_new)
    _y_new_sklearn = float(model.predict(np.array([[_xn]]))[0])
    _new_df = pd.DataFrame({"x": [_xn], "y": [_y_new_sklearn]})

    _ymin = float(min(y.min(), _y_new_sklearn, (coef * _x_line + intercept).min()))
    _ymax = float(max(y.max(), _y_new_sklearn, (coef * _x_line + intercept).max()))
    _ypad = (_ymax - _ymin) * 0.12

    _x_dom = [_xmin - _pad, _xmax + _pad]
    _y_dom = [_ymin - _ypad, _ymax + _ypad]

    _residuals = (
        alt.Chart(_scatter_df)
        .mark_rule(color="#ef4444", opacity=0.4, strokeDash=[3, 2])
        .encode(x="x:Q", y="y:Q", y2="y_pred:Q")
    )
    _pts = (
        alt.Chart(_scatter_df)
        .mark_circle(size=140, color="#1f77b4", stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=_x_dom), title=x_label),
            y=alt.Y("y:Q", scale=alt.Scale(domain=_y_dom), title=y_label),
            tooltip=[
                alt.Tooltip("x:Q", format=".3f"),
                alt.Tooltip("y:Q", format=".3f"),
                alt.Tooltip("y_pred:Q", format=".3f"),
                alt.Tooltip("residual:Q", format=".3f"),
            ],
        )
    )
    _line = (
        alt.Chart(_line_df)
        .mark_line(color="#ef4444", strokeWidth=3)
        .encode(x="x:Q", y="y:Q")
    )
    _diamond = (
        alt.Chart(_new_df)
        .mark_point(
            shape="diamond",
            size=420,
            color="#0ea5e9",
            filled=True,
            stroke="white",
            strokeWidth=2,
        )
        .encode(
            x="x:Q",
            y="y:Q",
            tooltip=[
                alt.Tooltip("x:Q", title="x_new", format=".3f"),
                alt.Tooltip("y:Q", title="predict(x_new)", format=".4f"),
            ],
        )
    )
    _diamond_label = (
        alt.Chart(_new_df.assign(label=[f"x={_xn:.2f} → ŷ={_y_new_sklearn:.3f}"]))
        .mark_text(
            align="left",
            dx=12,
            dy=-12,
            fontSize=12,
            fontWeight="bold",
            color="#0369a1",
        )
        .encode(x="x:Q", y="y:Q", text="label:N")
    )

    chart = (_residuals + _line + _pts + _diamond + _diamond_label).properties(
        width=620,
        height=380,
        title=f"sklearn 拟合直线  ŷ = {coef:.4f}·x + ({intercept:.4f})",
    )
    return (chart,)


@app.cell
def _(mo):
    mo.md("## 3️⃣ 拟合可视化")
    return


@app.cell
def _(chart, mo):
    mo.ui.altair_chart(chart)
    return


@app.cell
def _(
    X,
    coef,
    eff_dataset,
    eff_fit_intercept,
    eff_x_new,
    intercept,
    mo,
    model,
    n_features,
    np,
    y,
):
    # ===== 参数面板 =====
    _xn = float(eff_x_new)
    _y_sklearn = float(model.predict(np.array([[_xn]]))[0])
    _y_manual = coef * _xn + intercept
    _consistent = abs(_y_sklearn - _y_manual) < 1e-9
    _mark = (
        '<span style="color:#16a34a;font-weight:700;">✓ 一致</span>'
        if _consistent
        else '<span style="color:#dc2626;font-weight:700;">✗ 不一致</span>'
    )

    _y_pred_all = model.predict(X)
    _mse = float(np.mean((y - _y_pred_all) ** 2))
    _r2 = float(model.score(X, y))

    _fi_badge = (
        '<span style="background:#dbeafe;color:#1e40af;padding:2px 8px;border-radius:5px;font-size:12px;">fit_intercept = True</span>'
        if eff_fit_intercept
        else '<span style="background:#fef3c7;color:#92400e;padding:2px 8px;border-radius:5px;font-size:12px;">fit_intercept = False（强制 b=0）</span>'
    )

    _panel = f"""
<div style="font-family:ui-monospace,monospace; font-size:14px; line-height:1.75;
            background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px;
            padding:14px 18px;">

<div style="margin-bottom:10px;">
  <span style="background:#ecfeff;color:#0e7490;padding:2px 8px;border-radius:5px;font-size:12px;margin-right:6px;">{eff_dataset}</span>
  {_fi_badge}
</div>

<table style="border-collapse:collapse;width:100%;">
<thead>
  <tr style="border-bottom:1px solid #d1d5db;">
    <th style="text-align:left;padding:6px 10px;color:#6b7280;font-weight:600;">模型属性</th>
    <th style="text-align:left;padding:6px 10px;color:#6b7280;font-weight:600;">sklearn 输出</th>
    <th style="text-align:left;padding:6px 10px;color:#6b7280;font-weight:600;">含义</th>
  </tr>
</thead>
<tbody>
  <tr><td style="padding:4px 10px;"><code>model.coef_</code></td>
      <td style="padding:4px 10px;"><b>[{coef:.4f}]</b></td>
      <td style="padding:4px 10px;color:#6b7280;">权重 w（每多 1 单位 x，y 变 w）</td></tr>
  <tr><td style="padding:4px 10px;"><code>model.intercept_</code></td>
      <td style="padding:4px 10px;"><b>{intercept:.4f}</b></td>
      <td style="padding:4px 10px;color:#6b7280;">截距 b（x=0 时 y 的基线）</td></tr>
  <tr><td style="padding:4px 10px;"><code>model.n_features_in_</code></td>
      <td style="padding:4px 10px;"><b>{n_features}</b></td>
      <td style="padding:4px 10px;color:#6b7280;">训练时见过的特征数</td></tr>
  <tr><td style="padding:4px 10px;"><code>model.score(X, y)</code></td>
      <td style="padding:4px 10px;"><b>R² = {_r2:.4f}</b>,&nbsp;MSE = {_mse:.4f}</td>
      <td style="padding:4px 10px;color:#6b7280;">拟合优度 / 平均平方误差</td></tr>
</tbody>
</table>

<div style="height:10px;border-top:1px dashed #d1d5db;margin:12px 0 8px 0;"></div>

<b>预测对账 · x_new = {_xn:.4f}</b>
<table style="border-collapse:collapse;width:100%;margin-top:6px;">
<tbody>
  <tr><td style="padding:4px 10px;color:#6b7280;width:48%;"><code>model.predict([[{_xn:.2f}]])</code></td>
      <td style="padding:4px 10px;"><b>{_y_sklearn:.6f}</b></td></tr>
  <tr><td style="padding:4px 10px;color:#6b7280;">手动 <code>coef_·x + intercept_</code></td>
      <td style="padding:4px 10px;"><b>{coef:.4f} × {_xn:.2f} + ({intercept:.4f}) = {_y_manual:.6f}</b></td></tr>
  <tr><td style="padding:4px 10px;color:#6b7280;">差值 |sklearn − 手动|</td>
      <td style="padding:4px 10px;">{abs(_y_sklearn - _y_manual):.2e} &nbsp;&nbsp; {_mark}</td></tr>
</tbody>
</table>

</div>
"""
    panel = mo.md(_panel)
    return (panel,)


@app.cell
def _(mo):
    mo.md("## 4️⃣ 模型属性 + 公式验算面板")
    return


@app.cell
def _(panel):
    panel
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ### 一句话总结

        > **`LinearRegression()` = 一个把 `coef_` 和 `intercept_` 算出来的对象**。
        > `fit()` 算两个数，`predict()` 套公式 `coef_·x + intercept_`。
        > 整套 API 4 行代码能跑通，6 个场景跑完覆盖所有要点。
        """
    )
    return


@app.cell
def _(mo):
    # ===== 📐 录屏 grid 布局参考（开发用 · 录屏隐藏）=====
    mo.accordion(
        {
            "📐 Grid 布局参考（16:9 · 录屏推荐）": mo.md(
                r"""
**目标 viewport**：1280px maxWidth · 24 columns · rowHeight 20 · 16:9 横屏。
三段式：标题 + 场景卡（顶）/ 控件区 + 数据表 + 5 步代码 + 拟合图（中）/ 模型属性面板 + 总结（底）。

### 横屏骨架

```
   0                       12                       24
y=0   ┌──────── 大标题 + 玩法说明 (cell 1) ─────────┐  h=4
y=4   ├──────── 场景说明 callout (cell 4) ──────────┤  h=5
y=9   ├──────── ## 1️⃣ 控件 (cell 16) ──────────────┤  h=1
y=10  ├──────── preset dropdown (cell 17) ──────────┤  h=2
y=12  ├──────── 锁定/自定义 提示横条 (cell 18) ─────┤  h=1
y=13  ├──── dataset / fit_intercept / x_new (19) ───┤  h=3
y=16  ├── 数据集小标题 (9)  │ ## 3️⃣ 拟合 (22) ─────┤  h=1
y=17  ├── 数据表 (11)       │                       │
      │   h=8               │  拟合图 chart (23)    │
y=25  ├── shape 提示 (12)   │  h=21                 │
      │   h=2               │                       │
y=27  ├── 5 步代码 (20)     │                       │
      │   h=11              │                       │
y=38  ├──────── ## 4️⃣ 模型属性 (cell 25) ──────────┤  h=1
y=39  ├──────── 参数面板 panel (cell 26) ───────────┤  h=11
y=50  ├──────── 一句话总结 (cell 27) ───────────────┤  h=4
```

### 镜头脚本（横屏切镜头 = 切场景）

| # | 时长 | 操作 | 教学焦点 |
|---|---|---|---|
| **A** | 0-30s | 选场景 1 · PPT 经典 | 4 行代码跑通 sklearn LR |
| **B** | 30-60s | 选场景 2 · 关 fit_intercept | 看 b=0 拟合明显失败 |
| **C** | 60-100s | 选场景 3 · 30 点加噪 | 参数稳，靠数据分布 |
| **D** | 100-140s | 选场景 4 · 加州房价 | API 不变，coef/intercept 自动算新 |
| **E** | 140-170s | 选场景 5 · 加州 + 关 b | 大数据 + 没中心化 = 别关 |
| **F** | 170-220s | 选场景 6 · 拖 x_new | 外推风险：LR 没「我不知道」机制 |

### position=null 的 cell

imports / SCENARIOS dict / preset 定义 / 控件定义 / DATA_FACTORY / X y 加载 /
table_view 计算 / model.fit / x_new slider 定义 / eff_x_new 解析 /
chart 计算 / panel 计算 / 本 accordion → 共 13 计算 cell + 1 accordion = 14 个 null
                """
            )
        },
        multiple=False,
    )
    return


if __name__ == "__main__":
    app.run()
