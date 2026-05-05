"""
一刀切 · 决策树最关键的决定（Step 1+2 选特征 + 定阈值）

外卖案例：评分 × 距离 → 点 / 不点
拖阈值 / 切特征 / 切镜头 → 看左右子集"基尼"和"信息增益"变化

核心隐喻：切豆腐——决策树每问一题 = 沿坐标轴砍一刀。
跑：cd 01-ML/04-DecisionTree/01-intro/demos && marimo run 01-cut-once.py --headless --port 2760 --no-token
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/01-cut-once.grid.json",
    css_file="custom.css",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import altair as alt
    import pandas as pd
    return alt, mo, np, pd


@app.cell
def _(np, pd):
    # ===== 外卖店数据：评分 / 距离 / 头像色 → 点(1) / 不点(0) =====
    # 真值规则：score = 0.6·评分_norm + 0.4·距离反_norm > 0.55 → 点
    # "头像色"（0=红/1=黄/2=蓝/3=绿，平台美工随手挑）= 纯噪声特征 → C 镜头白切
    _rng = np.random.default_rng(11)
    N = 60
    _ratings = _rng.uniform(3.0, 5.0, N)
    _distances = _rng.uniform(0.5, 8.0, N)
    _r_norm = (_ratings - 3.0) / 2.0
    _d_norm = (8.0 - _distances) / 7.5
    _score = 0.6 * _r_norm + 0.4 * _d_norm
    _noise = _rng.normal(0, 0.06, N)
    _labels = (_score + _noise > 0.55).astype(int)
    _bg_color = _rng.choice([0, 1, 2, 3], N).astype(float)

    df = pd.DataFrame({
        "评分": _ratings,
        "距离": _distances,
        "头像色": _bg_color,
        "label": _labels,
    })
    df["类别"] = df["label"].map({1: "点", 0: "不点"})
    return (df,)


@app.cell(hide_code=True)
def title(effective_feat, effective_thr, info_gain, mo, shot):
    # 标题随 shot 联动 → 切 shot 时录屏区有可见反馈（_5 §2 dynamic title）
    _verdict = (
        "切得干净" if info_gain > 0.15
        else "切得还行" if info_gain > 0.05
        else "几乎白切"
    )
    _gain_color = (
        "#16a34a" if info_gain > 0.15
        else ("#f59e0b" if info_gain > 0.05 else "#94a3b8")
    )
    _shot_label = shot.value.split(" · ")[0]
    mo.md(
        f"### 🎬 {_shot_label} · {effective_feat} ≤ {effective_thr:.2f} · "
        f"<span style='color:{_gain_color}'>信息增益 {info_gain:.3f}</span>"
        f" · {_verdict}"
    ).style(margin="0", padding="4px 12px", font_size="15px", line_height="1.3")
    return


@app.cell
def _(mo):
    # ===== 控件定义（实际渲染在 controls cell）=====
    _S = dict(show_value=True, full_width=True)
    feat = mo.ui.radio(
        options=["评分", "距离", "头像色"],
        value="评分",
        label="特征",
        inline=False,
    )
    rating_thr = mo.ui.slider(3.0, 5.0, step=0.1, value=4.5, label="评分 ≤", **_S)
    dist_thr = mo.ui.slider(0.5, 8.0, step=0.5, value=3.5, label="距离 ≤", **_S)
    bg_thr = mo.ui.slider(0.0, 3.0, step=1.0, value=1.5, label="头像色 ≤", **_S)

    shot = mo.ui.dropdown(
        options=[
            "A · 切评分（好刀）",
            "B · 切距离（次刀）",
            "C · 切头像色（白切）",
            "D · 自定义",
        ],
        value="A · 切评分（好刀）",
        label="🎬 镜头",
    )
    return bg_thr, dist_thr, feat, rating_thr, shot


@app.cell
def controls(bg_thr, dist_thr, feat, mo, rating_thr):
    # 录屏区 · 左 sidebar 控件（紧凑垂直排列）
    _h = lambda s: mo.md(s).style(
        margin="0", padding="0", font_size="11px",
        font_weight="700", color="#6b7280", letter_spacing="0.05em",
    )
    _div = mo.md("").style(
        border_top="1px solid #e5e7eb", margin="4px 0", padding="0", height="1px",
    )
    mo.vstack(
        [_h("特征"), feat, _div, _h("阈值"), rating_thr, dist_thr, bg_thr],
        gap=0,
        align="stretch",
    )
    return


@app.cell
def _(bg_thr, dist_thr, feat, rating_thr, shot):
    # ===== effective_X 模式：A/B/C preset 锁定，D 释放滑块 =====
    _PRESET = {
        "A · 切评分（好刀）": ("评分", 4.5),
        "B · 切距离（次刀）": ("距离", 3.5),
        "C · 切头像色（白切）": ("头像色", 1.5),
    }
    if shot.value in _PRESET:
        effective_feat, effective_thr = _PRESET[shot.value]
    else:
        # D 自定义：用 radio + 对应 slider
        _slider_map = {
            "评分": rating_thr,
            "距离": dist_thr,
            "头像色": bg_thr,
        }
        effective_feat = feat.value
        effective_thr = float(_slider_map[effective_feat].value)
    return effective_feat, effective_thr


@app.cell
def _(df, effective_feat, effective_thr):
    # ===== 切分计算 · 基尼 + 信息增益 =====
    _x = df[effective_feat].values
    _y = df["label"].values
    mask_left = _x <= effective_thr

    def _gini(arr):
        if len(arr) == 0:
            return 0.0
        p = float(arr.mean())
        return 2 * p * (1 - p)

    n_total = len(_y)
    n_left = int(mask_left.sum())
    n_right = n_total - n_left
    g_parent = _gini(_y)
    g_left = _gini(_y[mask_left]) if n_left > 0 else 0.0
    g_right = _gini(_y[~mask_left]) if n_right > 0 else 0.0
    g_weighted = (n_left / n_total) * g_left + (n_right / n_total) * g_right
    info_gain = g_parent - g_weighted

    p1_left = int((_y[mask_left] == 1).sum()) if n_left > 0 else 0
    p0_left = n_left - p1_left
    p1_right = int((_y[~mask_left] == 1).sum()) if n_right > 0 else 0
    p0_right = n_right - p1_right
    return (
        g_left, g_parent, g_right, info_gain,
        n_left, n_right, n_total,
        p0_left, p0_right, p1_left, p1_right,
    )


@app.cell
def _(alt, df, effective_feat, effective_thr, pd):
    # ===== chart_scatter · 评分 × 距离 散点图 + 切分线 =====
    # 数据空间固定 [3,5] × [0,8.5]，所有 mark 加 clip=True
    _x_dom = [2.9, 5.1]
    _y_dom = [0, 8.5]

    _scatter = (
        alt.Chart(df)
        .mark_circle(size=180, opacity=0.88, stroke="white", strokeWidth=1.2)
        .encode(
            x=alt.X("评分:Q", scale=alt.Scale(domain=_x_dom), title="评分（星）"),
            y=alt.Y("距离:Q", scale=alt.Scale(domain=_y_dom), title="距离（km）"),
            color=alt.Color(
                "类别:N",
                scale=alt.Scale(domain=["点", "不点"], range=["#16a34a", "#dc2626"]),
                legend=alt.Legend(orient="top-right", title=None),
            ),
            tooltip=["评分:Q", "距离:Q", "头像色:Q", "类别:N"],
        )
    )

    # 切分线（评分 = 竖线 / 距离 = 横线 / 头像色 = 不在散点图坐标内 → 透明）
    if effective_feat == "评分":
        _cut = pd.DataFrame({"x": [effective_thr, effective_thr], "y": _y_dom})
        _line = (
            alt.Chart(_cut)
            .mark_line(stroke="#dc2626", strokeWidth=3, strokeDash=[6, 4], clip=True)
            .encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=_x_dom)),
                y=alt.Y("y:Q", scale=alt.Scale(domain=_y_dom)),
            )
        )
    elif effective_feat == "距离":
        _cut = pd.DataFrame({"x": _x_dom, "y": [effective_thr, effective_thr]})
        _line = (
            alt.Chart(_cut)
            .mark_line(stroke="#dc2626", strokeWidth=3, strokeDash=[6, 4], clip=True)
            .encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=_x_dom)),
                y=alt.Y("y:Q", scale=alt.Scale(domain=_y_dom)),
            )
        )
    else:
        # 头像色 = 离散特征，散点图坐标系无法画切分线 → 用 size 着色区分
        _cut = pd.DataFrame({"x": [3.0], "y": [4.0]})  # placeholder
        _line = alt.Chart(_cut).mark_point(opacity=0).encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=_x_dom)),
            y=alt.Y("y:Q", scale=alt.Scale(domain=_y_dom)),
        )

    chart_scatter = (_scatter + _line).properties(width=420, height=440).resolve_scale(color="independent")
    return (chart_scatter,)


@app.cell
def _(alt, p0_left, p0_right, p1_left, p1_right, pd):
    # ===== chart_purity · 左 / 右子集类别 stacked bar =====
    _rows = [
        {"side": "左 (≤)", "类别": "点", "n": p1_left},
        {"side": "左 (≤)", "类别": "不点", "n": p0_left},
        {"side": "右 (>)", "类别": "点", "n": p1_right},
        {"side": "右 (>)", "类别": "不点", "n": p0_right},
    ]
    _df_bar = pd.DataFrame(_rows)

    chart_purity = (
        alt.Chart(_df_bar)
        .mark_bar(stroke="white", strokeWidth=1.5)
        .encode(
            y=alt.Y("side:N", title=None, sort=["左 (≤)", "右 (>)"]),
            x=alt.X(
                "n:Q",
                title="比例",
                stack="normalize",
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color(
                "类别:N",
                scale=alt.Scale(domain=["点", "不点"], range=["#16a34a", "#dc2626"]),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=["side:N", "类别:N", "n:Q"],
        )
        .properties(width=320, height=200)
    )
    return (chart_purity,)


@app.cell
def stage(chart_purity, chart_scatter, mo):
    # 🎬 中央舞台 · Strategy B 双槽：散点（左）+ 纯度 bar（右）
    _slot1 = mo.ui.altair_chart(chart_scatter)
    _slot2 = mo.ui.altair_chart(chart_purity)
    mo.hstack([_slot1, _slot2], gap=0.5, widths=[1.3, 1], align="start")
    return


@app.cell
def _(
    effective_feat,
    effective_thr,
    g_left,
    g_parent,
    g_right,
    info_gain,
    mo,
    n_left,
    n_right,
    n_total,
    p0_left,
    p0_right,
    p1_left,
    p1_right,
):
    # ===== panel · 底部数字面板（切前 → 左/右 → 信息增益）=====
    _gain_color = (
        "#16a34a" if info_gain > 0.15
        else ("#f59e0b" if info_gain > 0.05 else "#94a3b8")
    )
    _verdict = (
        "✓ 干净" if info_gain > 0.15
        else "△ 一般" if info_gain > 0.05
        else "✗ 白切"
    )
    panel = mo.md(
        f"""
<div style="font-family:ui-monospace,monospace; font-size:13px; line-height:1.45;
        background:#f9fafb; border:1px solid #e5e7eb; border-radius:6px;
        padding:6px 12px; margin:0; display:flex; gap:18px; flex-wrap:wrap; align-items:center;">
  <span><b>当前刀</b> {effective_feat} ≤ <code>{effective_thr:.2f}</code></span>
  <span><b>切前</b> {n_total} 个 · 基尼 <code>{g_parent:.3f}</code></span>
  <span>→ <span style="color:#2563eb;font-weight:600">左 {n_left}</span>（点 {p1_left} / 不点 {p0_left}）·
        基尼 <code>{g_left:.3f}</code></span>
  <span><span style="color:#dc2626;font-weight:600">右 {n_right}</span>（点 {p1_right} / 不点 {p0_right}）·
        基尼 <code>{g_right:.3f}</code></span>
  <span style="margin-left:auto"><b>信息增益</b>
        <span style="color:{_gain_color};font-size:15px;font-weight:700">{info_gain:.3f}</span>
        · {_verdict}</span>
</div>
        """
    )
    return (panel,)


@app.cell
def panel_view(panel):
    # panel 在录屏区底部渲染
    panel
    return


@app.cell
def truth_hint(mo):
    # 真值提示（录屏外）
    mo.md(
        """<div style="background:#dbeafe;color:#1e40af;
            padding:6px 12px;border-radius:6px;font-size:12px;line-height:1.45;">
        🎯 <b>真值规则</b>：综合得分 = 0.6·评分 + 0.4·距离反 > 0.55 → 点（评分权重主导）·
        <b>"头像色"（0=红/1=黄/2=蓝/3=绿，平台美工随机挑）= 纯噪声特征</b>，验证白切
        </div>"""
    )
    return


@app.cell
def shot_picker(shot):
    # 镜头切换器（录屏外，Playwright 通过 selector 操控）
    shot
    return


@app.cell
def narration(mo, shot):
    # 口播稿（录屏外，按 shot 切换）
    _scripts = {
        "A": """
**🎬 A · 切评分 4.5（好刀）**

> "决策树第一刀挑哪个特征切，决定 80% 成败。看评分这一刀——
>  左集（评分<4.5）几乎全是不点，纯净；
>  右集（评分≥4.5）开始混杂，但已经把'下单候选'圈出来。
>  信息增益 = 切前基尼 − 加权(左基尼, 右基尼)，**这一刀价值高**。"

🎯 这是 ID3/CART 训练时**算法会自动选**的第一刀（因为它增益最大）。
""",
        "B": """
**🎬 B · 切距离 3.5（次优刀）**

> "评分外，距离也是有用特征——
>  左集（距离<3.5km）偏点；右集（>3.5km）偏不点。
>  增益不如评分大，但绝对不是噪声。
>  实际训练里，这一刀会出现在**树的第二层**（评分切完之后再用距离细分）。"

🔑 **特征顺序的根因**：评分增益更大 → 评分先切 → 距离后切。
""",
        "C": """
**🎬 C · 切头像色（白切）**

> "头像色（红/黄/蓝/绿）由平台美工随手挑——
>  跟好不好吃 / 卫不卫生 / 贵不贵都**没有任何因果链**。
>  切完左右两边的'点 vs 不点'比例几乎和切之前一样 → 信息增益≈0。
>  真实算法会自动绕开它，训练好的树里**根本不会出现'头像色'节点**。"

🚫 这就是"噪声特征"的范本——和"是否堂食"（食安信号 / 真信号但弱）截然不同。
""",
        "D": """
**🎬 D · 自定义**

> "自己拖滑块体会信息增益怎么变。
>  - 评分阈值 3.5 vs 4.5 vs 5.0 → 哪个最优？
>  - 切到极端值（评分 ≤ 3.0）→ 一边空一边全部 → 增益→0（白切）。
>  - 决策树训练 = **遍历所有特征 × 所有阈值**，挑增益最大的那对。"

🎯 任务：找出能让信息增益超过镜头 A 的组合（提示：很难，4.5 已接近最优）。
""",
    }
    _key = shot.value[0] if shot.value else "A"
    mo.md(_scripts.get(_key, "")).style(
        font_size="14px", line_height="1.55", margin="0", padding="12px 20px",
        background="#fffbeb", border_radius="8px",
        border_left="4px solid #fbbf24",
    )
    return


@app.cell
def layout_doc(mo):
    # 📐 录屏 grid 布局参考（折叠，录屏外）
    mo.accordion(
        {
            "📐 录屏布局参考（grid 设计意图）": mo.md(
                """
**目标**：1280×720（16:9 录屏）。控件 4col 左 + 主图 28col 中 + panel 底全宽。

```
   0          4                                32
0  ┌── 标题 (h=3 全宽 32) ──────────────────────┐
3  ├ 控件     │ 主图 stage（chart_scatter +     │
   │ h=28     │ chart_purity hstack 双槽）      │
   │ 4col     │ h=28 / 28col                    │
31 ├──── panel 底部数字面板 (h=3 全宽) ─────────┤
34 │（录屏区 0~36 内的留白）                    │
36 ├── shot_picker / truth_hint (h=2) ─────────┤  ← 录屏外
38 ├── narration (h=10) ───────────────────────┤
48
```

**镜头脚本**：
| # | preset | 教学焦点 |
|---|---|---|
| A | 评分 ≤ 4.5 | 好刀 · 信息增益高 |
| B | 距离 ≤ 3.5 | 次刀 · 增益中 |
| C | 头像色 ≤ 0.5 | 白切 · 增益≈0 |
| D | 自定义 | 自己探索 |
"""
            )
        },
        multiple=False,
    )
    return


if __name__ == "__main__":
    app.run()
