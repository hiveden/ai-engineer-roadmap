"""
3D 切豆腐 · N 维特征空间的决策树切分

外卖案例升级到 3 个特征：评分 × 距离 × 价格 → 点 / 不点
2D 一刀切是"切线"，3D 一刀切是"切面"，N 维就是切超平面。

核心隐喻：决策树的"刀"在任意维度都是轴对齐超平面。
跑：cd 01-ML/04-DecisionTree/01-intro/demos && marimo run 02-cube-3d.py --headless --port 2761 --no-token
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/02-cube-3d.grid.json",
    css_file="custom.css",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import altair as alt
    import pandas as pd
    import plotly.graph_objects as go
    return alt, go, mo, np, pd


@app.cell
def _(np, pd):
    # ===== 3D 外卖店数据：评分 / 距离 / 价格 → 点(1) / 不点(0) =====
    # 真值规则：score = 0.5·评分_norm + 0.3·距离反_norm + 0.2·价格反_norm > 0.5
    # 评分(50%) > 距离(30%) > 价格(20%) → 三特征对应"重要性递减"
    _rng = np.random.default_rng(11)
    N = 60
    _ratings = _rng.uniform(3.0, 5.0, N)
    _distances = _rng.uniform(0.5, 8.0, N)
    _prices = _rng.uniform(20, 100, N)
    _r_norm = (_ratings - 3.0) / 2.0
    _d_norm = (8.0 - _distances) / 7.5
    _p_norm = (100.0 - _prices) / 80.0
    _score = 0.5 * _r_norm + 0.3 * _d_norm + 0.2 * _p_norm
    _noise = _rng.normal(0, 0.05, N)
    _labels = (_score + _noise > 0.50).astype(int)

    # 第 4 特征：头像色（0=红/1=黄/2=蓝/3=绿，平台美工随手挑 → 纯噪声）
    _bg_color = _rng.choice([0, 1, 2, 3], N).astype(int)

    df = pd.DataFrame({
        "评分": _ratings,
        "距离": _distances,
        "价格": _prices,
        "头像色": _bg_color.astype(float),
        "label": _labels,
    })
    df["类别"] = df["label"].map({1: "点", 0: "不点"})
    return (df,)


@app.cell(hide_code=True)
def title(effective_cuts, info_gain, info_gain_d, mo, shot):
    # 标题随 shot 联动 + 显示当前切面 / 信息增益（D 镜头用累计增益）
    _gain = info_gain_d if shot.value.startswith("D") else info_gain
    _gain_label = "累计信息增益" if shot.value.startswith("D") else "信息增益"
    _gain_color = (
        "#16a34a" if _gain > 0.15
        else ("#f59e0b" if _gain > 0.05 else "#94a3b8")
    )
    _verdict = (
        "切得干净" if _gain > 0.15
        else "切得还行" if _gain > 0.05
        else "几乎白切"
    )
    _shot_label = shot.value.split(" · ")[0]
    _cut_str = " ∧ ".join(effective_cuts) if effective_cuts else "—"
    mo.md(
        f"### 🎬 {_shot_label} · {_cut_str} · "
        f"<span style='color:{_gain_color}'>{_gain_label} {_gain:.3f}</span>"
        f" · {_verdict}"
    ).style(margin="0", padding="4px 12px", font_size="15px", line_height="1.3")
    return


@app.cell
def _(mo):
    # ===== 控件定义 =====
    _S = dict(show_value=True, full_width=True)
    feat = mo.ui.radio(
        options=["评分", "距离", "价格", "头像色"],
        value="评分",
        label=None,
        inline=False,
    )
    rating_thr = mo.ui.slider(3.0, 5.0, step=0.1, value=4.5, label="评分 ≤", **_S)
    dist_thr = mo.ui.slider(0.5, 8.0, step=0.5, value=3.5, label="距离 ≤", **_S)
    price_thr = mo.ui.slider(20.0, 100.0, step=5.0, value=60.0, label="价格 ≤", **_S)
    bg_thr = mo.ui.slider(0.0, 3.0, step=1.0, value=1.5, label="头像色 ≤", **_S)

    shot = mo.ui.dropdown(
        options=[
            "A · 切评分平面",
            "B · 切距离平面",
            "C · 切价格平面",
            "D · 两刀连切（4 区）",
            "E · 切头像色（白切）",
            "F · 自定义",
        ],
        value="A · 切评分平面",
        label="🎬 镜头",
    )
    return bg_thr, dist_thr, feat, price_thr, rating_thr, shot


@app.cell
def controls(bg_thr, dist_thr, feat, mo, price_thr, rating_thr):
    # 录屏区 · 左 sidebar
    _h = lambda s: mo.md(s).style(
        margin="0", padding="0", font_size="11px",
        font_weight="700", color="#6b7280", letter_spacing="0.05em",
    )
    _div = mo.md("").style(
        border_top="1px solid #e5e7eb", margin="4px 0", padding="0", height="1px",
    )
    mo.vstack(
        [_h("特征"), feat, _div, _h("阈值"), rating_thr, dist_thr, price_thr, bg_thr],
        gap=0,
        align="stretch",
    )
    return


@app.cell
def _(bg_thr, dist_thr, feat, price_thr, rating_thr, shot):
    # ===== effective: shot preset 锁定 / F 自定义释放 =====
    # primary_feat / primary_thr: 主切面（plotly + 纯度 bar 计算依据）
    if shot.value.startswith("A"):
        primary_feat, primary_thr = "评分", 4.5
        effective_cuts = ["评分 ≤ 4.5"]
    elif shot.value.startswith("B"):
        primary_feat, primary_thr = "距离", 3.5
        effective_cuts = ["距离 ≤ 3.5"]
    elif shot.value.startswith("C"):
        primary_feat, primary_thr = "价格", 60.0
        effective_cuts = ["价格 ≤ 60"]
    elif shot.value.startswith("D"):
        # 两刀连切：评分 4.5 + 距离 3.5（直觉特征顺序）
        primary_feat, primary_thr = "评分", 4.5
        effective_cuts = ["评分 ≤ 4.5", "距离 ≤ 3.5"]
    elif shot.value.startswith("E"):
        primary_feat, primary_thr = "头像色", 1.5
        effective_cuts = ["头像色 ≤ 1.5（红+黄 vs 蓝+绿）"]
    else:
        # F 自定义
        _slider_map = {
            "评分": rating_thr, "距离": dist_thr,
            "价格": price_thr, "头像色": bg_thr,
        }
        primary_feat = feat.value
        primary_thr = float(_slider_map[primary_feat].value)
        effective_cuts = [f"{primary_feat} ≤ {primary_thr:.1f}"]
    return effective_cuts, primary_feat, primary_thr


@app.cell
def _(df, primary_feat, primary_thr):
    # ===== 单刀基尼 / 信息增益（A/B/C/E 用主切面计算）=====
    _x = df[primary_feat].values
    _y = df["label"].values
    mask_left = _x <= primary_thr

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
def _(df, np):
    # ===== D 镜头：两刀连切 4 区域 + 加权基尼 =====
    # 评分 ≤ 4.5 + 距离 ≤ 3.5 把 3D 空间切 4 块
    _r = df["评分"].values
    _d = df["距离"].values
    _y = df["label"].values

    def _gini(arr):
        if len(arr) == 0:
            return 0.0
        p = float(arr.mean())
        return 2 * p * (1 - p)

    _quads = {
        "评分≤4.5 ∧ 距离≤3.5": (_r <= 4.5) & (_d <= 3.5),
        "评分≤4.5 ∧ 距离>3.5": (_r <= 4.5) & (_d > 3.5),
        "评分>4.5 ∧ 距离≤3.5": (_r > 4.5) & (_d <= 3.5),
        "评分>4.5 ∧ 距离>3.5": (_r > 4.5) & (_d > 3.5),
    }
    quad_stats = []
    _g_after = 0.0
    for _name, _mask in _quads.items():
        _n = int(_mask.sum())
        if _n == 0:
            quad_stats.append((_name, 0, 0, 0, 0.0))
            continue
        _p1 = int((_y[_mask] == 1).sum())
        _p0 = _n - _p1
        _g = _gini(_y[_mask])
        quad_stats.append((_name, _n, _p1, _p0, _g))
        _g_after += (_n / len(_y)) * _g

    _g_parent_d = _gini(_y)
    info_gain_d = _g_parent_d - _g_after
    return info_gain_d, quad_stats


@app.cell
def _(df, go, np, primary_feat, primary_thr, shot):
    # ===== fig_3d · plotly 3D 散点 + 当前切分平面 =====
    # 第 4 维"头像色"通过 marker symbol 区分 4 类：● 红 / ■ 黄 / ◆ 蓝 / ✕ 绿
    _r = df["评分"].values
    _d = df["距离"].values
    _p = df["价格"].values
    _l = df["label"].values
    _bg = df["头像色"].values.astype(int)

    fig_3d = go.Figure()

    # 拆 4 组 trace 按头像色：保留单一 label color = 类别（点/不点）
    _bg_specs = [
        (0, "circle", "🔴 红"),
        (1, "square", "🟡 黄"),
        (2, "diamond", "🔵 蓝"),
        (3, "cross", "🟢 绿"),
    ]
    for _bg_val, _sym, _bg_label in _bg_specs:
        _mask = _bg == _bg_val
        if _mask.sum() == 0:
            continue
        _colors = ["#16a34a" if _l[i] == 1 else "#dc2626" for i in range(len(_l)) if _mask[i]]
        fig_3d.add_trace(go.Scatter3d(
            x=_r[_mask], y=_d[_mask], z=_p[_mask],
            mode="markers",
            marker=dict(
                size=5, symbol=_sym, color=_colors,
                line=dict(color="white", width=1.2), opacity=0.9,
            ),
            text=[
                f"评分 {a:.1f} / 距离 {b:.1f}km / 价格 {c:.0f}元 / "
                f"头像色 {_bg_label} → {'点' if v==1 else '不点'}"
                for a, b, c, v in zip(_r[_mask], _d[_mask], _p[_mask], _l[_mask])
            ],
            hoverinfo="text",
            name=_bg_label,
            showlegend=False,
        ))

    # 切分平面（轴对齐 + 半透明）
    _R_DOM = [3.0, 5.0]
    _D_DOM = [0.5, 8.0]
    _P_DOM = [20.0, 100.0]

    def _plane(feat, thr, color):
        if feat == "评分":
            yy, zz = np.meshgrid(_D_DOM, _P_DOM)
            xx = np.full_like(yy, thr, dtype=float)
        elif feat == "距离":
            xx, zz = np.meshgrid(_R_DOM, _P_DOM)
            yy = np.full_like(xx, thr, dtype=float)
        else:  # 价格
            xx, yy = np.meshgrid(_R_DOM, _D_DOM)
            zz = np.full_like(xx, thr, dtype=float)
        return go.Surface(
            x=xx, y=yy, z=zz,
            colorscale=[[0, color], [1, color]],
            opacity=0.35, showscale=False, hoverinfo="skip", name=f"{feat} ≤ {thr}",
        )

    if shot.value.startswith("D"):
        # 两刀：评分 4.5 + 距离 3.5
        fig_3d.add_trace(_plane("评分", 4.5, "#dc2626"))
        fig_3d.add_trace(_plane("距离", 3.5, "#2563eb"))
    elif primary_feat == "头像色":
        # 头像色是离散第 4 维，3D 空间无切面 → 不画
        pass
    else:
        # 单刀：主切面
        fig_3d.add_trace(_plane(primary_feat, primary_thr, "#dc2626"))

    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(title="评分（星）", range=_R_DOM),
            yaxis=dict(title="距离（km）", range=_D_DOM),
            zaxis=dict(title="价格（元）", range=_P_DOM),
            camera={"eye": {"x": 1.6, "y": -1.6, "z": 0.9}},
            aspectmode="cube",
        ),
        font=dict(family="PingFang SC, Heiti SC, Arial Unicode MS, DejaVu Sans"),
        height=460,
        margin=dict(l=0, r=0, b=0, t=10),
        showlegend=False,
    )
    return (fig_3d,)


@app.cell
def _(alt, p0_left, p0_right, p1_left, p1_right, pd, quad_stats, shot):
    # ===== chart_purity · 单刀左/右 stacked bar 或 D 四区 =====
    if shot.value.startswith("D"):
        # 4 区域比例
        _rows = []
        _short = {
            "评分≤4.5 ∧ 距离≤3.5": "高分近店",
            "评分≤4.5 ∧ 距离>3.5": "高分远店",
            "评分>4.5 ∧ 距离≤3.5": "低分近店",
            "评分>4.5 ∧ 距离>3.5": "低分远店",
        }
        # 注意：上述 short 的"高/低分"实际反了——评分≤4.5 是低分，>4.5 是高分。修正：
        _short = {
            "评分≤4.5 ∧ 距离≤3.5": "①低分·近",
            "评分≤4.5 ∧ 距离>3.5": "②低分·远",
            "评分>4.5 ∧ 距离≤3.5": "③高分·近",
            "评分>4.5 ∧ 距离>3.5": "④高分·远",
        }
        for _name, _n, _p1, _p0, _g in quad_stats:
            _sname = _short.get(_name, _name)
            _rows.append({"side": _sname, "类别": "点", "n": _p1})
            _rows.append({"side": _sname, "类别": "不点", "n": _p0})
        _df_bar = pd.DataFrame(_rows)
        _sort = ["③高分·近", "④高分·远", "①低分·近", "②低分·远"]
    else:
        _rows = [
            {"side": "左 (≤)", "类别": "点", "n": p1_left},
            {"side": "左 (≤)", "类别": "不点", "n": p0_left},
            {"side": "右 (>)", "类别": "点", "n": p1_right},
            {"side": "右 (>)", "类别": "不点", "n": p0_right},
        ]
        _df_bar = pd.DataFrame(_rows)
        _sort = ["左 (≤)", "右 (>)"]

    chart_purity = (
        alt.Chart(_df_bar)
        .mark_bar(stroke="white", strokeWidth=1.5)
        .encode(
            y=alt.Y("side:N", title=None, sort=_sort),
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
        .properties(width=320, height=380)
    )
    return (chart_purity,)


@app.cell
def stage(chart_purity, fig_3d, mo):
    # 🎬 中央舞台 · 双槽：左 plotly 3D / 右纯度 bar
    _slot1 = mo.ui.plotly(fig_3d)
    _slot2 = mo.ui.altair_chart(chart_purity)
    mo.hstack([_slot1, _slot2], gap=0.5, widths=[1.4, 1], align="start")
    return


@app.cell
def _(
    effective_cuts,
    g_left,
    g_parent,
    g_right,
    info_gain,
    info_gain_d,
    mo,
    n_left,
    n_right,
    n_total,
    p0_left,
    p0_right,
    p1_left,
    p1_right,
    quad_stats,
    shot,
):
    # ===== panel · 单刀 vs D 四区 不同显示 =====
    _is_D = shot.value.startswith("D")
    _gain_used = info_gain_d if _is_D else info_gain
    _gain_color = (
        "#16a34a" if _gain_used > 0.15
        else ("#f59e0b" if _gain_used > 0.05 else "#94a3b8")
    )
    _verdict = (
        "✓ 干净" if _gain_used > 0.15
        else "△ 一般" if _gain_used > 0.05
        else "✗ 白切"
    )

    if _is_D:
        # 4 区基尼摘要
        _quad_html = " · ".join([
            f"<span>{_name.replace('评分','R').replace('距离','D').replace(' ∧ ','·')[:28]} "
            f"<code>n={_n}, g={_g:.2f}</code></span>"
            for (_name, _n, _p1, _p0, _g) in quad_stats
        ])
        _content = f"""
  <span><b>两刀连切</b> {' ∧ '.join(effective_cuts)}</span>
  <span><b>切前</b> {n_total} 个 · 基尼 <code>{g_parent:.3f}</code></span>
  <span style="margin-left:auto"><b>累计信息增益</b>
        <span style="color:{_gain_color};font-size:15px;font-weight:700">{_gain_used:.3f}</span>
        · {_verdict}</span>
        """
    else:
        _content = f"""
  <span><b>当前切面</b> {effective_cuts[0]}</span>
  <span><b>切前</b> {n_total} 个 · 基尼 <code>{g_parent:.3f}</code></span>
  <span>→ <span style="color:#2563eb;font-weight:600">左 {n_left}</span>（点 {p1_left} / 不点 {p0_left}）·
        基尼 <code>{g_left:.3f}</code></span>
  <span><span style="color:#dc2626;font-weight:600">右 {n_right}</span>（点 {p1_right} / 不点 {p0_right}）·
        基尼 <code>{g_right:.3f}</code></span>
  <span style="margin-left:auto"><b>信息增益</b>
        <span style="color:{_gain_color};font-size:15px;font-weight:700">{_gain_used:.3f}</span>
        · {_verdict}</span>
        """

    panel = mo.md(
        f"""<div style="font-family:ui-monospace,monospace; font-size:13px; line-height:1.45;
        background:#f9fafb; border:1px solid #e5e7eb; border-radius:6px;
        padding:6px 12px; margin:0; display:flex; gap:18px; flex-wrap:wrap; align-items:center;">
{_content}
</div>"""
    )
    return (panel,)


@app.cell
def panel_view(panel):
    panel
    return


@app.cell
def truth_hint(mo):
    # 真值提示（录屏外）
    mo.md(
        """<div style="background:#dbeafe;color:#1e40af;
            padding:6px 12px;border-radius:6px;font-size:12px;line-height:1.45;">
        🎯 <b>真值规则</b>：score = 0.5·评分 + 0.3·距离反 + 0.2·价格反 > 0.5 → 点 ·
        <b>"头像色"（红/黄/蓝/绿，平台美工随手挑）= 第 4 个纯噪声特征</b>（散点 ● 红 / ■ 黄 / ◆ 蓝 / ✕ 绿）·
        信息增益 A &gt; B &gt; C ≫ E
        </div>"""
    )
    return


@app.cell
def shot_picker(shot):
    # 镜头切换器（录屏外）
    shot
    return


@app.cell
def narration(mo, shot):
    # 口播稿（录屏外）
    _scripts = {
        "A": """
**🎬 A · 切评分平面**

> "决策树到 N 维空间，'刀'就是**轴对齐超平面**——
>  3D 里是平面，4D 里是体，依此类推，但永远沿坐标轴。
>  现在切评分=4.5，得到一个**垂直评分轴**的红色平面，把 3D 立方体切两半。
>  右半（评分>4.5）大多绿点，左半混杂——和 2D 单刀一样，**只是从'切线'升维到'切面'**。"

🔑 N 维决策树的本质：每次问一个特征，等价于沿那根轴砍一刀超平面。
""",
        "B": """
**🎬 B · 切距离平面**

> "换距离切——平面**垂直距离轴**（Y 方向）。
>  距离近的半边（左下）偏绿，距离远的半边偏红，但区分度不如评分。
>  信息增益小一点：**距离对'点不点'的贡献小于评分**。"

🎯 第二位特征 → 树的第二层会用到。
""",
        "C": """
**🎬 C · 切价格平面**

> "再换价格切——平面**垂直价格轴**（Z 方向）。
>  增益更小，但**仍非零**——价格便宜的店稍微更可能下单。
>  这是树的第三层特征：评分先切，距离次切，价格补刀。"

📐 决策树第 3 层之后，每个叶子样本数已经很少，再切容易过拟合 → 引出**剪枝**。
""",
        "D": """
**🎬 D · 两刀连切（4 区）**

> "两刀连续：先切评分 4.5（红面），再切距离 3.5（蓝面）。
>  3D 空间被切成 **4 个区域**：
>  - ③ 高分·近 → 几乎全绿（点）
>  - ① 低分·近 / ② 低分·远 → 偏红（不点）
>  - ④ 高分·远 → 仍混杂，需要再切（→ 第三刀切价格）"

🌳 这就是决策树的递归本质：每个混杂区域继续找最优特征切，直到纯。
""",
        "E": """
**🎬 E · 切头像色（白切）**

> "第 4 个特征'头像色'由平台美工随手挑（红/黄/蓝/绿）——
>  跟好不好吃 / 卫不卫生都**没有任何因果链**，纯噪声。
>  3D 空间里它**不是坐标轴**（散点形状区分：● 红 / ■ 黄 / ◆ 蓝 / ✕ 绿），
>  所以这一刀**不画切面**。看右 bar：
>  '红+黄' vs '蓝+绿' 两组里点的比例几乎一样 → **信息增益≈0**。"

🚫 决策树训练时自动跳过这种特征 → 真实树**不会出现"头像色"节点**。
🆚 区分："头像色"= 纯噪声；"是否堂食"= **真信号但弱**（食安信号被评分吸收）。
""",
        "F": """
**🎬 F · 自定义**

> "拖滑块自由切。试试：
>  - 评分阈值 4.0 vs 4.5 vs 5.0 → 哪个增益最大？
>  - 切价格 80 vs 60 vs 40 → 价格阈值有'最优点'吗？
>  - 决策树训练 = **每个节点遍历所有特征 × 所有阈值**，挑增益最大的那对。"

🎯 任务：找出能比镜头 A（评分 4.5）增益更高的单刀（提示：很难）。
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
    # 录屏布局参考（折叠）
    mo.accordion(
        {
            "📐 录屏布局参考（grid 设计意图）": mo.md(
                """
**目标**：1280×720。控件 5col + plotly 3D + Altair bar 双槽。

```
   0          5                                32
0  ┌── 标题 (h=3 全宽 32) ──────────────────────┐
3  ├ 控件     │ stage（fig_3d │ chart_purity）  │
   │ h=28     │ hstack widths=[1.4,1]          │
   │ 5col     │ h=28 / 27col                   │
31 ├──── panel 底部数字面板 (h=3 全宽) ─────────┤
36 ├── shot_picker / truth_hint / narration ──┤  ← 录屏外
```

**镜头脚本**：
| # | 切分 | 教学焦点 |
|---|---|---|
| A | 评分 ≤ 4.5 | 单刀切面 · 3D 切线升维 |
| B | 距离 ≤ 3.5 | 第二特征 · 增益次之 |
| C | 价格 ≤ 60 | 第三特征 · 增益小但非零 |
| D | 评分 + 距离 | 两刀 4 区 · 递归直觉 |
| E | 头像色 ≤ 1.5 | 第 4 维噪声（纯随机色）· 白切验证（无切面）|
| F | 自定义 | 探索增益最大的单刀 |
"""
            )
        },
        multiple=False,
    )
    return


if __name__ == "__main__":
    app.run()
