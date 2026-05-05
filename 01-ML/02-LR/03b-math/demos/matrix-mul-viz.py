"""
向量 + 矩阵 · 纯数学可视化 · 03b-math 配套 demo

教学核心：矩阵乘法 = 批量加权求和 → ŷ = X·w 一口气算 4 个样本。
Strategy B 双槽：每镜头 chart + 状态卡。
  A · 向量几何：箭头图 + 点积/范数/cosθ
  B · 矩阵乘法：颜色块 A·w + 行展开公式
  C · LR 预演：颜色块 X·w + 样本预测展开

跑：
  cd 01-ML/02-LR/03b-math/demos
  marimo run --port 2763 --headless --no-token matrix-mul-viz.py
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/matrix-mul-viz.grid.json",
    css_file="custom.css",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import altair as alt
    import pandas as pd

    return alt, mo, np, pd


@app.cell(hide_code=True)
def title(mo):
    mo.md(
        r"### 向量 + 矩阵 · 数据视角 · 矩阵乘 = 批量加权求和 → $\hat{y} = X \cdot w$"
    ).style(margin="0", padding="4px 12px", font_size="15px", line_height="1.3")
    return


@app.cell
def _(mo, np):
    _S = dict(show_value=True, full_width=True)
    a_x = mo.ui.slider(-5.0, 5.0, step=0.1, value=4.0, label="aₓ", **_S)
    a_y = mo.ui.slider(-5.0, 5.0, step=0.1, value=0.0, label="aᵧ", **_S)
    b_x = mo.ui.slider(-5.0, 5.0, step=0.1, value=2.0, label="bₓ", **_S)
    b_y = mo.ui.slider(-5.0, 5.0, step=0.1, value=3.0, label="bᵧ", **_S)
    _PRESETS = {
        "普通": np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]]),
        "恒等": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        "对角": np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]]),
        "反向": np.array([[-1.0, 1.0], [1.0, -1.0], [2.0, 2.0]]),
    }
    A_preset = mo.ui.dropdown(
        options=_PRESETS, value="普通", label="A", full_width=True
    )
    w1 = mo.ui.slider(-3.0, 3.0, step=0.1, value=1.5, label="w₁", **_S)
    w2 = mo.ui.slider(-3.0, 3.0, step=0.1, value=-0.5, label="w₂", **_S)
    row_focus = mo.ui.slider(0, 2, step=1, value=0, label="行 i", **_S)
    sample_focus = mo.ui.slider(0, 3, step=1, value=0, label="样本 i", **_S)
    shot = mo.ui.dropdown(
        options=["A · 向量", "B · 矩阵乘", "C · LR 预演"],
        value="A · 向量",
        label="🎬 镜头",
    )
    return A_preset, a_x, a_y, b_x, b_y, row_focus, sample_focus, shot, w1, w2


@app.cell
def controls(
    A_preset,
    a_x,
    a_y,
    b_x,
    b_y,
    mo,
    row_focus,
    sample_focus,
    w1,
    w2,
):
    _h = lambda s: mo.md(s).style(
        margin="0", padding="0", font_size="11px",
        font_weight="700", color="#6b7280", letter_spacing="0.05em",
    )
    _div = mo.md("").style(
        border_top="1px solid #e5e7eb", margin="4px 0", padding="0", height="1px",
    )
    mo.vstack(
        [
            _h("向量"),
            a_x, a_y, b_x, b_y,
            _div,
            _h("矩阵"),
            A_preset, w1, w2,
            _div,
            _h("高亮"),
            row_focus, sample_focus,
        ],
        gap=0,
        align="stretch",
    )
    return


@app.cell
def _(a_x, a_y, b_x, b_y, np):
    a_vec = np.array([float(a_x.value), float(a_y.value)])
    b_vec = np.array([float(b_x.value), float(b_y.value)])
    dot_ab = float(np.dot(a_vec, b_vec))
    norm_a = float(np.linalg.norm(a_vec))
    norm_b = float(np.linalg.norm(b_vec))
    sum_ab = a_vec + b_vec
    if norm_a > 1e-9 and norm_b > 1e-9:
        cos_theta = dot_ab / (norm_a * norm_b)
    else:
        cos_theta = 0.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return a_vec, b_vec, cos_theta, dot_ab, norm_a, norm_b, sum_ab


@app.cell
def _(a_vec, alt, b_vec, np, pd, sum_ab):
    _arrows = pd.DataFrame([
        {"x": 0, "y": 0, "x2": a_vec[0], "y2": a_vec[1], "name": "a", "color": "#3b82f6"},
        {"x": 0, "y": 0, "x2": b_vec[0], "y2": b_vec[1], "name": "b", "color": "#ef4444"},
        {"x": 0, "y": 0, "x2": sum_ab[0], "y2": sum_ab[1], "name": "a+b", "color": "#8b5cf6"},
    ])
    _paral = pd.DataFrame([
        {"x": a_vec[0], "y": a_vec[1], "x2": sum_ab[0], "y2": sum_ab[1]},
        {"x": b_vec[0], "y": b_vec[1], "x2": sum_ab[0], "y2": sum_ab[1]},
    ])
    _norm_a_sq = float(np.dot(a_vec, a_vec))
    if _norm_a_sq > 1e-9:
        _proj_scale = float(np.dot(a_vec, b_vec)) / _norm_a_sq
        _proj_pt = a_vec * _proj_scale
    else:
        _proj_pt = np.array([0.0, 0.0])
    _proj_seg = pd.DataFrame([{"x": 0.0, "y": 0.0, "x2": float(_proj_pt[0]), "y2": float(_proj_pt[1])}])
    _perp_seg = pd.DataFrame([{"x": float(b_vec[0]), "y": float(b_vec[1]), "x2": float(_proj_pt[0]), "y2": float(_proj_pt[1])}])
    _proj_pt_df = pd.DataFrame([{"x": float(_proj_pt[0]), "y": float(_proj_pt[1]), "text": "proj"}])
    _ends = pd.DataFrame([
        {"x": a_vec[0], "y": a_vec[1], "name": "a", "color": "#3b82f6"},
        {"x": b_vec[0], "y": b_vec[1], "name": "b", "color": "#ef4444"},
        {"x": sum_ab[0], "y": sum_ab[1], "name": "a+b", "color": "#8b5cf6"},
    ])
    _all_x = [a_vec[0], b_vec[0], sum_ab[0], 0.0]
    _all_y = [a_vec[1], b_vec[1], sum_ab[1], 0.0]
    _W = max(max(abs(v) for v in _all_x + _all_y), 3.0) * 1.25

    _axis_x = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color="#9ca3af", strokeWidth=0.7, opacity=0.6)
        .encode(y="y:Q")
    )
    _axis_y = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(color="#9ca3af", strokeWidth=0.7, opacity=0.6)
        .encode(x="x:Q")
    )
    _arrow_lines = (
        alt.Chart(_arrows)
        .mark_rule(strokeWidth=3.5, opacity=0.92)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-_W, _W]), title="x"),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[-_W, _W]), title="y"),
            x2="x2:Q", y2="y2:Q",
            color=alt.Color("name:N", scale=alt.Scale(
                domain=["a", "b", "a+b"], range=["#3b82f6", "#ef4444", "#8b5cf6"]
            ), legend=alt.Legend(title="向量", orient="top-right")),
            tooltip=[alt.Tooltip("name:N"), alt.Tooltip("x2:Q", format=".2f"), alt.Tooltip("y2:Q", format=".2f")],
        )
    )
    _paral_lines = (
        alt.Chart(_paral)
        .mark_rule(color="#8b5cf6", strokeWidth=1.4, strokeDash=[5, 4], opacity=0.55)
        .encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
    )
    _proj_line = (
        alt.Chart(_proj_seg)
        .mark_rule(color="#10b981", strokeWidth=5, opacity=0.55)
        .encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
    )
    _perp_line = (
        alt.Chart(_perp_seg)
        .mark_rule(color="#9ca3af", strokeWidth=1.2, strokeDash=[3, 3], opacity=0.7)
        .encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
    )
    _proj_pt_mark = (
        alt.Chart(_proj_pt_df)
        .mark_point(shape="circle", size=80, filled=True, color="#10b981", stroke="white", strokeWidth=1)
        .encode(x="x:Q", y="y:Q", tooltip=[alt.Tooltip("text:N"), alt.Tooltip("x:Q", format=".2f"), alt.Tooltip("y:Q", format=".2f")])
    )
    _arrow_heads = (
        alt.Chart(_ends)
        .mark_point(shape="triangle", size=200, filled=True, opacity=0.95)
        .encode(
            x="x:Q", y="y:Q",
            color=alt.Color("name:N", scale=alt.Scale(
                domain=["a", "b", "a+b"], range=["#3b82f6", "#ef4444", "#8b5cf6"]
            ), legend=None),
        )
    )
    _labels = (
        alt.Chart(_ends)
        .mark_text(fontSize=15, fontWeight="bold", dx=12, dy=-8)
        .encode(
            x="x:Q", y="y:Q", text="name:N",
            color=alt.Color("name:N", scale=alt.Scale(
                domain=["a", "b", "a+b"], range=["#3b82f6", "#ef4444", "#8b5cf6"]
            ), legend=None),
        )
    )
    chart_vec = (
        (_axis_x + _axis_y + _paral_lines + _perp_line + _arrow_lines
         + _proj_line + _arrow_heads + _proj_pt_mark + _labels)
        .properties(width=480, height=400)
        .resolve_scale(color="independent")
    )
    return (chart_vec,)


@app.cell
def _(A_preset, np, w1, w2):
    A_mat = np.array(A_preset.value, dtype=float)
    w_vec = np.array([float(w1.value), float(w2.value)])
    Aw = A_mat @ w_vec
    return A_mat, Aw, w_vec


@app.cell
def _(A_mat, Aw, alt, pd, row_focus, w_vec):
    _i_focus = int(row_focus.value)
    _A_cells = []
    for _i in range(3):
        for _j in range(2):
            _A_cells.append({"col": _j, "row": _i, "val": float(A_mat[_i, _j]),
                             "block": "A", "label": f"{A_mat[_i, _j]:.2f}"})
    _w_cells = []
    for _k in range(2):
        _w_cells.append({"col": 3, "row": _k, "val": float(w_vec[_k]),
                          "block": "w", "label": f"{w_vec[_k]:.2f}"})
    _Aw_cells = []
    for _i in range(3):
        _Aw_cells.append({"col": 5, "row": _i, "val": float(Aw[_i]),
                           "block": "Aw", "label": f"{Aw[_i]:.2f}"})
    _cells_df = pd.DataFrame(_A_cells + _w_cells + _Aw_cells)

    _highlight = []
    for _j in range(2):
        _highlight.append({"col": _j, "row": _i_focus})
    for _k in range(2):
        _highlight.append({"col": 3, "row": _k})
    _highlight.append({"col": 5, "row": _i_focus})
    _hi_df = pd.DataFrame(_highlight)

    _rect = (
        alt.Chart(_cells_df)
        .mark_rect(stroke="white", strokeWidth=2)
        .encode(
            x=alt.X("col:O", axis=None),
            y=alt.Y("row:O", axis=None, sort=[0, 1, 2]),
            color=alt.Color("val:Q", scale=alt.Scale(scheme="redblue", domain=[-6, 6], reverse=True),
                            legend=alt.Legend(title="数值", orient="right")),
            tooltip=[alt.Tooltip("block:N"), alt.Tooltip("row:O"), alt.Tooltip("val:Q", format=".2f")],
        )
    )
    _text = (
        alt.Chart(_cells_df)
        .mark_text(fontSize=15, fontWeight="bold", color="#0f172a")
        .encode(x="col:O", y=alt.Y("row:O", sort=[0, 1, 2]), text="label:N")
    )
    _hi_rect = (
        alt.Chart(_hi_df)
        .mark_rect(stroke="#dc2626", strokeWidth=4, fill="transparent")
        .encode(x="col:O", y=alt.Y("row:O", sort=[0, 1, 2]))
    )
    chart_matmul = (
        (_rect + _text + _hi_rect)
        .properties(width=480, height=300, title="A (3×2) · w (2×1) = Aw (3×1)")
        .resolve_scale(color="independent")
    )
    return (chart_matmul,)


@app.cell
def _(np, w_vec):
    X_lr = np.array([
        [1.0, 2.0],
        [-1.0, 3.0],
        [2.0, -1.0],
        [0.5, 1.5],
    ])
    y_hat = X_lr @ w_vec
    return X_lr, y_hat


@app.cell
def _(X_lr, alt, pd, sample_focus, w_vec, y_hat):
    _i_focus = int(sample_focus.value)
    _X_cells = []
    for _i in range(4):
        for _j in range(2):
            _X_cells.append({"col": _j, "row": _i, "val": float(X_lr[_i, _j]),
                              "block": "X", "label": f"{X_lr[_i, _j]:.2f}"})
    _w_cells = []
    for _k in range(2):
        _w_cells.append({"col": 3, "row": _k, "val": float(w_vec[_k]),
                          "block": "w", "label": f"{w_vec[_k]:.2f}"})
    _y_cells = []
    for _i in range(4):
        _y_cells.append({"col": 5, "row": _i, "val": float(y_hat[_i]),
                          "block": "ŷ", "label": f"{y_hat[_i]:.2f}"})
    _cells_df = pd.DataFrame(_X_cells + _w_cells + _y_cells)

    _highlight = []
    for _j in range(2):
        _highlight.append({"col": _j, "row": _i_focus})
    for _k in range(2):
        _highlight.append({"col": 3, "row": _k})
    _highlight.append({"col": 5, "row": _i_focus})
    _hi_df = pd.DataFrame(_highlight)

    _rect = (
        alt.Chart(_cells_df)
        .mark_rect(stroke="white", strokeWidth=2)
        .encode(
            x=alt.X("col:O", axis=None),
            y=alt.Y("row:O", axis=None, sort=[0, 1, 2, 3]),
            color=alt.Color("val:Q", scale=alt.Scale(scheme="redblue", domain=[-6, 6], reverse=True),
                            legend=alt.Legend(title="数值", orient="right")),
            tooltip=[alt.Tooltip("block:N"), alt.Tooltip("row:O"), alt.Tooltip("val:Q", format=".2f")],
        )
    )
    _text = (
        alt.Chart(_cells_df)
        .mark_text(fontSize=15, fontWeight="bold", color="#0f172a")
        .encode(x="col:O", y=alt.Y("row:O", sort=[0, 1, 2, 3]), text="label:N")
    )
    _hi_rect = (
        alt.Chart(_hi_df)
        .mark_rect(stroke="#16a34a", strokeWidth=4, fill="transparent")
        .encode(x="col:O", y=alt.Y("row:O", sort=[0, 1, 2, 3]))
    )
    chart_lr = (
        (_rect + _text + _hi_rect)
        .properties(width=480, height=340, title="X (4×2) · w (2×1) = ŷ (4×1)")
        .resolve_scale(color="independent")
    )
    return (chart_lr,)


@app.cell
def _(Aw, cos_theta, dot_ab, mo, norm_a, np, shot, y_hat):
    _key = shot.value[0] if shot.value else "A"
    if _key == "A":
        _theta = float(np.degrees(np.arccos(cos_theta)))
        _text = (
            f"<b>向量</b> a·b={dot_ab:.3f} | ‖a‖={norm_a:.3f} "
            f"| cosθ={cos_theta:.3f}（{_theta:.1f}°）"
            f"<br><span style='color:#374151'>→ <b>点积 = 投影长 × ‖a‖</b>"
            f"；cosθ→0 时 a⊥b</span>"
        )
    elif _key == "B":
        _text = (
            f"<b>矩阵</b> Aw = ({Aw[0]:.2f}, {Aw[1]:.2f}, {Aw[2]:.2f})ᵀ"
            f"<br><span style='color:#374151'>→ <b>(Aw)ᵢ = 第 i 行 · w</b>"
            f"；红框扫到哪行 结果算到哪</span>"
        )
    else:
        _text = (
            f"<b>LR</b> ŷ = ({y_hat[0]:.2f}, {y_hat[1]:.2f}, "
            f"{y_hat[2]:.2f}, {y_hat[3]:.2f})ᵀ"
            f"<br><span style='color:#374151'>→ <b>ŷ = X·w</b>"
            f" 4 样本一口气算完 = numpy <code>X @ w</code></span>"
        )
    panel = mo.md(
        f"""<div style="font-family:ui-monospace,monospace; font-size:13px; line-height:1.45;
            background:#f9fafb; border:1px solid #e5e7eb; border-radius:6px;
            padding:6px 12px; margin:0;">
        {_text}
        </div>"""
    )
    return (panel,)


@app.cell
def truth_hint(mo):
    mo.md("""
    <div style="background:#dbeafe;color:#1e40af;border-left:4px solid #3b82f6;
    padding:6px 14px;border-radius:6px;font-size:13px;line-height:1.4;margin:0;">
    🎯 <b>数据视角</b>：矩阵=样本集（行=样本 列=特征），矩阵乘=批量加权求和
    &nbsp;·&nbsp; <b>不教</b>变换视角（3b1b）
    </div>
    """)
    return


@app.cell
def shot_picker(shot):
    shot
    return


@app.cell
def stage(
    A_mat,
    Aw,
    X_lr,
    chart_lr,
    chart_matmul,
    chart_vec,
    cos_theta,
    dot_ab,
    mo,
    norm_a,
    norm_b,
    np,
    row_focus,
    sample_focus,
    shot,
    sum_ab,
    w_vec,
    y_hat,
):
    _key = shot.value[0] if shot.value else "A"
    if _key == "A":
        _slot1 = mo.ui.altair_chart(chart_vec)
        _theta = float(np.degrees(np.arccos(cos_theta)))
        if abs(cos_theta) < 0.05:
            _rel = "≈ 垂直"
        elif cos_theta > 0.95:
            _rel = "≈ 同向"
        elif cos_theta < -0.95:
            _rel = "≈ 反向"
        else:
            _rel = f"夹角 {_theta:.1f}°"
        _slot2 = mo.md(f"""
    <div style="font-family:ui-monospace,monospace; font-size:14px; line-height:1.7; padding:12px 16px;">
    <b style="font-size:15px">📐 向量运算</b><br><br>
    <b>点积</b> a·b = <b>{dot_ab:.3f}</b><br>
    <b>范数</b> ‖a‖ = {norm_a:.3f}，‖b‖ = {norm_b:.3f}<br>
    <b>夹角</b> cosθ = {cos_theta:.3f}（{_rel}）<br>
    <b>加法</b> a+b = ({sum_ab[0]:.2f}, {sum_ab[1]:.2f})<br><br>
    <div style="background:#f0fdf4;padding:8px 12px;border-radius:6px;font-size:13px">
    🟢 <b>绿色粗线</b> = b 在 a 上的投影<br>
    投影长 × ‖a‖ = a·b（点积几何含义）
    </div>
    </div>""")

    elif _key == "B":
        _slot1 = mo.ui.altair_chart(chart_matmul)
        _i = int(row_focus.value)
        _row = A_mat[_i]
        _result = float(Aw[_i])
        _slot2 = mo.md(f"""
    <div style="font-family:ui-monospace,monospace; font-size:14px; line-height:1.7; padding:12px 16px;">
    <b style="font-size:15px">🔴 第 {_i} 行展开</b><br><br>
    行向量 = ({_row[0]:.2f}, {_row[1]:.2f})<br>
    w = ({w_vec[0]:.2f}, {w_vec[1]:.2f})ᵀ<br><br>
    (Aw)<sub>{_i}</sub> = {_row[0]:.2f}·{w_vec[0]:.2f} + {_row[1]:.2f}·{w_vec[1]:.2f}<br>
    = <b style="color:#dc2626;font-size:17px">{_result:.3f}</b><br><br>
    <div style="color:#6b7280;font-size:13px">
    完整 Aw = ({Aw[0]:.2f}, {Aw[1]:.2f}, {Aw[2]:.2f})ᵀ
    </div>
    <div style="background:#fef3c7;padding:8px 12px;border-radius:6px;font-size:13px;margin-top:8px">
    矩阵乘 = 左矩阵每行 dot 右向量<br>
    红框第 i 行 → 结果第 i 元素
    </div>
    </div>""")

    else:
        _slot1 = mo.ui.altair_chart(chart_lr)
        _i = int(sample_focus.value)
        _row = X_lr[_i]
        _yi = float(y_hat[_i])
        _all = " ; ".join(
            f'<span style="color:{"#16a34a" if k == _i else "#374151"};'
            f'font-weight:{"700" if k == _i else "400"}">'
            f"ŷ<sub>{k}</sub>={float(y_hat[k]):.2f}</span>"
            for k in range(4)
        )
        _slot2 = mo.md(f"""
    <div style="font-family:ui-monospace,monospace; font-size:14px; line-height:1.7; padding:12px 16px;">
    <b style="font-size:15px">🟢 样本 {_i} 预测</b><br><br>
    x<sub>{_i}</sub> = ({_row[0]:.2f}, {_row[1]:.2f})<br>
    w = ({w_vec[0]:.2f}, {w_vec[1]:.2f})ᵀ<br><br>
    ŷ<sub>{_i}</sub> = {w_vec[0]:.2f}·{_row[0]:.2f} + {w_vec[1]:.2f}·{_row[1]:.2f}<br>
    = <b style="color:#16a34a;font-size:17px">{_yi:.3f}</b><br><br>
    <div style="color:#6b7280;font-size:13px">
    所有预测：{_all}
    </div>
    <div style="background:#dcfce7;padding:8px 12px;border-radius:6px;font-size:13px;margin-top:8px">
    标量 ŷᵢ = w₁·xᵢ₁ + w₂·xᵢ₂<br>
    写成矩阵 → <b>ŷ = X·w</b>（一行代码）
    </div>
    </div>""")

    mo.hstack([_slot1, _slot2], gap=0.5, widths="equal", align="start")
    return


@app.cell
def panel_view(panel):
    panel
    return


@app.cell
def narration(mo, shot):
    _scripts = {
        "A": """
    **🎬 A · 向量几何**（45 秒）

    > "两个向量 a、b 从原点出发。
    >  紫色箭头 = a+b（平行四边形对角线）。
    >  绿色粗线 = b 在 a 方向上的投影段。
    >  投影长 × ‖a‖ = 点积 a·b —— 这是点积的几何含义。
    >  拖滑块让 cosθ→0 → 绿线消失 → a·b=0。"

    🔑 **桥梁**：下个镜头把'两向量点积'做 m 次 → 矩阵乘法
    """,
        "B": """
    **🎬 B · 矩阵乘法**（60 秒）

    > "矩阵 A (3×2) × 向量 w (2×1) = Aw (3×1)。
    >  就是把镜头 A 的'两向量点积'做 3 次。
    >  A 的每一行和 w 做点积 → 结果的每个元素。
    >  红框扫到第 i 行 → 结果第 i 个元素亮起。
    >  切 A 预设看不同矩阵，拖 w₁/w₂ 看结果实时变。"

    🔑 **形状对齐**：(3×2)·(2×1) = (3×1)；中间的 2 必须相等
    """,
        "C": """
    **🎬 C · LR 矩阵化预演**（60 秒）

    > "把 A 换成 X（4 样本 × 2 特征），w 不变。
    >  X·w = ŷ —— 4 个样本的预测一口气算完。
    >  标量版 ŷᵢ = w₁·xᵢ₁ + w₂·xᵢ₂ 写成矩阵就是 ŷ = X·w。
    >  这就是 numpy X @ w 比 Python for 快百倍的原因。"

    🔑 **核心**：LR 多元预测的全部 = 一次矩阵乘法
    """,
    }
    _key = shot.value[0] if shot.value else "A"
    mo.md(_scripts.get(_key, "")).style(
        font_size="15px", line_height="1.6", margin="0", padding="14px 24px",
        background="#fffbeb", border_radius="8px",
        border_left="4px solid #fbbf24",
    )
    return


@app.cell
def layout_doc(mo):
    mo.accordion(
        {
            "📐 录屏布局参考（grid 设计意图）": mo.md(
                r"""
    **目标 viewport**：1280×720（16:9 单屏不滚动）。
    左 sidebar 4col 控件 + 中右 stage 双槽 + 底 panel；narration / shot / truth_hint 全部录屏外。

    ### 横屏骨架

    ```
       0       4                              32
    y=0   ┌──── 标题 (h=3) ────────────────────────┐
    y=3   ├ controls ┬── stage 双槽 (h=26) ───────┤
      │ 4col     │ A: chart_vec | 向量状态卡    │
      │ 向量     │ B: chart_matmul | 行展开     │
      │ 矩阵     │ C: chart_lr | 样本预测展开   │
      │ 高亮     │                              │
    y=29  ├──────────┴── panel (h=3) ──────────────┤  ← 720 内
    y=36  ├──── truth_hint (录屏外) ───────────────┤
    y=39  ├──── shot dropdown (录屏外) ────────────┤
    y=42  ├──── narration 口播稿 (录屏外) ─────────┤
    ```

    ### 镜头脚本（Strategy B 双槽）

    | # | 时长 | slot1 | slot2 | 教学焦点 |
    |---|---|---|---|---|
    | **A** | 0-45s | chart_vec | 点积/范数/cosθ | 向量点积 = 投影 |
    | **B** | 45-105s | chart_matmul | 行展开公式 | 矩阵乘 = 逐行点积 |
    | **C** | 105-165s | chart_lr | 样本预测展开 | ŷ=X·w 一口气算完 |
                """
            )
        },
        multiple=False,
    )
    return


if __name__ == "__main__":
    app.run()
