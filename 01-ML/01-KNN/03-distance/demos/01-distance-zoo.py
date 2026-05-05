"""
距离动物园 · 4 种距离同台对比 + 闵可夫斯基 p 演化

互动：
- 拖动两个点 A/B 看 4 种距离实时计算
- 等距线（同一距离值的点轨迹）：圆 / 菱形 / 方
- 闵可夫斯基 p 从 0.5 到 10：看形状如何演化

跑：marimo edit 01-distance-zoo.py
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/01-distance-zoo.grid.json",
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
    mo.md("""
    # 距离动物园 · 4 种距离同台对比

    > KNN 默认用**欧氏距离**（直线），但 sklearn 实际是 `metric='minkowski', p=2`——
    > 闵可夫斯基（Minkowski）通过参数 p **统一**了欧氏 / 曼哈顿 / 切比雪夫三种距离。

    | 名字 | 公式 | 几何 |
    |---|---|---|
    | **曼哈顿** L1 | $\sum \lvert a_i - b_i
    vert$ | 网格街区 |
    | **欧氏** L2 | $\sqrt{\sum (a_i - b_i)^2}$ | 直线 |
    | **切比雪夫** L∞ | $\max \lvert a_i - b_i
    vert$ | 国王步数 |
    | **闵可夫斯基** | $(\sum \lvert a_i - b_i
    vert^p)^{1/p}$ | 参数 p 统一上面三种 |
    """)
    return


@app.cell
def _(mo):
    ax = mo.ui.slider(0, 10, value=5, step=0.1, label="A.x")
    ay = mo.ui.slider(0, 10, value=5, step=0.1, label="A.y")
    bx = mo.ui.slider(0, 10, value=8, step=0.1, label="B.x")
    by = mo.ui.slider(0, 10, value=7, step=0.1, label="B.y")
    p_slider = mo.ui.slider(0.5, 10.0, value=2.0, step=0.1, label="闵可夫斯基 p")
    mo.hstack(
        [ax, ay, bx, by, p_slider],
        widths=[1, 1, 1, 1, 1.4],  # p 的中文 label 较长，多分点空间
        justify="space-around",
    )
    return ax, ay, bx, by, p_slider


@app.cell
def _(ax, ay, bx, by, np, p_slider):
    # 纯计算：4 种距离 + 当前闵可夫斯基 Lp
    a = np.array([ax.value, ay.value])
    b = np.array([bx.value, by.value])
    diff = np.abs(a - b)
    d_man = float(diff.sum())  # 曼哈顿 L1
    d_euc = float(np.sqrt((diff ** 2).sum()))  # 欧氏 L2
    d_che = float(diff.max())  # 切比雪夫 L∞
    p = p_slider.value
    d_min = float((diff ** p).sum() ** (1 / p))  # 闵可夫斯基 Lp
    return a, b, d_che, d_euc, d_man, d_min, p


@app.cell
def _(a, b, d_che, d_euc, d_man, d_min, mo, p):
    # 实时距离状态卡（独立 mo.md cell）
    _dx = abs(a[0] - b[0])
    _dy = abs(a[1] - b[1])
    if abs(p - 1.0) < 0.05:
        _equiv = "（≈ 曼哈顿）"
    elif abs(p - 2.0) < 0.05:
        _equiv = "（≈ 欧氏）"
    elif p >= 8:
        _equiv = "（→ 切比雪夫，越大越接近）"
    else:
        _equiv = ""

    mo.md(
        f"""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:6px 10px;font-size:12px;line-height:1.5;">
    <strong>实时距离</strong> · A=({a[0]:.1f}, {a[1]:.1f}) · B=({b[0]:.1f}, {b[1]:.1f})<br>
    • <span style="color:#9333ea;font-weight:600;">曼哈顿 L1</span>: |Δx|+|Δy| = {_dx:.1f}+{_dy:.1f} = <code>{d_man:.3f}</code><br>
    • <span style="color:#16a34a;font-weight:600;">欧氏 L2</span>: √(Δx²+Δy²) = √({_dx**2:.2f}+{_dy**2:.2f}) = <code>{d_euc:.3f}</code><br>
    • <span style="color:#dc2626;font-weight:600;">切比雪夫 L∞</span>: max(|Δx|,|Δy|) = <code>{d_che:.3f}</code><br>
    • <span style="color:#2563eb;font-weight:600;">闵可夫斯基 L{p:.1f}</span>: (|Δx|^{p:.1f}+|Δy|^{p:.1f})^(1/{p:.1f}) = <code>{d_min:.3f}</code> {_equiv}
    </div>
        """
    )
    return


@app.cell
def _(d_che, d_euc, d_man, mo):
    # 左图标题
    mo.md(f"<div style='font-size:13px;font-weight:600;color:#334155;'>距离路径 · <span style='color:#9333ea'>紫={d_man:.2f}</span> · <span style='color:#16a34a'>绿={d_euc:.2f}</span> · <span style='color:#dc2626'>红={d_che:.2f}</span></div>")
    return


@app.cell
def _(a, alt, b, d_che, d_euc, d_man, pd):
    # 左图：A、B + 三种距离路径（曼哈顿 L 形、欧氏直线、切比雪夫对角）
    _df_pts = pd.DataFrame({
        "x": [a[0], b[0]],
        "y": [a[1], b[1]],
        "label": ["A", "B"],
        "color": ["#dc2626", "#2563eb"],
    })

    # 曼哈顿路径：A → (B.x, A.y) → B（紫色 L 形）
    _man_path = pd.DataFrame({
        "x": [a[0], b[0], b[0]],
        "y": [a[1], a[1], b[1]],
        "idx": [0, 1, 2],
    })
    # 欧氏路径：A → B 直线（绿）
    _euc_path = pd.DataFrame({
        "x": [a[0], b[0]],
        "y": [a[1], b[1]],
        "idx": [0, 1],
    })
    # 切比雪夫路径：取 max 维度，沿该维度直走（红色，含对角段）
    if abs(a[0] - b[0]) >= abs(a[1] - b[1]):
        _sx = 1 if b[0] > a[0] else -1
        _diag_end_x = a[0] + _sx * abs(a[1] - b[1])
        _diag_end_y = b[1]
    else:
        _sy = 1 if b[1] > a[1] else -1
        _diag_end_x = b[0]
        _diag_end_y = a[1] + _sy * abs(a[0] - b[0])
    _che_path = pd.DataFrame({
        "x": [a[0], _diag_end_x, b[0]],
        "y": [a[1], _diag_end_y, b[1]],
        "idx": [0, 1, 2],
    })

    # 自适应坐标系：A 居中 + W = max(d_*) * 1.25
    # 与右图 _base_x/_base_y 算法完全一致 → 左右两图坐标系同步、视觉对齐
    _W = max(d_man, d_euc, d_che, 0.5) * 1.25
    _base_x = alt.Scale(domain=[a[0] - _W, a[0] + _W])
    _base_y = alt.Scale(domain=[a[1] - _W, a[1] + _W])

    _man_line = alt.Chart(_man_path).mark_line(stroke="#9333ea", strokeWidth=2.5, opacity=0.7, clip=True).encode(
        x=alt.X("x:Q", scale=_base_x, title="x"),
        y=alt.Y("y:Q", scale=_base_y, title="y"),
        order="idx:Q",
    )
    _euc_line = alt.Chart(_euc_path).mark_line(stroke="#16a34a", strokeWidth=2.5, opacity=0.85, clip=True).encode(
        x=alt.X("x:Q", scale=_base_x), y=alt.Y("y:Q", scale=_base_y), order="idx:Q",
    )
    _che_line = alt.Chart(_che_path).mark_line(
        stroke="#dc2626", strokeWidth=2.5, strokeDash=[5, 3], opacity=0.7, clip=True,
    ).encode(x=alt.X("x:Q", scale=_base_x), y=alt.Y("y:Q", scale=_base_y), order="idx:Q")

    _pts = alt.Chart(_df_pts).mark_circle(size=180, stroke="black", strokeWidth=1, clip=True).encode(
        x=alt.X("x:Q", scale=_base_x), y=alt.Y("y:Q", scale=_base_y),
        color=alt.Color("label:N", scale=alt.Scale(domain=["A", "B"], range=["#dc2626", "#2563eb"]), legend=None),
        tooltip=["label:N", "x:Q", "y:Q"],
    )
    _text = alt.Chart(_df_pts).mark_text(dx=10, dy=-8, fontSize=11, fontWeight="bold", clip=True).encode(
        x=alt.X("x:Q", scale=_base_x), y=alt.Y("y:Q", scale=_base_y), text="label:N",
        color=alt.Color("label:N", scale=alt.Scale(domain=["A", "B"], range=["#dc2626", "#2563eb"]), legend=None),
    )

    _chart_paths = (_man_line + _che_line + _euc_line + _pts + _text).properties(
        width=220, height=220,
    )
    _chart_paths
    return


@app.cell
def _(mo, p):
    # 右图标题
    mo.md(f"<div style='font-size:13px;font-weight:600;color:#334155;'>等距线 L{p:.1f} · <span style='color:#2563eb'>蓝=当前</span> · <span style='color:#94a3b8'>淡=L1/L2/L∞ 参考</span></div>")
    return


@app.cell
def _(a, alt, d_che, d_euc, d_man, d_min, np, p, pd):
    # 右图：以 A 为圆心、半径 = 闵可夫斯基距离（当前 p）的等距线
    # 等距集合：{q : ||q - a||_p = d_min}
    # 对每个角度 θ 解 r 满足 ((|cos|·r)^p + (|sin|·r)^p)^(1/p) = d_min
    # 即 r = d_min / (|cos θ|^p + |sin θ|^p)^(1/p)
    _thetas = np.linspace(0, 2 * np.pi, 240)
    _cos_t = np.cos(_thetas)
    _sin_t = np.sin(_thetas)
    _denom = (np.abs(_cos_t) ** p + np.abs(_sin_t) ** p) ** (1 / p)
    _r_at = d_min / np.maximum(_denom, 1e-9)
    _df_iso = pd.DataFrame({
        "x": a[0] + _r_at * _cos_t,
        "y": a[1] + _r_at * _sin_t,
        "idx": np.arange(len(_thetas)),
    })

    _df_pts = pd.DataFrame({
        "x": [a[0]],
        "y": [a[1]],
        "label": ["A"],
    })

    # 自适应坐标系：A 居中 + 半径取所有距离上界（含 p<1 凹星形）
    # → A/B 距离小（state 2 d=0.51）也能看清等距线形状；p 拖动时 W 由 d_man 主导，坐标系基本不跳
    _W = max(d_man, d_euc, d_che, d_min) * 1.25
    _base_x = alt.Scale(domain=[a[0] - _W, a[0] + _W])
    _base_y = alt.Scale(domain=[a[1] - _W, a[1] + _W])

    def _iso_for(p_ref, d_ref, color, dash):
        _dr = (np.abs(_cos_t) ** p_ref + np.abs(_sin_t) ** p_ref) ** (1 / p_ref)
        _rr = d_ref / np.maximum(_dr, 1e-9)
        _df = pd.DataFrame({
            "x": a[0] + _rr * _cos_t,
            "y": a[1] + _rr * _sin_t,
            "idx": np.arange(len(_thetas)),
        })
        return alt.Chart(_df).mark_line(stroke=color, strokeWidth=1.0, strokeDash=dash, opacity=0.4, clip=True).encode(
            x=alt.X("x:Q", scale=_base_x), y=alt.Y("y:Q", scale=_base_y), order="idx:Q",
        )

    _iso_p1 = _iso_for(1, d_man, "#9333ea", [3, 3])
    _iso_p2 = _iso_for(2, d_euc, "#16a34a", [3, 3])
    _iso_pinf = _iso_for(20, d_che, "#dc2626", [3, 3])

    _iso_main = alt.Chart(_df_iso).mark_line(
        stroke="#2563eb", strokeWidth=3, opacity=0.95, clip=True,
    ).encode(
        x=alt.X("x:Q", scale=_base_x, title="x"),
        y=alt.Y("y:Q", scale=_base_y, title="y"),
        order="idx:Q",
    )

    _pts_iso = alt.Chart(_df_pts).mark_circle(size=200, stroke="black", strokeWidth=1, clip=True).encode(
        x=alt.X("x:Q", scale=_base_x), y=alt.Y("y:Q", scale=_base_y),
        color=alt.value("#dc2626"),
    )

    _chart_iso = (_iso_p1 + _iso_p2 + _iso_pinf + _iso_main + _pts_iso).properties(
        width=220, height=220,
    )
    _chart_iso
    return


@app.cell
def _(mo):
    # 演化图标题
    mo.md("""
    <div style='font-size:13px;font-weight:600;color:#334155;'>单位等距线随 p 演化 · p=0.5 星 → 1 菱 → 2 圆 → ∞ 方</div>
    """)
    return


@app.cell
def _(alt, np, pd):
    # 第三个图：单独看等距线随 p 演化
    # 固定半径 1，以原点为圆心，画 p=0.5, 1, 1.5, 2, 3, 5, 20 七条线
    _thetas = np.linspace(0, 2 * np.pi, 360)
    _cos_t = np.cos(_thetas)
    _sin_t = np.sin(_thetas)

    _rows = []
    for _p_val in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 20.0]:
        _denom = (np.abs(_cos_t) ** _p_val + np.abs(_sin_t) ** _p_val) ** (1 / _p_val)
        _r = 1.0 / np.maximum(_denom, 1e-9)
        for _i in range(len(_thetas)):
            _rows.append({
                "p": f"p={_p_val}",
                "p_val": _p_val,
                "x": _r[_i] * _cos_t[_i],
                "y": _r[_i] * _sin_t[_i],
                "idx": _i,
            })
    _df_evo = pd.DataFrame(_rows)

    _palette = alt.Scale(
        domain=["p=0.5", "p=1.0", "p=1.5", "p=2.0", "p=3.0", "p=5.0", "p=20.0"],
        range=["#a78bfa", "#9333ea", "#0ea5e9", "#16a34a", "#f59e0b", "#ea580c", "#dc2626"],
    )

    alt.Chart(_df_evo).mark_line(strokeWidth=2.0).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[-1.4, 1.4]), title="x"),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[-1.4, 1.4]), title="y"),
        color=alt.Color("p:N", scale=_palette, legend=alt.Legend(title="p 参数")),
        order="idx:Q",
        detail="p:N",
    ).properties(
        width=300, height=220,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### 3 张图怎么读

    1. **左上图（距离路径）**：A 到 B 的三条物理路径
    - **紫 L 形**：曼哈顿——只能横竖走（街区出租车）
    - **绿直线**：欧氏——可以斜着走（飞鸟视角）
    - **红虚对角**：切比雪夫——国王走棋（一步可以斜着走，但单步距离=max(|Δx|,|Δy|)）

    2. **右上图（等距线）**：以 A 为圆心，距离 = 当前 L_p 的所有点的轨迹
    - p=1 → 菱形（◇）
    - p=2 → 圆（○）
    - p→∞ → 正方形（□）
    - 拖 p 滑块看蓝色实线在三者之间过渡

    3. **底图（演化全谱）**：固定半径=1，p 从 0.5 到 20 的形状演化
    - p<1：星形（凹陷）—— 数学上是有效距离但不满足三角不等式（"伪距离"）
    - p=1：菱形顶点对齐坐标轴
    - p=2：完美圆
    - p>2：圆角方
    - p→∞：标准方
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "三种距离的几何直觉": mo.md(
                "**曼哈顿**：你在曼哈顿打车，司机只能沿街区走。从 (0,0) 到 (3,4)，开车 |3|+|4|=7 个街区，"
                "不管路径怎么绕，距离都是 7（只要不走回头路）。\n\n"
                "**欧氏**：直升机视角，直接飞过去。同样 (0,0) 到 (3,4)，飞行距离 = √(9+16) = 5。\n\n"
                "**切比雪夫**：象棋国王一步能走 8 个方向，每步距离 = 1 格（包括对角）。"
                "从 (0,0) 到 (3,4) 国王要走 max(3,4) = 4 步——前 3 步斜着走（同时缩 x 和 y），最后 1 步只缩 y。\n\n"
                "**闵可夫斯基**：把上面三种用一个公式统一。p 越大，"
                "**最大维度**对总距离的影响越大；p=∞ 时其他维度完全被忽略，只看最大那一维。"
            ),
            "什么时候用哪个？": mo.md(
                "| 距离 | 适用场景 |\n"
                "|---|---|\n"
                "| 欧氏 (L2) | 默认，特征量纲对齐时最稳 |\n"
                "| 曼哈顿 (L1) | 高维稀疏数据（特征独立性强）；离群值更鲁棒 |\n"
                "| 切比雪夫 (L∞) | 国际象棋类问题；棋盘格世界 |\n"
                "| 闵可夫斯基 | 调 p 是另一个超参（GridSearchCV 试 p∈{1, 2, 3}） |\n\n"
                "**重要陷阱**：所有距离都对**特征量纲**敏感——评分 0-10 + 票房 0-1e9 时，"
                "票房会主导一切。先标准化（StandardScaler）再算距离才有意义，这是下一章 04b-scaling 的主题。"
            ),
            "p<1 为什么算「伪距离」": mo.md(
                "数学上的距离要满足 4 条公理：\n\n"
                "1. 非负 $d(a,b) \\ge 0$\n"
                "2. 同一性 $d(a,a) = 0$\n"
                "3. 对称 $d(a,b) = d(b,a)$\n"
                "4. **三角不等式** $d(a,c) \\le d(a,b) + d(b,c)$\n\n"
                "p<1 时**三角不等式不成立**——绕路反而更短，违反「两点之间直线最短」的直觉。\n\n"
                "几何上 p<1 的等距线呈**星形凹陷**，从原点到顶点的距离 < 顶点到顶点的距离。"
                "实务中 p<1 几乎不用（KNN 不接受），仅作为数学拓展了解。"
            ),
        },
        multiple=True,
    )
    return


@app.cell
def _(mo):
    # 📐 Grid 布局参考（开发用 · 录屏隐藏）
    mo.md("""
    ## 📐 Grid 布局（16:9 · 录屏推荐）

    ```
       0                12                24
    0  ┌──────────── 控件（h=5）──────────┐
    5  ├──────── 实时距离卡（h=4）────────┤
    9  ├── 左图标题 ───┬── 右图标题 ──────┤
    10 ├── 左图 (h=20)─┼── 右图 (h=20) ───┤
    30 ├──────── 演化图标题（h=1）────────┤
    31 ├──────── 演化图（h=20）───────────┤
    51 └─ 三图说明 / accordion / 此 cell：null（录屏隐藏）
    ```

    cell 顺序（14 个）：
    1. imports（null）
    2. 顶部介绍 md（null）
    3. 控件
    4. 计算（null · 纯逻辑）
    5. 实时距离卡
    6. 左图标题
    7. 左图
    8. 右图标题
    9. 右图
    10. 演化图标题
    11. 演化图
    12. 三图怎么读（null）
    13. accordion FAQ（null）
    14. 本 cell · ASCII 参考（null）
    """)
    return


if __name__ == "__main__":
    app.run()
