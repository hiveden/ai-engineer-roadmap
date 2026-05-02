"""
KNN 接近程度 · 二维欧式距离演示

互动：拖动新电影坐标 → 实时看 8 部历史电影距离重排 → KNN 投票预测喜好
跑：marimo edit 02-proximity.py
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/02-proximity.grid.json",
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
    # KNN 接近程度 · 二维欧式距离

    用 9 部历史电影（评分 + 主演吸引度）预测你对新电影的喜好。
    拖动滑块改变新电影坐标，看**距离重排**和**邻居投票**怎么变。
    """)
    return


@app.cell
def _(np):
    # 9 部历史电影 (rating, lead_actor) + 喜好标签 (1=喜欢, 0=不喜欢)
    movies = [
        ("流浪地球2", 8.3, 9.0, 1),
        ("阿凡达",   7.8, 8.5, 1),
        ("泰坦尼克", 8.0, 9.5, 1),
        ("复联4",    8.4, 9.2, 1),
        ("沙丘2",    8.5, 8.0, 1),
        ("哥斯拉",   6.4, 7.0, 0),
        ("长津湖",   7.4, 8.0, 0),
        ("寒战",     7.2, 6.5, 0),
        ("喜羊羊",   5.5, 4.0, 0),
    ]
    names = [m[0] for m in movies]
    X = np.array([[m[1], m[2]] for m in movies])
    y = np.array([m[3] for m in movies])
    return X, names, y


@app.cell
def _(mo):
    new_rating = mo.ui.slider(4.0, 10.0, value=8.5, step=0.1, label="评分")
    new_lead = mo.ui.slider(4.0, 10.0, value=9.5, step=0.1, label="主演吸引度")
    k_slider = mo.ui.slider(1, 9, value=3, step=1, label="k 邻居数")

    mo.hstack(
        [new_rating, new_lead, k_slider],
        widths=[1, 1, 1],
        justify="space-around",
    )
    return k_slider, new_lead, new_rating


@app.cell
def _(X, k_slider, mo, new_lead, new_rating, np, y):
    query = np.array([new_rating.value, new_lead.value])
    dists = np.linalg.norm(X - query, axis=1)
    sorted_idx = np.argsort(dists)
    top_labels = y[sorted_idx[: k_slider.value]]
    likes = int((top_labels == 1).sum())
    dislikes = int((top_labels == 0).sum())
    prediction = 1 if likes > dislikes else 0

    label = "会喜欢" if prediction == 1 else "不会喜欢"
    bg = "#e8f5e9" if prediction == 1 else "#ffebee"
    bd = "#2ca02c" if prediction == 1 else "#d62728"
    txt = "#166534" if prediction == 1 else "#991b1b"

    mo.md(
        f'<div style="padding:6px 12px;background:{bg};border-left:4px solid {bd};'
        f'border-radius:4px;font-size:14px;color:{txt};">'
        f"<strong>{label}</strong> · 喜欢 {likes} / 不喜欢 {dislikes}"
        f" · 新电影 (<code>{new_rating.value:.1f}, {new_lead.value:.1f}</code>)"
        f" · k=<code>{k_slider.value}</code>"
        "</div>"
    )
    return dists, query, sorted_idx


@app.cell
def _(
    X,
    alt,
    k_slider,
    mo,
    names,
    new_lead,
    new_rating,
    pd,
    query,
    sorted_idx,
    y,
):
    df_pts = pd.DataFrame({
        "name": names,
        "rating": X[:, 0],
        "lead": X[:, 1],
        "label": ["喜欢" if lbl == 1 else "不喜欢" for lbl in y],
    })
    line_rows = []
    for line_rank, idx in enumerate(sorted_idx, start=1):
        line_rows.append({
            "x0": query[0], "y0": query[1],
            "x1": X[idx, 0], "y1": X[idx, 1],
            "kind": "top-k" if line_rank <= k_slider.value else "其他",
        })
    df_lines = pd.DataFrame(line_rows)
    df_query = pd.DataFrame({"rating": [new_rating.value], "lead": [new_lead.value]})

    base_x = alt.Scale(domain=[4, 10.5])
    base_y = alt.Scale(domain=[3, 11])

    lines_chart = alt.Chart(df_lines).mark_rule(strokeDash=[3, 2]).encode(
        x=alt.X("x0:Q", scale=base_x, title="评分"),
        y=alt.Y("y0:Q", scale=base_y, title="主演吸引度"),
        x2="x1:Q", y2="y1:Q",
        color=alt.Color("kind:N",
            scale=alt.Scale(domain=["top-k", "其他"], range=["#f59e0b", "#cccccc"]),
            legend=None),
        size=alt.Size("kind:N",
            scale=alt.Scale(domain=["top-k", "其他"], range=[2.5, 1]),
            legend=None),
    )
    pts_chart = alt.Chart(df_pts).mark_circle(size=200, stroke="black", strokeWidth=1).encode(
        x=alt.X("rating:Q", scale=base_x),
        y=alt.Y("lead:Q", scale=base_y),
        color=alt.Color("label:N",
            scale=alt.Scale(domain=["喜欢", "不喜欢"], range=["#2ca02c", "#d62728"]),
            legend=alt.Legend(title="历史标签")),
        tooltip=["name", "rating", "lead", "label"],
    )
    text_chart = alt.Chart(df_pts).mark_text(dx=10, dy=-8, fontSize=11).encode(
        x="rating:Q", y="lead:Q", text="name:N",
    )
    query_chart = alt.Chart(df_query).mark_point(
        shape="diamond", size=400, color="#1f77b4", filled=True, stroke="black", strokeWidth=1.5
    ).encode(x="rating:Q", y="lead:Q")

    chart = (lines_chart + pts_chart + text_chart + query_chart).properties(
        width=600, height=320
    )
    chart_section_style = (
        "border-left:3px solid #6366f1;padding:2px 10px;"
        "font-weight:600;font-size:14px;margin-bottom:4px;"
    )
    mo.vstack([
        mo.md(f'<div style="{chart_section_style}">距离可视化（黄色 = top-k）</div>'),
        chart,
    ])
    return


@app.cell
def _(X, dists, k_slider, mo, names, query, sorted_idx, y):
    rows = []
    for rank, i in enumerate(sorted_idx, start=1):
        if y[i] == 1:
            label_cell = (
                '<span style="background:#d4edda;color:#166534;'
                'padding:1px 8px;border-radius:3px;font-weight:600;">喜欢</span>'
            )
        else:
            label_cell = (
                '<span style="background:#f8d7da;color:#991b1b;'
                'padding:1px 8px;border-radius:3px;font-weight:600;">不喜欢</span>'
            )
        d_rating = X[i, 0] - query[0]
        d_lead = X[i, 1] - query[1]
        if rank <= k_slider.value:
            dist_cell = (
                f'<span style="background:#fef3c7;padding:1px 6px;border-radius:3px;'
                f'color:#92400e;font-weight:600;">{dists[i]:.3f}</span>'
            )
        else:
            dist_cell = f"{dists[i]:.3f}"
        rows.append(
            f"| {rank} | {names[i]} "
            f"| {X[i, 0]:.1f} | {X[i, 1]:.1f} "
            f"| {d_rating:+.1f} | {d_lead:+.1f} "
            f"| {dist_cell} | {label_cell} |"
        )

    detail_section_style = (
        "border-left:3px solid #6366f1;padding:2px 10px;"
        "font-weight:600;font-size:14px;margin-bottom:4px;"
    )
    table_md = (
        "| 排名 | 电影 | 评分 | 主演 | Δ评分 | Δ主演 | 距离 | 历史标签 |\n"
        "|---|---|---|---|---|---|---|---|\n"
        + "\n".join(rows)
        + f"\n\n> 距离 = √(Δ评分² + Δ主演²) · 黄色 = 前 k={k_slider.value} 个最近邻"
    )

    mo.vstack([
        mo.md(f'<div style="{detail_section_style}">距离明细（升序）</div>'),
        mo.md(table_md),
    ])
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "玩法建议": mo.md(
                "1. **基线**：新电影 (8.5, 9.5)，k=3 → 看 3 部最近邻全是「喜欢」\n\n"
                "2. **拖到分歧区**：把评分调到 7.5 附近，主演 7.5 → 邻居开始混杂\n\n"
                "3. **k 影响**：k=1 单点决定 → k=9 全员投票\n\n"
                "4. **喜羊羊角落**：新电影拉到 (5.5, 4.0) 附近 → 全是不喜欢\n\n"
                "5. **公式验算**：随便挑一行，距离 = √[(评分差)² + (主演差)²]"
            ),
            "数学": mo.md(
                "二维欧式距离：\n\n"
                "$$d = \\sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$$\n\n"
                "推广 N 维：$d = \\sqrt{\\sum_{i=1}^n (a_i - b_i)^2}$"
            ),
        },
        multiple=True,
    )
    return


if __name__ == "__main__":
    app.run()
