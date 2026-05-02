"""
KNN K 值调优演示 · 决策边界可视化

互动：拖 k 滑块 → 决策区域形状变化 → 看过拟合 / 欠拟合
数据：make_blobs 两簇高斯 + 6 个高分烂片噪声（N=126）
跑：marimo edit 03-k-tuning.py
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium", css_file="marimo.css")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    from sklearn.datasets import make_blobs

    return alt, make_blobs, mo, np, pd


@app.cell
def _(mo):
    mo.md("""
    # KNN K 值调优 · 决策边界可视化

    > 02 用 9 部真实电影建立距离直觉，03 切到 **126 部合成观影记录**做严肃 k 调优——
    > 真实样本太少看不出 k 的影响，需要更密的数据才能完整演示 bias-variance 谱。

    右上「高评分 + 高吸引度」是喜欢区，左下是不喜欢区，中间犬牙交错带 + 6 个"高分烂片"噪声点考验 KNN 鲁棒性。

    拖 k 滑块看决策区域形状：**k 太小** → 边界扭曲过拟合（噪声变孤岛）；
    **k 太大** → 边界过度平滑欠拟合（接近多数类塌陷）。
    """)
    return


@app.cell
def _(make_blobs, np):
    # 合成数据：两簇高斯（喜欢 / 不喜欢）+ 6 个高分烂片噪声
    X, y = make_blobs(
        n_samples=120,
        centers=[(8.5, 9.0), (6.0, 5.5)],
        cluster_std=1.3,  # 适度重叠：保持"绿背景+红孤岛"经典形态，k 影响仍可见
        random_state=42,
    )
    y = 1 - y  # 翻转：(8.5,9.0) 簇 → 喜欢=1

    # 6 个"高分烂片"噪声（喜欢簇中心 (8.5, 9.0) 半径 1.2 六边形顶点，标 0）
    # —— 包在密集喜欢邻居中演示 k=1 红色孤岛、k 适中时被淹没
    noise_pts = np.array([
        (9.7, 9.0),  (9.1, 10.04), (7.9, 10.04),
        (7.3, 9.0),  (7.9, 7.96),  (9.1, 7.96),
    ])
    X = np.vstack([X, noise_pts])
    y = np.concatenate([y, np.zeros(6, dtype=int)])
    return X, y


@app.cell
def _(X, mo, y):
    # 数据信息卡片（页面常驻显示，不藏 accordion）
    n_total = len(X)
    n_like = int((y == 1).sum())
    n_dislike = int((y == 0).sum())
    mo.md(
        f"""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px 14px;font-size:13px;line-height:1.7;">
    <strong style="color:#0f172a;">数据集</strong>（make_blobs 两簇高斯 + 6 个标错噪声）<br>
    • 样本量 <code>N={n_total}</code> · <span style="color:#16a34a;">喜欢 {n_like}</span> / <span style="color:#dc2626;">不喜欢 {n_dislike}</span><br>
    • 喜欢簇中心 <code>(rating=8.5, lead=9.0)</code> · 不喜欢簇中心 <code>(6.0, 5.5)</code> · <code>cluster_std=1.3</code>（适度重叠：保持"绿背景 + 红孤岛"形态）<br>
    • 噪声：6 个"高分烂片"散布在喜欢簇中心半径 1.2 的六边形顶点上，标签翻为不喜欢
    </div>
        """
    )
    return


@app.cell
def _(mo):
    section_style = (
        "border-left:2px solid #6366f1;padding:1px 8px;"
        "font-weight:600;font-size:12px;color:#475569;margin-bottom:2px;"
    )
    k_slider = mo.ui.slider(1, 125, value=11, step=2, label="k 邻居数")
    weighted_switch = mo.ui.switch(value=False, label="距离加权投票（关=多数票）")
    new_rating = mo.ui.slider(3.0, 11.5, value=7.5, step=0.1, label="新电影评分")
    new_lead = mo.ui.slider(3.0, 12.5, value=8.0, step=0.1, label="主演吸引度")
    mo.hstack(
        [
            mo.vstack([
                mo.md(f'<div style="{section_style}">KNN 参数</div>'),
                k_slider,
                weighted_switch,
            ]),
            mo.vstack([
                mo.md(f'<div style="{section_style}">新电影查询 🔷</div>'),
                new_rating,
                new_lead,
            ]),
        ],
        widths=[1, 1],
        justify="space-around",
    )
    return k_slider, new_lead, new_rating, weighted_switch


@app.cell
def _(X, k_slider, mo, new_lead, new_rating, np, weighted_switch, y):
    # 新电影查询：当前 k 下的预测
    query = np.array([new_rating.value, new_lead.value])
    q_dists = np.linalg.norm(X - query, axis=1)
    q_top_k = np.argsort(q_dists)[: k_slider.value]
    q_top_y = y[q_top_k]
    q_top_d = q_dists[q_top_k]
    q_likes = int((q_top_y == 1).sum())
    q_dislikes = k_slider.value - q_likes

    if weighted_switch.value:
        # 距离加权：weight = 1 / (d + eps)，按权重求和比较
        _w = 1.0 / (q_top_d + 1e-9)
        q_score_like = float(_w[q_top_y == 1].sum())
        q_score_dislike = float(_w[q_top_y == 0].sum())
        q_pred = "喜欢" if q_score_like > q_score_dislike else "不喜欢"
        _vote_str = (
            f'喜欢分 <strong style="color:#16a34a;">{q_score_like:.2f}</strong> / '
            f'不喜欢分 <strong style="color:#dc2626;">{q_score_dislike:.2f}</strong> '
            f'<span style="color:#64748b;">（票数：{q_likes}/{q_dislikes}）</span>'
        )
    else:
        q_pred = "喜欢" if q_likes > k_slider.value / 2 else "不喜欢"
        _vote_str = (
            f'<strong style="color:#16a34a;">{q_likes}</strong> 喜欢 / '
            f'<strong style="color:#dc2626;">{q_dislikes}</strong> 不喜欢'
        )

    _bg = "#e8f5e9" if q_pred == "喜欢" else "#ffebee"
    _bd = "#16a34a" if q_pred == "喜欢" else "#dc2626"
    _txt = "#166534" if q_pred == "喜欢" else "#991b1b"
    _mode = "距离加权" if weighted_switch.value else "多数票"

    mo.md(
        f'<div style="padding:8px 14px;background:{_bg};border-left:4px solid {_bd};'
        f'border-radius:4px;font-size:14px;color:{_txt};margin:6px 0;">'
        f"<strong>新电影预测：{q_pred}</strong>"
        f" · 评分 <code>{new_rating.value:.1f}</code> · 吸引度 <code>{new_lead.value:.1f}</code>"
        f" · k=<code>{k_slider.value}</code> · 模式 <code>{_mode}</code>"
        f" · {_vote_str}"
        "</div>"
    )
    return q_top_k, query


@app.cell
def _(X, k_slider, np, weighted_switch, y):
    # LOOCV：每个样本"留 1 后"的预测 + 邻居票数/加权得分 + top-1 邻居索引
    preds = np.zeros(len(X), dtype=int)
    vote_likes = np.zeros(len(X), dtype=int)  # 多数票：喜欢邻居数
    score_like = np.zeros(len(X), dtype=float)  # 加权：喜欢得分
    score_dislike = np.zeros(len(X), dtype=float)  # 加权：不喜欢得分
    nearest_idx = np.zeros(len(X), dtype=int)
    for _i in range(len(X)):
        _d = np.linalg.norm(X - X[_i], axis=1)
        _d[_i] = np.inf  # 排除自己
        _top = np.argsort(_d)[: k_slider.value]
        _ty = y[_top]
        _td = _d[_top]
        _likes = int((_ty == 1).sum())
        vote_likes[_i] = _likes
        # 加权得分（无论开关都算，便于错分表展示）
        _w = 1.0 / (_td + 1e-9)
        score_like[_i] = float(_w[_ty == 1].sum())
        score_dislike[_i] = float(_w[_ty == 0].sum())
        if weighted_switch.value:
            preds[_i] = 1 if score_like[_i] > score_dislike[_i] else 0
        else:
            preds[_i] = 1 if _likes > k_slider.value / 2 else 0
        nearest_idx[_i] = _top[0]
    accuracy = float((preds == y).mean())
    return accuracy, nearest_idx, preds, score_dislike, score_like, vote_likes


@app.cell
def _(accuracy, k_slider, mo):
    if k_slider.value <= 3:
        warn = "⚠️ k 太小 · 过拟合（边界扭曲，噪声形成孤岛）"
        bg, bd, txt = "#fff7ed", "#f97316", "#9a3412"
    elif k_slider.value >= 101:
        warn = "⚠️ k 接近 N · 严重欠拟合（趋向多数类全图塌陷）"
        bg, bd, txt = "#fff7ed", "#f97316", "#9a3412"
    elif k_slider.value >= 41:
        warn = "○ k 偏大 · 边界过度平滑（细节丢失但准确率仍在平台）"
        bg, bd, txt = "#fef3c7", "#eab308", "#854d0e"
    else:
        warn = "✓ k 适中"
        bg, bd, txt = "#e8f5e9", "#2ca02c", "#166534"

    mo.md(
        f'<div style="padding:6px 12px;background:{bg};border-left:4px solid {bd};'
        f'border-radius:4px;font-size:14px;color:{txt};">'
        f"<strong>{warn}</strong> · k=<code>{k_slider.value}</code>"
        f" · LOOCV 准确率 <code>{accuracy:.0%}</code>"
        "</div>"
    )
    return


@app.cell
def _(X, alt, k_slider, np, pd, q_top_k, query, weighted_switch, y):
    # 决策边界网格（mark_rect 平铺，无缝隙）
    n_grid = 60
    grid_x = np.linspace(3, 11.5, n_grid)
    grid_y = np.linspace(3, 12.5, n_grid)
    step_x = (11.5 - 3) / (n_grid - 1)
    step_y = (12.5 - 3) / (n_grid - 1)
    mesh_rows = []
    for gx in grid_x:
        for gy in grid_y:
            _d = np.linalg.norm(X - np.array([gx, gy]), axis=1)
            _top_k = np.argsort(_d)[: k_slider.value]
            _ty = y[_top_k]
            if weighted_switch.value:
                _w = 1.0 / (_d[_top_k] + 1e-9)
                _sl = float(_w[_ty == 1].sum())
                _sd = float(_w[_ty == 0].sum())
                _pred = "喜欢" if _sl > _sd else "不喜欢"
            else:
                _pred = "喜欢" if (_ty == 1).sum() > k_slider.value / 2 else "不喜欢"
            mesh_rows.append({
                "x": gx, "y": gy,
                "x_end": gx + step_x, "y_end": gy + step_y,
                "pred": _pred,
            })
    df_mesh = pd.DataFrame(mesh_rows)

    is_neighbor = np.zeros(len(X), dtype=bool)
    is_neighbor[q_top_k] = True

    df_pts = pd.DataFrame({
        "rating": X[:, 0],
        "lead": X[:, 1],
        "label": ["喜欢" if lbl == 1 else "不喜欢" for lbl in y],
        "is_neighbor": is_neighbor,
    })
    df_query = pd.DataFrame({
        "rating": [query[0]],
        "lead": [query[1]],
    })

    # k 邻居半径圆（圆心=查询点，半径=top-k 中最远那个的距离）
    k_radius = float(np.linalg.norm(X[q_top_k[-1]] - query))
    _theta = np.linspace(0, 2 * np.pi, 120)
    df_circle = pd.DataFrame({
        "idx": np.arange(len(_theta)),
        "rating": query[0] + k_radius * np.cos(_theta),
        "lead": query[1] + k_radius * np.sin(_theta),
    })

    base_x = alt.Scale(domain=[3, 11.5])
    base_y = alt.Scale(domain=[3, 12.5])
    palette = alt.Scale(domain=["喜欢", "不喜欢"], range=["#2ca02c", "#d62728"])

    # 决策区域
    region = alt.Chart(df_mesh).mark_rect(opacity=0.35, stroke=None).encode(
        x=alt.X("x:Q", scale=base_x, title="评分"),
        y=alt.Y("y:Q", scale=base_y, title="主演吸引度"),
        x2="x_end:Q",
        y2="y_end:Q",
        color=alt.Color("pred:N", scale=palette, legend=None),
    )

    # 普通训练点（非 top-k 邻居）
    pts_normal = alt.Chart(df_pts[~df_pts.is_neighbor]).mark_circle(
        size=60, stroke="black", strokeWidth=0.5, opacity=0.55
    ).encode(
        x="rating:Q", y="lead:Q",
        color=alt.Color("label:N", scale=palette,
                        legend=alt.Legend(title="历史标签")),
        tooltip=["rating:Q", "lead:Q", "label:N"],
    )

    # top-k 邻居高亮（金色粗边）
    pts_neighbor = alt.Chart(df_pts[df_pts.is_neighbor]).mark_circle(
        size=130, stroke="#fbbf24", strokeWidth=2.2, opacity=1.0
    ).encode(
        x="rating:Q", y="lead:Q",
        color=alt.Color("label:N", scale=palette, legend=None),
        tooltip=["rating:Q", "lead:Q", "label:N"],
    )

    # 查询点（新电影，蓝色菱形）
    query_pt = alt.Chart(df_query).mark_point(
        shape="diamond", size=400, color="#1f77b4", filled=True,
        stroke="black", strokeWidth=1.5,
    ).encode(x="rating:Q", y="lead:Q")

    # k 半径圆（虚线蓝圈，标识投票范围）
    radius_circle = alt.Chart(df_circle).mark_line(
        stroke="#1f77b4", strokeDash=[5, 3], strokeWidth=1.5, opacity=0.7,
    ).encode(
        x=alt.X("rating:Q", scale=base_x),
        y=alt.Y("lead:Q", scale=base_y),
        order="idx:Q",
    )

    chart = (region + pts_normal + pts_neighbor + radius_circle + query_pt).properties(
        width=620, height=440,
        title=f"k={k_slider.value} · 模式={'距离加权' if weighted_switch.value else '多数票'} · 半径={k_radius:.2f} · 金边=top-k 邻居 · 蓝菱形=新电影"
    )
    chart
    return


@app.cell
def _(X, accuracy, k_slider, mo, nearest_idx, np, pd, preds, score_dislike, score_like, vote_likes, weighted_switch, y):
    # 实时数据表：混淆矩阵 + 错分样本明细（随 k 变化）
    real_like = y == 1
    pred_like = preds == 1
    tp = int((real_like & pred_like).sum())
    tn = int((~real_like & ~pred_like).sum())
    fp = int((~real_like & pred_like).sum())
    fn = int((real_like & ~pred_like).sum())

    wrong_mask = preds != y
    n_wrong = int(wrong_mask.sum())

    wrong_indices = np.where(wrong_mask)[0]
    if n_wrong > 0:
        # 最近邻列：显示 top-1 邻居的索引、坐标、标签、距离
        nearest_strs = []
        for _idx in wrong_indices:
            _ni = int(nearest_idx[_idx])
            _dist = float(np.linalg.norm(X[_idx] - X[_ni]))
            _label = "喜欢" if y[_ni] == 1 else "不喜欢"
            nearest_strs.append(f"#{_ni + 1} ({X[_ni, 0]:.2f}, {X[_ni, 1]:.2f}) {_label} · d={_dist:.2f}")

        if weighted_switch.value:
            vote_col_name = f"k={k_slider.value} 加权得分"
            vote_col_data = [
                f"喜欢 {score_like[i]:.2f} / 不喜欢 {score_dislike[i]:.2f}"
                f" · 票 {int(vote_likes[i])}/{k_slider.value - int(vote_likes[i])}"
                for i in wrong_indices
            ]
        else:
            vote_col_name = f"k={k_slider.value} 邻居票"
            vote_col_data = [
                f"{int(vote_likes[i])} 喜欢 / {k_slider.value - int(vote_likes[i])} 不喜欢"
                for i in wrong_indices
            ]

        wrong_df = pd.DataFrame({
            "#": wrong_indices + 1,
            "rating": np.round(X[wrong_mask, 0], 2),
            "lead": np.round(X[wrong_mask, 1], 2),
            "真实": ["喜欢" if v == 1 else "不喜欢" for v in y[wrong_mask]],
            "预测": ["喜欢" if v == 1 else "不喜欢" for v in preds[wrong_mask]],
            "最近邻 (top-1)": nearest_strs,
            vote_col_name: vote_col_data,
        }).sort_values("rating").reset_index(drop=True)
        wrong_table = mo.ui.table(wrong_df, page_size=15, selection=None)
    else:
        wrong_table = mo.md("🎉 **全部分类正确**（这种情况罕见，通常 k 偏小或数据无噪声）")

    mo.vstack([
        mo.md(f"""
    ### 实时分类数据（k={k_slider.value}, LOOCV 准确率 {accuracy:.1%}）

    **混淆矩阵**

    | 真实 \\\\ 预测 | 喜欢 | 不喜欢 | 合计 |
    |---|---|---|---|
    | **喜欢** | <span style="color:#16a34a;font-weight:600;">{tp}</span> ✓ | <span style="color:#dc2626;">{fn}</span> ✗ | {tp + fn} |
    | **不喜欢** | <span style="color:#dc2626;">{fp}</span> ✗ | <span style="color:#16a34a;font-weight:600;">{tn}</span> ✓ | {fp + tn} |
    | **合计** | {tp + fp} | {fn + tn} | {len(y)} |

    **错分样本 {n_wrong} 个**（拖 k 看哪些样本"摇摆"——k 小时孤岛/边界点错，k 大时多数类样本错）
        """),
        wrong_table,
    ])
    return


@app.cell
def _(mo):
    # 玩法指南（常驻页面，不藏 accordion）
    mo.md("""
    ### 拖动 k 看四个关键档位

    | k 档位 | 现象 | 教学要点 |
    |---|---|---|
    | **k=1** | 决策区域贴每个点，6 个噪声点在喜欢区炸出**红色孤岛** | 过拟合（high variance）：噪声被全盘吸收 |
    | **k=11** ≈√126 | 边界平滑，红色孤岛被喜欢邻居投票淹没 | bias-variance 拐点，LOOCV ≈ 89% |
    | **k=51** | 边界继续平滑，但准确率仍在 89-90% 高位 | 平台期：k 适度偏大不致命，但细节丢失 |
    | **k=121** | 邻居接近全集，多数类（不喜欢 66 > 喜欢 60）主导 → 准确率**暴跌到 52%** | 欠拟合（high bias）：模型退化为常数预测 |

    **为什么 k=5 之后边界变化变小？** 这是 KNN 的本性——边界一旦从 wiggly 平滑下来就进入**平台期**，
    要看到下一次显著变化必须把 k 拖到接近 N（k>100）。bias-variance 不是连续滑动的，是阶段性的。
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "📖 如何阅读这张图（看不懂先点这）": mo.md(
                "决策边界图叠了**两层**，初学者经常混在一起：\n\n"
                "| 图层 | 是什么 | 怎么生成 |\n"
                "|---|---|---|\n"
                "| **圆点**（带黑边） | 训练数据 = 126 部电影的真实标签 | 直接画在 (rating, lead) 上 |\n"
                "| **背景色**（淡绿/淡红块） | 模型对**假想点**的预测 | 把平面切 60×60 网格，每格问 KNN：\"如果有部新电影评分=X、吸引度=Y，你预测它喜欢吗？\" |\n\n"
                "**关键认知**：\n\n"
                "1. **背景色 ≠ 数据密度，≠ 概率云**。它是\"假如新电影落这里我会判什么\"的全平面打分图。\n\n"
                "2. **决策边界**就是背景色变化的地方——绿淡红的分界线 = 模型从\"喜欢\"切换到\"不喜欢\"的位置。\n\n"
                "3. **KNN 没有真正的训练**——它的\"模型\"就是训练数据本身 + 一条规则（找 k 个最近邻投票）。"
                "改 k 只是改投票圈大小，不重算什么权重。这种算法叫 **lazy learner**。\n\n"
                "4. **拖 k 滑块**：3600 个网格点全部重新投票 → 背景色刷新。圆点位置不动（数据没变）。\n\n"
                "5. **任何分类器都能画这种图**——决策树、SVM、神经网络都行。这是机器学习的**通用可视化语言**，"
                "不是 KNN 专属。看 sklearn 的 [Classifier comparison](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) 一图九算法。\n\n"
                "6. **2D 只是为了能画**。真实数据可能 100 维，决策边界是 100 维超平面——画不出来但概念一样。"
            ),
            "距离加权投票 vs 多数票": mo.md(
                "**多数票**（默认）：top-k 邻居每人 1 票，谁多谁赢。简单但**远近一视同仁**——\n"
                "k=11 时第 1 近邻和第 11 近邻同等权重，浪费了距离信息。\n\n"
                "**距离加权**：每个邻居的票数 = $1 / (距离 + \\epsilon)$，**越近的邻居权重越大**。\n\n"
                "$$\\text{score}_{c} = \\sum_{i \\in \\text{top-k}} \\frac{1}{d_i + \\epsilon} \\cdot \\mathbf{1}[y_i = c]$$\n\n"
                "**何时有用**：\n\n"
                "- **平票打破**：偶数 k 时距离加权天然不会平票（除非所有邻居等距离，几率为 0）\n"
                "- **大 k 局部敏感**：k=51 多数票退化成大区域多数类；加权后仍以最近的几个邻居为主导\n"
                "- **稀疏数据**：远邻居信息少，加权自动降低其影响\n\n"
                "**何时副作用**：\n\n"
                "- 噪声点距离查询点很近时，加权会**放大噪声影响**\n"
                "- 当 k=1 时多数票和加权完全等价（只有 1 个邻居）\n\n"
                "**操作**：拖滑块到 k=21，开关切换看决策边界形状变化——加权后边界更紧贴训练点。"
            ),
            "数学：LOOCV 与经验法则": mo.md(
                "**LOOCV (Leave-One-Out CV)**：每次留 1 个样本当测试，剩 N-1 训练 → 跑 N 次 → 平均准确率\n\n"
                "**经验法则**：$k \\approx \\sqrt{N}$（N 是样本数）。本例 N=126 → 推荐 k≈11"
            ),
            "为什么底部一直是红？": mo.md(
                "这**不是 bug，是几何必然**。\n\n"
                "数据生成时不喜欢簇的中心是 (rating=6.0, lead=5.5)，喜欢簇中心是 (8.5, 9.0)——"
                "样本本来就**没有任何喜欢点出现在 lead<3 的区域**。\n\n"
                "KNN 只能看邻居投票。当某个区域周围所有最近邻都是不喜欢，无论 k=1 还是 k=125，"
                "都不可能预测出'喜欢'。这是**数据决定的下限**，不是 KNN 算法的失败。\n\n"
                "**对比真实场景**：如果你的训练数据某个特征区间没有正样本，模型在该区间的预测必然偏向负类——"
                "这是数据覆盖不足的问题，不是模型能力的问题。"
            ),
        },
        multiple=False,
    )
    return


if __name__ == "__main__":
    app.run()
