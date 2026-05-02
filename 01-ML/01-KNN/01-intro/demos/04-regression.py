"""
KNN 回归演示 · 预测连续值

互动：拖动新电影坐标 → 实时预测观看完成度（%）→ 看 k 大小对回归预测的影响
跑：marimo edit 04-regression.py
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

    return alt, mo, np, pd


@app.cell
def _(mo):
    mo.md("""
    # KNN 回归 · 预测连续值

    > 02/03 的 KNN 是**分类**（输出 喜欢 / 不喜欢）。本 demo 切到**回归**——
    > 输出**连续值**（观看完成度 0-100%）。算法相同，预测从"邻居投票"变成"邻居求平均"。

    数据：80 部电影的 (评分, 主演吸引度) → 你的观看完成度（%）。
    拖动新电影坐标看 KNN 怎么预测；拖 k 看 bias-variance 在回归里的形态。
    """)
    return


@app.cell
def _(np):
    # 合成数据：完成度 ≈ 5·rating + 5·lead − 30 + 噪声
    rng = np.random.default_rng(42)
    n = 80
    X = np.column_stack([
        rng.uniform(3.0, 11.0, n),
        rng.uniform(3.0, 12.0, n),
    ])
    target = 5 * X[:, 0] + 5 * X[:, 1] - 30 + rng.normal(0, 8, n)
    target = np.clip(target, 0, 100)
    return X, target


@app.cell
def _(X, mo, target):
    mo.md(
        f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px 14px;font-size:13px;line-height:1.7;">
<strong>数据集</strong>（连续目标值）<br>
• 样本量 <code>N={len(X)}</code> · 目标 = 观看完成度（0-100%）<br>
• 真实关系：完成度 ≈ <code>5·rating + 5·lead − 30 + N(0, 8)</code>，clip 到 [0, 100]<br>
• 训练数据完成度：min <code>{target.min():.1f}</code> / max <code>{target.max():.1f}</code> / 均值 <code>{target.mean():.1f}</code>
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
    k_slider = mo.ui.slider(1, 80, value=7, step=1, label="k 邻居数")
    weighted_switch = mo.ui.switch(value=False, label="距离加权（关=简单平均）")
    new_rating = mo.ui.slider(3.0, 11.0, value=7.5, step=0.1, label="新电影评分")
    new_lead = mo.ui.slider(3.0, 12.0, value=8.0, step=0.1, label="主演吸引度")
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
def _(X, k_slider, mo, new_lead, new_rating, np, target, weighted_switch):
    # 新电影查询：当前 k 下的回归预测
    query = np.array([new_rating.value, new_lead.value])
    q_dists = np.linalg.norm(X - query, axis=1)
    q_top_k = np.argsort(q_dists)[: k_slider.value]
    q_targets = target[q_top_k]
    q_d = q_dists[q_top_k]

    if weighted_switch.value:
        _w = 1.0 / (q_d + 1e-9)
        q_pred = float((_w * q_targets).sum() / _w.sum())
    else:
        q_pred = float(q_targets.mean())

    q_min = float(q_targets.min())
    q_max = float(q_targets.max())
    q_std = float(q_targets.std())

    _mode = "距离加权" if weighted_switch.value else "简单平均"
    mo.md(
        f'<div style="padding:8px 14px;background:#eff6ff;border-left:4px solid #2563eb;'
        f'border-radius:4px;font-size:14px;margin:6px 0;">'
        f'<strong style="color:#1e40af;">预测观看完成度：{q_pred:.1f}%</strong>'
        f" · 评分 <code>{new_rating.value:.1f}</code> · 吸引度 <code>{new_lead.value:.1f}</code>"
        f" · k=<code>{k_slider.value}</code> · 模式 <code>{_mode}</code><br>"
        f"<span style=\"color:#64748b;font-size:13px;\">"
        f"top-{k_slider.value} 邻居完成度：min <code>{q_min:.1f}</code> · "
        f"max <code>{q_max:.1f}</code> · 标准差 <code>{q_std:.1f}</code>"
        f"</span>"
        "</div>"
    )
    return q_top_k, query


@app.cell
def _(X, k_slider, np, target, weighted_switch):
    # LOOCV：每个样本"留 1 后"的回归预测
    preds_reg = np.zeros(len(X))
    for _i in range(len(X)):
        _d = np.linalg.norm(X - X[_i], axis=1)
        _d[_i] = np.inf
        _top = np.argsort(_d)[: k_slider.value]
        _t = target[_top]
        if weighted_switch.value:
            _w = 1.0 / (_d[_top] + 1e-9)
            preds_reg[_i] = (_w * _t).sum() / _w.sum()
        else:
            preds_reg[_i] = _t.mean()

    mae = float(np.abs(preds_reg - target).mean())
    rmse = float(np.sqrt(((preds_reg - target) ** 2).mean()))
    _ss_res = float(((target - preds_reg) ** 2).sum())
    _ss_tot = float(((target - target.mean()) ** 2).sum())
    r2 = float(1 - _ss_res / _ss_tot)
    return mae, preds_reg, r2, rmse


@app.cell
def _(k_slider, mae, mo, r2, rmse):
    if k_slider.value <= 2:
        warn = "⚠️ k 小 · 高方差（预测过于贴噪声，决策面斑驳）"
        bg, bd, txt = "#fff7ed", "#f97316", "#9a3412"
    elif k_slider.value >= 30:
        warn = "⚠️ k 大 · 高偏差（预测趋向全局均值，决策面单调）"
        bg, bd, txt = "#fff7ed", "#f97316", "#9a3412"
    else:
        warn = "✓ k 适中"
        bg, bd, txt = "#e8f5e9", "#2ca02c", "#166534"
    mo.md(
        f'<div style="padding:6px 12px;background:{bg};border-left:4px solid {bd};'
        f'border-radius:4px;font-size:14px;color:{txt};">'
        f"<strong>{warn}</strong> · k=<code>{k_slider.value}</code>"
        f" · LOOCV <strong>MAE</strong> <code>{mae:.2f}%</code>"
        f" · <strong>RMSE</strong> <code>{rmse:.2f}%</code>"
        f" · <strong>R²</strong> <code>{r2:.3f}</code>"
        "</div>"
    )
    return


@app.cell
def _(X, alt, k_slider, np, pd, q_top_k, query, target, weighted_switch):
    # 决策面：网格上每点的回归预测，色阶填充
    n_grid = 50
    grid_x = np.linspace(3, 11, n_grid)
    grid_y = np.linspace(3, 12, n_grid)
    step_x = (11 - 3) / (n_grid - 1)
    step_y = (12 - 3) / (n_grid - 1)
    rows = []
    for gx in grid_x:
        for gy in grid_y:
            _d = np.linalg.norm(X - np.array([gx, gy]), axis=1)
            _top = np.argsort(_d)[: k_slider.value]
            _t = target[_top]
            if weighted_switch.value:
                _w = 1.0 / (_d[_top] + 1e-9)
                _p = float((_w * _t).sum() / _w.sum())
            else:
                _p = float(_t.mean())
            rows.append({
                "x": gx, "y": gy,
                "x_end": gx + step_x, "y_end": gy + step_y,
                "pred": _p,
            })
    df_mesh = pd.DataFrame(rows)

    is_neighbor = np.zeros(len(X), dtype=bool)
    is_neighbor[q_top_k] = True

    df_pts = pd.DataFrame({
        "rating": X[:, 0],
        "lead": X[:, 1],
        "完成度": target,
        "is_neighbor": is_neighbor,
    })
    df_query = pd.DataFrame({"rating": [query[0]], "lead": [query[1]]})

    # k 半径圆
    k_radius = float(np.linalg.norm(X[q_top_k[-1]] - query))
    _theta = np.linspace(0, 2 * np.pi, 120)
    df_circle = pd.DataFrame({
        "idx": np.arange(len(_theta)),
        "rating": query[0] + k_radius * np.cos(_theta),
        "lead": query[1] + k_radius * np.sin(_theta),
    })

    base_x = alt.Scale(domain=[3, 11])
    base_y = alt.Scale(domain=[3, 12])
    color_scale = alt.Scale(scheme="viridis", domain=[0, 100])

    # 预测面
    region = alt.Chart(df_mesh).mark_rect(opacity=0.55, stroke=None).encode(
        x=alt.X("x:Q", scale=base_x, title="评分"),
        y=alt.Y("y:Q", scale=base_y, title="主演吸引度"),
        x2="x_end:Q",
        y2="y_end:Q",
        color=alt.Color("pred:Q", scale=color_scale,
                        legend=alt.Legend(title="预测完成度 %")),
    )

    # 普通训练点
    pts_normal = alt.Chart(df_pts[~df_pts.is_neighbor]).mark_circle(
        size=80, stroke="black", strokeWidth=0.7, opacity=0.95,
    ).encode(
        x="rating:Q", y="lead:Q",
        color=alt.Color("完成度:Q", scale=color_scale, legend=None),
        tooltip=["rating:Q", "lead:Q", "完成度:Q"],
    )

    # top-k 邻居高亮
    pts_neighbor = alt.Chart(df_pts[df_pts.is_neighbor]).mark_circle(
        size=180, stroke="#fbbf24", strokeWidth=2.5, opacity=1.0,
    ).encode(
        x="rating:Q", y="lead:Q",
        color=alt.Color("完成度:Q", scale=color_scale, legend=None),
        tooltip=["rating:Q", "lead:Q", "完成度:Q"],
    )

    # k 半径圆
    radius_circle = alt.Chart(df_circle).mark_line(
        stroke="#dc2626", strokeDash=[5, 3], strokeWidth=1.5, opacity=0.8,
    ).encode(
        x=alt.X("rating:Q", scale=base_x),
        y=alt.Y("lead:Q", scale=base_y),
        order="idx:Q",
    )

    # 查询点（红菱形，与训练点 viridis 色阶区分）
    query_pt = alt.Chart(df_query).mark_point(
        shape="diamond", size=400, color="#dc2626", filled=True,
        stroke="black", strokeWidth=1.8,
    ).encode(x="rating:Q", y="lead:Q")

    chart = (region + pts_normal + pts_neighbor + radius_circle + query_pt).properties(
        width=620, height=440,
        title=(
            f"KNN 回归预测面（k={k_slider.value}, "
            f"{'加权' if weighted_switch.value else '平均'}, "
            f"半径={k_radius:.2f}）· 红菱形=新电影 · 金边=top-k 邻居"
        ),
    )
    chart
    return


@app.cell
def _(mo):
    mo.md(
        """
### 拖动 k 看回归的 bias-variance

| k | 现象 | 教学要点 |
|---|---|---|
| **k=1** | 决策面斑驳，每格颜色 = 最近训练点的完成度 | 高方差：预测完全跟着噪声 |
| **k=7** ≈√80 | 决策面平滑，仍保留局部结构（左下偏冷 / 右上偏暖） | bias-variance 拐点 |
| **k=20** | 决策面进一步光滑，渐变更柔 | 偏差略增、方差大降 |
| **k=80** | 决策面变成**单一颜色 = 全局均值**（≈42%） | 极端欠拟合：完全忽略特征 |

**回归 vs 分类的 k 影响**：分类靠投票（离散），多数决一旦稳定就稳定；
回归靠**平均**（连续），**每个邻居都贡献数值**——所以 k 一变化预测立刻有连续位移，
决策面比分类敏感得多。
        """
    )
    return


@app.cell
def _(mo, preds_reg, target):
    import altair as _alt
    import pandas as _pd
    # 残差散点图：真实 vs 预测，越贴对角线越准
    df_resid = _pd.DataFrame({
        "真实完成度": target,
        "预测完成度": preds_reg,
        "残差": preds_reg - target,
    })
    line_df = _pd.DataFrame({"x": [0, 100], "y": [0, 100]})

    diag = _alt.Chart(line_df).mark_line(stroke="#94a3b8", strokeDash=[4, 3]).encode(
        x="x:Q", y="y:Q",
    )
    pred_pts = _alt.Chart(df_resid).mark_circle(size=70, opacity=0.7).encode(
        x=_alt.X("真实完成度:Q", scale=_alt.Scale(domain=[0, 100])),
        y=_alt.Y("预测完成度:Q", scale=_alt.Scale(domain=[0, 100])),
        color=_alt.Color("残差:Q", scale=_alt.Scale(scheme="redyellowblue", domainMid=0)),
        tooltip=["真实完成度:Q", "预测完成度:Q", "残差:Q"],
    )
    resid_chart = (diag + pred_pts).properties(
        width=420, height=380,
        title="LOOCV 残差图：贴对角线 = 预测准 · 离对角线越远误差越大",
    )
    mo.vstack([
        mo.md("### LOOCV 预测 vs 真实（拖 k 看点云形态变化）"),
        resid_chart,
    ])
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "📖 KNN 回归怎么算的": mo.md(
                "**简单平均**（多数模式）：\n\n"
                "$$\\hat{y}(\\mathbf{x}) = \\frac{1}{k} \\sum_{i \\in \\text{top-k}(\\mathbf{x})} y_i$$\n\n"
                "**距离加权**：\n\n"
                "$$\\hat{y}(\\mathbf{x}) = \\frac{\\sum_{i \\in \\text{top-k}} w_i \\cdot y_i}{\\sum_{i \\in \\text{top-k}} w_i}, \\quad w_i = \\frac{1}{d_i + \\epsilon}$$\n\n"
                "区别于分类：回归预测**连续值**（实数），不需要投票；多数票变成数值平均，加权变成加权平均。"
            ),
            "评估指标解读": mo.md(
                "- **MAE**（Mean Absolute Error）：预测值与真实值的绝对差均值。"
                "单位 = 目标值单位（这里 %）。MAE=5 意味着平均预测偏差 5 个百分点。\n\n"
                "- **RMSE**（Root Mean Squared Error）：均方根误差。对**大误差更敏感**，"
                "因为先平方再开方。一般 RMSE ≥ MAE，差距大说明有少数大误差点。\n\n"
                "- **R²**（决定系数）：模型解释方差的比例。\n"
                "    - R² = 1：完美拟合\n"
                "    - R² = 0：等同于「全部预测为均值」的 baseline\n"
                "    - R² < 0：比 baseline 还差（KNN k 极大时会出现）"
            ),
            "什么时候用 KNN 回归": mo.md(
                "**适合**：\n\n"
                "- 数据局部相似但全局复杂（非线性关系）\n"
                "- 训练成本接近 0（lazy learner，不算参数）\n"
                "- 目标值的「局部空间相似性」假设成立\n\n"
                "**不适合**：\n\n"
                "- **高维**（curse of dimensionality）：维度高了距离失去意义\n"
                "- **大数据**：预测要算所有距离，O(N) 慢\n"
                "- **外推（extrapolation）**：KNN 的预测**永远不会超出训练数据 target 范围**——"
                "这是它的硬限制。线性回归能预测 [0, 100] 之外的值，KNN 不行"
            ),
        },
        multiple=False,
    )
    return


if __name__ == "__main__":
    app.run()
