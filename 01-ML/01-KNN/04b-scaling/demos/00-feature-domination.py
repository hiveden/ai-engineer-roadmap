"""
量纲独裁可视化 · 健康预测 8 人

核心交互：
  · 拖视力滑块大幅变化 (0.3 ↔ 1.8) → top-3 邻居几乎不变（视力权重 2.25 太小）
  · 拖体重滑块小幅变化 (70 ↔ 80) → top-3 邻居全换（体重权重 2025 独裁）

跑：marimo run 00-feature-domination.py --port 2750 --no-token
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/00-feature-domination.grid.json",
    css_file="marimo.css",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _(mo):
    mo.md("# 量纲独裁可视化 · 体重权重 vs 视力权重 = 900 倍")
    return


@app.cell
def _(mo):
    info_style = (
        "border-left:3px solid #6366f1;padding:6px 12px;"
        "background:#f5f3ff;border-radius:4px;font-size:14px;line-height:1.6;"
    )
    mo.md(
        f'<div style="{info_style}">'
        "<strong>训练集</strong>：8 人 · 健康 4 / 不健康 4 ｜ "
        "<strong>特征</strong>：身高(cm) / 体重(kg) / 视力 ｜ "
        "<strong>跨度对比</strong>：身高=15 / 体重=45 / 视力=1.5 ｜ "
        "<strong>权重 = 跨度²</strong>："
        '<span style="background:#fee2e2;padding:1px 6px;border-radius:3px;color:#991b1b;font-weight:600;">体重 2025</span> · '
        "身高 225 · 视力 2.25"
        "</div>"
    )
    return


@app.cell
def _(mo):
    section_style = (
        "border-left:3px solid #6366f1;padding:2px 10px;"
        "font-weight:600;font-size:14px;margin-bottom:4px;"
    )

    new_weight = mo.ui.slider(55, 115, value=75, step=1, label="新人体重 (kg)")
    new_vision = mo.ui.slider(0.2, 2.0, value=1.7, step=0.1, label="新人视力")

    mo.hstack(
        [
            mo.vstack([
                mo.md(f'<div style="{section_style}">新人特征（身高固定 173 cm · K=3）</div>'),
                mo.hstack([new_weight, new_vision], widths=[1, 1], justify="start", gap=4),
            ]),
        ],
        justify="start",
    )
    return new_vision, new_weight


@app.cell
def _(new_vision, new_weight, np):
    NEW_HEIGHT = 173  # 固定，专注体重 × 视力对比

    X_data = np.array([
        [175, 70,  1.5],
        [180, 78,  1.2],
        [170, 65,  1.8],
        [168, 72,  1.5],
        [165, 95,  0.4],
        [172, 100, 0.3],
        [178, 110, 0.5],
        [167, 92,  0.6],
    ])
    y_data = np.array([1, 1, 1, 1, 2, 2, 2, 2])

    new_person = np.array([NEW_HEIGHT, new_weight.value, new_vision.value])

    diffs = X_data - new_person  # (8, 3)
    sq = diffs ** 2  # (8, 3)
    dists = np.sqrt(sq.sum(axis=1))
    sorted_idx = np.argsort(dists)
    K = 3
    top_idx = sorted_idx[:K]
    top_labels = y_data[top_idx]
    healthy = int((top_labels == 1).sum())
    unhealthy = int((top_labels == 2).sum())
    prediction = 1 if healthy > unhealthy else 2

    # 三项平方占比（用 top-1 邻居的差异分解，最直观）
    nearest = top_idx[0]
    sq_h, sq_w, sq_v = sq[nearest]
    total = sq_h + sq_w + sq_v
    pct_h = sq_h / total * 100 if total > 0 else 0
    pct_w = sq_w / total * 100 if total > 0 else 0
    pct_v = sq_v / total * 100 if total > 0 else 0

    return (
        K,
        NEW_HEIGHT,
        X_data,
        dists,
        nearest,
        new_person,
        pct_h,
        pct_v,
        pct_w,
        prediction,
        sorted_idx,
        sq_h,
        sq_v,
        sq_w,
        top_idx,
        y_data,
    )


@app.cell
def _(X_data, mo, new_weight, new_vision, plt, top_idx, y_data):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    colors = ["#2ca02c" if c == 1 else "#d62728" for c in y_data]
    ax.scatter(X_data[:, 1], X_data[:, 2], c=colors, s=120, edgecolors="k", linewidths=0.8, zorder=3)

    # top-3 黄圈高亮
    ax.scatter(
        X_data[top_idx, 1], X_data[top_idx, 2],
        s=320, facecolors="none", edgecolors="#fbbf24", linewidths=3, zorder=4,
    )

    # 新人黑星
    ax.scatter(new_weight.value, new_vision.value, c="black", s=260,
               marker="*", edgecolors="white", linewidths=1.2, zorder=5)

    # 编号标注
    for _i, (_w, _v) in enumerate(zip(X_data[:, 1], X_data[:, 2])):
        ax.annotate(str(_i + 1), (_w, _v), fontsize=8, ha="center", va="center", color="white", fontweight="bold", zorder=6)

    ax.set_xlabel("Weight (kg)  · range=45")
    ax.set_ylabel("Vision  · range=1.5")
    ax.set_xlim(50, 120)
    ax.set_ylim(0, 2.2)
    ax.grid(alpha=0.3)
    ax.set_facecolor("#fafafa")

    plt.tight_layout()
    mo.mpl.interactive(fig)
    fig
    return


@app.cell
def _(
    K,
    NEW_HEIGHT,
    X_data,
    dists,
    mo,
    nearest,
    new_vision,
    new_weight,
    pct_h,
    pct_v,
    pct_w,
    prediction,
    sq_h,
    sq_v,
    sq_w,
    top_idx,
    y_data,
):
    label = "健康" if prediction == 1 else "不健康"
    bg = "#e8f5e9" if prediction == 1 else "#ffebee"
    bd = "#2ca02c" if prediction == 1 else "#d62728"
    txt = "#166534" if prediction == 1 else "#991b1b"

    pred_card = (
        f'<div style="padding:6px 12px;background:{bg};border-left:4px solid {bd};'
        f'border-radius:4px;font-size:14px;color:{txt};margin-bottom:6px;">'
        f"<strong>预测：{label}</strong> · 新人 ({NEW_HEIGHT}, "
        f"<code>{new_weight.value}</code>, <code>{new_vision.value:.1f}</code>) "
        f"· top-{K} 邻居 = "
        + " / ".join(
            ("健康" if y_data[i] == 1 else "不健康") + f"#{i + 1}"
            for i in top_idx
        )
        + "</div>"
    )

    # 距离公式拆分（用最近邻 top-1）
    h_n, w_n, v_n = X_data[nearest]
    diff_h = NEW_HEIGHT - h_n
    diff_w = new_weight.value - w_n
    diff_v = new_vision.value - v_n

    def bar(pct, color):
        return (
            f'<div style="background:#e5e7eb;border-radius:3px;height:10px;width:80px;'
            f'display:inline-block;vertical-align:middle;overflow:hidden;">'
            f'<div style="background:{color};width:{pct:.1f}%;height:100%;"></div></div>'
        )

    formula_md = (
        f"#### 距离公式拆分（到 top-1 邻居 #{nearest + 1}）\n\n"
        f"| 维度 | 差 | 差² | 占比 | 视觉 |\n"
        f"|---|---|---|---|---|\n"
        f"| 身高 | {diff_h:+.1f} | {sq_h:.2f} | {pct_h:.1f}% | {bar(pct_h, '#9ca3af')} |\n"
        f"| **体重** | **{diff_w:+.1f}** | **{sq_w:.2f}** | **{pct_w:.1f}%** | {bar(pct_w, '#dc2626')} |\n"
        f"| 视力 | {diff_v:+.2f} | {sq_v:.4f} | {pct_v:.1f}% | {bar(pct_v, '#9ca3af')} |\n"
    )

    insight = ""
    if pct_w > 90:
        insight = (
            '<div style="margin-top:6px;padding:4px 10px;background:#fef3c7;'
            'border-left:3px solid #f59e0b;border-radius:3px;font-size:13px;color:#92400e;">'
            f"⚠ 体重项贡献 <strong>{pct_w:.0f}%</strong> → 距离 ≈ 体重差，其余维度被淹没"
            "</div>"
        )

    # 距离明细 top-3
    _detail_rows = []
    for _rank, _i in enumerate(top_idx, start=1):
        _kind = "健康" if y_data[_i] == 1 else "不健康"
        _h, _w, _v = X_data[_i]
        _detail_rows.append(
            f"| {_rank} | #{_i + 1} | {_h:.0f} / {_w:.0f} / {_v:.1f} | {_kind} | {dists[_i]:.3f} |"
        )
    detail_md = (
        "#### top-3 邻居\n\n"
        "| 排名 | 编号 | 身高/体重/视力 | 标签 | 距离 |\n"
        "|---|---|---|---|---|\n"
        + "\n".join(_detail_rows)
    )

    mo.md(pred_card + formula_md + insight + "\n\n" + detail_md)
    return


@app.cell
def _(mo):
    # 📐 Grid 布局参考（开发用 · 录屏隐藏 position=null）
    mo.md(
        """
## 📐 Grid 布局（16:9 · maxWidth=1280）

```
   0           14          24
0  ┌──────── 标题（h=2）──────────┐
2  ├────── 数据信息卡（h=4）──────┤
6  ├──────── 控件行（h=4）────────┤
10 ├── 散点图 ──────┬─ 状态/公式 ─┤
24                                  (h=14)
```

cells 顺序：imports / 标题 / 信息卡 / 控件 / 计算(隐藏) / 散点图 / 状态卡 / 本 cell(隐藏)

### 演示路径

| 步骤 | 操作 | 看 |
|---|---|---|
| baseline | 体重 85 / 视力 1.0 | top-3 邻居 + 体重项占比 |
| 视力大变 | 体重不动 / 视力 0.3 → 1.8 | top-3 几乎不变 |
| 体重微变 | 视力不动 / 体重 70 → 80 | top-3 全换 |
"""
    )
    return


if __name__ == "__main__":
    app.run()
