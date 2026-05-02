"""
KNN 缩放对比 · 健康预测案例（紧凑版）

紧凑布局：
  - 控件横排两列
  - 预测 callout 居中
  - 散点图小尺寸
  - 详情合并到一个折叠面板

跑：marimo run 02-knn-scaling.py --port 2718 --no-token
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium", css_file="marimo.css")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    return MinMaxScaler, StandardScaler, mo, np, plt


@app.cell
def _(mo):
    mo.md("""
    # KNN 缩放对比 · 健康预测
    """)
    return


@app.cell
def _(mo):
    k_slider = mo.ui.slider(1, 7, value=3, step=1, label="k 邻居数")
    scaler_choice = mo.ui.dropdown(
        options=["None", "MinMaxScaler", "StandardScaler"],
        value="None",
        label="缩放",
    )
    outlier_switch = mo.ui.switch(value=False, label="注入异常值 (体重 250kg)")

    new_height = mo.ui.slider(160, 195, value=173, step=1, label="身高 cm")
    new_weight = mo.ui.slider(50, 130, value=80, step=1, label="体重 kg")
    new_vision = mo.ui.slider(0.1, 2.0, value=1.0, step=0.1, label="视力")

    section_style = (
        "border-left:3px solid #6366f1;padding:2px 10px;"
        "font-weight:600;font-size:14px;margin-bottom:4px;"
    )
    mo.hstack(
        [
            mo.vstack([
                mo.md(f'<div style="{section_style}">模型参数</div>'),
                k_slider,
                scaler_choice,
                outlier_switch,
            ]),
            mo.vstack([
                mo.md(f'<div style="{section_style}">新人特征</div>'),
                new_height,
                new_weight,
                new_vision,
            ]),
        ],
        justify="space-around",
        widths=[1, 1],
    )
    return (
        k_slider,
        new_height,
        new_vision,
        new_weight,
        outlier_switch,
        scaler_choice,
    )


@app.cell
def _(MinMaxScaler, StandardScaler, np, outlier_switch, scaler_choice):
    X_raw = np.array([
        [175, 70,  1.5],
        [180, 78,  1.2],
        [170, 65,  1.8],
        [168, 72,  1.5],
        [165, 95,  0.4],
        [172, 100, 0.3],
        [178, 110, 0.5],
        [167, 92,  0.6],
    ])
    y_raw = np.array([1, 1, 1, 1, 2, 2, 2, 2])

    if outlier_switch.value:
        X_data = np.vstack([X_raw, [[170, 250, 1.0]]])
        y_data = np.append(y_raw, 2)
    else:
        X_data = X_raw.copy()
        y_data = y_raw.copy()

    if scaler_choice.value == "MinMaxScaler":
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_data)
    elif scaler_choice.value == "StandardScaler":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_data)
    else:
        scaler = None
        X_scaled = X_data.copy()
    return X_data, X_scaled, scaler, y_data


@app.cell
def _(
    X_scaled,
    k_slider,
    mo,
    new_height,
    new_vision,
    new_weight,
    np,
    outlier_switch,
    scaler,
    scaler_choice,
    y_data,
):
    new_person = np.array([[new_height.value, new_weight.value, new_vision.value]])
    new_scaled = scaler.transform(new_person) if scaler else new_person

    dists = np.linalg.norm(X_scaled - new_scaled, axis=1)
    sorted_idx = np.argsort(dists)
    top_labels = y_data[sorted_idx[: k_slider.value]]
    healthy = int((top_labels == 1).sum())
    unhealthy = int((top_labels == 2).sum())
    prediction = 1 if healthy > unhealthy else 2

    label = "健康" if prediction == 1 else "不健康"
    outlier_tag = (
        ' · <span style="background:#fef3c7;padding:1px 6px;border-radius:3px;color:#92400e;">异常</span>'
        if outlier_switch.value else ""
    )
    bg_color = "#e8f5e9" if prediction == 1 else "#ffebee"
    border_color = "#2ca02c" if prediction == 1 else "#d62728"
    text_color = "#166534" if prediction == 1 else "#991b1b"

    mo.md(
        f'<div style="padding:6px 12px;background:{bg_color};'
        f'border-left:4px solid {border_color};border-radius:4px;'
        f'font-size:14px;line-height:1.4;color:{text_color};">'
        f"<strong>{label}</strong> · 健康 {healthy} / 不健康 {unhealthy}"
        f" · k=<code>{k_slider.value}</code>"
        f" · <code>{scaler_choice.value}</code>{outlier_tag}"
        "</div>"
    )
    return dists, new_scaled, sorted_idx


@app.cell
def _(
    X_data,
    X_scaled,
    new_height,
    new_scaled,
    new_weight,
    plt,
    scaler_choice,
    y_data,
):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    colors = ["#2ca02c" if c == 1 else "#d62728" for c in y_data]

    axes[0].scatter(X_data[:, 0], X_data[:, 1], c=colors, s=80, edgecolors="k")
    axes[0].scatter(new_height.value, new_weight.value, c="blue", s=180,
                    marker="*", edgecolors="k", linewidths=1.2)
    axes[0].set(xlabel="height", ylabel="weight", title="Raw")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=colors, s=80, edgecolors="k")
    axes[1].scatter(new_scaled[0, 0], new_scaled[0, 1], c="blue", s=180,
                    marker="*", edgecolors="k", linewidths=1.2)
    axes[1].set(xlabel="height'", ylabel="weight'", title=f"After {scaler_choice.value}")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig
    return


@app.cell
def _(
    X_data,
    X_scaled,
    dists,
    k_slider,
    mo,
    np,
    scaler,
    scaler_choice,
    sorted_idx,
    y_data,
):
    # ---- 距离明细（top-k 用黄色高亮背景） ----
    dist_rows = []
    for rank, i in enumerate(sorted_idx, start=1):
        h, w, v = X_data[i]
        kind = "健康" if y_data[i] == 1 else "不健康"
        if rank <= k_slider.value:
            dist_cell = (
                f'<span style="background:#fef3c7;padding:1px 6px;border-radius:3px;'
                f'color:#92400e;font-weight:600;">{dists[i]:.3f}</span>'
            )
        else:
            dist_cell = f"{dists[i]:.3f}"
        dist_rows.append(
            f"| {rank} | {h:.0f} | {w:.0f} | {v:.1f} | {kind} | {dist_cell} |"
        )
    dist_md = (
        "| 排名 | 身高 | 体重 | 视力 | 标签 | 距离 |\n"
        "|---|---|---|---|---|---|\n"
        + "\n".join(dist_rows)
        + f"\n\n> 黄色高亮 = 前 k={k_slider.value} 个最近邻"
    )

    # ---- 缩放参数 ----
    if scaler_choice.value == "MinMaxScaler":
        params_md = (
            "| 特征 | min | max | 跨度 |\n"
            "|---|---|---|---|\n"
            f"| 身高 | {scaler.data_min_[0]:.2f} | {scaler.data_max_[0]:.2f} | {scaler.data_range_[0]:.2f} |\n"
            f"| 体重 | {scaler.data_min_[1]:.2f} | {scaler.data_max_[1]:.2f} | {scaler.data_range_[1]:.2f} |\n"
            f"| 视力 | {scaler.data_min_[2]:.2f} | {scaler.data_max_[2]:.2f} | {scaler.data_range_[2]:.2f} |\n"
            "\n公式 `x' = (x − min) / (max − min)`"
        )
    elif scaler_choice.value == "StandardScaler":
        params_md = (
            "| 特征 | μ | σ |\n"
            "|---|---|---|\n"
            f"| 身高 | {scaler.mean_[0]:.2f} | {np.sqrt(scaler.var_[0]):.2f} |\n"
            f"| 体重 | {scaler.mean_[1]:.2f} | {np.sqrt(scaler.var_[1]):.2f} |\n"
            f"| 视力 | {scaler.mean_[2]:.2f} | {np.sqrt(scaler.var_[2]):.2f} |\n"
            "\n公式 `x' = (x − μ) / σ`"
        )
    else:
        params_md = "未缩放——直接用原始量纲"

    # ---- 缩放前后对比 ----
    compare_rows = "\n".join([
        (
            f"| {i+1} | {X_data[i, 0]:.0f} | {X_data[i, 1]:.0f} | {X_data[i, 2]:.1f} | "
            f"{X_scaled[i, 0]:.3f} | {X_scaled[i, 1]:.3f} | {X_scaled[i, 2]:.3f} | "
            f"{'健康' if y_data[i] == 1 else '不健康'} |"
        )
        for i in range(len(X_data))
    ])
    compare_md = (
        "| # | 身高 | 体重 | 视力 | 身高' | 体重' | 视力' | 标签 |\n"
        "|---|---|---|---|---|---|---|---|\n"
        + compare_rows
    )

    detail_section_style = (
        "border-left:3px solid #6366f1;padding:2px 10px;"
        "font-weight:600;font-size:14px;margin-bottom:4px;"
    )
    side_by_side = mo.hstack(
        [
            mo.vstack([
                mo.md(f'<div style="{detail_section_style}">距离明细</div>'),
                mo.md(dist_md),
            ]),
            mo.vstack([
                mo.md(f'<div style="{detail_section_style}">缩放前后对比</div>'),
                mo.md(compare_md),
            ]),
        ],
        widths=[1, 1.2],
        justify="space-between",
        gap=2,
    )

    mo.vstack([
        side_by_side,
        mo.accordion(
            {
                f"缩放参数（{scaler_choice.value}）": mo.md(params_md),
                "玩法建议": mo.md(
                    "1. 基线：默认 → 看预测\n\n"
                    "2. 量纲独裁：保持 `None`，调体重 50→130 → 预测翻转\n\n"
                    "3. 切 `MinMax` / `Standard` → 看距离重排\n\n"
                    "4. 异常值 + `MinMax` → 崩；异常值 + `Standard` → 稳\n\n"
                    "5. k=1 vs k=7 → 投票稳定性"
                ),
            },
            multiple=True,
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
