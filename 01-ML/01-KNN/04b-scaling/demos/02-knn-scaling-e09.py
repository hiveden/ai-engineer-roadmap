"""
KNN 缩放对比 · 健康预测案例 · E09 版（v3 · 16:9 横屏 + E09 主轴）

E08 / E09 共数据 + 共组件，分别独立 layout：
  - E08（归一化）：02-knn-scaling.py
  - E09（标准化 + 高斯）：02-knn-scaling-e09.py（本文件）

E09 定制（与 E08 差异）：
  - 默认 scaler = StandardScaler（开机即 baseline-z）
  - 默认新人特征 = 175 / 85 / 1.7（对齐 script.json seg 7 实测数字）
  - 玩法 accordion 改 E09 三步：baseline-z → outlier-z → compare-mm
  - 缩放参数表 StandardScaler 分支加 μ/σ 同涨金句
  - layout 升级 16:9（1280×720 · 36 行 · 2×2 主体网格）

跑：
  marimo edit 02-knn-scaling-e09.py --port 2718              # 调布局
  marimo run  02-knn-scaling-e09.py --port 2752 --no-token --headless  # 录屏
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/02-knn-scaling-e09.grid.json",
    css_file="marimo.css",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    return MinMaxScaler, StandardScaler, mo, np, plt


@app.cell
def _(mo):
    mo.md("# KNN 缩放对比 · 健康预测")
    return


@app.cell
def _(mo):
    k_slider = mo.ui.slider(1, 7, value=3, step=1, label="k", show_value=True)
    scaler_choice = mo.ui.dropdown(
        options=["None", "MinMaxScaler", "StandardScaler"],
        value="StandardScaler",
        label="缩放",
    )
    outlier_switch = mo.ui.switch(value=False, label="异常值 250kg")
    mo.hstack([k_slider, scaler_choice, outlier_switch], widths=[1, 1, 1], gap=2)
    return k_slider, outlier_switch, scaler_choice


@app.cell
def _(mo):
    new_height = mo.ui.slider(160, 195, value=175, step=1, label="身高", show_value=True)
    new_weight = mo.ui.slider(50, 130, value=85, step=1, label="体重", show_value=True)
    new_vision = mo.ui.slider(0.1, 2.0, value=1.7, step=0.1, label="视力", show_value=True)
    mo.hstack([new_height, new_weight, new_vision], widths=[1, 1, 1], gap=2)
    return new_height, new_vision, new_weight


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
        f'<div style="padding:8px 14px;background:{bg_color};'
        f'border-left:4px solid {border_color};border-radius:4px;'
        f'font-size:15px;line-height:1.5;color:{text_color};">'
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
def _(X_data, dists, k_slider, mo, sorted_idx, y_data):
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
        "**距离明细**（黄色高亮 = 前 k 近邻）\n\n"
        "| 排名 | 身高 | 体重 | 视力 | 标签 | 距离 |\n"
        "|---|---|---|---|---|---|\n"
        + "\n".join(dist_rows)
    )
    mo.md(dist_md)
    return


@app.cell
def _(mo, np, scaler, scaler_choice):
    if scaler_choice.value == "MinMaxScaler":
        _params_md = (
            "**缩放参数**（MinMaxScaler · 公式 `x' = (x − min) / (max − min)`）\n\n"
            "| 特征 | min | max | 跨度 |\n"
            "|---|---|---|---|\n"
            f"| 身高 | {scaler.data_min_[0]:.2f} | {scaler.data_max_[0]:.2f} | {scaler.data_range_[0]:.2f} |\n"
            f"| 体重 | {scaler.data_min_[1]:.2f} | {scaler.data_max_[1]:.2f} | {scaler.data_range_[1]:.2f} |\n"
            f"| 视力 | {scaler.data_min_[2]:.2f} | {scaler.data_max_[2]:.2f} | {scaler.data_range_[2]:.2f} |"
        )
    elif scaler_choice.value == "StandardScaler":
        _params_md = (
            "**缩放参数**（StandardScaler · 公式 `x' = (x − μ) / σ`）\n\n"
            "| 特征 | μ | σ |\n"
            "|---|---|---|\n"
            f"| 身高 | {scaler.mean_[0]:.2f} | {np.sqrt(scaler.var_[0]):.2f} |\n"
            f"| 体重 | {scaler.mean_[1]:.2f} | {np.sqrt(scaler.var_[1]):.2f} |\n"
            f"| 视力 | {scaler.mean_[2]:.2f} | {np.sqrt(scaler.var_[2]):.2f} |\n\n"
            "<span style=\"color:#6b7280;font-size:13px;\">"
            "异常值进入：μ 与 σ 同涨 → 尺子变厚，正常样本不被压角落</span>"
        )
    else:
        _params_md = "**缩放参数**\n\n未缩放——直接用原始量纲"
    mo.md(_params_md)
    return


@app.cell
def _(X_data, X_scaled, mo, y_data):
    compare_rows = "\n".join([
        (
            f"| {i+1} | {X_data[i, 0]:.0f} | {X_data[i, 1]:.0f} | {X_data[i, 2]:.1f} | "
            f"{X_scaled[i, 0]:.3f} | {X_scaled[i, 1]:.3f} | {X_scaled[i, 2]:.3f} | "
            f"{'健康' if y_data[i] == 1 else '不健康'} |"
        )
        for i in range(len(X_data))
    ])
    compare_md = (
        "**缩放前后对比**\n\n"
        "| # | 身高 | 体重 | 视力 | 身高' | 体重' | 视力' | 标签 |\n"
        "|---|---|---|---|---|---|---|---|\n"
        + compare_rows
    )
    mo.md(compare_md)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "玩法建议（E09 三步）": mo.md(
                "1. **baseline-z**：默认 `StandardScaler` + 异常 OFF → 看 μ/σ 与预测\n\n"
                "2. **outlier-z**：打开异常开关 → 体重 μ 跳到 103.56 / σ 跳到 53.72；"
                "正常样本仍均匀散在零附近，异常值落在 +2.73σ\n\n"
                "3. **compare-mm**：异常保持 ON，缩放切回 `MinMaxScaler` → "
                "8 个正常样本被挤到 [0, 0.24] 角落 → 与上一步标准化散点对比"
            ),
        },
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
## Grid 布局参考（E09 v3 · 严格 16:9 · 1280×720 · 36 行）

```
   0                   12                   24
 0 ┌────────────── 标题 h=2 ────────────────┐  cell 2
 2 ├── 模型参数 h=4 ──┬── 新人特征 h=4 ─────┤  cell 3 + 4
 6 ├────────────── 预测卡 h=2 ──────────────┤  cell 6
 8 ├ 双散点 Raw|Std    ┬─── 距离明细 ───────┤  cell 7 + 8
   │   h=14            │   h=14             │
22 ├── 缩放参数表 ─────┬── 缩放前后对比 ────┤  cell 9 + 10
   │   μ/σ  h=14       │   h=14             │
36
```

cell 1 (imports) / 5 (计算) / 11 (玩法 accordion) / 12 (本块) → null

横屏叙事配对：
- 第一行内容（散点 + 距离）：视觉证据 + 数字证据
- 第二行内容（参数 + 对比）：尺子是什么 + 尺子套上去后
"""
    )
    return


if __name__ == "__main__":
    app.run()
