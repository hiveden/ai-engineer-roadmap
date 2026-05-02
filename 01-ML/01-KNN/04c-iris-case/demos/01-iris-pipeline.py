"""
鸢尾花完整 ML pipeline · KNN + 标准化实战

互动：拖 k / test_size / random_state、切换标准化、选 2 维投影看决策边界
对照：4 维全特征训练 vs 当前 2 维投影训练，准确率差距体现"维度信息量"

跑：marimo edit 01-iris-pipeline.py
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
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        classification_report,
    )

    return (
        KNeighborsClassifier,
        StandardScaler,
        accuracy_score,
        alt,
        classification_report,
        confusion_matrix,
        load_iris,
        mo,
        np,
        pd,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(
        """
    # 鸢尾花完整 ML Pipeline · KNN 实战

    > 6 步标准流程：**加载 → 拆分 → 标准化 → 训练 → 评估 → 预测**
    >
    > 你能调的：训练/测试比例、k、random_state、标准化开关、2D 投影特征。
    > 你能看的：准确率、混淆矩阵、分类报告、决策边界（2D 投影）。
        """
    )
    return


@app.cell
def _(load_iris):
    iris = load_iris()
    feature_names = list(iris.feature_names)  # 4 个英文名
    feature_names_zh = ["花萼长 sepal_len", "花萼宽 sepal_wid", "花瓣长 petal_len", "花瓣宽 petal_wid"]
    target_names = list(iris.target_names)  # ['setosa', 'versicolor', 'virginica']
    X_full = iris.data  # (150, 4)
    y_full = iris.target  # (150,)
    return X_full, feature_names_zh, target_names, y_full


@app.cell
def _(X_full, feature_names_zh, mo, target_names, y_full):
    mo.md(
        f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px 14px;font-size:13px;line-height:1.7;">
<strong>数据集</strong>：sklearn `load_iris()` · 经典 ML hello world<br>
• 样本量 <code>N={len(X_full)}</code> · 4 个特征 · 3 类（每类 {(y_full == 0).sum()}/{(y_full == 1).sum()}/{(y_full == 2).sum()}）<br>
• 特征：{" / ".join(feature_names_zh)}<br>
• 类别：<span style="color:#dc2626;">{target_names[0]}</span> · <span style="color:#16a34a;">{target_names[1]}</span> · <span style="color:#2563eb;">{target_names[2]}</span>
</div>
        """
    )
    return


@app.cell
def _(feature_names_zh, mo):
    section_style = (
        "border-left:2px solid #6366f1;padding:1px 8px;"
        "font-weight:600;font-size:12px;color:#475569;margin-bottom:2px;"
    )
    test_size = mo.ui.slider(0.1, 0.5, value=0.3, step=0.05, label="test_size 测试集占比")
    k_slider = mo.ui.slider(1, 30, value=5, step=2, label="k 邻居数")
    seed = mo.ui.slider(0, 100, value=22, step=1, label="random_state")
    scale_switch = mo.ui.switch(value=True, label="StandardScaler 标准化")

    feat_x = mo.ui.dropdown(options=feature_names_zh, value=feature_names_zh[2], label="2D 投影 X")
    feat_y = mo.ui.dropdown(options=feature_names_zh, value=feature_names_zh[3], label="2D 投影 Y")

    mo.hstack(
        [
            mo.vstack([
                mo.md(f'<div style="{section_style}">数据拆分</div>'),
                test_size, seed,
            ]),
            mo.vstack([
                mo.md(f'<div style="{section_style}">KNN 参数</div>'),
                k_slider, scale_switch,
            ]),
            mo.vstack([
                mo.md(f'<div style="{section_style}">2D 投影特征</div>'),
                feat_x, feat_y,
            ]),
        ],
        widths=[1, 1, 1],
        justify="space-around",
    )
    return feat_x, feat_y, k_slider, scale_switch, seed, test_size


@app.cell
def _(
    KNeighborsClassifier,
    StandardScaler,
    X_full,
    accuracy_score,
    k_slider,
    scale_switch,
    seed,
    test_size,
    train_test_split,
    y_full,
):
    # 4 维全特征训练（与下面 2D 投影做对比）
    Xtr, Xte, ytr, yte = train_test_split(
        X_full, y_full,
        test_size=test_size.value,
        random_state=seed.value,
        stratify=y_full,
    )
    if scale_switch.value:
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
    else:
        Xtr_s, Xte_s = Xtr, Xte
    model_4d = KNeighborsClassifier(n_neighbors=k_slider.value)
    model_4d.fit(Xtr_s, ytr)
    yhat_4d = model_4d.predict(Xte_s)
    acc_4d = float(accuracy_score(yte, yhat_4d))
    return Xtr, Xte, acc_4d, scaler, yhat_4d, ytr, yte


@app.cell
def _(
    KNeighborsClassifier,
    StandardScaler,
    X_full,
    accuracy_score,
    feat_x,
    feat_y,
    feature_names_zh,
    k_slider,
    scale_switch,
    seed,
    test_size,
    train_test_split,
    y_full,
):
    # 仅 2 维投影训练（演示决策边界 + 与 4D 对比）
    ix = feature_names_zh.index(feat_x.value)
    iy = feature_names_zh.index(feat_y.value)
    if ix == iy:
        # 防呆：两个轴选了同一个，强制 X 用 idx, Y 用 idx+1
        iy = (ix + 1) % 4
    X2 = X_full[:, [ix, iy]]
    Xtr2, Xte2, ytr2, yte2 = train_test_split(
        X2, y_full,
        test_size=test_size.value,
        random_state=seed.value,
        stratify=y_full,
    )
    if scale_switch.value:
        scaler2 = StandardScaler()
        Xtr2_s = scaler2.fit_transform(Xtr2)
        Xte2_s = scaler2.transform(Xte2)
    else:
        scaler2 = None
        Xtr2_s, Xte2_s = Xtr2, Xte2
    model_2d = KNeighborsClassifier(n_neighbors=k_slider.value)
    model_2d.fit(Xtr2_s, ytr2)
    yhat_2d = model_2d.predict(Xte2_s)
    acc_2d = float(accuracy_score(yte2, yhat_2d))
    return (
        X2,
        Xtr2_s,
        Xte2_s,
        acc_2d,
        ix,
        iy,
        model_2d,
        scaler2,
        yhat_2d,
        ytr2,
        yte2,
    )


@app.cell
def _(acc_2d, acc_4d, k_slider, mo, scale_switch):
    delta = (acc_4d - acc_2d) * 100
    if delta > 5:
        diff_msg = f"<span style='color:#dc2626;'>⚠️ 4D 比 2D 高 {delta:.1f} pp，**说明剩余 2 维有信息**</span>"
    elif delta > 0:
        diff_msg = f"<span style='color:#16a34a;'>✓ 4D 比 2D 高 {delta:.1f} pp，但差距小</span>"
    else:
        diff_msg = f"<span style='color:#16a34a;'>2D 已经够（运气或这两维信息量大）</span>"
    mo.md(
        f"""
<div style="padding:8px 14px;background:#eff6ff;border-left:4px solid #2563eb;border-radius:4px;font-size:14px;">
<strong>测试集准确率对比</strong> · k=<code>{k_slider.value}</code> · 标准化 <code>{'ON' if scale_switch.value else 'OFF'}</code><br>
• <strong>4 维全特征</strong>（标准训练）：<code>{acc_4d:.1%}</code><br>
• <strong>2 维投影特征</strong>（仅当前可视化的两维）：<code>{acc_2d:.1%}</code><br>
{diff_msg}
</div>
        """
    )
    return


@app.cell
def _(
    X2,
    Xtr2_s,
    alt,
    feat_x,
    feat_y,
    model_2d,
    np,
    pd,
    scale_switch,
    scaler2,
    target_names,
    ytr2,
    yte2,
    Xte2_s,
):
    # 决策边界（2D 投影模型）
    # 用标准化后的特征空间生成网格 → 反标准化回原始特征空间显示
    if scale_switch.value:
        xmin, xmax = float(Xtr2_s[:, 0].min()) - 0.5, float(Xtr2_s[:, 0].max()) + 0.5
        ymin, ymax = float(Xtr2_s[:, 1].min()) - 0.5, float(Xtr2_s[:, 1].max()) + 0.5
    else:
        xmin = float(X2[:, 0].min()) - 0.5
        xmax = float(X2[:, 0].max()) + 0.5
        ymin = float(X2[:, 1].min()) - 0.5
        ymax = float(X2[:, 1].max()) + 0.5
    n_grid = 60
    gx = np.linspace(xmin, xmax, n_grid)
    gy = np.linspace(ymin, ymax, n_grid)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.c_[GX.ravel(), GY.ravel()]
    grid_pred = model_2d.predict(grid_pts)
    # 反标准化回原始特征空间显示坐标
    if scale_switch.value:
        grid_pts_disp = scaler2.inverse_transform(grid_pts)
    else:
        grid_pts_disp = grid_pts
    step_x_disp = (grid_pts_disp[:, 0].max() - grid_pts_disp[:, 0].min()) / (n_grid - 1)
    step_y_disp = (grid_pts_disp[:, 1].max() - grid_pts_disp[:, 1].min()) / (n_grid - 1)
    df_mesh = pd.DataFrame({
        "x": grid_pts_disp[:, 0],
        "y": grid_pts_disp[:, 1],
        "x_end": grid_pts_disp[:, 0] + step_x_disp,
        "y_end": grid_pts_disp[:, 1] + step_y_disp,
        "pred": [target_names[p] for p in grid_pred],
    })

    # 训练点 + 测试点
    Xtr2_disp = scaler2.inverse_transform(Xtr2_s) if scale_switch.value else Xtr2_s
    Xte2_disp = scaler2.inverse_transform(Xte2_s) if scale_switch.value else Xte2_s
    df_train = pd.DataFrame({
        "x": Xtr2_disp[:, 0],
        "y": Xtr2_disp[:, 1],
        "label": [target_names[c] for c in ytr2],
        "kind": ["train"] * len(ytr2),
    })
    df_test = pd.DataFrame({
        "x": Xte2_disp[:, 0],
        "y": Xte2_disp[:, 1],
        "label": [target_names[c] for c in yte2],
        "kind": ["test"] * len(yte2),
    })

    palette = alt.Scale(
        domain=target_names,
        range=["#dc2626", "#16a34a", "#2563eb"],
    )

    region = alt.Chart(df_mesh).mark_rect(opacity=0.35, stroke=None).encode(
        x=alt.X("x:Q", title=feat_x.value),
        y=alt.Y("y:Q", title=feat_y.value),
        x2="x_end:Q", y2="y_end:Q",
        color=alt.Color("pred:N", scale=palette, legend=None),
    )
    train_pts = alt.Chart(df_train).mark_circle(
        size=80, stroke="black", strokeWidth=0.6, opacity=0.85,
    ).encode(
        x="x:Q", y="y:Q",
        color=alt.Color("label:N", scale=palette, legend=alt.Legend(title="类别")),
        tooltip=["x:Q", "y:Q", "label:N", "kind:N"],
    )
    test_pts = alt.Chart(df_test).mark_point(
        size=160, shape="cross", strokeWidth=2.5, filled=False,
    ).encode(
        x="x:Q", y="y:Q",
        color=alt.Color("label:N", scale=palette, legend=None),
        tooltip=["x:Q", "y:Q", "label:N", "kind:N"],
    )

    chart = (region + train_pts + test_pts).properties(
        width=560, height=420,
        title=f"2D 决策边界（仅用 {feat_x.value} × {feat_y.value} 训练 KNN）· 圆=训练 · 十字=测试",
    )
    chart
    return


@app.cell
def _(confusion_matrix, mo, pd, target_names, yhat_4d, yte):
    cm = confusion_matrix(yte, yhat_4d)
    n_test = int(cm.sum())
    rows = []
    for i, real in enumerate(target_names):
        for j, pred in enumerate(target_names):
            rows.append({
                "真实": real,
                "预测": pred,
                "count": int(cm[i, j]),
                "is_diag": i == j,
            })
    df_cm = pd.DataFrame(rows)

    import altair as _alt
    heat = _alt.Chart(df_cm).mark_rect(stroke="white", strokeWidth=2).encode(
        x=_alt.X("预测:N", sort=target_names, title="预测类别"),
        y=_alt.Y("真实:N", sort=target_names, title="真实类别"),
        color=_alt.Color(
            "count:Q",
            scale=_alt.Scale(scheme="blues"),
            legend=_alt.Legend(title="样本数"),
        ),
        tooltip=["真实:N", "预测:N", "count:Q"],
    )
    text = _alt.Chart(df_cm).mark_text(fontSize=18, fontWeight="bold").encode(
        x="预测:N", y="真实:N",
        text="count:Q",
        color=_alt.condition(
            "datum.count > 5",
            _alt.value("white"),
            _alt.value("#0f172a"),
        ),
    )
    cm_chart = (heat + text).properties(
        width=320, height=300,
        title=f"4 维全特征 · 测试集混淆矩阵（N={n_test}）",
    )
    mo.vstack([
        mo.md("### 混淆矩阵（4 维模型）"),
        cm_chart,
    ])
    return


@app.cell
def _(classification_report, mo, target_names, yhat_4d, yte):
    rep = classification_report(
        yte, yhat_4d,
        target_names=target_names,
        digits=3,
        zero_division=0,
    )
    mo.md(
        "### 分类报告（4 维模型）\n\n```\n" + rep + "\n```\n\n"
        "**precision**：预测为某类中真正是该类的比例（避免误报）\n\n"
        "**recall**：真正某类中被预测出来的比例（避免漏检）\n\n"
        "**f1-score**：precision 和 recall 的调和平均\n\n"
        "**support**：测试集中该类样本数"
    )
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "📖 这个 demo 在演示什么": mo.md(
                "**完整 ML pipeline 6 步**：\n\n"
                "1. **加载** `load_iris()` 拿数据 + 特征名 + 类别名\n"
                "2. **拆分** `train_test_split(test_size, random_state, stratify=y)` —— "
                "`stratify` 保证训练/测试集每类比例一致\n"
                "3. **标准化** `StandardScaler().fit_transform(Xtr)` —— "
                "**只能在训练集 fit，测试集只 transform**（防数据泄漏）\n"
                "4. **训练** `KNeighborsClassifier(k).fit(Xtr_s, ytr)` —— KNN 实际只是存数据\n"
                "5. **评估** `accuracy_score(yte, yhat)` + 混淆矩阵 + 分类报告\n"
                "6. **预测** `model.predict(new_X)` 输出类别 / `predict_proba` 输出概率"
            ),
            "为什么 stratify=y": mo.md(
                "三类各 50 个，random 拆分可能某类训练只有 30 个、测试占 20 个；"
                "另一类训练 45 个、测试 5 个——比例失衡导致评估有偏。\n\n"
                "`stratify=y` 强制按 y 的比例分层抽样，每类训练/测试比例都是 70/30。"
                "**类别不平衡时这是必做项**。"
            ),
            "为什么标准化对 KNN 关键": mo.md(
                "KNN 算距离时，量纲大的特征会主导。\n\n"
                "iris 4 维量纲都是 cm，差距不大（~1-7 cm），所以标准化开关切换准确率变化不大；"
                "但如果加入「票房」（亿元）+「评分」（0-10）这种**量纲悬殊**的特征，"
                "不标准化时 KNN 几乎只看票房——这就是 04b 章节强调的核心。\n\n"
                "**测试**：把标准化关掉，4 维准确率应该几乎不变（量纲已对齐）；"
                "下章 04b-scaling 会换数据集让差距显著。"
            ),
            "怎么读决策边界图": mo.md(
                "**圆点 = 训练样本**（带黑边），**十字 = 测试样本**。\n\n"
                "**背景色 = 决策区域**：那一片区域内任何点会被预测成对应颜色的类。\n\n"
                "**测试十字落在错色区域 = 该测试样本被分错**——能直观找出哪些点最难分。\n\n"
                "切换 X/Y 轴投影维度，看哪两维最容易分类（一般是花瓣长 × 花瓣宽，setosa 几乎完美隔离）。"
            ),
        },
        multiple=False,
    )
    return


if __name__ == "__main__":
    app.run()
