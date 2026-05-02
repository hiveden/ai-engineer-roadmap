"""
超参调优 · CV + GridSearchCV + 手写数字 KNN

3 个 tab：
1. CV 折数对评分稳定性的影响（多次 random_state 看方差）
2. GridSearchCV 在 (k, weights, p) 网格上的热力图 + 最优组合
3. 手写数字识别（load_digits 8x8）实时 KNN 预测 + top-k 邻居可视化

跑：marimo edit 01-cv-gridsearch.py
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
    from sklearn.datasets import load_iris, load_digits
    from sklearn.model_selection import (
        train_test_split,
        cross_val_score,
        GridSearchCV,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier

    return (
        GridSearchCV,
        KNeighborsClassifier,
        StandardScaler,
        alt,
        cross_val_score,
        load_digits,
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
    # 超参调优 · CV + GridSearchCV + 手写数字

    > **交叉验证（CV）** 解决「单次拆分运气问题」——多次切分取均值。
    > **GridSearchCV** 解决「超参组合搜索」——网格暴力试 + 每组用 CV 评分。
    > **手写数字** 把上面工具串到真实任务里。
    >
    > 三个 tab 独立体验，建议从左到右走一遍。
        """
    )
    return


@app.cell
def _(load_digits, load_iris):
    iris = load_iris()
    digits = load_digits()
    return digits, iris


# ============================================================
# Tab 1：CV 折数 vs 单次拆分的方差对比
# ============================================================

@app.cell
def _(mo):
    cv_section_style = (
        "border-left:2px solid #6366f1;padding:1px 8px;"
        "font-weight:600;font-size:12px;color:#475569;margin-bottom:2px;"
    )
    cv_k = mo.ui.slider(1, 30, value=5, step=2, label="k 邻居数")
    cv_folds = mo.ui.dropdown(
        options=["2", "3", "5", "10", "LOO"], value="5", label="CV 折数"
    )
    cv_n_repeats = mo.ui.slider(1, 30, value=20, step=1, label="不同 random_state 重复次数")
    cv_controls = mo.vstack([
        mo.md(f'<div style="{cv_section_style}">CV 实验参数</div>'),
        mo.hstack([cv_k, cv_folds, cv_n_repeats], widths=[1, 1, 1]),
    ])
    return cv_controls, cv_folds, cv_k, cv_n_repeats


@app.cell
def _(
    KNeighborsClassifier,
    StandardScaler,
    alt,
    cross_val_score,
    cv_folds,
    cv_k,
    cv_n_repeats,
    iris,
    np,
    pd,
    train_test_split,
):
    # 实验：固定 k，多次 random_state 比较「单次拆分」vs「CV 平均」的稳定性
    _X = iris.data
    _y = iris.target
    _n_repeats = cv_n_repeats.value

    _rows = []
    for _seed in range(_n_repeats):
        _Xtr, _Xte, _ytr, _yte = train_test_split(
            _X, _y, test_size=0.3, random_state=_seed, stratify=_y
        )
        _sc = StandardScaler()
        _Xtr_s = _sc.fit_transform(_Xtr)
        _Xte_s = _sc.transform(_Xte)
        _clf = KNeighborsClassifier(n_neighbors=cv_k.value)
        _clf.fit(_Xtr_s, _ytr)
        _single_score = float(_clf.score(_Xte_s, _yte))

        if cv_folds.value == "LOO":
            _cv = len(_Xtr_s)
        else:
            _cv = int(cv_folds.value)
        _cv_scores = cross_val_score(
            KNeighborsClassifier(n_neighbors=cv_k.value),
            _Xtr_s, _ytr, cv=_cv,
        )
        _cv_mean = float(_cv_scores.mean())

        _rows.append({"seed": _seed, "kind": "单次 train/test", "score": _single_score})
        _rows.append({"seed": _seed, "kind": f"CV-{cv_folds.value} 平均", "score": _cv_mean})

    _df = pd.DataFrame(_rows)
    _single_std = float(_df[_df.kind == "单次 train/test"].score.std())
    _cv_std = float(_df[_df.kind == f"CV-{cv_folds.value} 平均"].score.std())
    _single_mean = float(_df[_df.kind == "单次 train/test"].score.mean())
    _cv_mean_total = float(_df[_df.kind == f"CV-{cv_folds.value} 平均"].score.mean())

    _palette = alt.Scale(
        domain=["单次 train/test", f"CV-{cv_folds.value} 平均"],
        range=["#dc2626", "#16a34a"],
    )
    _strip = alt.Chart(_df).mark_circle(size=80, opacity=0.7).encode(
        x=alt.X("score:Q", scale=alt.Scale(domain=[0.7, 1.02]), title="测试集 / CV 准确率"),
        y=alt.Y("kind:N", title=""),
        color=alt.Color("kind:N", scale=_palette, legend=None),
        tooltip=["seed:Q", "kind:N", "score:Q"],
    )
    _box = alt.Chart(_df).mark_boxplot(extent="min-max", size=30, opacity=0.4).encode(
        x="score:Q",
        y=alt.Y("kind:N", title=""),
        color=alt.Color("kind:N", scale=_palette, legend=None),
    )
    cv_chart = (_box + _strip).properties(
        width=520, height=180,
        title=f"k={cv_k.value} · {_n_repeats} 次重复实验 · 看分布带宽（越窄越稳）",
    )
    cv_summary_md = (
        f"**单次 train/test**：均值 `{_single_mean:.3f}`，标准差 `{_single_std:.4f}`（带宽大 = 看运气）\n\n"
        f"**CV-{cv_folds.value} 平均**：均值 `{_cv_mean_total:.3f}`，标准差 `{_cv_std:.4f}`（带宽小 = 评分稳）\n\n"
        f"→ CV 把单次拆分变成 {cv_folds.value} 次取平均，标准差按 ~$1/\\sqrt{{n}}$ 缩。"
        "看 4 折降一半噪声，看 16 折降到 1/4。"
    )
    return cv_chart, cv_summary_md


@app.cell
def _(cv_chart, cv_controls, cv_summary_md, mo):
    tab1 = mo.vstack([
        cv_controls,
        cv_chart,
        mo.md(cv_summary_md),
        mo.md(
            "**为什么不直接用单次 train/test？** 同一个 k=5 鸢尾花模型，"
            "切训练/测试时换 6 个 random_state（7, 20, 0, 42, 1, 100）"
            "实测准确率从 0.90 跳到 1.00——10 个百分点的波动只是抽样运气。"
            "CV 通过多次切+平均把这波动从 ~0.04 降到 ~0.01 量级。"
        ),
    ])
    return (tab1,)


# ============================================================
# Tab 2：GridSearchCV 热力图
# ============================================================

@app.cell
def _(
    GridSearchCV,
    KNeighborsClassifier,
    StandardScaler,
    alt,
    iris,
    pd,
):
    # 在标准化后的 iris 上做 GridSearchCV
    _X = iris.data
    _y = iris.target
    _sc = StandardScaler()
    _X_s = _sc.fit_transform(_X)

    _param_grid = {
        "n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 29],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    }
    _gs = GridSearchCV(
        KNeighborsClassifier(),
        _param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=1,
    )
    _gs.fit(_X_s, _y)
    _cv_results = _gs.cv_results_

    _grid_rows = []
    for _i in range(len(_cv_results["mean_test_score"])):
        _grid_rows.append({
            "n_neighbors": int(_cv_results["param_n_neighbors"][_i]),
            "weights": _cv_results["param_weights"][_i],
            "p": int(_cv_results["param_p"][_i]),
            "mean": float(_cv_results["mean_test_score"][_i]),
            "std": float(_cv_results["std_test_score"][_i]),
        })
    df_grid = pd.DataFrame(_grid_rows)
    df_grid["combo"] = df_grid["weights"] + "·p=" + df_grid["p"].astype(str)

    best_params = _gs.best_params_
    best_score = float(_gs.best_score_)

    _heat = alt.Chart(df_grid).mark_rect(stroke="white", strokeWidth=1).encode(
        x=alt.X("n_neighbors:O", title="k (n_neighbors)"),
        y=alt.Y("combo:N", title="weights · p", sort=["uniform·p=1", "uniform·p=2", "distance·p=1", "distance·p=2"]),
        color=alt.Color(
            "mean:Q",
            scale=alt.Scale(scheme="viridis", domain=[df_grid["mean"].min(), df_grid["mean"].max()]),
            legend=alt.Legend(title="5 折 CV 准确率"),
        ),
        tooltip=["n_neighbors:O", "weights:N", "p:O", "mean:Q", "std:Q"],
    )
    _text = alt.Chart(df_grid).mark_text(fontSize=11, fontWeight="bold").encode(
        x="n_neighbors:O",
        y=alt.Y("combo:N", sort=["uniform·p=1", "uniform·p=2", "distance·p=1", "distance·p=2"]),
        text=alt.Text("mean:Q", format=".3f"),
        color=alt.condition(
            "datum.mean > 0.95",
            alt.value("#0f172a"),
            alt.value("white"),
        ),
    )
    grid_chart = (_heat + _text).properties(
        width=560, height=200,
        title=f"GridSearchCV(5-fold) on iris · {len(df_grid)} 组超参（每组训练 5 次 = {len(df_grid)*5} 次）",
    )
    return best_params, best_score, df_grid, grid_chart


@app.cell
def _(best_params, best_score, df_grid, grid_chart, mo):
    worst_score = float(df_grid["mean"].min())
    tab2 = mo.vstack([
        grid_chart,
        mo.md(
            f"""
**最优组合**：<code>{best_params}</code> · 5 折 CV 准确率 <code>{best_score:.3f}</code>
**最差组合**：<code>{worst_score:.3f}</code> · 跨度 <code>{(best_score - worst_score) * 100:.1f}</code> pp

**怎么读**：
- **行 = (weights, p) 组合**，**列 = k 值**，**色深 = CV 准确率**
- 横向看：k 太小（1）准确率波动；k 适中（5-15）平稳；k 太大（29）开始下降
- 纵向看：`distance` 加权一般略好或持平 `uniform`；p=1 vs p=2 在 iris 上差距小
- 实战中 GridSearchCV 还会试 `metric` (manhattan / chebyshev / minkowski)、
  `algorithm` (auto / ball_tree / kd_tree) 等更多参数 → 网格指数膨胀
        """
        ),
        mo.accordion({
            "为什么 grid search 训练次数 = grid 大小 × CV 折数": mo.md(
                "GridSearchCV 对 grid 中**每一种**超参组合，都用 cv 折交叉验证评分。\n\n"
                "例：grid 36 组 × cv=5 = **180 次模型训练 + 评估**。\n\n"
                "iris 数据集小训练快，每次 ~ms 级；换成 MNIST 60k 样本训练 5 折就要数分钟。"
                "这是为什么生产中常用 `RandomizedSearchCV`（随机采样）或 Bayesian Optimization 替代暴力网格。"
            ),
            "GridSearchCV 后还要不要再 fit 一次": mo.md(
                "`GridSearchCV` 的 `refit=True`（默认）会**自动**用最优参数在**全训练集**上"
                "重新训练一次，得到最终模型。\n\n"
                "调用 `gs.predict(X_new)` 用的就是这个最终模型，**不是** CV 过程中临时的 5 个子模型。\n\n"
                "如果 `refit=False`，需要你手动从 `gs.best_params_` 取参数自己 fit。"
            ),
        }),
    ])
    return (tab2,)


# ============================================================
# Tab 3：手写数字识别（load_digits 8x8）
# ============================================================

@app.cell
def _(KNeighborsClassifier, digits, train_test_split):
    Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(
        digits.data, digits.target,
        test_size=0.25, random_state=42, stratify=digits.target,
    )
    digits_model = KNeighborsClassifier(n_neighbors=3, weights="distance")
    digits_model.fit(Xd_tr, yd_tr)
    digits_acc = float(digits_model.score(Xd_te, yd_te))
    return Xd_te, Xd_tr, digits_acc, digits_model, yd_te, yd_tr


@app.cell
def _(Xd_te, mo, yd_te):
    pick = mo.ui.slider(0, len(Xd_te) - 1, value=0, step=1, label="测试样本编号")
    return (pick,)


@app.cell
def _(
    Xd_te,
    Xd_tr,
    alt,
    digits_acc,
    digits_model,
    mo,
    np,
    pd,
    pick,
    yd_te,
    yd_tr,
):
    # 当前测试样本
    sample = Xd_te[pick.value].reshape(8, 8)
    true_label = int(yd_te[pick.value])
    pred_label = int(digits_model.predict(Xd_te[pick.value:pick.value + 1])[0])
    proba = digits_model.predict_proba(Xd_te[pick.value:pick.value + 1])[0]

    # 8x8 主图
    rows = []
    for i in range(8):
        for j in range(8):
            rows.append({"row": i, "col": j, "v": float(sample[i, j])})
    df_main = pd.DataFrame(rows)

    main_img = alt.Chart(df_main).mark_rect().encode(
        x=alt.X("col:O", title=None, axis=None),
        y=alt.Y("row:O", title=None, axis=None, sort="ascending"),
        color=alt.Color(
            "v:Q",
            scale=alt.Scale(scheme="greys", domain=[0, 16]),
            legend=None,
        ),
    ).properties(
        width=200, height=200,
        title=f"测试样本 #{pick.value} · 真实={true_label} · 预测={pred_label} {'✓' if pred_label == true_label else '✗'}",
    )

    # top-3 最近邻
    dists = np.linalg.norm(Xd_tr - Xd_te[pick.value], axis=1)
    top3 = np.argsort(dists)[:3]

    nbr_dfs = []
    for rank, idx in enumerate(top3):
        nbr = Xd_tr[idx].reshape(8, 8)
        for i in range(8):
            for j in range(8):
                nbr_dfs.append({
                    "rank": rank + 1,
                    "label": int(yd_tr[idx]),
                    "dist": float(dists[idx]),
                    "row": i, "col": j,
                    "v": float(nbr[i, j]),
                })
    df_nbr = pd.DataFrame(nbr_dfs)

    nbr_img = alt.Chart(df_nbr).mark_rect().encode(
        x=alt.X("col:O", title=None, axis=None),
        y=alt.Y("row:O", title=None, axis=None, sort="ascending"),
        color=alt.Color("v:Q", scale=alt.Scale(scheme="greys", domain=[0, 16]), legend=None),
        column=alt.Column(
            "rank:O",
            title=None,
            header=alt.Header(
                labelExpr="'#'+datum.value",
                labelFontSize=11,
            ),
        ),
    ).properties(width=100, height=100)

    nbr_meta = pd.DataFrame({
        "rank": [1, 2, 3],
        "label": [int(yd_tr[i]) for i in top3],
        "dist": [float(dists[i]) for i in top3],
    })
    nbr_md_lines = "\n".join(
        f"- **#{r.rank}** 标签 = <code>{r.label}</code> · 距离 = <code>{r.dist:.2f}</code>"
        for r in nbr_meta.itertuples()
    )

    # 概率分布柱状
    df_proba = pd.DataFrame({"class": list(range(10)), "prob": proba})
    proba_chart = alt.Chart(df_proba).mark_bar().encode(
        x=alt.X("class:O", title="预测类别"),
        y=alt.Y("prob:Q", title="概率", scale=alt.Scale(domain=[0, 1])),
        color=alt.condition(
            f"datum['class'] == {true_label}",
            alt.value("#16a34a"),  # 真实类绿色
            alt.value("#94a3b8"),
        ),
        tooltip=["class:O", "prob:Q"],
    ).properties(width=380, height=180, title="predict_proba 输出（k=3 加权）")

    tab3 = mo.vstack([
        mo.md(
            f"### 手写数字 KNN (k=3, 加权)\n\n"
            f"**测试集准确率** <code>{digits_acc:.1%}</code> · 测试集大小 <code>{len(Xd_te)}</code> · 训练集 <code>{len(Xd_tr)}</code>"
        ),
        pick,
        mo.hstack([main_img, proba_chart], widths=[1, 1.4], justify="start"),
        mo.md("**Top-3 最近邻**（训练集中离当前测试样本最近的 3 个）"),
        nbr_img,
        mo.md(nbr_md_lines),
    ])
    return (tab3,)


# ============================================================
# 总装
# ============================================================

@app.cell
def _(mo, tab1, tab2, tab3):
    mo.ui.tabs(
        {
            "1️⃣ CV 折数对方差": tab1,
            "2️⃣ GridSearchCV 热力图": tab2,
            "3️⃣ 手写数字 KNN": tab3,
        }
    )
    return


if __name__ == "__main__":
    app.run()
