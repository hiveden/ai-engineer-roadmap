"""
GridSearchCV 热力图 · 36 组超参 × cv=5 = 180 次训练

静态结果图：(weights, p) × n_neighbors 网格上的 5 折 CV 准确率热力图。
最优组合 distance·p=2·k=9 · CV 0.967。

跑：marimo run 02-gridsearch.py --headless --port 2756 --no-token
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/02-gridsearch.grid.json",
    css_file="marimo.css",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    from sklearn.datasets import load_iris
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier

    return (
        GridSearchCV,
        KNeighborsClassifier,
        StandardScaler,
        alt,
        load_iris,
        mo,
        pd,
    )


@app.cell
def _(GridSearchCV, KNeighborsClassifier, StandardScaler, load_iris, pd):
    iris = load_iris()
    _X = iris.data
    _y = iris.target
    _X_s = StandardScaler().fit_transform(_X)

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
    return best_params, best_score, df_grid


@app.cell
def _(mo):
    mo.md(
        """
        # GridSearchCV 热力图 · 36 组 × cv=5 = 180 次训练

        > 网格搜索 = 穷举所有超参组合，每组用 CV 评分 · `best_params_` / `best_score_` / `best_estimator_` 三件套
        """
    )
    return


@app.cell
def _(alt, df_grid):
    _sort = ["uniform·p=1", "uniform·p=2", "distance·p=1", "distance·p=2"]
    _heat = alt.Chart(df_grid).mark_rect(stroke="white", strokeWidth=1).encode(
        x=alt.X("n_neighbors:O", title="k (n_neighbors)"),
        y=alt.Y("combo:N", title="weights · p", sort=_sort),
        color=alt.Color(
            "mean:Q",
            scale=alt.Scale(scheme="viridis", domain=[df_grid["mean"].min(), df_grid["mean"].max()]),
            legend=alt.Legend(title="5 折 CV 准确率"),
        ),
        tooltip=["n_neighbors:O", "weights:N", "p:O", "mean:Q", "std:Q"],
    )
    _text = alt.Chart(df_grid).mark_text(fontSize=11, fontWeight="bold").encode(
        x="n_neighbors:O",
        y=alt.Y("combo:N", sort=_sort),
        text=alt.Text("mean:Q", format=".3f"),
        color=alt.condition(
            "datum.mean > 0.95",
            alt.value("#0f172a"),
            alt.value("white"),
        ),
    )
    grid_chart = (_heat + _text).properties(width="container", height=320)
    grid_chart
    return


@app.cell
def _(best_params, best_score, df_grid, mo):
    worst_score = float(df_grid["mean"].min())
    mo.md(
        f"""
        <div style="background:#eff6ff;border-left:4px solid #2563eb;border-radius:4px;padding:10px 14px;font-size:14px;">
        <strong>最优组合</strong> <code>{best_params}</code> · 5 折 CV 准确率 <code>{best_score:.3f}</code><br>
        <strong>最差组合</strong> <code>{worst_score:.3f}</code> · 跨度 <code>{(best_score - worst_score) * 100:.1f}</code> pp
        </div>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        **怎么读**：行 = (weights, p) 组合，列 = k 值，色深 = CV 准确率。
        横向看：k=1 波动大、k 适中（5-15）平台期、k=29 在 uniform 行明显下降。
        纵向看：`distance` 加权一般略好或持平 `uniform`；p=1 vs p=2 在 iris 上差距小。
        实战还会试 `metric` (manhattan / chebyshev / minkowski)、`algorithm` (auto / ball_tree / kd_tree) 等更多参数 → 网格指数膨胀。
        """
    )
    return


@app.cell
def _(mo):
    mo.accordion(
        {
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
        },
        multiple=False,
    )
    return


@app.cell
def _(mo):
    # 📐 Grid 布局参考（开发用 · 录屏隐藏）
    mo.md(
        """
        ## 📐 Grid 布局（16:9 · 录屏推荐）

        ```
           0                                      24
        0  ┌───────── 标题（h=3）────────────────┐
        3  ├───────── 热力图（h=14）─────────────┤
        17 ├───────── best/worst 卡（h=3）──────┤
        20 ├───────── 怎么读 mo.md（h=3）────────┤
        23 ├───────── accordion（h=3）───────────┤
        ```
        """
    )
    return


if __name__ == "__main__":
    app.run()
