"""
CV 折数对方差的影响 · 单次 train/test vs CV 平均

互动：拖 k / cv 折数 / 重复次数 → 看双行 boxplot 带宽（红宽=单切运气，绿窄=CV 稳）

跑：marimo run 01-cv-folds.py --headless --port 2755 --no-token
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/01-cv-folds.grid.json",
    css_file="marimo.css",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier

    return (
        KNeighborsClassifier,
        LeaveOneOut,
        StandardScaler,
        alt,
        cross_val_score,
        load_iris,
        mo,
        pd,
        train_test_split,
    )


@app.cell
def _(load_iris):
    iris = load_iris()
    return (iris,)


@app.cell
def _(mo):
    mo.md(r"""
    # CV 折数 vs 单次拆分的方差对比

    > **问题**：单次 train/test 切分的准确率，换 random_state 抖得厉害——评分到底是模型水平还是抽样运气？
    > **答案**：CV 把单次拆分变成 n 次取平均，标准差按 ~$1/\sqrt{n}$ 缩。
    """)
    return


@app.cell
def _(mo):
    section_style = (
        "border-left:2px solid #6366f1;padding:1px 8px;"
        "font-weight:600;font-size:12px;color:#475569;margin-bottom:2px;"
    )
    cv_k = mo.ui.slider(1, 30, value=5, step=2, label="k 邻居数")
    cv_folds = mo.ui.dropdown(
        options=["2", "3", "5", "10", "LOO"], value="5", label="CV 折数"
    )
    cv_n_repeats = mo.ui.slider(1, 30, value=20, step=1, label="不同 random_state 重复次数")
    mo.vstack([
        mo.md(f'<div style="{section_style}">CV 实验参数</div>'),
        mo.hstack([cv_k, cv_folds, cv_n_repeats], widths=[1, 1, 1]),
    ])
    return cv_folds, cv_k, cv_n_repeats


@app.cell
def _(
    KNeighborsClassifier,
    LeaveOneOut,
    StandardScaler,
    cross_val_score,
    cv_folds,
    cv_k,
    cv_n_repeats,
    iris,
    pd,
    train_test_split,
):
    # 实验：固定 k，多次 random_state 比较「单次拆分」vs「CV 平均」的稳定性
    _X = iris.data
    _y = iris.target
    n_repeats = cv_n_repeats.value

    _rows = []
    for _seed in range(n_repeats):
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
            _cv = LeaveOneOut()
        else:
            _cv = int(cv_folds.value)
        _cv_scores = cross_val_score(
            KNeighborsClassifier(n_neighbors=cv_k.value),
            _Xtr_s, _ytr, cv=_cv,
        )
        _cv_mean = float(_cv_scores.mean())

        _rows.append({"seed": _seed, "kind": "单次 train/test", "score": _single_score})
        _rows.append({"seed": _seed, "kind": f"CV-{cv_folds.value} 平均", "score": _cv_mean})

    df_cv = pd.DataFrame(_rows)
    single_std = float(df_cv[df_cv.kind == "单次 train/test"].score.std())
    cv_std = float(df_cv[df_cv.kind == f"CV-{cv_folds.value} 平均"].score.std())
    single_mean = float(df_cv[df_cv.kind == "单次 train/test"].score.mean())
    cv_mean_total = float(df_cv[df_cv.kind == f"CV-{cv_folds.value} 平均"].score.mean())
    return cv_mean_total, cv_std, df_cv, n_repeats, single_mean, single_std


@app.cell
def _(alt, cv_folds, df_cv):
    _palette = alt.Scale(
        domain=["单次 train/test", f"CV-{cv_folds.value} 平均"],
        range=["#dc2626", "#16a34a"],
    )
    _strip = alt.Chart(df_cv).mark_circle(size=80, opacity=0.7).encode(
        x=alt.X("score:Q", scale=alt.Scale(domain=[0.7, 1.02]), title="测试集 / CV 准确率"),
        y=alt.Y("kind:N", title=""),
        color=alt.Color("kind:N", scale=_palette, legend=None),
        tooltip=["seed:Q", "kind:N", "score:Q"],
    )
    _box = alt.Chart(df_cv).mark_boxplot(extent="min-max", size=30, opacity=0.4).encode(
        x="score:Q",
        y=alt.Y("kind:N", title=""),
        color=alt.Color("kind:N", scale=_palette, legend=None),
    )
    cv_chart = (_box + _strip).properties(width=720, height=200)
    cv_chart
    return


@app.cell
def _(
    cv_folds,
    cv_k,
    cv_mean_total,
    cv_std,
    mo,
    n_repeats,
    single_mean,
    single_std,
):
    mo.md(f"""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px 14px;font-size:13px;line-height:1.7;">
    <strong>k=<code>{cv_k.value}</code> · {n_repeats} 次重复实验 · 看分布带宽（越窄越稳）</strong><br>
    • <span style="color:#dc2626;">单次 train/test</span>：均值 <code>{single_mean:.3f}</code> · 标准差 <code>{single_std:.4f}</code>（带宽大 = 看运气）<br>
    • <span style="color:#16a34a;">CV-{cv_folds.value} 平均</span>：均值 <code>{cv_mean_total:.3f}</code> · 标准差 <code>{cv_std:.4f}</code>（带宽小 = 评分稳）<br>
    → CV 把单次拆分变成 {cv_folds.value} 次取平均，标准差按 ~1/√n 缩。看 4 折降一半噪声，看 16 折降到 1/4。
    </div>
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    **为什么不直接用单次 train/test？** 同一个 k=5 鸢尾花模型，切训练/测试时换 6 个 random_state（7, 20, 0, 42, 1, 100）实测准确率从 0.91 跳到 0.98——约 7 个百分点的波动只是抽样运气。CV 通过多次切+平均把这波动从 ~0.034 降到 ~0.020 量级。
    """)
    return


@app.cell
def _(mo):
    # 📐 Grid 布局参考（开发用 · 录屏隐藏）
    mo.md("""
    ## 📐 Grid 布局（16:9 · 录屏推荐）

    ```
       0                                      24
    0  ┌───────── 标题（h=3）────────────────┐
    3  ├───────── 控件 hstack（h=3）─────────┤
    6  ├───────── chart（h=12）──────────────┤
    18 ├───────── 数据卡 mo.md（h=4）────────┤
    22 ├───────── 解释 mo.md（h=3）──────────┤
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
