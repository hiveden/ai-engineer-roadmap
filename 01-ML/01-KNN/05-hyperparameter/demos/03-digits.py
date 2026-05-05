"""
手写数字识别 · 8×8 → 64 维 KNN

互动：拖 pick → 看测试样本 / proba 分布 / top-3 邻居
模型：KNeighborsClassifier(k=3, weights="distance") · 测试集准确率 0.9844

跑：marimo run 03-digits.py --headless --port 2757 --no-token
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/03-digits.grid.json",
    css_file="marimo.css",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier

    return (
        KNeighborsClassifier,
        alt,
        load_digits,
        mo,
        np,
        pd,
        train_test_split,
    )


@app.cell
def _(KNeighborsClassifier, load_digits, train_test_split):
    digits = load_digits()
    Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(
        digits.data, digits.target,
        test_size=0.25, random_state=42, stratify=digits.target,
    )
    digits_model = KNeighborsClassifier(n_neighbors=3, weights="distance")
    digits_model.fit(Xd_tr, yd_tr)
    digits_acc = float(digits_model.score(Xd_te, yd_te))
    return Xd_te, Xd_tr, digits_acc, digits_model, yd_te, yd_tr


@app.cell
def _(mo):
    mo.md(
        """
        # 手写数字 KNN · 8×8 → 64 维

        > sklearn `load_digits` 1797 张 8×8 灰度图（原 MNIST 28×28=784 维的简化版） · 展平 64 维 + 欧氏距离 + k=3 加权投票
        """
    )
    return


@app.cell
def _(Xd_te, Xd_tr, digits_acc, mo):
    mo.md(
        f"""
        <div style="background:#eff6ff;border-left:4px solid #2563eb;border-radius:4px;padding:10px 14px;font-size:14px;">
        <strong>测试集准确率</strong> <code>{digits_acc:.4f}</code>（{int(round(digits_acc * len(Xd_te)))} / {len(Xd_te)} 对，错 {len(Xd_te) - int(round(digits_acc * len(Xd_te)))} 张）
        · 训练集 <code>{len(Xd_tr)}</code> · 测试集 <code>{len(Xd_te)}</code> · k=3 加权欧氏
        </div>
        """
    )
    return


@app.cell
def _(Xd_te, mo):
    pick = mo.ui.slider(0, len(Xd_te) - 1, value=0, step=1, label="测试样本编号")
    pick
    return (pick,)


@app.cell
def _(Xd_te, digits_model, pick, yd_te):
    sample = Xd_te[pick.value].reshape(8, 8)
    true_label = int(yd_te[pick.value])
    pred_label = int(digits_model.predict(Xd_te[pick.value:pick.value + 1])[0])
    proba = digits_model.predict_proba(Xd_te[pick.value:pick.value + 1])[0]
    is_correct = pred_label == true_label
    return is_correct, pred_label, proba, sample, true_label


@app.cell
def _(is_correct, mo, pick, pred_label, true_label):
    _mark = "✓" if is_correct else "✗"
    _color = "#16a34a" if is_correct else "#dc2626"
    mo.md(
        f"""
        <div style="font-size:15px;line-height:1.8;padding:6px 10px;">
        <strong>测试样本 #{pick.value}</strong> · 真实 = <code>{true_label}</code> ·
        预测 = <code style="color:{_color};">{pred_label}</code>
        <span style="color:{_color};font-size:18px;font-weight:bold;">{_mark}</span>
        </div>
        """
    )
    return


@app.cell
def _(alt, pd, sample):
    _rows = []
    for _i in range(8):
        for _j in range(8):
            _rows.append({"row": _i, "col": _j, "v": float(sample[_i, _j])})
    _df_main = pd.DataFrame(_rows)
    main_img = alt.Chart(_df_main).mark_rect().encode(
        x=alt.X("col:O", title=None, axis=None),
        y=alt.Y("row:O", title=None, axis=None, sort="ascending"),
        color=alt.Color(
            "v:Q",
            scale=alt.Scale(scheme="greys", domain=[0, 16]),
            legend=None,
        ),
    ).properties(width=240, height=240)
    main_img
    return


@app.cell
def _(alt, pd, proba, true_label):
    _df_proba = pd.DataFrame({"class": list(range(10)), "prob": proba})
    proba_chart = alt.Chart(_df_proba).mark_bar().encode(
        x=alt.X("class:O", title="预测类别"),
        y=alt.Y("prob:Q", title="概率", scale=alt.Scale(domain=[0, 1])),
        color=alt.condition(
            f"datum['class'] == {true_label}",
            alt.value("#16a34a"),
            alt.value("#94a3b8"),
        ),
        tooltip=["class:O", "prob:Q"],
    ).properties(width=440, height=240)
    proba_chart
    return


@app.cell
def _(mo):
    mo.md("**Top-3 最近邻**（训练集中离当前测试样本最近的 3 个，按欧氏距离）")
    return


@app.cell
def _(Xd_te, Xd_tr, np, pd, pick, yd_tr):
    _dists = np.linalg.norm(Xd_tr - Xd_te[pick.value], axis=1)
    top3 = np.argsort(_dists)[:3]
    nbr_dfs = []
    for _rank, _idx in enumerate(top3):
        _nbr = Xd_tr[_idx].reshape(8, 8)
        for _i in range(8):
            for _j in range(8):
                nbr_dfs.append({
                    "rank": _rank + 1,
                    "label": int(yd_tr[_idx]),
                    "dist": float(_dists[_idx]),
                    "row": _i, "col": _j,
                    "v": float(_nbr[_i, _j]),
                })
    df_nbr = pd.DataFrame(nbr_dfs)
    nbr_meta = pd.DataFrame({
        "rank": [1, 2, 3],
        "label": [int(yd_tr[_i]) for _i in top3],
        "dist": [float(_dists[_i]) for _i in top3],
    })
    return df_nbr, nbr_meta


@app.cell
def _(alt, df_nbr):
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
    ).properties(width=120, height=120)
    nbr_img
    return


@app.cell
def _(mo, nbr_meta):
    _md_lines = "\n".join(
        f"- **#{_r.rank}** 标签 = <code>{_r.label}</code> · 距离 = <code>{_r.dist:.2f}</code>"
        for _r in nbr_meta.itertuples()
    )
    mo.md(_md_lines)
    return


@app.cell
def _(mo):
    # 📐 Grid 布局参考（开发用 · 录屏隐藏）
    mo.md(
        """
        ## 📐 Grid 布局（16:9 · 录屏推荐）

        ```
           0           10                       24
        0  ┌────── 标题（h=3）─────────────────┐
        3  ├────── 准确率卡（h=3）──────────────┤
        6  ├────── pick slider（h=2）───────────┤
        8  ├────── 当前样本说明（h=2）──────────┤
        10 ├── 主图(8×8) ─┬── proba bar ───────┤
        10 │  (h=12)      │  (h=12)             │
        22 ├────── top-3 标题（h=2）────────────┤
        24 ├────── top-3 邻居图（h=8）──────────┤
        32 ├────── 邻居元数据 md（h=3）─────────┤
        ```
        """
    )
    return


if __name__ == "__main__":
    app.run()
