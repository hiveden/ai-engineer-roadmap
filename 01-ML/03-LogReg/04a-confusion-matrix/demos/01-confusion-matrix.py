"""
混淆矩阵交互 · 拖阈值 → 4 格 TP/FN/FP/TN 实时变

互动：
- 拖阈值 τ → 矩阵数字 + 散点图样本归属同步变
- 切场景（疾病筛查 / 反欺诈 / 默认）看 FN / FP 代价非对称
跑：marimo edit 01-confusion-matrix.py --port 2740 --headless --no-token
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/01-confusion-matrix.grid.json",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    return alt, mo, np, pd


@app.cell
def _(mo):
    mo.md(r"""
    # 混淆矩阵 · 一个准确率盖不住两种错

    **TP / FN / FP / TN** 把"真实 × 预测"二维交叉摊开。
    第二字母看预测（P/N），第一字母看对错（T/F）。

    拖阈值 → 格子里的数字实时变；右图每个点的颜色对应它落在哪格。
    """)
    return


@app.cell
def _(mo):
    tau = mo.ui.slider(0.05, 0.95, value=0.5, step=0.05,
                       label="阈值 τ", show_value=True)
    scenario = mo.ui.dropdown(
        options={"默认（FN = FP）": "default",
                 "疾病筛查（FN ≫ FP）": "medical",
                 "反欺诈拦截（FP ≫ FN）": "fraud"},
        value="默认（FN = FP）", label="业务场景")
    show_labels = mo.ui.switch(value=True, label="显示样本标签")
    mo.hstack([tau, scenario, show_labels],
              widths=[2, 1.5, 1], justify="space-around")
    return scenario, show_labels, tau


@app.cell
def _(np):
    # 单一来源：合成预测概率分布（正例分布偏右、负例偏左但有重叠）
    _rng = np.random.default_rng(13)
    _n_pos, _n_neg = 50, 50
    _p_pos = np.clip(_rng.beta(5, 2, _n_pos), 0.02, 0.98)
    _p_neg = np.clip(_rng.beta(2, 5, _n_neg), 0.02, 0.98)
    p_all = np.concatenate([_p_pos, _p_neg])
    y_all = np.array([1] * _n_pos + [0] * _n_neg)
    idx_all = np.arange(len(y_all))
    return idx_all, p_all, y_all


@app.cell
def _(p_all, tau, y_all):
    y_pred = (p_all >= tau.value).astype(int)
    # 4 格分类
    tp = int(((y_all == 1) & (y_pred == 1)).sum())
    fn = int(((y_all == 1) & (y_pred == 0)).sum())
    fp = int(((y_all == 0) & (y_pred == 1)).sum())
    tn = int(((y_all == 0) & (y_pred == 0)).sum())
    n_total = tp + fn + fp + tn
    acc = (tp + tn) / n_total
    return acc, fn, fp, n_total, tn, tp, y_pred


@app.cell
def _(fn, fp, tn, tp):
    # 给每个样本打标记 TP/FN/FP/TN
    def cell_of(y, p):
        if y == 1 and p == 1:
            return "TP"
        if y == 1 and p == 0:
            return "FN"
        if y == 0 and p == 1:
            return "FP"
        return "TN"
    cell_palette = {"TP": "#16a34a", "FN": "#dc2626",
                    "FP": "#ea580c", "TN": "#64748b"}
    cell_counts = {"TP": tp, "FN": fn, "FP": fp, "TN": tn}
    return cell_counts, cell_of, cell_palette


@app.cell
def _(alt, fn, fp, mo, pd, tn, tp):
    # 视图 1：4 格混淆矩阵热图（行真列预 · 用颜色块 + 数字）
    _rows = [
        {"真实": "正例", "预测": "正例", "格": "TP", "n": tp, "解读": "命中"},
        {"真实": "正例", "预测": "负例", "格": "FN", "n": fn, "解读": "漏报"},
        {"真实": "负例", "预测": "正例", "格": "FP", "n": fp, "解读": "误报"},
        {"真实": "负例", "预测": "负例", "格": "TN", "n": tn, "解读": "正确放过"},
    ]
    _df = pd.DataFrame(_rows)
    _rect = alt.Chart(_df).mark_rect(stroke="white", strokeWidth=2).encode(
        x=alt.X("预测:N", sort=["正例", "负例"], title="预测",
                axis=alt.Axis(orient="top", labelFontSize=13)),
        y=alt.Y("真实:N", sort=["正例", "负例"], title="真实",
                axis=alt.Axis(labelFontSize=13)),
        color=alt.Color("格:N",
            scale=alt.Scale(domain=["TP", "FN", "FP", "TN"],
                            range=["#16a34a", "#dc2626", "#ea580c", "#64748b"]),
            legend=None),
    )
    _txt_n = alt.Chart(_df).mark_text(
        fontSize=42, fontWeight="bold", color="white", dy=-10).encode(
        x=alt.X("预测:N", sort=["正例", "负例"]),
        y=alt.Y("真实:N", sort=["正例", "负例"]),
        text="n:Q",
    )
    _txt_lbl = alt.Chart(_df).mark_text(
        fontSize=14, color="white", dy=22).encode(
        x=alt.X("预测:N", sort=["正例", "负例"]),
        y=alt.Y("真实:N", sort=["正例", "负例"]),
        text=alt.Text("解读:N"),
    )
    _txt_code = alt.Chart(_df).mark_text(
        fontSize=12, color="white", dy=42, opacity=0.85).encode(
        x=alt.X("预测:N", sort=["正例", "负例"]),
        y=alt.Y("真实:N", sort=["正例", "负例"]),
        text="格:N",
    )
    chart_matrix = (_rect + _txt_n + _txt_lbl + _txt_code).properties(
        width=320, height=320)
    return (chart_matrix,)


@app.cell
def _(alt, cell_of, idx_all, p_all, pd, show_labels, tau, y_all, y_pred):
    # 视图 2：样本按预测概率排开（一维 strip + jitter），按格染色，阈值竖线
    _rng = __import__("numpy").random.default_rng(2)
    _df = pd.DataFrame({
        "id": idx_all,
        "p": p_all,
        "真实": ["正" if y == 1 else "负" for y in y_all],
        "格": [cell_of(y_all[i], y_pred[i]) for i in range(len(idx_all))],
        "y_pos": [
            (0.55 + 0.4 * _rng.random()) if y_all[i] == 1
            else (0.0 + 0.4 * _rng.random())
            for i in range(len(idx_all))
        ],
    })
    _pts = alt.Chart(_df).mark_circle(size=120, stroke="white",
                                       strokeWidth=1.2, opacity=0.95).encode(
        x=alt.X("p:Q", scale=alt.Scale(domain=[0, 1]),
                title="预测概率 P(y=1)"),
        y=alt.Y("y_pos:Q", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(values=[0.2, 0.75], labelExpr=
                              "datum.value > 0.5 ? '真实正例' : '真实负例'",
                              title=None)),
        color=alt.Color("格:N",
            scale=alt.Scale(domain=["TP", "FN", "FP", "TN"],
                            range=["#16a34a", "#dc2626", "#ea580c", "#64748b"]),
            legend=alt.Legend(title="混淆格", orient="right")),
        tooltip=["id:Q", alt.Tooltip("p:Q", format=".3f"), "真实:N", "格:N"],
    )
    _vl = alt.Chart(pd.DataFrame({"x": [tau.value]})).mark_rule(
        stroke="#0f172a", strokeWidth=2.5, strokeDash=[6, 4]).encode(x="x:Q")
    _lbl = alt.Chart(pd.DataFrame({
        "x": [tau.value], "y": [0.97], "label": [f"τ = {tau.value:.2f}"]
    })).mark_text(align="center", fontSize=12, fontWeight="bold",
                  color="#0f172a", dy=-5).encode(
        x="x:Q", y="y:Q", text="label:N")
    _sep = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
        stroke="#cbd5e1", strokeWidth=1).encode(y="y:Q")
    if show_labels.value:
        chart_strip = (_pts + _vl + _lbl + _sep).properties(
            width=540, height=320)
    else:
        chart_strip = (_pts.encode(tooltip=alt.value(None))
                       + _vl + _lbl + _sep).properties(
            width=540, height=320)
    return (chart_strip,)


@app.cell
def _(chart_matrix, chart_strip, mo):
    mo.hstack([chart_matrix, chart_strip],
              widths=[1, 1.5], justify="space-around")
    return


@app.cell
def _(acc, cell_counts, mo, n_total, scenario, tau):
    # 状态卡：准确率 + 业务代价
    tp_, fn_, fp_, tn_ = (
        cell_counts["TP"], cell_counts["FN"],
        cell_counts["FP"], cell_counts["TN"]
    )
    cost_w = {"default": (1, 1), "medical": (10, 1), "fraud": (1, 10)}[scenario.value]
    cost_label = {"default": "FN/FP 代价相同",
                  "medical": "1 漏诊 = 10 误诊（医疗）",
                  "fraud": "1 误报 = 10 漏报（反欺诈）"}[scenario.value]
    business_cost = fn_ * cost_w[0] + fp_ * cost_w[1]
    cost_color = ("#16a34a" if business_cost <= 10
                  else "#eab308" if business_cost <= 30 else "#dc2626")
    _box = "padding:10px 14px;border-radius:6px;font-size:13px;line-height:1.7;"
    mo.md(f"""
<div style="display:flex;gap:10px;margin:6px 0;">
  <div style="flex:1;background:#f8fafc;border:1px solid #e2e8f0;{_box}">
    <strong>当前阈值</strong> τ = <code>{tau.value:.2f}</code><br>
    准确率 = <strong style="color:#2563eb;">{acc:.1%}</strong>
    （{tp_+tn_} / {n_total}）<br>
    对角线（对）= {tp_+tn_} · 反对角线（错）= {fn_+fp_}
  </div>
  <div style="flex:1;background:#fef2f2;border-left:4px solid #dc2626;{_box}">
    <strong>错误结构</strong><br>
    漏报 FN = <strong style="color:#dc2626;">{fn_}</strong>
    （把病人判健康）<br>
    误报 FP = <strong style="color:#ea580c;">{fp_}</strong>
    （把健康判病人）
  </div>
  <div style="flex:1.2;background:#fff7ed;border-left:4px solid {cost_color};{_box}">
    <strong>业务代价</strong>（{cost_label}）<br>
    总代价 = <strong style="color:{cost_color};">{business_cost}</strong>
    = {fn_}×{cost_w[0]} + {fp_}×{cost_w[1]}<br>
    <em>调阈值看代价怎么变 → 阈值不该默认 0.5</em>
  </div>
</div>
""")
    return


@app.cell
def _(mo):
    mo.accordion({
        "命名规则口诀：第二字母看预测，第一字母看对错": mo.md(r"""
- **TP** = 预测 P，对了 → 实际也是正
- **FN** = 预测 N，错了 → 实际是正（漏报）
- **FP** = 预测 P，错了 → 实际是负（误报）
- **TN** = 预测 N，对了 → 实际也是负

读 FN：先看 N（模型说"负"），再看 F（错了，所以实际是正），即"把正例判成负"= 漏报。

约定：**正例 = 关心的少数类 / 高代价类**——癌症 = 恶性，反欺诈 = 欺诈，缺陷检测 = 缺陷。
"""),
        "为什么准确率不够看": mo.md(r"""
反例：体检人群癌症患病率 1%。
"永远输出健康"的笨模型在 100 人测试集上：
- 99 个真健康全猜对（TN=99）
- 1 个真癌症漏掉（FN=1）
- 准确率 = 99% ✓

**业务上是灾难**：所有病人 100% 漏诊。准确率把"对在哪类、错在哪类"全抹平了。
混淆矩阵 4 格分开摆，错误结构一目了然。

把上面 demo 阈值拖到 0.95：FN 暴涨（变成"不敢判正例"），但因为正负样本平衡，
准确率不一定垮——准确率不敏感的盲区。
"""),
        "sklearn 的 confusion_matrix 行列顺序坑": mo.md(r"""
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred, labels=["恶性", "良性"])
#               预测 恶性  预测 良性
# 真实 恶性        TP        FN
# 真实 良性        FP        TN
```

**坑 1**：不传 `labels` 参数 → sklearn 按字母序排，行列顺序可能和你脑里的"正例在前"不一致。
**坑 2**：标签是字符串时（"恶性"/"良性"）必须显式 `labels=` 才能锁定正例位置。
**坑 3**：`pos_label` 在 `precision_score`/`recall_score` 里要求 = 你的正例标签。
"""),
        "调阈值的代价权衡（业务驱动）": mo.md(r"""
**疾病筛查**（漏诊代价 ≫ 误诊）：
- 阈值低（0.2-0.3） → 多预测正例 → FN ↓ FP ↑
- "宁可错杀一千，不能漏掉一个"

**反欺诈拦截**（误报代价 ≫ 漏报）：
- 阈值高（0.7-0.9） → 少预测正例 → FP ↓ FN ↑
- 误拦正常用户的成本（投诉、流失）大于偶尔放过欺诈

**生产代码不要靠 `predict()` 默认 0.5**：
```python
proba = clf.predict_proba(X)[:, 1]
y_pred = (proba >= 0.3).astype(int)  # 自定义阈值
```
"""),
    }, multiple=False)
    return


@app.cell
def _(mo):
    mo.md("""
## Grid 布局参考（16:9 · 36 行）

```
   0           12          24
0  ┌───── 标题 h=4 全宽 ─────┐
4  ├ τ ┬ scenario ┬ labels h=4 ─┤
8  ├──── 状态卡 h=5 全宽 ────┤
13 ├ 矩阵热图(10) ┬ 散点 strip(14) ─┤  ← h=18
31 ├──── accordion h=5 全宽 ────┤
36
```
""")
    return


if __name__ == "__main__":
    app.run()
