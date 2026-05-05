"""
LR 癌症分类最小 pipeline · sklearn 内置威斯康星乳腺癌数据集

互动：
- 拖测试集比例 / 阈值 → 实时 train + 评估
- 看 coef_ 排行（哪个特征最强 → "恶性"）
跑：marimo edit 02-cancer-pipeline.py --port 2736 --headless --no-token
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/02-cancer-pipeline.grid.json",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score)
    return (LogisticRegression, StandardScaler, accuracy_score, alt,
            f1_score, load_breast_cancer, mo, np, pd, precision_score,
            recall_score, roc_auc_score, train_test_split)


@app.cell
def _(mo):
    mo.md(r"""
    # LR 癌症分类 · 最小完整 pipeline

    威斯康星乳腺癌数据集（569 样本 × 30 特征）。**正例 = 恶性 malignant**。

    步骤：`load → split → StandardScaler.fit_transform → LR.fit → predict_proba → 评估`

    拖测试集比例和阈值看指标怎么变；右图按 |w| 排前 10 个最重要的特征。
    """)
    return


@app.cell
def _(mo):
    test_ratio = mo.ui.slider(0.1, 0.5, value=0.2, step=0.05,
                              label="测试集比例", show_value=True)
    tau = mo.ui.slider(0.1, 0.9, value=0.5, step=0.05,
                       label="阈值 τ", show_value=True)
    log_C = mo.ui.slider(-2.0, 2.0, value=0.0, step=0.25,
                         label="log₁₀(C)", show_value=True)
    mo.hstack([test_ratio, tau, log_C],
              widths=[1, 1, 1], justify="space-around")
    return log_C, tau, test_ratio


@app.cell
def _(load_breast_cancer):
    data = load_breast_cancer()
    feature_names = list(data.feature_names)
    X_full = data.data
    y_full = data.target  # 0=malignant, 1=benign（sklearn 约定）
    # 翻转标签：让正例 = 恶性（业务直觉，少数类 = 关心类）
    # sklearn 的 target=0 是恶性、1 是良性，恰好恶性少（212 vs 357）
    # 我们把 y 反转：恶性=1, 良性=0
    y_full = 1 - y_full
    n_total = len(y_full)
    n_pos = int(y_full.sum())
    n_neg = n_total - n_pos
    return X_full, feature_names, n_neg, n_pos, n_total, y_full


@app.cell
def _(LogisticRegression, StandardScaler, X_full, log_C, test_ratio,
      train_test_split, y_full):
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=test_ratio.value,
        random_state=42, stratify=y_full)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    C = float(10 ** log_C.value)
    clf = LogisticRegression(solver="lbfgs", penalty="l2",
                             C=C, max_iter=2000)
    clf.fit(X_train_s, y_train)
    proba_test = clf.predict_proba(X_test_s)[:, 1]
    coef = clf.coef_.ravel()
    intercept = float(clf.intercept_[0])
    return C, X_test_s, clf, coef, intercept, proba_test, y_test


@app.cell
def _(accuracy_score, f1_score, precision_score, proba_test, recall_score,
      roc_auc_score, tau, y_test):
    y_pred = (proba_test >= tau.value).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_test, proba_test),  # AUC 用 proba 不依赖阈值
    }
    cm_tp = int(((y_test == 1) & (y_pred == 1)).sum())
    cm_fn = int(((y_test == 1) & (y_pred == 0)).sum())
    cm_fp = int(((y_test == 0) & (y_pred == 1)).sum())
    cm_tn = int(((y_test == 0) & (y_pred == 0)).sum())
    return cm_fn, cm_fp, cm_tn, cm_tp, metrics, y_pred


@app.cell
def _(alt, metrics, pd):
    # 视图 1：5 指标条形图
    _df = pd.DataFrame({"指标": list(metrics.keys()),
                        "值": list(metrics.values())})
    _bars = alt.Chart(_df).mark_bar(color="#2563eb").encode(
        x=alt.X("值:Q", scale=alt.Scale(domain=[0, 1]),
                title="测试集得分"),
        y=alt.Y("指标:N", sort=["Accuracy", "Precision", "Recall", "F1", "AUC"],
                title=None),
    )
    _text = alt.Chart(_df).mark_text(align="left", dx=4,
                                      fontSize=12, fontWeight="bold").encode(
        x="值:Q", y="指标:N",
        text=alt.Text("值:Q", format=".3f"))
    chart_metrics = (_bars + _text).properties(width=320, height=240)
    return (chart_metrics,)


@app.cell
def _(alt, coef, feature_names, np, pd):
    # 视图 2：top-10 |w| 特征
    _abs_w = np.abs(coef)
    _idx = np.argsort(_abs_w)[::-1][:10]
    _df = pd.DataFrame({
        "特征": [feature_names[i] for i in _idx],
        "w": [coef[i] for i in _idx],
        "方向": ["→ 恶性" if coef[i] > 0 else "→ 良性" for i in _idx],
    })
    _bars = alt.Chart(_df).mark_bar().encode(
        x=alt.X("w:Q", title="权重 w（标准化特征）"),
        y=alt.Y("特征:N", sort=_df["特征"].tolist(), title=None),
        color=alt.Color("方向:N",
            scale=alt.Scale(domain=["→ 恶性", "→ 良性"],
                            range=["#dc2626", "#16a34a"]),
            legend=alt.Legend(orient="top")),
        tooltip=["特征:N", alt.Tooltip("w:Q", format=".3f"), "方向:N"],
    )
    _zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        stroke="#0f172a", strokeWidth=1).encode(x="x:Q")
    chart_coef = (_bars + _zero).properties(width=440, height=320)
    return (chart_coef,)


@app.cell
def _(chart_coef, chart_metrics, mo):
    mo.hstack([chart_metrics, chart_coef],
              widths=[1, 1.4], justify="space-around")
    return


@app.cell
def _(C, cm_fn, cm_fp, cm_tn, cm_tp, intercept, metrics, mo, n_neg,
      n_pos, n_total, tau, y_test):
    # 状态卡：数据集 + 模型 + 评估
    n_test = len(y_test)
    auc = metrics["AUC"]
    auc_color = "#16a34a" if auc >= 0.95 else (
        "#eab308" if auc >= 0.85 else "#dc2626")
    _box = "padding:10px 14px;border-radius:6px;font-size:13px;line-height:1.7;"
    mo.md(f"""
<div style="display:flex;gap:10px;margin:6px 0;">
  <div style="flex:1;background:#f8fafc;border:1px solid #e2e8f0;{_box}">
    <strong>数据集</strong>（威斯康星乳腺癌）<br>
    总样本 {n_total} = 恶性 <strong style="color:#dc2626;">{n_pos}</strong> +
    良性 <strong style="color:#16a34a;">{n_neg}</strong><br>
    特征 30 维 · 测试集 {n_test}（stratified）
  </div>
  <div style="flex:1;background:#eff6ff;border-left:4px solid #2563eb;{_box}">
    <strong>模型</strong><br>
    <code>LogisticRegression(C={C:.3g}, solver='lbfgs', penalty='l2')</code><br>
    截距 b = <code>{intercept:.3f}</code> · τ = <code>{tau.value:.2f}</code>
  </div>
  <div style="flex:1;background:#f0fdf4;border-left:4px solid {auc_color};{_box}">
    <strong>测试集评估</strong>（AUC = <span style="color:{auc_color};">{auc:.3f}</span>）<br>
    TP={cm_tp} · FN=<strong style="color:#dc2626;">{cm_fn}</strong>
    （漏诊 ⚠️）<br>
    FP={cm_fp} · TN={cm_tn}
  </div>
</div>
""")
    return


@app.cell
def _(mo):
    mo.accordion({
        "完整 pipeline 代码（可拷贝）": mo.md(r"""
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

data = load_breast_cancer()
X, y = data.data, 1 - data.target   # 翻转 → 1=恶性

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr)
X_te_s = scaler.transform(X_te)

clf = LogisticRegression(C=1.0, max_iter=2000).fit(X_tr_s, y_tr)
proba = clf.predict_proba(X_te_s)[:, 1]
y_pred = (proba >= 0.5).astype(int)

print(classification_report(y_te, y_pred, target_names=["良性", "恶性"]))
print(f"AUC = {roc_auc_score(y_te, proba):.3f}")
```

**生产管线**用 `Pipeline([("sc", StandardScaler()), ("lr", LR(...))]`)
把标准化和模型绑死，避免训测漏标准化或泄漏。
"""),
        "为什么必须先标准化（StandardScaler）": mo.md(r"""
LR 损失对 `||w||²` 加正则惩罚，**所有 w 共用一个 C**。如果不同特征 scale 差很多
（"半径" 是 1-30 范围，"光滑度" 是 0.001-0.4），相同 C 对它们的"惩罚力度"差几个
数量级 → 大 scale 特征系数被压得过狠，小 scale 特征反而过拟合。

工程后果：
- 不标准化时 lbfgs 容易 `ConvergenceWarning`
- coef_ 大小没法直接对比"哪个特征更重要"

`StandardScaler` 把每列变成 mean=0, std=1，所有特征同尺度。
**只在训练集上 fit**（避免测试集信息泄漏），测试集 transform 用相同的 mean / std。

```python
# ❌ 错：在 X 上 fit（泄漏）
scaler.fit(X)
# ✅ 对：只在训练集 fit
scaler.fit(X_train)
```
"""),
        "标签翻转：sklearn 的 target 约定 vs 业务正例": mo.md(r"""
sklearn 的 `load_breast_cancer().target`：
- **0 = malignant（恶性）**
- 1 = benign（良性）

但**业务直觉的"正例"是关心的少数 / 高代价类 = 恶性**。所以本 demo 翻转：
```python
y = 1 - data.target   # 1=恶性, 0=良性
```

**为什么重要**：`predict_proba(X)[:, 1]` 是 `classes_[1]` 的概率。如果不翻转，
"P(y=1)" 就是"良性概率"——对漏诊召回率（医疗主指标）的解读全反了。

**生产纪律**：项目开始时**显式定义正例**（写注释 + assert），别靠数据集默认。
```python
assert clf.classes_.tolist() == [0, 1]
assert "恶性" 对应 1   # 写文档里
```
"""),
        "调阈值的医疗 vs 工程权衡": mo.md(r"""
默认 τ=0.5 在医疗筛查里**通常太高**：漏诊（FN）代价 ≫ 误诊（FP）。

把 demo τ 拖到 0.3：
- Recall 通常 ↑（少漏诊）
- Precision 通常 ↓（多误诊）
- AUC 不变（AUC 与阈值无关）

**生产代码模式**：
```python
proba = clf.predict_proba(X)[:, 1]
# 业务规则：召回率不能低于 95%
from sklearn.metrics import precision_recall_curve
P, R, T = precision_recall_curve(y_val, proba)
# 找满足 R >= 0.95 的最大 P 对应阈值
idx = np.where(R >= 0.95)[0][-1]
tau_chosen = T[idx]
```

调阈值不重训，只是改部署侧的判定门槛——LR 的部署灵活性优势。
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
4  ├ test ┬ τ ┬ logC h=4 ──┤
8  ├──── 状态卡 h=5 全宽 ────┤
13 ├ 5 指标(10) ┬ top-10 coef(14) ─┤  ← h=18
31 ├──── accordion h=5 全宽 ────┤
36
```
""")
    return


if __name__ == "__main__":
    app.run()
