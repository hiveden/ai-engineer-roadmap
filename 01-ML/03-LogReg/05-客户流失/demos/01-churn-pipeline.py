"""
电信客户流失 · 端到端 LR pipeline（合成数据近似真实 Telco churn 分布）

互动：
- 拖阈值 τ → 评估指标 + 混淆矩阵 + 留存动作覆盖率同步变
- 切类别权重（balanced / 默认）看不平衡处理
跑：marimo edit 01-churn-pipeline.py --port 2750 --headless --no-token
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    layout_file="layouts/01-churn-pipeline.grid.json",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score,
                                 roc_curve)
    return (LogisticRegression, StandardScaler, accuracy_score, alt,
            f1_score, mo, np, pd, precision_score, recall_score,
            roc_auc_score, roc_curve, train_test_split)


@app.cell
def _(mo):
    mo.md(r"""
    # 电信客户流失 · 端到端 LR pipeline

    7000 客户 × 12 特征（合成近似 Telco Churn 分布）。**正例 = 流失 Churn=Yes**（约 26%）。

    `raw → drop ID → one-hot → split + 标准化 → LR + class_weight → 评估`

    业务诉求：找高流失风险客户做留存（折扣 / 客服回访），**Recall 优先**。
    """)
    return


@app.cell
def _(mo):
    tau = mo.ui.slider(0.1, 0.9, value=0.5, step=0.05,
                       label="阈值 τ", show_value=True)
    class_weight_kind = mo.ui.dropdown(
        options={"None（默认）": "none",
                 "balanced（按频率反比）": "balanced"},
        value="None（默认）", label="class_weight")
    log_C = mo.ui.slider(-2.0, 2.0, value=0.0, step=0.25,
                         label="log₁₀(C)", show_value=True)
    mo.hstack([tau, class_weight_kind, log_C],
              widths=[1, 1.3, 1], justify="space-around")
    return class_weight_kind, log_C, tau


@app.cell
def _(np, pd):
    # 合成数据：模拟真实 Telco churn 分布（基础流失率 ~26%）
    rng = np.random.default_rng(42)
    n = 7000

    # 数值特征
    tenure = rng.integers(0, 73, size=n)  # 在网月数
    monthly_charges = rng.uniform(18, 120, size=n).round(2)
    senior = rng.choice([0, 1], size=n, p=[0.84, 0.16])

    # 类别特征
    contract = rng.choice(["Month-to-month", "One year", "Two year"],
                          size=n, p=[0.55, 0.21, 0.24])
    internet = rng.choice(["Fiber optic", "DSL", "No"],
                          size=n, p=[0.44, 0.34, 0.22])
    payment = rng.choice(["Electronic check", "Mailed check",
                          "Bank transfer", "Credit card"],
                         size=n, p=[0.34, 0.23, 0.22, 0.21])
    paperless = rng.choice(["Yes", "No"], size=n, p=[0.59, 0.41])
    partner = rng.choice(["Yes", "No"], size=n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.30, 0.70])
    online_security = rng.choice(["Yes", "No", "No internet"],
                                 size=n, p=[0.29, 0.49, 0.22])
    tech_support = rng.choice(["Yes", "No", "No internet"],
                              size=n, p=[0.29, 0.49, 0.22])

    # 合成 Churn 概率（参考真实数据相关性）
    z = -1.0
    z = z + (-0.06) * tenure  # 在网越久越不易流失
    z = z + 0.012 * (monthly_charges - 65)  # 高月费略增流失
    z = z + 0.4 * senior
    z = z + np.where(contract == "Month-to-month", 1.5, 0)
    z = z + np.where(contract == "Two year", -1.2, 0)
    z = z + np.where(internet == "Fiber optic", 0.8, 0)
    z = z + np.where(payment == "Electronic check", 0.6, 0)
    z = z + np.where(paperless == "Yes", 0.3, 0)
    z = z + np.where(partner == "Yes", -0.3, 0)
    z = z + np.where(dependents == "Yes", -0.4, 0)
    z = z + np.where(online_security == "Yes", -0.5, 0)
    z = z + np.where(tech_support == "Yes", -0.5, 0)
    z = z + rng.normal(0, 0.6, size=n)  # 噪声

    p = 1 / (1 + np.exp(-z))
    churn = (rng.uniform(0, 1, size=n) < p).astype(int)

    df = pd.DataFrame({
        "customerID": [f"C{i:05d}" for i in range(n)],
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "SeniorCitizen": senior,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment,
        "PaperlessBilling": paperless,
        "Partner": partner,
        "Dependents": dependents,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "Churn": np.where(churn == 1, "Yes", "No"),
    })
    return (df,)


@app.cell
def _(df, pd):
    # 数据基本处理：drop ID，类别 one-hot，Churn → 1/0
    raw_n = len(df)
    pos_count = int((df["Churn"] == "Yes").sum())
    neg_count = raw_n - pos_count
    churn_rate = pos_count / raw_n

    df_clean = df.drop(columns=["customerID"]).copy()
    y = (df_clean.pop("Churn") == "Yes").astype(int).to_numpy()

    cat_cols = ["Contract", "InternetService", "PaymentMethod",
                "PaperlessBilling", "Partner", "Dependents",
                "OnlineSecurity", "TechSupport"]
    df_X = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True,
                          dtype=float)
    feat_names = list(df_X.columns)
    X = df_X.to_numpy()
    return X, churn_rate, feat_names, neg_count, pos_count, raw_n, y


@app.cell
def _(LogisticRegression, StandardScaler, X, class_weight_kind, log_C,
      train_test_split, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)
    cw = None if class_weight_kind.value == "none" else "balanced"
    C = float(10 ** log_C.value)
    clf = LogisticRegression(solver="lbfgs", penalty="l2",
                             C=C, class_weight=cw, max_iter=2000)
    clf.fit(X_tr_s, y_tr)
    proba_te = clf.predict_proba(X_te_s)[:, 1]
    coef = clf.coef_.ravel()
    intercept = float(clf.intercept_[0])
    return C, clf, coef, intercept, proba_te, y_te


@app.cell
def _(accuracy_score, f1_score, precision_score, proba_te, recall_score,
      roc_auc_score, tau, y_te):
    y_pred = (proba_te >= tau.value).astype(int)
    tp = int(((y_te == 1) & (y_pred == 1)).sum())
    fn = int(((y_te == 1) & (y_pred == 0)).sum())
    fp = int(((y_te == 0) & (y_pred == 1)).sum())
    tn = int(((y_te == 0) & (y_pred == 0)).sum())
    metrics = {
        "Accuracy": accuracy_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred, zero_division=0),
        "Recall": recall_score(y_te, y_pred, zero_division=0),
        "F1": f1_score(y_te, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_te, proba_te),
    }
    # 业务覆盖率：被推留存动作的客户占比 = (TP+FP)/N
    n_te = len(y_te)
    n_targeted = tp + fp
    target_rate = n_targeted / n_te
    return fn, fp, metrics, n_targeted, n_te, target_rate, tn, tp


@app.cell
def _(alt, pd, proba_te, roc_curve, tau, y_te):
    # 视图 1：ROC 曲线 + 当前点
    fpr, tpr, thr = roc_curve(y_te, proba_te)
    _df = pd.DataFrame({"FPR": fpr, "TPR": tpr, "thr": thr})
    line = alt.Chart(_df).mark_line(stroke="#2563eb", strokeWidth=2.5).encode(
        x=alt.X("FPR:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("TPR:Q", scale=alt.Scale(domain=[0, 1])),
    )
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        stroke="#94a3b8", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
    # 当前阈值附近的点
    _i = (abs(thr - tau.value)).argmin()
    cur = alt.Chart(pd.DataFrame({"FPR": [fpr[_i]], "TPR": [tpr[_i]]})).mark_circle(
        color="#dc2626", size=200, stroke="white", strokeWidth=2).encode(
        x="FPR:Q", y="TPR:Q")
    chart_roc = (line + diag + cur).properties(width=320, height=320)
    return (chart_roc,)


@app.cell
def _(alt, coef, feat_names, np, pd):
    # 视图 2：top-10 |w| 推动流失 / 留存
    abs_w = np.abs(coef)
    idx = np.argsort(abs_w)[::-1][:10]
    _df = pd.DataFrame({
        "特征": [feat_names[i] for i in idx],
        "w": [coef[i] for i in idx],
        "推向": ["流失风险 ↑" if coef[i] > 0 else "留存倾向 ↑" for i in idx],
    })
    bars = alt.Chart(_df).mark_bar().encode(
        x=alt.X("w:Q", title="权重 w（标准化特征 + one-hot）"),
        y=alt.Y("特征:N", sort=_df["特征"].tolist(), title=None),
        color=alt.Color("推向:N",
            scale=alt.Scale(domain=["流失风险 ↑", "留存倾向 ↑"],
                            range=["#dc2626", "#16a34a"]),
            legend=alt.Legend(orient="top")),
        tooltip=["特征:N", alt.Tooltip("w:Q", format=".3f"), "推向:N"],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        stroke="#0f172a").encode(x="x:Q")
    chart_coef = (bars + zero).properties(width=480, height=320)
    return (chart_coef,)


@app.cell
def _(chart_coef, chart_roc, mo):
    mo.hstack([chart_roc, chart_coef],
              widths=[1, 1.4], justify="space-around")
    return


@app.cell
def _(C, churn_rate, class_weight_kind, fn, fp, intercept, metrics, mo,
      n_targeted, n_te, neg_count, pos_count, raw_n, target_rate, tau,
      tn, tp):
    # 状态卡：数据集 + 评估 + 业务
    auc = metrics["AUC"]
    rec = metrics["Recall"]
    prec = metrics["Precision"]
    auc_color = "#16a34a" if auc >= 0.8 else (
        "#eab308" if auc >= 0.7 else "#dc2626")
    rec_color = "#16a34a" if rec >= 0.7 else (
        "#eab308" if rec >= 0.5 else "#dc2626")
    cw_label = "无" if class_weight_kind.value == "none" else "balanced"
    _box = "padding:10px 14px;border-radius:6px;font-size:13px;line-height:1.7;"
    mo.md(f"""
<div style="display:flex;gap:10px;margin:6px 0;">
  <div style="flex:1.1;background:#f8fafc;border:1px solid #e2e8f0;{_box}">
    <strong>数据集</strong>（合成 Telco Churn）<br>
    {raw_n} 客户 = 流失 <strong style="color:#dc2626;">{pos_count}</strong>
    （{churn_rate:.1%}）+ 留存 {neg_count}<br>
    测试集 {n_te} · class_weight = <code>{cw_label}</code> · C = {C:.3g}
  </div>
  <div style="flex:1;background:#eff6ff;border-left:4px solid {auc_color};{_box}">
    <strong>测试集评估</strong>（τ = {tau.value:.2f}）<br>
    AUC = <strong style="color:{auc_color};">{auc:.3f}</strong>
    · F1 = {metrics["F1"]:.3f}<br>
    Precision = {prec:.1%} · <strong style="color:{rec_color};">Recall = {rec:.1%}</strong>
  </div>
  <div style="flex:1.1;background:#fff7ed;border-left:4px solid #ea580c;{_box}">
    <strong>留存动作覆盖</strong>（TP+FP）/N<br>
    被推送动作 = <strong>{n_targeted}</strong> / {n_te}（<strong>{target_rate:.1%}</strong>）<br>
    其中真流失 TP={tp} · 误打扰 FP={fp} · 漏掉 FN=<strong style="color:#dc2626;">{fn}</strong>
  </div>
</div>
""")
    return


@app.cell
def _(df, mo):
    # 数据预览（前 6 行）
    mo.md("### 原始数据预览（合成）")
    return


@app.cell
def _(df, mo):
    mo.ui.table(df.head(6), show_column_summaries=False, page_size=6,
                selection=None)
    return


@app.cell
def _(mo):
    mo.accordion({
        "完整 pipeline 代码": mo.md(r"""
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 1. 数据基本处理
df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
y = (df.pop("Churn") == "Yes").astype(int)

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
X = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)

# 2. 切分 + 标准化（只在 train 上 fit）
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr)
X_te_s = scaler.transform(X_te)

# 3. LR + 不平衡处理
clf = LogisticRegression(C=1.0, class_weight="balanced",
                         max_iter=2000).fit(X_tr_s, y_tr)

# 4. 评估
proba = clf.predict_proba(X_te_s)[:, 1]
y_pred = (proba >= 0.4).astype(int)   # 业务调阈值偏 R
print(classification_report(y_te, y_pred,
      target_names=["留存", "流失"]))
print(f"AUC = {roc_auc_score(y_te, proba):.3f}")
```
"""),
        "为什么流失场景偏 Recall（漏报代价大）": mo.md(r"""
业务循环：
1. 模型挑高流失风险用户
2. 营销团队发券 / 客服回访
3. 客户被挽回 = 留存收益

**漏报 FN 代价**：真流失客户没被识别 → 客户实际流失 → 终身价值（LTV）损失，
通常一个流失客户损失 = 几个月营收。

**误报 FP 代价**：留存客户被误识别 → 发了优惠券但本来也不会流失 → 让利成本，
通常 = 一张券 ≪ 终身价值。

所以业务调阈值通常 τ < 0.5 偏 Recall，可以接受 P 低一点。
**调阈值不重训**——`predict_proba` 一次跑完，τ 只在部署侧改。

**进阶**：用 cost-weighted classification + class_weight="balanced" 双重处理。
"""),
        "class_weight='balanced' 怎么工作": mo.md(r"""
默认 LR 损失里每个样本权重 1。极不平衡时（流失率 26% 还算缓和；
信用卡欺诈通常 < 1%），多数类样本数量碾压，模型学到"全猜留存"也准确率高。

`class_weight="balanced"` 把每类权重设为 `n_total / (n_classes * n_class_i)`：
- 类 0（多数）权重 = 1 / 0.74 ≈ 1.35
- 类 1（少数）权重 = 1 / 0.26 ≈ 3.85

少数类样本误分类代价 ≈ 3 倍。等价于"复制少数类样本到平衡"。

**何时用**：
- 类别比例 > 5:1 → 强烈推荐
- 类别比例 ≈ 1:1 → 不必（balanced 几乎不变）
- 已用阈值调节做不平衡 → 二选一即可，避免双重补偿

**对比**：把 demo 切到 `balanced` → Recall 通常 ↑、Precision ↓、AUC 几乎不变。
"""),
        "特征重要性解读（从系数到业务）": mo.md(r"""
right 图按 |w| 排前 10。**因为已标准化，coef 大小可直接对比**：

预期看到的"流失风险 ↑"特征：
- `Contract_Month-to-month`（最强，月付无锁定）
- `InternetService_Fiber optic`（光纤价格高）
- `PaymentMethod_Electronic check`（年龄段年轻 + 易切换）
- `PaperlessBilling_Yes`（数字化用户更易流失）
- `SeniorCitizen`

预期"留存倾向 ↑"特征：
- `tenure`（在网越久越不流失）
- `Contract_Two year`（合约锁定）
- `Dependents_Yes` / `Partner_Yes`（家庭用户粘性高）
- `OnlineSecurity_Yes` / `TechSupport_Yes`（用了附加服务粘性高）

**业务洞察**：
- 留存动作可针对"月付 + 电子支票 + 高月费 + 短期"用户做精准营销
- 推附加服务（OnlineSecurity / TechSupport）本身就是反流失杠杆
- 长合约客户不需要打扰——别误打扰 (FP) 反而引发反弹
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
4  ├ τ ┬ class_weight ┬ logC h=4 ─┤
8  ├──── 状态卡 h=6 全宽 ────┤
14 ├ ROC(10) ┬ top-10 coef(14) ─┤  ← h=14
28 ├──── 数据预览 h=4 ────┤
32 ├──── accordion h=4 全宽 ────┤
36
```
""")
    return


if __name__ == "__main__":
    app.run()
