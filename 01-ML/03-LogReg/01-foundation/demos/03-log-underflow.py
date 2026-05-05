"""
log 拯救 underflow · 工程刚需演示

互动：拖 N / P → 看直接连乘 vs log 版的反差，N / P 极端时直接版跌入 float64 下溢区
跑：marimo edit 03-log-underflow.py --port 2733 --headless --no-token
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    import sys
    return alt, mo, np, pd, sys


@app.cell
def _(mo):
    mo.md(
        r"""
    # log 拯救 underflow · 工程刚需演示

    > 概率连乘场景（朴素贝叶斯 / 最大似然 / Logistic 损失），样本数 N 一上千，
    > 直接 $\prod P_i$ 容易跌进 **float64 下溢区（约 2.2e-308）** 塌成 0——不是数学优雅，是工程刚需。

    $$\underbrace{L = \prod_{i=1}^{N} P_i = P^N}_{\text{直接：N 大就 underflow}} \quad\xrightarrow{\;\ln\;}\quad \underbrace{\ln L = \sum_{i=1}^{N} \ln P_i = N \cdot \ln P}_{\text{log 版：永远是有限负数}}$$
    """
    )
    return


@app.cell
def _(mo):
    preset = mo.ui.dropdown(
        options={
            "自由": None,
            "正常 (10, 0.5)": (10, 0.5),
            "逼近极限 (1000, 0.5)": (1000, 0.5),
            "真 underflow (10000, 0.1)": (10000, 0.1),
            "存活 (10000, 0.99)": (10000, 0.99),
        },
        value="自由", label="预设场景",
    )
    N = mo.ui.slider(1, 10000, value=1000, step=10, label="N（样本数）")
    P = mo.ui.slider(0.01, 0.99, value=0.5, step=0.01, label="P（每样本概率）")
    mo.hstack([preset, N, P], widths=[1, 1, 1], justify="space-around")
    return N, P, preset


@app.cell
def _(N, P, preset):
    if preset.value is None:
        N_use, P_use, mode_label = int(N.value), float(P.value), "手动"
    else:
        _n, _p = preset.value
        N_use, P_use, mode_label = int(_n), float(_p), "预设"
    return N_use, P_use, mode_label


@app.cell
def _(N_use, P_use, np, sys):
    UF = sys.float_info.min  # 约 2.2e-308
    with np.errstate(under="ignore", over="ignore", invalid="ignore"):
        try:
            direct_result = float(P_use) ** int(N_use)
        except (OverflowError, ValueError):
            direct_result = 0.0
    log_result = float(N_use) * float(np.log(P_use))
    with np.errstate(under="ignore"):
        log_then_exp = float(np.exp(log_result))
    is_uf_direct = (direct_result < UF) or (direct_result == 0.0)
    is_uf_lte = (log_then_exp < UF) or (log_then_exp == 0.0)
    return direct_result, is_uf_direct, is_uf_lte, log_result, log_then_exp


@app.cell
def _(N_use, P_use, direct_result, is_uf_direct, is_uf_lte, log_result, log_then_exp, mo, mode_label):
    _UF = '<span style="background:#dc2626;color:white;padding:2px 6px;border-radius:3px;font-size:11px;font-weight:600;">UNDERFLOW</span>'
    _OK = '<span style="background:#16a34a;color:white;padding:2px 6px;border-radius:3px;font-size:11px;font-weight:600;">OK</span>'
    _ds = f"{direct_result:.3e}" if direct_result > 0 else "0.0"
    _df = (f"{direct_result:.20f}"[:24] + "...") if direct_result > 0 else "0.000..."
    _ls = f"{log_then_exp:.3e}" if log_then_exp > 0 else "0.0"
    _bd, _bl = (_UF if is_uf_direct else _OK), (_UF if is_uf_lte else _OK)
    mo.md(
        f"""
    **当前**：N=<code>{N_use}</code> · P=<code>{P_use}</code> · 模式 <code>{mode_label}</code>

    | 方法 | 科学计数 | 浮点 | 状态 | 备注 |
    |---|---|---|---|---|
    | 直接 `P**N` | `{_ds}` | `{_df}` | {_bd} | N 大必塌；float64 下限 ≈ 2.2e-308 |
    | `exp(N·lnP)` | `{_ls}` | — | {_bl} | 中间稳定，exp 回原空间仍 underflow |
    | log `N·lnP` | `{log_result:.3e}` | `{log_result:.4f}` | {_OK} | **工程标准**：永远有限负数 |
    """
    )
    return


@app.cell
def _(alt, direct_result, is_uf_direct, is_uf_lte, log_result, log_then_exp, np, pd):
    _floor = -320.0
    _sl = lambda v: _floor if (v is None or v <= 0) else float(np.log10(v))
    df_bar = pd.DataFrame({
        "method": ["直接连乘 P^N", "log 后取指数 exp(N·lnP)", "直接 log N·lnP"],
        "log10_abs": [_sl(direct_result), _sl(log_then_exp),
                      _sl(abs(log_result)) if log_result != 0 else _floor],
        "status": ["UNDERFLOW" if is_uf_direct else "OK",
                   "UNDERFLOW" if is_uf_lte else "OK", "OK"],
        "raw": [f"{direct_result:.3e}", f"{log_then_exp:.3e}", f"{log_result:.3f}"],
    })
    _ymax = max(5.0, float(df_bar["log10_abs"].max()) + 5.0)
    bar_chart = alt.Chart(df_bar).mark_bar(stroke="black", strokeWidth=0.5).encode(
        x=alt.X("method:N", title=None, sort=None, axis=alt.Axis(labelAngle=0, labelFontSize=11)),
        y=alt.Y("log10_abs:Q", title="log10 |结果|（越低越接近 underflow）",
                scale=alt.Scale(domain=[_floor, _ymax])),
        color=alt.Color("status:N",
                        scale=alt.Scale(domain=["OK", "UNDERFLOW"], range=["#16a34a", "#dc2626"]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["method", "raw", "log10_abs", "status"],
    ).properties(width=560, height=240,
                 title="三种算法量级对比（y 是 log10，-300 已塌到 float 底）")
    bar_chart
    return


@app.cell
def _(N_use, P_use, alt, np, pd):
    n_grid = np.unique(np.linspace(1, 10000, 100).astype(int))
    log_vals = n_grid.astype(float) * float(np.log(P_use))
    with np.errstate(under="ignore"):
        direct_vals = np.exp(log_vals)
    df_scan = pd.DataFrame({"N": n_grid, "direct": direct_vals.astype(float),
                            "log_val": log_vals.astype(float)})
    rule = alt.Chart(pd.DataFrame({"N": [N_use]})).mark_rule(
        color="#1f77b4", strokeWidth=1.5, strokeDash=[4, 3]).encode(x="N:Q")
    top = alt.Chart(df_scan).mark_line(color="#dc2626", strokeWidth=2).encode(
        x=alt.X("N:Q", title="N（样本数）"),
        y=alt.Y("direct:Q", title="直接 P^N（线性 · 看到塌零悬崖）"),
        tooltip=["N", "direct"],
    ).properties(width=560, height=180,
                 title=f"顶：直接连乘 P^N（P={P_use}）— y 轴塌到 0")
    bot = alt.Chart(df_scan).mark_line(color="#16a34a", strokeWidth=2).encode(
        x=alt.X("N:Q", title="N（样本数）"),
        y=alt.Y("log_val:Q", title="N·lnP（健康负数）"),
        tooltip=["N", "log_val"],
    ).properties(width=560, height=180, title="底：log 版 N·lnP — 永远干净斜线")
    alt.vconcat(top + rule, bot + rule).resolve_scale(x="shared")
    return


@app.cell
def _(N_use, P_use, direct_result, is_uf_direct, log_result, mo):
    _uf = '<span style="background:#dc2626;color:white;padding:1px 5px;border-radius:3px;font-size:11px;">UNDERFLOW</span>'
    _v = (f'直接 = <code style="color:#dc2626;">{direct_result:.3e}</code> {_uf}'
          if is_uf_direct else f'直接 = <code>{direct_result:.3e}</code>（还活着但已经很小）')
    mo.md(
        f'<div style="background:#f8fafc;border-left:4px solid #1f77b4;padding:10px 14px;'
        f'border-radius:4px;font-size:14px;line-height:1.7;">'
        f'<strong>解读</strong> · N={N_use}, P={P_use}：<br>{_v}；'
        f'log 版 = <code style="color:#16a34a;">{log_result:.2f}</code> '
        f'<span style="color:#16a34a;font-weight:600;">健康</span><br>'
        f'<span style="color:#64748b;font-size:12px;">工程教训：永远在 log 空间累加，'
        f'要原概率时再 <code>exp</code>——但通常你<strong>根本不需要</strong>原概率，'
        f'分类只看相对大小（argmax），log 后保序。</span></div>'
    )
    return


@app.cell
def _(mo):
    mo.accordion({
        "为什么 float64 下限是 2.2e-308": mo.md(
            "IEEE 754 双精度：1 符号 + 11 指数 + 52 尾数。最小正规数 = $2^{-1022} \\approx 2.225 \\times 10^{-308}$。"
            "比这小进入 **subnormal**（精度逐步丢失），再小直接变 0。`sys.float_info.min` = 2.2250738585072014e-308。"
        ),
        "为什么 log 后取指数 也会塌": mo.md(
            "`exp(N·lnP)` 中间步骤 `N·lnP` 健康（如 -2300），但最后 `exp(-2300)` 仍小于 float 下限。"
            "正解：**全程留在 log 空间**。比较概率比 log 概率（保序）；归一化用 **log-sum-exp**。"
        ),
        "log-sum-exp 技巧（朴素贝叶斯归一化）": mo.md(
            "算 $P(c|x) = e^{a_c} / \\sum_k e^{a_k}$，分子分母都可能 over/underflow。"
            "令 $M = \\max_k a_k$，则 $\\log \\sum_k e^{a_k} = M + \\log \\sum_k e^{a_k - M}$。"
            "右边 $a_k - M \\leq 0$，`exp` 永远 $\\leq 1$，绝不 overflow；最大那项 = 1，绝不全塌零。"
            "scipy.special.logsumexp / torch.logsumexp 都是这招。"
        ),
    }, multiple=False)
    return


if __name__ == "__main__":
    app.run()
