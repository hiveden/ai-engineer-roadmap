"""
概率交互 demo · 蒙特卡洛验证独立 vs 不独立

互动：拖 N + 切换放回/不放回 → 看频率曲线收敛到两条不同的理论线
跑：marimo edit 04-probability-cards.py --port 2734 --headless --no-token
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
    return alt, mo, np, pd


@app.cell
def _(mo):
    mo.md(
        r"""
    # 概率 demo · 抽两张牌全 A 的频率收敛

    一副 52 张牌（4 张 A）。**抽两张全是 A** 有两种算法，差在「第二次抽样是否独立」：

    $$
    P(\text{两张全 A}) = \begin{cases}
    \dfrac{4}{52} \times \dfrac{4}{52} \approx 0.5917\% & \text{放回（独立）} \\[6pt]
    \dfrac{4}{52} \times \dfrac{3}{51} \approx 0.4525\% & \text{不放回（条件依赖）}
    \end{cases}
    $$

    拖 N、切换模式 → 蓝色累积频率曲线收敛到绿色理论线。
    放回比不放回**高 ~31%**：不放回时第二次"还能抽到 A"的可能性变小（少一张 A，少一张牌）。
    """
    )
    return


@app.cell
def _(mo):
    N = mo.ui.slider(10, 100000, value=10000, step=10, label="试验次数 N")
    replace = mo.ui.radio(
        {"放回（独立）": True, "不放回（条件依赖）": False},
        value="放回（独立）", label="抽样方式",
    )
    seed = mo.ui.number(value=42, label="随机种子（改变可重新跑）")
    mo.hstack([N, replace, seed], widths=[2, 1, 1], justify="space-around")
    return N, replace, seed


@app.cell
def _(N, np, replace, seed):
    # 向量化蒙特卡洛：52 张牌编号 0..51，A 是 0..3；N=100k 也秒回
    rng = np.random.default_rng(int(seed.value))
    n_trials = int(N.value)
    if replace.value:
        c1 = rng.integers(0, 52, size=n_trials)
        c2 = rng.integers(0, 52, size=n_trials)
    else:
        # 不放回：每行 argsort 一组随机数取前 2 = 无放回抽 2
        perms = np.argsort(rng.random((n_trials, 52)), axis=1)
        c1, c2 = perms[:, 0], perms[:, 1]
    hits = (c1 < 4) & (c2 < 4)
    cumfreq = np.cumsum(hits) / np.arange(1, n_trials + 1)

    p_indep = (4 / 52) * (4 / 52)  # ≈ 0.005917
    p_dep = (4 / 52) * (3 / 51)    # ≈ 0.004525
    p_theory = p_indep if replace.value else p_dep
    n_hits = int(hits.sum())
    freq_final = float(cumfreq[-1])
    abs_err = abs(freq_final - p_theory)
    rel_err = abs_err / p_theory
    return abs_err, cumfreq, freq_final, n_hits, p_dep, p_indep, p_theory, rel_err


@app.cell
def _(N, cumfreq, np, p_theory, pd, replace):
    # 频率曲线下采样：N>1000 时取 200 个 log-spaced 点，避免画 100k 点卡
    n_total = int(N.value)
    if n_total > 1000:
        idx = np.unique(np.round(np.logspace(0, np.log10(n_total), 200)).astype(int))
        idx = np.clip(idx, 1, n_total) - 1
    else:
        idx = np.arange(n_total)
    n_sampled_pts = len(idx)

    df_curve = pd.DataFrame({
        "trial": (idx + 1).astype(int),
        "cumfreq": cumfreq[idx],
        "kind": ["累积频率"] * len(idx),
    })
    _theory_label = f"理论值 ({'放回' if replace.value else '不放回'})"
    df_theory = pd.DataFrame({
        "trial": [1, n_total],
        "cumfreq": [p_theory, p_theory],
        "kind": [_theory_label, _theory_label],
    })
    return df_curve, df_theory, n_sampled_pts


@app.cell
def _(N, alt, df_curve, df_theory, p_theory, replace):
    palette = alt.Scale(
        domain=["累积频率", f"理论值 ({'放回' if replace.value else '不放回'})"],
        range=["#1f77b4", "#16a34a"],
    )
    y_max = max(p_theory * 3, 0.012)

    line_freq = alt.Chart(df_curve).mark_line(strokeWidth=2).encode(
        x=alt.X("trial:Q", title="试验序号（log）",
                scale=alt.Scale(type="log", domain=[1, int(N.value)])),
        y=alt.Y("cumfreq:Q", title="累积频率", scale=alt.Scale(domain=[0, y_max])),
        color=alt.Color("kind:N", scale=palette, legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("trial:Q", title="试验"),
                 alt.Tooltip("cumfreq:Q", title="频率", format=".5f")],
    )
    line_theory = alt.Chart(df_theory).mark_line(strokeWidth=2, strokeDash=[6, 4]).encode(
        x=alt.X("trial:Q", scale=alt.Scale(type="log")),
        y="cumfreq:Q",
        color=alt.Color("kind:N", scale=palette, legend=None),
    )
    chart_curve = (line_freq + line_theory).properties(
        width=600, height=320,
        title=f"累积频率 → 理论值（{'放回（独立）' if replace.value else '不放回（条件依赖）'}）",
    )
    chart_curve
    return


@app.cell
def _(alt, p_dep, p_indep, pd, replace):
    df_bar = pd.DataFrame({
        "mode": ["放回 (独立)", "不放回 (条件)"],
        "prob": [p_indep, p_dep],
        "selected": ["当前" if replace.value else "对照",
                     "对照" if replace.value else "当前"],
    })
    bar = alt.Chart(df_bar).mark_bar().encode(
        x=alt.X("mode:N", title=None, sort=["放回 (独立)", "不放回 (条件)"]),
        y=alt.Y("prob:Q", title="理论概率", axis=alt.Axis(format=".2%")),
        color=alt.Color("selected:N",
            scale=alt.Scale(domain=["当前", "对照"], range=["#1f77b4", "#cbd5e1"]),
            legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("mode:N", title="模式"),
                 alt.Tooltip("prob:Q", title="概率", format=".5f")],
    )
    text = alt.Chart(df_bar).mark_text(dy=-8, fontSize=12, fontWeight="bold").encode(
        x="mode:N", y="prob:Q", text=alt.Text("prob:Q", format=".4%"),
        color=alt.value("#0f172a"),
    )
    diff_pct = (p_indep - p_dep) / p_dep * 100
    chart_bar = (bar + text).properties(
        width=380, height=260,
        title=f"理论值对比 · 放回比不放回高 {diff_pct:.1f}%",
    )
    chart_bar
    return


@app.cell
def _(N, abs_err, freq_final, mo, n_hits, n_sampled_pts, p_theory, rel_err, replace):
    mode_str = "放回（独立）" if replace.value else "不放回（条件依赖）"
    _ok = rel_err < 0.1
    _bg = "#eff6ff" if _ok else "#fff7ed"
    _bd = "#1f77b4" if _ok else "#f97316"
    _txt = "#1e3a8a" if _ok else "#9a3412"
    _verdict = "✓ 实测接近理论" if _ok else "⚠ 误差偏大，加大 N 再看"
    mo.md(
        f"""
<div style="padding:10px 14px;background:{_bg};border-left:4px solid {_bd};border-radius:4px;font-size:14px;color:{_txt};line-height:1.8;">
<strong>{_verdict}</strong> · 模式 <code>{mode_str}</code> · N=<code>{int(N.value):,}</code><br>
• 命中数：<strong>{n_hits}</strong> / {int(N.value):,} → 实测频率 <strong>{freq_final:.5f}</strong>（{freq_final * 100:.4f}%）<br>
• 理论值：<strong>{p_theory:.5f}</strong>（{p_theory * 100:.4f}%）· 绝对误差 <code>{abs_err:.5f}</code> · 相对误差 <code>{rel_err:.2%}</code><br>
• 曲线下采样到 <code>{n_sampled_pts}</code> 个点（log-spaced，避免 altair 卡）
</div>

**教学解读**：N 越大 → 实测越接近理论（大数定律）；切换放回/不放回 → 收敛到两条不同理论线。
03 概率讲解里的反例 `4/52 × 4/52` vs `4/52 × 3/51` 是**独立性假设是否成立**的物理体现——
朴素贝叶斯之所以"朴素"就是强行用左边算法，即使现实更接近右边。
        """
    )
    return


@app.cell
def _(mo):
    mo.accordion({
        "为什么不放回概率更小？": mo.md(
            "**第一次抽到 A**：4/52，两种模式相同。\n\n"
            "**第二次抽到 A**：放回还是 4/52；不放回少一张 A 少一张牌 → 3/51。\n\n"
            "**条件概率视角**：$P(B|A) \\ne P(B)$ 时 A、B **不独立**——"
            "知道 A 发生改变了 B 的概率分布。"
        ),
        "和朴素贝叶斯什么关系？": mo.md(
            "朴素贝叶斯算 $P(x_1, x_2, \\dots, x_n | y)$ 时**强行假设特征独立**：\n\n"
            "$$P(x_1, \\dots, x_n | y) \\approx \\prod_i P(x_i | y)$$\n\n"
            "对应这里的 **放回模式**——每个特征当成「独立抽样」。"
            "现实中特征常相关（如「下雨」和「带伞」），更接近 **不放回** 的条件依赖。\n\n"
            "朴素贝叶斯 work 的秘密：**分类任务只需排序正确**——即使联合概率算偏了，最大那个类通常还是对的。"
        ),
        "蒙特卡洛 vs 解析解": mo.md(
            "**解析解**：$\\binom{4}{2} / \\binom{52}{2} = 6/1326 = 0.4525\\%$（不放回精确）。\n\n"
            "**蒙特卡洛**：跑 N 次数频率，N→∞ 收敛到解析解（大数定律）。\n\n"
            "**为什么还要它**：解析解算不动时（多步骤、连续、复杂依赖）；"
            "验证解析推导（这个 demo）；强化学习 / 贝叶斯推断（MCMC、粒子滤波）大量出现。"
        ),
    }, multiple=False)
    return


if __name__ == "__main__":
    app.run()
