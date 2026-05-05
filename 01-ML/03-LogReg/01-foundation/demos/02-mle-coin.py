"""
MLE 抛硬币交互 demo · 在似然曲线上找峰值

互动：改观测序列 + 拖 θ → 实时看 L(θ) / ln L(θ) 双曲线 + 反差面板
跑：marimo edit 02-mle-coin.py --port 2732 --headless --no-token
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
        # MLE 抛硬币 · 在似然曲线上找峰值

        > 把「让观测最不像意外」具象化：拖 θ 看 L(θ) 反差，验证 MLE 给出的 θ\* = k/n。

        $$L(\theta) = \prod_{i=1}^{n} P(x_i \mid \theta) = \theta^{k}(1-\theta)^{n-k}, \quad
        \theta^{*} = \arg\max_{\theta} L(\theta) = \frac{k}{n}$$

        - **L(θ)** 把整段观测压成一个数：θ 越接近经验频率 k/n，这个数越大
        - **ln L(θ)** 数值更平稳，但峰值位置完全相同（log 单调）
        """
    )
    return


@app.cell
def _(mo):
    obs_text = mo.ui.text(value="正反反正正正", label="观测序列（正/反任意组合）", full_width=True)
    preset = mo.ui.dropdown(
        options={
            "自由": "FREE",
            "θ=0.1（偏反）": "P01",
            "θ=0.5（公平）": "P05",
            "θ*=k/n（最优）": "OPTIMAL",
            "θ=0.9（偏正）": "P09",
        },
        value="自由",
        label="预设 θ",
    )
    theta = mo.ui.slider(0.01, 0.99, value=0.5, step=0.01, label="θ（猜的正面概率）")
    mo.hstack([obs_text, preset, theta], widths=[1.4, 1, 1.2], justify="space-around")
    return obs_text, preset, theta


@app.cell
def _(mo, obs_text, preset, theta):
    # 单一来源：解析观测 + 算 n / k / θ_use
    raw = obs_text.value or ""
    n = sum(1 for c in raw if c in ("正", "反"))
    k = raw.count("正")
    bad_obs = (n == 0) or (k == 0) or (k == n)
    theta_star = (k / n) if n > 0 else 0.5

    _pmap = {"FREE": None, "P01": 0.1, "P05": 0.5, "OPTIMAL": theta_star, "P09": 0.9}
    _p = _pmap.get(preset.value, None)
    theta_use = float(min(max(theta.value if _p is None else _p, 0.01), 0.99))
    mode_label = "手动滑块" if preset.value == "FREE" else f"预设：{preset.value}"

    if bad_obs:
        warn = mo.md(
            '<div style="padding:8px 14px;background:#fff7ed;border-left:4px solid #f97316;'
            'border-radius:4px;font-size:13px;color:#9a3412;">'
            f"⚠️ 观测异常（n={n}, k={k}）：需既有「正」也有「反」才能演示峰值。"
            "试试默认值「正反反正正正」。</div>"
        )
    else:
        warn = mo.md("")
    warn
    return bad_obs, k, mode_label, n, theta_star, theta_use


@app.cell
def _(k, n, np, pd, theta_star, theta_use):
    # θ 网格 + 曲线/标记数据
    grid = np.linspace(0.01, 0.99, 100)
    L_vals = grid**k * (1.0 - grid) ** (n - k) if n > 0 else np.zeros_like(grid)
    if n > 0 and 0 < k < n:
        lnL_vals = k * np.log(grid) + (n - k) * np.log(1.0 - grid)
    else:
        lnL_vals = np.full_like(grid, np.nan)
    df_curve = pd.DataFrame({"theta": grid, "L": L_vals, "lnL": lnL_vals})

    def _eval(t):
        L_t = float(t**k * (1.0 - t) ** (n - k)) if n > 0 else 0.0
        ln_t = (
            float(k * np.log(t) + (n - k) * np.log(1.0 - t))
            if (n > 0 and 0 < k < n)
            else float("nan")
        )
        return L_t, ln_t

    L_use, lnL_use = _eval(theta_use)
    L_star, lnL_star = _eval(theta_star)
    df_use = pd.DataFrame({"theta": [theta_use], "L": [L_use], "lnL": [lnL_use]})
    df_star = pd.DataFrame({"theta": [theta_star], "L": [L_star], "lnL": [lnL_star]})
    return L_star, df_curve, df_star, df_use, lnL_star


@app.cell
def _(alt, df_curve, df_star, df_use, k, mo, n):
    # 视图 1+2：L(θ) 与 ln L(θ) 双曲线
    def _chart(y, y_title, color, title, dn=False):
        cu, us, st = (df_curve, df_use, df_star)
        if dn:
            cu, us, st = cu.dropna(subset=[y]), us.dropna(subset=[y]), st.dropna(subset=[y])
        tip = [alt.Tooltip("theta:Q", format=".4f"), alt.Tooltip(f"{y}:Q", format=".4e")]
        line = alt.Chart(cu).mark_line(color=color, strokeWidth=2).encode(
            x=alt.X("theta:Q", scale=alt.Scale(domain=[0, 1]), title="θ"),
            y=alt.Y(f"{y}:Q", title=y_title),
        )
        pu = alt.Chart(us).mark_circle(size=200, color="red").encode(
            x="theta:Q", y=f"{y}:Q", tooltip=tip)
        ps = alt.Chart(st).mark_point(
            shape="diamond", size=300, color="green", filled=True,
            stroke="black", strokeWidth=1,
        ).encode(x="theta:Q", y=f"{y}:Q", tooltip=tip)
        return (line + pu + ps).properties(width=420, height=260, title=title)

    chart_L = _chart("L", "L(θ)", "#1f77b4",
                     f"L(θ) = θ^k · (1-θ)^(n-k)，n={n}, k={k}（红=当前，绿钻=θ*）")
    chart_lnL = (
        _chart("lnL", "ln L(θ)", "#9467bd", "ln L(θ)（量纲温和、峰值同位）", dn=True)
        if (n > 0 and 0 < k < n) else mo.md("_（边界数据，ln L 跳过）_")
    )
    mo.vstack([chart_L, chart_lnL])
    return


@app.cell
def _(L_star, bad_obs, k, lnL_star, mo, n, np, theta_star, theta_use):
    # 视图 3：数值反差面板
    if bad_obs:
        panel = mo.md("_（观测异常，反差表跳过）_")
    else:
        def _row(label, t, hl=False):
            L_t = float(t**k * (1.0 - t) ** (n - k))
            ln_t = float(k * np.log(t) + (n - k) * np.log(1.0 - t))
            ratio = f"{(L_t / L_star):.4f}" if L_star > 0 else "—"
            cells = (label, f"{L_t:.6g}", f"{ln_t:.4f}", ratio)
            return "| " + " | ".join(f"**{c}**" if hl else c for c in cells) + " |"

        rows = [
            _row("θ=0.1", 0.1),
            _row("θ=0.5", 0.5),
            _row(f"θ*=k/n={theta_star:.4f}", theta_star),
            _row("θ=0.9", 0.9),
            _row(f"当前 θ_use={theta_use:.4f}", theta_use, hl=True),
        ]
        table_md = (
            "| θ 取值 | L(θ) | ln L(θ) | L / L_max |\n|---|---|---|---|\n"
            + "\n".join(rows)
        )
        panel = mo.md(
            f"**反差面板**（最优 L_max = {L_star:.6g}，ln L_max = {lnL_star:.4f}）\n\n"
            + table_md
            + "\n\n_当前 θ\\_use 行加粗。L/L_max 越接近 1 越像「最不意外」。_"
        )
    panel
    return


@app.cell
def _(bad_obs, k, mo, mode_label, n, theta_star, theta_use):
    # 底部解读
    if bad_obs:
        msg, bg, bd, txt = "⚠️ 观测异常，无法给出 MLE 解读", "#fff7ed", "#f97316", "#9a3412"
    else:
        msg = (f"观测 n=<code>{n}</code> 次中 k=<code>{k}</code> 正、"
               f"(n-k)=<code>{n - k}</code> 反 → MLE 给出 θ* = k/n = "
               f"<strong>{theta_star:.4f}</strong> · 当前 θ_use = <code>{theta_use:.4f}</code>"
               f" · 输入模式 <code>{mode_label}</code>")
        bg, bd, txt = "#e8f5e9", "#16a34a", "#166534"
    mo.md(f'<div style="padding:8px 14px;background:{bg};border-left:4px solid {bd};'
          f'border-radius:4px;font-size:14px;color:{txt};margin:6px 0;">{msg}</div>')
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "为什么 MLE = k/n？": mo.md(
                r"对 ln L(θ) = k ln θ + (n-k) ln(1-θ) 求导："
                r"$\frac{d\ln L}{d\theta} = \frac{k}{\theta} - \frac{n-k}{1-\theta} = 0$ → θ\*=k/n。"
            ),
            "为什么用 ln L？": mo.md(
                "**数值稳定**（n=100 时 L≈1e-30 下溢，ln L≈-70 正常）+ "
                "**乘积变求和** $\\ln \\prod_i P_i = \\sum_i \\ln P_i$ + "
                "**峰值同位**（log 单调）。训练 LR/NN 都最小化 -ln L（cross-entropy）。"
            ),
            "推到逻辑回归": mo.md(
                "抛硬币 P(x|θ)=θ^x(1-θ)^(1-x) 即 Bernoulli。LR 把固定 θ 换成 σ(w·x+b)："
                r"$L(w,b)=\prod_i \sigma(z_i)^{y_i}(1-\sigma(z_i))^{1-y_i}$，"
                "取 -ln 即二分类交叉熵。MLE 思路一脉相承。"
            ),
        },
        multiple=False,
    )
    return


if __name__ == "__main__":
    app.run()
