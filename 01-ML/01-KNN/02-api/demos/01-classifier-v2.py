"""
KNN 分类 API · sklearn 三件套（marimo 录屏 demo · v2 单屏幻灯片）

设计：
  - viewport 1920×1080，每 state = 一张全屏幻灯片
  - 深色统一主题：bg #0b1220 / card #131c2e / accent yellow/green/red
  - dropdown CSS 隐藏（仅作 Playwright 切 state 接口）
  - 9 step：1·数据X / 2·X+y / 3·import / 4·构造 / 5·fit /
            6·predict / 7·输出 / 8·计时 / 9·fit 内部
  - 散点图手画 SVG（深色原生），不依赖 altair
  - 不用 marimo hstack，整块 mo.Html + CSS Grid 铺满 1920

跑：
  cd 01-ML/01-KNN/02-api/demos
  marimo run 01-classifier-v2.py --headless --port 2762 --no-token
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    return mo, np, pd


@app.cell
def _(mo):
    # 全局 reset：去 marimo 默认 max-width / padding，铺满 1920×1080；
    # 隐藏 nav / dropdown UI（dropdown 仅做 Playwright 接口）
    mo.Html(
        """
<style>
  html, body {
    margin: 0 !important;
    padding: 0 !important;
    background: #0b1220 !important;
    color: #e2e8f0 !important;
    font-family: -apple-system, "PingFang SC", "Inter", sans-serif !important;
  }
  #App, main, .marimo-cell, .Cell,
  [data-testid="cell-output"],
  [data-testid="cell-output"] > div,
  [data-testid="cell-output"] .markdown {
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    background: #0b1220 !important;
  }
  .marimo-cell, .Cell { border: none !important; }
  /* 隐藏含 dropdown 的整个 cell（含 label / wrapper） */
  .marimo-cell:has([data-testid="marimo-plugin-dropdown"]),
  [data-testid="cell-output"]:has([data-testid="marimo-plugin-dropdown"]) {
    position: fixed !important;
    left: -9999px !important;
    top: -9999px !important;
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
    opacity: 0 !important;
  }
  /* select 本身保持 enabled（Playwright select_option 用） */
  [data-testid="marimo-plugin-dropdown"] {
    pointer-events: auto !important;
    visibility: visible !important;
  }
  /* 干掉 cell 右上角操作菜单（"..." 按钮） */
  button[data-testid="cell-actions-button"],
  [data-testid="cell-action-menu"],
  [data-testid="cell-actions-dropdown"],
  .cell-actions, .cell-toolbar { display: none !important; }
  /* 隐藏顶部导航 / sidebar */
  nav, header, aside, .marimo-app-bar, [data-testid="app-bar"] {
    display: none !important;
  }
</style>
"""
    )
    return


@app.cell
def _(pd):
    movies = [
        ("流浪地球2", 8.4, 9.0, "喜欢"),
        ("阿凡达", 8.7, 8.5, "喜欢"),
        ("泰坦尼克", 9.5, 9.5, "喜欢"),
        ("复联4", 8.5, 9.2, "喜欢"),
        ("沙丘2", 8.0, 8.0, "喜欢"),
        ("哥斯拉", 6.4, 7.0, "不喜欢"),
        ("长津湖", 7.4, 8.0, "不喜欢"),
        ("寒战", 7.5, 6.5, "不喜欢"),
        ("上海堡垒", 2.9, 3.5, "不喜欢"),
    ]
    df = pd.DataFrame(movies, columns=["电影", "评分", "主演吸引度", "标签"])
    new_movie = (8.5, 9.5)  # 流浪地球 3
    return df, new_movie


@app.cell
def _(mo):
    step = mo.ui.dropdown(
        options=[
            "1·数据 X",
            "2·标签 y",
            "3·import",
            "4·构造",
            "5·fit",
            "6·predict",
            "7·输出",
            "8·计时",
            "9·fit 内部",
        ],
        value="1·数据 X",
        label="step",
    )
    step
    return (step,)


@app.cell
def _(df, mo, new_movie, np, step):
    # ─── 共用：舞台壳 ───────────────────────────────────────
    def stage(title: str, body_html: str, caption: str = "") -> str:
        return f"""
<div style="
  width:1920px; height:1080px;
  background:#0b1220; color:#e2e8f0;
  display:flex; flex-direction:column;
  box-sizing:border-box; padding:60px 96px 56px 96px;
  font-family:-apple-system,'PingFang SC','Inter',sans-serif;
">
  <div style="font-size:24px; color:#94a3b8; letter-spacing:1px; margin-bottom:36px;">{title}</div>
  <div style="flex:1; display:flex; align-items:center; justify-content:center; min-height:0;">
    {body_html}
  </div>
  <div style="font-size:22px; color:#64748b; margin-top:36px; min-height:32px; text-align:center;">{caption}</div>
</div>
"""

    # ─── 共用：电影表格（state 1 / 2）──────────────────────
    def render_table(show_y: bool) -> str:
        head = (
            "<th style='padding:18px 36px;text-align:left;color:#64748b;font-weight:500;border-bottom:1px solid #1e293b;'>电影</th>"
            "<th style='padding:18px 36px;text-align:right;color:#64748b;font-weight:500;border-bottom:1px solid #1e293b;'>评分</th>"
            "<th style='padding:18px 36px;text-align:right;color:#64748b;font-weight:500;border-bottom:1px solid #1e293b;'>主演吸引度</th>"
        )
        if show_y:
            head += (
                "<th style='padding:18px 36px;text-align:center;color:#fbbf24;font-weight:600;"
                "border-bottom:1px solid #1e293b;background:rgba(251,191,36,0.08);'>标签 y</th>"
            )

        rows = ""
        for _, r in df.iterrows():
            cells = (
                f"<td style='padding:16px 36px;color:#e2e8f0;border-bottom:1px solid #1e293b;font-size:24px;'>{r['电影']}</td>"
                f"<td style='padding:16px 36px;text-align:right;color:#94a3b8;border-bottom:1px solid #1e293b;font-family:\"SF Mono\",monospace;font-size:24px;'>{r['评分']}</td>"
                f"<td style='padding:16px 36px;text-align:right;color:#94a3b8;border-bottom:1px solid #1e293b;font-family:\"SF Mono\",monospace;font-size:24px;'>{r['主演吸引度']}</td>"
            )
            if show_y:
                color = "#22c55e" if r["标签"] == "喜欢" else "#ef4444"
                cells += (
                    f"<td style='padding:16px 36px;text-align:center;color:{color};font-weight:600;"
                    f"border-bottom:1px solid #1e293b;background:rgba(251,191,36,0.06);font-size:24px;'>{r['标签']}</td>"
                )
            rows += f"<tr>{cells}</tr>"

        return f"""
<div style="background:#131c2e;border:1px solid #1e293b;border-radius:16px;padding:8px 0 0 0;min-width:1100px;">
  <table style="width:100%;border-collapse:collapse;">
    <thead><tr>{head}</tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>
"""

    # ─── 共用：代码块 ───────────────────────────────────────
    def render_code(lines, active_idx=None, max_w: int = 1400) -> str:
        if active_idx is None:
            active = set()
        elif isinstance(active_idx, int):
            active = {active_idx}
        else:
            active = set(active_idx)

        out_rows = []
        for i, line in enumerate(lines):
            is_a = i in active
            bg = "rgba(251,191,36,0.14)" if is_a else "transparent"
            border = (
                "border-left:5px solid #fbbf24;"
                if is_a
                else "border-left:5px solid transparent;"
            )
            color = "#fbbf24" if is_a else "#cbd5e1"
            weight = "font-weight:600;" if is_a else ""
            content = (
                line.replace(" ", "&nbsp;").replace("<", "&lt;").replace(">", "&gt;")
                if line
                else "&nbsp;"
            )
            out_rows.append(
                f"<div style=\"font-family:'SF Mono',Menlo,monospace;font-size:26px;"
                f"background:{bg};{border}color:{color};{weight}"
                f'padding:16px 36px;line-height:1.55;">{content}</div>'
            )

        return f"""
<div style="background:#0f172a;border:1px solid #1e293b;border-radius:16px;
            padding:24px 0;min-width:980px;max-width:{max_w}px;width:100%;">
  {''.join(out_rows)}
</div>
"""

    # ─── 共用：散点图 SVG ───────────────────────────────────
    def render_scatter(query=None, top5=(), highlight_neighbors=False) -> str:
        W, H = 1080, 720
        x_min, x_max = 2.0, 11.0
        y_min, y_max = 2.0, 11.0
        pad_l, pad_r, pad_t, pad_b = 96, 56, 56, 88

        def x2px(x: float) -> float:
            return pad_l + (x - x_min) / (x_max - x_min) * (W - pad_l - pad_r)

        def y2px(y: float) -> float:
            return H - pad_b - (y - y_min) / (y_max - y_min) * (H - pad_t - pad_b)

        parts = [
            f'<svg viewBox="0 0 {W} {H}" '
            f'style="width:100%;max-width:1080px;height:auto;'
            f'background:#131c2e;border:1px solid #1e293b;border-radius:16px;">'
        ]
        # 网格
        for v in range(2, 12, 2):
            parts.append(
                f'<line x1="{x2px(v):.1f}" y1="{pad_t}" x2="{x2px(v):.1f}" y2="{H-pad_b}" '
                f'stroke="#1e293b" stroke-width="1"/>'
            )
            parts.append(
                f'<line x1="{pad_l}" y1="{y2px(v):.1f}" x2="{W-pad_r}" y2="{y2px(v):.1f}" '
                f'stroke="#1e293b" stroke-width="1"/>'
            )
        # 轴标
        for v in range(2, 12, 2):
            parts.append(
                f'<text x="{x2px(v):.1f}" y="{H-pad_b+30}" '
                f'fill="#64748b" font-size="18" text-anchor="middle">{v}</text>'
            )
            parts.append(
                f'<text x="{pad_l-18}" y="{y2px(v)+6:.1f}" '
                f'fill="#64748b" font-size="18" text-anchor="end">{v}</text>'
            )
        # 轴标题
        parts.append(
            f'<text x="{(pad_l + W - pad_r) / 2:.1f}" y="{H-22}" '
            f'fill="#94a3b8" font-size="20" text-anchor="middle">评分</text>'
        )
        parts.append(
            f'<text x="28" y="{H/2:.1f}" '
            f'fill="#94a3b8" font-size="20" text-anchor="middle" '
            f'transform="rotate(-90 28 {H/2:.1f})">主演吸引度</text>'
        )
        # 5 邻居线（先画线，被点遮一部分没关系）
        if query is not None and len(top5) > 0:
            qx, qy = query
            for i in top5:
                tx = float(df.iloc[i]["评分"])
                ty = float(df.iloc[i]["主演吸引度"])
                parts.append(
                    f'<line x1="{x2px(qx):.1f}" y1="{y2px(qy):.1f}" '
                    f'x2="{x2px(tx):.1f}" y2="{y2px(ty):.1f}" '
                    f'stroke="#fbbf24" stroke-dasharray="6 6" stroke-width="2.5" opacity="0.85"/>'
                )
        # 9 数据点
        for idx, row in df.iterrows():
            px, py = x2px(row["评分"]), y2px(row["主演吸引度"])
            color = "#22c55e" if row["标签"] == "喜欢" else "#ef4444"
            is_neighbor = highlight_neighbors and idx in top5
            if is_neighbor:
                parts.append(
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="17" '
                    f'fill="{color}" stroke="#fbbf24" stroke-width="3.5"/>'
                )
            else:
                parts.append(
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="14" '
                    f'fill="{color}" opacity="0.9"/>'
                )
            # 电影名
            parts.append(
                f'<text x="{px+22:.1f}" y="{py+6:.1f}" '
                f'fill="#cbd5e1" font-size="16">{row["电影"]}</text>'
            )
        # 新点（流浪地球 3）
        if query is not None:
            qx, qy = query
            cx, cy = x2px(qx), y2px(qy)
            parts.append(
                f'<polygon points="{cx:.1f},{cy-20:.1f} {cx+20:.1f},{cy:.1f} '
                f'{cx:.1f},{cy+20:.1f} {cx-20:.1f},{cy:.1f}" '
                f'fill="#fbbf24" stroke="#0b1220" stroke-width="3"/>'
            )
            parts.append(
                f'<text x="{cx+30:.1f}" y="{cy+6:.1f}" '
                f'fill="#fbbf24" font-size="18" font-weight="600">流浪地球 3 ?</text>'
            )
        # 图例
        parts.append(
            f'<g transform="translate({W-pad_r-200}, {pad_t+8})">'
            f'<rect x="0" y="0" width="180" height="78" fill="#0f172a" '
            f'stroke="#1e293b" rx="8"/>'
            f'<circle cx="20" cy="22" r="10" fill="#22c55e"/>'
            f'<text x="40" y="27" fill="#cbd5e1" font-size="16">喜欢</text>'
            f'<circle cx="20" cy="54" r="10" fill="#ef4444"/>'
            f'<text x="40" y="59" fill="#cbd5e1" font-size="16">不喜欢</text>'
            f"</g>"
        )
        parts.append("</svg>")
        return "".join(parts)

    # ─── 9 个 state ─────────────────────────────────────────
    s = step.value.split("·")[0]

    six_lines = [
        "from sklearn.neighbors import KNeighborsClassifier",
        "",
        "model = KNeighborsClassifier(n_neighbors=5)",
        "model.fit(X, y)",
        "prediction = model.predict([[8.5, 9.5]])",
        "print(prediction)",
    ]

    if s == "1":
        body = render_table(show_y=False)
        html = stage(
            "Cell 1 · 数据 X · 9 部电影 × 2 个特征",
            body,
            "X 必须是 2D：每行一个样本，每列一个特征",
        )
    elif s == "2":
        body = render_table(show_y=True)
        html = stage(
            "Cell 1 · 加上标签 y · 离散类别",
            body,
            "y 必须是 1D：每个样本对应一个标签",
        )
    elif s == "3":
        body = render_code(six_lines, active_idx=0, max_w=1500)
        html = stage(
            "Cell 2 · 第 1 行 · import",
            body,
            "sklearn 把所有算法打包成一个个类",
        )
    elif s == "4":
        body = render_code(six_lines, active_idx=2, max_w=1500)
        html = stage(
            "Cell 2 · 第 2 行 · 构造（K = 5）",
            body,
            "三件套之一：构造 — 把超参数装进 model 对象",
        )
    elif s == "5":
        scatter = render_scatter()
        code = render_code(six_lines, active_idx=3)
        body = f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:48px;width:100%;
            max-width:1740px;align-items:center;">
  <div>{scatter}</div>
  <div>{code}</div>
</div>
"""
        html = stage(
            "Cell 2 · 第 3 行 · fit · 把 9 个样本喂给 model",
            body,
            "三件套之二：fit — KNN 这一步几乎只是存数据",
        )
    elif s == "6":
        X_arr = df[["评分", "主演吸引度"]].to_numpy()
        q = np.array(new_movie)
        d = np.linalg.norm(X_arr - q, axis=1)
        top5 = list(np.argsort(d)[:5])
        scatter = render_scatter(
            query=new_movie, top5=top5, highlight_neighbors=True
        )
        code = render_code(six_lines, active_idx=4)
        body = f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:48px;width:100%;
            max-width:1740px;align-items:center;">
  <div>{scatter}</div>
  <div>{code}</div>
</div>
"""
        html = stage(
            "Cell 2 · 第 4 行 · predict · 流浪地球 3 + 5 个最近邻",
            body,
            "三件套之三：predict — 找 K 个最近邻 → 多数表决",
        )
    elif s == "7":
        body = """
<div style="text-align:center;">
  <div style="font-size:22px;color:#64748b;letter-spacing:3px;margin-bottom:32px;">PREDICTION</div>
  <div style="font-family:'SF Mono',Menlo,monospace;font-size:160px;font-weight:700;
              color:#22c55e;line-height:1;text-shadow:0 0 32px rgba(34,197,94,0.25);">
    ['喜欢']
  </div>
  <div style="font-size:24px;color:#94a3b8;margin-top:56px;line-height:1.6;">
    5 个最近邻里 4 个标 <span style="color:#22c55e;font-weight:600;">喜欢</span> · 多数表决
  </div>
</div>
"""
        html = stage("Cell 2 · 输出", body, "")
    elif s == "8":
        # fit ≈ 109 µs / predict ≈ 667 µs，柱长按比例（fit 8% / predict 50%）
        body = """
<div style="width:100%;max-width:1320px;">
  <div style="font-size:18px;color:#64748b;letter-spacing:3px;margin-bottom:48px;
              text-align:center;">TIMEIT · 1000 次平均</div>
  <div style="display:flex;flex-direction:column;gap:48px;">
    <div style="display:flex;align-items:center;gap:36px;">
      <div style="width:160px;font-family:'SF Mono',monospace;font-size:32px;color:#94a3b8;">fit</div>
      <div style="flex:1;height:56px;background:#131c2e;border:1px solid #1e293b;
                  border-radius:8px;padding:4px;">
        <div style="width:8%;height:100%;background:linear-gradient(90deg,#22c55e,#16a34a);
                    border-radius:6px;"></div>
      </div>
      <div style="width:240px;font-family:'SF Mono',monospace;font-size:32px;color:#22c55e;
                  font-weight:600;text-align:right;">≈ 109 µs</div>
    </div>
    <div style="display:flex;align-items:center;gap:36px;">
      <div style="width:160px;font-family:'SF Mono',monospace;font-size:32px;color:#94a3b8;">predict</div>
      <div style="flex:1;height:56px;background:#131c2e;border:1px solid #1e293b;
                  border-radius:8px;padding:4px;">
        <div style="width:50%;height:100%;background:linear-gradient(90deg,#ef4444,#dc2626);
                    border-radius:6px;"></div>
      </div>
      <div style="width:240px;font-family:'SF Mono',monospace;font-size:32px;color:#ef4444;
                  font-weight:600;text-align:right;">≈ 667 µs</div>
    </div>
  </div>
  <div style="text-align:center;margin-top:72px;font-size:28px;color:#fbbf24;font-weight:600;">
    predict 慢 6× — 反直觉
  </div>
</div>
"""
        html = stage(
            "Cell 3 · 给 fit 和 predict 计时",
            body,
            "KNN 的 fit 几乎什么都没干",
        )
    elif s == "9":
        body = """
<div style="display:flex;flex-direction:column;gap:32px;
            font-family:'SF Mono',Menlo,monospace;font-size:36px;">
  <div style="background:#131c2e;border:1px solid #1e293b;border-radius:16px;
              padding:36px 56px;">
    <span style="color:#fbbf24;">model._fit_X</span><span style="color:#64748b;">.shape</span>
    <span style="color:#94a3b8;">  =  </span>
    <span style="color:#22c55e;font-weight:600;">(9, 2)</span>
  </div>
  <div style="background:#131c2e;border:1px solid #1e293b;border-radius:16px;
              padding:36px 56px;">
    <span style="color:#fbbf24;">model.classes_</span>
    <span style="color:#94a3b8;">          =  </span>
    <span style="color:#22c55e;font-weight:600;">array(['不喜欢', '喜欢'])</span>
  </div>
  <div style="font-size:22px;color:#64748b;line-height:1.7;margin-top:24px;
              font-family:-apple-system,'PingFang SC',sans-serif;">
    fit 把 X 原封不动存了 — 整个训练集本身就是模型
  </div>
</div>
"""
        html = stage(
            "Cell 4 · fit 之后看模型对象内部",
            body,
            "lazy learner 的本质：训练 = 存数据",
        )
    else:
        html = stage("", "", "")

    mo.Html(html)
    return


if __name__ == "__main__":
    app.run()
