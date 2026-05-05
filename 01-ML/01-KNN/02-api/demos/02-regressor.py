"""
KNN 回归 API · 找不同游戏（marimo 录屏 demo）

横屏布局：
  左：随 step 切换的可视化（标签 vs 星数对照表 / 散点图 + 5 邻居 / 输出对比卡）
  右：上下两份代码（E04 KNeighborsClassifier / E05 KNeighborsRegressor）
      同步高亮当前讲到的行；类名差异加红框

录屏：
  cd 01-ML/01-KNN/02-api/demos
  marimo run 02-regressor.py --headless --port 2763 --no-token
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full", layout_file="layouts/02-regressor.grid.json")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt

    return alt, mo, np, pd


@app.cell
def _(mo):
    mo.Html(
        """
    <style>
      main, [data-testid="cell-output"] .markdown { max-width: 1600px !important; }
      .marimo-cell, .Cell { max-width: 1600px !important; }
    </style>
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    # KNN 回归 API · 找不同游戏

    E05 录屏 demo · 同 9 部电影、同 sklearn，y 从分类标签换成连续打分
    """)
    return


@app.cell
def _(pd):
    movies = [
        ("流浪地球2", 8.4, 9.0, "喜欢", 4.5),
        ("阿凡达", 8.7, 8.5, "喜欢", 4.0),
        ("泰坦尼克", 9.5, 9.5, "喜欢", 4.3),
        ("复联4", 8.5, 9.2, "喜欢", 4.5),
        ("沙丘2", 8.0, 8.0, "喜欢", 4.5),
        ("哥斯拉", 6.4, 7.0, "不喜欢", 3.0),
        ("长津湖", 7.4, 8.0, "不喜欢", 2.5),
        ("寒战", 7.5, 6.5, "不喜欢", 3.0),
        ("上海堡垒", 2.9, 3.5, "不喜欢", 1.5),
    ]
    df = pd.DataFrame(
        movies, columns=["电影", "评分", "主演吸引度", "E04 标签 y", "E05 星数 y"]
    )
    new_movie = (8.5, 9.5)
    return df, new_movie


@app.cell
def _(mo):
    step = mo.ui.dropdown(
        options=[
            "1·数据 X 不变",
            "2·y 换星数",
            "3·import 差异",
            "4·构造差异",
            "5·fit & predict 一字不动",
            "6·输出 4.36",
            "7a·5 邻居 · 起始全表",
            "7b·5 邻居 · 复联4",
            "7c·5 邻居 · 流浪地球2",
            "7d·5 邻居 · 泰坦尼克",
            "7e·5 邻居 · 阿凡达",
            "7f·5 邻居 · 沙丘2",
            "7g·5 邻居 · 全表打分",
            "7h·平均 = 4.36",
        ],
        value="1·数据 X 不变",
        label="录屏叙事步骤",
    )
    step
    return (step,)


@app.cell
def _(alt, df, mo, new_movie, np, pd, step):
    s = step.value.split("·")[0]

    # ─── 左侧可视化 ─────────────────────────────────────────
    if s in ("1", "2"):
        # 表格对照：E04 标签 vs E05 星数
        if s == "1":
            _view = df[["电影", "评分", "主演吸引度"]]
            _viz_title = "X · 9 部电影 × 2 个特征（与 E04 完全相同）"
        else:
            _view = df  # 含两列 y
            _viz_title = "y · 左列分类标签（E04） / 右列连续星数（E05）"
        _viz = mo.ui.table(_view, page_size=10, selection=None, show_column_summaries=False)
    elif s in ("3", "4", "5"):
        # 中间这几步只看代码差异，左侧给个静态卡片提示
        _hints = {
            "3": ("import 这一行变了", "类名结尾：Classifier → Regressor"),
            "4": ("构造这一行变了", "类名跟着 import 一起换；n_neighbors 不动"),
            "5": ("剩下两行一字不动", "fit 和 predict 完全一样"),
        }
        _t, _sub = _hints[s]
        _viz = mo.md(
            f"""
    <div style="padding:80px 40px;background:#0f172a;color:#e2e8f0;border-radius:12px;border:1px solid #334155;text-align:center;">
      <div style="font-size:13px;color:#64748b;text-transform:uppercase;letter-spacing:2px;margin-bottom:24px;">右侧代码 →</div>
      <div style="font-size:32px;color:#fbbf24;font-weight:600;margin-bottom:18px;">{_t}</div>
      <div style="font-size:18px;color:#cbd5e1;line-height:1.7;">{_sub}</div>
    </div>
    """
        )
        _viz_title = "右侧代码对比"
    elif s == "6":
        # 输出对比卡：['喜欢'] vs [4.36]
        _viz = mo.md(
            """
    <div style="display:flex;flex-direction:column;gap:16px;">
      <div style="padding:32px 40px;background:#dcfce7;border:2px solid #22c55e;border-radius:12px;">
    <div style="font-size:13px;color:#15803d;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;">E04 · 分类输出</div>
    <div style="font-family:'SF Mono',Menlo,monospace;font-size:42px;color:#166534;font-weight:bold;">['喜欢']</div>
    <div style="font-size:13px;color:#15803d;margin-top:8px;">字符串数组（类别标签）</div>
      </div>
      <div style="padding:32px 40px;background:#dbeafe;border:2px solid #3b82f6;border-radius:12px;">
    <div style="font-size:13px;color:#1d4ed8;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;">E05 · 回归输出</div>
    <div style="font-family:'SF Mono',Menlo,monospace;font-size:42px;color:#1e3a8a;font-weight:bold;">[4.36]</div>
    <div style="font-size:13px;color:#1d4ed8;margin-top:8px;">浮点数（连续值）</div>
      </div>
    </div>
    """
        )
        _viz_title = "predict 输出形式：分类 → 回归"
    elif s.startswith("7"):
        # ─── 多 sub-state：7a-7h 逐邻居高亮 ───
        _sub_key = s[1:]  # "a".."h"
        _highlight_map = {"b": 0, "c": 1, "d": 2, "e": 3, "f": 4}
        _highlight_row = _highlight_map.get(_sub_key, -1)
        _all_stars = (_sub_key == "g")
        _show_avg = (_sub_key == "h")

        _scatter = (
            alt.Chart(df)
            .mark_circle(size=300, opacity=0.85)
            .encode(
                x=alt.X("评分:Q", scale=alt.Scale(domain=[2, 11]), axis=alt.Axis(title="评分")),
                y=alt.Y("主演吸引度:Q", scale=alt.Scale(domain=[2, 11]), axis=alt.Axis(title="主演吸引度")),
                color=alt.Color("E05 星数 y:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[1, 5]), legend=alt.Legend(orient="top-right", title="星数")),
                tooltip=["电影", "评分", "主演吸引度", "E05 星数 y"],
            )
        )
        _X = df[["评分", "主演吸引度"]].to_numpy()
        _q = np.array(new_movie)
        _d = np.linalg.norm(_X - _q, axis=1)
        _top5 = np.argsort(_d)[:5]

        _new_df = pd.DataFrame({"评分": [new_movie[0]], "主演吸引度": [new_movie[1]]})
        _new_pt = (
            alt.Chart(_new_df)
            .mark_point(size=600, shape="diamond", color="#fbbf24", strokeWidth=4, filled=True)
            .encode(x=alt.X("评分:Q"), y=alt.Y("主演吸引度:Q"))
        )

        _line_data = pd.DataFrame([
            {"评分": new_movie[0], "主演吸引度": new_movie[1],
             "评分_to": float(_X[i][0]), "主演吸引度_to": float(_X[i][1])}
            for i in _top5
        ])

        _layer_list = [_scatter, _new_pt]
        if _highlight_row >= 0:
            _other_lines = _line_data.drop(_highlight_row).reset_index(drop=True)
            _active_line = _line_data.iloc[[_highlight_row]]
            _layer_list.append(
                alt.Chart(_other_lines).mark_rule(stroke="#fbbf24", strokeDash=[4, 4], strokeWidth=1.5)
                .encode(x="评分:Q", y="主演吸引度:Q", x2="评分_to:Q", y2="主演吸引度_to:Q")
            )
            _layer_list.append(
                alt.Chart(_active_line).mark_rule(stroke="#fbbf24", strokeDash=[4, 4], strokeWidth=4)
                .encode(x="评分:Q", y="主演吸引度:Q", x2="评分_to:Q", y2="主演吸引度_to:Q")
            )
            _hi_idx = _top5[_highlight_row]
            _hi_df = pd.DataFrame({"评分": [float(_X[_hi_idx][0])], "主演吸引度": [float(_X[_hi_idx][1])]})
            _layer_list.append(
                alt.Chart(_hi_df).mark_circle(size=400, stroke="#000", strokeWidth=3, color="#fbbf24")
                .encode(x="评分:Q", y="主演吸引度:Q")
            )
        else:
            _layer_list.append(
                alt.Chart(_line_data).mark_rule(stroke="#fbbf24", strokeDash=[4, 4], strokeWidth=2)
                .encode(x="评分:Q", y="主演吸引度:Q", x2="评分_to:Q", y2="主演吸引度_to:Q")
            )

        _chart = alt.layer(*_layer_list).resolve_scale(color="independent").properties(width=540, height=320)

        _stars = [df["E05 星数 y"].iloc[i] for i in _top5]
        _names = [df["电影"].iloc[i] for i in _top5]
        _avg = sum(_stars) / 5

        _table_rows = ""
        for k, (i, n) in enumerate(zip(_top5, _names)):
            _is_hi = (k == _highlight_row)
            _is_dim = (_highlight_row >= 0 and not _is_hi)
            if _is_hi:
                _rs = "background:#fef3c7;border-left:4px solid #f59e0b;"
                _ns = "color:#1e293b;font-weight:700;"
                _ds = "color:#92400e;font-weight:600;"
                _ss = "color:#d97706;font-weight:700;"
            elif _is_dim:
                _rs = "opacity:0.4;border-left:4px solid transparent;"
                _ns = "color:#cbd5e1;"
                _ds = "color:#94a3b8;"
                _ss = "color:#94a3b8;"
            elif _all_stars:
                _rs = ""
                _ns = "color:#cbd5e1;"
                _ds = "color:#94a3b8;"
                _ss = "color:#fbbf24;font-weight:700;background:#fef3c7;border-radius:4px;padding:2px 6px;"
            else:
                _rs = ""
                _ns = "color:#cbd5e1;"
                _ds = "color:#94a3b8;"
                _ss = "color:#fbbf24;font-weight:600;"
            _table_rows += (
                f"<tr style='{_rs}'>"
                f"<td style='padding:4px 12px;{_ns}'>{n}</td>"
                f"<td style='padding:4px 12px;text-align:right;{_ds}'>{_d[i]:.2f}</td>"
                f"<td style='padding:4px 12px;text-align:right;{_ss}'>{_stars[k]}</td></tr>"
            )

        if _show_avg:
            _avg_html = (
                f'<div style="margin-top:16px;text-align:right;">'
                f'<span style="display:inline-block;padding:8px 22px;border-radius:8px;'
                f'background:#064e3b;font-size:30px;color:#22c55e;font-weight:700;'
                f'box-shadow:0 0 16px rgba(34,197,94,0.35);">平均 = {_avg:.2f}</span></div>'
            )
        else:
            _avg_html = (
                f'<div style="margin-top:8px;text-align:right;font-size:16px;color:#22c55e;font-weight:600;">'
                f'平均 = {_avg:.2f}</div>'
            )

        _verify_html = mo.md(
            f"""
    <div style="padding:16px 20px;background:#0f172a;color:#e2e8f0;border-radius:12px;border:1px solid #334155;font-family:'SF Mono',Menlo,monospace;margin-top:8px;">
      <div style="font-size:13px;color:#64748b;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;">5 个最近邻 · 平均</div>
      <table style="border-collapse:collapse;width:100%;font-size:14px;">
    <thead><tr style="border-bottom:1px solid #334155;">
      <th style='padding:6px 12px;text-align:left;color:#64748b;font-weight:normal;'>电影</th>
      <th style='padding:6px 12px;text-align:right;color:#64748b;font-weight:normal;'>距离</th>
      <th style='padding:6px 12px;text-align:right;color:#64748b;font-weight:normal;'>星数</th>
    </tr></thead>
    <tbody>{_table_rows}</tbody>
      </table>
      {_avg_html}
    </div>
    """
        )
        _viz = mo.vstack([mo.ui.altair_chart(_chart), _verify_html])
        _viz_title = "5 个最近邻 · 求平均 = 4.36"
    else:
        _viz = mo.md("")
        _viz_title = ""

    # ─── 右侧：上下两份代码同步对比 ───────────────────────
    # active_lines: 哪些行高亮（在 e04/e05 同 idx 的行同时高亮）
    # diff_lines: 哪些行加红框（仅在 import / 构造 行）
    if s == "3":
        _active = {0}  # import 行
        _diff = {0}
    elif s == "4":
        _active = {2}  # 构造行
        _diff = {2}
    elif s == "5":
        _active = {3, 4}  # fit + predict
        _diff = set()
    elif s == "6":
        _active = {5}  # print
        _diff = set()
    else:
        _active = set()
        _diff = set()

    def _render_block(label, color, lines, active, diff, diff_token=None):
        rows = []
        for i, line in enumerate(lines):
            is_a = i in active
            is_d = i in diff and diff_token
            bg = "#fef3c7" if is_a else "transparent"
            border = (
                "border-left:4px solid #f59e0b;"
                if is_a
                else "border-left:4px solid transparent;"
            )
            text_c = "#1e293b" if is_a else "#cbd5e1"
            weight = "font-weight:600;" if is_a else ""
            content = (
                line.replace(" ", "&nbsp;").replace("<", "&lt;").replace(">", "&gt;")
                if line
                else "&nbsp;"
            )
            # 在差异行内单独圈 diff_token（不圈整行）
            if is_d:
                _badge = (
                    f'<span style="outline:2px solid #dc2626;outline-offset:2px;'
                    f'background:#fee2e2;color:#991b1b;border-radius:3px;'
                    f'padding:1px 4px;font-weight:700;">{diff_token}</span>'
                )
                content = content.replace(diff_token, _badge)
            rows.append(
                f'<div style="font-family:\'SF Mono\',Menlo,Consolas,monospace;font-size:15px;'
                f"background:{bg};{border}color:{text_c};{weight}"
                f'padding:6px 16px;line-height:1.55;">{content}</div>'
            )
        label_html = (
            f'<div style="font-size:11px;color:{color};text-transform:uppercase;'
            f'letter-spacing:2px;padding:8px 16px 4px 16px;font-family:\'SF Mono\',monospace;">{label}</div>'
        )
        return (
            '<div style="background:#0f172a;border-radius:8px;border:1px solid #334155;'
            'overflow-x:auto;white-space:nowrap;width:720px;min-width:720px;margin-bottom:12px;">'
            + label_html
            + "".join(rows)
            + "</div>"
        )

    if s.startswith("7"):
        # 5 邻居验证用专门的 kneighbors 代码
        _kn_lines = [
            "# Cell 3 · 手验闭环",
            "import numpy as np",
            "",
            "distances, indices = model.kneighbors([[8.5, 9.5]])",
            "y_5 = [y[i] for i in indices[0]]",
            "print(f'平均 = {np.mean(y_5):.2f}')",
            "# → 4.36",
        ]
        _code_html = _render_block(
            "Cell 3 · KNN 回归内部 = 平均", "#3b82f6", _kn_lines, set(), set()
        )
        _code_title = "kneighbors · 5 邻居 + 平均"
    else:
        # 三件套对照（E04 / E05），按 _active 同步高亮
        _e04_lines = [
            "from sklearn.neighbors import KNeighborsClassifier",
            "",
            "model = KNeighborsClassifier(n_neighbors=5)",
            "model.fit(X, y)",
            "prediction = model.predict([[8.5, 9.5]])",
            "print(prediction)",
        ]
        _e05_lines = [
            "from sklearn.neighbors import KNeighborsRegressor",
            "",
            "model = KNeighborsRegressor(n_neighbors=5)",
            "model.fit(X, y)",
            "prediction = model.predict([[8.5, 9.5]])",
            "print(prediction)",
        ]
        _e04_html = _render_block(
            "E04 · 分类（KNeighborsClassifier）",
            "#22c55e",
            _e04_lines,
            _active,
            _diff,
            diff_token="KNeighborsClassifier",
        )
        _e05_html = _render_block(
            "E05 · 回归（KNeighborsRegressor）",
            "#3b82f6",
            _e05_lines,
            _active,
            _diff,
            diff_token="KNeighborsRegressor",
        )
        _code_html = _e04_html + _e05_html
        _code_title = "Cell 2 · 找不同游戏"

    # ─── 横屏拼接 ────────────────────────────────────────
    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md(f"#### 左 · {_viz_title}"),
                    _viz,
                ]
            ),
            mo.vstack(
                [
                    mo.md(f"#### 右 · {_code_title}"),
                    mo.Html(_code_html),
                ]
            ),
        ],
        widths=[500, 720],
        gap=2,
        align="start",
    )
    return


if __name__ == "__main__":
    app.run()
