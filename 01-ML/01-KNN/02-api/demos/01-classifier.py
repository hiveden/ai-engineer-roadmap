"""
KNN 分类 API · sklearn 三件套（marimo 录屏 demo）

横屏布局：
  左：随 step 切换的可视化（表格 / 模型卡 / 散点图 / 输出 / 计时对照 / fit 内部）
  右：完整代码块不动，按 step 高亮当前讲到的行

录屏（按 cue 词切 dropdown）：
  cd 01-ML/01-KNN/02-api/demos
  marimo run 01-classifier.py --headless --port 2760 --no-token
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt

    return alt, mo, np, pd


@app.cell
def _(mo):
    # 全局 CSS：固定 main 容器宽度 + 禁用 markdown 默认 max-width
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
    mo.md(
        """
        # KNN 分类 API · sklearn 三件套

        E04 录屏 demo · 9 部电影 + 流浪地球 3，预测会不会喜欢
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
        label="录屏叙事步骤",
    )
    step
    return (step,)


@app.cell
def _(alt, df, mo, new_movie, np, pd, step):
    # ─── 左侧可视化 ─────────────────────────────────────────
    s = step.value.split("·")[0]

    if s == "1":
        _viz = mo.ui.table(
            df[["电影", "评分", "主演吸引度"]],
            page_size=10,
            selection=None,
            show_column_summaries=False,
        )
        _viz_title = "数据集 X · 9 部电影 × 2 个特征"
    elif s == "2":
        _viz = mo.ui.table(
            df, page_size=10, selection=None, show_column_summaries=False
        )
        _viz_title = "数据集 X + y · 加上标签列"
    elif s in ("3", "4"):
        if s == "3":
            _cls_line = "KNeighborsClassifier  &nbsp;<span style='color:#94a3b8;font-size:14px;'># 类已导入</span>"
            _sub = "等待实例化"
        else:
            _cls_line = (
                'KNeighborsClassifier(<span style="color:#fbbf24;">n_neighbors=5</span>)'
            )
            _sub = "模型对象创建（K=5）"
        _viz = mo.md(
            f"""
<div style="padding:60px;background:#0f172a;color:#e2e8f0;border-radius:12px;text-align:center;border:1px solid #334155;">
  <div style="font-family:'SF Mono',Menlo,monospace;font-size:28px;color:#fbbf24;margin-bottom:18px;">model</div>
  <div style="font-family:'SF Mono',Menlo,monospace;font-size:22px;color:#f1f5f9;line-height:1.7;">{_cls_line}</div>
  <div style="font-size:14px;color:#94a3b8;margin-top:24px;">{_sub}</div>
</div>
"""
        )
        _viz_title = "模型对象"
    elif s in ("5", "6"):
        _scatter = (
            alt.Chart(df)
            .mark_circle(size=300, opacity=0.85)
            .encode(
                x=alt.X("评分:Q", scale=alt.Scale(domain=[2, 11]), axis=alt.Axis(title="评分")),
                y=alt.Y("主演吸引度:Q", scale=alt.Scale(domain=[2, 11]), axis=alt.Axis(title="主演吸引度")),
                color=alt.Color(
                    "标签:N",
                    scale=alt.Scale(
                        domain=["喜欢", "不喜欢"], range=["#22c55e", "#ef4444"]
                    ),
                    legend=alt.Legend(orient="top-right"),
                ),
                tooltip=["电影", "评分", "主演吸引度", "标签"],
            )
        )
        if s == "5":
            _chart = _scatter.properties(width=560, height=480)
            _viz_title = "散点图 · fit 后的训练集分布"
        else:
            X_arr = df[["评分", "主演吸引度"]].to_numpy()
            q = np.array(new_movie)
            d = np.linalg.norm(X_arr - q, axis=1)
            top5 = np.argsort(d)[:5]

            new_df = pd.DataFrame(
                {
                    "评分": [new_movie[0]],
                    "主演吸引度": [new_movie[1]],
                    "电影": ["流浪地球3"],
                }
            )
            _new_pt = (
                alt.Chart(new_df)
                .mark_point(
                    size=600, shape="diamond", color="#fbbf24", strokeWidth=4, filled=True
                )
                .encode(
                    x=alt.X("评分:Q", axis=alt.Axis(title="评分")),
                    y=alt.Y("主演吸引度:Q", axis=alt.Axis(title="主演吸引度")),
                )
            )
            # line_data 用与 scatter 同名 field（"评分"/"主演吸引度"），避免 layer axis title 串字
            _line_data = pd.DataFrame(
                [
                    {
                        "评分": new_movie[0],
                        "主演吸引度": new_movie[1],
                        "评分_to": float(X_arr[i][0]),
                        "主演吸引度_to": float(X_arr[i][1]),
                    }
                    for i in top5
                ]
            )
            _lines = (
                alt.Chart(_line_data)
                .mark_rule(stroke="#fbbf24", strokeDash=[4, 4], strokeWidth=2)
                .encode(
                    x="评分:Q",
                    y="主演吸引度:Q",
                    x2="评分_to:Q",
                    y2="主演吸引度_to:Q",
                )
            )
            # scatter 放第一层让 axis title 用 scatter 的设置（评分/主演吸引度）
            _chart = (
                (_scatter + _lines + _new_pt)
                .resolve_scale(color="independent")
                .properties(width=560, height=480)
            )
            _viz_title = "predict · 流浪地球 3 + 5 邻居（黄虚线）"
        _viz = mo.ui.altair_chart(_chart)
    elif s == "7":
        _viz = mo.md(
            """
<div style="padding:80px 60px;background:#dcfce7;border:3px solid #22c55e;border-radius:12px;text-align:center;">
  <div style="font-size:18px;color:#15803d;margin-bottom:24px;font-family:monospace;">prediction =</div>
  <div style="font-family:'SF Mono',Menlo,monospace;font-size:72px;color:#166534;font-weight:bold;letter-spacing:2px;">['喜欢']</div>
  <div style="font-size:14px;color:#15803d;margin-top:32px;">5 个最近邻里 4 个标"喜欢" → 多数表决</div>
</div>
"""
        )
        _viz_title = "predict 输出"
    elif s == "8":
        _viz = mo.md(
            """
<div style="padding:50px;background:#0f172a;color:#e2e8f0;border-radius:12px;font-family:'SF Mono',Menlo,monospace;border:1px solid #334155;">
  <div style="font-size:13px;color:#64748b;text-transform:uppercase;letter-spacing:2px;margin-bottom:28px;">timeit · 1000 次平均</div>
  <div style="font-size:26px;line-height:2.2;">
    <span style="color:#94a3b8;">fit&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span style="color:#22c55e;font-weight:bold;">≈ 109 µs</span> &nbsp;<span style="background:#22c55e;color:#022c22;padding:3px 8px;border-radius:3px;">█</span><br>
    <span style="color:#94a3b8;">predict&nbsp;</span><span style="color:#ef4444;font-weight:bold;">≈ 667 µs</span> &nbsp;<span style="background:#ef4444;color:#450a0a;padding:3px 8px;border-radius:3px;">██████</span> &nbsp;<span style="color:#fbbf24;font-size:22px;">慢 6×</span>
  </div>
  <div style="font-size:14px;color:#64748b;margin-top:32px;line-height:1.6;">predict 比 fit 慢——跟一般直觉完全相反。<br>因为 KNN 的 fit 几乎什么都没干。</div>
</div>
"""
        )
        _viz_title = "fit vs predict 计时对照"
    elif s == "9":
        _viz = mo.md(
            """
<div style="padding:50px;background:#0f172a;color:#e2e8f0;border-radius:12px;font-family:'SF Mono',Menlo,monospace;border:1px solid #334155;">
  <div style="font-size:13px;color:#64748b;text-transform:uppercase;letter-spacing:2px;margin-bottom:28px;">fit 之后看模型对象内部</div>
  <div style="font-size:22px;line-height:2.2;">
    <span style="color:#fbbf24;">model._fit_X</span><span style="color:#94a3b8;">.shape</span> &nbsp;= &nbsp;<span style="color:#22c55e;font-weight:bold;">(9, 2)</span><br>
    <span style="color:#fbbf24;">model.classes_</span> &nbsp; &nbsp; &nbsp;= &nbsp;<span style="color:#22c55e;font-weight:bold;">['不喜欢' '喜欢']</span>
  </div>
  <div style="font-size:14px;color:#64748b;margin-top:32px;line-height:1.6;">fit 把 X 原封不动存了——9 行 2 列<br>整个训练集本身就是模型</div>
</div>
"""
        )
        _viz_title = "fit 后模型内部"
    else:
        _viz = mo.md("")
        _viz_title = ""

    # ─── 右侧代码块（按 step 切换 + 高亮当前行）────────────
    if s == "1":
        # 只显示 X
        _code_lines = [
            ("# Cell 1 · 数据准备", False),
            ("X = [[8.4, 9.0],   # 流浪地球 2", False),
            ("     [8.7, 8.5],   # 阿凡达", False),
            ("     [9.5, 9.5],   # 泰坦尼克", False),
            ("     [6.4, 7.0],   # 哥斯拉", False),
            ("     [2.9, 3.5],   # 上海堡垒", False),
            ("     [8.0, 8.0],   # 沙丘 2", False),
            ("     [8.5, 9.2],   # 复联 4", False),
            ("     [7.5, 6.5],   # 寒战", False),
            ("     [7.4, 8.0]]   # 长津湖", False),
        ]
        _code_title = "Cell 1 · 数据准备（X）"
    elif s == "2":
        # X + y，y 整段高亮
        _code_lines = [
            ("# Cell 1 · 数据准备", False),
            ("X = [[8.4, 9.0], [8.7, 8.5], [9.5, 9.5],", False),
            ("     [6.4, 7.0], [2.9, 3.5],", False),
            ("     [8.0, 8.0], [8.5, 9.2],", False),
            ("     [7.5, 6.5], [7.4, 8.0]]", False),
            ("", False),
            ("y = ['喜欢', '喜欢', '喜欢',", True),
            ("     '不喜欢', '不喜欢',", True),
            ("     '喜欢', '喜欢',", True),
            ("     '不喜欢', '不喜欢']", True),
        ]
        _code_title = "Cell 1 · 数据准备（X + y）"
    elif s == "8":
        # 用 IPython magic，行短不会溢出
        _code_lines = [
            ("# Cell 3 · 给 fit 和 predict 分别计时", False),
            ("%timeit model.fit(X, y)", False),
            ("%timeit model.predict([[8.5, 9.5]])", False),
        ]
        _code_title = "Cell 3 · 计时"
    elif s == "9":
        _code_lines = [
            ("# fit 后看模型对象的属性", False),
            ("print('model._fit_X.shape :', model._fit_X.shape)", False),
            ("print('model.classes_     :', model.classes_)", False),
        ]
        _code_title = "Cell 4 · fit 内部"
    else:
        # 三件套 6 行（1-7 步用同一份代码块）
        _active_idx = {"3": 0, "4": 2, "5": 3, "6": 4, "7": 5}.get(s, -1)
        _code_lines = [
            ("from sklearn.neighbors import KNeighborsClassifier", _active_idx == 0),
            ("", False),
            ("model = KNeighborsClassifier(n_neighbors=5)   # 1. 构造", _active_idx == 2),
            ("model.fit(X, y)                                # 2. 训练", _active_idx == 3),
            ("prediction = model.predict([[8.5, 9.5]])       # 3. 预测", _active_idx == 4),
            ("print(prediction)", _active_idx == 5),
        ]
        _code_title = "Cell 2 · sklearn 三件套"

    _rows = []
    for _line, _active in _code_lines:
        _bg = "#fef3c7" if _active else "transparent"
        _border = (
            "border-left:4px solid #f59e0b;"
            if _active
            else "border-left:4px solid transparent;"
        )
        _color = "#1e293b" if _active else "#cbd5e1"
        _weight = "font-weight:600;" if _active else ""
        _content = (
            _line.replace(" ", "&nbsp;").replace("<", "&lt;").replace(">", "&gt;")
            if _line
            else "&nbsp;"
        )
        _rows.append(
            f'<div style="font-family:\'SF Mono\',Menlo,Consolas,monospace;font-size:17px;'
            f"background:{_bg};{_border}color:{_color};{_weight}"
            f'padding:8px 16px;line-height:1.6;transition:all 0.2s;">{_content}</div>'
        )
    _code_html = (
        '<div style="background:#0f172a;border-radius:8px;padding:18px;'
        "border:1px solid #334155;width:720px;min-width:720px;"
        'overflow-x:auto;white-space:nowrap;">'
        + "".join(_rows)
        + "</div>"
    )

    # ─── 横屏拼接（固定宽度，不受 viewport 影响）────────
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
        widths=[600, 720],
        gap=2,
        align="start",
    )
    return


if __name__ == "__main__":
    app.run()
