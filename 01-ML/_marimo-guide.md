# Marimo 实用速查手册

## Cell 基础

| 规则 | 要点 | 代码 |
|---|---|---|
| 装饰器 | 函数即 cell | `@app.cell` `def _(x): ...; return (y,)` |
| 返回值 | 元组导出全局变量 | `return (y,)` 单值也要逗号 |
| 局部变量 | `_` 前缀仅本 cell 可见 | `_tmp = 100` |
| 函数封装 | 内部变量不污染全局 | `def _(): ...; return ax` |
| 隐藏 | `mo.md("...").style("display:none")` 或 cell 头部 `# hide_code` | — |
| 最后表达式 | 自动渲染为 cell 输出 | `chart`（不要 `print(chart)`）|

## 反应式 + 变量冲突

| 规则 | 含义 |
|---|---|
| 静态依赖图 | 变量名匹配 → 自动追踪下游 cell |
| **每变量仅一处定义** | 同名 `x` 出现在两 cell 即报错 |
| **不追踪 mutation** | `lst.append()` 跨 cell 不触发，重新赋值才会 |
| 单 cell 内可变 | `df["new"]=...` 同 cell 内合法 |
| 无 magic | 不支持 `%pip` `%matplotlib`；用 `subprocess` / `os` |

## UI 控件清单

```python
mo.ui.slider(0, 100); mo.ui.number(); mo.ui.text(); mo.ui.text_area()
mo.ui.dropdown(options=["A","B"]); mo.ui.multiselect(options=[...])
mo.ui.switch(); mo.ui.checkbox(); mo.ui.radio(options=[...])
mo.ui.date(); mo.ui.file(); mo.ui.file_browser(); mo.ui.code_editor()
mo.ui.button(label="x"); mo.ui.run_button(); mo.ui.refresh()
mo.ui.array([s1, s2]); mo.ui.dictionary({"k": s1}); mo.ui.form(elem)
```

读值：`slider.value`。容器：`arr.value` 是 list / dict。

## 布局原语

| API | 用途 | 代码 |
|---|---|---|
| `mo.hstack` | 横排 | `mo.hstack([a, b])` |
| `mo.vstack` | 竖排 | `mo.vstack([a, b])` |
| `mo.ui.tabs` | 标签页 | `mo.ui.tabs({"T1": a, "T2": b})` |
| `mo.accordion` | 折叠 | `mo.accordion({"详情": content})` |
| `mo.callout` | 提示框 | `mo.callout("note", kind="info")` |
| `mo.sidebar` | 侧边栏 | `mo.sidebar([nav])` |
| `mo.nav_menu` | 导航 | `mo.nav_menu({"Home":"/"})` |
| `mo.lazy` | 延迟渲染 | `mo.lazy(lambda: heavy())` |

## 数据展示

| API | 场景 | 代码 |
|---|---|---|
| `mo.md` | Markdown / LaTeX / f-string 嵌 UI | `mo.md(f"x={slider}")` |
| `mo.ui.table` | 可选可排序表格 | `mo.ui.table(df, selection="multi")` |
| `mo.json` | 折叠 JSON | `mo.json(data)` |
| `mo.tree` | 树形结构 | `mo.tree(["a", {"b": [1,2]}])` |
| `mo.sql` | SQL on df / DuckDB | `mo.sql("SELECT * FROM df")` |

## 图表（Altair 一等公民）

| 库 | 互动 | 代码 |
|---|---|---|
| Altair | 选择自动回 Python（`chart.value` → DataFrame）| `chart = mo.ui.altair_chart(alt.Chart(df).mark_point().encode(...))` |
| Plotly | 仅散点/柱/直方/热力/树状/旭日支持选择 | `mo.ui.plotly(go.Figure(...))` |
| Matplotlib | 末行返回 Axes/Figure；**别用 `plt.show()`** | `ax.plot(...); ax` |

## 状态（mo.state）

| 何时用 | 99% 不用，优先 reactive 变量 |
|---|---|
| 用例 | 历史累积 / 两控件互相同步 / 显式循环 |
| API | `g, s = mo.state(0)`；`s(1)` 或 `s(lambda v: v+1)` |
| 自循环 | `mo.state(0, allow_self_loops=True)` |

```python
get_n, set_n = mo.state(0)
btn = mo.ui.button(on_change=lambda _: set_n(lambda v: v+1))
mo.md(f"count={get_n()}")
```

## 缓存

| API | 持久化 | 代码 |
|---|---|---|
| `@mo.cache` | 内存（重启失效）| `@mo.cache\ndef f(x): ...` |
| `@mo.persistent_cache` | 磁盘 `__marimo__/cache/` | `@mo.persistent_cache\ndef f(x): ...` |
| `@mo.lru_cache(maxsize=128)` | 有界内存 | `@mo.lru_cache(maxsize=128)` |
| context manager | 块级 | `with mo.persistent_cache("k"): ...` |

注：参数需可哈希；DataFrame 用 hash 不稳定，建议先序列化为 key。

## 样式

| 配置 | 代码 |
|---|---|
| 注入 CSS | `marimo.App(css_file="custom.css")` |
| 注入 `<head>` | `marimo.App(html_head_file="head.html")` |
| 公共 CSS 变量 | `--marimo-monospace-font` / `--marimo-text-font` / `--marimo-heading-font` |
| 强制主题 | 脚本头 `# [tool.marimo.display]\n# theme = "dark"` |
| 选择器 | `[data-cell-name='x'] { ... }` |

## 模式与启动命令

| 模式 | 命令 | 说明 |
|---|---|---|
| edit | `marimo edit nb.py` | 开发；带可写编辑器 |
| run（app）| `marimo run nb.py` | 部署只读应用 |
| script | `python nb.py` | 当普通脚本跑，print 到 terminal |
| wasm | `marimo edit --sandbox` 或导出 HTML | Pyodide 浏览器端，无 IO |
| convert | `marimo convert x.ipynb` | jupyter → marimo |
| export | `marimo export html\|ipynb\|script nb.py -o out` | 静态产物 |
| 自检 | `marimo check nb.py` | 静态分析（依赖/冲突）|

常用参数：`--port 2718` / `--headless` / `--no-token`（关闭 CSRF token）/ `--watch`（文件变更自动 reload）。

## 测试

| 方式 | 代码 |
|---|---|
| pytest | `pytest nb.py`（marimo 文件即 Python，可被收集）|
| 函数级 | cell 内 `def test_x(): assert ...`，pytest 直接发现 |
| doctest | docstring 内 `>>>` 由 `pytest --doctest-modules` 跑 |

## 常见陷阱

| 陷阱 | 解法 |
|---|---|
| matplotlib fig 不显示 | 末行返回 `ax` 或 `fig`，删掉 `plt.show()` |
| 跨 cell 改 list/dict 不刷新 | 改为重新赋值新对象 |
| 同名变量冲突报错 | 改名 / 加 `_` 前缀 / 函数封装 |
| 全局命名空间污染 | 用 `def _(): ...; return v` 包一层 |
| 中文字体方框 | matplotlib：`plt.rcParams["font.sans-serif"]=["PingFang SC"]` |
| inline `style=` 不生效 | 用 `.style({"color":"red"})` 或 css_file |
| class 名被 hash | marimo 给输出加哈希前缀；用 `[data-cell-name]` 选 |
| Plotly 选择无效 | 仅特定图类型支持，必要时切 Altair |
| 缓存未命中 | 参数含不可哈希对象（DataFrame）；先转 key |
| WASM 失败 | wasm 模式无文件系统/网络 socket，用 fetch 或 pyodide micropip |

## 实战教训

| 主题 | 教训 |
|---|---|
| **字体定制** | 公共 CSS 变量只 3 个（text/mono/heading），其他 class 名 hash 化跨版本会断 |
| **图表库选择** | **Altair 一等公民**，比 matplotlib 在 marimo 渲染更稳；选择互动也只 Altair 完整支持 |
| **`width="container"`** | 不可靠，需父容器显式 CSS 才生效；推荐固定 px 值 |
| **自检** | 改完跑 `python -c "from x import app; app.run()"` 静态检查，能抓变量冲突 / 语法错误，不用启 marimo edit |

## 文档链接

- Reactivity: https://docs.marimo.io/guides/reactivity/
- Interactivity: https://docs.marimo.io/guides/interactivity/
- Working with data: https://docs.marimo.io/guides/working_with_data/
- Plotting: https://docs.marimo.io/guides/working_with_data/plotting/
- State: https://docs.marimo.io/guides/state/
- Caching API: https://docs.marimo.io/api/caching/
- Theming: https://docs.marimo.io/guides/configuration/theming/
- Coming from Jupyter: https://docs.marimo.io/guides/coming_from/jupyter/
