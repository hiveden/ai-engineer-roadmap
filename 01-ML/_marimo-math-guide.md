# Marimo 数学/算法交互 Demo 设计手册

> **设计套路** · 把数学概念变成"5 秒看懂"的交互。配套 [`_marimo-guide.md`](./_marimo-guide.md)（API 速查）

## 索引（按需取读）

| § | 内容 | 何时读 |
|---|---|---|
| 何时用 | 用 marimo 做数学 demo 的判定 | 选工具时 |
| 设计三原则（5 秒测试）| 单焦点 / 直觉锚 / 反馈即时 | 起手设计 |
| 工具链选型 | matplotlib vs altair vs plotly | 画图选型 |
| 标配可视化模式 | 函数图 / 等值线 / 散点 + 决策边界 | 找模板 |
| 交互状态决策树 | slider / button / dropdown 选择 | 选控件 |
| 启动调试套路 | 端口 / hot reload / 嵌入 | 调试卡住 |
| 字体 / 显示 | 中文字体 / DPI / 视频友好 | 渲染问题 |
| 数据范围 + 数值稳定 | log / clip / 防 NaN | 数学退化点 |
| 文件结构建议 | imports / 控件 / 计算 / 渲染 | 写新 demo |
| 教学叙事模板 | 引导观众视线的 demo 套路 | 写讲解 demo |
| 反模式（避免）| 已踩过的坑 | 排错 |
| 复用片段（粘贴即用）| 现成代码 | 快速搭起 |
| 参考实现 | 已有 demo 的优秀样例 | 找灵感 |

> 控件 / 布局 API → [`_marimo-guide.md`](./_marimo-guide.md)；录屏友好的 cell 拆分 → [`_2-demo-guide.md`](./_2-demo-guide.md)

---

## 何时用 marimo 做数学 demo

| 适用 | 不适用 |
|---|---|
| 参数空间可视化（loss / 决策边界 / 概率分布）| 录视频讲解 → Manim |
| 算法步骤分解（GD / KNN / EM / 反向传播）| 几何拖拽（点线圆）→ GeoGebra |
| Loss / metric 实时反馈 | 静态信息图 → matplotlib + 导出 |
| 预设场景对比（不同 lr / 起点 / 超参）| 视频/PPT 嵌入 → 导出 HTML |

## 设计三原则（5 秒测试）

| 原则 | 反例 | 正例 |
|---|---|---|
| **单通道输入** | 滑块 + state + 按钮三个通道改同一对参数 | 滑块=唯一起点，其它都是它的派生 |
| **多视图破抽象** | 一张 2D 等高线讲完所有事 | 1D 切片 + 2D 等高线 + 3D 曲面 + 业务图 同步联动 |
| **预设 + 帧滑块** | 让用户自己摸索学习率 | 6 个典型 preset（完美/龟速/振荡/飞出/远距）+ 帧 0→30 慢动作 |

## 工具链选型

| 任务 | 推荐 | 备注 |
|---|---|---|
| 业务散点 + 拟合线 + hover | **altair** | mo.ui.altair_chart 选择回 Python |
| 等高线地形图（俯视） | **matplotlib contourf** | 加 white contour 线条 + log levels 谷底分层 |
| 3D 曲面（透视拖旋） | **plotly Surface** | 只此一家，scene.camera 控视角 |
| 1D 抛物线/曲线切片 | **altair** | 轻量，叠 mark_circle 标当前点 |
| 几何动画（向量/旋转） | **matplotlib FuncAnimation 导 gif** | marimo 内不流畅，外部生成贴 |
| 符号公式实时渲染 | **sympy + mo.md(sp.latex(...))** | f-string 嵌即可 |

## 标配可视化模式

| 模式 | 用途 | 实现要点 |
|---|---|---|
| **残差方块** | 让"平方误差"看得见 | `mark_square` + `size=alt.Size('err_sq', range=[20, 4000])`，搭红色虚线连真实/预测点 |
| **抛物线切片** | 把 2D loss 拆成两个 1D 直觉 | 固定一参，扫另一参，画 line + 当前红点 + 最优绿钻石 |
| **等高线 + 轨迹** | 看"碗"形 + GD 路径 | log-spaced levels；GD 轨迹 `'o-'` 白色叠在上面 |
| **3D 曲面 + 漂浮点** | 拖拽看几何形状 | plotly Surface + Scatter3d 当前/最优 + 白线轨迹 |
| **预设 dropdown + 帧 slider** | 一键复现典型情况 + 慢动作 | 预跑全轨迹，frame 取 history[:f+1] |
| **模式徽章** | 提示当前用滑块还是预设 | mo.md 里嵌 span 带背景色（蓝=手动 / 黄=预设） |

## 交互状态决策树

```
用户输入 → 输出？
├─ 实时联动（loss 数字、图形态）
│   └─ 直接读 slider.value，不用 state
├─ 累积历史（GD 一步一步走）
│   └─ ❌ 不要 state + 按钮 ✅ 改"自动跑 N 步" + frame 回放
└─ 多控件互相同步（拖滑块 = 改起点 = 重算白线）
    └─ 单一来源原则：滑块=输入，其它都是计算结果
```

**核心教训**：`mo.state` + `run_button` 适合"探索式累积"（笔记本场景），不适合"教学演示"。教学要**所见即所得**，每个滑块改动应立刻体现在所有视图。

## 启动调试套路

| 步骤 | 命令 |
|---|---|
| 自测（不开浏览器） | `marimo export script nb.py -o /tmp/_lint.py && python /tmp/_lint.py` 跑通=所有 cell 无错 |
| 启动（项目 venv 直跑） | `.venv/bin/marimo edit --port 2721 --headless --no-token nb.py` |
| 端口管理 | `lsof -iTCP:2721 -sTCP:LISTEN \| awk '{print $2}' \| xargs kill` 只杀指定端口 |
| 多端口共存 | 每个 demo 独占一个端口（2718/2719/2720...），避免 take-over session 弹窗 |
| 强刷 token | 重启后浏览器报 token 不匹配 → 强刷一次 |

## 字体 / 显示

| 问题 | 解法 |
|---|---|
| matplotlib 中文方框 | `plt.rcParams["font.sans-serif"]=["PingFang SC","Heiti SC","Arial Unicode MS","DejaVu Sans"]` |
| altair 表格 pyarrow 警告 | 安装 `pyarrow` 或忽略（CSV fallback 不影响功能）|
| plotly 中文标题 | layout.font.family="PingFang SC" |
| 坐标轴单位差异导致"方块"变形 | 用 `size=` 编码（屏幕像素）而非 `x2,y2`（数据空间）|

## 数据范围 + 数值稳定

| 场景 | 处理 |
|---|---|
| GD 发散（"飞出去"预设） | 每步 `np.clip(val, -1e6, 1e6)`，避免 NaN/inf 让 plotly/contourf 爆炸 |
| log color scale 谷底 | `np.exp(np.linspace(np.log(min+0.1), np.log(max), N))` 避免 log(0) |
| 网格分辨率 | 教学 50×50 足够，不要 200×200（卡 + 看不出差别）|

## 文件结构建议

```
sandbox/math-demos/
├── loss-landscape.py        # 简单版（只演示一个概念）
├── loss-landscape-v2.py     # 进阶版（多视图 + preset）
└── README.md                # demo 索引 + 端口分配
```

每个 demo 文件头部 docstring 写清：
1. 演示什么概念
2. 启动命令（含端口）
3. 互动玩法（≤ 5 条）

## 教学叙事模板

| 阶段 | 内容 |
|---|---|
| **抓眼球** | 一句话场景：用 `y=wx+b` 拟合 10 个点，让红方块越小越好 |
| **建直觉** | 1D 抛物线（B1/B2）→ "调一个旋钮 loss 怎么变" |
| **升维** | 2D 等高线（C）→ "两个旋钮拼成地形" |
| **震撼** | 3D 曲面（D）→ "拖拽看碗的形状" |
| **对比锚点** | 5 个 preset → "lr 太小/太大/合适 各什么样" |
| **回放** | 帧滑块 0→30 → "GD 每一步在哪" |

## 反模式（避免）

- ❌ 一上来 3D 曲面：新手判读 (w,b,loss) 三坐标慢，先 1D → 2D → 3D 渐进
- ❌ 滑块语义双重：既当"红点位置"又当"GD 起点"→ 必有一处违反直觉
- ❌ 只给"自由探索"无 preset：用户没参照系，调几下就放弃
- ❌ 一个图塞 8 种 mark：信息密度高 ≠ 易懂，分多图同步联动
- ❌ 中文标题 + matplotlib 默认字体：方框雪崩
- ❌ `mo.state` + `run_button` 做实时反馈：滞后感强，改用纯 reactive

## 复用片段（粘贴即用）

```python
# 残差方块 + 拟合线（A 模式）
df['err_sq'] = (df.y - df.y_pred) ** 2
squares = alt.Chart(df).mark_square(opacity=0.35, color='#ef4444').encode(
    x='x', y='y', size=alt.Size('err_sq:Q', scale=alt.Scale(range=[20,4000]), legend=None)
)

# 等高线 + GD 轨迹（C 模式 · matplotlib）
levels = np.exp(np.linspace(np.log(L.min()+0.1), np.log(L.max()), 18))
ax.contourf(W, B, L, levels=levels, cmap='viridis_r', alpha=0.85)
ax.contour(W, B, L, levels=levels, colors='white', linewidths=0.5, alpha=0.4)
ax.plot(hist[:,0], hist[:,1], 'o-', color='white', markersize=3)

# 3D 曲面（D 模式 · plotly）
fig.add_trace(go.Surface(x=ws, y=bs, z=L, colorscale='Viridis', reversescale=True,
    contours={"z":{"show":True,"usecolormap":True,"project_z":True}}))
fig.update_layout(scene={"camera":{"eye":{"x":1.6,"y":-1.6,"z":1.0}}})

# 预设 + 帧 slider（统一 GD 来源）
if preset.value is None:
    w0, b0, lr_use = w.value, b.value, lr.value
else:
    w0, b0, lr_use = preset.value
traj = [(w0, b0)]
cw, cb = w0, b0
for _ in range(30):
    g_w = 2*np.mean((cw*x + cb - y)*x); g_b = 2*np.mean(cw*x + cb - y)
    cw -= lr_use*g_w; cb -= lr_use*g_b
    cw = float(np.clip(cw, -1e6, 1e6)); cb = float(np.clip(cb, -1e6, 1e6))
    traj.append((cw, cb))
f = min(frame.value, len(traj)-1)
cur_w, cur_b = traj[f]
history = traj[:f+1]
```

## 参考实现

- `sandbox/math-demos/loss-landscape-v2.py` — 完整范本：A+B+C+D 四视图 + 6 预设 + 帧回放
