# 03b-math demo 开发日志

文件：`demos/mse-residual-squares.py` · 端口 2730 · ~290 行 / 14 cell

## review 必修项落地

| # | 必修项 | 落地 | 说明 |
|---|---|---|---|
| 1 | 删除 `frame` 滑块 | ✓ | 控件表只保留 `preset` / `k_slider` / `b_slider` 三项 |
| 2 | 预设/手动状态转移 | ✓（变通） | 见下文「偏离说明」 |
| 3 | 子图标题自解释 | ✓ | A：「数据点 + 当前拟合（红线）+ 最优拟合（绿虚）+ 残差平方（红方块越大误差越大）」；B1：「固定 b={cur_b:.2f}，loss 随 k 变化（红=当前 k，绿钻=最优 k\*={k_opt:.2f}）」；B2 同结构 |
| 4 | 指标面板完整 | ✓ | 模式徽章 + 当前 k/b/loss + 最优 k\*/b\*/loss\* + 当前/最优比值，颜色随 gap 变（绿/黄/红） |
| 5 | 中文字体 | ✓ | matplotlib `font.sans-serif=[PingFang SC, Heiti SC, Arial Unicode MS, DejaVu Sans]` 已设；图表全部 altair（无 matplotlib 渲染中文） |
| 6 | 模式徽章 | ✓ | 控件区上方、指标面板内各一处。蓝底「🔵 手动模式」/ 黄底「🟡 预设模式」 |

## review 软建议落地情况

| # | 软建议 | 是否采纳 | 说明 |
|---|---|---|---|
| 1 | 数据点数加到 8-10 | ✗ | 保留 5 点，对齐 design 第 87 行 + slide 24 算例。后续如果觉得方块阵列太稀疏可微调 |
| 2 | 绿虚线/绿钻石 hover tooltip | ✓ | A 视图绿虚线、B1/B2 绿钻石都加了 altair tooltip（最优值 + 切片 loss） |
| 3 | LaTeX 损失公式 | ✓ | 顶部 mo.md 渲染 $L(k,b) = \frac{1}{n}\sum (y_i - kx_i - b)^2$；底部回顾区还展开闭式解公式 |
| 4 | 配色 a11y | 部分 | 绿钻石加了 `stroke=black` 黑边框增强对色弱友好度；红方块未加斜纹（altair 不原生支持，性价比低，跳过） |
| 5 | 预设标签精简 | ✓ | 改为 ✋ 手动 / ✓ 完美 / 📉 k 偏小 / 📈 b 偏大 / 🚀 远距 / 🔄 反向 |

## 偏离 design / review 的决策

### 1. 预设状态下「滑块禁用态」改为「滑块隐藏 + 只读值展示」

**review 原话**（必修 2.a）：选预设 → 滑块「显式同步到预设值 + 显示禁用态（半透明 + 不可拖）」

**实际做法**：选预设 → 滑块**整体不渲染**，改用只读 mo.md 显示「k = 0.97（已锁定） · b = 1.00（已锁定）」+ 黄色徽章「🟡 预设模式 · 切回 ✋ 手动 解锁滑块」

**理由**：
- marimo `mo.ui.slider` 原生 API 没有 `disabled` 参数（实测 0.23.x 系列），无法做半透明禁用态
- 通过 mo.state 强制设值会触发任务明令禁止的「mo.state + run_button 反模式」+ 滑块和预设相互回写容易死循环
- 「隐藏 + 显示锁定值 + 文字提示如何解锁」达成同等教学目标（用户清楚看到参数值已被预设固定，不会困惑），实现简单可靠

### 2. 「拖滑块自动切回 ✋ 手动」未实现

**review 原话**（必修 2.b）：用户拖动滑块 → preset dropdown 自动切回 "✋ 手动"

**实际做法**：在预设模式下滑块根本不可见（不渲染），所以"拖滑块"动作在该模式下不存在；用户必须先在 dropdown 选 ✋ 手动 才能见到滑块。

**理由**：与决策 1 同根 —— marimo 反应式架构里让 dropdown 接收滑块事件需要 mo.state 双向同步，违反"单通道输入 + 不用 state"原则。隐藏滑块直接堵死了状态漂移的可能。

### 3. 数据中心化（x_data 改为 [-3, 3] 而非身高 [160, 180]）

**design 原话**（第 80-85 行）：`x_data = np.array([160, 166, 172, 174, 180])`、`w_true=0.97, b_true=-95`

**实际做法**：沿用范本 v2 的中心化 `np.linspace(-3, 3, 5)` + `b_true=1.0`，最优 k\*≈0.97 / b\*≈1.0 落在控件 [-2,6] / [-4,6] 范围内可见。

**理由**：原始尺度下 b_opt = -95 落在滑块 b ∈ [-4, 6] 之外，B2 抛物线根本看不到绿钻石，A 视图直线斜率被 y 轴尺度压平 — 教学冲击力归零。中心化不改变 LR 的几何意义，README 已经说"教学上等价"，对应 PPT 的概念（误差/方块/抛物线）一一保留。

### 4. 同时去掉 design 范本中的 `lr` 滑块（范本对照新增条目）

范本 v2 有 `lr` 滑块（学习率），03b 不涉及 GD 故无意义，删除。控件最终就 `preset` + `k_slider` + `b_slider` 三项，符合 review 必修 1 精神。

## 自检结果

```
$ marimo export script mse-residual-squares.py -o /tmp/_lint_03b.py
（无输出 = 成功）

$ python3 /tmp/_lint_03b.py
[W ...altair_transformer] Failed to convert data to arrow format, falling back to CSV: No module named 'pyarrow'
（仅 pyarrow 缺失警告 · CSV fallback 不影响功能 · _marimo-math-guide.md line 73 已说明可忽略）
```

## 服务启动

```
$ nohup .venv/bin/marimo run ... --port 2730 ... &
$ curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:2730
200
```

进程持续运行（pid 已 disown，log 在 /tmp/marimo-03b.log）。注意 IPv4 显式访问，`localhost` 在本机偶发解析到 IPv6 返回 000，用 `127.0.0.1` 稳。
