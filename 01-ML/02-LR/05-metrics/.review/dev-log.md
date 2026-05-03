# Dev Log · metric-vs-outlier.py

> P3 实现日志 · review 总评 FAIL → 按 review 必修项重新实现 · 不照原 design 做

## review 3 必修项 → 实现映射

### 必修 1：三指标合并到同一柱图共享线性 y 轴

**design 原方案**（FAIL）：散点 + MAE 柱 + MSE 柱 + RMSE 柱 四列横排，每柱独立 y 轴 auto-scale → 倍数差被坐标轴消化，5 秒测试归零。

**本实现**：
- 第二行单张合并柱图 `chart_bars`（占满 760px 宽 × 300px 高）
- 横轴 = `指标:N`（MAE / MSE / RMSE 三个 category），`xOffset` 在每个 category 内并列 baseline 灰柱 + current 蓝柱
- y 轴 = `值:Q`（线性，共享）→ MSE 一柱顶天，MAE/RMSE 几乎贴底
- 倍数标签 `mark_text` 红色 `#dc2626` 加粗 18px 贴 current 柱顶（不藏在数字面板里）
- 布局两行：第一行散点 + 拟合线（占满宽度），第二行合并柱图（占满宽度），横向拉满让柱高差最大化

**位置**：`chart_bars` cell（合并柱图），`mo.vstack` 布局 cell（两行结构）。

### 必修 2：单一来源 + 模式徽章 + 删双异常点预设

**design 原方案**（FAIL）：preset / x_outlier / y_outlier 同时存在，主从不清。还有「双异常」预设破坏单点隐喻。

**本实现**：
- `PRESETS` dict：`"✋ 手动"` 映射 `None`，其余 4 个预设直接给 `(x_out, y_out)` 元组
- 解析 cell 单一来源原则：
  ```python
  if preset.value is None:
      x_out, y_out = x_out_slider.value, y_out_slider.value
      is_preset = False
  else:
      x_out, y_out = preset.value   # 预设直接出值，slider 此刻被忽略
      is_preset = True
  ```
- 下游所有计算（拟合、指标、绘图）只读 `x_out / y_out`，不读 slider/preset 本身
- **模式徽章**：`mo.md` 嵌 `<span>`，蓝色 `#dbeafe` = 手动 / 黄色 `#fef3c7` = 预设，并显示当前预设名
- **删除「双异常点」预设**：v1 不上，已挪到 README 对应章节的扩展位

### 必修 3：预设倍数指数递进（3x / 8x / 25x，不是线性微调）

**design 原方案**（FAIL）：轻微 y=14（贴拟合线）/ 中等 y=22 / 极端 y=28 → 倍数 ~2x / 5x / 20x，前两档差异不够。

**本实现**（拟合线 y=2x+5，x=5 处 y≈15）：

| 预设 | (x_out, y_out) | 偏离 | 预期 MSE 倍数 |
|---|---|---|---|
| ○ 无异常 | (5.0, 15.0) | ≈0（贴拟合线） | ~1x baseline |
| ▲ 轻微 | (5.0, 21.0) | +6 | ~3x |
| ▲▲ 中等 | (5.0, 26.0) | +11 | ~8x |
| 💥 极端 | (5.0, 35.0) | +20 | ~25x |

每档跳一个数量级，对比戏剧性拉满。15 个正常点 baseline MSE ≈ 1，含异常时 N=16，单点平方误差 e²/16 直接堆到 current MSE 上：e=20 → 400/16 = 25 → 25x ✓。

## 可选增强（已落地）

### 散点图叠平方残差方块 `mark_square`

按手册 §复用片段 1 实现：
- `df_all` 包含每个点的 `err_sq = (y_true - y_pred)²`
- `mark_square(opacity=0.30, color='#ef4444')` + `size = alt.Size('err_sq:Q', scale=alt.Scale(range=[20, 6000]))`
- 异常点处冒巨大红方块（极端预设下 size 接近 6000 上限），正常点方块小到几乎看不见 → **直接图解平方放大效应**
- 加 `mark_rule` 红色虚线（`strokeDash=[3,2]`）连接真实点 ↔ 拟合线对应预测点，让「误差长度」也可见

### 异常点标记

不用普通 mark_point，用自定义五角星 SVG path（`M0,-1L...Z`）+ 红色填充 + 白边 size=600，区别于普通蓝圆，符合 design「红色五角星」核心隐喻。

### 信息面板 `mo.callout`

按 `mag_mse` 自适应 kind：
- `> 5` → `warn`：直接断言「MSE 已被绑架，改用 MAE」
- `2~5` → `info`：「明显放大但可接受」
- `< 2` → `success`：「不严重，试试极端预设」

## 实现期 bug 修复

- design 第 95-98 行 `np.vstack([x_base, [x_outlier]])` 维度错误（x_base 1D + 标量 vstack 出 (2, N)，而 `LinearRegression.fit` 要 (N+1, 1)）
- 已改为 `np.append(x_base, x_out).reshape(-1, 1)` + `np.append(y_base, y_out)`，形状正确

## 字符串约定

- 所有面向用户的字符串用「」中文角引号或单引号，避免与 docstring `"""` 双引号冲突（如「无异常」「轻微」「中等」「极端」）
- 中文字体配置：`PingFang SC / Heiti SC / Arial Unicode MS / DejaVu Sans` 链式 fallback

## 自测

```bash
.venv/bin/marimo export script 01-ML/02-LR/05-metrics/demos/metric-vs-outlier.py -o /tmp/_lint_05.py
# → 无输出 = 导出成功

.venv/bin/python /tmp/_lint_05.py
# → 仅 6 条 pyarrow CSV fallback 警告（marimo guide 已注明可忽略）
# → 无 Python 错误，所有 cell 静态执行通过
```

## 启动

```bash
nohup .venv/bin/marimo run 01-ML/02-LR/05-metrics/demos/metric-vs-outlier.py \
  --headless --no-token --port 2732 > /tmp/marimo-05.log 2>&1 &
disown
```

- 端口 2732 LISTEN 确认（lsof）
- `curl http://127.0.0.1:2732` → HTTP 200
- 进程持久（disown 已脱离 shell）

## 行数

约 280 行（含 docstring / 注释 / 多 cell 结构），单文件可独立运行。
