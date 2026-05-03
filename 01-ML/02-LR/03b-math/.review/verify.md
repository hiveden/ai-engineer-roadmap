# 03b-math Verify Report

> 验收对象：`demos/mse-residual-squares.py`（444 行 / 14 cell）
> 验收时间：2026-05-02
> 验收者：P4 验收 agent（只读 + 报告）

## 总评 [PASS]

review 6 项必修全部落地（其中必修 2「拖滑块自动切回手动」用「预设态隐藏滑块」做了等效变通，dev-log 已说明且实测达成同等教学目标，验收承认）。服务健康（端口 2730 LISTEN / HTTP 200）。代码无 mo.state/run_button 反模式，cell 划分清爽，自解释标题完整。教学价值上 5 秒测试 3 个判断点全部成立，预设跨度足够拉差异。建议合入。

---

## V1 · review 必修核查 [6/6 ✓]

| # | 必修项 | 落地 | 证据（行号） |
|---|---|---|---|
| 1 | 删除 `frame` 滑块 | ✓ | 全文搜 `frame`，仅 line 99 注释「review 必修 1：删除 frame 滑块」。控件区只剩 `preset` / `k_slider` / `b_slider`（line 97-101）。✓ |
| 2 | 预设/手动状态转移明确 | ✓（变通） | line 106-140 cell 实现「if preset.value is None：渲染滑块 + 蓝徽章；else：隐藏滑块 + 黄徽章 + 只读值展示」。dev-log「偏离决策 1/2」承认未做「禁用半透明」+ 未做「拖滑块自动切回手动」，改为「预设模式直接不渲染滑块」。**验收意见**：变通方案达成教学目标——用户清楚看到 k/b 已被预设锁定（line 132-135 mo.md 显式打印「k=0.97（已锁定）」），切回手动按钮提示在徽章里（line 130「切回 ✋ 手动 解锁滑块」）。比强行 mo.state 双向同步更可靠。✓ |
| 3 | 子图标题自解释 | ✓ | A 标题 line 276：「A · 数据点 + 当前拟合（红线）+ 最优拟合（绿虚）+ 残差平方（红方块越大误差越大）」；B1 line 341：动态 f-string「B1 · 固定 b={cur_b:.2f}，loss 随 k 变化（红=当前 k，绿钻=最优 k\*={k_opt:.2f}）」；B2 line 400 同结构。陌生人 5 秒看懂无术语。✓ |
| 4 | 指标面板完整 | ✓ | line 187-200 mo.md 渲染：模式徽章（line 193）+ 当前 k/b（line 194）+ 当前 Loss 大字号 + 颜色（line 195）+ 最优 k\*/b\*/Loss\*（line 196）+ 当前/最优比值（line 197）。颜色阈值见 line 172-177（gap<0.05 绿 / <0.5 黄 / 否则红）。✓ |
| 5 | 中文字体 | ✓ | line 40-41：`plt.rcParams["font.sans-serif"]=["PingFang SC","Heiti SC","Arial Unicode MS","DejaVu Sans"]` + `axes.unicode_minus=False`。所有图表用 altair（中文友好），matplotlib 仅作字体兜底配置存在，未实际渲染中文图。✓ |
| 6 | 模式徽章有效区分 | ✓ | 控件区徽章 line 114-118（蓝 #dbeafe/#1e40af 「🔵 手动模式」）vs line 127-131（黄 #fef3c7/#92400e 「🟡 预设模式」）；指标面板内徽章 line 179-185 同配色复现。两处同步，5 秒可识别。✓ |

---

## V2 · 服务健康

| 检查项 | 结果 |
|---|---|
| 端口 2730 | **LISTEN**（pid 20832 Python，IPv4 localhost） |
| HTTP 状态 | **200** |
| 进程存活 | ✓（lsof 输出 LISTEN 状态） |

---

## V3 · 代码质量

| 维度 | 结果 |
|---|---|
| 行数 / cell 数 | 444 行 / 14 cell（与 dev-log 报「~290 行」有出入，实际更长但可接受——多出来的行主要在 A 视图 altair 多 layer 组合 line 205-278 和指标面板 HTML line 187-200） |
| 反模式扫描（mo.state / run_button） | ✓ 0 匹配。纯 reactive，滑块和 dropdown 直驱所有图表 |
| 私有变量命名 | ✓ cell 内临时变量都用 `_` 前缀（`_df`、`_xs`、`_pts`、`_squares`、`_rules`、`_line_cur`、`_line_opt`、`_ks`、`_losses`、`_df_k`、`_parabola`、`_cur_dot`、`_opt_dot`、`_k_slice_opt`、`_opt_loss_at_kopt` 等）。跨 cell 暴露的只有 `chart_data` / `chart_k_slice` / `chart_b_slice` / `cur_k` / `cur_b` / `cur_loss` / `is_preset` / `k_opt` / `b_opt` / `loss_opt` / `x_data` / `y_data` / `y_pred_cur` / `PRESETS` / `preset` / `k_slider` / `b_slider`，全部命名清晰，含义自明 |
| 数据/参数合理性 | 5 样本对齐 design + slide 24（line 69 `np.linspace(-3, 3, 5)`），中心化到 0 附近偏离原 design 但 dev-log 决策 3 已说明（原 b_opt=-95 落在滑块外不可见，中心化后 k\*≈0.97/b\*≈1.0 落在 [-2,6]/[-4,6] 滑块范围内可见）。教学等价性站得住 |
| clip 防爆 | ✓ line 164、286、351 三处 `np.clip(loss, 0, 1e8)`，极端预设（如 🚀 远距 k=-1.5/b=4.5）也不会让 altair y 轴爆 |

---

## V4 · 教学价值

design 第 104-110 行 5 秒测试 3 判断点：

| 判断点 | 视觉信号 | 验收 |
|---|---|---|
| ① 红方块的含义 | A 视图 5 个红方块 size ∝ err²（line 242-244 size scale [50,3000]）+ 红色虚线连真实点和预测点（line 251-255 mark_rule strokeDash）+ 标题明示「红方块越大误差越大」（line 276） | ✓ 陌生人 5 秒理解「方块=误差量度」无悬念 |
| ② 滑块作用 | 拖 k → A 视图红线斜率变（line 218-219 + 256-260）+ B1 红点左右移（line 304-315）+ 指标面板 Loss 数字+颜色变。手动模式徽章蓝色（line 114-118）vs 预设模式徽章黄色（line 127-131）一眼识别当前是谁主控 | ✓ 「旋钮改变直线和 loss」5 秒成立 |
| ③ 最优解位置 | A 视图绿虚线（line 261-271）+ B1/B2 绿钻石（line 318-336、377-395）同色 #10b981 同 stroke=black 黑边框；标题里「绿钻=最优 k\*」明确锚定（line 341、400） | ✓ 「绿=特别的最优」可建立连接。review 维度 1 担心的「绿虚线 vs 绿钻石不同形」靠子图标题文字补强 |

**预设跨度核查**（line 89-96 PRESETS）：
- ✓ 完美 (0.97, 1.0) → loss≈0（接近 loss\*）
- 📉 k 偏小 (0.3, 1.0) → A 红线明显平于绿虚，方块中等
- 📈 b 偏大 (0.97, 4.0) → A 红线整体上移，方块中等
- 🚀 远距 (-1.5, 4.5) → 红线反向 + 大幅偏离，方块巨大
- 🔄 反向 (-0.8, 2.0) → 反向但偏离没那么远

跨度对比明显，loss 数量级从 ≈0 到几十够撑起教学锚点。✓

**反应式流畅度**（基于代码逻辑推断）：
- 单一来源（line 144-155 cell）：preset.value None ⇒ 用滑块；非 None ⇒ 用预设值。无双向回写
- 拖滑块只触发依赖 k_slider/b_slider 的 cell 重算，120 点抛物线（line 284、349）+ 5 点散点 altair 重渲染轻量，应当流畅
- 切预设也只触发同一条链路重算，无 race

---

## 残留 issue（不阻塞，可选优化）

1. **dev-log line 3 报「~290 行」与实际 444 行有出入**：建议下次写 dev-log 时跑一下 `wc -l` 再写。不影响功能
2. **review 必修 2 的「拖滑块自动切回手动」纯字面意义未实现**：dev-log 决策 2 用「预设态隐藏滑块」堵死状态漂移路径，等效但与 review 字面要求不一致。验收承认变通——marimo 反应式架构里强行做反向同步会引入 mo.state 反模式，性价比低。后续若 marimo 升级支持 slider `disabled` 参数，可以重做成「半透明禁用」更接近 review 原意
3. **数据点保持 5 个未采纳软建议 1（加到 8-10）**：dev-log 决策保留对齐 slide 24 算例，可接受。后续如学生反馈方块阵列稀疏可微调
4. **A 视图 size 编码 range=[50, 3000]**（line 244）和 design 第 57 行一致，但极小误差（如 ✓ 完美预设）下方块下限 50 仍有可见尺寸——并非 0，理论上「完美拟合方块消失」的视觉冲击会打折。可考虑 range=[10, 3000] 让完美态方块更小
5. **B1 视图绿钻石的 y 坐标用全局 k\* 在 cur_b 切片上的 loss**（line 317 `_opt_loss_at_kopt`），不是该切片的最低点。设计上对——展示的是「全局最优 k\* 在我当前选的 b 切片上的 loss」，与 A 视图绿虚线语义统一。但新手可能困惑「为啥绿钻不在抛物线最低点」（仅当 cur_b≠b\* 时偏离）。子图标题已用「最优 k\*」明示，可接受
6. **pyarrow 缺失警告**（dev-log line 67）：CSV fallback 不影响功能，按 _marimo-math-guide.md line 73 可忽略。如需去掉警告，`uv pip install pyarrow` 即可

---

**结论：PASS · 6/6 必修达成 · 端口 2730 LISTEN / HTTP 200**
