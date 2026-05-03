# 07b-regularization Demo 开发日志

> 文件：`demos/l1-l2-shrinkage.py` · 端口 2734 · 行数 419（去注释/空行 ~330）

---

## review 必修 3 项 · 逐条落实

### ✅ 必修 1 · B 视图零柱视觉信号

**原问题**：红色虚线空心框（`#dc2626`）与无正则线红（`#ef4444`）撞色，且 `fillOpacity=0` 在白底偏弱。

**实现**（cell `chart_B`，`zero_marks` 层）：
- 改用「黄色实心填充 + 黑边」：`fill="#fde047"`（鲜黄）`fillOpacity=0.9` + `stroke="#000000"` `strokeWidth=1.8`
- 零柱位置画一个跨 0 轴的小矩形（高度 ±6% 全局 `_w_max`），在视觉上呈现「被划掉的格子」
- 矩形上方叠加 `mark_text(text="0", fontWeight="bold", color="#000000")`
- 完全脱离红色家族，与 L1 主色橙 / 无正则红都不冲突

### ✅ 必修 2 · preset / slider 共存徽章语义

**原问题**：用户选预设后再拖滑块，`cur_lambda` 仍取 preset.value，滑块「看似无效」。

**实现**（cell 模式徽章）：
- `is_preset=True` 时：黄色徽章 `🟡 预设模式 · {preset_name} · λ=...`，**附加灰色小字提示** `（手动滑块已锁定，选 ✋ 手动 解锁）`
- `is_preset=False` 时：蓝色徽章 `🔵 手动模式 · log₁₀(λ)=... · λ=...`
- 徽章右侧再加一个迷你 chip `当前聚焦：L1/L2`（响应 toggle）
- 配色取自 04b-gd 范本（蓝 `#dbeafe/#1e40af`，黄 `#fef3c7/#92400e`），跨 demo 心智一致

### ✅ 必修 3 · B 视图标准化空间标注

**原问题**：标题「权重对比」让学生误以为是原始 x^k 系数。

**实现**（cell `chart_B` 顶部 `TitleParams`）：
- 主标题：`B · 权重柱图 · L1「黄底黑边」= 被砍掉 (zero)`
- **副标题（紫色 `#9333ea` 警示色）**：`⚠ 标准化空间下系数 (PolynomialFeatures+StandardScaler 后) · 仅供稀疏性对比，非原始 x^k 系数`
- 紫色而非灰色是为了让该警示**在第二眼**就被看到（不淹没在常规副标题灰里）

---

## 可选建议采纳情况

| 建议 | 状态 | 说明 |
|---|---|---|
| 1 · C 视图 L1+L2 同框对比 | ✅ 采纳 | 选中类型实线粗 (sw=2.8)，未选中虚线细淡 (sw=1.5, opacity=0.45)，便于对比甜蜜点位置 |
| 2 · log 滑块标签强化 | 部分采纳 | label 改为 `正则强度 log₁₀(λ)`；徽章常驻显示 `log₁₀(λ)=X.XX · λ=Y` 双形式 |
| 3 · toggle 联动 A/B 高亮 | ✅ 采纳 | A 图选中模型 `opacity=1.0/strokeWidth=3.0`，未选中 `opacity=0.35/strokeWidth=1.8`；B 图同理（无正则始终满）|
| 4 · 弱档 alpha 调到 1e-2 | ✅ 采纳 | 预设「🌱 弱」从 log₁₀(λ)=-3 (1e-3) 改为 -2 (1e-2)，与「无正则」拉开 100×，与「适中」差 10×，4 档分布均匀 |
| 5 · 与 07a pipeline 一致性 | N/A | 07a-overfit/demos/ 当前为空，无需对齐；本 demo 用 `PolynomialFeatures(include_bias=False) + StandardScaler` 作为锚点 |

---

## 数据 / pipeline 关键点

- 数据：`np.random.seed(666)`，`N=100`，`x ~ U(-3,3)`，`y = 0.5x² + x + 2 + N(0,1)`，70/30 split (`random_state=5`)
- pipeline：`PolynomialFeatures(degree=10, include_bias=False) → StandardScaler → {LinearRegression | Lasso | Ridge}`
- Lasso：`max_iter=50000, tol=1e-4`，`ConvergenceWarning` 已 `warnings.filterwarnings` 屏蔽
- 零阈值：`ZERO_TOL=1e-6`
- C 图预计算：30 λ × 2 正则 = 60 fit，启动一次性跑完（约 3-5 秒），后续仅当前 λ 重 fit

---

## 自检 / 启动

```bash
# 自检
cd 01-ML/02-LR/07b-regularization/demos
.venv/bin/marimo export script l1-l2-shrinkage.py -o /tmp/_lint_07b.py  # ✅ 无错
.venv/bin/python3 /tmp/_lint_07b.py  # ✅ EXIT=0 (仅 pyarrow fallback 警告，不影响功能)

# 启动
nohup .venv/bin/marimo run 01-ML/02-LR/07b-regularization/demos/l1-l2-shrinkage.py \
  --headless --no-token --port 2734 > /tmp/marimo-07b.log 2>&1 &
# curl http://127.0.0.1:2734 → 200 OK
```

---

## 已知小坑（开发中踩过的）

1. **`color_scale` 命名冲突**：A 图与 B 图 cell 都用 `color_scale` 当 alt.Scale 变量名，marimo reactive 不允许跨 cell 同名 → B 图改 `_color_scale_b`（cell 私有）
2. **`_rows` 不可作为 return 值**：`_` 前缀变量是 cell 私有，不能 `return (_rows,)` 给下游 → 直接在同 cell 内构造 `df_curve = pd.DataFrame(_build_curve())`
3. **`localhost` curl 失败 / `127.0.0.1` 成功**：marimo headless 默认只监听 IPv4，macOS curl 默认偏好 IPv6 (::1) → 服务实际正常，校验需用 `127.0.0.1`

---

## 视图分工 · 教学叙事映射

| 视图 | 抽象层 | 5 秒任务 |
|---|---|---|
| A · 散点+三线 | 业务输出 | 看「无正则疯狂抖 → L1/L2 贴真值」 |
| B · 权重柱（主舞台）| 参数空间 | 看「L1 黄底黑边零柱 vs L2 整体矮化」 |
| C · λ-MSE U 形 | 超参扫描 | 看「test MSE U 形 + L1 vs L2 谷底位置不同」 |

对应 PPT slide 109-120 全程命中。
