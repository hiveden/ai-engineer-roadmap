# Verify · poly-degree-overfit demo

> P4 阶段 · 只读验收 · 4 维度核查 · 不改代码

**结论**：**PASS**（4 项必修全部落地 / 服务健康 / 代码与数据一致 / 教学锚点清晰）

---

## V1 · review 4 必修项核查

| # | 必修项 | 代码位置（行号） | 状态 |
|---|---|---|---|
| 1 | 测试 MSE 最低点绿钻石标记 + annotation | `poly-degree-overfit.py:300-325` `best_marker`（`mark_point shape="diamond" size=300 color="#10b981" filled=True`）+ `best_text`（`mark_text dy=-18`，文字 `"最优 degree={best_degree}"`） | ✅ done |
| 2 | preset 用 button 调 `set_value` 单一来源 | `poly-degree-overfit.py:74-103` 4 个 `mo.ui.button` 的 `on_click=lambda _: degree_slider.set_value(N)`；下游所有计算均只读 `degree_slider.value`（L126），无 `mo.state` / `run_button` 反模式 | ✅ done |
| 3 | degree 范围 1-12（不是 1-15） | `poly-degree-overfit.py:78` `mo.ui.slider(1, 12, ...)`；预计算 L147 `np.arange(1, 13)`；右图轴 L269-270 `domain=[0.5, 12.5]` + `values=range(1,13)` 全链一致 | ✅ done |
| 4 | 左右图 hstack 同 cell | `poly-degree-overfit.py:335-343` 单独 cell，`mo.hstack([mo.ui.altair_chart(chart_left), mo.ui.altair_chart(chart_right)], gap=1, widths=[1,1])` ，注释明确写 "review 必修 4：避免渲染撕裂" | ✅ done |

---

## V2 · 服务健康（端口 2733）

```
$ curl -s -o /dev/null -w "HTTP %{http_code}" http://127.0.0.1:2733
HTTP 200
```

✅ 服务在线，HTTP 200。

---

## V3 · 代码质量

### 数据生成（与 design §2 / dev-log 一致）

| 校验点 | 期望 | 代码（L62-71） | 状态 |
|---|---|---|---|
| 公式 | `y = 0.5x² + x + 2 + N(0,1)` | `y_all = 0.5 * X_all**2 + X_all + 2 + np.random.normal(0, 1, 100)` | ✅ |
| 样本数 | N=100 | `np.random.uniform(-3, 3, 100)` | ✅ |
| 拆分 | 70/30 | `train_test_split(..., test_size=0.3, random_state=5)` | ✅ |
| 种子 | seed=666 + random_state=5 | `np.random.seed(666)` + `random_state=5` | ✅ |

### RankWarning 处理

`poly-degree-overfit.py:37-40` 用 `warnings.filterwarnings("ignore", category=np.exceptions.RankWarning)`（带 `hasattr(np, "exceptions")` 兼容老/新 numpy 的 fallback 到 message 匹配）。
另外 L140 `np.clip(y_fitted, -20, 25)` 防止 high-degree 数值飞出图框。✅

### 预设 4 档（degree=1/2/5/12）

`poly-degree-overfit.py:87-102` 四个按钮：
- `btn_under` → 1（欠拟合）
- `btn_just` → 2（刚好）
- `btn_mild` → 5（微过拟合）
- `btn_severe` → 12（严重过拟合）

完全符合 design §3.2 + review 必修。✅

### 其他代码亮点（非阻塞，加分项）

- 计算 fan-out 单点：`degree / coeffs / train_mse / test_mse / gap` 全在 L123-141 一个 cell，下游视图只 read，无重复计算；
- 预计算所有 degree=1..12 MSE 用独立 cell（L144-162），`best_degree = argmin(test_mses)` 动态计算（不是硬编码 2）；
- 真实曲线 `y=0.5x²+x+2` 淡绿 dashed `strokeDash=[6,4] opacity=0.7`（L218-222）= "真理基线"；
- 状态徽章 3 色阈值 `0.15 / 0.4`（L168-173）按 dev-log 已校准。

---

## V4 · 教学价值

### 数值校验（独立复跑 seed=666 / random_state=5）

```
d= 1 train=3.147 test=2.918 gap=-0.229
d= 2 train=1.150 test=0.983 gap=-0.166  ← test MSE 最低 = 绿钻石位置
d= 3 train=1.104 test=1.110 gap=+0.006
d= 5 train=1.101 test=1.099 gap=-0.001
d= 8 train=1.034 test=1.392 gap=+0.358
d=10 train=1.015 test=1.339 gap=+0.324
d=12 train=1.013 test=1.401 gap=+0.388
best_degree = 2, best_test_mse = 0.9832
```

✅ 与 dev-log 表格逐行一致；✅ `best_degree=2` 与"刚好"预设 + 默认滑块值完全对齐 → 用户打开页面**第一眼**红点 = 绿钻石 = degree=2，"理想状态"作为锚点设定到位。

### U 形清晰度

- 测试 MSE：`2.918 → 0.983 → 1.110 → 1.099 → 1.392 → 1.339 → 1.401` —— 从 d=1 下到 d=2 的最低点，再缓慢上升到 d=8+ 的高位，**U 形成立**。
- 训练 MSE：`3.147 → 1.150 → ... → 1.013` —— 单调下降（带噪声小波动），符合 high-variance 模型"训练越深越好"的预期。
- 注意 d=3/5 的 test MSE 仍接近最低点（设计文档已预警"degree=5 仍接近最优"是 test set 巧合），不影响 U 形整体——用户在 d=8/10/12 能清楚看到测试线远高于训练线（gap +0.32~0.39）。✅

### 甜蜜点视觉锚定

- 绿钻石 `size=300 color=#10b981 shape=diamond`，加 `mark_text dy=-18` 显示 "最优 degree=2"
- 当前红点 `size=280 color=#ef4444`，与绿钻石形状/颜色对比鲜明
- 默认 degree=2 → 打开页面瞬间红点贴在绿钻石上 → "刚好"作为锚点 + 拖滑块离开后红点偏移即可对照 ✅
- 底部玩法 L360 已显式说明："红点和绿钻石的相对位置就是「调参方向」" → 教学语言固化 ✅

### 训练/测试 gap 数字提示

- 状态徽章（L175-185）一行显示 `degree | train MSE | test MSE | gap | 状态标签`
- gap 颜色三色梯度：`<0.15` 绿（拟合良好）/ `<0.4` 黄（微过拟合）/ `≥0.4` 红（严重过拟合）
- gap 数字 `font-size:18px` 比其他数字突出，且着色与状态条联动 → 5 秒可读 ✅

---

## 总结

| 维度 | 判定 |
|---|---|
| V1 review 4 必修项 | 4/4 PASS |
| V2 服务健康（2733） | PASS（HTTP 200） |
| V3 代码质量（数据/RankWarning/预设） | PASS |
| V4 教学价值（U 形/甜蜜点/gap 提示） | PASS |

**最终判定：PASS** —— 可以进入 P5（章节集成 / README 链接 / commit）。

无阻塞项。可选增强（非必要）：

- 当前 d=3/5 的 test MSE 与 d=2 几乎并列（test set 巧合），如果想让 U 形更"教科书"，未来可考虑换个 `random_state` 让 d=3-5 略高于 d=2 ——但这会偏离 README 底稿数值，**不建议**为了视觉牺牲与底稿的对齐性。
- 玩法文案 L353 "degree=1 → 12" 与界面一致 ✅，无需调整。
