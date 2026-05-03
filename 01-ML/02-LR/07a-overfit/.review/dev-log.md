# Dev Log · poly-degree-overfit demo

> P3 阶段 · 实现 + 自测 + 启服务

## 文件

- 路径：`01-ML/02-LR/07a-overfit/demos/poly-degree-overfit.py`
- 行数：约 280 行（含注释/docstring/空行）
- 端口：**2733**

## review 4 项必修 · 落地核对

| # | 必修 | 实现位置 | 状态 |
|---|---|---|---|
| 1 | 右图加 test MSE 最低点绿钻石 + annotation | 右图 cell：`best_marker`（mark_point shape=diamond, color=#10b981, size=300）+ `best_text` mark_text 标注 "最优 degree=N" | done |
| 2 | preset ↔ slider 单一来源 | 4 个 `mo.ui.button`，`on_click` 调 `degree_slider.set_value(...)`；下游所有计算只读 `degree_slider.value` | done |
| 3 | degree 范围 1-15 → 1-12 | `mo.ui.slider(1, 12, ...)`；预计算循环 `np.arange(1, 13)` | done |
| 4 | 左右图 hstack 同 cell 输出 | 单独 cell：`mo.hstack([mo.ui.altair_chart(chart_left), mo.ui.altair_chart(chart_right)], ...)` | done |

## 可选建议落地

- 真实曲线 `y=0.5x²+x+2` 淡绿 dashed → 已加（左图 `line_true`）
- y 轴固定 [-6, 12] / x 轴 [-3.2, 3.2] → 已加 `alt.Scale(domain=...)`
- gap 阈值落地后微调：0.15 / 0.4（依实际 gap 分布，原 0.2/0.5 偏紧）
- MSE 显示 `.3f` → 已用
- 默认 `degree=2`（"刚好"作为打开页面第一眼的锚点）→ 已用
- `np.polyfit` 高 degree 数值稳定：用 `warnings.filterwarnings` 屏蔽 RankWarning + 拟合曲线 `np.clip(-20, 25)` 防止视觉飞出

## 数值校验（seed=666, random_state=5）

| degree | train MSE | test MSE | gap | 说明 |
|---|---|---|---|---|
| 1 | 3.147 | 2.918 | -0.229 | 欠拟合：都很大 |
| 2 | 1.150 | 0.983 | -0.166 | **最优**（test 最低，绿钻石位置） |
| 3 | 1.104 | 1.110 | +0.006 | 接近最优 |
| 5 | 1.101 | 1.099 | -0.001 | 仍接近最优（test set 巧合） |
| 8 | 1.034 | 1.392 | +0.358 | 微过拟合（黄） |
| 10 | 1.015 | 1.339 | +0.324 | 微过拟合 |
| 12 | 1.013 | 1.401 | +0.388 | 严重过拟合（接近红阈值） |

best_degree = 2，绿钻石锚点教学意图清晰。

## 自检

```
marimo export script poly-degree-overfit.py -o /tmp/_lint_07a.py  # 0 错
python /tmp/_lint_07a.py                                          # 0 错（仅 pyarrow CSV fallback 警告）
```

## 服务

```
nohup .venv/bin/marimo run ... --port 2733 ... &
curl http://127.0.0.1:2733  →  HTTP 200
```

## 备注

- `localhost` 在本机解析为 IPv6 时 curl 拿不到响应，用 `127.0.0.1` 验证 OK
- pyarrow 警告无害（DataFrame 自动 fallback 到 CSV transformer，不影响渲染）
