# E08 · MAE / MSE / RMSE + 异常点 · 计划

> **状态**：Step 2 占位骨架。Step 3 写 script.json 时按 `_3-script-guide.md` 展开本文件。

## 覆盖

- `02-LR/05-metrics/01-MAE.md` —【知道】平均绝对误差
- `02-LR/05-metrics/02-MSE.md` —【知道】均方误差
- `02-LR/05-metrics/03-RMSE.md` —【知道】均方根误差
- `02-LR/05-metrics/04-三种指标比较.md` —【了解】对比 / 选用

## 教学目标（3-5 条）

1. 知道 MAE / MSE / RMSE 三种回归评估指标的公式与单位
2. 知道 RMSE = √MSE，单位回到 y 本身（解决 MSE 平方单位的不直观）
3. 理解 MSE 对异常点**极敏感**（平方放大）；MAE **不敏感**（线性求和）
4. 知道为什么有异常点时推荐 **MAE**（loss 不被极端值绑架）
5. 知道工程实践：数据干净用 RMSE 报告 / 异常点多用 MAE / LR 内部训练一律 MSE

## 衔接钩

- **承上**（呼应 E07 cta）："上一期对比完三种 GD + 正规方程，模型训练完了——但好不好得有评估指标。这一期讲 MAE / MSE / RMSE 三件套。"
- **启下**（埋伏到 E09）："三种指标有了，但单元数据集太抽象——下期上真实案例：**加州房价完整 pipeline**（替代废弃的波士顿数据集），数据加载 / 标准化 / 训练 / 评估一站跑通。"

## 本期主线案例

**对比场景**：拟合 y = 2x + 5 + 噪声

- **场景 1（噪声小）**：MAE ≈ RMSE，两者差异小
- **场景 2（含异常点）**：RMSE ≈ 2 × MAE，差距拉开
- **MSE 几乎爆炸**——异常点贡献了大头

**简短数字例**：误差 [1, 3] → MAE = 2，RMSE = √5 ≈ 2.24（直觉锚）

## Demo 片段

- `02-LR/05-metrics/demos/metric-vs-outlier.py`（端口 2759）
- 主轴：拖动异常点 → 三个指标实时变化
- 预期：MAE 缓慢上升 / MSE 急剧爆炸 / RMSE 中间
- Step 3 写 script 前需实跑：
  - 干净数据下三个指标实测值
  - 加入 1 个 / 多个异常点的指标变化数字

## 砍掉（待 Step 3 确认）

- 评估函数 vs loss 函数概念差异 → 一句话带过（"它们公式一样，叫法不同看用途"）
- sklearn API 完整签名 `LinearRegression(fit_intercept=True)` 等 → 推迟 E09 代码段一并讲
- learning_rate='invscaling' / 'constant' 等 SGDRegressor 细节 → E09 讲
- slide 90 sklearn API 对比 → 不本期讲，留给 E09
- R²（决定系数）→ 不在教材 5 章范围，**不讲**

## 段落骨架（待 Step 3 展开）

| id | type | topic | 源 | 字数估 |
|---|---|---|---|---|
| 1 | hook | 训练完看损失值——但单位 y² 不直观 | A | ~120 |
| 2 | content | MAE：平均绝对误差，单位 = y 本身 | A | ~180 |
| 3 | content | MSE：均方误差，平方放大大误差（LR 训练 loss） | A | ~220 |
| 4 | content | RMSE：均方根，开方回到 y 单位（评估首选） | A | ~200 |
| 5 | content | Demo · 异常点拖动：三指标实时变化 | B | ~340 |
| 6 | content | 选用建议：干净 RMSE / 异常 MAE / 训练 MSE | A | ~200 |
| 7 | cta | 三件套有了——下期真实案例加州房价完整 pipeline | A | ~120 |

合计 ~1380 字 + demo 录屏 ~3 min → 估 **~12 min**

## 待办

- [ ] 实跑 metric-vs-outlier.py 校准三指标实测数字
- [ ] script.json 正稿
- [ ] 与 E09 hook 对齐"三指标 → 加州房价案例"主线
