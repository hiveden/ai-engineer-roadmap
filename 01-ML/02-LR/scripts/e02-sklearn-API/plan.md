# E02 · sklearn API 4 行跑通 · 计划

> **状态**：Step 2 占位骨架。Step 3 写 script.json 时按 `_3-script-guide.md` 展开本文件。

## 覆盖

- `02-LR/02-api/01-LinearRegression.md` —【实操】sklearn API 5 步流程

## 教学目标（3-5 条）

1. 实操 sklearn 的 `LinearRegression` 5 步流程：导入 → 准备数据 → 实例化 → fit → predict
2. 知道 X 必须是 2D（n 样本 × n 特征），y 是 1D
3. 知道 `coef_`（权重）、`intercept_`（截距）的含义和形态
4. 知道 sklearn 内部用 SVD 解正规方程（不直接求逆，更稳）

## 衔接钩

- **承上**（呼应 E01 cta）："上一期我们讲了 LR 是什么——从一元到多元。但每次手算 w、b 都太累——这一期我们用 sklearn 4 行代码搞定。"
- **启下**（埋伏到 E03）："API 把直线找到了——但它内部到底是怎么找的？什么是'好直线'？下期我们打开黑盒，从损失函数讲起。"

## 本期主线案例

**沿用 E01 身高体重数据**（保持跨期数据一致 · _0-workflow §3.5 铁律）：

```python
from sklearn.linear_model import LinearRegression

x = [[160], [166], [172], [174], [180]]   # 必须 2D
y = [56.3, 60.6, 65.1, 68.5, 75]

model = LinearRegression()
model.fit(x, y)
print('w:', model.coef_, 'b:', model.intercept_)
print('预测 176:', model.predict([[176]]))
```

预期输出：w ≈ 0.93、b ≈ -93.5、预测 ≈ 70.2 kg（与 E01 手算结果对齐）。

## Demo 片段

- `02-LR/02-api/demos/api-walkthrough.py`（端口 2756）
- 主轴：5 步流程图 + 6 个场景（一元 / 多元 / shape 错误演示 / fit_intercept / coef_ 形态等）
- Step 3 写 script 前需实跑核对 6 场景实际呈现 + 数字精度

## 砍掉（待 Step 3 确认）

- 标准化讨论（slide 16 思考题）→ 一句话提"一元、量纲单一可不做"，推迟到 E09
- "fit 比 predict 慢一个数量级"等量化对比 → 不写，避免 KNN E03/E04 同类型踩坑
- SGDRegressor 的 API 介绍 → 推迟到 E07（vs 正规方程那期）

## 段落骨架（待 Step 3 展开）

| id | type | topic | 源 | 字数估 |
|---|---|---|---|---|
| 1 | hook | 上期手算太累 → 4 行代码出结果 | A | ~120 |
| 2 | content | 5 步流程图 + 一元身高体重 demo 跑通 | B | ~280 |
| 3 | content | coef_ / intercept_ 解读 + 与 E01 手算对齐 | B | ~220 |
| 4 | content | 多元扩展（demo 第 6 场景）+ X 2D 形态强调 | B | ~260 |
| 5 | content | sklearn 内部 = SVD + 正规方程（埋钩） | A | ~160 |
| 6 | cta | 4 行很爽——但 fit 内部在做什么？下期"打开黑盒" | A | ~140 |

合计 ~1180 字 + demo 录屏 ~3 min → 估 **~12 min**

## 待办

- [ ] 实跑 api-walkthrough.py 6 场景，记录每场景 expect 数字
- [ ] script.json 正稿
- [ ] 与 E01 手算 (k≈0.93, b≈-93.5, 预测 70.2) 对齐——E02 demo 跑出实际值后改写
