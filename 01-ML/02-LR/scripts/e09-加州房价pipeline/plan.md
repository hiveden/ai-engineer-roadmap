# E09 · 加州房价完整 pipeline · 计划

> **状态**：Step 2 占位骨架。Step 3 写 script.json 时按 `_3-script-guide.md` 展开本文件。

## 覆盖

- `02-LR/06-boston/01-API复习.md` —【知道】LinearRegression / SGDRegressor 再过一遍
- `02-LR/06-boston/02-案例背景.md` —【知道】数据集背景（Boston 已废弃 → California）
- `02-LR/06-boston/03-案例分析.md` —【实操】数据分割 + 标准化 + 回归预测 + 评估 4 步
- `02-LR/06-boston/04-性能评估.md` —【实操】MSE / MAE / RMSE / R²
- `02-LR/06-boston/05-代码实现.md` —【实操】完整 6 步 pipeline 代码

## 教学目标（3-5 条）

1. 实操完整 sklearn pipeline：fetch_california_housing → train_test_split → StandardScaler → fit → predict → 评估
2. 知道为什么 Boston 数据集被废弃（伦理问题，含种族变量），加州房价是替代
3. 实操 LinearRegression（正规方程）+ SGDRegressor（梯度下降）两种解法对比
4. 知道结果几乎一致——验证凸优化问题两种解法都收敛到同一最优解
5. 知道实战中**标准化必须做**（多元量纲悬殊会让 GD 震荡 / 让 LR 系数不可解读）

## 衔接钩

- **承上**（呼应 E08 cta）："上一期讲完 MAE / MSE / RMSE 三件套——这期上真实案例：**加州房价 8 特征完整 pipeline**。"
- **启下**（埋伏到 E10）："案例跑完看似完美——但等等，这里出现一个反直觉现象：**训练误差越小，测试误差反而越大**。下期我们看 LR 的另一面：欠拟合 / 过拟合 / U 形曲线。"

## 本期主线案例

**加州房价 California Housing**：8 特征 + 1 标签，20640 样本

- 8 特征：MedInc / HouseAge / AveRooms / AveBedrms / Population / AveOccup / Latitude / Longitude
- 标签：房价中位数（10 万美元单位）

**完整 6 步 pipeline**（教材代码）：

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 1 加载  2 拆分  3 标准化  4 训练（两种解法）  5 评估  6 预测新样本
```

**预期输出**（参考值）：

```
正规方程: MSE=0.555  MAE=0.533  RMSE=0.745  R²=0.596
SGD:      MSE=0.557  MAE=0.534  RMSE=0.746  R²=0.594
```

## Demo 片段

- **无独立 marimo demo**（沿用 KNN E04/E05 模式：jupyter 录屏）
- 主轴：jupyter notebook 6 步代码逐 cell 跑通 + print 输出贴合讲解节奏
- Step 3 写 script 前需实跑校准：
  - 8 特征实际名字（教材给的可能与 sklearn 当前版本略有差异）
  - 两种解法的实测 MSE / MAE / RMSE / R² 数字（_0-workflow §3.5 铁律——凭印象写的数字必须实测对齐）
  - 第 6 步预测新样本的输出
- **可选**：是否补 marimo demo 由用户决定（README 已标灰色地带 2）

## 砍掉（待 Step 3 确认）

- Boston 13 特征字段表（CRIM / ZN / INDUS / ...）→ 一句话"原数据集已废弃，看 California 8 特征即可"，不逐项念
- slide 95-96 PPT 原版 load_boston 代码 → 直接用 California 替代版（教材已替换）
- slide 98 自检题（B - 均方误差）→ 不进视频
- "为啥正规方程预测误差小"思考题（slide 95 注释）→ 一句话提"两种解法结果几乎一致"，不展开 SVD vs SGD 收敛细节
- R² 公式细节 → 一句话"决定系数，越接近 1 越好"，公式留 latex 字幕不口播

## 段落骨架（待 Step 3 展开）

| id | type | topic | 源 | 字数估 |
|---|---|---|---|---|
| 1 | hook | E08 三件套有了——上真实案例：加州房价 | A | ~140 |
| 2 | content | 数据背景：Boston 废弃 → California 8 特征 | A | ~200 |
| 3 | content | Step 1-2：fetch + train_test_split | B | ~220 |
| 4 | content | Step 3：StandardScaler 为什么必做 | B | ~280 |
| 5 | content | Step 4：两种解法 LinearRegression / SGDRegressor | B | ~280 |
| 6 | content | Step 5：四指标评估（MSE / MAE / RMSE / R²） | B | ~260 |
| 7 | content | 两种解法结果几乎一致——凸优化的体现 | A | ~180 |
| 8 | content | Step 6：预测新样本（8 维输入 → 房价输出） | B | ~200 |
| 9 | cta | 看似完美——但训练越准测试越差是什么情况？下期 U 形曲线 | A | ~180 |

合计 ~1940 字 + jupyter 录屏 ~5 min → 估 **~18 min**

## 待办

- [ ] 实跑教材完整 6 步代码，校准 4 指标实测数字
- [ ] script.json 正稿
- [ ] 决定 demo 模式：jupyter 录屏 vs 补 marimo（用户审）
- [ ] 与 E10 hook 对齐"看似完美 → 过拟合反直觉"主线
