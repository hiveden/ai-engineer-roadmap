# E11 · L1 / L2 正则化（Lasso / Ridge）· 计划

> **状态**：Step 2 占位骨架。Step 3 写 script.json 时按 `_3-script-guide.md` 展开本文件。
>
> **02-LR 章末期**——cta 承接到 03-LogReg 章节首期。

## 覆盖

- `02-LR/07b-regularization/01-L1正则化.md` — Lasso（特征选择）
- `02-LR/07b-regularization/02-L2正则化.md` — Ridge（权重收缩）
- `02-LR/07b-regularization/03-正则化案例.md` —【实操】LR / Lasso / Ridge 在多项式过拟合数据上对比

## 教学目标（3-5 条）

1. 理解正则化核心思想：在损失函数后加惩罚项让权重不要太大 → 模型简化
2. 理解 L1（Lasso）= 权重绝对值之和 → 等量收缩 → **稀疏解**（特征自动剔除）
3. 理解 L2（Ridge）= 权重平方和 → 按比例收缩 → **整体平滑**（权重小但不为 0）
4. 知道几何直觉：L1 等高线 = 菱形（顶点在轴上 → 切点稀疏）/ L2 等高线 = 圆（切点不在轴上）
5. 知道 sklearn API：`Lasso(alpha=...)` / `Ridge(alpha=...)`，alpha 越大正则越强；工程默认 L2

## 衔接钩

- **承上**（呼应 E10 cta）："上一期看完过拟合 4 类解决办法——前 3 类是事后补救，最工程的招是**正则化**：从损失函数本身下手。这一期 L1 / L2 双拳。"
- **启下**（埋伏到 03-LogReg 章节首期）："02-LR 章 11 期到此完结——LR 是回归任务的基线武器。但现实中很多问题是**离散类别**（垃圾邮件 / 肿瘤良恶 / 客户流失）——下一章我们把 LR 加 sigmoid 推广到分类问题：**逻辑回归**（Logistic Regression，LogReg）。"

## 本期主线案例

**沿用 E10 抛物线 + 噪声合成数据**（保持跨期一致 · _0-workflow §3.5 铁律）：

```python
np.random.seed(666)
x = np.random.uniform(-3, 3, size=100).reshape(-1, 1)
y = 0.5 * x.ravel()**2 + x.ravel() + 2 + np.random.normal(0, 1, 100)
```

**三方案对比**（10 次多项式 + 标准化）：

| 方案 | 权重特征 | 测试 MSE | 状态 |
|---|---|---|---|
| **纯 LR** | 高次项权重大、抖动 | ~1.5+ | 过拟合 |
| **Lasso α=0.1** | 大部分高次项 = 0（稀疏） | ~1.0 | 自动选特征 |
| **Ridge α=1.0** | 所有权重压小但 ≠ 0 | ~1.0 | 整体收缩 |

## Demo 片段

- `02-LR/07b-regularization/demos/l1-l2-shrinkage.py`（端口 2761）
- 主轴：拖动 alpha → L1 / L2 权重柱状图实时变化（L1 高次项归零 vs L2 压小）
- 可能含几何直觉子图：菱形 vs 圆形等高线
- Step 3 写 script 前需实跑：
  - alpha 各档实测权重值
  - L1 第一个变零的 alpha 阈值
  - L2 各档权重压缩比

## 砍掉（待 Step 3 确认）

- L1 在 w=0 处导数不存在（sign 函数细节）→ 一句话"用次梯度求解"，不展开
- 完整 sign 函数定义 → 直觉"正负号"，不写公式
- ElasticNet（L1 + L2 混合）→ 一句话"还有 ElasticNet 综合两者"，不本期讲
- LassoCV / RidgeCV 自动调 alpha → 一句话"工程实践用 CV 自动选"，KNN E11a 已讲过 CV
- alpha 与 lambda 命名差异（教材两边混用）→ 统一用 alpha（sklearn 命名）
- normalize=True 已弃用警告 → 一句话提"sklearn 1.0+ 改用 Pipeline + StandardScaler"
- slide 118-120 自检题 → 不进视频

## 段落骨架（待 Step 3 展开）

| id | type | topic | 源 | 字数估 |
|---|---|---|---|---|
| 1 | hook | E10 过拟合最工程的招——正则化登场 | A | ~140 |
| 2 | content | 核心思想：损失 + λ·惩罚项 | A | ~200 |
| 3 | content | L1 = |w| 之和 → 等量收缩 → 稀疏 | A | ~240 |
| 4 | content | L2 = w² 之和 → 按比例收缩 → 平滑 | A | ~240 |
| 5 | content | 几何直觉：菱形 vs 圆形等高线 | A | ~200 |
| 6 | content | Demo · 拖 alpha：L1 vs L2 权重变化 | B | ~360 |
| 7 | content | sklearn API：Lasso / Ridge 工程默认 L2 | A | ~200 |
| 8 | cta | 02-LR 章完结 → 下章 LR + sigmoid = 逻辑回归 | A | ~180 |

合计 ~1760 字 + demo 录屏 ~3-4 min → 估 **~15 min**

## 待办

- [ ] 实跑 l1-l2-shrinkage.py 校准 alpha 各档实测权重
- [ ] script.json 正稿
- [ ] 与 03-LogReg/scripts/e01 hook 对齐"LR + sigmoid → LogReg"主线（Step 2 拆 03-LogReg 时再回填）
