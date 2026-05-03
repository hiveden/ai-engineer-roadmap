# LogReg 视频脚本

> AI 老师讲解 + 后续补 demo 录屏拼接，7 期覆盖整个 LogReg 章。
>
> **写新期前必读**：[`../../_0-workflow.md`](../../_0-workflow.md) + [`../../_3-script-guide.md`](../../_3-script-guide.md) + [`../../_4-recording-guide.md`](../../_4-recording-guide.md) + KNN scripts e01-e11 正稿样本。
>
> 立场：AI 老师做分享，第一人称用"我们"，每期结尾"如果有不对欢迎指正"。

## 拆分依据复盘

03-LogReg **没有 demos/ 目录**，§2 第 2 条（demo 覆盖）暂不适用，按 **教材边界 + 时长 + 概念递进 + 衔接钩** 拍板。整体 7 期，期号从 e01 起（每个算法独立期号空间，与 KNN 一致）：

1. **01-foundation 5 个子文件不硬塞 1 期**——按概念耦合度拆 2 期：
   - e01「应用场景 + sigmoid」是 LR 主体引子，强耦合；sigmoid 篇幅适中（118 行），合一期 ~12 min
   - e02「概率 + MLE + 对数函数」三者强耦合（MLE 推导直接用条件概率，取对数是 MLE 数值化必经一步）→ 一期讲清"为什么 LR 损失要长成 logloss 那样"的根
2. **02-原理 独立 e03**：假设函数 + 决策边界 + 对数似然损失推导，是算法核心，不与 API 合并以免冲淡数学链
3. **03-api 独立 e04**：sklearn 第一次 LR 落地，API 介绍 + 癌症案例代码一气呵成（~18 min），不与原理合是因为 e03 已重
4. **04a + 04b 合 e05**：04a 单概念 135 行偏短，与 PRF 强耦合（TP/FP/TN/FN 是 P/R/F1 的拆解），合一期讲透"准确率不够 → 引入 4 格 → 衍生 P/R/F1"完整链
5. **04c 独立 e06**：ROC/AUC 概念重（315 行），含 2 个 API（classification_report、roc_auc_score），独立成期足量
6. **05-客户流失 独立 e07**：综合实战（296 行 EDA + 编码 + 训练 + 评估），独立收尾

后续如补 demos/ 再视情况合并/微调。

## 7 期索引

| 期 | 章节 | 覆盖 md | Demo | 时长 | 核心概念 | 状态 |
|---|---|---|---|---|---|---|
| **e01-引入与sigmoid** | 01-foundation | 01-应用场景 / 02-sigmoid函数 | 待补 | ~12 min | LR 是分类不是回归 / sigmoid 压区间 | 占位 |
| **e02-概率MLE对数** | 01-foundation | 03-概率 / 04-极大似然估计 / 05-对数函数 | 待补 | ~18 min | MLE 选参数 / 取对数化连乘为求和 | 占位 |
| **e03-原理与损失** | 02-原理 | 01-原理 / 02-损失函数（实质内容在 README） | 待补 | ~20 min | 假设函数 + 决策边界 / 对数似然损失推导 | 占位 |
| **e04-API与癌症案例** | 03-api | 01-API介绍 / 02-癌症分类案例（实质内容在 README） | 待补（癌症 pipeline） | ~18 min | LogisticRegression API 关键参数 / 二分类完整流程 | 占位 |
| **e05-混淆矩阵与PRF** | 04a + 04b | 01-混淆矩阵 / 01-precision / 02-recall / 03-f1-score（实质内容在 README） | 待补 | ~20 min | TP/FP/TN/FN / Precision-Recall 取舍 / F1 调和 | 占位 |
| **e06-ROC-AUC** | 04c | 01-roc / 02-绘制 / 03-auc / 04-分类报告api / 05-auc计算api（实质内容在 README） | 待补 | ~22 min | TPR/FPR 阈值扫描 / AUC 几何与概率两种解读 | 占位 |
| **e07-电信客户流失** | 05-客户流失 | 01-数据集 / 02-处理流程 / 03-案例实现（实质内容在 README） | 待补（电信 pipeline） | ~22 min | 类别不平衡处理 / 完整工业流（EDA → 编码 → 训练 → 评估） | 占位 |

合计 ~132 min，平均 ~19 min/期，全部落在 10-25 min 完播率甜点区间。

## 衔接钩链路

- e01 cta: "sigmoid 把 z 压成概率，但 z 怎么算的 w 哪来？要看 LR 怎么从数据里学这套权重，得先补一个数学工具：极大似然估计" → e02 hook 回应
- e02 cta: "我们手上有 sigmoid 假设、有 MLE 思想、有取对数的工具——下期把它们拼成 LR 完整推导，看损失函数怎么长出来" → e03 hook
- e03 cta: "数学讲完了，下期上手 sklearn，用 30 行代码做一个癌症分类" → e04 hook
- e04 cta: "癌症案例准确率 0.97 看着很美，但医疗场景要警惕——漏诊一个病人和误报一个健康人代价完全不同。下期看为什么准确率不够" → e05 hook
- e05 cta: "P/R/F1 都依赖一个固定阈值 0.5——但模型输出的是连续概率，阈值能不能扫一遍看整体？下期 ROC 曲线" → e06 hook
- e06 cta: "评估工具齐了，下期把 LR 全套用到一个真实业务场景：电信客户流失预测" → e07 hook
- e07 cta: "LR 整章收官——下一个算法 ..."（决策树？）

## 目录约定

每期一个子目录：

```
eXX-名称/
├── plan.md       ← Step 2 占位（教材范围 / 核心问题 / demo 钩子 / 衔接）
├── script.json   ← Step 3 主产物
└── recording/    ← Step 6/8 录屏产物
```

## 立场要点

- **第一人称：我们**（"我们来看"/"我们注意"）
- **AI 老师**做主讲，但不端权威——结尾留"如有不对欢迎指正"
- **教学语气**可断言，但不强势；用"想象"/"假设"/"注意"做引导
- **不引官方文档**——直接讲概念
- **术语首次出现中英双标**：逻辑回归（Logistic Regression，LR）、混淆矩阵（Confusion Matrix）、ROC 曲线（Receiver Operating Characteristic）、AUC（Area Under Curve）

## 待补 demo 提示

03-LogReg 当前缺 demos/，建议后续按期补：

- e01: sigmoid 函数交互图（拖 z 看输出 / 调温度系数）
- e02: MLE 抛硬币动画（拖 θ 看似然函数曲线 + 取对数前后对比）
- e03: LR 决策边界 2D 可视化 + 损失曲面
- e04: 癌症数据集 marimo pipeline（来自 README 代码）
- e05: 混淆矩阵交互式（拖阈值看 TP/FP/TN/FN 实时变）
- e06: ROC 曲线扫阈值动画 + AUC 阴影
- e07: 电信流失 marimo pipeline（来自 README 代码）

补齐后视必要性合并/拆分对应期。
