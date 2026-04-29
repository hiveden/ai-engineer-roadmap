# LESSON-PLAN · 第 9 节 · 时序预测（Time Series Forecasting）

> **占位文档**（2026-04-24 建）：本节尚未正式开讲。首次开讲时按模板 v0.2 骨架（[`../_shared/LESSON-PLAN-TEMPLATE.md`](../_shared/LESSON-PLAN-TEMPLATE.md)）补全 §0~§9，并合并 [`./10-time-series-digest.md`](./10-time-series-digest.md) 的三源素材。
>
> **本文档当前唯一作用**：承接 [`../00-mental-model/01-LESSON-PLAN.md` §8.4](../00-mental-model/01-LESSON-PLAN.md) 的大纲接收指派，确保首次开讲时不遗漏。

> 📎 **大纲接收**（2026-04-24 决策）：本节**无特定 §6 / §7 接收指派**。但有一个 **IID 反例锚点** 值得在首次开讲时唤起——大纲 §3 与 DISTILL §9 都强调"训练/测试独立同分布"，而时序数据天然违反 IID（未来样本不能漏进训练集），这是典型**特征泄露 Data Leakage** 事故源头。可借此顺带重锚 IID 原则。
>
> 详见 [`../00-mental-model/01-LESSON-PLAN.md` §8.4](../00-mental-model/01-LESSON-PLAN.md)。

---

## TODO · 首次开讲时

- [ ] 按模板 v0.2 骨架填 §0~§9
- [ ] 合并 [`10-time-series-digest.md`](./10-time-series-digest.md) 三源素材
- [ ] 重锚 IID 反例：时序数据的切分方式（时间切分，不是随机切分）与常规 ML 不同
- [ ] 深度配置：LESSON-PLAN 01 §3.1 表里标注为 **L1**（离核心方向远）——按 L1 扫即可
