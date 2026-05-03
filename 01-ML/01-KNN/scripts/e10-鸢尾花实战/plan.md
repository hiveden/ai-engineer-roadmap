# E10 · 鸢尾花实战 · 计划

## 覆盖

- `04c-iris-case/01-鸢尾花案例.md`（全）
  - 数据集介绍：150 条 / 4 特征 / 3 类
  - 6 步标准 ML 流程：加载 → 拆分 → 标准化 → 训练 → 评估 → 预测
  - sklearn API 关键：`load_iris` / `train_test_split(stratify)` / `StandardScaler` / `KNeighborsClassifier` / `accuracy_score` / `predict_proba`

## Demo 片段

- `04c-iris-case/demos/01-iris-pipeline.py`

**实际 UI**（按这个写录屏稿，**已读 demo 代码**）：
- **顶部 6 步标准流程介绍**（mo.md 卡片：N=150，4 特征，3 类）
- **3 列控件**（hstack）：
  - 数据拆分：`test_size` 滑块（0.1-0.5, step=0.05, default=0.3）+ `random_state` 滑块（0-100, default=22）
  - KNN 参数：`k` 滑块（1-30, step=2, default=5）+ `StandardScaler 标准化` 开关（default=ON）
  - 2D 投影特征：`feat_x` 下拉 / `feat_y` 下拉（4 选 2，default 花瓣长 × 花瓣宽）
- **准确率对比卡**：4 维全特征 vs 2 维投影 acc + 差距高亮（>5pp 红色警告）
- **决策边界图**：60×60 网格 mark_rect 三色背景（红=setosa / 绿=versicolor / 蓝=virginica）+ 训练点（圆带黑边）+ 测试点（十字 size=160）
- **混淆矩阵**：3×3 蓝色热力图 + 数字
- **分类报告**：sklearn classification_report 文本 + precision/recall/f1/support 4 个指标说明
- **底部 accordion**：4 个折叠（演示什么 / stratify=y / 标准化对 KNN / 决策边界怎么读）

**关键档位**（demo 自带的教学结构）：

| 档位 | 现象 | 教学要点 |
|---|---|---|
| 默认 (k=5, std=ON, scoring 花瓣 × 花瓣) | 4 维 acc ≈ 95-100%, 2 维 acc 接近 | setosa 完美隔离，versicolor/virginica 微重叠 |
| feat 切到 sepal_len × sepal_wid | 2 维 acc 跌到 ~75%, 4 维不变 | 维度信息量差异 |
| 标准化 OFF | 4 维 acc 几乎不变（量纲都是 cm） | 提示 e09 标准化的核心场景在量纲悬殊时 |
| k 拖到 1 / 29 | 边界过拟合 / 欠拟合 | 回扣 e02 |

**states 操作清单**（草拟，正稿前实跑校准）：

```yaml
states:
  1. intro      — 静帧介绍 UI（6 步流程卡 + 数据集卡 + 3 列控件）
  2. baseline   — k=5, test_size=0.3, seed=22, std=ON, feat 花瓣长 × 花瓣宽
                  expect: 4 维 acc ~95%+ · 决策边界三色清晰 · 混淆矩阵接近对角
  3. feat-bad   — 切换到 sepal_len × sepal_wid（信息量差的两维）
                  expect: 2 维 acc 跌到 ~75% · 红绿蓝大量重叠 · 4 维 acc 不变（重点对比）
  4. scale-off  — 切回花瓣长 × 花瓣宽，标准化关掉
                  expect: 4 维 acc 几乎不变（iris 量纲都是 cm）— 引出 e09 钩子
  5. k-extreme  — k=1（决策面碎裂） / k=29（边界平滑）
                  expect: 视觉回扣 e02 过/欠拟合
```

## 砍掉

- `model.predict_proba` 详细输出 → 教学已在 e02/e03 累积理解，本期口播一句带过
- `classification_report` 详细字段（precision/recall/f1）展开 → 02-LR 章节会专门讲（评估指标体系），本期只点出"混淆矩阵 + 分类报告"两件套
- demo accordion 4 个折叠详解 → 视频用口播一句话提及，深读留给观众自己点开

## 口吻提示

- E10 核心命题：**前 9 期所有概念在这里第一次组装成完整流水线**——加载 / 拆分 / 标准化 / 训练 / 评估 / 预测，6 步全亮起来
- 鸢尾花是 ML hello world：1936 年 Fisher 数据集，所有 ML 教材的入门必经
- **stratify** 第一次正式介绍：分层抽样，按类别比例切分（中英双标）
- **混淆矩阵 confusion matrix** 第一次正式介绍（中英双标，但不展开 precision/recall）
- 切到 demo 后强调："4 维全特征训练 vs 仅看 2 维投影——准确率差距告诉我们多余的两维有没有信息"
- 标准化开关切换 acc 不变 → 设伏笔："iris 量纲都是 cm 看不出标准化威力，所以 e09 用了量纲悬殊的数据演示——这一期我们看到第一次完整跑通的现场"
- cta 末句："下一期我们解决一个新问题——刚才 k 我们随手设了 5，到底是不是最优？怎么不靠手感、靠数据自己挑出最优 K——交叉验证完整版来了。如果讲得有不对的地方，欢迎在评论区指正。"

## 段落骨架（7 段，~15 min）

| id | type | topic | 源 | 字数 |
|---|---|---|---|---|
| 1 | hook | 接 e09：工具齐了，本期实战 + 6 步流水线 | A | ~140 |
| 2 | content | 鸢尾花数据集介绍（150/4/3） | A | ~200 |
| 3 | content | 6 步流程 + sklearn API 一一对应（含 stratify） | A | ~280 |
| 4 | content | **Demo · baseline 跑通**（4D acc + 决策边界 + 混淆矩阵） | **B** | ~340 |
| 5 | content | **Demo · 2D 投影对比**：花瓣 vs 花萼，维度信息量 | **B** | ~280 |
| 6 | content | **Demo · 标准化开关 + k 极端**：iris 量纲对齐 + 回扣 e02 | **B** | ~280 |
| 7 | cta | 概念回顾 + 钩 e11a 交叉验证 | A | ~280 |

合计 ~1800 字 + demo 录屏 ~5 min → 估算 **~14-15 min**

## 待办

- [ ] script.json 正稿
- [ ] 实跑 demo 校准 baseline / feat-bad 准确率数字（feat-bad 实测 ~75% 是估值）
- [ ] 6 步流程动画素材（A 段核心视觉，每步亮起对应 sklearn API 名）
- [ ] 决策边界三色 + 训练圆/测试十字图例
