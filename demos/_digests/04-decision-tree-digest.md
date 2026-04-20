# L1 备课摘要 · 决策树（Decision Tree）

> 这是 demo-04-decision-tree 的 L1 素材包，不是最终 LESSON-PLAN。
> 未来制作 LESSON-PLAN 时从此文件起手，按模板 v0.2 三源定位表填槽。
> 数据源：原始培训笔记 `assets/source-materials/第4阶段-机器学习/决策树.md` + 第一版改编 `demos/05-Decision-Trees-and-Rule-Extraction.md` + 联网搜索交叉验证。

---

## 1. 候选数据集清单（对应模板 §0.3）

| 候选 | 规模（行×列） | 特点 | 教学价值（能讲什么钩子） | 适合上课场景 | 数据源 URL |
|---|---|---|---|---|---|
| **Titanic 泰坦尼克号** | 891 × 12 | 二分类 survive；混合数值+类别（性别/舱位）；含缺失值 | 原始笔记的"官方案例"；能讲**缺失值处理 + 类别编码 + 可视化 if-else 规则**；输出每条规则对应监管叙事 | 想要对齐原始笔记 / 讲可解释性 | [Kaggle Titanic](https://www.kaggle.com/competitions/titanic/data) |
| **Breast Cancer Wisconsin** | 569 × 30 | 二分类 benign/malignant；全数值 | 已在 demo-02 用过。**强建议不复用**，除非要做"同一数据 KNN/LR/DT 横向对比" | 仅对比多模型时 | [sklearn load_breast_cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) |
| **贷款申请 15 条表格**（原始笔记作业 3） | 15 × 4 | 多分类 approve/reject；纯类别特征（年龄/有工作/有房/信贷）；小到能手算 | **手推 ID3 信息熵 + 信息增益**的唯一合适样本；可在白板上完整走一遍 | 想手推数学 / 带学员走一次熵计算 | 原始笔记 `决策树.md` 作业 3 |
| **California Housing** | 20640 × 8 | 回归任务；房价中位数 | **CART 回归树**教学锚；能复现"线性回归是直线，回归树是阶梯"的戏剧对比 | 讲 CART 既分类又回归时 | [sklearn fetch_california_housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) |
| **UCI Adult / Census Income** | 48842 × 14 | 二分类 income >50K；含敏感特征（种族、性别）；混合类型 | **反面教材钩子**——演示单棵树如何在敏感特征上 leak bias，对接 Apple Card / COMPAS 讨论 | 想讲公平性 + 可解释性双刃剑 | [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult) |
| **sklearn load_iris** | 150 × 4 | 3 分类；全数值 | **决策边界可视化**最干净的数据集，能画 2D 切分图；树深 3 就 95%+ | 零基础第一次看树结构 | [sklearn load_iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) |

**现场选型规则**：
- 想对齐原始笔记正统路线 → **Titanic**
- 想手推数学（信息熵/基尼） → **贷款 15 条**
- 讲 CART 回归树 → **California Housing**
- 讲业界翻车 / 公平性 → **Adult**
- 想画图讲决策边界 → **Iris**

**默认推荐**：第一轮 **Titanic**（剧情好 + 规则抽取能出"女性儿童优先"的人话叙事），第二轮可插 **California Housing** 展示 CART 回归。

---

## 2. 算法现代实践 · 时效性素材（2024-2025）

决策树在 2026 年的真实工程地位**不是"被淘汰的老算法"**，而是分成三条清晰的生存线：

1. **弱学习器基石** — XGBoost / LightGBM / CatBoost 全部以 CART 决策树为 base learner。2024 年 Kaggle 表格数据竞赛前 10 名里有 7 个是 boosting 方案，依然跑赢 tabular Transformer（TabNet、FT-Transformer）。LightGBM 在亿级行场景下已成事实标准，histogram-based split + leaf-wise 生长把 XGBoost 甩在 2-5x 速度后。[XGBoost vs LightGBM 对比](https://dataheadhunters.com/academy/xgboost-vs-lightgbm-gradient-boosting-in-the-spotlight/) / [LightGBM NeurIPS 论文](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)

2. **可解释 AI（XAI）主力** — 强监管行业（信贷、保险、医疗、司法）的决策链必须可审计。单棵浅树或 SHAP-on-ensemble 是唯一合规路径。2023 Frontiers 综述系统梳理了决策树从"效率"走向"负责任 AI"的演进。[Frontiers · Decision trees: from efficient prediction to responsible AI](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1124553/full)

3. **规则抽取（Rule Extraction）** — 从黑盒模型或 DRL policy 里反向抽出 if-else 规则。2022 年 RuleCOSI+ 把 tree ensemble 压缩成人类可读规则集；2025 年 ScienceDirect 上有从深度强化学习能源管理策略里抽取决策规则的工作。决策树成为"解释其他模型"的工具。[RuleCOSI+](https://www.sciencedirect.com/science/article/abs/pii/S1566253522001129) / [DRL energy management rule extraction 2025](https://www.sciencedirect.com/science/article/abs/pii/S0378778825002440)

4. **FIGS (Fast Interpretable Greedy-tree Sums)** — 2022 年 Berkeley 提出，CART 的现代扩展，用"一组小树的和"代替单棵大树，兼顾可解释性和精度，被 NIH 用在儿科临床决策支持。[Fast Interpretable Greedy-Tree Sums](https://pmc.ncbi.nlm.nih.gov/articles/PMC11848335/)

**给 Alex 的一句话结论**：**CART 不是历史文物，它是整个表格 ML 生态的细胞**。学决策树不是为了单棵上生产，是为了理解 XGBoost/LightGBM 和 SHAP 的工作机制。

---

## 3. 常见学员追问 · 下限答案

### Q1 · 决策树怎么选分裂特征？信息增益 / 基尼 / 熵到底是什么
**下限答案**：三者都是衡量"这一刀切完左右两堆数据有没有更纯"的指标。
- **信息熵 H = -Σ p·log(p)**：不确定度。全是一类 → 熵=0；均匀分布 → 熵最大
- **信息增益 = 分裂前熵 - 分裂后加权熵**：ID3 用这个
- **信息增益率 = 信息增益 / 特征熵**：C4.5 用这个，惩罚"取值太多"的特征（比如 user_id）
- **基尼 Gini = 1 - Σ p²**：CART 用这个，数学上和熵接近但**免去 log 运算**，sklearn 默认
- **经验结论**：Gini 和 Entropy 的精度差别实测可忽略，Gini 快一点所以是默认。选哪个不是业务决策。
URL：[sklearn 1.10 Decision Trees 文档](https://scikit-learn.org/stable/modules/tree.html) / [Gini vs Entropy 对比](https://www.garysieling.com/blog/sklearn-gini-vs-entropy-criteria/)

### Q2 · 过拟合怎么办？剪枝是什么？
**下限答案**：决策树不约束就会长到每个叶子只剩 1 条样本（训练 100% / 测试崩盘）。两种剪枝：
- **预剪枝**：边长边判断，不增益就停（`max_depth` / `min_samples_split` / `min_samples_leaf`）— 工程熔断器
- **后剪枝**：先长完再砍（sklearn 的 `ccp_alpha` 成本复杂度剪枝）— 工程 GC
- **ccp_alpha 实战**：`clf.cost_complexity_pruning_path(X, y)` 给出一串候选 alpha，用 CV 选最优。sklearn 官方 breast cancer 示例里 ccp_alpha=0.015 是甜区。
URL：[sklearn Post pruning with cost complexity](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html) / [ccp_alpha 选择](https://www.geeksforgeeks.org/machine-learning/how-to-choose-a-in-cost-complexity-pruning/)

### Q3 · 为什么随机森林比单棵决策树好？
**下限答案**：单树方差高（换个训练子集就长成另一棵）。Breiman 2001 把**Bagging（自助采样）+ 特征随机**组合起来——每棵树只看一部分样本 + 一部分特征，**降低树与树之间的相关性**。不相关的错误在投票时互相抵消，variance 大幅下降。关键洞察：树多了不会过拟合（Breiman 原话"Random Forests do not overfit as more trees are added"）。
URL：[Breiman Random Forests Wikipedia](https://en.wikipedia.org/wiki/Random_forest) / [Google ML Decision Forests](https://developers.google.com/machine-learning/decision-forests/random-forests)

### Q4 · CART 和 ID3 / C4.5 区别？现在还需要学 ID3 吗？
**下限答案**：生产上**只用 CART**（sklearn 默认，XGBoost/LightGBM 基座）。
- **ID3 (1975)**：信息增益；只能离散特征；偏爱取值多的字段（会选中 user_id）— 上古
- **C4.5 (1993)**：信息增益率；能处理连续值和缺失值；吃内存 — 过渡
- **CART (1984)**：基尼；严格二叉树；分类+回归都支持；工程化最好 — 现代标配

学 ID3/C4.5 不是为了用，是为了理解"分裂依据"的演进动机。

### Q5 · 决策树能做回归吗？
**下限答案**：能。**CART 回归树**把基尼换成**平方损失 MSE**，叶节点输出从"多数类"换成"样本均值"。构建过程：排序连续特征 → 相邻值取中点作为候选分裂点 → 选 MSE 最小的点切。预测时落到哪个叶节点就输出那个叶子的样本均值。所以回归树的输出是**阶梯函数**（不连续），这和线性回归的连续直线形成戏剧对比。sklearn API：`DecisionTreeRegressor`。
URL：[sklearn tree 文档 §1.10.2](https://scikit-learn.org/stable/modules/tree.html#regression)

### Q6 · 类别特征能直接喂进去吗？
**下限答案**：**sklearn 的 DecisionTreeClassifier 不原生支持类别变量**，必须先编码。三种选择：
- **OneHotEncoder**：最常见，但对树不友好——每次只能按"是不是某一类"切，需要更深的树才能还原 N-way split；高基数特征（城市、商品 ID）灾难
- **OrdinalEncoder**：赋整数编号，但树会误把它当成有序的
- **原生类别支持**：`HistGradientBoostingClassifier(categorical_features=[...])` 或 LightGBM `categorical_feature=` 参数 — 这才是工业解法

这是 sklearn 决策树相对 LightGBM 的一个真实工程短板，不是文档细节。
URL：[sklearn Categorical Feature Support](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_categorical.html) / [Categorical in Decision Tree](https://medium.com/@dyahayusekarkinasih/categorical-feature-in-decision-tree-classifier-3ad0c42c6dcc)

### Q7 · 决策树可解释性 vs 黑盒模型的 trade-off 怎么做？
**下限答案**：三档选型：
1. **强监管 + 精度要求不极致**（信贷评分卡 / 医疗分诊）→ 单棵浅树（max_depth=5-8），每条路径直接给法务看
2. **要精度 + 保留可解释**（互联网风控）→ XGBoost / LightGBM + **SHAP 解释器**。树集成负责精度，SHAP 负责事后归因
3. **纯精度导向**（推荐系统、非监管场景）→ 直接上深度模型 / 大 ensemble

关键：**单棵决策树的"可解释"不是免费的**——一旦 max_depth>10 或特征 >50，人类读路径的能力也崩了。可解释性是和"复杂度预算"强耦合的。
URL：[Rule-Based Systems & XAI](https://fxis.ai/edu/rule-based-systems-decision-trees-guide/)

### Q8 · 决策树的特征重要性怎么看？
**下限答案**：`clf.feature_importances_` — 基于每个特征在所有分裂中带来的基尼/熵减少的加权和。但有两个坑：
- **对高基数特征有偏**（取值多的特征天生有更多分裂机会）
- **相关特征会稀释重要性**（两个强相关特征会互相抢贡献）

更稳健的替代：**Permutation Importance**（`sklearn.inspection.permutation_importance`）或 SHAP values。
URL：[sklearn DecisionTreeClassifier 文档](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

---

## 4. 业界翻车 / 反面教材

### 4.1 · Apple Card 性别歧视事件（2019，Goldman Sachs）
David Heinemeier Hansson（Ruby on Rails 作者）公开 tweet：自己和妻子提交几乎相同的财务信息，自己的 Apple Card 额度是妻子的 **20 倍**，即使妻子信用评分更高。Steve Wozniak 也有同样遭遇。纽约州金融服务部立案调查。
**为什么和决策树相关**：Goldman Sachs 的信用评分模型里大量使用 tree-based 方法（XGBoost / Random Forest）。即使模型**未显式使用性别特征**，也可能通过代理变量（邮编、消费类别、职业编码）间接编码性别——**这正是单棵决策树的"可解释性"被监管要求的原因**：能打开每条路径看是哪些特征在驱动决策。
URL：[Washington Post 报道](https://www.washingtonpost.com/business/2019/11/11/apple-card-algorithm-sparks-gender-bias-allegations-against-goldman-sachs/) / [MIT Tech Review](https://www.technologyreview.com/2019/11/11/131983/apple-card-is-being-investigated-over-claims-it-gives-women-lower-credit-limits/) / [AI Incident DB #92](https://incidentdatabase.ai/cite/92/)

### 4.2 · COMPAS 累犯预测算法种族偏见（ProPublica 2016 调查）
Northpointe 的 COMPAS 系统在美国司法系统广泛用于保释 / 量刑 / 假释决策。ProPublica 分析 10000+ 佛罗里达被告数据发现：**黑人被告被错误标记为"高风险但实际未再犯"的概率几乎是白人的 2 倍**；白人被告则更容易被错误标记为"低风险"。
**为什么和决策树相关**：COMPAS 底层是基于决策树的风险打分。规则如"年龄<25 + 有前科 → 高风险"看似中立，但历史数据中的系统性偏差被树学进去后成为"合法化的歧视"。ProPublica 事件后触发了整个 Fair-ML 子领域，是决策树"可解释"也**救不了算法公平**的经典案例。
URL：[ProPublica · Machine Bias](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing) / [ProPublica 方法论](https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm)

### 4.3 · Drools / 规则引擎在金融反欺诈的规则爆炸
非公开报告案例通用模式：银行初期用决策树导出规则灌进 Drools，上线几个月后规则数从 30 条膨胀到 300+，规则间互相冲突、优先级不清、维护成本指数级上涨，最终必须回到"用模型重新训 + 规则冻结"的模式。**教训**：决策树的"可解释性"要配套"规则治理（Rule Governance）"流程，否则 if-else 变成意大利面。

---

## 5. 推荐阅读清单

1. **sklearn 官方 Decision Trees 文档** — API + 参数 + 复杂度分析一站式。https://scikit-learn.org/stable/modules/tree.html
2. **sklearn Post pruning with cost complexity pruning 教程** — `ccp_alpha` 实战模板。https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
3. **Breiman · Random Forests (2001)** — 原始论文，理解 bagging + feature randomness 的 variance reduction。https://en.wikipedia.org/wiki/Random_forest（含原论文引用）
4. **Frontiers 综述 · Decision trees: from efficient prediction to responsible AI (2023)** — 决策树从"效率工具"转向"可解释 AI 主力"的学术视角综述。https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1124553/full
5. **LightGBM NeurIPS 2017 论文** — 理解 histogram-based split / leaf-wise growth 为什么能 10x 于 XGBoost。https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree
6. **ProPublica · Machine Bias (2016)** — 算法公平性必读非技术读物。https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing

---

## 6. 盲区填补要点（原始笔记 + 第一版改编都没覆盖）

1. **成本复杂度剪枝 `ccp_alpha`**：原始笔记只讲概念性的预剪枝/后剪枝，**没给 sklearn 实际 API**。第一版改编列了 `max_depth`/`min_samples_*` 但**漏了 `ccp_alpha`**——这是 sklearn 0.22+ 的主流后剪枝接口，必须补。
2. **类别特征处理的工程短板**：原始笔记完全不提编码问题；第一版改编也没展开。实际上 sklearn 决策树**不原生支持类别变量**是一个会被学员立即问到的坑，应显式给 OneHot / Ordinal / HistGBT 原生支持三档选型。
3. **特征重要性的偏差陷阱**：两份源都没提 `feature_importances_` 对高基数特征有偏，应在 §4 或 §6.2 里补 Permutation Importance 作为稳健替代。
4. **决策边界可视化**：原始笔记有一些图但没给代码；第一版改编完全没有。**`plot_tree` + `export_text` + 决策边界二维图**是决策树教学的"可视化刚需"，LESSON-PLAN 应该在 §1.2 demo 里直接给。
5. **对 XGBoost / LightGBM 的过渡**：第一版改编 §7 提了"升维到集成学习"，但**没讲清为什么 XGBoost 的弱学习器是 CART 而不是 ID3/C4.5**——因为 CART 的平方损失直接对应 gradient boosting 的 residual 拟合。这是跨章节的概念桥，LESSON-PLAN 要明确点破。
6. **公平性 / Bias 讨论**：两份源都 0 提及。对 Alex 这种做 AI Agent 架构的工程师，Apple Card / COMPAS 这类"可解释但不公平"的反例比数学公式更值得记忆——要进 §6 反面教材。
7. **与 SHAP / LIME 的配合**：可解释性不等于"只能用浅树"。现代解法是"XGBoost + SHAP"——树负责精度，SHAP 负责事后解释。这是 Alex 未来在 RAG/Agent 架构里处理可解释需求的真实工具链。

---

## 7. 与模板 §8.9 三源定位映射

| 来源 | 文件 / URL | 未来 LESSON-PLAN 用在哪 |
|---|---|---|
| **原始培训笔记** | `assets/source-materials/第4阶段-机器学习/决策树.md` | §2 信息熵/信息增益/基尼的公式推导 · §2.2 相亲例子业务锚 · §3 贷款 15 条手推 ID3 · §5.1 记忆层题（熵公式、三算法对比表）· §3 泰坦尼克号官方案例 · §6 剪枝的"好瓜坏瓜"案例 |
| **第一版改编** | `demos/05-Decision-Trees-and-Rule-Extraction.md` | §1.1 if-else 业务锚 · §2.1 黑盒视图（三要素表）· §2 算法对比类比 MySQL 引擎 · §3 过拟合/剪枝的 Redis 缓存类比 · §4 PMML vs ONNX 跨语言落地 · §5 风控系统架构图 · §7 升维到集成学习的过渡 |
| **联网搜索交叉验证** | 本文件第 2-5 节所有 URL | §0.3 候选数据集 · §3 常见追问的 URL 溯源 · §4 Apple Card / COMPAS 反面教材 · §5 权威阅读清单 · §6.1 ccp_alpha 工程细节 · §6.5 XGBoost/LightGBM 过渡桥 · §6.6 公平性讨论 |

### 本摘要引用的外部 URL 清单（供未来 LESSON-PLAN 复用）

**sklearn 官方**：
- [sklearn Decision Trees 文档](https://scikit-learn.org/stable/modules/tree.html)
- [DecisionTreeClassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [Post pruning cost complexity pruning](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html)
- [Categorical Feature Support in Gradient Boosting](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_categorical.html)
- [fetch_california_housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- [load_iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
- [load_breast_cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

**学院派 / 权威综述**：
- [Frontiers · Decision trees: from efficient prediction to responsible AI](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1124553/full)
- [LightGBM NeurIPS 论文](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
- [Random Forest Wikipedia（含 Breiman 原论文）](https://en.wikipedia.org/wiki/Random_forest)
- [Google ML · Decision Forests](https://developers.google.com/machine-learning/decision-forests/random-forests)
- [Fast Interpretable Greedy-Tree Sums (FIGS)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11848335/)
- [RuleCOSI+ 规则抽取](https://www.sciencedirect.com/science/article/abs/pii/S1566253522001129)

**教程 / Q&A**：
- [Gini vs Entropy 对比](https://www.garysieling.com/blog/sklearn-gini-vs-entropy-criteria/)
- [ccp_alpha 选择指南](https://www.geeksforgeeks.org/machine-learning/how-to-choose-a-in-cost-complexity-pruning/)
- [Rule-Based Systems & XAI](https://fxis.ai/edu/rule-based-systems-decision-trees-guide/)
- [XGBoost vs LightGBM 对比](https://dataheadhunters.com/academy/xgboost-vs-lightgbm-gradient-boosting-in-the-spotlight/)
- [Categorical Feature in Decision Tree Classifier](https://medium.com/@dyahayusekarkinasih/categorical-feature-in-decision-tree-classifier-3ad0c42c6dcc)

**业界翻车 / 反面教材**：
- [ProPublica · Machine Bias (COMPAS)](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)
- [ProPublica · How We Analyzed COMPAS](https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm)
- [Washington Post · Apple Card gender bias](https://www.washingtonpost.com/business/2019/11/11/apple-card-algorithm-sparks-gender-bias-allegations-against-goldman-sachs/)
- [MIT Tech Review · Apple Card](https://www.technologyreview.com/2019/11/11/131983/apple-card-is-being-investigated-over-claims-it-gives-women-lower-credit-limits/)
- [AI Incident DB #92 · Apple Card](https://incidentdatabase.ai/cite/92/)

**数据集**：
- [Kaggle Titanic](https://www.kaggle.com/competitions/titanic/data)
- [UCI Adult/Census Income](https://archive.ics.uci.edu/dataset/2/adult)

---

## 附：迭代记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-04-20 | v0.1 | 首版。基于原始笔记 + 第一版改编 + 8 组联网搜索交叉验证。目标是制作 demo-04 LESSON-PLAN 时的起手素材库。 |
