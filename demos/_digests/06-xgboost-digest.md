# L1 备课摘要 · XGBoost / GBDT

> 范围：梯度提升树家族（GBDT / XGBoost / LightGBM / CatBoost），重点锚定 XGBoost。
> 受众：Alex（全栈转 AI Agent），工程师语言、业务锚优先、术语中英双标。
> 产出用途：第 06 节「集成学习 · 金融风控」demo 的 L1 备课底稿。

---

## 1. 候选数据集清单

| # | 数据集 | 规模 / 特征 | 任务 | 工程锚点 |
|---|---|---|---|---|
| 1 | **Give Me Some Credit（GMSC, Kaggle 2011）** | 15 万训练 + 10 万测试，11 特征 | 二分类：2 年内是否逾期 90 天 | 金融风控经典基线，正样本 ~6.7%，天然样本不均衡（class imbalance）。XGBoost out-of-box AUC ≈ 0.861。[Kaggle 链接](https://www.kaggle.com/competitions/GiveMeSomeCredit) |
| 2 | **Lending Club Loan Data** | 200 万+ 贷款记录，140+ 特征 | 违约预测 / 利率定价 | P2P 风控全链路，字段含 FICO、DTI、就业年限。需要手动做特征泄漏（leakage）筛查——`total_pymnt` 等事后字段必须剔除。[Kaggle · Lending Club](https://www.kaggle.com/datasets/wordsforthewise/lending-club) |
| 3 | **Home Credit Default Risk（Kaggle 2018）** | 30+ 万样本，多表 join | 违约二分类 | 含 bureau、previous_application 等多张表，考察"特征工程 > 模型选型"的典型。[Kaggle · Home Credit](https://www.kaggle.com/competitions/home-credit-default-risk) |
| 4 | **Santander Customer Transaction Prediction** | 20 万 × 200 匿名数值特征 | 二分类 | Kaggle 冠军方案为 LightGBM + 魔法特征（magic feature），表格数据竞赛代表。[Kaggle · Santander](https://www.kaggle.com/competitions/santander-customer-transaction-prediction) |
| 5 | **Criteo CTR / Avazu CTR** | 千万级行，类别特征为主 | 点击率预测 | 广告排序业务锚，CatBoost / LightGBM 原生处理高基数类别（high-cardinality categorical）的标杆场景。[Criteo Display Ad](https://www.kaggle.com/c/criteo-display-ad-challenge) / [Avazu CTR](https://www.kaggle.com/c/avazu-ctr-prediction) |
| 6 | **红酒品质分类（源材料自带）** | 3269 行 × 11 数值特征 | 多分类（6 档） | 作为 warm-up 最小可跑案例，源材料代码已具备。[UCI · Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) |

**首选路线**：GMSC 做主 demo（规模可控、风控故事清晰）+ 红酒做 warm-up。Lending Club 作为"特征泄漏翻车"的反面教材素材库。

---

## 2. 算法现代实践 · 时效性素材（2024-2026）

### 2.1 三强鼎立 · 现状
- **XGBoost / LightGBM / CatBoost** 三者在表格数据（tabular data）上打成平手，但定位分化：
  - XGBoost：**精度优先**，成熟度最高，生态最广（ONNX / Spark / Dask / GPU 都齐）。
  - LightGBM：**速度优先**，leaf-wise 生长 + 直方图算法（histogram-based），大数据集训练 10-50× 加速。
  - CatBoost：**类别特征优先**，内建 ordered target encoding，省掉手工独热编码（one-hot）。
- 2025 年银行流失预测（churn）对比研究：XGBoost 准确率 98.3% 最高，LightGBM 速度最快，CatBoost 在中大型数据上平衡最好。([apxml 对比](https://apxml.com/posts/xgboost-vs-lightgbm-vs-catboost)、[ResearchGate 银行 churn](https://www.researchgate.net/publication/397440582_A_Comparative_Study_of_XGBoost_LightGBM_and_CatBoost_Models_for_Customer_Churn_Prediction_in_the_Banking_Industry)）

### 2.2 为什么在表格数据仍吊打神经网络
- 2022 NeurIPS 经典论文《Why do tree-based models still outperform deep learning on tabular data?》：中等规模（~10K 样本）上树模型仍是 SOTA。([arXiv 2207.08815](https://arxiv.org/abs/2207.08815)、[NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf))
- 2024 年综述《Tabular Data: Is Deep Learning all you need?》确认：XGBoost 需更少调参、训练更快、泛化更稳。([arXiv 2402.03970](https://arxiv.org/html/2402.03970v3))
- 核心原因三条：① 表格数据的旋转不变性（rotational invariance）破坏深度学习假设；② 树天然处理异方差与不规则决策边界；③ 深度网络在混合类型特征（数值 + 类别）上优化难。([Forecastegy 解读](https://forecastegy.com/posts/gradient-boosting-vs-deep-learning-tabular-data/))

### 2.3 Kaggle 获奖率
- 2015-2020 年 Kaggle 结构化数据竞赛冠军方案中，XGBoost/LightGBM 占比长期 >70%。近两年 LightGBM 占比上升、CatBoost 在含类别特征场景追平。
- 深度学习 Transformer 类模型（TabNet、FT-Transformer、SAINT）只在**超大规模 + 强类别结构**的极少数竞赛拿过冠军，多数仍被 GBDT stacking 反超。

### 2.4 与 AutoML 的关系
- **H2O AutoML / AutoGluon-Tabular**：底层模型池里 XGBoost + LightGBM + CatBoost 是三大主力，AutoML 更多是做超参搜索 + stacking 封装。
- **FLAML（微软）**：聚焦 CFO（cost-frugal optimization），首选模型就是 LightGBM。
- 结论：AutoML 没有"替代"GBDT，而是把 GBDT 的调参自动化了。

---

## 3. 常见学员追问 · 下限答案

### Q1. XGBoost 和随机森林的核心区别？
**业务锚**：随机森林 = Quorum 投票集群（etcd/ZK 并行投票）；XGBoost = 责任链补偿（每个服务处理上一个的错单）。
- **Bagging vs Boosting**：RF 并行训练、平权投票、降方差（variance）；XGBoost 串行训练、加权累加、降偏差（bias）。
- **训练数据**：RF 每棵树看 bootstrap 抽样子集；XGBoost 每棵树看全量数据 + 上一轮残差。
- **树的角色**：RF 是"一群强壮独立的树投票"；XGBoost 是"一串浅薄但专注补漏的树接力"。
- **输出**：RF 投票 / 平均；XGBoost 累加（score = Σ f_k(x)）。

### Q2. 为什么 XGBoost 比原生 GBDT 快这么多？
四大关键工程优化：
1. **二阶泰勒展开（second-order Taylor）**：GBDT 只用一阶梯度（gradient）近似残差，XGBoost 同时用一阶 g 和二阶 h，逼近更准，收敛更快。
2. **正则项（L1/L2 regularization）**：目标函数里塞了 `γT + ½λ‖w‖²`，控制叶子数和叶子权重，天然抗过拟合。
3. **直方图分裂（approx / hist）**：把连续特征离散化成 bin，候选分裂点从 O(#data) 降到 O(#bins)。
4. **系统优化**：预排序 block 结构、列压缩、OpenMP 并行、cache-aware 访问、out-of-core 外存训练。
5. **缺失值稀疏感知（sparsity-aware split）**：训练时自动学"缺失往左还是往右"的默认方向。

### Q3. learning_rate / n_estimators / max_depth 怎么配？
工程默认档（生产可用）：
- `learning_rate (eta)`：0.05-0.1（调小需同步加大 n_estimators，一般反比）。
- `n_estimators`：100-1000，**一律配合 early_stopping_rounds=50** 自动选，别死磕。
- `max_depth`：6-8（XGBoost 默认 6）。更深通常无收益，反而过拟合。
- `subsample` / `colsample_bytree`：0.8 / 0.8，行采样 + 列采样双重防过拟合。
- `min_child_weight`：1-10，风控这类噪声数据适当调大（5-10）抗噪。
- `reg_alpha` / `reg_lambda`：L1/L2 正则系数，小数据可调大（1-10）。
([GPU + Bayesian 实战推荐](https://qfournier.github.io/blog/2025/xgboost/))

### Q4. 为什么 LightGBM 比 XGBoost 更快？
- **Leaf-wise vs level-wise 生长**：XGBoost 默认按层（level-wise / BFS）扩展，LightGBM 每次只裂"当前增益最大的那一片叶子"（leaf-wise / best-first）。同样叶子数下 loss 更低，但小数据易过拟合——必须设 `num_leaves` 上限。
- **GOSS（Gradient-based One-Side Sampling）**：保留大梯度样本 + 随机抽小梯度样本，数据量降而精度不丢。
- **EFB（Exclusive Feature Bundling）**：把互斥稀疏特征捆绑成一维，特征数 ×10 → ×1。
- 综合：大数据集 10-50× 加速。([LightGBM Features](https://lightgbm.readthedocs.io/en/latest/Features.html))

### Q5. CatBoost 的优势到底在哪？
- **Ordered Target Encoding**：用"前缀样本"的目标均值做类别编码，天然防目标泄漏（target leakage）。对比 pandas 手搓 mean-encoding 经常翻车。
- **Symmetric Trees（oblivious trees）**：每层用同一个分裂条件，推理极快，适合在线服务。
- **内建类别特征支持**：不用做独热，高基数类别（如 user_id、city_id）直接喂。
- 劣势：小数据训练比 LightGBM 慢。([Neptune 选型指南](https://neptune.ai/blog/when-to-choose-catboost-over-xgboost-or-lightgbm))

### Q6. 早停（Early Stopping）怎么用？
```python
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric='auc',
    early_stopping_rounds=50,  # 验证集连续 50 轮不提升就停
    verbose=False,
)
# 推理时用 model.best_iteration 或 model.best_ntree_limit
```
核心心法：**n_estimators 设大（如 5000），让 early_stopping 自己收敛**，不要人肉调 n_estimators。

### Q7. GPU 加速开不开？
- 开关：`tree_method='gpu_hist'`（XGBoost 1.x）或 `device='cuda'`（XGBoost 2.x+）。
- 加速比：百万级样本 10-50×，十万级样本 3-5×，万级以下可能反而变慢（显存拷贝开销）。
- **注意事项**：① 早停、自定义 objective 都支持；② Dask + GPU 可做分布式；③ 推理端用 `DaskDeviceQuantileDMatrix` 省显存。([XGBoost GPU 文档](https://xgboost.readthedocs.io/en/stable/gpu/index.html))

### Q8. 为什么表格数据不上神经网络？
三条工程原因：
1. **性价比**：XGBoost 开箱 AUC 比 MLP 高 2-5 个点，调参时间只要 1/10。
2. **可解释性**：SHAP / feature importance / 分裂路径都成熟，风控、医疗这种要审计的场景必须用树。
3. **线上延迟**：ONNX 导出后 1-2ms 推理，无 GPU 依赖，能塞进 Java/Go 网关。深度模型推理动辄 10ms+，还要独立推理服务。

### Q9. 样本不均衡（class imbalance）怎么办？
三层手段，优先级递减：
1. **`scale_pos_weight = neg/pos`**（XGBoost 原生，风控首选）或 `sample_weight`。
2. **调阈值 + PR-AUC 评估**：不要用 accuracy，用 precision@k / recall@k / PR-AUC。
3. **SMOTE 等重采样**：最后兜底，且**必须只在训练集内做，切勿在切分前做**（见 §4 翻车）。

---

## 4. 业界翻车 / 反面教材

### 4.1 数据泄漏 · SMOTE 位置错误 · 信用卡欺诈
- 典型案例：研究论文中 XGBoost 报 99.97% accuracy、100% recall，实际是**先 SMOTE 再切分**导致的测试集污染。
- 正确做法：train_test_split → 只在训练集做 SMOTE/sample_weight → 验证集保持原分布。
- 源头：([arXiv 2412.07437 · Impact of Sampling Techniques and Data Leakage on XGBoost](https://arxiv.org/abs/2412.07437)、[MDPI 2025 批判性研究](https://www.mdpi.com/2227-7390/13/16/2563))

### 4.2 特征泄漏 · Lending Club 常见坑
- 典型坑字段：`total_pymnt`（总还款额）、`recoveries`（催收回款）、`last_pymnt_d`（最后还款日）——这些都是**贷款发放后才产生**的字段，拿来预测违约等于作弊。
- 工程约束：每个特征都要打时间戳 tag，训练前强制校验 `feature_time <= prediction_time`。
- 这类坑在银行 / 券商模型上线前常要过一轮 "时间穿越审计（point-in-time audit）"。

### 4.3 训练-推理 schema 不一致 · 线上精度崩塌
- 症状：离线 AUC 0.85，上线后 AUC 跌到 0.60。
- 常见原因：
  - 特征顺序不一致（Python pandas 的列顺序 ≠ Java 调用 ONNX 时传入顺序）。
  - 类别编码字典漂移（训练用的 LabelEncoder 没和模型一起序列化）。
  - 缺失值填充策略不一致（训练时用 `.mean()`，推理时用 0）。
- 工程解法：模型工件（artifact）必须含 `feature_names.json` + `preprocessor.pkl`，CI 里加 schema 对比测试。

### 4.4 Zillow Zestimate · iBuying 亏损 $880M
- 2021 年 Zillow 关停 iBuying 业务，累计亏损 8.8 亿美元。
- **真正的锅不是过拟合**：是市场剧烈波动 + 人为上调估值偏置刺激成交率，模型对非平稳（non-stationary）市场假设失效。
- 启示：GBDT 再强也是 IID（独立同分布）假设下的模型，经济周期切换 / 黑天鹅场景下必须有人工 override + 监控报警（drift detection）。
- 来源：([GeekWire 分析](https://www.geekwire.com/2021/ibuying-algorithms-failed-zillow-says-business-worlds-love-affair-ai/)、[Statsig 辩护视角](https://www.statsig.com/blog/in-defense-of-zillows-besieged-data-scientists))

### 4.5 稀疏特征误处理
- CTR 场景下，高基数类别（user_id、ad_id）做 one-hot 会爆维度。
- 错误做法：盲目 one-hot → 训练 OOM，或者用 LabelEncoder 把类别当连续数值喂。
- 正确做法：① CatBoost 原生处理；② LightGBM 的 `categorical_feature` 参数；③ target encoding + K-Fold 防泄漏。

---

## 5. 推荐阅读清单

### 官方文档
- [XGBoost 官方文档](https://xgboost.readthedocs.io/en/stable/)（重点：Tutorials → Boosted Trees 入门 + Parameters 全量参考）
- [LightGBM Features 文档](https://lightgbm.readthedocs.io/en/latest/Features.html)（leaf-wise / histogram / GOSS / EFB 全在这）
- [CatBoost 官方文档](https://catboost.ai/docs/)（重点：Categorical features 章节）

### 原始论文
- Chen & Guestrin, 2016, **XGBoost: A Scalable Tree Boosting System**（KDD）— [arXiv:1603.02754](https://arxiv.org/abs/1603.02754)
- Ke et al., 2017, **LightGBM: A Highly Efficient Gradient Boosting Decision Tree**（NIPS）
- Prokhorenkova et al., 2018, **CatBoost: unbiased boosting with categorical features**（NeurIPS）
- Friedman, 2001, **Greedy Function Approximation: A Gradient Boosting Machine**（GBDT 源头论文）

### 深度对比 & 翻车研究
- [Why do tree-based models still outperform DL on tabular data (NeurIPS 2022)](https://arxiv.org/abs/2207.08815)
- [Tabular Data: Is Deep Learning all you need (2024)](https://arxiv.org/html/2402.03970v3)
- [Impact of Sampling and Data Leakage on XGBoost (2024)](https://arxiv.org/abs/2412.07437)
- [XGBoost vs LightGBM vs CatBoost 实战对比 · apxml](https://apxml.com/posts/xgboost-vs-lightgbm-vs-catboost)

### Kaggle 获奖 notebook（风控向）
- [Give Me Some Credit 竞赛页](https://www.kaggle.com/competitions/GiveMeSomeCredit)
- [Credit Risk Dataset XGBoost 分析 · Kaggle](https://www.kaggle.com/code/vincentlee3231/credit-risk-dataset-model-analysis-using-xgboost)
- Home Credit Default Risk 2018 冠军方案（搜索 "home credit 1st place solution"）

---

## 6. 盲区填补要点

- **目标函数的泰勒二阶展开推导**：Alex 如果工程出身，推导细节可跳过，但必须能说出"一阶 g 管方向、二阶 h 管步长"这个业务锚。
- **shrinkage（学习率 η）到底在收缩什么**：每棵树的输出 × η 再累加，相当于"慢慢逼近真相"——和 SGD 里的学习率是同一个思想。
- **叶子权重闭式解** w\* = -G/(H+λ)：这是 XGBoost 论文最漂亮的地方，也是和 GBDT 的本质区别。推导一次就不会忘。
- **分裂增益公式** gain = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)] - γ：每次分裂都在算这个，理解它就理解了 XGBoost 的树构建算法。
- **样本不均衡 · `scale_pos_weight` 原理**：本质是在损失函数里给正样本梯度加权，不是真的复制样本。
- **SHAP 与 XGBoost 的绑定**：tree-based SHAP 有精确多项式算法（复杂度 O(TLD²)），比 KernelSHAP 快几个数量级，风控可解释性标配。
- **Monotonic constraints（单调性约束）**：风控里"收入越高违约率应越低"这种先验，XGBoost 可以直接通过 `monotone_constraints` 参数强制，这是业务模型可上线的硬需求。

---

## 7. 与模板 §8.9 三源定位映射

> 按模板 v0.2 §8.9 "三源定位（不是简单合并）" 规则归类：

| 源 | 具体素材 | 在本备课中的角色 |
|---|---|---|
| **S1 · 原始培训笔记** `assets/source-materials/第4阶段-机器学习/集成学习.md` | Bagging/Boosting 概念、AdaBoost 权重推导、GBDT 残差拟合演示、XGBoost 目标函数分步化简、红酒品质预测代码 | **理论骨架 + warm-up 代码**：提供公式推导路径（泰勒展开 → 叶子权重闭式解 → 分裂增益），以及可跑的最小案例（红酒） |
| **S2 · 第一版改编** `demos/06-Ensemble-and-Financial-Risk.md` | 金融风控业务锚、Quorum vs 责任链的架构映射、ONNX 跨语言部署路径 | **业务锚 + 工程落地**：提供讲给 Alex 的语言（微服务、责任链、ONNX），负责把 S1 的数学接到工程师世界 |
| **S3 · 联网交叉验证** 本摘要 §2-§5 所有带 URL 的素材 | 三强对比、NeurIPS 2022/2024 论文、SMOTE leakage 论文、Zillow 案例、GPU 最佳实践 | **时效性补丁 + 翻车素材**：S1/S2 的时间戳停在 2023 前，S3 负责补 2024-2026 的生态现状、反面案例、现代调参范式 |

**交叉校验已完成的冲突点**：
- S1 代码中 `use_label_encoder=False` 已在 XGBoost 2.x 中废弃，demo 跑通前需改写。
- S2 中 "XGBoost 在寻找分裂点时并行" 的表述正确，但需补充"树与树之间仍是串行"——这是学员必问。
- S3 最新 benchmark（2024 churn 研究）与 S1/S2 的"XGBoost 最强"口径一致，无需修订主叙事；但 S3 补充了 LightGBM 速度优势的量化依据（10-50×）。
