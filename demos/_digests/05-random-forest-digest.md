# L1 备课摘要 · 随机森林（Random Forest）

> 这是 demo-05-random-forest 的 L1 素材包，不是最终 LESSON-PLAN。
> 本文档只负责"原料齐全 + URL 落地"，教学叙事结构等 LESSON-PLAN 阶段再拼。

---

## 1. 候选数据集清单（§0.3）

随机森林的教学价值在于：① 展示"bagging + 特征随机"的方差削减机制，② 展示特征重要性（feature importance）的**工程诱惑 + 陷阱**，③ 作为 KNN 之后的"第一个真正能上线的结构化数据算法"。数据集选型围绕这三个教学目标。

| 候选 | 规模 | 任务 | 教学价值（能讲什么钩子） | 适合场景 | 数据源 URL |
|---|---|---|---|---|---|
| **Titanic 泰坦尼克号** | 891 × 12 | 二分类（生存预测） | 原始培训笔记默认数据集；混合类型特征（数值+类别+缺失）；有"Sex/Pclass"这种强特征，能演示 feature importance 直观排序 | 零基础 / 想跟原教材对齐 | [Kaggle Titanic](https://www.kaggle.com/c/titanic) |
| **Wine Quality 红酒品质** | 4898 × 12 | 多分类（1-10 品质） | 原始笔记 XGBoost 章节用的数据；全数值特征，适合演示 RF 不需要标准化（对比 KNN 必须 scale） | 想和 demo-03 KNN 做对照组 | [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) |
| **German Credit 德国信贷** | 1000 × 20 | 二分类（违约预测） | **业界最经典风控数据集**；类别极度不平衡（好客户:坏客户 = 7:3）；能引爆"accuracy 虚高 + 必须看 recall/PR-AUC"的话题 | 想走风控主线 / 对接 demo-02 评估指标 | [UCI Statlog German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) |
| **Credit Card Fraud 信用卡欺诈** | 284807 × 30 | 二分类（欺诈） | **极端不平衡**（欺诈 0.17%）；直接复现"全预测为 0 也有 99.83% accuracy"的反教材；PCA 脱敏后的 V1-V28 是讲 feature importance 陷阱的好素材 | 想讲不平衡 + SMOTE + class_weight | [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **California Housing** | 20640 × 8 | 回归 | 演示 RandomForestRegressor；可与 demo-01 线性回归对照（RF 捕捉非线性 + 交互项） | 想对比 RF 回归 vs 线性回归 | [sklearn fetch_california_housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) |

**默认推荐**：第一轮 **German Credit**（1000 条够快 + 风控故事线和 Alex 转型方向贴）。Alex 想对比"RF vs 线性回归捕捉非线性"转 **California Housing**。想炸不平衡话题转 **Credit Card Fraud**。

---

## 2. 算法现代实践 · 时效性素材（2024-2025）

RF 在 2024-2025 的定位：**不是 SOTA，而是"上线首选 baseline + 可解释性护城河"**。在 Kaggle 竞赛（XGBoost/LightGBM/CatBoost 统治）和研究论文（Transformer/TabPFN 冲击）之外的**工业真实场景**，RF 依然是大量生产系统的主力。

### 2.1 金融风控 · 信用评分 / 反欺诈
- **2025 RF 信用风险模型**：在微金融机构客户信用风险分级中，RF 模型达到 **89% 整体准确率**，precision/recall 在各类别之间平衡——工业界可直接上线的典型表现。见 [Customer Credit Risk Levels Using Random Forest (2025)](https://www.researchgate.net/publication/398125736_Classification_of_Customer_Credit_Risk_Levels_Using_the_Random_Forest_Method_A_Case_Study_on_Microfinance_Institutions)。
- **企业财务数据风险预警**：RF 在企业级财务数据风险预警模型中，accuracy 和 recall 可达 **90%+，最高 97%**。见 [IBM · What is Random Forest](https://www.ibm.com/think/topics/random-forest)。
- **SHAP + RF 合规解释**：信用决策拒绝时必须向客户和监管解释原因（debt-to-income ratio / delinquency history / credit length），SHAP 把 RF 变成"可审计"的黑盒。见 [Xoriant · Random Forest Use Cases](https://www.xoriant.com/blog/random-forest-algorithm)。

### 2.2 Kaggle 表格竞赛 · RF 在集成里的角色
- **2024 AutoGluon 横扫**：AutoGluon 在 18 个 tabular 比赛中拿下 15 个奖牌 / 7 金——底层 stack 里 RF 是**固定成员**，但不再是主力。见 [ML Contests · State of ML Competitions 2024](https://mlcontests.com/state-of-machine-learning-competitions-2024/)。
- **2025 Kaggle Podcast 冠军方案**：Grandmaster Chris Deotte 用 **3 层 stack / 72 个模型** 拿下第一——XGBoost + LightGBM + CatBoost + NN + TabPFN + KNN + SVR + Ridge + **RF**，最后 Ridge + GBM 聚合。见 [Kaggle Playground 2025 Winning Strategies](https://medium.com/@gauurab/kaggle-playground-how-top-competitors-actually-win-in-2025-c75d4b380bb5)。
- **NVIDIA 的 Kaggle Grandmaster Playbook**：RF 现在被定位成"快速 baseline + 异质性补充"而非核心模型。见 [NVIDIA · Kaggle Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)。

### 2.3 推荐冷启动 baseline
- RF 在**物品冷启动 / 用户冷启动 / 双冷启动**三个场景都有系统性研究——用户画像特征 + 物品特征进 RF 做 rating 预测，不依赖协同过滤历史。见 [Aggregated Recommendation through Random Forests (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4142736/)。
- 工程定位："冷启动第一版就上线"——RF 不依赖 embedding、不依赖用户历史序列，只要有特征就能训。

### 2.4 sklearn 1.1 默认值变更（工程师必知）
- `RandomForestClassifier` 的 `max_features` 从 `"auto"` 改为 `"sqrt"`（功能等价，但老代码会报 deprecation 警告）。见 [sklearn 1.8.0 RandomForestClassifier 文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)。

---

## 3. 常见学员追问 · 下限答案

> 按 Alex 追问优先级排序。每条带"最短能糊弄过去的答案 + URL 兜底"。

### Q1 · Bagging 和 Boosting 什么区别？
**下限答案**：
- Bagging（RF 代表）= **并行** + **有放回采样** + **平权投票**。每棵树独立，训练可以多线程。降方差（variance）。
- Boosting（XGBoost/GBDT 代表）= **串行** + **全量数据** + **加权投票**。后面的树拟合前面的残差。降偏差（bias）。
- **工程类比**：Bagging = ZooKeeper 集群投票（Quorum）；Boosting = 责任链模式（前一个服务处理不了的丢给下一个）。

**URL**：[GeeksforGeeks · Random Forest vs XGBoost](https://www.geeksforgeeks.org/machine-learning/difference-between-random-forest-vs-xgboost/) / [Xoriant · XGBoost vs LightGBM vs RF](https://www.xoriant.com/blog/gradient-boosting-in-machine-learning-xgboost-lightgbm-random-forest-explained)

### Q2 · 为什么随机森林"不怕过拟合"？
**下限答案**：这话**只对了一半**。
- 对的部分：**加树不会过拟合**——Breiman 2001 原论文用大数定律证明，树数 n_estimators 增加时 generalization error 收敛到极限值。见 [Breiman 2001 Random Forests (PDF)](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)。
- 错的部分：RF **依然能过拟合**——如果单棵树 `max_depth=None` 无限深 + 数据噪声大，整个森林的 bias 继承自单棵树，test/train gap 会拉大。修复手段：`max_depth` / `min_samples_leaf` / `min_samples_split`。见 [mljar · Does Random Forest overfit?](https://mljar.com/blog/random-forest-overfitting/) / [alexmolas · Can Random Forests overfit?](https://www.alexmolas.com/2022/12/26/random-forest-overfit.html)。

### Q3 · n_estimators 怎么选？
**下限答案**：
- 默认 100。范围一般 50-400。
- **有饱和点**——加到一定程度后性能提升可忽略，但训练时间线性增长。
- 调参法：从 100 起步，画 `n_estimators vs OOB/CV score` 曲线，找拐点。
- 工业界经验值：结构化数据 200-500 够用；如果 500 还没饱和，问题多半不在 n_estimators（特征工程更重要）。

**URL**：[Configure n_estimators (SKLearner)](https://sklearner.com/sklearn-randomforestclassifier-n_estimators-parameter/) / [GeeksforGeeks · Effects of Depth and Number of Trees](https://www.geeksforgeeks.org/machine-learning/the-effects-of-the-depth-and-number-of-trees-in-a-random-forest/)

### Q4 · 特征重要性怎么算？靠谱吗？
**下限答案**：**默认的不靠谱，必须切换到 permutation importance**。
- sklearn 默认 `feature_importances_` = **MDI（Mean Decrease in Impurity）**，基于训练集上每次分裂的基尼增益累加。
- **两大致命缺陷**：
  1. **偏向高基数特征**——连续数值 / 高唯一值分类特征会被系统性高估。极端情况：一列纯随机生成的 ID 都能排进 top-3。
  2. **训练集计算**——过拟合时，噪声特征在训练集上也能刷出高分，泛化能力**完全没体现**。
- **正确姿势**：`sklearn.inspection.permutation_importance`，在**测试集**上打乱某列，看 score 掉多少。缺点：相关特征会互相掩盖（permute A 时 B 还在，B 还能撑住）。

**URL**：[sklearn · Permutation Importance vs MDI](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html) / [sklearn · Permutation Feature Importance](https://scikit-learn.org/stable/modules/permutation_importance.html) / [explained.ai · Beware Default RF Importances](https://explained.ai/rf-importance/)

### Q5 · 为什么要做特征随机采样（max_features）？
**下限答案**：**解决"强特征垄断"问题**。
- 如果只 bagging 样本不 bagging 特征——假设有一列超强特征，每棵树第一刀都切它，**所有树变得高度相关**，bagging 就只剩"重复投票"没有"多样性求平均"。
- `max_features=sqrt(n)`（分类默认）/ `n/3`（回归，R 和 h2o 默认）= 每次分裂只从随机 k 个特征里挑最优，**强制其他特征也有出场机会**，树之间去相关（decorrelate），方差削减才生效。
- 这就是"随机森林"名字里**两重随机**：样本随机（bagging）+ 特征随机（max_features）。

**URL**：[sklearn MOOC · Random Forests](https://inria.github.io/scikit-learn-mooc/python_scripts/ensemble_random_forest.html) / [Lorentzen · Feature Subsampling for RF Regression](https://lorentzen.ch/index.php/2021/08/19/feature-subsampling-for-random-forest-regression/)

### Q6 · 树数多 = 效果好吗？有没有饱和点？
**下限答案**：**有饱和点，且加树不降 bias 只降 variance**。
- bias 天花板 = 单棵树的 bias。加多少树都破不了。
- variance 随树数增加而降，但边际递减——每多一棵树贡献越来越小。
- 典型 saturation：n_estimators 200-400 后几乎不动。
- 工程成本：训练时间 + 模型大小（部署 ONNX / PMML 时模型文件可能几十 MB 到几百 MB）都线性增长。
- **所以不要盲目加树**，先看曲线。

**URL**：[Crunching the Data · Number of trees in RF](https://crunchingthedata.com/number-of-trees-in-random-forests/) / [Baekho Lab · RF Hyperparameters (2025)](https://baekholab.com/2025/03/22/part-6-random-forest-hyperparameters-tuning-strategies/)

### Q7 · OOB (Out-of-Bag) 是什么？为什么可以代替 CV？
**下限答案**：**bagging 的免费验证集**。
- 有放回采样时，每棵树约 **37% 样本没被抽到**（`(1-1/n)^n → 1/e ≈ 0.368`）——这些就是该树的 OOB 样本。
- 每条训练样本天然有一批"没见过它的树"，用这些树做预测 → 得到无偏的泛化误差估计。
- 使用：`RandomForestClassifier(oob_score=True)`，然后读 `clf.oob_score_`。
- **和 CV 的差异**：
  - OOB 是 RF/bagging 专属，CV 通用
  - OOB 免费（训练时顺带计算），CV 需要重复训练 k 次
  - 大样本下 OOB 收敛到 leave-one-out CV
- **工程价值**：数据量大、训练贵时，OOB 代替 5-fold CV 省 5 倍训练时间。

**URL**：[sklearn · OOB Errors for RF](https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html) / [Wikipedia · Out-of-bag error](https://en.wikipedia.org/wiki/Out-of-bag_error) / [Analytics Vidhya · OOB Score](https://www.analyticsvidhya.com/blog/2020/12/out-of-bag-oob-score-in-the-random-forest-algorithm/)

### Q8 · 和 XGBoost/LightGBM 对比，什么时候选 RF？
**下限答案**：**选 RF 的四个典型场景**。
1. **第一版 baseline / 时间紧**：RF 默认参数性能 = 调优后 97-99%；XGBoost 默认参数 = 调优后 90-95%。RF 开箱即用。
2. **数据噪声大 / 样本少**：RF 天然鲁棒（bagging 削方差），XGBoost 容易过拟合噪声。
3. **可解释性要求高（风控 / 医疗）**：RF + SHAP 的组合比 XGBoost + SHAP 更稳定。
4. **训练要多线程但推理延迟不敏感**：RF 树间独立，训练天然多线程。

**选 XGBoost/LightGBM 的场景**：
- 大数据量 + 愿意调参 → 精度天花板比 RF 高 2-5%
- LightGBM 在十亿级数据 / CTR 场景几乎无替代

**URL**：[MCP Analytics · XGBoost vs RF](https://mcpanalytics.ai/articles/xgboost-vs-random-forest-comparison) / [Medium · Tree-Based Models Showdown](https://medium.com/@sebuzdugan/random-forest-xgboost-vs-lightgbm-vs-catboost-tree-based-models-showdown-d9012ac8717f) / [apxml · XGBoost vs LightGBM vs CatBoost](https://apxml.com/posts/xgboost-vs-lightgbm-vs-catboost)

---

## 4. 业界翻车 / 反面教材

### 4.1 · 特征重要性误用 · 高基数列"假冠军"
**现象**：某团队用 RF 做用户流失预测，feature importance 显示"user_id" 排第一名。
**真相**：user_id 是高基数分类列，MDI 天然偏向——每次分裂它都能劈出纯节点，累计基尼增益爆表，但**毫无泛化价值**。
**修复**：切 permutation importance + 在测试集上算 + 删 ID 类特征。
**URL**：[explained.ai · Beware Default RF Importances](https://explained.ai/rf-importance/) / [sklearn · Permutation vs MDI](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html)

### 4.2 · 不平衡数据 accuracy 虚高 · 信用卡欺诈场景
**现象**：Credit Card Fraud 数据集欺诈比例 0.17%。RF 默认参数直接训，accuracy 99.8%+——老板看 dashboard 表示满意。
**真相**：模型把所有样本预测为"非欺诈"，一个欺诈都没抓到。recall = 0%，precision 无定义。
**修复**：
- 评估换 **PR-AUC / F1 / Recall**（不能看 accuracy）
- 训练加 `class_weight='balanced'` 或 SMOTE 重采样
- 业务决策看 confusion matrix 不看单一指标

**URL**：[Nature · Credit Card Fraud Detection with RF+SMOTE (2025)](https://www.nature.com/articles/s41598-025-00873-y) / [Keylabs · Handling Imbalanced Data](https://keylabs.ai/blog/handling-imbalanced-data-to-improve-precision/) / [AIMS · RF interpretability under class imbalance (2024)](https://www.aimspress.com/article/doi/10.3934/DSFE.2024019?viewType=HTML)

### 4.3 · 数据泄漏 · "在线模型上线就崩"
**现象**：训练集 AUC 0.95，测试集 AUC 0.93，上线后线上 AUC 0.62。
**真相**：特征里混入了**未来信息**——例如"用户是否投诉"这列在训练时用的是事后标注，但预测时拿不到。RF 对相关特征不敏感不代表对泄漏不敏感——有泄漏它照样会学，而且学得特别好（误导性高）。
**修复**：
- 所有特征都要问一句"**预测时间点**这列能拿到吗？"
- 时序数据用 **time-based split** 而不是随机 split
- 检查高相关特征（`|corr(X_i, y)| > 0.9`）——往往是泄漏信号
- 用 pipeline 把预处理（scaler/encoder）包进 CV 折内，避免 preprocessing leakage

**URL**：[Springer · Don't push the button! Data leakage risks (2025)](https://link.springer.com/article/10.1007/s10462-025-11326-3) / [ScienceDirect · Leakage and reproducibility crisis in ML](https://www.sciencedirect.com/science/article/pii/S2666389923001599) / [Machine Learning Mastery · Data Leakage](https://machinelearningmastery.com/data-leakage-machine-learning/)

---

## 5. 推荐阅读清单

1. **Breiman, Leo (2001). "Random Forests". *Machine Learning* 45 (1): 5-32**. 原始论文，必读——尤其"加树不过拟合"的证明部分。[PDF](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
2. **sklearn 官方 · RandomForestClassifier 文档**。API + 默认值 + 实现细节权威源。[Link](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
3. **sklearn MOOC · Random Forests 章节**（Inria 官方课程）。讲"为什么特征随机采样能 decorrelate"讲得最清楚。[Link](https://inria.github.io/scikit-learn-mooc/python_scripts/ensemble_random_forest.html)
4. **Terence Parr et al. · "Beware Default Random Forest Importances"**。feature importance 陷阱的工业界权威吐槽文。[Link](https://explained.ai/rf-importance/)
5. **Google Developers · Decision Forests**（ML 课程模块）。工程视角，讲 RF 在 production 的定位。[Link](https://developers.google.com/machine-learning/decision-forests/random-forests)

---

## 6. 盲区填补要点（原始笔记 + 改编版都没覆盖）

原始培训笔记 `集成学习.md` 和改编版 `06-Ensemble-and-Financial-Risk.md` 的共同缺口，LESSON-PLAN 必须补：

1. **OOB score 完全没提**。原始笔记 + 改编版都没讲 `oob_score=True`，这是 RF **最工程向的特性**（免费验证集），必须讲，能省 Alex 未来大量 CV 训练时间。
2. **feature importance 的 MDI 陷阱没讲**。原始笔记只字未提 `feature_importances_`；改编版提了但没说"默认不靠谱"。必须在 LESSON-PLAN 里铺 permutation importance 对照 demo。
3. **类别不平衡处理**。泰坦尼克号本身不算不平衡（生存率 38%），但 Alex 的目标方向是风控/反欺诈，`class_weight='balanced'` + PR-AUC 评估是**必讲**。
4. **n_estimators 饱和曲线**。原始笔记说"一般选取值较大"——太模糊，需要给 Alex 具体的调参手法（画曲线找拐点）。
5. **max_features 为什么是 sqrt**。原始笔记列了 API 参数但没讲**机制**——去相关化是 RF 名字里"随机"的核心，讲不透这点，Alex 就只是背了个默认值。
6. **RF vs XGBoost 的选型表**。原始笔记把 RF/AdaBoost/GBDT/XGBoost 串着讲，没给"什么时候选谁"的工程决策框架——Alex 作为架构师最需要这个。
7. **sklearn 版本迁移坑**：`max_features="auto"` 在 sklearn 1.3+ 已 deprecated，改为 `"sqrt"`。老代码迁移时要提醒。
8. **Software 1.0 vs 2.0 映射**。改编版提了"Bagging = Quorum 集群"但没延伸到"为什么 2025 年还要学 RF"——锚点要和 Alex 的 AI Agent 方向对接：**RF 是 LLM Agent 里"工具调用分类 / 意图识别"的常见 fallback 模型**（轻量、可解释、离线部署）。

---

## 7. 与模板 §8.9 三源定位映射

| 来源 | 文件 / URL | 本 LESSON-PLAN 将用在哪 |
|---|---|---|
| **原始培训笔记** | `assets/source-materials/第4阶段-机器学习/集成学习.md`（§"随机森林"章节，L107-L256） | §2.2 构建过程五步（样本随机 + 特征随机 + CART + 投票）· §2.3 API 参数表 · §5.1 记忆层（两重随机 / n_estimators / max_features 的角色） |
| **第一版改编** | `demos/06-Ensemble-and-Financial-Risk.md`（§2 Bagging 部分 L19-L35） | §1.1 业务锚（Bagging = Quorum 投票集群）· §3.4 Software 1.0 vs 2.0 映射 · §2.1 黑盒视图（并行性 / 样本随机 / 特征随机的工程类比） |
| **联网搜索交叉验证** | 详见 §2 / §3 / §4 全文嵌入 URL | §0.3 数据集清单（Titanic / German Credit / CCFraud）· §3.2 permutation importance 对照 demo · §4 术语卡（OOB / max_features / feature importance / class_weight）· §6 反面教材（MDI 陷阱 / 不平衡 accuracy 虚高 / 数据泄漏） |

### 联网 URL 清单（本次 agent 交付）

**算法原理 / 权威源**：
- [Breiman 2001 · Random Forests 原论文](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
- [sklearn · RandomForestClassifier 文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [sklearn MOOC · Random Forests](https://inria.github.io/scikit-learn-mooc/python_scripts/ensemble_random_forest.html)
- [Google Developers · Decision Forests / Random Forests](https://developers.google.com/machine-learning/decision-forests/random-forests)
- [IBM · What Is Random Forest](https://www.ibm.com/think/topics/random-forest)

**OOB**：
- [sklearn · OOB Errors for Random Forests](https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html)
- [Wikipedia · Out-of-bag error](https://en.wikipedia.org/wiki/Out-of-bag_error)
- [Analytics Vidhya · Out-of-Bag Score](https://www.analyticsvidhya.com/blog/2020/12/out-of-bag-oob-score-in-the-random-forest-algorithm/)

**Feature Importance 陷阱**：
- [sklearn · Permutation Importance vs MDI](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html)
- [sklearn · Permutation Feature Importance 用户指南](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [explained.ai · Beware Default RF Importances](https://explained.ai/rf-importance/)

**n_estimators / max_features / 超参**：
- [Configure n_estimators (SKLearner)](https://sklearner.com/sklearn-randomforestclassifier-n_estimators-parameter/)
- [GeeksforGeeks · Effects of Depth and Number of Trees](https://www.geeksforgeeks.org/machine-learning/the-effects-of-the-depth-and-number-of-trees-in-a-random-forest/)
- [Crunching the Data · Number of trees](https://crunchingthedata.com/number-of-trees-in-random-forests/)
- [Baekho Lab · RF Hyperparameters Tuning 2025](https://baekholab.com/2025/03/22/part-6-random-forest-hyperparameters-tuning-strategies/)
- [Lorentzen · Feature Subsampling for RF Regression](https://lorentzen.ch/index.php/2021/08/19/feature-subsampling-for-random-forest-regression/)

**过拟合争议**：
- [mljar · Does Random Forest overfit?](https://mljar.com/blog/random-forest-overfitting/)
- [alexmolas · Can Random Forests overfit?](https://www.alexmolas.com/2022/12/26/random-forest-overfit.html)

**RF vs XGBoost/LightGBM**：
- [GeeksforGeeks · Random Forest vs XGBoost](https://www.geeksforgeeks.org/machine-learning/difference-between-random-forest-vs-xgboost/)
- [MCP Analytics · XGBoost vs Random Forest](https://mcpanalytics.ai/articles/xgboost-vs-random-forest-comparison)
- [Medium · Tree-Based Models Showdown](https://medium.com/@sebuzdugan/random-forest-xgboost-vs-lightgbm-vs-catboost-tree-based-models-showdown-d9012ac8717f)
- [Xoriant · XGBoost vs LightGBM vs RF](https://www.xoriant.com/blog/gradient-boosting-in-machine-learning-xgboost-lightgbm-random-forest-explained)
- [apxml · XGBoost vs LightGBM vs CatBoost](https://apxml.com/posts/xgboost-vs-lightgbm-vs-catboost)

**工业场景 / 2024-2025 实战**：
- [Xoriant · RF Algorithm Use Cases](https://www.xoriant.com/blog/random-forest-algorithm)
- [ResearchGate · Customer Credit Risk with RF (2025)](https://www.researchgate.net/publication/398125736_Classification_of_Customer_Credit_Risk_Levels_Using_the_Random_Forest_Method_A_Case_Study_on_Microfinance_Institutions)
- [ML Contests · State of ML Competitions 2024](https://mlcontests.com/state-of-machine-learning-competitions-2024/)
- [NVIDIA · Kaggle Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- [Kaggle Playground 2025 Winning Strategies](https://medium.com/@gauurab/kaggle-playground-how-top-competitors-actually-win-in-2025-c75d4b380bb5)
- [PMC · Aggregated Recommendation through Random Forests](https://pmc.ncbi.nlm.nih.gov/articles/PMC4142736/)

**不平衡 / 反面教材**：
- [Nature · Credit Card Fraud Detection with RF+SMOTE (2025)](https://www.nature.com/articles/s41598-025-00873-y)
- [Keylabs · Handling Imbalanced Data](https://keylabs.ai/blog/handling-imbalanced-data-to-improve-precision/)
- [AIMS · RF Interpretability under Class Imbalance](https://www.aimspress.com/article/doi/10.3934/DSFE.2024019?viewType=HTML)

**数据泄漏**：
- [Springer · Data Leakage Risks in ML (2025)](https://link.springer.com/article/10.1007/s10462-025-11326-3)
- [ScienceDirect · Leakage and Reproducibility Crisis](https://www.sciencedirect.com/science/article/pii/S2666389923001599)
- [Machine Learning Mastery · Data Leakage](https://machinelearningmastery.com/data-leakage-machine-learning/)

**数据集**：
- [Kaggle · Titanic](https://www.kaggle.com/c/titanic)
- [UCI · Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)
- [UCI · Statlog German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- [Kaggle · Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [sklearn · fetch_california_housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
