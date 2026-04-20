# L1 备课摘要 · PCA / 特征降维

> 定位：stage-1 第 8 节「PCA / 特征降维」L1 级备课调研。为正式 LESSON-PLAN 提供素材库，不等于教案本身。

## 1. 候选数据集清单

| 数据集 | 规模 | 教学价值 | 入口 |
|---|---|---|---|
| **Iris** | 150 × 4 | 最小验证：4 维 → 2 维，4 类别色散图最直观，和 KNN/决策树互通 | `sklearn.datasets.load_iris` |
| **MNIST / Digits** | 1797 × 64（sklearn 小版本） / 70k × 784（完整版） | 手写数字降维可视化，10 类在 2D 平面的分离度是 PCA vs t-SNE vs UMAP 的经典对比题 | `sklearn.datasets.load_digits` / `fetch_openml('mnist_784')` |
| **Faces / Olivetti** | 400 × 4096 | **Eigenfaces**（特征脸）历史名案——1991 Turk & Pentland，把每张脸重构成 N 个特征脸的线性组合，是讲"主成分=方向"最直观的素材 | `sklearn.datasets.fetch_olivetti_faces` |
| **Breast Cancer Wisconsin** | 569 × 30 | 和 demo-02 同数据集，30 维 → 2 维后良/恶肿瘤仍可视分开，复用感强 | `sklearn.datasets.load_breast_cancer` |
| **高维基因表达 / scRNA-seq** | 数千样本 × 2 万基因 | 真实高维场景，也是 UMAP 在生物圈走红的原因（单细胞聚类已是 UMAP 标配） | Scanpy / PBMC 3k |
| **LLM Embedding（OpenAI text-embedding-3 / MiniLM）** | N × 384 / 768 / 1536 / 3072 | **和 demo-03 KNN 直接对接**：把 embedding 用 PCA 压到 100 维后入库，是 2025 RAG 工程标配 | 自造 + sentence-transformers |

## 2. 算法现代实践 · 时效性素材

### 2.1 PCA 在 2024-2026 的地位
不是"被 embedding 淘汰"，而是**退到 embedding 下游**：
- 上游：Transformer 的 Embedding 层已经做了"学习型降维"，取代了 PCA 对原始文本/图像做降维的角色
- 下游：embedding 本身 768/1536/3072 维太贵，PCA 重新回到"embedding 压缩器"的位置

### 2.2 PCA 在向量数据库 / RAG 的角色（桥接 demo-03 KNN）
2025 的一批论文直接把 PCA 塞进 RAG pipeline：
- **PCA-RAG (arXiv 2504.08386, 2025)**：对 embedding 做 PCA 压缩后做检索，平衡质量与成本
- **Embedding 存储优化 (arXiv 2505.00105, 2025)**：系统对比 PCA / Kernel PCA / UMAP / Random Projection / Autoencoder，结论是 **PCA 是性价比最高的降维手段**；float8 量化 + PCA 保留 50% 维度 = 8× 总压缩，性能损失小于单独用 int8
- **极端案例**：3072 维 embedding 降到 110 维，检索质量仍可接受
- **Compressing LLMs with PCA (arXiv 2508.04307, 2025)**：70 维 PCA 后 embedding 喂给 2 层 transformer，20 Newsgroups 上 76.62% 准确率；decoder-only 模型在 70 维 PCA embedding 上仍能生成连贯 token，和完整 MiniLM 表示的余弦相似度 > 97%

> 工程锚：**PCA 现在是 Vector DB 的"JPEG 压缩层"**，上游 embedding 是"光学传感器原始信号"，下游 HNSW/IVF 是"CDN 分发"。

### 2.3 PCA vs t-SNE vs UMAP 选型

| 维度 | PCA | t-SNE | UMAP |
|---|---|---|---|
| 类型 | 线性 | 非线性 | 非线性 |
| 保留结构 | **全局方差** | 局部邻域 | 全局+局部平衡 |
| 速度 | 最快（解析解） | 最慢 | 快（approx. t-SNE 10×） |
| 确定性 | 确定性 | 随机（每次不同） | 半确定 |
| 能 transform 新样本 | **能**（直接投影） | **不能**（需重训） | 能（官方实现支持） |
| 可解释性 | 主成分 = 原特征线性组合 | 无 | 无 |
| 适用场景 | **特征工程 / pipeline 一环 / 去噪** | 小到中数据集探索性可视化 | 大数据集可视化 + 单细胞生物 |

**一句话选型**：进 pipeline 选 PCA；画图讲故事选 UMAP；小数据找局部簇选 t-SNE。

### 2.4 降维在 ML pipeline 中的位置
- **特征工程（主）**：高维稀疏 → 低维稠密，下游模型训练更快、更稳
- **去噪**：丢掉小方差分量等于丢掉噪声维度
- **可视化**：降到 2D/3D 肉眼看聚类是否真的成簇
- **存储压缩**：embedding 入库前降维省 RAM / 省磁盘

## 3. 常见学员追问 · 下限答案

1. **PCA 是线性还是非线性？**
   线性。它只做正交旋转+投影。非线性降维走 Kernel PCA（核技巧）/ t-SNE / UMAP / Autoencoder。（出处：[sklearn KernelPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html) / [sklearn TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) / [sklearn Manifold learning](https://scikit-learn.org/stable/modules/manifold.html) / [UMAP 原论文 arXiv:1802.03426](https://arxiv.org/abs/1802.03426)）

2. **保留多少个主成分？**
   工程默认：累计方差贡献率（explained_variance_ratio_ 累加）≥ 95%。
   实操 3 个标尺：① 95% 方差；② 肘部法（碎石图，scree plot 拐点）；③ 下游模型 CV 分数随 k 的曲线。sklearn 里 `PCA(n_components=0.95)` 直接传小数。

3. **PCA 之前必须标准化吗？**
   **必须**。PCA 基于方差，未标准化时量纲大的特征（如"收入"元 vs "年龄"岁）会主宰主成分方向，结果失真。标准做法：`StandardScaler().fit_transform(X)` 再 PCA。BytePlus / sklearn 官方 common_pitfalls 都把这条列为头号坑。（出处：[BytePlus · Common PCA mistakes](https://www.byteplus.com/en/topic/399421) / [sklearn Common Pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) / [sklearn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html) / [microbiozindia · Common mistakes in PCA](https://microbiozindia.com/common-mistakes-in-pca-and-how-to-avoid-them/)）

4. **特征值 / 特征向量 / 主成分 什么关系？**
   对协方差矩阵做特征分解：**特征向量 = 主成分方向**（新坐标轴），**特征值 = 该方向上的方差大小**。按特征值从大到小排序取前 k 个就是"前 k 主成分"。`explained_variance_ratio_ = 特征值 / 特征值之和`。（出处：[sklearn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) / [Wikipedia · Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) / [PSU STAT505 · Eigenvalues and Eigenvectors](https://online.stat.psu.edu/stat505/lesson/4/4.5) / [Pressbooks · PCA (Linear Algebra and Applications)](https://pressbooks.pub/linearalgebraandapplications/chapter/principal-component-analysis/)）

5. **PCA 之后特征还能解释吗？**
   通常**失去直观可解释性**。新主成分是原特征的线性组合（系数在 `components_` 里），想解释就去看每个主成分在哪些原特征上权重高——但往往是几十个特征的加权和，没法说"这是收入轴"。需要可解释降维请用 Sparse PCA 或特征选择。（出处：[sklearn SparsePCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html) / [sklearn Decomposition · Sparse PCA](https://scikit-learn.org/stable/modules/decomposition.html)）

6. **Kernel PCA / Sparse PCA 什么时候用？**
   - **Kernel PCA**：数据非线性分布（如同心圆），需先通过核函数映射到高维再 PCA（[sklearn KernelPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html) / [sklearn · Kernel PCA 示例](https://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html)）
   - **Sparse PCA**：要可解释性，强制每个主成分只依赖少数几个原特征（[sklearn SparsePCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html) / [sklearn MiniBatchSparsePCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchSparsePCA.html)）
   - **Incremental PCA**：数据装不进内存，流式处理（[sklearn IncrementalPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html) / [sklearn · Incremental PCA 示例](https://scikit-learn.org/stable/auto_examples/decomposition/plot_incremental_pca.html)）

7. **t-SNE 和 UMAP 对比 PCA 各自场景？**
   见 §2.3 表。补一句：**t-SNE 不能做 `.transform(new_data)`**——训练-推理分离的 ML pipeline 里这是硬伤，UMAP 和 PCA 都没这个问题。

8. **PCA 降维后再跑 KNN 有意义吗？**
   **有**，且是 2025 RAG 的主流做法。KNN 在高维空间里受"维度灾难"拖累（距离失真，§4 翻车清单），PCA 压到 50-100 维既省内存又常常提点。直接桥到 demo-03。

## 4. 业界翻车 / 反面教材

1. **未标准化直接 PCA**（最常见）
   量纲不一时主成分完全跟着大量纲特征走。sklearn 官方 common_pitfalls 页面、BytePlus、microbiozindia 等都列为头号错误。（出处：[sklearn Common Pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) / [BytePlus · Common PCA mistakes](https://www.byteplus.com/en/topic/399421) / [microbiozindia · Common mistakes in PCA](https://microbiozindia.com/common-mistakes-in-pca-and-how-to-avoid-them/)）

2. **在整个数据集上 fit PCA 再 split → 数据泄漏（data leakage）**
   PCA 的均值、标准差、主成分方向都是"训练侧参数"。如果先 `fit_transform(X)` 再切 train/test，测试集信息已经进入训练——模型表面指标虚高，上线崩盘。**正确做法**：`Pipeline([StandardScaler, PCA, Model])` + 只在 train 上 fit。（出处：[sklearn Common Pitfalls · data leakage](https://scikit-learn.org/stable/common_pitfalls.html) / [MachineLearningMastery · Data Preparation Without Leakage](https://machinelearningmastery.com/data-preparation-without-data-leakage/) / [TDS · Data Leakage in Preprocessing](https://towardsdatascience.com/data-leakage-in-preprocessing-explained-a-visual-guide-with-code-examples-33cbf07507b7/)）

3. **训练-推理使用不同均值/方差（状态漂移）**
   和 demo-03 KNN 同类问题：训练时 scaler/PCA 的 mean_/components_ 不持久化，上线重新 fit 导致投影方向不一样，推理时向量空间对不上。**正确做法**：`joblib.dump(pipeline)` 或导出 ONNX，确保 Python 训练 → Go/Java/Node 推理用同一套参数。（出处：[sklearn PCA · components_/mean_ 属性](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) / [sklearn Common Pitfalls · inconsistent preprocessing](https://scikit-learn.org/stable/common_pitfalls.html#inconsistent-preprocessing)）

4. **过度降维导致信息损失崩盘**
   盲目追求"压缩率"，把 95% 方差降到 60%，下游分类/检索直接掉点。典型症状：离线评测还行，A/B 实验数据一上就输给未降维版本。（出处：[sklearn PCA · explained_variance_ratio_](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) / [Embedding Storage Optimization arXiv:2505.00105](https://arxiv.org/html/2505.00105) / [Milvus · Reduce embedding size](https://milvus.io/ai-quick-reference/how-do-you-reduce-the-size-of-embeddings-without-losing-information)）

5. **把 PCA 用在明显非线性数据上**
   例如同心圆、瑞士卷（Swiss roll）数据，PCA 只能拿到方差最大的线性方向，丢掉流形结构。换 Kernel PCA / UMAP。（出处：[sklearn Manifold learning · Swiss roll](https://scikit-learn.org/stable/modules/manifold.html) / [sklearn · Kernel PCA 示例](https://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html) / [Raschka · RBF Kernel PCA 教程](https://sebastianraschka.com/Articles/2014_kernel_pca.html) / [GeeksforGeeks · Swiss Roll LLE](https://www.geeksforgeeks.org/machine-learning/swiss-roll-reduction-with-lle-in-scikit-learn/)）

6. **用 PCA 的主成分当"业务指标"解释给产品/老板**
   第 5 问已说过，PC1/PC2 不是可直接叙事的业务变量。这种误用常见于 BI 看板，给出看似"深度洞察"实则玄学的数字。（出处：[BytePlus · Common PCA mistakes](https://www.byteplus.com/en/topic/399421) / [sklearn SparsePCA · 可解释性替代](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html) / [microbiozindia · Common mistakes in PCA](https://microbiozindia.com/common-mistakes-in-pca-and-how-to-avoid-them/)）

## 5. 推荐阅读清单

- **[sklearn.decomposition.PCA 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)** — 参数与属性，`explained_variance_ratio_` / `components_` / `mean_` 先熟记
- **[sklearn Common Pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)** — 官方列明的 PCA/数据泄漏坑
- **[PCA-RAG paper (arXiv 2504.08386, 2025)](https://arxiv.org/html/2504.08386v1)** — PCA 在 RAG 里的工程价值
- **[Embedding Storage Optimization (arXiv 2505.00105, 2025)](https://arxiv.org/html/2505.00105)** — PCA vs 其他降维方案在 embedding 压缩上的系统对比
- **[Compressing LLMs with PCA (arXiv 2508.04307, 2025)](https://arxiv.org/html/2508.04307v1)** — 70 维 PCA embedding 仍可训 transformer
- **[UMAP 原论文 (McInnes et al., 2018)](https://arxiv.org/abs/1802.03426)** — UMAP 数学基础
- **[PCA vs UMAP vs t-SNE · biostatsquid](https://biostatsquid.com/pca-umap-tsne-comparison/)** — 三者并列对比最清晰的科普
- **[Which Dimension Reduction Method Should I Use? · Duke](https://sites.duke.edu/dimensionreduction/)** — Duke 决策树式选型指南
- **[Milvus · How to reduce embedding size](https://milvus.io/ai-quick-reference/how-do-you-reduce-the-size-of-embeddings-without-losing-information)** — 向量数据库厂商视角
- **[Zilliz · Dimensionality reduction via PCA/autoencoder](https://zilliz.com/ai-faq/how-can-one-reduce-the-dimensionality-or-size-of-embeddings-through-methods-like-pca-or-autoencoders-to-make-a-largescale-problem-more-tractable-without-too-much-loss-in-accuracy)** — Vector DB 工程实践
- **[BytePlus · Common PCA mistakes](https://www.byteplus.com/en/topic/399421)** — 翻车清单
- Jay Alammar 的博客虽然暂无 PCA 专篇，但 [Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) 是 embedding 几何直觉的最佳入门，可以和 PCA 的"方差方向"对照讲

## 6. 盲区填补要点

1. **PCA ↔ Embedding ↔ Vector DB 三者的工程关系**（demo-03 桥梁）
   "PCA 是 embedding 的 JPEG 压缩层"这个锚必须讲透——Alex 背景里 embedding 是 LLM 产物，PCA 是统计学老东西，但 2025 它们在 RAG pipeline 是上下游关系，不是取代关系。

2. **PCA 的 `mean_` / `components_` 是"训练产物"这件事**
   像模型权重一样需要持久化和版本管理。这是跨语言部署（Python 训练 → Go/Java 推理）时最容易丢的状态。同 KNN 的 scaler 漂移问题，教学上可以复用同一个"状态漂移"锚。

3. **维度灾难（curse of dimensionality）的直觉**
   高维空间里"所有点都差不多远"——KNN 在原始 784 维 MNIST 上效果不如 50 维 PCA 后。这是为什么"PCA + KNN"组合在 demo-03 收尾时值得演示一次。

4. **PCA 给 LLM 新人的入口直觉**
   embedding 的 1536 维里不是每一维都有独立语义，它们高度相关；PCA 告诉你"真正有效自由度只有 100-200 维"。这是讲 embedding 几何的低门槛切入点。

5. **不讲的东西**（避免过度展开）
   SVD 与 PCA 的数学等价、协方差矩阵特征分解推导、Kernel trick 细节——L1 级不讲，放到 §2.3 数学附录备用。

## 7. 与模板 §8.9 三源定位映射

| 模板章节 | 主要来源 | 补充来源 | 本摘要对应位置 |
|---|---|---|---|
| §1 钩子 · 业务锚 | 第一版改编（JPEG 压缩 / One-Hot 崩溃） | 联网搜索（2025 embedding 压缩论文） | §2.1 / §2.2 / §6.1 |
| §2.1 黑盒视图 | 第一版改编（"降维打击"/"有损压缩"类比） | — | §2.1 |
| §2.2 核心机制 | 原始培训笔记（方差最大化、正交变换） | — | §3.4 / §3.5 |
| §2.3 数学附录 | 原始培训笔记（协方差/特征值/特征向量） | 联网搜索（SVD 实现细节） | §3.4 |
| §3 坏→好递进 | 第一版改编（One-Hot → PCA → Embedding） | 联网搜索（PCA-RAG、PCA+量化混合压缩） | §2.1 / §2.2 / §6.1 |
| §4 术语定义 | 原始培训笔记（主成分、方差贡献率、低方差过滤、皮尔逊/斯皮尔曼） | 第一版改编（稠密 vs 稀疏、Embedding 术语） | §3 所有 8 问 |
| §4 术语追问答案 | — | **联网搜索全部附 URL**（sklearn / arXiv / biostatsquid / Duke） | §3 + §5 |
| §5.1 记忆层题 | 原始培训笔记（Iris PCA 代码、VarianceThreshold 代码） | — | §1 候选数据集 |
| §6 常见坑 | 原始笔记（"注意标准化"） + 改编版（跨语言状态漂移） | 联网搜索（sklearn common_pitfalls / BytePlus / data leakage 案例） | §4 全部 6 条 |

> 三源分工核查：原始笔记贡献数学骨架与基础代码；第一版改编贡献"JPEG 压缩 / Embedding 桥梁"工程锚；联网搜索贡献 2025 时效素材（PCA-RAG、embedding 压缩论文、t-SNE/UMAP 对比、翻车清单 URL）。每一节都至少用到 2 源，§4 翻车 / §5 阅读清单同时咬合 3 源。
