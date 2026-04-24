## 99 · 心智模型蒸馏

> **这份文档是什么**：阶段一（Machine Learning Foundation）入门后真正内化下来的心智模型，面向未来的复习和同路人的阅读。
>
> **和同目录其他文档的区别**：
> - `01-LESSON-PLAN.md` 是**阶段一教学大纲**（学习方法论 + A/B/C 必记清单 + 自评表 + 进度追踪）
> - 这份 DISTILL 是**蒸馏后的结晶**——只保留经过实际 Q&A 踩坑验证过的核心心智，每个技术论断都挂到可查的外部权威源
>
> **阅读视角**：第三人称客观陈述。踩过的坑升维成「⚠️ 陷阱」旁注。
>
> **组织方式**：以"什么是机器学习"为锚（零节），其余所有段落按**严格依赖顺序**排列——先讲数据（原料），再讲任务分类（流派），再讲过程（流程），再讲结果（产物），再讲工程落地、姿态、位置、杠杆。每一节只依赖零节总纲 + 前面章节的概念，不提前使用后面才讲的东西。
>
> **脚注说明**：技术论断用 `[^n]` 标记，GitHub / Obsidian / Typora 等渲染器会显示为上标，鼠标悬浮预览外网权威源（Wikipedia / sklearn 官方 / Karpathy 原文等），点击跳转文末。学习轨迹（session / highlights）另列「学习轨迹」小节。

---

### 零、什么是机器学习

Tom Mitchell 1997 年给的经典定义：[^1]

> "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E."

翻译成工程师语言：**机器学习 = 通过数据投喂让算法自己总结出处理逻辑的方法**。

和传统编程的分水岭在"**处理逻辑从哪来**"：

| | 传统编程（Software 1.0） | 机器学习（Software 2.0[^2]） |
|---|---|---|
| 处理逻辑 | 人写（每一行都是自己的思考逻辑） | 算法自己总结（从数据里反推） |
| 输入 | 规则 + 数据 | 数据 + 答案 |
| 输出 | 按规则处理后的结果 | **一套规则**（模型文件） |
| 调试 | 读代码、加日志 | 加数据、调特征 |

> 📎 **用词说明**：Karpathy 原文严格把 "Software 2.0" 定义为**神经网络权重作为代码**的范式[^2]。本文档沿用社区惯例，把 Software 2.0 宽泛扩展到所有"数据驱动、权重被算法学出来"的机器学习范式（包括决策树、线性回归等），以突出它和 Software 1.0 的对比结构。严格引用 Karpathy 时请回到原文。

"算法自己总结"的过程有个工程名字——**训练循环（training loop）**：

```
  数据投喂  →  算法选择  →  产生处理逻辑
                                 ↓
  修正处理逻辑  ←  指标检查  ←  对输入产生输出
```

对应到标准 ML 术语：

- 数据投喂 = 训练数据
- 算法选择 = 模型类型选择（决策树 / 线性回归 / 神经网络 ...）[^3]
- 指标检查 = 损失函数 / 评估指标
- 修正处理逻辑 = 梯度下降 / 反向传播

所以"机器学习"不是一个算法、不是一个库、不是一个领域。它是一种**让计算机从数据里反推规则的编程范式**。

> 🧭 **锚点句**（记住这一句，后面所有段落都挂在它下面）：
>
> "函数的处理逻辑是人写的，模型的处理逻辑是算法自己总结的。"

---

### 一、原料 · 特征 vs 标签

训练循环吃的"数据投喂"具体是什么？是一张带答案的表：

| 城市 | 面积 | 年限 | 是否靠地铁 | **成交价** |
|---|---|---|---|---|
| 上海 | 89㎡ | 8 年 | 是 | **680 万** |
| 杭州 | 110㎡ | 3 年 | 否 | **420 万** |

- **特征（Feature）**：前几列，算法的输入信号
- **标签（Label）**：最后一列，算法要学着预测的目标

数学上监督学习的训练样本被描述为 `(x_i, y_i)` 配对，`x_i` 是特征向量，`y_i` 是标签。[^8]

**训练的本质 = 给算法看一大堆 (feature, label) 配对，让它自己摸出 feature → label 的映射**。

#### 🪤 陷阱 1：相关 ≠ 因果

特征和标签之间需要的是**统计相关性**，不是因果关系。[^9]

**经典反例（Wikipedia 也收录）**：冰淇淋销量 ↔ 溺水人数高度相关——可以训出"预测"模型。但冰淇淋不会导致溺水，两者都是"夏天"这个混淆变量（confounding variable）的结果。

**后果**：如果把模型输出当因果读，会做出荒谬决策。例："流失用户里'打过客服电话'是最强特征"→ 产品经理："那关掉客服电话就不流失了？" ❌ 错，打电话是流失的**症状**不是**原因**。因果关系是独立的、更难的领域（因果推断 Causal Inference），不是 ML。

#### 💡 洞察：label 不是数据天生的，是工程师的选择

同一张 5 列的表，指定哪一列是 `y` 决定整个任务在做什么：

- `y = 是否复购` → 训出「复购预测器」
- `y = 月消费` → 训出「消费金额预测器」
- `y = 城市` → 训出「根据行为反推用户城市的模型」

**从物理上看 feature 和 label 就是一样的数据**，让它们变得不一样的**不是数据，是工程师定义问题的方式**。这个认知是 AI Agent 工程师 80% 的工作——**定义问题比写代码难**。

---

### 二、流派 · 监督 vs 无监督

按"有没有 label"划分，机器学习分成两大流派：[^8][^10]

| | 监督学习[^8] | 无监督学习[^10] |
|---|---|---|
| 训练数据 | 有 (feature, **label**) 配对 | **只有 feature**，没 label |
| 算法在学 | feature → label 的映射 | 数据本身的结构 |
| 典型任务 | 分类（垃圾邮件）、回归（房价） | 聚类（用户分群）、降维 |
| 评估难度 | ⭐ 简单（对答案算准确率） | ⭐⭐⭐⭐ 难（没答案可对）[^11] |

#### ⚠️ 陷阱：feature 表几乎一样，差别只在"那一列"

反直觉的地方：监督和无监督的 feature 表**长得几乎一模一样**，差别只是监督版多了最后一列 label。

**区分不在"给了多少数据"，在"有没有显式指定一列作为目标"**（伪代码，`fit` 表示"开始拟合/训练"）：

```python
# 监督
X = df[['年龄', '城市', '月消费', '浏览次数']]
y = df['是否复购']
fit(X, y)   # 指定了 y

# 无监督
X = df[['年龄', '城市', '月消费', '浏览次数']]
fit(X)      # 没指定 y
```

把 label 列和 feature 一起塞进 `X` 不指定 `y`，监督算法会退化成无监督——它会把本该是 label 的那列当普通字段用。

#### 🪤 陷阱：无监督的评估没有 ground truth

聚类（clustering，无监督学习的典型任务）把样本分成 5 群，**怎么知道这 5 群对不对**？没有标准答案可对比。

sklearn 官方给出的补救是：[^11]
- **有 ground truth 时**：Rand Index、Mutual Information 等
- **没 ground truth 时**：Silhouette Coefficient（轮廓系数）——衡量"群内样本紧密 + 群间样本分离"的程度

工业界更常用**下游任务反推**——分完群后做精准营销 A/B 测试，看转化率提升了多少。算法只负责"凑堆"（"群 1 有 80 万人"），**人负责解读**每个群的业务含义（称作 persona labeling）。

---

### 三、流程 · 训练 / 测试 / 推理（三元结构）

有了原料（feature/label）和流派（监督/无监督），接下来的问题是：**具体怎么跑起来**？

初学者容易把 ML 流程理解成"训练 vs 测试"两段，**真实结构是三段**（更严格讲是四段：训练 / 验证 / 测试 / 推理）：[^6]

```
训练 Training  →  验证 Validation  →  测试 Testing  →  推理 Inference
（学规则）       （调超参数）        （最终评分）      （生产：用规则做预测）
```

| 阶段 | 数据 | 有无标签 | 目的 |
|---|---|---|---|
| 训练 | 训练集 | 有 | 让算法从 (feature, label) 里摸出映射规律 |
| 验证 | 验证集 | 有 | 调超参数、早停，避免过拟合 |
| 测试 | 测试集 | 有 | 用没见过的带标签数据评估准确率 |
| 推理 | 线上真实请求 | **无**（label 正是要预测的东西） | 生产环境做预测 |

**为什么必须分开**：如果拿训练数据做测试，算法完全可以"死记硬背"得到 100% 准确率——这叫**过拟合（Overfitting）**[^7]，上线遇到没见过的数据就翻车。

**训练完成的物理含义**：把"处理逻辑固化"——训练完成 = 把映射规则冻结到一个文件里。于是这个文件可以序列化、可以跨语言加载。下一节就讲这个文件。

#### ⚠️ 陷阱：测试 ≠ 推理

两个动作都是"把数据喂进去看输出"，但：
- **测试**有标准答案（测试集的 label），输出用来算准确率
- **推理**没标准答案（线上请求只有 feature，label 就是你要预测的），输出直接返回给用户

混成一回事，会导致讨论"模型上线后怎么评估"时方向错乱。

---

### 四、产物 · 模型是一个函数

训练跑完，**产出的东西叫"模型"**。模型从后端工程师的角度看就是一个函数 `y = f(x)`：

- **输入** `x`：一个浮点数数组（即上文讲的特征向量）
- **输出** `y`：一个数字（回归）或一组类别概率（分类）
- **中间**：一堆数学变换，存在一个文件里（`.pkl` / `.onnx`[^4] / `.pmml`[^5]）
- **调用**：`y = predict(x)`，和 `Math.sqrt(4)` 没有本质区别

**结构完全一致（输入/处理/输出），区别只在处理逻辑的来源**——这就是"模型"和"函数"的唯一差别，也是"机器学习"和"传统编程"的唯一差别。

#### ⚠️ 陷阱：模型 ≠ 黑盒

决策树、线性回归**是完全白盒**——可以直接打印权重、导出 if-else 规则。"黑盒"描述的是**复杂度**（深度网络、Transformer 参数多到人类读不懂），不是模型的本质属性。心智建立时不要把"不可解释"写进模型的定义里。

---

### 五、落地 · 离线训练 / 在线推理

模型是个文件，这意味着**训练**和**推理**可以物理隔离在两条链路：

```
【离线侧 · Python 主场】
  历史数据 → 特征工程 → 训练 → 模型文件（.onnx / .pmml） → 模型仓库

【在线侧 · Java/Go/Node 主场】
  请求 → 实时聚合特征 → 加载模型 → predict() → 返回 + 监控
```

**对后端开发者的意义**：常见存量技能（MySQL/Redis/Kafka/gRPC/Docker/k8s/Prometheus）直接复用。**新学的只有 4 件事**：

1. 训练侧 API（sklearn / PyTorch / HuggingFace）
2. 模型工件格式：
   - **ONNX**[^4]（Open Neural Network Exchange）——Linux Foundation AI 项目，跨框架跨语言标准
   - **PMML**[^5]（Predictive Model Markup Language）——DMG 组织的 XML 标准，金融/传统企业主流
3. 推理侧 SDK（onnxruntime 在 Java/Go 里怎么集成）
4. **训推一致性** —— 线下训练和线上推理的特征处理逻辑必须完全一致

#### 🪤 两个必须掌握的生产故障类型 · 📎 未实操 TODO

> 以下两项**概念认识即可，未做过实操**。等做真实生产项目时回来深挖。

- [ ] **Data Leakage（特征泄露）**[^12]：训练时不小心用了"未来才能知道的字段"，离线指标完美，上线翻车
- [ ] **Concept Drift / Model Drift（模型衰退）**[^13]：上线后数据分布会变，模型准确率持续下滑，需要监控 + 定期重训

---

### 六、姿态 · 不穷尽，凑相关信号

传统编程要**穷尽** case：

```python
if 用户VIP and 余额>0 and 非黑名单 and ...:
    通过
```

漏一个分支就是 bug。

机器学习是反过来的姿态——**不需要穷尽原因，只需要凑出"够用"的相关信号**。算法会自己从这些信号里摸规律。代价是：模型永远不会 100% 准，只会"在大多数情况下大概率对"。

对习惯了命令式编程的开发者，这是最大的心智切换——**从"追求正确"到"追求够用且可监控"**。

---

### 七、位置 · LLM 时代的基础设施

**错误认知**："ChatGPT 出现了 → 传统 ML 过时了 → 做 AI 就是调 LLM API"。

**真实产业现状**：

- 银行/电商的核心风控仍在 XGBoost[^14]——亚毫秒延迟、可解释性、监管要求
- Agent 内部的路由/意图识别大量用传统 ML（LLM 太贵太慢）
- RAG 的召回层本质是向量检索 + **BM25**[^15]（1990 年代信息检索算法）
- 大模型微调用梯度下降，评估用 Precision/Recall/AUC——全是传统 ML 概念

**结论**：LLM 没消灭传统 ML，它把传统 ML 从"主角"变成了"**基础设施**"。

#### Agent 系统的真实分工

```
          ┌─────────────────────────┐
          │   LLM（大脑 / 慢系统）    │   复杂推理、对话、规划
          │   贵 / 慢 / 不可解释      │
          └────────────┬────────────┘
                       │
           ┌───────────┴───────────┐
           ↓                       ↓
   ┌───────────────┐       ┌───────────────┐
   │  传统 ML 模型  │       │   规则引擎     │
   │  小脑 / 反射   │       │   if-else     │
   │  快/便宜/可解释 │       │   写死的逻辑   │
   └───────────────┘       └───────────────┘
   路由分类、评分、           硬约束、
   过滤、检索                 合规校验
```

LLM 是"大脑"，传统 ML 是"小脑"，规则引擎是"反射"。没有"小脑"的 Agent 系统是昂贵、缓慢、不可控的玩具。

#### LLM 术语都是传统 ML 的儿子

| LLM 术语 | 出处 |
|---|---|
| LLM 微调 overfit | 过拟合[^7] |
| RAG 召回率低 | Recall（分类指标） |
| Reranker 精度不够 | Precision / AUC |
| Embedding 漂移 | Model Drift[^13] |
| 训推不一致事故 | 训推一致性 |
| 预训练数据污染 | Data Leakage[^12] |

不学传统 ML，等于没学英语字母表就读莎士比亚。

---

### 八、杠杆 · 后端开发者的 AI 切入点 · 📎 远景定位（未实操）

> **这一节是远景定位**，不是已掌握的技能。列出来是为了看清护城河在哪。

调 LLM API 的人满大街，培训班 3 个月就能批量生产。

真正稀缺的能力：**"把 sklearn 训出的模型，导出成 ONNX，在 Java Spring Boot 里加载，跑出 1ms 的 P99 延迟，扛住 10 万 QPS，还能实时监控模型衰退。"**

能力栈清单（当前状态）：

- [x] Python ML 训练（demo-01/02/03 已跑）
- [ ] 模型工件序列化（ONNX / PMML 导出）
- [x] Java/Go 后端工程能力（后端开发存量）
- [x] 高并发系统设计（存量）
- [x] 监控 / 灰度 / 回滚（存量）
- [ ] **把上面五项串起来跑通一条真实链路** —— 计划在 demo-05/06 或真实项目时做

**后端开发者切入 AI 最大的杠杆点不是"会调 LLM"，而是"懂 ML + 会工程化落地"**——传统 ML 是这个杠杆的支点。

---

### QA

#### Q1. 什么是机器学习？

Tom Mitchell 的定义：**一个程序在经验 E 的加持下，在任务 T 上的表现 P 能随经验而提升，就叫"学习"**。[^1]

工程师语言：**通过数据投喂让算法自己总结出处理逻辑的方法**。给算法一堆带答案的数据（特征 + 标签），让它通过"输入 → 输出 → 指标检查 → 修正处理逻辑"的循环，反推出一套能预测新数据的规则。这套规则被固化成一个文件，就是"模型"。

和传统编程的差别不在技术栈，在**处理逻辑的来源**——1.0 是人写的，2.0 是算法自己总结的。[^2]

#### Q2. 特征和标签之间需要有因果关系吗？

**不需要，统计相关性就够了**。ML 模型不关心为什么相关，它只关心能不能从特征里挖出预测标签的规律。但要记住：**模型输出的是相关，不是因果**。拿相关性去指导决策（"关掉客服电话就能减少流失"）会出大问题——因果推断是独立的、更难的领域。[^9]

#### Q3. label 是数据天生的属性吗？

**不是**。从物理上看 label 和 feature 都是 Excel 里的列，没区别。让某一列成为 label 的**不是数据，是工程师的意图**——声明"我要用 X 预测 y"的那一刻，y 才成为 label。

同一张表换一列当 y，整个模型的目标就变了。这就是为什么 "problem framing（问题定义）" 是 AI 工程师的核心能力。

#### Q4. 如果把所有字段都丢给模型训练，会变成无监督吗？

**不会**。监督/无监督的区分不在"给了多少列"，而在"**有没有显式指定某一列作为目标**"。[^8][^10]

```python
# 监督（指定 y）
fit(X, y)

# 无监督（没指定 y）
fit(X)
```

把 label 列和其他列一起塞进 `X` 且不指定 `y`，监督算法会退化成无监督。

#### Q5. 无监督模型分出的群没有名字，怎么用？

算法只负责**凑堆**（告诉你"群 1 有 80 万人、群 2 有 120 万人"），**人负责解读**（看每个群的特征分布，手动起名）。这个过程叫 **persona labeling**，是业务+工程师的工作。

评估上，sklearn 官方给的方案：没 ground truth 时用 Silhouette Coefficient；工业界更常用**下游 A/B 测试反推**（看转化率）。[^11]

#### Q6. 为什么训练数据和测试数据必须分开？

算法在训练时会不断调整内部参数。如果拿同一批数据做测试，它完全可以"死记硬背"得到 100% 准确率——这叫**过拟合**[^7]，上线遇到没见过的数据就翻车。测试集的唯一作用是模拟"上线遇到新数据"这件事，所以必须是模型没见过的。[^6]

#### Q7. "训练完成"在物理上意味着什么？

**把处理逻辑冻结到一个文件里**。训练前的模型内部参数一直在变，训练完成 = 参数定住 + 序列化成文件（`.pkl` / `.onnx` / `.pmml`）。正是因为"规则被固化成文件"，模型才能跨语言加载——Python 训的模型可以在 Java / Go / Node 里跑推理（ONNX[^4] / PMML[^5]）。

#### Q8. 模型和普通函数的本质区别是什么？

结构完全一致（输入 / 处理 / 输出），区别只在**处理逻辑的来源**：函数的逻辑是人手写的，模型的逻辑是从数据里被算法总结出来的。这也是 Software 1.0 与 2.0 的分水岭。[^2]

#### Q9. 在 LLM 时代学传统 ML 是不是过时了？

**没有**。LLM 把传统 ML 从「主角」变成「基础设施」。基础设施层永远需要懂底层的人。Agent 系统是「LLM + 传统 ML + 规则引擎」的混合体，不是纯 LLM。结构化数据场景（风控、推荐、评分卡、时序异常检测）XGBoost[^14] / 逻辑回归仍在吊打 LLM——更快、更准、更便宜、更可解释。

---

### 九、必扫概念 · 边界声明

> 这一节列**直播课 day-01 讲到、但本 DISTILL 主体未深入**的概念。复习时扫一遍名字，有模糊的点脚注查资料。每条格式：**名字** + 一句话 + 📖 课件出处 + 🔗 外网权威源。

#### AI / ML / DL 三者关系

三层包含：**AI ⊃ ML ⊃ DL**。AI 是目标（让机器"智能"），ML 是实现途径（从数据学规律），DL 是 ML 的神经网络分支（多层 ANN / CNN / RNN / Transformer / Mamba）。
- 📖 课件 §1 · 🔗 [Wikipedia: AI](https://en.wikipedia.org/wiki/Artificial_intelligence) · [DL](https://en.wikipedia.org/wiki/Deep_learning)

#### ML 主要应用领域

- **CV 计算机视觉**：图像识别、人脸检测
- **NLP 自然语言处理**：机器翻译、文本分类
- **音文互转**：TTS / STT
- **数据挖掘和数据分析**
- 📖 课件 §2

#### AI 发展史 4 阶段

**1950-1970 符号主义**（专家系统、1956 达特茅斯 AI 元年）→ **1980-2000 统计主义**（SVM、1997 深蓝）→ **2010s 神经网络**（2012 AlexNet、2016 AlphaGo）→ **2017-至今 大规模预训练**（Transformer、BERT、GPT、ChatGPT）。
- 📖 课件 §2 · 🔗 [Wikipedia: History of AI](https://en.wikipedia.org/wiki/History_of_artificial_intelligence)

#### AI 三要素 · 数据 / 算法 / 算力

- **数据**：教材——海量 + 清洗标注
- **算法**：解决方案——论文和模型
- **算力**：引擎——CPU（I/O 密集）/ GPU（计算密集、训练主力）/ NPU（边缘推理）/ TPU（Google 大型网络训练专用）
- 📖 课件 §2

#### 完整 5 类算法分类

DISTILL 二节只讲了前两类。完整分类：

| 类别 | 有无标签 | 典型场景 |
|---|---|---|
| 有监督 Supervised | 有完整标签 | 分类 / 回归（已讲 ✅）|
| 无监督 Unsupervised | 无标签 | 聚类 / 降维（已讲 ✅）|
| **半监督 Semi-supervised** | 少量标签 + 大量无标签 | 降低标注成本 |
| **自监督 Self-supervised** | 从数据自身造标签 | **LLM 预训练基础** |
| **强化学习 RL** | 环境反馈奖励 | AlphaGo / 自动驾驶 / RLHF |

- 📖 课件 §4 · 🔗 [Semi-supervised](https://en.wikipedia.org/wiki/Weak_supervision) · [Self-supervised](https://en.wikipedia.org/wiki/Self-supervised_learning) · [RL](https://en.wikipedia.org/wiki/Reinforcement_learning)

#### IID 独立同分布 · 训练集/测试集核心原则

训练集和测试集必须 **Independent and Identically Distributed**：
- **独立**：测试集数据**不能出现在**训练集里（否则等于"考过的原题"）
- **同分布**：训练和测试数据来自**同一概率分布**（光照 / 时段 / 场景一致）

违反 IID 的典型事故：时序数据"未来"漏进训练集；训练集是爬虫数据 / 测试集是真实用户。
- 📖 课件 §3 · 🔗 [Wikipedia: IID random variables](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)

#### 奥卡姆剃刀 · 模型简化原则

**"如无必要，勿增实体"**——给定两个**泛化能力相同**的模型，选**更简单**的那个。对应 bias-variance tradeoff：模型太复杂 → 过拟合 → 噪声占主导。
- 📖 课件 §7 · 🔗 [Wikipedia: Occam's razor](https://en.wikipedia.org/wiki/Occam%27s_razor)

#### 建模流程 6 步

```
获取数据 → 数据基本处理 → 特征工程 → 训练 → 评估 → 上线/预测
    ↑           ↑              ↑
  匹配业务   异常值/缺失值    最耗时（80%）
```
- 📖 课件 §5

#### 数据基本处理 · 异常值 / 缺失值

- **异常值**：离群点、记录错误——**删除** or **修正**
- **缺失值**：空字段——**删除**（`dropna`）or **填充**（`fillna`，均值 / 中位数 / 业务默认值）
- 📖 课件 §5

#### "数据和特征决定上限" · ML 经典格言

> **数据和特征决定了机器学习的上限，模型和算法只是逼近这个上限而已。**

换算法通常只能提升几个百分点；换好特征 / 更干净数据 能提升一大档。
- 📖 课件 §6

---

**算法级必扫**（散到各算法章的边界声明 / KNOWLEDGE-MAP）：
- `01-linear-regression`：Ridge / Lasso / ElasticNet、多重共线性、SGD 变体
- `02-logistic-regression`：Log-Loss、MLE、Softmax、ROC/AUC、Calibration、One-Hot
- `03-knn`：K 值选型、距离度量、Wine 标准化、Lazy vs Eager、ANN / Vector DB / Embedding（详见 [`../03-knn/02-KNOWLEDGE-MAP.txt`](../03-knn/02-KNOWLEDGE-MAP.txt)）
- `04-09` 待开讲章节各自管理

**工具与生态**：
- **环境**：scikit-learn / Anaconda / Jupyter / PyCharm
- **资源**：[Arxiv](https://arxiv.org/) · [Papers With Code](https://paperswithcode.com/) · [HuggingFace](https://huggingface.co/)
- **顶会**：CVPR / ECCV / ICCV（CV）· NeurIPS / ICML / ICLR / AAAI（通用）· ACL / EMNLP（NLP）

> 📎 分散策略的**设计说明**见 [`01-LESSON-PLAN.md` §8](./01-LESSON-PLAN.md)。

---

### 学习轨迹（内部溯源）

这份文档不是理论综述，是基于真实 Q&A 通关过的内容蒸馏而来。以下内部文档记录了每个论断在哪次 session 被验证、学习者用自己的话如何表达：

- [learning-sessions/2026-04-07-session-01.md](../../learning-sessions/2026-04-07-session-01.md) — Q1 模型=函数、Q2 训练 vs 推理。含「函数的处理逻辑是人写的，模型的处理逻辑是算法自己总结的」原话出处，及"数据投喂 + 指标检查 + 修正处理逻辑"（即 training loop 的自主重建）
- [learning-sessions/2026-04-07-session-03.md](../../learning-sessions/2026-04-07-session-03.md) — Q3 特征 vs 标签（房价场景）、Q D 监督 vs 无监督（电商分群）、狗头玩笑触发的"label 是工程师的选择"洞察
- [highlights/2026-04-07-ML-入门---特征-标签-监督学习.md](../../highlights/2026-04-07-ML-入门---特征-标签-监督学习.md) — session-03 完整对话原文
- [00-mental-model/01-LESSON-PLAN.md](./01-LESSON-PLAN.md) — 阶段一教学大纲（合并自原 00-Orientation + 01-Overview-for-Developers）：学习方法论、A/B/C 必记清单、RDBMS 术语映射、部署三架构、特征工程五分支

---

[^1]: **Tom Mitchell 机器学习定义** — [Wikipedia: Machine learning](https://en.wikipedia.org/wiki/Machine_learning)
[^2]: **Software 2.0 范式** — [Andrej Karpathy · "Software 2.0" (Medium, 2017)](https://karpathy.medium.com/software-2-0-a64152b37c35)
[^3]: **sklearn 监督学习 API 和算法清单** — [scikit-learn docs: Supervised learning](https://scikit-learn.org/stable/supervised_learning.html)
[^4]: **ONNX（Open Neural Network Exchange）标准** — [onnx.ai](https://onnx.ai/)
[^5]: **PMML（Predictive Model Markup Language）标准** — [Data Mining Group · PMML](https://dmg.org/pmml/)
[^6]: **训练/验证/测试集三元拆分** — [Wikipedia: Training, validation, and test data sets](https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets)
[^7]: **过拟合（Overfitting）** — [Wikipedia: Overfitting](https://en.wikipedia.org/wiki/Overfitting)
[^8]: **监督学习 + feature/label 关系（含 `x_i` / `y_i` 数学形式化）** — [Wikipedia: Supervised learning](https://en.wikipedia.org/wiki/Supervised_learning)
[^9]: **相关 ≠ 因果（含冰淇淋/溺水混淆变量例子）** — [Wikipedia: Correlation does not imply causation](https://en.wikipedia.org/wiki/Correlation_does_not_imply_causation)
[^10]: **无监督学习** — [Wikipedia: Unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning)
[^11]: **聚类评估（含 Silhouette Coefficient 无 ground truth 方案）** — [scikit-learn docs: Clustering performance evaluation](https://scikit-learn.org/stable/modules/clustering.html)
[^12]: **Data Leakage（特征泄露）** — [Wikipedia: Leakage (machine learning)](https://en.wikipedia.org/wiki/Leakage_(machine_learning))
[^13]: **Concept Drift / Model Drift（模型衰退）** — [Wikipedia: Concept drift](https://en.wikipedia.org/wiki/Concept_drift)
[^14]: **XGBoost（工业界主力表格数据模型）** — [xgboost.ai](https://xgboost.ai/)
[^15]: **BM25（RAG 召回层的经典信息检索算法）** — [Wikipedia: Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
