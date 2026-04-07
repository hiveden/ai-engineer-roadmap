# AI Engineer Roadmap (面向全栈开发与架构师的 AI 落地指南)

**【适用对象与目标定位】**
本项目**绝非**为算法工程师、发论文的研究员、或者底层算子开发者准备的数学/统计学宝典。
本指南的唯一受众是：**拥有多年后端/前端架构经验（Java、Go、Node.js、React 等），致力于将 AI 大模型、向量数据库和深度学习组件无缝接入现有千万级高并发系统的 AI 应用全栈开发工程师与系统架构师。**

我们将抛弃传统学术派死磕偏导数和激活函数的模式，直接切入核心：**如何跨越内存边界与生态重力，将 AI 模型作为黑盒微服务或智能 API，安全、极速地集成到现有的企业级技术栈中。**

---

## 🚪 从这里开始

如果你是第一次打开这个仓库，**请严格按这个顺序读**：

1. 👉 **[00 - 起点定向](./01-Machine-Learning-Foundation/00-Orientation.md)** — 在动手学 ML 之前**必须**先看的元学习指南
   - 时代语境（为什么 LLM 时代还要学传统 ML）
   - 学习方法（4 层路径 + 必背点 + 验证 demo 阶梯）
   - 自评表（动态发现自己的知识盲区）
   - 问题停车场（持续追加你的"我不懂"）
2. 然后再读阶段一的具体章节（01-08）
3. **不要跳级**——本指南的整个学习曲线是为"自下而上"设计的，跳过 00 直接读 01 会看不出味道

---

## 🎯 核心理念：Software 2.0 时代的架构范式

1. **语言即职责（Role-based Runtime）**：
   - **Python**：研发实验室与数据管道（Data Pipeline）。全栈工程师必须跨越 Pandas 深浅拷贝带来的 OOM（内存溢出）陷阱，写出高吞吐的预处理服务。
   - **Go / Java / Node.js**：AI 基础设施与业务网关。负责高并发长连接（SSE）、流式转发与智能代理编排，利用 ONNX Runtime 实现零网络 I/O 的毫秒级本地推理。
   - **React / 前端**：智能交互与端侧推理。负责 Agent 状态可视化与基于 WebGPU 的端侧直接计算。

2. **跨越物理与数据边界**：
   - **显存与 GC（垃圾回收）的博弈**：理解大规模 Tensor 处理时的内存布局，避免 JVM/V8 引擎的长尾延迟（STW/OOM）。
   - **零拷贝（Zero-copy）数据流转**：抛弃低效的 JSON/REST 序列化，拥抱 Apache Arrow/Parquet 列式内存格式。

3. **状态管理与持久化执行**：
   - 应对 AI Agent 动辄几分钟的长周期、多步调用，构建强状态的分布式工作流引擎。

---

## 🗺️ 学习路线

本指南将传统培训班的冗长课程体系，浓缩重构为四大工程化模块。**进度追踪以「内容是否已纳入 git」为准**——见下文「仓库哲学」。

### 阶段一：传统机器学习的工程化视角 ✅ 内容齐全

> **核心目标：** 理解模型的本质是"黑盒函数"；掌握特征工程与列式数据流转；学会跨语言的模型导出（ONNX）与服务加载。

* 👉 **[00 - 起点定向：在动手学 ML 之前必须先看这一篇](./01-Machine-Learning-Foundation/00-Orientation.md)**
* [x] [01 - 机器学习概述：开发者的降维指南](./01-Machine-Learning-Foundation/01-Overview-for-Developers.md)
* [x] [02 - KNN 与现代向量检索（Vector Search）的底层逻辑](./01-Machine-Learning-Foundation/02-KNN-and-Vector-Search.md)
* [x] [03 - 线性回归工程化：从加权累加器到 ONNX 导出](./01-Machine-Learning-Foundation/03-Linear-Regression-Engineering.md)
* [x] [04 - 逻辑回归与分类指标：阈值博弈与 AUC 物理意义](./01-Machine-Learning-Foundation/04-Logistic-Regression-and-Classification-Metrics.md)
* [x] [05 - 决策树与规则抽取：可解释性是金融风控的硬通货](./01-Machine-Learning-Foundation/05-Decision-Trees-and-Rule-Extraction.md)
* [x] [06 - 集成学习（Ensemble）：从随机森林到 XGBoost 的风控霸权](./01-Machine-Learning-Foundation/06-Ensemble-and-Financial-Risk.md)
* [x] [07 - 朴素贝叶斯与特征降维 (PCA)：通往大模型时代的序曲](./01-Machine-Learning-Foundation/07-PCA-Bayes-and-LLM-Prelude.md)
* [x] [08 - 阶段一收官实战：聚类算法与时间序列工程落地](./01-Machine-Learning-Foundation/08-Clustering-and-Time-Series-Project.md)

### 阶段二：深度学习与跨语言部署落地 ⏳ 计划中

> **核心目标：** 将神经网络视为带有状态的架构组件；全栈工程师必懂的 Python 内存防坑指南；掌握 PyTorch 张量操作与模型工程化落地。
>
> ⚠️ 本阶段章节**尚未经过实际学习验证**，目前只是本地草稿（位于本地的 `assets/plan/`，未纳入 git）。等学到该阶段时会从草稿升级为正式章节并合入。

* [ ] 01 - AI 数据流基石：全栈工程师的 Numpy 与 Pandas 内存防坑指南
* [ ] 02 - PyTorch 极速上手：将张量（Tensor）与计算图视为带状态的系统组件
* [ ] 03 - CNN 与 RNN 架构解构：图像特征滑动窗口与时序微服务的状态管理
* [ ] 04 - 模型部署实战：从 Python 原型到 Go/Java/Node 的毫秒级本地加载 (ONNX Runtime)

### 阶段三：自然语言处理与大模型前置知识 ⏳ 计划中

> **核心目标：** 搞懂计算机如何理解人类语言；彻底理解 Embedding（词向量）与 Transformer 架构原理；理解预训练生态的演进史。
>
> ⚠️ 同上，本阶段章节尚未学习验证。

* [ ] 01 - 词嵌入（Embedding）：将文本转化为多维浮点数数组
* [ ] 02 - Transformer 架构解密：大语言模型（LLM）跳动的心脏
* [ ] 03 - 迁移学习与预训练模型：从 FastText 极速分类到 BERT/GPT 的生态演进

### 阶段四：大语言模型、RAG 与 Agent 实战 ⏳ 计划中

> **核心目标：** 掌握当前时代全栈开发者最大的红利区——企业级大模型应用与 AI 网关架构开发。
>
> ⚠️ 同上，本阶段章节尚未学习验证。

* [ ] 01 - Prompt Engineering：将自然语言作为代码的一部分
* [ ] 02 - 实战：基于 RAG (检索增强生成) 的 SmartRecruit 智能简历推荐系统（整合 Elasticsearch、Milvus 向量库与 BGE-M3 模型）
* [ ] 03 - AI Agent 开发：Function Calling 让模型自动调用你的后端 CRUD 接口
* [ ] 04 - 实战：基于 Multi-Agent 与 MCP 协议的 SmartVoyage 分布式智能旅行助手（多代理协同架构落地）

### 阶段五 / 六：模型推理优化 / 知识图谱 ⏳ 远期计划

> 这两个阶段的具体章节会在阶段四完成后再细化设计——避免过早规划。

---

## 📐 仓库哲学

> **这是一个学习仓库，不是软件仓库。**
> **Git 是"已验证学习进度"的实时投影——只有真正学过、迭代过的内容才入 git。**

这个原则意味着：

1. **你 clone 下来看到的内容 = 作者实际"走过"的内容**——没看到的章节是因为还没学到那一步，不是"功能不全"
2. **草稿和原始素材不入 git**：未经验证的章节草稿放在本地的 `assets/plan/`，培训班原始素材放在本地的 `assets/source-materials/`，两者都 gitignored
3. **README 上 `[x]` 的章节 = 内容真的进了 git 可点击；`[ ]` 的章节 = 还在草稿态**
4. **每完成一次学习会话**，就在 `learning-sessions/` 下生成一份结构化报告，作为元认知工具

这套哲学的好处：**仓库永远是诚实的、永远是干净的、永远是可发布的**。

---

## 📚 学习会话日志

`learning-sessions/` 目录是本项目的**元认知工具**——它存的不是"学了什么"（那是各 stage 章节的事），而是"**怎么学的**" + "**学的过程中我注意到了自己什么**"。

每场学习会话结束时生成一份结构化报告，含 Q&A 原文、纠错过程、暴露的盲区、自评分变化、下次起点，最后留一段「💡 元反思」由学习者亲自填写。

> 👉 [`learning-sessions/README.md`](./learning-sessions/README.md) 解释完整机制
> 👉 [`learning-sessions/_template.md`](./learning-sessions/_template.md) 是模板
> 👉 [`learning-sessions/2026-04-07-session-01.md`](./learning-sessions/2026-04-07-session-01.md) 是第一份真实样本

---

## 🛠️ 项目结构

```
ai-engineer-roadmap/
├── README.md                          # 本文件 · 全局导航
├── CLAUDE.md                          # 给 Claude Code 的项目指令
├── GEMINI.md                          # 给 Gemini CLI 的项目指令
├── .gitignore                         # 草稿/素材/IDE 配置一律排除
│
├── 01-Machine-Learning-Foundation/    # ✅ 阶段一 · 已纳入 git
│   ├── 00-Orientation.md              #   ⭐ 起点定向（必先读）
│   ├── 01-Overview-for-Developers.md
│   ├── 02-KNN-and-Vector-Search.md
│   ├── 03-Linear-Regression-Engineering.md
│   ├── 04-Logistic-Regression-and-Classification-Metrics.md
│   ├── 05-Decision-Trees-and-Rule-Extraction.md
│   ├── 06-Ensemble-and-Financial-Risk.md
│   ├── 07-PCA-Bayes-and-LLM-Prelude.md
│   └── 08-Clustering-and-Time-Series-Project.md
│
├── learning-sessions/                 # ✅ 元认知工具 · 已纳入 git
│   ├── README.md
│   ├── _template.md
│   └── YYYY-MM-DD-session-NN.md
│
└── assets/                            # 🚫 全部 gitignored
    ├── source-materials/              #   原始培训班素材（输入，不公开）
    └── plan/                          #   未学习的章节草稿（待升级）
        ├── 02-Deep-Learning-Engineering/
        ├── 03-NLP-and-Transformer/
        ├── 04-LLM-and-Agent-Architecture/
        ├── 05-Model-Inference-and-Optimization/
        └── 06-Knowledge-Graph-and-IE/
```

文件命名约定：`NN-Kebab-Case-Title.md`（NN 为该阶段内序号，从 01 起）。
