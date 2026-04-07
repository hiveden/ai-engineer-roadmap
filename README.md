# AI Engineer Roadmap (面向全栈开发与架构师的 AI 落地指南)

**【适用对象与目标定位】**
本项目**绝非**为算法工程师、发论文的研究员、或者底层算子开发者准备的数学/统计学宝典。
本指南的唯一受众是：**拥有多年后端/前端架构经验（Java、Go、Node.js、React 等），致力于将 AI 大模型、向量数据库和深度学习组件无缝接入现有千万级高并发系统的 AI 应用全栈开发工程师与系统架构师。**

我们将抛弃传统学术派死磕偏导数和激活函数的模式，直接切入核心：**如何跨越内存边界与生态重力，将 AI 模型作为黑盒微服务或智能 API，安全、极速地集成到现有的企业级技术栈中。**

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

## 🗺️ 课程重构目录

本指南将传统培训班的冗长课程体系，浓缩重构为四大工程化模块：

### 阶段一：传统机器学习的工程化视角 (Machine Learning Foundation)
> **核心目标：** 理解模型的本质是“黑盒函数”；掌握特征工程与列式数据流转；学会跨语言的模型导出（ONNX）与服务加载。

* 👉 **[00 - 起点定向：在动手学 ML 之前必须先看这一篇](./01-Machine-Learning-Foundation/00-Orientation.md)** （含自评表 + 问题停车场）
* [x] [01 - 机器学习概述：开发者的降维指南](./01-Machine-Learning-Foundation/01-Overview-for-Developers.md)
* [x] [02 - KNN 与现代向量检索（Vector Search）的底层逻辑](./01-Machine-Learning-Foundation/02-KNN-and-Vector-Search.md)
* [x] [03 - 线性回归工程化：从加权累加器到 ONNX 导出](./01-Machine-Learning-Foundation/03-Linear-Regression-Engineering.md)
* [x] [04 - 逻辑回归与分类指标：阈值博弈与 AUC 物理意义](./01-Machine-Learning-Foundation/04-Logistic-Regression-and-Classification-Metrics.md)
* [x] [05 - 决策树与规则抽取：可解释性是金融风控的硬通货](./01-Machine-Learning-Foundation/05-Decision-Trees-and-Rule-Extraction.md)
* [x] [06 - 集成学习（Ensemble）：从随机森林到 XGBoost 的风控霸权](./01-Machine-Learning-Foundation/06-Ensemble-and-Financial-Risk.md)
* [x] [07 - 朴素贝叶斯与特征降维 (PCA)：通往大模型时代的序曲](./01-Machine-Learning-Foundation/07-PCA-Bayes-and-LLM-Prelude.md)
* [x] [08 - 阶段一收官实战：聚类算法与时间序列工程落地](./01-Machine-Learning-Foundation/08-Clustering-and-Time-Series-Project.md)

### 阶段二：深度学习与跨语言部署落地 (Deep Learning Engineering)
> **核心目标：** 将神经网络视为带有状态的架构组件；全栈工程师必懂的 Python 内存防坑指南；掌握 PyTorch 张量操作与模型工程化落地。

* [x] [01 - AI 数据流基石：全栈工程师的 Numpy 与 Pandas 内存防坑指南](./02-Deep-Learning-Engineering/01-Numpy-Pandas-Memory-Traps.md)
* [x] [02 - PyTorch 极速上手：将张量（Tensor）与计算图视为带状态的系统组件](./02-Deep-Learning-Engineering/02-PyTorch-Stateful-Tensors.md)
* [x] [03 - CNN 与 RNN 架构解构：图像特征滑动窗口与时序微服务的状态管理](./02-Deep-Learning-Engineering/03-CNN-and-RNN-Architectures.md)
* [x] [04 - 模型部署实战：从 Python 原型到 Go/Java/Node 的毫秒级本地加载 (ONNX Runtime)](./02-Deep-Learning-Engineering/04-ONNX-Runtime-Deployment.md)

### 阶段三：自然语言处理与大模型前置知识 (NLP and Transformer)
> **核心目标：** 搞懂计算机如何理解人类语言；彻底理解 Embedding（词向量）与 Transformer 架构原理；理解预训练生态的演进史。

* [x] [01 - 词嵌入（Embedding）：将文本转化为多维浮点数数组](./03-NLP-and-Transformer/01-Embedding-and-Tokenization.md)
* [x] [02 - Transformer 架构解密：大语言模型（LLM）跳动的心脏](./03-NLP-and-Transformer/02-Transformer-Architecture.md)
* [ ] 03 - 迁移学习与预训练模型：从 FastText 极速分类到 BERT/GPT 的生态演进

### 阶段四：大语言模型、RAG 与 Agent 实战 (LLM and Agent Architecture)
> **核心目标：** 掌握当前时代全栈开发者最大的红利区——企业级大模型应用与 AI 网关架构开发。

* [ ] 01 - Prompt Engineering：将自然语言作为代码的一部分
* [ ] 02 - 实战：基于 RAG (检索增强生成) 的 SmartRecruit 智能简历推荐系统（整合 Elasticsearch、Milvus 向量库与 BGE-M3 模型）
* [ ] 03 - AI Agent 开发：Function Calling 让模型自动调用你的后端 CRUD 接口
* [ ] 04 - 实战：基于 Multi-Agent 与 MCP 协议的 SmartVoyage 分布式智能旅行助手（多代理协同架构落地）

---

## 🛠️ 项目结构

- `README.md`：全局学习路线与核心架构范式（当前文件）。
- `各阶段目录`：存放重构后的高信息密度 Markdown 笔记和架构讲义。
- `code-demos/`：存放跨语言（Python -> Go/Java/Node/React）打通的实战代码片段。