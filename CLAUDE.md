# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库性质

这是一份**写作型仓库**（Markdown 笔记 + 少量 code-demos），不是软件项目。没有 build/test/lint 流水线。产物是《AI Engineer Roadmap》——把传统 AI 培训班课件重构为面向**全栈开发工程师 / 系统架构师**的工程化讲义。

## 目标受众与红线（最重要）

**唯一受众**：拥有多年 Java / Go / Node.js / React 架构经验、要把 AI 作为黑盒组件集成到企业级系统的全栈工程师与架构师。

**绝对禁止的越界行为**（来自 GEMINI.md，等同于 CLAUDE 行为约束）：
- 不写偏导数、梯度下降、损失函数等学术数学推导
- 不堆砌算法岗面试冷僻知识点
- 不培养"算法工程师 / 研究员"
- 不写论文风格的理论铺陈

**必须做的事**：
- 用工程师熟悉的语言解释概念——微服务、数据库、API、内存管理、GC、序列化、零拷贝、ONNX、长连接
- 强调跨语言落地：Python 训练 → ONNX/PMML 导出 → Go/Java/Node.js 加载推理
- 对比 Software 1.0（rule-based）vs Software 2.0（data-driven），回答"在现有架构里它能替换什么"
- 直击高并发、显存/GC 博弈、流式 SSE、Apache Arrow 列式数据等架构痛点

## 写作风格

- 资深架构师 / Tech Lead 语气：专业、直接、高信息密度，少废话
- 开始修改文件前，**用一句话简述修改策略**
- 输出纯 Markdown，存到对应阶段目录
- 中文为主，技术术语和代码标识符保留原文

## 目录结构

```
01-Machine-Learning-Foundation/   # 阶段一：传统 ML 的工程化视角
02-Deep-Learning-Engineering/     # 阶段二：DL 与跨语言部署
03-NLP-and-Transformer/           # 阶段三：NLP / Transformer 前置
04-LLM-and-Agent-Architecture/    # 阶段四：LLM / RAG / Agent 实战
05-Model-Inference-and-Optimization/
06-Knowledge-Graph-and-IE/
assets/source-materials/          # 原始培训班课件（pptx/pdf/md），仅作输入素材
<阶段>/code-demos/                # 跨语言实战代码片段（Python → Go/Java/Node/React）
```

文件命名：`NN-Kebab-Case-Title.md`（NN 为该阶段内序号）。`README.md` 是全局学习路线总纲——新增/重构章节后**必须同步更新 README 的目录复选框**。

## 重构工作流

每节内容的典型处理流程：
1. 用户指向 `assets/source-materials/` 下的某节原始课件
2. 剔除学术冗余 → 用工程类比重写概念
3. 补充该模型/算法在实际业务架构中的位置（特征工程管道、风控服务、推荐系统、Agent 工具节点等）
4. 加入跨语言落地路径（导出格式、运行时、并发模型）
5. 写入对应阶段目录，更新 `README.md` 状态

## code-demos 约定

- Python demos 配 `requirements.txt`，确定性可运行
- 目的不是教学完整训练，而是**展示集成点**：模型加载、推理调用、跨语言导出
