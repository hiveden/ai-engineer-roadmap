# AI Engineer Roadmap

面向全栈开发与架构师的 AI 落地指南。受众、红线、协作约定见 [`CLAUDE.md`](./CLAUDE.md)。

---

## 🚪 从这里开始

第一次打开这个仓库：

1. 👉 [00 - 起点定向](./01-Machine-Learning-Foundation/00-Orientation.md) — 动手学 ML 之前的元学习指南
2. [01 - 机器学习概述](./01-Machine-Learning-Foundation/01-Overview-for-Developers.md) — 开发者视角的 ML 定义与工程落地路径
3. 进 [`demos/`](./demos/) 做 10 节算法实操课（每节一个 demo + 一份 `LESSON-PLAN.md`）

---

## 🎯 核心理念：Software 2.0 架构范式

1. **语言即职责**
   - Python：研发实验室与数据管道（Pandas 深浅拷贝的 OOM 陷阱、高吞吐预处理服务）
   - Go / Java / Node.js：AI 基础设施与业务网关（ONNX Runtime 实现零网络 I/O 本地推理）
   - React / 前端：Agent 状态可视化、WebGPU 端侧推理

2. **跨越物理与数据边界**
   - 显存 vs GC：避免 JVM/V8 的 STW 长尾延迟
   - 零拷贝：抛弃 JSON/REST，拥抱 Apache Arrow / Parquet 列式格式

3. **持久化执行**：AI Agent 动辄几分钟的长周期调用，需要强状态的分布式工作流引擎

---

## 🗺️ 学习路线

> 进度追踪以「内容是否已纳入 git」为准 — 见下文「仓库哲学」。
> 阶段二以后的章节**尚未经过实际学习验证**，只是本地草稿（`assets/plan/`，gitignored）。

### 阶段一：传统机器学习的工程化视角 · 进行中（2/10）

核心目标：理解模型 = 黑盒函数；掌握特征工程与列式数据流转；学会跨语言模型导出（ONNX）与服务加载。

* 👉 [00 - 起点定向](./01-Machine-Learning-Foundation/00-Orientation.md)
* [x] [01 - 机器学习概述](./01-Machine-Learning-Foundation/01-Overview-for-Developers.md)

**10 节算法实操课**（[详见 `demos/`](./demos/README.md)）：

* [x] 第 1 节 · 线性回归 — [`demos/demo-01-house-price/`](./demos/demo-01-house-price/)（加州房价 / 回归）
* [x] 第 2 节 · 逻辑回归 — [`demos/demo-02-breast-cancer/`](./demos/demo-02-breast-cancer/)（乳腺癌 / 二分类）
* [ ] 第 3 节 · KNN — [`demos/02-KNN-and-Vector-Search.md`](./demos/02-KNN-and-Vector-Search.md)
* [ ] 第 4 节 · 决策树 — [`demos/05-Decision-Trees-and-Rule-Extraction.md`](./demos/05-Decision-Trees-and-Rule-Extraction.md)
* [ ] 第 5/6 节 · 随机森林 / XGBoost — [`demos/06-Ensemble-and-Financial-Risk.md`](./demos/06-Ensemble-and-Financial-Risk.md)
* [ ] 第 7/8 节 · 朴素贝叶斯 / PCA — [`demos/07-PCA-Bayes-and-LLM-Prelude.md`](./demos/07-PCA-Bayes-and-LLM-Prelude.md)
* [ ] 第 9/10 节 · K-Means / 时间序列 — [`demos/08-Clustering-and-Time-Series-Project.md`](./demos/08-Clustering-and-Time-Series-Project.md)

### 阶段二：深度学习与跨语言部署落地 ⏳

* [ ] 01 - Numpy / Pandas 内存防坑指南
* [ ] 02 - PyTorch：张量与计算图作为带状态系统组件
* [ ] 03 - CNN / RNN 架构解构
* [ ] 04 - 模型部署：Python 原型 → Go/Java/Node 毫秒级本地加载（ONNX Runtime）

### 阶段三：NLP 与大模型前置 ⏳

* [ ] 01 - 词嵌入（Embedding）
* [ ] 02 - Transformer 架构
* [ ] 03 - 迁移学习：FastText → BERT/GPT

### 阶段四：LLM / RAG / Agent 实战 ⏳

* [ ] 01 - Prompt Engineering
* [ ] 02 - 实战：RAG 智能简历推荐（ES + Milvus + BGE-M3）
* [ ] 03 - AI Agent：Function Calling + 后端 CRUD 集成
* [ ] 04 - 实战：Multi-Agent + MCP 协议（SmartVoyage 旅行助手）

### 阶段五 / 六：推理优化 / 知识图谱 ⏳

阶段四完成后再细化设计。

---

## 📐 仓库哲学

> 学习仓库，不是软件仓库。Git 是"已验证学习进度"的实时投影——只有真正学过、迭代过的内容才入 git。

1. 你 clone 下来看到的 = 作者实际走过的。没看到的章节是还没学到
2. 草稿 / 原始素材不入 git（`assets/plan/`、`assets/source-materials/` 均 gitignored）
3. `[x]` 的章节内容真的进了 git；`[ ]` 的还在草稿态
4. 每次学习会话产生一份结构化报告 → `learning-sessions/`，作为元认知工具

---

## 🛠️ 项目结构

```
ai-engineer-roadmap/
├── README.md                          # 本文件 · 全局导航
├── CLAUDE.md                          # 给 Claude Code 的项目指令
├── GEMINI.md                          # 给 Gemini CLI 的项目指令
│
├── 01-Machine-Learning-Foundation/    # 阶段一主线（元指南 + 总览）
│   ├── 00-Orientation.md              #   起点定向（必先读）
│   ├── 01-Overview-for-Developers.md  #   开发者视角总览
│   └── COURSE-MAPPING.md              #   直播课 ↔ 仓库双时间线
│
├── demos/                             # 10 节算法实操课
│   ├── README.md                      #   L3 定义 + 10 算法清单
│   ├── demo-01-house-price/           #   第 1 节 · 线性回归
│   │   ├── LESSON-PLAN.md
│   │   └── code/                      #     step1~4 + run.py
│   ├── demo-02-breast-cancer/         #   第 2 节 · 逻辑回归
│   │   ├── LESSON-PLAN.md
│   │   ├── NOTES.md                   #     置信度专题
│   │   └── code/                      #     step1~4
│   └── 02/05/06/07/08-*.md            #   待做 demo 的讲义资料
│
├── learning-sessions/                 # 元认知工具 · 学习会话日志
│   ├── README.md                      #   日志机制说明
│   ├── _template.md                   #   报告模板
│   └── YYYY-MM-DD-session-NN.md
│
├── highlights/                        # 触发到我的对话原文存档
│   ├── README.md
│   └── YYYY-MM-DD-<title>.md
│
└── assets/                            # 全部 gitignored
    ├── source-materials/              #   原始培训班素材
    └── plan/                          #   未学习的章节草稿
```

文件命名约定：`NN-Kebab-Case-Title.md`（NN 为阶段内序号，从 01 起）。
