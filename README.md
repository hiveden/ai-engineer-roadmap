# AI Engineer Roadmap

面向开发与架构师的 AI 落地指南。受众、红线、协作约定见 [`CLAUDE.md`](./CLAUDE.md)。

---

## 🚪 从这里开始

**当前状态**：v0 自学方案已验证失败（详见 [`_v0/sessions/2026-04-26-session-13.md`](./_v0/sessions/2026-04-26-session-13.md)），仓库正以 [`机器学习学习框架.md`](./机器学习学习框架.md) 为试行框架重构。第一根算法柱子从 KNN 起步，3 个算法验证后固化为一期架构。

阅读顺序：

1. 👉 [`机器学习学习框架.md`](./机器学习学习框架.md) — 唯一方法论权威（横向四层 + 纵向柱子 + 三子框架）
2. [`CLAUDE.md`](./CLAUDE.md) — 协作协议
3. [`_v0/README.md`](./_v0/README.md) — 失败归档说明（不参考其规则）

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

### 阶段一：传统机器学习的工程化视角 · v1 试行中

按 [`机器学习学习框架.md`](./机器学习学习框架.md) 的横向四层 + 纵向柱子框架重构。

**纵向柱子（算法）—— 试行 3 根后固化框架**：

* [ ] 第 1 根 · KNN（重启）
* [ ] 第 2 根 · 待 v1 框架推进时定
* [ ] 第 3 根 · 待 v1 框架推进时定
* [ ] 4-N 根 · 框架固化后展开

**横向四层（最小启动包随柱子追加，不一次建完）**：

* [ ] 数学语言层（线代 / 微积分 / 概率统计）
* [ ] 特征工程层（数据预处理 / 特征构造 / 选择 / 降维）
* [ ] 模型训练层（损失函数 / 优化方法 / 训练流程 / 正则化）
* [ ] 模型评估层（评估指标 / 误差诊断 / 模型对比）

横向具体放置位置（仓库顶层独立 vs ML 子层）待 v1 试行期间敲定。

### 阶段二：深度学习与跨语言部署落地 ⏳

* [ ] 01 - PyTorch：张量与计算图作为带状态系统组件
* [ ] 02 - CNN / RNN 架构解构
* [ ] 03 - 模型部署：Python 原型 → Go/Java/Node 毫秒级本地加载（ONNX Runtime）

### 阶段三：NLP 与大模型前置 ⏳

* [ ] 01 - 词嵌入（Embedding）
* [ ] 02 - Transformer 架构
* [ ] 03 - 迁移学习：FastText → BERT/GPT

### 阶段四：LLM / RAG / Agent 实战 ⏳

* [ ] 01 - Prompt Engineering
* [ ] 02 - 实战：RAG 智能简历推荐（ES + Milvus + BGE-M3）
* [ ] 03 - AI Agent：Function Calling + 后端 CRUD 集成
* [ ] 04 - 实战：Multi-Agent + MCP 协议

### 阶段五 / 六：推理优化 / 知识图谱 ⏳

阶段四完成后再细化设计。

---

## 📐 仓库哲学

> 学习仓库，不是软件仓库。Git 是"已验证学习进度"的实时投影——只有真正学过、迭代过的内容才入 git。

1. 你 clone 下来看到的 = 作者实际走过的。没看到的章节是还没学到
2. 草稿 / 原始素材不入 git（`assets/` 已 gitignored）
3. `[x]` 的章节内容真的进了 git；`[ ]` 的还在草稿态
4. 每次学习会话产生一份结构化报告 → `learning-sessions/`，作为元认知工具
5. 验证失败的方案归档进 `_v0/`，**不参考其规则**

---

## 🛠️ 项目结构

```
ai-engineer-roadmap/
├── README.md                      # 本文件 · 全局导航
├── CLAUDE.md                      # 给 Claude Code 的协作协议
├── 机器学习学习框架.md              # 唯一方法论权威（v1 试行）
│
├── 01-ML/                          # 阶段一主线（v1 试行中，从 KNN 起步）
│
├── learning-sessions/              # 元认知工具 · 学习会话日志（重启编号）
│
├── _v0/                           # v0 失败归档 · 不参考其规则
│   ├── README.md                  #   归档说明
│   ├── 01-ML/                     #   旧 ML 工作区
│   ├── sessions/                  #   session-01~13
│   ├── extracts/                  #   pptx 提取临时产物
│   └── grading-rules.md           #   L0-L4 评分表（已被二元判停取代）
│
├── scripts/                       # 工具脚本（pptx 提取 / session 导出）
└── assets/                        # 原始素材（gitignored）
```

文件命名约定：`NN-Kebab-Case-Title.md`（NN 为阶段内序号，从 01 起）。
