# ML 视频生产工作流

> **总入口** · 教材 → 视频成片。任何 ML 章节（KNN / LR / LogReg / DT / ...）通用

```
1.讲解+Demo → 2.拆分 → 3.写脚本 → 3.5.实测复核 → 4.TTS → 5.调布局 → 6.录屏 → 7.预览审核 → 8.产物
  人工+Agent    人工     Agent      人工+Agent      外部     人工        Agent     人工         Agent
```

人工 gate：**1 / 3 / 3.5 / 5 / 7**，不可跳过。

> **铁律**：Step 4 TTS 一旦推出，脚本即冻结。任何 demo 没跑过就写进脚本的量化描述，必须在 3.5 复核中拦下，否则只能花钱重生成 wav。

## 何时用

- 启动新章节（先看这份定流程）
- 卡在某一步不知道对应 skill 文档（用下表跳转）
- 跨步骤问题（如 step 3 写脚本时怀疑 demo 数据错 → 回 step 1）

## Step → Skill 映射（按需加载子文档）

> **文件编号 = Step 编号**。Step 2 / 3.5 / 5 / 7 暂无独立 guide，规则散在 _0 / _2 节内（下表"位置"列指明）。

| Step | 子 skill | 位置 | 何时读 |
|---|---|---|---|
| 1a 讲解段 | [`_1-explanation-guide.md`](./_1-explanation-guide.md) | 独立 guide | 写讲解段 |
| 1b demo 制作 | [`_2-demo-guide.md`](./_2-demo-guide.md) + [`_marimo-guide.md`](./_marimo-guide.md) + [`_marimo-math-guide.md`](./_marimo-math-guide.md) | 独立 guide | 写 marimo demo |
| 2 拆分 | — | 散 _0 §2（5 条依据）| 拆期 |
| 3 写脚本 | [`_3-script-guide.md`](./_3-script-guide.md) | 独立 guide | 写 plan.md / script.json |
| 3.5 实测复核 | — | 散 _0 §3.5（待抽出 _3.5-fact-check-guide.md）| TTS 前最后一关 |
| 4 TTS | 外部（tts-agent-harness）| — | 推 script.json |
| 5 调布局 | — | 散 _2 §4（grid layout）| marimo edit 调比例 |
| 6 录屏 | [`_6-recording-guide.md`](./_6-recording-guide.md) | 独立 guide | TTS 就绪后做 mp4 |
| 7 预览审核 | — | 散 _0 §7（6 项验证）| preview.html 同步播 |
| 8a 推下游 | [`_8a-pipeline-sync-guide.md`](./_8a-pipeline-sync-guide.md) | 独立 guide | mp4 推 astral-pipeline |
| 8b 发布 | [`_8b-publish-guide.md`](./_8b-publish-guide.md) | 独立 guide | 写 publish.json |
| 横切 | [`_架构规则.md`](./_架构规则.md) | 宪法 | 创建新章节文件结构 |

---

## Step 1 · 讲解段 + Demo 制作

> **PPT/教材原话整理 ≠ 完成 Step 1**——必须按下面两份 guide 加工成讲解段 + 互动 demo。

两个并行子项（`人工+Agent`）：

- **1a 讲解段写作**：按 [`_1-explanation-guide.md`](./_1-explanation-guide.md)（三层结构：直觉锚 → 概念 → 推广 + 关键启示金句）
- **1b Demo 制作**：按 [`_2-demo-guide.md`](./_2-demo-guide.md)（marimo demo grid 友好原则，**一个 cell = grid 最小可摆放单元**）

写完各自按指南末尾的验证清单过一遍。

---

## Step 2 · 拆分依据（5 条）

1. **教材章节边界**：1 md 通常 1 期，太短合（KNN 03-distance 4 md 合 e06）/ 太长拆（KNN 05-hyperparameter 拆 e11a + e11b）
2. **demo 覆盖**：含独立 demo 通常独立成期
3. **时长 10-25 min**（B 站完播率甜点）
4. **概念递进**：每期 1-2 核心问题
5. **衔接钩**：上期 cta 伏笔 → 下期 hook 回应

输出：`scripts/README.md` 索引 + `scripts/eXX-期名/` 骨架

---

## Step 3 · 写脚本

按 [`_3-script-guide.md`](./_3-script-guide.md)。输出 `eXX/plan.md` + `script.json`。

---

## Step 3.5 · 实测复核（TTS 前最后一道关）

> **专杀凭空数字。**写脚本时凭印象写下的量化描述，必须**全部回到 demo 真实输出**对一遍。

跑 demo 一遍，逐条核对脚本里**每一处量化描述**：

| 类型 | 例子 | 怎么核 |
|---|---|---|
| 具体数字 | "K=80 准确率约 50%" | demo 跑出来是不是 50% |
| 倍数关系 | "predict 比 fit 慢一个数量级" | %timeit 实测算倍数 |
| 比例描述 | "5 个邻居 4 个不喜欢" | demo 真的是 4:1 吗 |
| 输出形态 | "输出 ['喜欢']" / "shape (9, 2)" | print 出来是不是这样 |
| 跨期一致 | "长津湖 = 不喜欢"（E02）vs "= 3.5 分"（E05） | 同一样本跨期标签互不矛盾 |

凡是**不能从 demo 输出复现**的描述 → 改成定性表达，或者重测改成对的数字。

历史踩过的坑（防再犯）：
- E03 `K=80 ≈ 50%` → 实测 41.6%
- E04 `fit 几十倍于 predict` → 实测 6 倍
- E05 长津湖 3.5 与 E02 label=0 不一致 → 改 2.5
- **E11a LOO 档位 demo 写 `cv=N`** 撞 stratified KFold `n_splits > 每类样本数` 报错 → 改 `LeaveOneOut()`

**边界档位铁律**：所有边界档位（LOO / k=1 / k=N / cv=2 / cv=数据集大小）必须实跑，不只 baseline。教材没人测过的档位经常崩。

**输出**：每条量化描述标 ✓，TTS 闸门才能开。

---

## Step 4 · TTS（外部）

推 `script.json` 给 `~/projects/tts-agent-harness`，等 `~/projects/astral-pipeline/mlXX/tts/.ready`。

产物：`shotYY.wav` + `subtitles.json`（字级时间戳）+ `durations.json`

---

## Step 5 · 调 demo 布局（人工）

```bash
marimo edit XX-name.py --port 2718
# Cmd+Shift+L 切 Grid → 调比例 → 自动保存 layouts/XX.grid.json
```

如已有 layouts.json 且 demo 未变 → 跳过。

---

## Step 6 · 录屏

按 [`_6-recording-guide.md`](./_6-recording-guide.md)。输出 `_recording/output/eXX_shotYY_cropped.mp4`。

---

## Step 7 · 预览审核（人工）

`_recording/output/preview.html` 同步播视频+音频+字幕。

验证 6 项：时长 = TTS / 裁剪正确 / 颜色对 / 拖动可见 / 静帧 ≥1.5s / 末尾不抖。

---

## Step 8 · 产物

本仓侧产物：

```
scripts/eXX-期名/
├── recording/
│   ├── shotYY.mp4    ← 拷自 _recording/output/
│   └── README.md     ← 下游使用说明
└── publish.json      ← 4 平台发布文案（douyin/xiaohongshu/wechat_channels/bilibili）
```

两个子任务：

- **8.1 推 recording 到 pipeline**：按 [`_8a-pipeline-sync-guide.md`](./_8a-pipeline-sync-guide.md)，跑 `bash scripts/_recording/sync-to-pipeline.sh <epXX> <mlXX>`，原子写 mp4 + manifest.json + .ready 到 `~/projects/astral-pipeline/<mlXX>/recording/`
- **8.2 写 publish.json**：按 [`_8b-publish-guide.md`](./_8b-publish-guide.md)，4 平台文案（bilibili 走 tags 数组 + 章节时间戳，其他三平台 # 内联在 description 末尾），不写 cover

---

## Step 8+ · 隐形子任务（每章/每会话收尾自检）

8 步主流程是**看得见的产出**（plan/script/recording/publish）。每章收尾时还有 4 类**隐形工作**经常被忽视——它们决定下一会话/下一开发者的接手成本：

| 类型 | 例 | 工作特征 | 决定 |
|---|---|---|---|
| **新增** | 新 demo / 新章节 / 新 guide | 从 0 到 1 | 当期可见 |
| **重构** | 字段契约演化 / 命名修正 / 文件重组 | 已有产物结构变更 | 下次扩展是否丝滑 |
| **整理** | 字段顺序统一 / commit 分组 / 文档导航段 | 形态没变但更易用 | 下次接手成本 |
| **沉淀** | 单次教训 → 通用规则（如 §3.5 "边界档位铁律"）| 抽象升级 | 下次会不会再踩同坑 |

后两类（整理 + 沉淀）最易被低估但价值最高。

收尾自检 4 问：

- 这次的临时决策有哪些应该回写到 guide？（沉淀）
- 字段 / 命名 / 路径有没有不一致需要批量修正？（整理）
- 单次踩的坑能不能升成可推广的规则？（沿用 §3.5 / `_4 §7` / `_5 §6` / `_6 §9` 失败模式表格式）（沉淀）
- commit 粒度是否按主题切、每个独立可回滚？（整理）

KNN 章实战示例：本章在 12 期 publish.json 一致化（4 平台 → 删 youtube → 加 description_inline → wechat 字符合规）的过程中触发了 3 轮 publish 契约重构 + 1 轮工作流文档（_0 / _5 / _6）整理 + E11a LOO 边界踩坑沉淀成铁律。这些不在 8 步任何一步，但占了收尾会话约一半工作量。

---

## 文档结构

```
01-ML/
├── _0-workflow.md          ← 入口（本文）
├── _1-explanation-guide.md ← Step 1 讲解段写作
├── _2-demo-guide.md        ← Step 1 demo 制作
├── _3-script-guide.md      ← Step 3 写脚本
├── _6-recording-guide.md   ← Step 6 录屏
├── _8a-pipeline-sync-guide.md ← Step 8.1 推 recording 到 pipeline
├── _8b-publish-guide.md     ← Step 8.2 写 publish.json
├── _marimo-guide.md            marimo 用法
├── _marimo-math-guide.md       marimo 数学动画
├── _架构规则.md                教材组织
└── 0X-ChY/
    ├── 教材 md / demos/*.py
    └── scripts/
        ├── README.md       ← 期数索引
        ├── eXX-期名/{plan.md, script.json, recording/}
        └── _recording/     ← 录屏脚本 + 中间产物
```

---

## Agent Prompt 模板

### Step 2 拆分

```
读 01-ML/_0-workflow.md §2 + 0X-ChY/ 教材 + demos/。
输出 0X-ChY/scripts/README.md 索引 + 每期目录骨架。
```

### Step 3 写脚本

```
读 01-ML/_3-script-guide.md（严格遵守）+ 已有正稿样本（KNN scripts/e01-e11）+ 教材 + demo 代码 + TTS subtitles（如有）。
输出 scripts/eXX-期名/plan.md + script.json。
```

### Step 3.5 实测复核

```
读 01-ML/_0-workflow.md §3.5 + scripts/eXX/script.json + demo 源码。
跑 demo（marimo run / jupyter nbconvert --execute）拿真实输出。
逐条核对所有量化描述（数字 / 倍数 / 比例 / 形态 / 跨期一致），不一致就改 script。
输出：复核报告 + 修订后的 script.json。TTS 才能开闸。
```

### Step 6 录屏

```
读 01-ML/_6-recording-guide.md（严格遵守）+ scripts/eXX/script.json segment {YY} text + notes + astral-pipeline/mlXX/tts/subtitles.json。
输出 scripts/_recording/eXX_shotYY.py + scripts/eXX-期名/recording/shotYY.mp4 + README.md。
```

---

## 贯穿原则

- **决策代理**：相近候选自选最优，不让用户在 A/B 间挑（见 `~/.claude/CLAUDE.md`）
- **数据一致性**：教材 / demo / 脚本三处数字一致，改一处全章同步
- **概念推迟**：教材标"后续展开" → 脚本也推迟
- **TTS 友好**：text 字段无 markdown / 中文标点 / 数字拆短句
