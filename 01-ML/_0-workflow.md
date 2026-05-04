# ML 视频生产工作流

> 教材 → 视频成片。任何 ML 章节（KNN / LR / LogReg / DT / ...）通用。

```
1.底稿+Demo → 2.拆分 → 3.写脚本 → 3.5.实测复核 → 4.TTS → 5.调布局 → 6.录屏 → 7.预览审核 → 8.产物
   人工        人工      Agent      人工+Agent      外部     人工        Agent     人工         Agent
```

人工 gate：**1 / 3 / 3.5 / 5 / 7**，不可跳过。

> **铁律**：Step 4 TTS 一旦推出，脚本即冻结。任何 demo 没跑过就写进脚本的量化描述，必须在 3.5 复核中拦下，否则只能花钱重生成 wav。

---

## Step 1 · 底稿 + Demo 制作

两个并行子项：

- **讲解段写作**：按 [`_1-explanation-guide.md`](./_1-explanation-guide.md)（教材 md 三段式讲解：场景→机制→命名）
- **Demo 制作**：按 [`_2-demo-guide.md`](./_2-demo-guide.md)（marimo demo grid 友好原则，**一个 cell = grid 最小可摆放单元**）

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

按 [`_4-recording-guide.md`](./_4-recording-guide.md)。输出 `_recording/output/eXX_shotYY_cropped.mp4`。

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

- **8.1 推 recording 到 pipeline**：按 [`_5-pipeline-sync-guide.md`](./_5-pipeline-sync-guide.md)，跑 `bash scripts/_recording/sync-to-pipeline.sh <epXX> <mlXX>`，原子写 mp4 + manifest.json + .ready 到 `~/projects/astral-pipeline/<mlXX>/recording/`
- **8.2 写 publish.json**：按 [`_6-publish-guide.md`](./_6-publish-guide.md)，4 平台文案（bilibili 走 tags 数组 + 章节时间戳，其他三平台 # 内联在 description 末尾），不写 cover

---

## 文档结构

```
01-ML/
├── _0-workflow.md          ← 入口（本文）
├── _1-explanation-guide.md ← Step 1 讲解段写作
├── _2-demo-guide.md        ← Step 1 demo 制作
├── _3-script-guide.md      ← Step 3 写脚本
├── _4-recording-guide.md   ← Step 6 录屏
├── _5-pipeline-sync-guide.md ← Step 8.1 推 recording 到 pipeline
├── _6-publish-guide.md     ← Step 8.2 写 publish.json
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
读 01-ML/_4-recording-guide.md（严格遵守）+ scripts/eXX/script.json segment {YY} text + notes + astral-pipeline/mlXX/tts/subtitles.json。
输出 scripts/_recording/eXX_shotYY.py + scripts/eXX-期名/recording/shotYY.mp4 + README.md。
```

---

## 贯穿原则

- **决策代理**：相近候选自选最优，不让用户在 A/B 间挑（见 `~/.claude/CLAUDE.md`）
- **数据一致性**：教材 / demo / 脚本三处数字一致，改一处全章同步
- **概念推迟**：教材标"后续展开" → 脚本也推迟
- **TTS 友好**：text 字段无 markdown / 中文标点 / 数字拆短句
