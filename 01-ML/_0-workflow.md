# ML 视频生产工作流

> **总入口** · 教材 → 视频成片。任何 ML 章节（KNN / LR / LogReg / DT / ...）通用

```
0.底稿装载 → 1.讲解+Demo → 2.拆分 → 3.写脚本 → 3.5.实测复核 → 4.TTS → 5.调布局 → 6.录屏 → 7.预览审核 → 8.产物
 人工+Agent    人工+Agent    人工     Agent      人工+Agent      外部     人工        Agent     人工         Agent
```

人工 gate：**1 / 3 / 3.5 / 5 / 7**，不可跳过。Step 0 是起手前置，不阻塞但底稿不全则 Step 1 写不出。

> **铁律**：Step 4 TTS 一旦推出，脚本即冻结。任何 demo 没跑过就写进脚本的量化描述，必须在 3.5 复核中拦下，否则只能花钱重生成 wav。

## 何时用

- 启动新章节（先看这份定流程）
- 卡在某一步不知道对应 skill 文档（用下表跳转）
- 跨步骤问题（如 step 3 写脚本时怀疑 demo 数据错 → 回 step 1）

## Step → Skill 映射（按需加载子文档）

> **文件编号 = Step 编号**。Step 2 / 3.5 / 7 暂无独立 guide，规则散在 _0 / _2 节内（下表"位置"列指明）。

| Step | 子 skill | 位置 | 何时读 |
|---|---|---|---|
| 0 底稿装载 | [`_0a-source-guide.md`](./_0a-source-guide.md) | 独立 guide | 起手新章节 / 底稿丢数据 / 拆子文件 |
| 1a 讲解段 | [`_1-explanation-guide.md`](./_1-explanation-guide.md) | 独立 guide | 写讲解段 |
| 1b demo 制作 | [`_2-demo-guide.md`](./_2-demo-guide.md) + [`_marimo-guide.md`](./_marimo-guide.md) + [`_marimo-math-guide.md`](./_marimo-math-guide.md) | 独立 guide | 写 marimo demo |
| 2 拆分 | — | 散 _0 §2（5 条依据）| 拆期 |
| 3 写脚本 | [`_3-script-guide.md`](./_3-script-guide.md) | 独立 guide | 写 plan.md / script.json |
| 3.5 实测复核 | — | 散 _0 §3.5（待抽出 _3.5-fact-check-guide.md）| TTS 前最后一关 |
| 4 TTS | 外部（tts-agent-harness）| — | 推 script.json |
| 5 调布局 | [`_5-layout-guide.md`](./_5-layout-guide.md) | 独立 guide | marimo edit 调录屏布局 |
| 6 录屏 | [`_6-recording-guide.md`](./_6-recording-guide.md) | 独立 guide | TTS 就绪后做 mp4 |
| 7 预览审核 | — | 散 _0 §7（6 项验证）| preview.html 同步播 |
| 8a 推下游 | [`_8a-pipeline-sync-guide.md`](./_8a-pipeline-sync-guide.md) | 独立 guide | mp4 推 astral-pipeline |
| 8b 发布 | [`_8b-publish-guide.md`](./_8b-publish-guide.md) | 独立 guide | 写 publish.json |
| 横切 | [`_架构规则.md`](./_架构规则.md) | 宪法 | 创建新章节文件结构 |

---

## Step 0 · 底稿装载与子文件拆分

> **PPT 原话 → md 章节起手**。每章必跑一遍——Step 1 写讲解段时引用的"## 底稿"段就是这一步的产物。

按 [`_0a-source-guide.md`](./_0a-source-guide.md) 走 6 节：

1. **PPT XML 扫描法**（三路提取：文字 / MathML / 备注）—— README 缺数据时用
2. **ASCII 图代替 PPT 原图** —— 可视化补回 README，不嵌外部图
3. **slide → 子文件归属表** —— 同章主题下放 / 跨章综合留 README
4. **子文件底稿同步格式** —— 完整复制 README 对应段，不简述
5. **PPT 错误处理** —— 忠实誊抄 + ⚠ 警告 + 正确解（已知 03b/04a 都有 PPT bug）
6. **收尾自检清单** —— grep 占位 / diff 底稿 / 检查归属

**输出**：章 README（含完整底稿）+ 子文件骨架（每个含底稿 + 待写讲解段）。

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

按 [`_5-layout-guide.md`](./_5-layout-guide.md) 严格走。如已有 layouts.json 且 demo 未变 → 跳过。

```bash
marimo edit XX-name.py --port 2718
# Cmd+Shift+L 切 Grid → 拖拽调位置 → 自动保存 layouts/XX.grid.json
```

关键决策（详细见 _5）：
- **录屏 1280×720** + ffmpeg upscale 到 1080p（marimo 默认组件按 ~1280px 设计，1920 容器会拉变形）
- **Strategy B 永远双槽** 是 LR 类对照教学的默认（slot1 主图 + slot2 辅助：表/公式/同模型俯视）
- **录屏区只放视觉演示**：标题 / 控件 / chart / 数字面板。提示组件（shot dropdown / narration / 真实参数）全部录屏区**下方**，不入画
- **sidebar 12-15%** + 极简 label（1-2 字符）+ slider `full_width=True` + 可选 `transform: scale(0.92)`
- **盖宋体**：`custom.css` 设 `--marimo-heading-font` / `--marimo-text-font`

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
├── _0a-source-guide.md     ← Step 0 底稿装载与拆分
├── _1-explanation-guide.md ← Step 1 讲解段写作
├── _2-demo-guide.md        ← Step 1 demo 制作
├── _3-script-guide.md      ← Step 3 写脚本
├── _5-layout-guide.md      ← Step 5 调录屏布局
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

---

## 变更日志

### 2026-05-05 · Step 0 抽出独立 guide（底稿装载与拆分）

LR 02-LR 03b/04a/04b 三轮重扫沉淀（PPT XML 扫描法 / ASCII 图代图 / slide → 子文件归属表 / PPT 错误处理）后两步收口。

**第 1 步**（先做）：规则写进 `_架构规则.md` v1.2 §16-19。

**第 2 步**（重构）：抽出独立 [`_0a-source-guide.md`](./_0a-source-guide.md)，定位 Step 0 起手指南，对齐"每个有内容的 Step 一个 _N-guide.md"既有模式：
- `_架构规则.md` §16-19 收回 → 一行 pointer + 头部 v1.2 描述指向 `_0a`
- `_0` 工作流图加 Step 0 / Step 表加一行 / 文档结构图加 _0a / 新增 §Step 0 段
- `_1-explanation-guide.md` 头部 + §9 + 末尾分工表加交叉引用

**核心知识沉淀**（详见 `_0a`）：
- PPT XML 三路提取：`<a:t>` 文字 + `<m:oMath>` MathML + notesSlides 备注
- MathML 渲染为伪 LaTeX（递归处理 sSup / f / nary / d / m）
- ASCII 图替代外部图：散点 / 抛物线 / 向量 / 矩阵阶梯都能画
- slide → 子文件归属：同章主题下放（即使编号不连续）/ 跨章综合留 README
- PPT 错误处理四步：忠实誊抄 + ⚠ 警告 + 官方答案 + 验证步骤
- 反例库：03b slide 23-25 实战 / 04a slide 56 PPT 答案错误（k=0.0397 代回不成立）/ 04b slide 67 下山 4 步骤

### 2026-05-05 · Step 5 抽出独立 guide
LR 章 `02-LR/01-intro/demos/k-to-w-migration.py` 录屏布局调试约 1.5 小时（约 25 轮 grid 重排）后两步沉淀。

**第 1 步**（先做）：规则写进 `_0` §Step 5（约 80 行）。

**第 2 步**（重构）：抽出独立 [`_5-layout-guide.md`](./_5-layout-guide.md)，对齐"每个有内容的 Step 一个 _N-guide.md"既有模式：
- _0 §Step 5 收回 ~10 行 pointer
- _0 文档结构图 / Step 表 / "哪些 Step 有独立 guide"段同步
- `_marimo-guide.md` §样式 扩充（custom.css 注入 / `[data-cell-name]` selector / `.style()` 链式 / 录屏类布局常用 hack 表）
- `_2-demo-guide.md` §4 / §5 加交叉引用（录屏类 demo → 看 _5）

核心知识沉淀（详见 `_5`）：
- 录屏 1280×720 + ffmpeg upscale 1.2× → 1920×1080（首选 16:9 尺寸，组件不变形）
- Strategy B 双槽：主区永远 2 槽（slot1 主图 + slot2 表/公式/俯视），画面稳定 + 教学密度翻倍
- 提示组件全部录屏区下方（shot dropdown / narration / 真实参数），不入画
- sidebar 12-15%：极简 label（1-2 字符）+ `full_width=True` + `transform: scale(0.92)`
- custom.css 三件套：锁画布宽 / 字体覆盖（仅 3 个公开 CSS 变量）/ 录屏区红虚线边框
- 失败模式 8 条：compaction / 双图溢出 / accordion / callout 空白 / 画布响应式 / container 自适应 / 宋体 / sm:pt-8 顶部留白
- 方法论：用 `marimo edit` 拖拽 grid 而非硬编码 json，避开 react-grid-layout compaction 雷

### 2026-05-05 · 子 Agent 验证 _5（同日补丁）
派 general-purpose agent 按 `_5-layout-guide.md` 改造 e02 `02-LR/02-api/demos/api-walkthrough.py`，验证文档可独立工作。Reflection 报告：`~/2026-05-05-step5-validation-report.md`。

发现 **3 处关键盲点**导致严格按 _5 走会卡住，必须回看参考实现或 DOM 探测。已据此修订：

| 修订 | 文件 |
|---|---|
| 新增 §11 "edit vs run 模式 DOM 差异" | `_5-layout-guide.md` |
| §3 加 `chart height ↔ cell h` 换算表 + 公式 `cell_h ≥ ceil((height + 80) / 20)` | `_5-layout-guide.md` |
| §4 sidebar `transform: scale` 加 run 模式失效警告 + cell 命名规则 | `_5-layout-guide.md` |
| §7 失败模式表新增 4 条（cell 内容下沉 / chart axis 截掉 / selector run 模式失效 / playwright dropdown 静默失败） | `_5-layout-guide.md` |
| §10 自检"切镜头"改为文本断言（4 张相同截图无法肉眼区分）| `_5-layout-guide.md` |
| §样式 标注 `[data-cell-name]` selector 仅 edit 模式生效 | `_marimo-guide.md` |

**沉淀的元教训**：marimo 平台知识不能依赖单 mode 验证。`edit` 模式开发体验和 `run` 模式录屏体验是**两套 DOM**，写规则要分别标注适用模式。

**子 Agent 验证模式有效性**：花 80 分钟 + 一个 sub-agent token 预算，换来文档质量从 80% stand-alone 升到 ~95%。下次新增 _N-guide.md 应当默认派 sub-agent 复检一遍再交付。

### 2026-05-05 · 元教训：写规则 ≠ 应用规则（同日补丁）
e02 api-walkthrough 改造过程中**主开发者（我）连续违规 6 次** _5 自己刚写的规则（preset 塞录屏内 / sidebar 5+ col 过宽 / label 带中文括号 / 没截图自检...），全部由 user 反馈才修正。

**最深层根因**：LLM **能"陈述"规则正确但"应用"规则失败**——文档生成是 generation task，应用规则是 reasoning + checklist task，是两件事。

**修法已沉淀进 `_5-layout-guide.md` §10 顶部"🚨 强制流程"段**：
- 新 demo 开工前抄 k-to-w 骨架（不重新发明）
- 打印 §10 自检 11 项到工作记忆顶
- 每改一处 grid.json / chart props / label → playwright 截图自检
- 不让 user 当 QA：交付前自己确保 11 项 ✓
- 列出 4 类典型违规模式作为 avoid 清单

**对其他 _N-guide.md 的启示**：含规则的 guide 都应该在末尾（gate 节）加"🚨 强制流程"段——明确告诉读者本节不是"参考"是"closing gate"。否则文档会被当成 generation reference 而非 application checklist。

### 2026-05-05 · 第二轮 sub-agent 验证（同日补丁 +e03）
派 sub-agent 按强制流程改造 e03 `03b-math/demos/mse-residual-squares.py`。Reflection：`~/2026-05-05-step5-validation-2-mse.md`。

**验证结果**：✅ 强制流程段 **6/6 全预防 e02 违规**——抄骨架（第 1 步）+ 截图自检（第 3 步）是最起作用的两条。

**最关键发现**：抄骨架收益 **~82 min（145% 加速）**——custom.css 三件套 / cell 命名 / Strategy B stage 模板 / sidebar 紧凑样式 / narration 字典结构都直接 copy。但数据 / chart / shot 内容设计抄不动（demo-specific）。

**新发现 4 个文档盲点已修订**：

| # | 盲点 | 修订位置 |
|---|---|---|
| D.1 | truth_hint 用 `mo.callout` 在 h≤5 cell 空白渲染（已沉淀但还踩）| §3 加"录屏外提示组件用纯 HTML div"主推模板 |
| D.2 | dropdown default value 改后必须重启 marimo（`--watch` 无效）| §11 加"切镜头实操流程"代码块 |
| D.3 | cell 顺序重排后 grid.json position 数组要全部重映射 | §4 加"@app.cell 顺序对齐 grid.json"铁律 |
| D.4 | 录屏区 720 边界视觉判定模糊（y=32+h=3 在临界） | §3 加"录屏外组件 y 坐标 ≥ 36"铁律 |

**工时**：sub-agent 95 min（vs 上轮 e02 80 min，多 15 min 是更严格的截图自检 + 详尽 reflection）。如果不抄骨架预计要 ~180 min。

**11 项 closing gate**：10/11 ✓ + 1 N/A（ffmpeg 不在本任务范围）。

**模式总结**：派 sub-agent 改造 + reflection 是 _N-guide 持续优化的高 ROI 闭环。每改造一个新 demo = 一次文档质量验证。预计 e04+ 后续 demo 改造时盲点会越来越少（接近收敛）。

### 2026-05-05 · 第三轮 6 agent 并行 + 用户验收闭环（同日）
创建 team `lr-demo-batch`，并行派 6 sub-agent 改造剩余 6 demo（e04a/e04b/e06/e08/e10/e11）。LR 章 demo 改造 100% 收口（除设计上无 demo 的 e05/e07/e09）。

**Sub-agent 工作流**：
1. TeamCreate + 6 TaskCreate 一一映射
2. Spawn 6 agents 携 team_name + 任务说明
3. 每 agent 独立 claim task → 改造 → 自检 → reflection → SendMessage 报告
4. team-lead（我）汇总 6 份 reflection 提取共性 + 新盲点
5. 串行用户验收 → 修复发现的 bug → shutdown team → 沉淀

**用户验收阶段发现 5 处问题修复**（agent 自检漏掉的）：
| demo | bug | 修法 |
|---|---|---|
| e08 | mark_text dy=-6 推 MSE label "7.5" 出 chart 边界 | y `Scale(domainMax=max*1.15)` 留白 |
| e10 | PRESET button on_click 调 `slider.set_value()` 不存在的 API | 改 `mo.state` 共享 |
| e10 | mo.state 改变让同 cell 内 shot dropdown 也重置 | 拆 widget 到独立 cell |
| e10 | shot + button 重复设计（主播切两次）| 删 button，shot 联动 effective_degree + dynamic title |
| e10 | 双 chart width=440 临界 1050px → X scroll | 缩到 width=400 |

**沉淀进 _5/_marimo-guide**：
- _5 §7 失败模式表新增 7 条（mark_text 溢出 / mo.state 多 widget / button setter 幻觉 / altair_chart duplicate signal / mo.md HTML block markdown 不解析 / altair y 轴拼接 / plotly 3D legend 溢出）
- _5 §2 加"shot 联动 effective_X 模式" + "Dynamic title"两个推荐模板
- _5 §7 加 chart 双图临界 1000px 提示
- `_marimo-guide` §实战教训新增 5 条 marimo API 行为坑

**核心元教训**：
1. **agent 自检 ≠ 用户验收**——agent 报告"11/11 ✓"但用户视觉验收发现 5 处问题。流程合规 ≠ 产物质量
2. **并行 6 agent 共发现 ~12 处新盲点 / 推荐模式**，比单 agent 收敛快 ~3 倍
3. **dynamic title 模式补足 shot 在录屏外的可见性问题**——shot 切换不再"无声"，title 文字随之变化给观众视觉反馈
4. **把"陈述规则 → 应用规则 → 用户验收 → 沉淀"形成完整闭环**：每轮验收发现的 bug 都修订进文档，下次新 demo 享受全部教训

**LR 章总产出**（9 个 demo + 3 个文档轮）：
- 9 个 marimo demo 全部按 _5 标准统一布局（k-to-w / api / mse / partial-deriv / matrix / gd-landscape / metric / poly-overfit / l1-l2）
- _5-layout-guide.md 从 v0 演进到 v4（含强制流程 / 11 项 closing gate / edit-vs-run §11 / 失败模式 8+7 条 / 推荐模板）
- 3 份累积 reflection 报告 + 1 份 v3 验证（共 ~6 万字 token 经验沉淀）
