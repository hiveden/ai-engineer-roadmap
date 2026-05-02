# KNN 视频脚本

> AI 老师讲解 + Demo 录屏拼接，11 期覆盖整个 KNN 章。
>
> 立场：AI 老师做分享，第一人称用"我们"，每期结尾"如果有不对欢迎指正"。

## 双源拼接

- **A 动画**：教材 md 转口播 + Remotion react 组件 + 时间轴代码生成
- **B Demo 录屏**：marimo 浏览器 + Playwright 驱动 + 录屏 + 静帧拉伸对齐
- **重叠优先用 B**（更具体）；衔接和纯概念用 A

## 录制 Pipeline（方案 D · 粗对齐 + 静帧拉伸）

教学视频不需要"句子和操作严格对齐"——观众容忍 0.5-1.5s 偏差。所以：

```
A 段（Remotion 动画）：
  口播稿 → TTS 时间戳 → react 组件按时间戳渲染 → 完全代码生成

B 段（Demo 录屏）：
  1. 写自由字数的口播稿（不被时长绑架）
  2. 写 states 操作清单：每段 demo 列 3-5 档关键状态
  3. Playwright 驱动 demo 经过这些 state，记录"开始/结束/稳定"3 个事件时间戳
  4. TTS 跑口播稿 → 拿到每句时间戳
  5. ffmpeg 按口播时间戳拼接：
     - 操作进行中（state 之间的 transition）→ 1× 速度
     - 渲染稳定的静帧 → 自动拉伸/截短匹配下句口播
```

**states 操作清单格式**（写在 segment.notes 里）：

```yaml
states:
  1. intro    — 静帧介绍 UI（不操作）
  2. baseline — rating=8.5, lead=9.5, k=5
                expect: 预测卡=会喜欢 · 5/0 · 距离 0.30/0.51/1.00/1.02/1.58
  3. dislike  — rating=2.9, lead=3.5, k=5
                expect: 预测卡翻红 · 1/4 不喜欢
  4. ...
```

**关键约束**：每个 state 必须给 `expect`（稳定后画面应有什么）—— Playwright 用来判断"渲染稳定"，剪辑时拿这个对齐口播锚点。

## 11 期索引

| 期 | 章节 | 覆盖 md | Demo | 时长 | 状态 |
|---|---|---|---|---|---|
| **e01-概念距离** | 01-intro | 01-概念 / 02-接近程度 / 02a-豆瓣手算 | `01-intro/demos/02-proximity.py` | ~18 min | 正稿 |
| **e02-k值加权** | 01-intro | 03a / 03b / 03c / 04-加权 | `01-intro/demos/03-k-tuning.py` | ~22 min | 占位 |
| **e03-工作流回归** | 01-intro | 05-工作流程 | `01-intro/demos/04-regression.py` | ~12 min | 占位 |
| e04-分类API | 02-api | 01-分类API | — | ~12 min | 待规划 |
| e05-回归API | 02-api | 02-回归API | — | ~10 min | 待规划 |
| e06-距离族 | 03-distance | 全章（4 种距离） | `03-distance/demos/01-distance-zoo.py` | ~22 min | 待规划 |
| e07-缩放动机 | 04b-scaling | 01-为什么预处理 | — | ~18 min | 待规划 |
| e08-归一化 | 04b-scaling | 02-归一化 | `04b-scaling/demos/02-knn-scaling.py`（部分） | ~22 min | 待规划 |
| e09-标准化高斯 | 04b-scaling | 03-标准化 / 04-高斯分布 | `04b-scaling/demos/02-knn-scaling.py`（部分） | ~25 min | 待规划 |
| e10-鸢尾花实战 | 04c-iris-case | 全章 | `04c-iris-case/demos/01-iris-pipeline.py` | ~15 min | 待规划 |
| e11-超参数搜索 | 05-hyperparameter | 01-CV / 02-网格搜索 / 03-手写数字 | `05-hyperparameter/demos/01-cv-gridsearch.py` | ~45 min（建议拆 2 段） | 待规划 |

## 目录约定

每期一个子目录：

```
e0X-名称/
├── script.json   ← 主产物（segments 数组：id/type/topic/text/notes）
└── plan.md       ← 编排计划（覆盖 md / demo 片段 / 砍掉什么 / 口吻提示）
```

## 格式参考

[`~/projects/script-agent-harness/episodes/agui01/script.json`](../../../../script-agent-harness/episodes/agui01/script.json) — 学习整理类教学脚本的格式范本。

注：当前格式还在迭代，**不严格遵循** script-agent-harness 的所有铁律（特别是 model B 的 "notes 不写视觉" 限制——本仓 notes 仍写 region/demo 切分指引，方便单仓闭环迭代）。格式稳定后再迁。

## script.json 字段

仿 agui01：

```jsonc
{
  "title": "...",
  "description": "...",
  "style": "tutorial",
  "form": "tutorial",
  "target_duration_s": 1080,
  "segments": [
    {
      "id": 1,
      "type": "hook" | "content" | "cta",
      "topic": "...",
      "text": "TTS 实际输入，纯口播文本，不含 [] tag",
      "notes": "字数估算 + [src: ...] + region 切分（title/body/figure 等）+ demo clip 引用"
    }
  ]
}
```

**notes 内 region 标记的扩展**（针对双源）：

- `title "..."` / `body "..."` / `figure "..."` / `quote "..."` — 动画段（A）
- `clip "demos/02-proximity.py · MM:SS-MM:SS · 简述"` — demo 录屏段（B）
- `formula "公式描述"` — 公式定格动画

## 立场要点

- **第一人称：我们**（"我们来看"/"我们注意"）
- **AI 老师**做主讲，但不端权威——结尾留"如有不对欢迎指正"
- **教学语气**可断言，但不强势；用"想象"/"假设"/"注意"做引导
- **不引官方文档**（这不是评测）——直接讲概念
- **术语首次出现中英双标**：K 近邻（K Nearest Neighbor，KNN）
