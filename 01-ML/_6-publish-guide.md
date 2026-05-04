# Publish 指南（skill）

> Step 8 子任务 2。每期一份 `scripts/eXX-期名/publish.json`，覆盖 4 个发布平台文案。
>
> 路径归口：本仓（不进 `script-agent-harness`）。下游 astral-video / 发布工具直接读本仓相对路径，或日后另起 sync 推到 `pipeline/<id>/publish/`。

## 1. 前置 checklist

- [ ] `scripts/eXX-期名/script.json` 已稳定（title / segments[].topic / text 不会再大改）
- [ ] `~/projects/astral-pipeline/<id>/tts/durations.json` 存在（bilibili 章节时间戳要算）
- [ ] 本期立场已对齐（默认教学/AI 老师；见 §2）

## 2. 立场（与 script.json 同源）

KNN 系列定位：**AI 老师做分享，不端权威**（见 `scripts/README.md` 立场段）。

publish 文案沿用同一口吻，**不照搬** `script-agent-harness/RULES.md` 的"实践者复盘"立场——后者适合 `meta/flash/brief/run` 等技术评测系列，与教学序列违和。

| 必做 | 必禁 |
|---|---|
| 第一人称"我们" / "我"（教学口吻） | 老师权威话术（"必须 / 一定要"） |
| 结尾留"如有不对欢迎指正" | "必看 / 不看后悔 / 锁定下期" |
| 用具体数字（"5 个邻居距离 0.30 / 0.51..."） | 含糊量化（"很多" / "几个" / "大幅") |
| 暴露不确定 / 互动邀请（"问题评论区见"） | 乞求话术（"求点赞 / 求关注"） |
| 直接讲概念 | 引官方文档作权威佐证（不是评测） |

封面文案（cover）**不写**。视觉决策归下游 astral-video / 封面工具（与 `script-agent-harness/CONTRACT.md` model B 一致）。

## 3. 契约（publish.json 字段结构）

```json
{
  "platforms": {
    "douyin":          { "title": "...", "description": "... \n\n#tag1 #tag2 ..." },
    "xiaohongshu":     { "title": "...", "description": "... \n\n#tag1 #tag2 ..." },
    "wechat_channels": { "title": "...", "description": "... \n\n#tag1 #tag2 ..." },
    "bilibili":        {
      "title": "...",
      "description": "...（多段、含章节时间戳，无 #）",
      "description_inline": "...（同 description 但换行折叠为空格，无 \\n）",
      "tags": ["a","b",...]
    }
  }
}
```

**4 平台**：douyin / xiaohongshu / wechat_channels / bilibili。**不做** youtube。

### 平台差异速查

| 平台 | tags 字段 | description 末尾 # | 章节时间戳 | description_inline |
|---|---|---|---|---|
| douyin | ❌ 无 | ✅ `#xxx` 内联，**≤ 5 个**（硬约束） | ❌ 无 | ❌ 无 |
| xiaohongshu | ❌ 无 | ✅ `#xxx` 内联，6-9 个 | ❌ 无 | ✅ 必写（描述同源，换行折叠） |
| wechat_channels | ❌ 无 | ✅ `#xxx` 内联，2-4 个 | ❌ 无 | ❌ 无 |
| bilibili | ✅ 数组 8-10 个 | ❌ 不写 # | ✅ 必写（`MM:SS 标题` 行） | ✅ 必写（描述同源，换行折叠） |

### 硬约束

- **抖音 description 末尾内联 # 数 ≤ 5**（平台规则）
- **bilibili tags 数组**只用于 bilibili，其他平台不要重复加 tags 字段
- **不写 cover 字段**
- **bilibili / xiaohongshu 必带 `description_inline`**：用于不支持换行的粘贴场景（某些 UI 文本框 / 表单 / 自动化脚本）。两个平台原始 description 都是多段格式（小红书带分点列表 + 末尾 # 标签，B 站带章节时间戳），inline 是同源粘贴版。生成规则见 §4 末

## 4. 章节时间戳（仅 bilibili）

从 `~/projects/astral-pipeline/<id>/tts/durations.json` 累加：

```python
import json
durs = json.load(open('/Users/xuelin/projects/astral-pipeline/ml01/tts/durations.json'))
t = 0
for shot in durs:
    m, s = divmod(int(t), 60)
    print(f'{m:02d}:{s:02d}  {shot["id"]}')
    t += shot['duration_s']
```

**时间戳取整到秒**（B 站章节就支持秒精度）。**topic 改写为口播式**而非 segment.topic 原文（segment topic 是给 script 用的，发布文案要更可读）：

| segment.topic（原） | bilibili 章节标题（改写） |
|---|---|
| `KNN 一句话定位 + 本期预告` | `KNN 一句话定位 + 本期预告` ← 直接用 |
| `Demo 录屏 · 二维欧氏距离 + 投票预测` | `marimo demo · 二维 KNN 投票预测` ← 简化 |
| `豆瓣手算 K=5 公式细节` | `K=5 多数票完整手算` ← 改更易懂 |

口径：去掉脚本里的内部记号（"Demo 录屏"等），用更广播友好的措辞。

### description_inline 生成规则（bilibili + xiaohongshu）

每次改 bilibili.description 或 xiaohongshu.description 后，同步重生成对应 description_inline：

```python
import re
def flatten(text):
    out = re.sub(r'\s*\n+\s*', ' ', text)   # 所有连续换行（含两侧空白）→ 单空格
    out = re.sub(r' {2,}', ' ', out)         # 多空格折叠
    return out.strip()
```

约定：description 是「展示版」（B 站多段 + 章节时间戳竖排 / 小红书分点列表 + 末尾 # 标签竖排），description_inline 是「粘贴版」（语义完全相同，仅去换行）。两份**必须语义一致**——禁止只改 description 不同步 inline，否则下游粘贴的版本和实际显示不一致。

douyin / wechat_channels 的 description 都是单段或两段，**不需要** inline 字段（直接用 description 即可）。

## 5. 平台风格差异

### douyin（短钩子）
- title ≤ 25 字，钩子型（"K 怎么选？拖一下就懂"）
- description 1-2 段流畅自然，120-200 字
- 末尾 ≤ 5 个 # 标签

### xiaohongshu（分点 + 字符装饰）
- title 12-18 字
- description 300-500 字，多分行 + 列表（`·` 项符 + emoji 数字 1️⃣2️⃣ 偶用）
- 末尾 6-9 个 # 标签
- 立场更亲切（"问题评论区见"）

### wechat_channels（朋友圈式）
- title 10-15 字
- description 100-150 字，单段流畅
- 末尾 2-4 个 # 标签（精选最核心的）
- **title 字符白名单**（平台硬约束）：中文 / 英文字母 / 数字 / 空格 / 书名号《》/ 引号 ""''「」 / 冒号 ：: / 加号 + / 问号 ？? / 百分号 % / 摄氏度 ℃。**禁用** `·` `-` `,` `，` `、` `。` `！` `；` `/` `(` `)` `[` `]` `…` 等其他符号。逗号用空格替代。
- 常见替换约定：
  - `·`（中点分隔）→ `：` 全角冒号
  - `-`（连字符）→ 空格 或 直接合并（如 `Min-Max` → `MinMax`、`Z-score` → `Z score`）
  - `+` 直接保留（白名单内）
- description 字段**不受**该限制（只约束 title）

### bilibili（专业长文 + 章节）
- title 30-50 字（含副标题，"｜" 分隔主副）
- description 400-800 字，分段叙述：上一期回顾 → 本期主旨 → 关键对比/数字 → 章节时间戳 → 下期预告 → 立场声明
- tags 数组 8-10 个
- description 内**不写 #**

## 6. 系列 IP 标签

KNN / 后续 ML 章节统一用：

```
AI老师讲ML
```

放在每平台标签的末尾（douyin / xiaohongshu / wechat_channels 内联，bilibili tags 数组）。

> 占位名，可改。要换需 sed 全章批量替换；改前先确认 KNN / LR / LogReg / DT 等章节都过一遍审稿，避免再返工。

## 7. 模板（e01 缩略版）

```json
{
  "platforms": {
    "douyin": {
      "title": "KNN 入门第一课：找最像的邻居怎么找？",
      "description": "机器学习入门第一站，K 近邻 KNN。一句话：要判断新样本属于哪一类...如有讲得不对的地方，评论区指正。\n\n#机器学习 #KNN #算法入门 #欧氏距离 #AI老师讲ML"
    },
    "xiaohongshu": {
      "title": "KNN 入门 · 距离到底怎么算",
      "description": "K 近邻 KNN 是机器学习里最直观的一个算法。这一期搞清两件事：\n\n1️⃣ 算法思想：...\n2️⃣ 怎么量化「像」：欧氏距离\n\n讲解路径：\n· 五角星 vs 三角形 · 建立分类直觉\n· 豆瓣评分一维 · 数字之差就是距离\n...\n如有不对欢迎指正，问题评论区见。\n\n#机器学习 #KNN #算法入门 #欧氏距离 #AI教程 #marimo #ML入门 #AI老师讲ML"
    },
    "wechat_channels": {
      "title": "KNN 入门 · 距离怎么算",
      "description": "机器学习入门第一站 K 近邻 KNN。这一期把「距离」讲透：从豆瓣一维评分到二维勾股到 N 维欧氏距离，公式同源。配 marimo 互动 demo。如有不对欢迎指正。\n\n#机器学习 #KNN #AI老师讲ML"
    },
    "bilibili": {
      "title": "KNN 第 1 期 · 找最像的邻居 + 距离怎么定义｜机器学习入门",
      "description": "K 近邻 KNN 是机器学习里最直观的一个算法。这一期作为 KNN 入门，把第一个核心问题讲透...\n\n章节时间戳：\n00:00 KNN 一句话定位 + 本期预告\n00:29 五角星 vs 三角形 · 分类直觉\n01:51 一维直觉 · 豆瓣评分\n02:52 升到二维 · 勾股定理\n03:37 marimo demo · 二维 KNN 投票预测\n04:53 推广 N 维 · 欧氏距离命名\n05:53 K=5 多数票完整手算\n07:25 总结 + 下期预告\n\n下一期讲 KNN 第二个核心问题：K 该取多少。\n\n这是 AI 老师做分享，不是权威定义——如有不对欢迎指正，问题评论区见。",
      "tags": ["机器学习", "KNN", "K近邻", "算法入门", "欧氏距离", "marimo", "AI教程", "ML入门", "AI老师讲ML"]
    }
  }
}
```

## 8. 验证清单

```bash
python3 -c "
import json, re, sys
for ep in ['e01-概念距离', 'e02-k值加权', 'e03-工作流回归']:
    p = json.load(open(f'/Users/xuelin/projects/ai-engineer-roadmap/01-ML/01-KNN/scripts/{ep}/publish.json'))
    plats = list(p['platforms'].keys())
    assert plats == ['douyin','xiaohongshu','wechat_channels','bilibili'], (ep, plats)
    for plat, d in p['platforms'].items():
        desc = d.get('description', '')
        n_hash = len(re.findall(r'#\S+', desc))
        if plat == 'douyin':
            assert n_hash <= 5, f'{ep}/{plat} inline # = {n_hash} > 5'
            assert 'tags' not in d, f'{ep}/{plat} 不应有 tags 数组'
        elif plat in ('xiaohongshu', 'wechat_channels'):
            assert n_hash >= 1, f'{ep}/{plat} 末尾应有内联 #'
            assert 'tags' not in d
            if plat == 'wechat_channels':
                # 微信视频号标题字符白名单（§5）
                ALLOWED = re.compile(r'[\\u4e00-\\u9fffA-Za-z0-9\\s《》\"\"\\'\\'「」：:+?？%℃]')
                bad = [c for c in d['title'] if not ALLOWED.match(c)]
                assert not bad, f'{ep}/{plat} title 含非法字符 {set(bad)}'
            if plat == 'xiaohongshu':
                inline = d.get('description_inline', '')
                assert inline, f'{ep}/{plat} 缺 description_inline'
                assert '\\n' not in inline, f'{ep}/{plat} description_inline 含换行'
                assert re.sub(r'\\s+','',desc) == re.sub(r'\\s+','',inline), \\
                    f'{ep}/{plat} inline 与 description 语义不一致'
        elif plat == 'bilibili':
            assert n_hash == 0, f'{ep}/{plat} bilibili description 不应含 #'
            assert isinstance(d.get('tags'), list) and len(d['tags']) >= 5
            assert re.search(r'\\d{2}:\\d{2}', desc), f'{ep}/{plat} 缺章节时间戳'
            inline = d.get('description_inline', '')
            assert inline, f'{ep}/{plat} 缺 description_inline'
            assert '\\n' not in inline, f'{ep}/{plat} description_inline 含换行'
            # 语义一致性 spot-check：折叠后字符集合应基本一致（容许多空格差）
            import re as _re
            assert _re.sub(r'\\s+','',desc) == _re.sub(r'\\s+','',inline), f'{ep}/{plat} inline 与 description 语义不一致'
print('OK')
"
```

应输出 `OK`。

## 9. 失败模式

| 失败 | 教训 |
|---|---|
| 抖音 description 末尾 # 写到第 6 个 | 平台只识别前 5 个，后面失效；脚本验证拦下 |
| bilibili 只写 tags 不写章节时间戳 | 失去 B 站强项（章节跳转 / SEO）；hardcoded 必写 |
| 把 segment.topic 原文照搬进 bilibili 章节 | "Demo 录屏 · ..." 这种内部记号不广播友好；改写 |
| 立场写成"实践者刚做完一期分享" | 与 KNN 教学定位违和；用 AI 老师立场 |
| 写了 cover 字段 | 视觉决策归下游，不写；script-agent-harness model B 一致 |
| 多平台用同一份 description 复制粘贴 | 风格差异是平台调性，不能省（短抖音 vs 长 B 站） |
| 章节时间戳算错 | 脚本 §4 直接累加 durations.json；不要凭印象写 |
| 系列标签每期不一致 | 全章统一一个 IP（当前 `AI老师讲ML`） |
| 改 description 忘改 description_inline | inline 是同源粘贴版，改 description 后必须重跑 §4 末 `flatten()` 同步 |
| wechat_channels.title 含 `·` / `-` / `、` 等被平台拒收 | 微信视频号标题字符白名单严格，见 §5；提交前用 §8 验证脚本 ALLOWED 正则审一遍 |
