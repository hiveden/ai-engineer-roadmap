# Highlights · 精彩片段对话

这个目录存放**触发到我的对话原文**——不是摘要、不是总结，是完整的 Q&A 实录。

和 `learning-sessions/` 的关系：
- **`learning-sessions/`** = 给下次 Claude 看的轻量交接报告（摘要 + 状态情报）
- **`highlights/`** = 给我自己回看 / 想分享出去的对话原文（完整、有语气、有节奏、有顿悟时刻）

两者是**互补**的：日志丢了语气保留了结构，原文保留了语气丢了结构。摘要永远不能替代 transcript。

---

## 怎么导出一段对话

用 [`scripts/export_session.py`](../scripts/export_session.py)，无依赖（仅 Python 3 标准库）。

```bash
# 导出最新一次会话，自动存到本目录
python3 scripts/export_session.py

# 给文件起个有意义的标题（推荐）
python3 scripts/export_session.py --title "ML 入门 - 特征 vs 标签那段"

# 指定 session id（前缀匹配，不用打全）
python3 scripts/export_session.py --session 9604

# 列出本项目所有 session
python3 scripts/export_session.py --list

# 包含工具调用摘要（默认只导对话文本）
python3 scripts/export_session.py --include-tools

# 自定义输出路径（不进 highlights/）
python3 scripts/export_session.py --output /tmp/foo.md
```

---

## 数据从哪来

Claude Code 把每次会话以 JSONL 存在：

```
~/.claude/projects/-Users-USER-projects-ai-engineer-roadmap/*.jsonl
```

每个 `.jsonl` = 一次完整的 session，每行 = 一条事件（user / assistant / tool 调用 / hook 等）。

`export_session.py` 干的事：
1. 找到对应 jsonl
2. 过滤掉 system 噪音（hook 事件、文件快照、tool_result）
3. 剥掉 `<system-reminder>` / `<command-*>` 这些 Claude Code 注入的内部标签
4. 渲染成可读的 markdown，按 `[N] 👤 User` / `[N] 🤖 Claude` 分块

---

## 命名约定

导出文件默认命名：`YYYY-MM-DD-<title-slug>.md`

如果不传 `--title`，会用 session id 前 8 位当 slug——可读性差。**推荐每次手动起标题**。

好的标题示例：
- ✅ `2026-04-07-ML 入门 - 特征/标签/监督学习.md`
- ✅ `2026-04-15-反思 - 为什么我总在追求新框架.md`
- ❌ `2026-04-07-9604d524.md`（默认 fallback，没意义）

---

## git 策略

`highlights/` **进 git**——它是用户主动挑选过的、有元认知价值的对话原文。
和 `learning-sessions/` 同等地位（已验证的学习产出）。

不进 git 的：
- `~/.claude/projects/.../*.jsonl` 原始 jsonl 本身（有路径、密钥、隐私风险）
- 未经挑选的"所有会话导出"批量产物
