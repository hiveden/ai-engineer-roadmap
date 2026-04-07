#!/usr/bin/env python3
"""
export_session.py — 把 Claude Code 的 JSONL 会话原文导出成可读的 Markdown。

用途：保留"触发到我"的对话原文，存进 highlights/ 目录或自定义路径。
确定性脚本，不让 LLM 干浏览器自动化或文件抽取这种不可靠的活。

用法：
    # 导出本项目最新一次会话
    python3 scripts/export_session.py

    # 导出指定 session id（前缀匹配即可）
    python3 scripts/export_session.py --session 9604

    # 列出本项目所有 session
    python3 scripts/export_session.py --list

    # 包含工具调用摘要（默认只导出对话文本）
    python3 scripts/export_session.py --include-tools

    # 自定义输出路径
    python3 scripts/export_session.py --output /tmp/foo.md

    # 自定义标题（写进文件头部）
    python3 scripts/export_session.py --title "ML 入门 - 特征 vs 标签那段"

无依赖：仅用 Python 3 标准库。
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_NAME = PROJECT_ROOT.name
# Claude Code 把项目路径转成 slug：/ 替换为 -（绝对路径开头的 / 自然变成开头的 -）
PROJECT_SLUG = str(PROJECT_ROOT).replace("/", "-")
SESSIONS_DIR = Path.home() / ".claude" / "projects" / PROJECT_SLUG
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "highlights"

# 匹配 <system-reminder>...</system-reminder>（含跨行）
SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
# 匹配 <command-...>...</command-...> 等系统标签
SYSTEM_TAG_RE = re.compile(r"<(command-[a-z]+|local-command-[a-z]+)>.*?</\1>", re.DOTALL)


def list_sessions() -> list[Path]:
    """返回本项目所有 jsonl 会话文件，按修改时间倒序"""
    if not SESSIONS_DIR.exists():
        return []
    return sorted(
        SESSIONS_DIR.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def find_session(session_id_prefix: str | None) -> Path | None:
    """根据前缀找一个 session 文件；不传则返回最新一个"""
    sessions = list_sessions()
    if not sessions:
        return None
    if session_id_prefix is None:
        return sessions[0]
    matches = [s for s in sessions if s.stem.startswith(session_id_prefix)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        print(f"❌ 没找到匹配 '{session_id_prefix}' 的 session", file=sys.stderr)
        return None
    print(f"❌ '{session_id_prefix}' 有多个匹配，请提供更长前缀：", file=sys.stderr)
    for m in matches:
        print(f"   {m.stem}", file=sys.stderr)
    return None


def clean_user_text(text: str) -> str:
    """剥掉 system-reminder 和 command 标签，保留真实用户输入"""
    text = SYSTEM_REMINDER_RE.sub("", text)
    text = SYSTEM_TAG_RE.sub("", text)
    return text.strip()


def extract_user_text(message: dict) -> str | None:
    """从 user 消息中提取真实文本。tool_result / 空消息返回 None"""
    content = message.get("content", "")
    if isinstance(content, str):
        cleaned = clean_user_text(content)
        return cleaned or None
    if isinstance(content, list):
        # tool_result-only 消息跳过；其他类型暂不处理
        text_parts = []
        for block in content:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        if text_parts:
            cleaned = clean_user_text("\n".join(text_parts))
            return cleaned or None
    return None


def extract_assistant_blocks(message: dict, include_tools: bool) -> list[str]:
    """从 assistant 消息中提取可渲染块"""
    content = message.get("content", "")
    if isinstance(content, str):
        return [content] if content.strip() else []
    if not isinstance(content, list):
        return []

    blocks = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "").strip()
            if text:
                blocks.append(text)
        elif btype == "tool_use" and include_tools:
            tool_name = block.get("name", "?")
            tool_input = block.get("input", {})
            # 折叠成简短摘要
            summary = json.dumps(tool_input, ensure_ascii=False)[:120]
            blocks.append(f"> 🛠 **tool call** · `{tool_name}` · `{summary}`")
        elif btype == "thinking" and include_tools:
            thinking = block.get("thinking", "").strip()
            if thinking:
                blocks.append(f"> 💭 *(thinking)* {thinking[:300]}")
    return blocks


def parse_session(jsonl_path: Path, include_tools: bool) -> tuple[list[dict], dict]:
    """
    解析一个 session jsonl，返回 (turns, meta)
    turns: [{"role": "user"|"assistant", "text": str, "ts": str}, ...]
    meta:  {"session_id": str, "started_at": str, "ended_at": str, "cwd": str, "git_branch": str}
    """
    turns = []
    meta = {
        "session_id": jsonl_path.stem,
        "started_at": "",
        "ended_at": "",
        "cwd": "",
        "git_branch": "",
    }

    with jsonl_path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            ts = d.get("timestamp", "")
            if ts:
                if not meta["started_at"]:
                    meta["started_at"] = ts
                meta["ended_at"] = ts
            if d.get("cwd") and not meta["cwd"]:
                meta["cwd"] = d["cwd"]
            if d.get("gitBranch") and not meta["git_branch"]:
                meta["git_branch"] = d["gitBranch"]

            mtype = d.get("type")
            message = d.get("message", {})

            if mtype == "user":
                text = extract_user_text(message)
                if text:
                    turns.append({"role": "user", "text": text, "ts": ts})
            elif mtype == "assistant":
                blocks = extract_assistant_blocks(message, include_tools)
                if blocks:
                    turns.append({"role": "assistant", "text": "\n\n".join(blocks), "ts": ts})

    return turns, meta


def format_timestamp(iso_ts: str) -> str:
    """ISO 时间转人类可读"""
    if not iso_ts:
        return ""
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso_ts


def render_markdown(turns: list[dict], meta: dict, title: str | None) -> str:
    """渲染成可读 markdown"""
    started = format_timestamp(meta["started_at"])
    ended = format_timestamp(meta["ended_at"])

    header = []
    if title:
        header.append(f"# {title}")
    else:
        header.append(f"# Session {meta['session_id'][:8]}")
    header.append("")
    header.append(f"> **导出时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
    header.append(f"> **会话起止**: {started} → {ended}  ")
    header.append(f"> **Session ID**: `{meta['session_id']}`  ")
    if meta["git_branch"]:
        header.append(f"> **Git Branch**: `{meta['git_branch']}`  ")
    header.append(f"> **轮次**: {len(turns)} 条消息")
    header.append("")
    header.append("---")
    header.append("")

    body = []
    for i, turn in enumerate(turns, 1):
        role_label = "👤 **User**" if turn["role"] == "user" else "🤖 **Claude**"
        body.append(f"## [{i}] {role_label}")
        body.append("")
        body.append(turn["text"])
        body.append("")
        body.append("---")
        body.append("")

    return "\n".join(header + body)


def main():
    parser = argparse.ArgumentParser(
        description="导出 Claude Code 会话原文为 Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--session", "-s", help="Session ID 前缀（不传则取最新）")
    parser.add_argument("--list", "-l", action="store_true", help="列出本项目所有 session")
    parser.add_argument("--output", "-o", help="输出路径（不传则存到 highlights/）")
    parser.add_argument("--title", "-t", help="自定义文件标题")
    parser.add_argument("--include-tools", action="store_true", help="包含工具调用摘要")
    args = parser.parse_args()

    if args.list:
        sessions = list_sessions()
        if not sessions:
            print(f"⚠️  {SESSIONS_DIR} 下没有 session 文件")
            return
        print(f"📂 {SESSIONS_DIR}\n")
        for s in sessions:
            mtime = datetime.fromtimestamp(s.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            size_kb = s.stat().st_size / 1024
            print(f"  {s.stem}  ·  {mtime}  ·  {size_kb:.0f} KB")
        return

    session_path = find_session(args.session)
    if session_path is None:
        sys.exit(1)

    print(f"📖 解析 {session_path.name} ...")
    turns, meta = parse_session(session_path, args.include_tools)
    print(f"   提取到 {len(turns)} 条消息")

    markdown = render_markdown(turns, meta, args.title)

    if args.output:
        out_path = Path(args.output)
    else:
        DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)
        date = format_timestamp(meta["started_at"])[:10] or datetime.now().strftime("%Y-%m-%d")
        slug = (args.title or meta["session_id"][:8]).replace(" ", "-").replace("/", "-")
        out_path = DEFAULT_OUTPUT_DIR / f"{date}-{slug}.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown)
    print(f"✅ 写入 {out_path}")
    print(f"   {out_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
