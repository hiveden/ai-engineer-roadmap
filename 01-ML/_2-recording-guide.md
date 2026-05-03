# 录屏指南（skill）

> Step 6 细则。需要 TTS 已就绪（`astral-pipeline/mlXX/tts/.ready`）。

## 1. 前置 checklist

- [ ] Python 3.10+ · Playwright + chromium · ffmpeg
- [ ] `astral-pipeline/mlXX/tts/.ready` 存在
- [ ] `astral-pipeline/mlXX/tts/{shotYY.wav, subtitles.json, durations.json}` 全在
- [ ] `scripts/eXX-期名/script.json` 含 demo 段 segment（notes 有 states + cue 词）
- [ ] 对应 demo `.py` 跑得通

## 2. 关键技术决策（必读）

### 2.1 macOS 系统代理（chromium 继承 socks5 失败）
```python
browser = p.chromium.launch(args=["--no-proxy-server"])
```
不要用 `curl localhost` 验存活（curl 走代理）→ 用 `lsof -i :PORT`。

### 2.2 marimo slider DOM
```html
<marimo-slider data-step="0.1" data-start="2" data-stop="10">
  <span role="slider" aria-valuenow="8.5">
```
**不是** `input[type="range"]`，**不能** `fill()`。用 focus + 键盘箭头：
```python
def set_slider(page, idx, target, step_delay_ms=30):
    sliders = page.locator('marimo-slider')
    aria = sliders.nth(idx).locator('[role="slider"]').first
    current = float(aria.get_attribute('aria-valuenow'))
    step = float(sliders.nth(idx).get_attribute('data-step'))
    presses = round((target - current) / step)
    aria.focus()
    key = "ArrowRight" if presses > 0 else "ArrowLeft"
    for _ in range(abs(presses)):
        aria.press(key)
        page.wait_for_timeout(step_delay_ms)
```

### 2.3 端口分离
| 端口 | 用途 |
|---|---|
| 2718 | `marimo edit`（人工调布局，**勿动**） |
| 2750 | `marimo run`（录屏专用） |

### 2.4 viewport / crop 默认
| 项 | 值 |
|---|---|
| viewport | 1920×1080 |
| crop | `1198:600:362:34`（marimo grid maxWidth=1200 居中） |

新 demo 首次录屏前用 Playwright `boundingBox` 实测确认。**不要改 grid 布局**（比例手调）。

### 2.5 ffmpeg 倒切
webm 加载头长度不确定 → 从尾倒切：
```bash
ffmpeg -sseof -78 -i raw.webm -vf "crop=1198:600:362:34" \
       -c:v libx264 -crf 18 -preset fast -y out.mp4
```

### 2.6 Altair 多层颜色冲突
```python
chart = (lines + pts + text + query).resolve_scale(color="independent")
```
否则 layer 间 color encoding 互相覆盖。录屏前先截图验证颜色对。

### 2.7 时间精确对齐
读 `subtitles.json` 找 cue 词时间戳，每个 state 操作完后 `wait_until(t)` 等到下一个 cue。录屏总长 = TTS shot 总长。

## 3. 标准动作 8 步

1. 读 `script.json` segment {YY} 的 text + notes（states + cue 词）
2. 读 `subtitles.json` shot{YY} 找 cue 词精确时间戳
3. 启 `marimo run XX.py --headless --port 2750 --no-token`
4. Probe DOM（首次写新 demo 录屏脚本必做）：
   ```python
   for i, s in enumerate(page.locator('marimo-slider').all()):
       print(i, s.get_attribute('data-label'), s.get_attribute('data-step'))
   ```
5. 写 `_recording/eXX_shotYY.py`（套 §5 模板）
6. 跑 → ffmpeg 切边
7. 拷产物：`cp output/eXX_shotYY_cropped.mp4 ../eXX-期名/recording/shotYY.mp4`
8. 写 `eXX-期名/recording/README.md`（§6 模板）

## 4. 验证清单

- [ ] 视频时长 = TTS shot 时长（差异 < 50ms，ffprobe 验）
- [ ] 裁剪尺寸正确（默认 1198×600）
- [ ] 颜色对（无 Altair 冲突）
- [ ] 滑块拖动可见（不是瞬移，step_delay_ms ≥ 30）
- [ ] 每 state 稳定 ≥ 1.5s
- [ ] 末尾不抖动（最后 state 设置完到 T_END 是静帧）

## 5. 录屏脚本模板

```python
"""eXX shotYY · demo XX-name · N states"""
from playwright.sync_api import sync_playwright, Page
import time, subprocess
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
MARIMO_URL = "http://localhost:2750"
VIEWPORT = {"width": 1920, "height": 1080}
LOAD_WAIT_S = 4.0
T_END = 78.0  # TTS shot 总长

# Cue 时间戳（从 subtitles.json 提取）
T_S2 = 39.23
T_S3 = 55.87
T_S4 = 68.61

# Slider 索引（按 demo mo.hstack/vstack 顺序）
SLIDER_RATING, SLIDER_LEAD, SLIDER_K = 0, 1, 2


def set_slider(page, idx, target, step_delay_ms=30):
    sliders = page.locator('marimo-slider')
    aria = sliders.nth(idx).locator('[role="slider"]').first
    current = float(aria.get_attribute('aria-valuenow'))
    step = float(sliders.nth(idx).get_attribute('data-step'))
    presses = round((target - current) / step)
    if presses == 0: return
    aria.focus()
    key = "ArrowRight" if presses > 0 else "ArrowLeft"
    for _ in range(abs(presses)):
        aria.press(key)
        page.wait_for_timeout(step_delay_ms)


def record():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-proxy-server"])
        ctx = browser.new_context(
            viewport=VIEWPORT,
            record_video_dir=str(OUTPUT_DIR),
            record_video_size=VIEWPORT,
        )
        page = ctx.new_page()
        page.goto(MARIMO_URL, wait_until="networkidle", timeout=30000)
        time.sleep(LOAD_WAIT_S)

        shot_start = time.monotonic()
        def wait_until(t):
            wait = t - (time.monotonic() - shot_start)
            if wait > 0: time.sleep(wait)

        # State 1: intro 静帧
        wait_until(T_S2)

        # State 2: baseline
        set_slider(page, SLIDER_RATING, 8.5)
        set_slider(page, SLIDER_LEAD, 9.5)
        set_slider(page, SLIDER_K, 5)
        wait_until(T_S3)

        # State 3: dislike
        set_slider(page, SLIDER_RATING, 2.9)
        set_slider(page, SLIDER_LEAD, 3.5)
        wait_until(T_S4)

        # State 4: single
        set_slider(page, SLIDER_K, 1)
        wait_until(T_END)

        page.close(); ctx.close(); browser.close()

        # ffmpeg 倒切 + crop + 转 mp4
        webm = sorted(OUTPUT_DIR.glob("*.webm"), key=lambda p: p.stat().st_mtime)[-1]
        raw = OUTPUT_DIR / "eXX_shotYY_raw.webm"
        if raw.exists(): raw.unlink()
        webm.rename(raw)

        out = OUTPUT_DIR / "eXX_shotYY_cropped.mp4"
        subprocess.run([
            "ffmpeg", "-sseof", f"-{int(T_END)}", "-i", str(raw),
            "-vf", "crop=1198:600:362:34",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-y", str(out),
        ], check=True, capture_output=True)
        print(f"✓ {out}")


if __name__ == "__main__":
    record()
```

## 6. 产物 README 模板

```md
# eXX Demo 录屏

## 文件

| 文件 | 规格 | 与 TTS 对齐 |
|---|---|---|
| `shotYY.mp4` | 1198×600 · H.264 · {T_END}s · ~XMB | ↔ `astral-pipeline/mlXX/tts/shotYY.wav` 同长 |

## 时间窗口

| 时间 | State | 画面 |
|---|---|---|
| 0 - T_S2 | intro | ... |
| T_S2 - T_S3 | baseline | ... |
| T_S3 - T_S4 | dislike | ... |
| T_S4 - T_END | single | ... |

## 下游用法

```bash
ffmpeg -i shotYY.mp4 -i ../../../astral-pipeline/mlXX/tts/shotYY.wav \
       -c:v copy -c:a aac -shortest shotYY_with_audio.mp4
```

## 推下游（可选）

```bash
cp shotYY.mp4 ~/projects/astral-pipeline/mlXX/recording/
touch ~/projects/astral-pipeline/mlXX/recording/.ready
```

## 重新生成

```bash
cd 01-ML/0X-ChY/XX-section/demos
marimo run XX-name.py --headless --port 2750 --no-token &
cd ../../../scripts/_recording
python3 eXX_shotYY.py
```
```

## 7. 失败模式

| 失败 | 教训 |
|---|---|
| `curl localhost:2750` 返回 000 | macOS socks5 拦截，用 `lsof` 不用 curl |
| `ERR_PROXY_CONNECTION_FAILED` | chromium 继承代理，必须 `--no-proxy-server` |
| `input[type="range"]` 找不到 | marimo 是 web component，用 `marimo-slider` + `[role="slider"]` |
| `slider.fill("8.5")` 没反应 | ARIA slider 不接 fill，必须 focus + 键盘 |
| 散点图变橙色 | Altair 多层冲突，加 `.resolve_scale(color='independent')` |
| `-ss N` 切多/少 | webm 加载头长度不定，用 `-sseof -N` 倒切 |
| 录屏比 TTS 长 1-2s | 同上 |
| 拖滑块"瞬移" | step_delay_ms 太小，30 ms 体验最佳 |
| 视频拉伸变形 | crop 宽高必须偶数（H.264 编码要求） |
