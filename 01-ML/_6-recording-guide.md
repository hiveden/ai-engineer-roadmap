# 录屏指南（skill）

> **Step 6** · TTS 就绪后用 Playwright + ffmpeg 制作带 highlight 的 demo 录屏 mp4
> 前置：`astral-pipeline/mlXX/tts/.ready` 存在

## 何时用

- TTS 就绪、demo 已固化、要做 `recording/shotYY.mp4`
- 设计 highlight overlay 同步字级口播 cue
- 调 ffmpeg 倒切 / CROP 偶数 / 视频对齐 TTS 长度
- chart 全挤同行 / 缩豆 / 控件溢出顶部 等录屏前问题排查

## 索引（按需取读）

| § | 内容 | 何时读 |
|---|---|---|
| 1 | 前置 checklist | 启动前 |
| 2.1-2.4 | macOS 代理 / slider DOM / 端口 / viewport-CROP | 首次写新录屏 |
| 2.5 | ffmpeg 倒切（精度毫秒级）| 视频对齐 TTS |
| 2.6 | Altair 多层颜色冲突 | 颜色变橙时 |
| 2.7 | 时间精确对齐（subtitles cue）| 写 wait_until |
| 2.8 | marimo-dropdown shadow DOM | 用 dropdown 时 |
| **2.9** | **highlight overlay 同步口播（核心）** | 设计 highlight |
| 2.9.5 | marimo run cwd 陷阱（layout_file 相对路径）| chart 全挤同行 |
| 2.9.6 | chart 自适应坐标系（防缩豆）| 交互小尺度缩豆 |
| 2.9.7 | 控件嵌套消除（防溢出 CROP 顶部）| CROP 顶部缺内容 |
| 3 | 标准动作 8 步 | 流程速查 |
| 4 | 验证清单 | 输出前自检 |
| 5 | 录屏脚本模板 | 写新 eXX_shotYY.py |
| 6 | 产物 README 模板 | 拷贝产物时 |
| 7 | 失败模式 | 遇到问题时 |

> demo 设计原则（cell 拆分 / grid layout / mo.md 陷阱）→ 见 [`_2-demo-guide.md`](./_2-demo-guide.md)

---

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

### 2.2 marimo slider / switch DOM

slider 和 switch 是**两个独立的 web component**，索引互不重叠：`mo.ui.slider` → `<marimo-slider>` 集合；`mo.ui.switch` → `<marimo-switch>` 集合。**不要**假设 `marimo-slider` 含所有控件。

```html
<marimo-slider data-step="0.1" data-start="2" data-stop="10">
  <span role="slider" aria-valuenow="8.5">

<marimo-switch>
  <button role="switch" aria-checked="false">
```

**slider** 不是 `input[type="range"]`，**不能** `fill()`。用 focus + 键盘箭头：
```python
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
```

**switch** 用 `click()` 切换 + 等渲染稳定（k 网格随开关重算耗时不定，800ms 经验值稳）：
```python
def set_switch(page, idx, target_value: bool):
    switches = page.locator('marimo-switch')
    btn = switches.nth(idx).locator('[role="switch"]').first
    current = btn.get_attribute('aria-checked') == 'true'
    if current != target_value:
        btn.click()
        page.wait_for_timeout(800)  # 渲染稳定
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

#### 2.4.1 严格 16:9 demo 简化（e09 案例）

如果 demo 是严格 16:9 一屏满（maxWidth=1280 + 总高 36 行 = 720px，见 `_2-demo-guide §4.2`）：

```python
VIEWPORT = {"width": 1280, "height": 720}
record_video_size = VIEWPORT       # 实测：device_scale=2 不影响视频尺寸
CROP = "1280:720:0:0"              # 等于不 crop，直接全屏
```

省掉 probe `boundingBox` + 居中偏移计算。Playwright `record_video_size` 取 viewport 原值（不被 device_scale_factor 放大），device_scale=2 只提升渲染 DPI、不增加视频文件像素。

### 2.5 ffmpeg 倒切（精度敏感）

webm 加载头长度不确定 → 从尾倒切。**不要**用 `int(T_END)`，T_END=45.3 会被截成 45，丢 300ms。

**推荐方案 B · ffprobe 取总长 + `-ss`/`-t`**（精度毫秒级，e03 实战已验证）：
```python
result = subprocess.run([
    "ffprobe", "-v", "error", "-show_entries", "format=duration",
    "-of", "default=noprint_wrappers=1:nokey=1", str(raw),
], capture_output=True, text=True)
total = float(result.stdout.strip())

subprocess.run([
    "ffmpeg",
    "-ss", str(total - T_END),
    "-t", str(T_END),
    "-i", str(raw),
    "-vf", f"crop={CROP}",
    "-c:v", "libx264", "-crf", "18", "-preset", "fast",
    "-y", str(out_mp4),
], check=True, capture_output=True)
```

**备选方案 A · 浮点 -sseof**（一行简洁，但有些 ffmpeg 旧版本对 `-sseof` 浮点支持不稳）：
```python
"-sseof", f"-{T_END}",  # 注意：不是 f"-{int(T_END)}"
```

### 2.6 Altair 多层颜色冲突
```python
chart = (lines + pts + text + query).resolve_scale(color="independent")
```
否则 layer 间 color encoding 互相覆盖。录屏前先截图验证颜色对。

### 2.7 时间精确对齐
读 `subtitles.json` 找 cue 词时间戳，每个 state 操作完后 `wait_until(t)` 等到下一个 cue。录屏总长 = TTS shot 总长。

### 2.8 marimo-dropdown shadow DOM

dropdown 自身 `display:contents` + `width=0 height=0`——常规 `click()` 报 `element is not visible`。原生 `<select>` 在 shadow root 内部。

**操作方式**：`evaluate` 进 shadow root 设 value + dispatch change：

```python
def set_dropdown(page, idx: int, value: str):
    page.evaluate("""(args) => {
        const dd = document.querySelectorAll('marimo-dropdown')[args.idx];
        const sel = dd.shadowRoot.querySelector('select');
        sel.value = args.value;
        sel.dispatchEvent(new Event('change', {bubbles: true}));
    }""", {"idx": idx, "value": value})
```

代价：值瞬变，无菜单展开动画。**录屏补偿**：切前蓝闪 dropdown 200ms 提示"用户在操作"。

### 2.9.5 marimo run 必须设 cwd（layout_file 相对路径陷阱）

`layout_file="layouts/XX.grid.json"` 是相对 demo 文件的路径。`marimo run` 默认 cwd=Python 启动目录（如 `_recording/`），相对路径解析失败 → **grid 完全不生效但不报错**：所有 chart 挤在同一行（看起来像 css 渲染 bug）。

```python
proc = subprocess.Popen(
    ["marimo", "run", str(DEMO), "--headless", ...],
    cwd=str(DEMO.parent),  # 关键：让 layout_file 相对路径生效
)
```

**症状**：probe 实测 N 个 canvas 全在同一 y（应该按 grid 分散）→ 立刻检查 cwd。

### 2.9.6 chart 自适应坐标系（防"缩豆"）

固定 `domain=[0, 10]` 在小尺度交互时（如 A/B 滑动到 (8.5, 9.5)/(8.4, 9.0)，d=0.5）→ 等距线圆 / 路径缩成右上角小豆，占画面 ~5%。

**统一公式**（左右两图 + 多 chart 协同时建议一致）：
```python
_W = max(d_man, d_euc, d_che, 0.5) * 1.25  # 上界 + padding，0.5 是最小 W 防过 zoom
_base_x = alt.Scale(domain=[a[0] - _W, a[0] + _W])
_base_y = alt.Scale(domain=[a[1] - _W, a[1] + _W])
# 所有 mark 加 clip=True，所有 layer 显式 scale=_base_x/_base_y
```

**为什么 W 用 d_man 上界**：L1 ≥ L2 ≥ L∞，d_man 是最大半径；p<1 凹星形 d_min 可能超 d_man，所以 max 含 d_min。

**好处**：
- 多 chart 共用同 domain → 观众视觉对齐（左图在哪、右图同一区域）
- 操作单一参数（如 p）时 W 由 d_man 主导不变 → 坐标轴不跳

**副作用**：A 拖动时坐标轴跟着平移——可接受，因为 highlight 黄框继续指向 chart 容器（位置不变），观众视线不丢。

### 2.9.7 控件嵌套是溢出顶部 CROP 的元凶

`mo.hstack([mo.vstack([分组标题, slider, slider]), ...])` 让控件 cell 高度从 ~30px 撑到 ~100px，CROP 默认从 y=80 起截 → 分组标题溢出顶部。

**修复**：
- 平铺单层 `mo.hstack([s1, s2, s3, s4, s5], widths=[1,1,1,1,1])` → cell 高 ~30px
- 中文 label 较长（如"闵可夫斯基 p"）给 widths 多分点：`[1,1,1,1,1.4]` 防换行
- CROP y=0 起 + h=640 完整保留控件区（防止任何残留溢出）

### 2.9 highlight overlay 同步口播（关键）

**录屏 ≠ 操作演示，是视觉指挥棒**——TTS 说哪 highlight 哪。e11a 第 1 版只切 3 次 dropdown，0~53s 全静帧错过"控件介绍 / 红行 / 绿行 / 数据卡"4 段口播——观众视线无引导，教学效果归零。

**实现**：`position: fixed` overlay + setTimeout 自动消失，不阻塞 marimo reactive。

```python
def highlight(page, x, y, w, h, duration_ms, color="#fbbf24"):
    page.evaluate("""(a) => {
        const o = document.createElement('div');
        o.style.cssText = `position:fixed;left:${a.x-6}px;top:${a.y-6}px;
          width:${a.w+12}px;height:${a.h+12}px;border:3px solid ${a.color};
          border-radius:8px;pointer-events:none;z-index:99999;
          box-shadow:0 0 24px ${a.color};opacity:0;transition:opacity .2s`;
        document.body.appendChild(o);
        requestAnimationFrame(() => o.style.opacity = '1');
        setTimeout(() => { o.style.opacity='0'; setTimeout(()=>o.remove(), 250); }, a.duration);
    }""", {"x":x,"y":y,"w":w,"h":h,"duration":duration_ms,"color":color})
```

**bbox 来源**：录屏前先跑 `eXX_target_probe.py` 探测：
- 控件区 `marimo-slider/dropdown` → `shadowRoot.querySelector('.mo-label')` 整体可视容器
- chart `vega-embed` 整体 bbox → 内部双行用 padding ~30px 缩边、二分上下
- 数据卡 → `[style*='f8fafc']` 或含特定文字的 div

**marimo run 模式 DOM 反直觉**（e06 实战）：
- `marimo-cell` 选择器在 run 模式下**不存在**（仅 edit 有）→ 用 canvas / [role='slider'] / 文字定位
- `marimo-slider` 是 `display:contents` → `bounding_box()` 返回 None → 用内部 `[role='slider']` 拿 16×16 thumb，再扩展或用 5 个 thumb 的最小外接矩形作 controls_band
- 数据卡 inline style 在 mo.md 渲染中可能被处理 → `div[style*='f8fafc']` 找不到，改用 JS evaluate 找含特定文字的最深 div：
  ```js
  [...document.querySelectorAll('div')]
    .filter(d => d.textContent.includes('实时距离') && d.textContent.length < 600)
    .sort((a, b) => a.textContent.length - b.textContent.length)[0]
  ```

**时间戳来源**：`subtitles.json` 字级时间，每个 cue 词触发对应 highlight。e11a shot06 39 条字幕 → ~12 个视觉动作（远不止 5 个 state）。

**色码约定**：

| 色 | 用途 |
|---|---|
| `#fbbf24` 黄 | 通用 highlight（控件、数据卡）|
| `#dc2626` 红 | 数据红行（口播说"红色"） |
| `#16a34a` 绿 | 数据绿行（口播说"绿色"） |
| `#3b82f6` 蓝 | "用户在操作"提示（切控件前 200ms 短闪） |

## 3. 标准动作 8 步

1. 读 `script.json` segment {YY} 的 text + notes（states + cue 词）
2. 读 `subtitles.json` shot{YY} 找 cue 词精确时间戳
3. 启 `marimo run XX.py --headless --port 2750 --no-token`
4. Probe DOM（首次写新 demo 录屏脚本必做）：
   ```python
   for i, s in enumerate(page.locator('marimo-slider').all()):
       print(i, s.get_attribute('data-label'), s.get_attribute('data-step'))
   ```
   **同时跑 highlight target probe**（按 §2.9 探测所有目标 bbox：控件 `.mo-label` shadow / chart `vega-embed` / 数据卡 `[style*='f8fafc']` / 红绿行二分），写到 `eXX_targets.json` 给录屏脚本读。
5. 写 `_recording/eXX_shotYY.py`（套 §5 模板）
6. 跑 → ffmpeg 切边
7. 拷产物：`cp output/eXX_shotYY_cropped.mp4 ../eXX-期名/recording/shotYY.mp4`
8. 写 `eXX-期名/recording/README.md`（§6 模板）

## 4. 验证清单

- [ ] 视频时长 = TTS shot 时长（差异 < 50ms，ffprobe 验）
- [ ] 裁剪尺寸正确（默认 1198×600；不同 maxWidth 实测）
- [ ] 颜色对（无 Altair 冲突）
- [ ] 滑块拖动可见（不是瞬移，step_delay_ms ≥ 30）
- [ ] 每 state 稳定 ≥ 1.5s
- [ ] 末尾不抖动（最后 state 设置完到 T_END 是静帧）
- [ ] **画面动作覆盖口播每段**（关键）：subtitles 字级每个 cue 都有视觉响应（highlight / 切控件 / move），多秒静帧错过讲解 = BLOCK。e11a 第 1 版 0~53s 全静帧只切 dropdown 是反例
- [ ] probe 输出 N 个 canvas y 坐标分散（grid 生效）—— 全挤同一 y 立刻查 `cwd=DEMO.parent`
- [ ] 交互后 chart 在最小尺度（如 d=0.5）形状仍占画面 ≥ 30%（自适应坐标系生效）
- [ ] 控件区高度 ≤ 50px（无嵌套 vstack）+ 中文 label 不换行

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


def set_switch(page, idx, target_value: bool):
    switches = page.locator('marimo-switch')
    btn = switches.nth(idx).locator('[role="switch"]').first
    current = btn.get_attribute('aria-checked') == 'true'
    if current != target_value:
        btn.click()
        page.wait_for_timeout(800)  # 渲染稳定


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
        # 方案 B：ffprobe 拿总长，-ss/-t 精确倒切（毫秒级）
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(raw),
        ], capture_output=True, text=True)
        total = float(probe.stdout.strip())
        subprocess.run([
            "ffmpeg",
            "-ss", str(total - T_END),
            "-t", str(T_END),
            "-i", str(raw),
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
| switch 控件被 `marimo-slider` locator 漏掉 | switch 是独立 `<marimo-switch>`，索引单独数；不在 marimo-slider 集合里 |
| 散点图变橙色 | Altair 多层冲突，加 `.resolve_scale(color='independent')` |
| `-ss N` 切多/少 | webm 加载头长度不定，用 `-sseof -N` 倒切 |
| `-sseof -{int(T_END)}` 丢小数（45.3 → 45） | 用 `-sseof -<float>` 或 ffprobe + `-ss / -t`（推荐） |
| 录屏比 TTS 长 1-2s | 同上 |
| 拖滑块"瞬移" | step_delay_ms 太小，30 ms 体验最佳 |
| 视频拉伸变形 | crop 宽高必须偶数（H.264 编码要求） |
| 多 chart 全挤同一行（grid 失效） | `marimo run` 没设 `cwd=DEMO.parent` → `layout_file` 相对路径解析失败 |
| chart 在交互小尺度下缩豆（5%） | 固定 `domain=[0,10]` 不适配 → 自适应 `[a±W, W=max(d_*)·1.25]` + clip=True |
| 控件区溢出 CROP 顶部 | 嵌套 `hstack([vstack([标题, slider]), ...])` 撑高 cell → 平铺单层 hstack |
| 中文 slider label 换行（"闵可夫斯基 p"） | widths 平均分配空间不够 → `widths=[1,1,1,1,1.4]` 给长 label 多分 |
| `div[style*='f8fafc']` 找不到数据卡 | mo.md 渲染处理 inline style → 用 JS textContent 反查最深含特定文字的 div |
| `marimo-cell` 选择器 count=0 | run 模式不渲染 marimo-cell（仅 edit 有）→ 用 canvas / [role='slider'] / 文字定位 |
| `marimo-slider.bounding_box()` 返回 None | display:contents → 用内部 `[role='slider']` thumb 或父 widget 容器 |
| state 1 静帧形状被坐标系裁切 | 改 demo 默认值（如 A=(2,3) → A=(5,5)）比改渲染逻辑成本更低 |
| demo 默认值升级为"开机演示状态"（如 e09 默认 = StandardScaler）→ 录屏 cue "切到 Standard"动作消失（已经是 Standard 了）| 录屏脚本 SETUP 阶段先把状态退回口播起点（dropdown→None），让 cue 时间点的切换动作可见。SETUP 不计入 shot 时间，靠 ffmpeg 倒切丢前面 |
| `marimo run` subprocess.Popen 没设 cwd → layout_file 相对路径解析失败但不报错（grid 完全失效，所有 cell 挤成一列）| `cwd=str(DEMO_PATH.parent)`（同 §2.9.5）|
| marimo dropdown 在 light DOM（version ≥ 0.23）下用 `marimo-dropdown select` selector 直接 `select_option(label=...)` 即可，不需要 evaluate 进 shadow root（§2.8 是旧版方案，新版不适用）| 优先试 `page.locator('marimo-dropdown select').select_option(label=label)`；不工作再回 §2.8 evaluate 方案 |
