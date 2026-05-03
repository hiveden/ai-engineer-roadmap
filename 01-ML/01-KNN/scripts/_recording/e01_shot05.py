"""
E01 shot05 录屏 · 02-proximity demo · 4 states
按 TTS shot05 的 cue 词时间戳对齐，录屏总长 = 78s（与音频同步）

cue 词时间戳（来自 ml01/tts/subtitles.json）:
  3.94s   "屏幕上是一张二维散点图"     → state 1 intro 开始
  39.23s  "我们把评分调到 8.5"         → state 2 baseline 触发
  55.87s  "现在把新电影拖到上海堡垒"   → state 3 dislike 触发
  68.61s  "再把 k 调到 1"              → state 4 single 触发
  77.55s  shot 末

前置：
  cd 01-ML/01-KNN/01-intro/demos
  marimo run 02-proximity.py --headless --port 2750 --no-token &

跑：
  python3 e01_shot05.py
  → 输出 output/e01_shot05.webm（约 82s，含 4s 加载头）
  → ffmpeg 切前 4s + crop → 78s 1198×600 mp4
"""

from playwright.sync_api import sync_playwright, Page
import time
from pathlib import Path
import subprocess

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

MARIMO_URL = "http://localhost:2750"
VIEWPORT = {"width": 1920, "height": 1080}
LOAD_WAIT_S = 4.0   # marimo 加载等待（后面 ffmpeg 切掉）

# Cue 词目标时间（相对 shot_start = 加载完成时刻）
T_BASELINE = 39.23
T_DISLIKE = 55.87
T_SINGLE = 68.61
T_END = 78.0

# slider 索引
SLIDER_RATING = 0
SLIDER_LEAD = 1
SLIDER_K = 2


def set_slider(page: Page, idx: int, target: float, step_delay_ms: int = 30):
    sliders = page.locator('marimo-slider')
    slider = sliders.nth(idx)
    aria = slider.locator('[role="slider"]').first
    current = float(aria.get_attribute('aria-valuenow'))
    step = float(slider.get_attribute('data-step'))
    diff = target - current
    presses = round(diff / step)
    if presses == 0:
        return
    aria.focus()
    key = "ArrowRight" if presses > 0 else "ArrowLeft"
    for _ in range(abs(presses)):
        aria.press(key)
        if step_delay_ms > 0:
            page.wait_for_timeout(step_delay_ms)


def record_e01_shot05():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-proxy-server"])
        context = browser.new_context(
            viewport=VIEWPORT,
            record_video_dir=str(OUTPUT_DIR),
            record_video_size=VIEWPORT,
        )
        page = context.new_page()

        print("→ 打开 demo")
        page.goto(MARIMO_URL, wait_until="networkidle", timeout=30000)
        time.sleep(LOAD_WAIT_S)

        # 真正"shot 时刻 0"（录屏内的 t=4s）
        shot_start = time.monotonic()

        def wait_until(t: float, label: str = ""):
            """等到相对 shot_start 的 t 秒。"""
            elapsed = time.monotonic() - shot_start
            wait = t - elapsed
            if wait > 0:
                print(f"  → 等到 t={t:.2f}s ({label}) · 等待 {wait:.2f}s")
                time.sleep(wait)
            else:
                print(f"  ⚠ t={t:.2f}s 已超时 {-wait:.2f}s")

        # State 1: intro 静帧（直到 baseline cue）
        print("→ State 1: intro 静帧（等到 baseline cue）")
        wait_until(T_BASELINE, "baseline cue")

        # State 2: baseline (rating=8.5, lead=9.5, k=5)
        print("→ State 2: baseline rating=8.5 lead=9.5 k=5")
        set_slider(page, SLIDER_RATING, 8.5)
        set_slider(page, SLIDER_LEAD, 9.5)
        set_slider(page, SLIDER_K, 5)
        wait_until(T_DISLIKE, "dislike cue")

        # State 3: dislike (rating=2.9, lead=3.5, k=5)
        print("→ State 3: dislike rating=2.9 lead=3.5 k=5")
        set_slider(page, SLIDER_RATING, 2.9)
        set_slider(page, SLIDER_LEAD, 3.5)
        wait_until(T_SINGLE, "single cue")

        # State 4: single (k=1)
        print("→ State 4: single k=1")
        set_slider(page, SLIDER_K, 1)
        wait_until(T_END, "shot 末")

        actual_duration = time.monotonic() - shot_start
        print(f"\n→ shot 实际时长: {actual_duration:.2f}s（目标 78s）")

        page.close()
        context.close()
        browser.close()

        # 找最新 webm，重命名 + ffmpeg 切前 4s + crop + 转 mp4
        webm_files = sorted(OUTPUT_DIR.glob("*.webm"), key=lambda p: p.stat().st_mtime)
        if not webm_files:
            print("⚠ 未找到 webm 输出")
            return

        raw = OUTPUT_DIR / "e01_shot05_raw.webm"
        if raw.exists():
            raw.unlink()
        webm_files[-1].rename(raw)

        print(f"\n→ ffmpeg 切前 {LOAD_WAIT_S}s + crop 1198×600 + 转 mp4")
        out_mp4 = OUTPUT_DIR / "e01_shot05_cropped.mp4"
        # 方案 B：ffprobe 拿 webm 总长，再 -ss/-t 精确倒切 T_END 秒（避开 -sseof int() 丢小数）
        probe = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(raw),
        ], capture_output=True, text=True)
        total = float(probe.stdout.strip())
        subprocess.run([
            "ffmpeg",
            "-ss", str(total - T_END),
            "-t", str(T_END),
            "-i", str(raw),
            "-vf", "crop=1198:600:362:34",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-y", str(out_mp4),
        ], check=True, capture_output=True)

        size_mb = out_mp4.stat().st_size / 1024 / 1024
        # ffprobe 看时长
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(out_mp4),
        ], capture_output=True, text=True)
        duration = float(result.stdout.strip())

        print(f"✓ {out_mp4} · {duration:.2f}s · {size_mb:.1f} MB")


if __name__ == "__main__":
    record_e01_shot05()
