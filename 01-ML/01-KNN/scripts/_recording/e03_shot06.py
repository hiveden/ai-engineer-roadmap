"""
E03 shot06 录屏 · 04-regression demo · K=1 vs K=80 极端（45.314s）

承接 shot05 末态：k=7 → 这个 shot 内拖到 k=1，再拖到 k=80。
新 context 默认就是 k=7（demo 默认值），无需 SETUP。

cue 时间戳（ml03/tts/subtitles.json）:
  0.00s   "现在拖 K"
  3.16s   "K=1"   → set k=1（拖 7→1，6 步 · 30ms = 0.18s）
  16.22s  "K=80"  → set k=80（拖 1→80，79 步 · 30ms = 2.37s）
  45.314s shot 末

slider 索引：k=0 / rating=1 / lead=2
"""
from playwright.sync_api import sync_playwright, Page
import time, subprocess
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

MARIMO_URL = "http://localhost:2752"
VIEWPORT = {"width": 1920, "height": 1080}
LOAD_WAIT_S = 4.0
T_END = 45.3  # shot06 总长 45.314s

T_K1 = 3.16
T_K80 = 16.22

SLIDER_K = 0
SLIDER_RATING = 1
SLIDER_LEAD = 2

CROP = "1278:1046:321:34"


def set_slider(page: Page, idx: int, target: float, step_delay_ms: int = 30):
    sliders = page.locator('marimo-slider')
    slider = sliders.nth(idx)
    aria = slider.locator('[role="slider"]').first
    current = float(aria.get_attribute('aria-valuenow'))
    step = float(slider.get_attribute('data-step'))
    presses = round((target - current) / step)
    if presses == 0:
        return
    aria.focus()
    key = "ArrowRight" if presses > 0 else "ArrowLeft"
    for _ in range(abs(presses)):
        aria.press(key)
        if step_delay_ms > 0:
            page.wait_for_timeout(step_delay_ms)


def set_switch(page: Page, idx: int, want_on: bool):
    sw = page.locator('marimo-switch').nth(idx)
    button = sw.locator('[role="switch"]').first
    checked = button.get_attribute('aria-checked') == "true"
    if checked != want_on:
        button.click()


def record():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-proxy-server"])
        ctx = browser.new_context(
            viewport=VIEWPORT,
            record_video_dir=str(OUTPUT_DIR),
            record_video_size=VIEWPORT,
        )
        page = ctx.new_page()

        print("→ 打开 demo")
        page.goto(MARIMO_URL, wait_until="networkidle", timeout=30000)
        time.sleep(LOAD_WAIT_S)

        # SETUP：起始 state = k=7（默认）
        print("→ SETUP（k=7 默认 / 静默）")
        set_switch(page, 0, False)
        set_slider(page, SLIDER_K, 7, step_delay_ms=0)
        set_slider(page, SLIDER_RATING, 7.5, step_delay_ms=0)
        set_slider(page, SLIDER_LEAD, 8.0, step_delay_ms=0)
        time.sleep(0.5)

        shot_start = time.monotonic()

        def wait_until(t: float, label: str = ""):
            elapsed = time.monotonic() - shot_start
            wait = t - elapsed
            if wait > 0:
                print(f"  → 等到 t={t:.2f}s ({label}) · 等待 {wait:.2f}s")
                time.sleep(wait)
            else:
                print(f"  ⚠ t={t:.2f}s 已超时 {-wait:.2f}s")

        # State 1: 静帧 k=7 直到 cue "K=1"
        print("→ State 1: 静帧 k=7 → 等到 cue K=1")
        wait_until(T_K1, "cue K=1")

        # State 2: k=1（拖 6 步）
        print("→ State 2: k=7 → k=1（6 步）")
        set_slider(page, SLIDER_K, 1, step_delay_ms=30)
        wait_until(T_K80, "cue K=80")

        # State 3: k=80（拖 79 步 · ~2.37s）
        print("→ State 3: k=1 → k=80（79 步 · ~2.4s）")
        set_slider(page, SLIDER_K, 80, step_delay_ms=30)

        wait_until(T_END, "shot 末")

        actual = time.monotonic() - shot_start
        print(f"\n→ shot 实际时长: {actual:.2f}s（目标 {T_END}s）")

        page.close(); ctx.close(); browser.close()

        webm_files = sorted(OUTPUT_DIR.glob("*.webm"), key=lambda p: p.stat().st_mtime)
        if not webm_files:
            print("⚠ 无 webm 输出"); return
        raw = OUTPUT_DIR / "e03_shot06_raw.webm"
        if raw.exists(): raw.unlink()
        webm_files[-1].rename(raw)

        out = OUTPUT_DIR / "e03_shot06_cropped.mp4"
        print(f"\n→ ffmpeg 倒切 {T_END}s + crop {CROP}")
        subprocess.run([
            "ffmpeg",
            "-sseof", f"-{int(T_END)}",
            "-i", str(raw),
            "-vf", f"crop={CROP}",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-y", str(out),
        ], check=True, capture_output=True)
        size_mb = out.stat().st_size / 1024 / 1024
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(out),
        ], capture_output=True, text=True)
        duration = float(result.stdout.strip())
        print(f"✓ {out} · {duration:.2f}s · {size_mb:.1f} MB")


if __name__ == "__main__":
    record()
