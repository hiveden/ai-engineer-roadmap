"""
E03 shot05 录屏 · 04-regression demo · 回归基线（53.014s）
口播主要是介绍 viridis 色阶、MAE 概念，画面基本静帧。
默认 state（k=7, weighted=off, rating=7.5, lead=8.0）就是 cue 16.26s "K 取 7" 想要的目标。

cue 时间戳（来自 ml03/tts/subtitles.json）:
  0.00s   "打开新的 demo"
  16.26s  "K 取 7" → 已默认 k=7（dummy 操作以确认/触发 reactive）
  17.42s  "看决策面"
  28.30s  "每一格的颜色对应这一格的预测完成度"
  53.014s shot 末

slider 索引（probe 实测）:
  idx 0 = k (1-80, step=1, val=7)
  idx 1 = new_rating (3-11, step=0.1, val=7.5)
  idx 2 = new_lead (3-12, step=0.1, val=8)
  marimo-switch[0] = weighted（默认 false）
"""
from playwright.sync_api import sync_playwright, Page
import time, subprocess
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

MARIMO_URL = "http://localhost:2752"
VIEWPORT = {"width": 1920, "height": 1080}
LOAD_WAIT_S = 4.0
T_END = 53.0  # shot05 总长 53.014s

# Cue 时间戳
T_K7_CUE = 16.26
T_DECISION_CUE = 17.42

SLIDER_K = 0
SLIDER_RATING = 1
SLIDER_LEAD = 2

# crop 实测：grid bbox 321,33,1278,1047 → 偶数对齐
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

        # SETUP：起始 state = 默认（k=7, weighted=off, rating=7.5, lead=8.0）
        # 默认就是这个值，但显式 set 一遍以保证幂等
        print("→ SETUP（默认 state · 静默）")
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

        # State: k=7 静帧到 shot 末
        # cue 16.26s 已经 k=7（默认），不需要操作。
        print("→ State: 静帧（k=7 默认） · 听口播介绍 viridis/MAE")
        wait_until(T_END, "shot 末")

        actual = time.monotonic() - shot_start
        print(f"\n→ shot 实际时长: {actual:.2f}s（目标 {T_END}s）")

        page.close(); ctx.close(); browser.close()

        webm_files = sorted(OUTPUT_DIR.glob("*.webm"), key=lambda p: p.stat().st_mtime)
        if not webm_files:
            print("⚠ 无 webm 输出"); return
        raw = OUTPUT_DIR / "e03_shot05_raw.webm"
        if raw.exists(): raw.unlink()
        webm_files[-1].rename(raw)

        out = OUTPUT_DIR / "e03_shot05_cropped.mp4"
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
