"""
E02 shot09 录屏 · 03-k-tuning demo · 加权破远近差异
T_END = 69.291s

cue（来自 ml02/tts/subtitles.json shot09）:
  26.06s "K 拖到 21"          → set k=21
  27.02s "把距离加权开关打开" → set weighted=on

起始 state：k=121（承接 shot07 末），weighted=off
SETUP：load → k=11 默认 → 拖到 121（不计入 record）
"""

from playwright.sync_api import sync_playwright, Page
import time, subprocess
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
MARIMO_URL = "http://localhost:2751"
VIEWPORT = {"width": 1920, "height": 1080}
LOAD_WAIT_S = 4.0
T_END = 69.291

T_K21 = 26.06
T_WEIGHTED = 27.02

SLIDER_K = 0
SLIDER_RATING = 1
SLIDER_LEAD = 2
SWITCH_WEIGHTED = 0

CROP = "1400:740:260:34"


def set_slider(page: Page, idx: int, target: float, step_delay_ms: int = 30):
    sliders = page.locator("marimo-slider")
    slider = sliders.nth(idx)
    aria = slider.locator('[role="slider"]').first
    current = float(aria.get_attribute("aria-valuenow"))
    step = float(slider.get_attribute("data-step"))
    presses = round((target - current) / step)
    if presses == 0:
        return
    aria.focus()
    key = "ArrowRight" if presses > 0 else "ArrowLeft"
    for _ in range(abs(presses)):
        aria.press(key)
        if step_delay_ms > 0:
            page.wait_for_timeout(step_delay_ms)


def set_switch(page: Page, idx: int, target: bool):
    sw = page.locator("marimo-switch").nth(idx)
    btn = sw.locator("button").first
    checked = btn.get_attribute("aria-checked") == "true"
    if checked != target:
        btn.click()
        page.wait_for_timeout(120)


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

        # SETUP: k=11 → 121（承接 shot07 末）
        print("→ SETUP: k=121")
        set_slider(page, SLIDER_K, 121, step_delay_ms=10)
        time.sleep(2.0)  # k=121 网格重算较慢

        shot_start = time.monotonic()

        def wait_until(t, label=""):
            wait = t - (time.monotonic() - shot_start)
            if wait > 0:
                print(f"  → wait {wait:.2f}s for t={t:.2f} ({label})")
                time.sleep(wait)
            else:
                print(f"  ⚠ t={t:.2f}s 已过 {-wait:.2f}s")

        # State 1: k=121 静帧（口播讲加权概念）
        wait_until(T_K21, "K=21 cue")

        # State 2: k=21（121 → 21，差 100，step=2，50 步 ≈ 1.5s）
        print("→ State 2: k=21")
        set_slider(page, SLIDER_K, 21)

        # T_WEIGHTED = 27.02，与 T_K21 间隔 < 1s，可能已经超过；wait_until 会跳过
        wait_until(T_WEIGHTED, "weighted on cue")

        # State 3: weighted on
        print("→ State 3: weighted=on")
        set_switch(page, SWITCH_WEIGHTED, True)

        wait_until(T_END, "shot 末")

        actual = time.monotonic() - shot_start
        print(f"\n→ 实际时长 {actual:.2f}s（目标 {T_END}s）")

        page.close(); ctx.close(); browser.close()

        webms = sorted(OUTPUT_DIR.glob("*.webm"), key=lambda p: p.stat().st_mtime)
        if not webms:
            print("⚠ 无 webm")
            return
        raw = OUTPUT_DIR / "e02_shot09_raw.webm"
        if raw.exists(): raw.unlink()
        webms[-1].rename(raw)

        out = OUTPUT_DIR / "e02_shot09_cropped.mp4"
        subprocess.run([
            "ffmpeg",
            "-sseof", f"-{T_END:.3f}",
            "-i", str(raw),
            "-t", f"{T_END:.3f}",
            "-vf", f"crop={CROP}",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-y", str(out),
        ], check=True, capture_output=True)

        size_mb = out.stat().st_size / 1024 / 1024
        dur = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(out)],
            capture_output=True, text=True
        ).stdout.strip())
        print(f"✓ {out} · {dur:.3f}s · {size_mb:.1f} MB")


if __name__ == "__main__":
    record()
