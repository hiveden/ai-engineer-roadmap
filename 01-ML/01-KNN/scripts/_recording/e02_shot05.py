"""
E02 shot05 录屏 · 03-k-tuning demo · K=1 过拟合
T_END = 45.5s

cue（来自 ml02/tts/subtitles.json shot05）:
  0.82s "K 拖到 1"  → set k=1
  之后 k=1 静帧到 45.5s

起始 state：默认（k=11, weighted=off, rating=7.5, lead=8.0）
"""

from playwright.sync_api import sync_playwright, Page
import time, subprocess
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
MARIMO_URL = "http://localhost:2751"
VIEWPORT = {"width": 1920, "height": 1080}
LOAD_WAIT_S = 4.0
T_END = 45.539  # 来自 durations.json shot05

T_K1 = 0.82  # cue "K 拖到 1"

# slider idx（probe 结果：3 个 slider）
SLIDER_K = 0
SLIDER_RATING = 1
SLIDER_LEAD = 2

# crop（probe boundingBox）
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

        # 起始 state 是默认（k=11, weighted=off），无需 SETUP
        shot_start = time.monotonic()

        def wait_until(t, label=""):
            wait = t - (time.monotonic() - shot_start)
            if wait > 0:
                print(f"  → wait {wait:.2f}s for t={t:.2f} ({label})")
                time.sleep(wait)
            else:
                print(f"  ⚠ t={t:.2f}s 已过 {-wait:.2f}s")

        # State 1: k=11 静帧（极短，0.82s 后即触发）
        wait_until(T_K1, "K=1 cue")

        # State 2: k=1（K 滑块从 11 → 1，每步 -2，共 5 步）
        print("→ State 2: k=1")
        set_slider(page, SLIDER_K, 1)

        # 静帧到末尾
        wait_until(T_END, "shot 末")

        actual = time.monotonic() - shot_start
        print(f"\n→ 实际时长 {actual:.2f}s（目标 {T_END}s）")

        page.close(); ctx.close(); browser.close()

        webms = sorted(OUTPUT_DIR.glob("*.webm"), key=lambda p: p.stat().st_mtime)
        if not webms:
            print("⚠ 无 webm")
            return
        raw = OUTPUT_DIR / "e02_shot05_raw.webm"
        if raw.exists(): raw.unlink()
        webms[-1].rename(raw)

        out = OUTPUT_DIR / "e02_shot05_cropped.mp4"
        # ffmpeg -sseof 倒切 T_END 秒 + crop（用浮点 -t 精确控时长）
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
