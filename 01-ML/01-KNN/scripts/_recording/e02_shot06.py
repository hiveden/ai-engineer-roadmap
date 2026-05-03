"""
E02 shot06 录屏 · 03-k-tuning demo · K=11 → K=51 平台期
T_END = 86.525s

cue（来自 ml02/tts/subtitles.json shot06）:
   1.78s "先到 11"     → 已经在 11，但 SETUP 需要从 k=1 切到 k=11（承接 shot05）
  61.50s "继续拖到 51" → set k=51

起始 state（每个 shot 独立 context，但承接逻辑上 shot05 末态是 k=1）
SETUP（不计入 record，由 ffmpeg -sseof 切掉）：
  载入 demo（k=11 默认）→ 立即拖到 k=1 → SETUP 结束 → shot_start

主操作：
  T_K11 = 1.78  → set k=11
  T_K51 = 61.50 → set k=51
"""

from playwright.sync_api import sync_playwright, Page
import time, subprocess
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
MARIMO_URL = "http://localhost:2751"
VIEWPORT = {"width": 1920, "height": 1080}
LOAD_WAIT_S = 4.0
T_END = 86.525

T_K11 = 1.78
T_K51 = 61.50

SLIDER_K = 0
SLIDER_RATING = 1
SLIDER_LEAD = 2

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

        # SETUP: k 从默认 11 拖到 1（承接 shot05 末态）
        print("→ SETUP: k=1")
        set_slider(page, SLIDER_K, 1)
        # 等 marimo 重算稳定
        time.sleep(1.0)

        # SETUP 完成，shot 计时起点
        shot_start = time.monotonic()

        def wait_until(t, label=""):
            wait = t - (time.monotonic() - shot_start)
            if wait > 0:
                print(f"  → wait {wait:.2f}s for t={t:.2f} ({label})")
                time.sleep(wait)
            else:
                print(f"  ⚠ t={t:.2f}s 已过 {-wait:.2f}s")

        # State 1: k=1 静帧（极短，1.78s 时拖到 11）
        wait_until(T_K11, "K=11 cue")

        # State 2: k=11
        print("→ State 2: k=11")
        set_slider(page, SLIDER_K, 11)

        # 静帧到 K=51 cue
        wait_until(T_K51, "K=51 cue")

        # State 3: k=51
        print("→ State 3: k=51")
        set_slider(page, SLIDER_K, 51)

        wait_until(T_END, "shot 末")

        actual = time.monotonic() - shot_start
        print(f"\n→ 实际时长 {actual:.2f}s（目标 {T_END}s）")

        page.close(); ctx.close(); browser.close()

        webms = sorted(OUTPUT_DIR.glob("*.webm"), key=lambda p: p.stat().st_mtime)
        if not webms:
            print("⚠ 无 webm")
            return
        raw = OUTPUT_DIR / "e02_shot06_raw.webm"
        if raw.exists(): raw.unlink()
        webms[-1].rename(raw)

        out = OUTPUT_DIR / "e02_shot06_cropped.mp4"
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
