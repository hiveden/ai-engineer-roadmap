"""
E03 shot07 录屏 · 04-regression demo · 残差图（29.453s）

承接 shot06 末态 k=80。新 context 需 SETUP（k=80 / 默认 weighted=off / rating=7.5 / lead=8.0）。
本 shot 不操作 slider，只把鼠标移到残差散点图上提升视觉聚焦。

cue 时间戳（ml03/tts/subtitles.json）:
  0.00s   "再看右边一张图叫残差图"
  21.78s  "残差图是回归任务最有用的诊断工具之一"
  29.453s shot 末

布局参考：残差散点图位于 grid 右侧（chart 宽 420 · grid 24 列中右侧约 10 列）。
策略：用 page.mouse.move 把鼠标移到残差散点图上，让某个圆点显示 tooltip 增加视觉焦点。
"""
from playwright.sync_api import sync_playwright, Page
import time, subprocess
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

MARIMO_URL = "http://localhost:2752"
VIEWPORT = {"width": 1920, "height": 1080}
LOAD_WAIT_S = 4.0
T_END = 29.5  # shot07 总长 29.453s

T_HOVER = 6.0       # 鼠标进入残差图区域
T_HOVER_MOVE = 14.0 # 慢慢挪到另一个点

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

        # SETUP：承接 shot06 末态 k=80（拖 79 步 · 静默）
        print("→ SETUP: k=7 → k=80（静默）")
        set_switch(page, 0, False)
        set_slider(page, SLIDER_RATING, 7.5, step_delay_ms=0)
        set_slider(page, SLIDER_LEAD, 8.0, step_delay_ms=0)
        set_slider(page, SLIDER_K, 80, step_delay_ms=0)
        time.sleep(0.8)  # 等图重渲

        # 找残差散点图位置（页面里第 2 个 vega/canvas，宽 420）
        # 用 evaluate 找所有 canvas 看哪个是残差图
        chart_centers = page.evaluate("""
        () => {
            const canvases = document.querySelectorAll('canvas');
            return Array.from(canvases).map(c => {
                const r = c.getBoundingClientRect();
                return {x: r.x, y: r.y, w: r.width, h: r.height,
                        cx: r.x + r.width/2, cy: r.y + r.height/2};
            });
        }
        """)
        print(f"→ canvases: {chart_centers}")
        # 残差图通常是第 2 个（决策面在前），且宽较小（~420）
        # 选取宽 < 500 的最右一个作为残差图
        residual_canvas = None
        for c in chart_centers:
            if 200 < c["w"] < 600 and c["h"] > 200:
                if residual_canvas is None or c["x"] > residual_canvas["x"]:
                    residual_canvas = c
        if residual_canvas is None and chart_centers:
            residual_canvas = chart_centers[-1]
        print(f"→ residual canvas: {residual_canvas}")

        shot_start = time.monotonic()

        def wait_until(t: float, label: str = ""):
            elapsed = time.monotonic() - shot_start
            wait = t - elapsed
            if wait > 0:
                print(f"  → 等到 t={t:.2f}s ({label}) · 等待 {wait:.2f}s")
                time.sleep(wait)
            else:
                print(f"  ⚠ t={t:.2f}s 已超时 {-wait:.2f}s")

        # 起始：鼠标停在残差图区域中部偏上（k=80 时点云聚成一条横线 ~y=均值）
        if residual_canvas:
            cx, cy = residual_canvas["cx"], residual_canvas["cy"]
            # 第 1 段：开场鼠标静止远离图
            page.mouse.move(50, 50)
            wait_until(T_HOVER, "进入残差图")

            # 第 2 段：移动到残差图中心
            print(f"→ mouse → ({cx:.0f}, {cy:.0f})")
            page.mouse.move(cx, cy, steps=20)
            wait_until(T_HOVER_MOVE, "横向移动")

            # 第 3 段：横向慢慢扫过点云（提示对角线/残差色）
            print(f"→ mouse 横扫")
            page.mouse.move(cx + residual_canvas["w"] * 0.2, cy - residual_canvas["h"] * 0.15, steps=30)

        wait_until(T_END, "shot 末")

        actual = time.monotonic() - shot_start
        print(f"\n→ shot 实际时长: {actual:.2f}s（目标 {T_END}s）")

        page.close(); ctx.close(); browser.close()

        webm_files = sorted(OUTPUT_DIR.glob("*.webm"), key=lambda p: p.stat().st_mtime)
        if not webm_files:
            print("⚠ 无 webm 输出"); return
        raw = OUTPUT_DIR / "e03_shot07_raw.webm"
        if raw.exists(): raw.unlink()
        webm_files[-1].rename(raw)

        out = OUTPUT_DIR / "e03_shot07_cropped.mp4"
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
