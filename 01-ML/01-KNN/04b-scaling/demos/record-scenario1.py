"""
场景一自动化录屏：拖动视力滑块 → 预测不变（证明体重独裁）

前置：
    1. 启动 marimo: marimo run 02-knn-scaling.py --port 2718 --no-token
    2. pip install playwright && playwright install chromium

跑：python record-scenario1.py
输出：./recordings/<timestamp>/scenario1.webm（自动）
转 mp4：ffmpeg -i scenario1.webm scenario1.mp4
"""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

MARIMO_URL = "http://localhost:2718"
OUTPUT_DIR = Path("./recordings")
OUTPUT_DIR.mkdir(exist_ok=True)


async def record():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            record_video_dir=str(OUTPUT_DIR),
            record_video_size={"width": 1280, "height": 800},
            viewport={"width": 1280, "height": 800},
        )
        page = await context.new_page()

        # 1. 打开 marimo
        await page.goto(MARIMO_URL)
        await page.wait_for_load_state("networkidle")
        await asyncio.sleep(3)  # 等 marimo 完全渲染

        # 2. 定位三个滑块（按页面出现顺序：身高、体重、视力）
        sliders = await page.locator("input[type=range]").all()
        print(f"找到 {len(sliders)} 个滑块")
        # 假设新人 3 个滑块在最后 3 个（前面有 k 滑块）
        # 顺序通常: k, 身高, 体重, 视力
        if len(sliders) < 4:
            print("⚠️ 滑块数量不对，请检查 UI")
            return
        vision_slider = sliders[3]

        # 3. 把视力从 0.1 慢慢拖到 2.0（每 0.1 一步，共 19 帧）
        for i in range(20):
            value = round(0.1 + i * 0.1, 2)
            await vision_slider.evaluate(
                """(el, v) => {
                    el.value = v;
                    el.dispatchEvent(new Event('input', {bubbles: true}));
                    el.dispatchEvent(new Event('change', {bubbles: true}));
                }""",
                value,
            )
            await asyncio.sleep(0.4)

        # 4. 结尾停留 3 秒看最终结果
        await asyncio.sleep(3)

        await context.close()
        await browser.close()
        print(f"✅ 视频已保存到 {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(record())
