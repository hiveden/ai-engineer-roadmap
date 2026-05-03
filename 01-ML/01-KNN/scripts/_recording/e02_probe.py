"""Probe 03-k-tuning DOM @ port 2751"""
from playwright.sync_api import sync_playwright
import time

URL = "http://localhost:2751"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-proxy-server"])
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.goto(URL, wait_until="networkidle", timeout=30000)
    time.sleep(4.5)

    print("=== marimo-slider ===")
    sliders = page.locator("marimo-slider")
    n = sliders.count()
    print(f"count: {n}")
    for i in range(n):
        s = sliders.nth(i)
        label = s.get_attribute("data-label")
        step = s.get_attribute("data-step")
        start = s.get_attribute("data-start")
        stop = s.get_attribute("data-stop")
        aria = s.locator('[role="slider"]').first
        valuenow = aria.get_attribute("aria-valuenow")
        print(f"  [{i}] label={label} step={step} start={start} stop={stop} value={valuenow}")

    print("\n=== marimo-switch ===")
    switches = page.locator("marimo-switch")
    nsw = switches.count()
    print(f"count: {nsw}")
    for i in range(nsw):
        s = switches.nth(i)
        label = s.get_attribute("data-label")
        # 找内部 button / input
        inner_html = s.inner_html()[:300]
        print(f"  [{i}] label={label}")
        print(f"      inner: {inner_html}")
        # 尝试找 button[role=switch]
        btn = s.locator('button').first
        try:
            checked = btn.get_attribute("aria-checked")
            print(f"      button aria-checked={checked}")
        except Exception as e:
            print(f"      button err: {e}")

    print("\n=== grid 容器 boundingBox ===")
    # marimo grid 通常是 .react-grid-layout
    for sel in [".react-grid-layout", "#App", "main", "marimo-cell-output-area"]:
        loc = page.locator(sel).first
        if loc.count() > 0:
            try:
                box = loc.bounding_box()
                print(f"  {sel}: {box}")
            except Exception as e:
                print(f"  {sel}: err {e}")

    # 实测内容底部边界（找最后一个 cell 的底部）
    print("\n=== 所有 grid item bottom 边界 ===")
    items = page.locator(".react-grid-item")
    cnt = items.count()
    print(f"react-grid-item count: {cnt}")
    max_bottom = 0
    for i in range(cnt):
        try:
            box = items.nth(i).bounding_box()
            if box:
                bottom = box["y"] + box["height"]
                if bottom > max_bottom:
                    max_bottom = bottom
                if i < 3 or i >= cnt - 3:
                    print(f"  [{i}] x={box['x']:.0f} y={box['y']:.0f} w={box['width']:.0f} h={box['height']:.0f} bottom={bottom:.0f}")
        except Exception:
            pass
    print(f"max_bottom: {max_bottom}")

    page.screenshot(path="/tmp/e02_probe.png", full_page=False)
    print("\nscreenshot → /tmp/e02_probe.png")

    page.close(); ctx.close(); browser.close()
