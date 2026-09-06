from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright


BASE_URL = "http://127.0.0.1:7860/"
ARTIFACTS = Path(__file__).resolve().parents[1] / "test_artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def _open(page):
    page.goto(BASE_URL, wait_until="networkidle", timeout=60_000)
    page.get_by_text("PATH-Mat Research Studio", exact=True).wait_for(timeout=20_000)


def test_desktop_main_flow_and_physical_gate():
    console_errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
        _open(page)
        page.screenshot(path=str(ARTIFACTS / "desktop_before.png"), full_page=True)

        assert page.get_by_role("button", name="开始分析").is_visible()
        page.get_by_role("button", name="开始分析").click()
        page.get_by_text("分析完成：两种可能结果很接近", exact=True).wait_for(timeout=60_000)
        assert page.get_by_text("分析结果", exact=True).count() >= 1

        page.get_by_text("资料与测试选项", exact=True).click()
        page.get_by_label("模拟：请求控制设备", exact=True).check()
        page.get_by_role("button", name="开始分析").click()
        page.get_by_text("当前请求包含设备操作", exact=True).wait_for(timeout=60_000)
        page.screenshot(path=str(ARTIFACTS / "desktop_physical_gate.png"), full_page=True)

        assert not [item for item in console_errors if "favicon" not in item.lower()]
        browser.close()


def test_mobile_layout_and_run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
        _open(page)
        page.screenshot(path=str(ARTIFACTS / "mobile_before.png"), full_page=True)
        overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        assert overflow <= 2
        page.get_by_role("button", name="开始分析").click()
        page.get_by_text("分析完成：两种可能结果很接近", exact=True).wait_for(timeout=60_000)
        assert page.get_by_text("四个问题，直接看答案", exact=True).is_visible()
        page.screenshot(path=str(ARTIFACTS / "mobile_after.png"), full_page=True)
        browser.close()


def test_3d_route_and_agent_pages_render_and_respond():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        _open(page)
        page.get_by_role("button", name="开始分析").click()
        page.get_by_text("分析完成：两种可能结果很接近", exact=True).wait_for(timeout=60_000)

        page.get_by_role("tab", name="02 · 3D 路线图", exact=True).click()
        route_plot = page.locator(".js-plotly-plot:visible").first
        route_plot.wait_for(timeout=30_000)
        assert route_plot.locator("canvas").count() >= 1
        box = route_plot.bounding_box()
        assert box
        page.mouse.move(box["x"] + box["width"] * 0.55, box["y"] + box["height"] * 0.5)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] * 0.68, box["y"] + box["height"] * 0.58, steps=8)
        page.mouse.up()
        page.screenshot(path=str(ARTIFACTS / "desktop_3d_route.png"), full_page=True)

        page.get_by_role("tab", name="03 · 协作流程", exact=True).click()
        agent_plot = page.locator(".js-plotly-plot:visible").first
        agent_plot.wait_for(timeout=30_000)
        assert agent_plot.locator("canvas").count() >= 1
        page.screenshot(path=str(ARTIFACTS / "desktop_3d_agents.png"), full_page=True)
        browser.close()


def test_model_guided_experiment_page_recalculates_first_round():
    console_errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
        _open(page)

        page.get_by_role("tab", name="05 · 模型指导实验", exact=True).click()
        page.get_by_role("button", name="生成模型指导实验计划", exact=True).click()
        page.get_by_text("模型指导的第一轮：先做 4 组条件", exact=True).wait_for(timeout=60_000)
        assert page.get_by_text("实际调用 EC-PathWM 18 次", exact=False).is_visible()
        assert page.get_by_text("模型排序分从 25.51 到 25.82", exact=False).is_visible()
        assert page.get_by_text("现有训练记录还不能比较空气和惰性气氛的差别", exact=False).is_visible()
        assert page.get_by_text("下载的 CSV 已经预留样品编号", exact=False).is_visible()
        assert page.get_by_text("25.82", exact=True).count() >= 1
        assert page.get_by_text("25.51", exact=True).count() >= 1
        assert page.get_by_text("第一轮先做", exact=True).count() >= 4
        download = page.locator("a[download]:visible").first
        download.wait_for(timeout=20_000)
        assert download.get_attribute("download", timeout=5_000).endswith(".csv")
        plot = page.locator(".js-plotly-plot:visible").first
        plot.wait_for(timeout=30_000)
        assert plot.locator("svg").count() >= 1
        page.screenshot(path=str(ARTIFACTS / "desktop_model_guided_experiments.png"), full_page=True)

        page.get_by_role("textbox").first.fill("950, 1100, 1250, 1400")
        page.get_by_role("button", name="生成模型指导实验计划", exact=True).click()
        page.get_by_text("你输入了 24 组候选条件", exact=False).wait_for(timeout=60_000)
        assert page.get_by_text("少安排 40 次实验", exact=False).is_visible()
        assert not [item for item in console_errors if "favicon" not in item.lower()]
        browser.close()


def test_model_guided_experiment_page_is_mobile_readable():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
        _open(page)
        page.locator("#pm-pages button:has(svg circle)").first.click()
        page.get_by_role("button", name="05 · 模型指导实验", exact=True).click()
        page.get_by_role("button", name="生成模型指导实验计划", exact=True).click()
        page.get_by_text("模型指导的第一轮：先做 4 组条件", exact=True).wait_for(timeout=60_000)
        plot = page.locator(".js-plotly-plot:visible").first
        plot.scroll_into_view_if_needed()
        plot.wait_for(timeout=30_000)
        assert plot.locator("svg").count() >= 1
        overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        assert overflow <= 2
        page.screenshot(path=str(ARTIFACTS / "mobile_model_guided_experiments.png"), full_page=True)
        browser.close()


def test_model_guided_experiment_page_is_readable_in_dark_theme():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 1000},
            device_scale_factor=1,
            color_scheme="dark",
        )
        page = context.new_page()
        _open(page)
        page.get_by_role("tab", name="05 · 模型指导实验", exact=True).click()
        page.get_by_role("button", name="生成模型指导实验计划", exact=True).click()

        result_heading = page.get_by_text("模型指导的第一轮：先做 4 组条件", exact=True)
        result_heading.wait_for(timeout=60_000)
        input_heading = page.get_by_text("设置候选实验范围", exact=True)
        atmosphere_label = page.get_by_text("空气", exact=True).first

        assert result_heading.evaluate("element => getComputedStyle(element).color") == "rgb(11, 31, 51)"
        assert input_heading.evaluate("element => getComputedStyle(element).color") == "rgb(11, 31, 51)"
        assert atmosphere_label.evaluate("element => getComputedStyle(element).color") == "rgb(11, 31, 51)"
        assert page.locator("a[download]:visible").count() == 1
        file_label = page.locator('[data-testid="block-label"]:visible').filter(
            has_text="下载实验安排与结果记录表（CSV）"
        )
        assert file_label.evaluate("element => getComputedStyle(element).color") == "rgb(11, 31, 51)"
        page.screenshot(
            path=str(ARTIFACTS / "desktop_model_guided_experiments_dark.png"),
            full_page=True,
        )
        context.close()
        browser.close()
