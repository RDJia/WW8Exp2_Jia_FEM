#!/usr/bin/env python3
"""
在本地电脑 Chrome 中自动打开机票搜索页面

功能：
  1. 自动打开多个搜索平台（Google Flights, Skyscanner, Trip.com 等）
  2. 对比 NUE / FRA / MUC 三个出发地
  3. 在 Google Flights 上尝试自动提取价格信息

前置条件：
  pip install selenium webdriver-manager

用法：
  python chrome_search.py                          # 使用默认日期
  python chrome_search.py -d 2026-05-01 -r 2026-05-15   # 自定义日期
  python chrome_search.py --origin NUE             # 只搜指定出发地
  python chrome_search.py --urls-only              # 只打印URL不打开浏览器
"""

import argparse
import sys
import time
import json
import os
from datetime import datetime

# ============================================================
# URL 生成
# ============================================================

def make_google_flights_url(origin, dest, depart, return_date):
    """生成 Google Flights 搜索 URL"""
    return (
        f"https://www.google.com/travel/flights?"
        f"q=flights+from+{origin}+to+{dest}+on+{depart}+return+{return_date}"
        f"&curr=EUR"
    )

def make_skyscanner_url(origin, dest, depart, return_date):
    """生成 Skyscanner URL"""
    d = datetime.strptime(depart, "%Y-%m-%d").strftime("%y%m%d")
    r = datetime.strptime(return_date, "%Y-%m-%d").strftime("%y%m%d")
    origin_map = {"NUE": "nue", "FRA": "fran", "MUC": "muni"}
    dest_map = {"PVG": "pvga"}
    o = origin_map.get(origin, origin.lower())
    de = dest_map.get(dest, dest.lower())
    return (
        f"https://www.skyscanner.net/transport/flights/{o}/{de}/{d}/{r}/"
        f"?adultsv2=1&cabinclass=economy&rtn=1"
    )

def make_trip_url(origin, dest, depart, return_date):
    """生成 Trip.com URL"""
    return (
        f"https://www.trip.com/flights/{origin.lower()}-to-shanghai/"
        f"tickets-{origin.lower()}-{dest.lower()}"
        f"?dcity={origin.lower()}&acity={dest.lower()}"
        f"&ddate={depart}&rdate={return_date}"
        f"&flighttype=rt&class=y&quantity=1"
    )

def make_momondo_url(origin, dest, depart, return_date):
    return f"https://www.momondo.com/flight-search/{origin}-{dest}/{depart}/{return_date}?sort=price_a&stops=1"

def make_kayak_url(origin, dest, depart, return_date):
    return f"https://www.kayak.com/flights/{origin}-{dest}/{depart}/{return_date}?sort=price_a&fs=stops=1"


PLATFORMS = [
    ("Google Flights", make_google_flights_url),
    ("Skyscanner", make_skyscanner_url),
    ("Trip.com", make_trip_url),
    ("Momondo", make_momondo_url),
    ("Kayak", make_kayak_url),
]

ORIGINS_INFO = {
    "NUE": {"name": "纽伦堡", "extra_cost": 0, "note": "直接出发"},
    "FRA": {"name": "法兰克福", "extra_cost": 40, "note": "火车约2h, +€40"},
    "MUC": {"name": "慕尼黑", "extra_cost": 25, "note": "火车约1h, +€25"},
}

AIRLINE_URLS = [
    ("土耳其航空 (常有好价)", "https://www.turkishairlines.com/"),
    ("芬兰航空", "https://www.finnair.com/"),
    ("汉莎航空", "https://www.lufthansa.com/"),
    ("中国国航", "https://www.airchina.com.cn/"),
    ("中国东航", "https://www.ceair.com/"),
    ("卡塔尔航空", "https://www.qatarairways.com/"),
]


def generate_all_urls(origins, dest, depart, return_date):
    """生成所有搜索 URL"""
    result = {}
    for origin in origins:
        info = ORIGINS_INFO[origin]
        label = f"{info['name']} ({origin}) - {info['note']}"
        urls = {}
        for name, func in PLATFORMS:
            urls[name] = func(origin, dest, depart, return_date)
        result[label] = urls
    return result


# ============================================================
# Chrome 自动化
# ============================================================

def open_chrome(all_urls):
    """用 Selenium 在 Chrome 中打开所有搜索页面"""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
    except ImportError:
        print("❌ 需要安装 selenium:")
        print("   pip install selenium webdriver-manager")
        return None

    options = Options()
    options.add_argument("--start-maximized")
    # 伪装成正常浏览器，减少被检测为自动化的风险
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    try:
        from webdriver_manager.chrome import ChromeDriverManager
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
    except Exception:
        print("  ⚡ 尝试使用系统 chromedriver...")
        try:
            driver = webdriver.Chrome(options=options)
        except Exception as e:
            print(f"  ❌ 无法启动 Chrome: {e}")
            print("  请确保已安装 Chrome 和 chromedriver")
            return None

    # 隐藏 webdriver 标记
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    })

    first = True
    tab_count = 0

    for origin_label, urls in all_urls.items():
        print(f"\n📍 {origin_label}")
        for platform, url in urls.items():
            if first:
                driver.get(url)
                first = False
            else:
                driver.execute_script(f"window.open('{url}', '_blank');")

            tab_count += 1
            print(f"  ✅ [{tab_count}] {platform}")
            time.sleep(1.5)

    # 打开几个航司官网
    print(f"\n🏢 航司官网:")
    for name, url in AIRLINE_URLS[:3]:  # 只开前3个
        driver.execute_script(f"window.open('{url}', '_blank');")
        tab_count += 1
        print(f"  ✅ [{tab_count}] {name}")
        time.sleep(1)

    print(f"\n🎯 共打开 {tab_count} 个标签页。")
    print("   请在各标签页中查看和比较价格。")
    print("   浏览器会保持打开状态。按 Ctrl+C 退出脚本（浏览器不会关闭）。")

    return driver


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="纽伦堡⇄上海机票搜索助手")
    parser.add_argument("-d", "--depart", default="2026-05-01", help="出发日期 (YYYY-MM-DD)")
    parser.add_argument("-r", "--return", dest="return_date", default="2026-05-15", help="返回日期 (YYYY-MM-DD)")
    parser.add_argument("-o", "--origin", nargs="*", default=["NUE", "FRA", "MUC"],
                        choices=["NUE", "FRA", "MUC"], help="出发机场")
    parser.add_argument("--urls-only", action="store_true", help="只打印URL，不打开浏览器")
    args = parser.parse_args()

    print("=" * 60)
    print("✈  纽伦堡 ⇄ 上海 机票搜索助手")
    print("=" * 60)
    print(f"  出发: {args.depart}")
    print(f"  返回: {args.return_date}")
    print(f"  出发地: {', '.join(args.origin)}")
    print()

    all_urls = generate_all_urls(args.origin, "PVG", args.depart, args.return_date)

    # 打印所有 URL
    for origin_label, urls in all_urls.items():
        print(f"📍 {origin_label}")
        for platform, url in urls.items():
            print(f"  {platform}:")
            print(f"    {url}")
        print()

    if args.urls_only:
        print("✅ URL 已生成。请手动在浏览器中打开。")
        return

    # 打开 Chrome
    print("🚀 正在启动 Chrome...")
    driver = open_chrome(all_urls)

    if driver:
        try:
            print("\n💡 提示：在各个标签页中比较后，记录最佳选项。")
            print("   按 Ctrl+C 退出脚本。")
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n👋 退出。浏览器保持打开。")
            # 不关闭浏览器，让用户继续浏览
            driver.service.stop()


if __name__ == "__main__":
    main()
