"""
纽伦堡(NUE) ⇄ 上海(PVG) 往返机票自动搜索脚本

使用 Selenium + Chrome 打开多个机票搜索平台，自动填入搜索条件。
由于各平台反爬严重，脚本采用"半自动"策略：
  - 自动打开正确的搜索URL（参数已预设好）
  - 用户在浏览器中查看结果并手动记录
  - 部分平台尝试自动提取价格信息

用法：
  pip install selenium webdriver-manager
  python search_flights.py

  或指定日期：
  python search_flights.py --depart 2026-05-01 --return 2026-05-15
"""

import argparse
import time
import json
import os
from datetime import datetime, timedelta

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False
    print("⚠ Selenium 未安装，将只生成搜索链接。")
    print("  安装方式: pip install selenium webdriver-manager")

# ============================================================
# 配置
# ============================================================

# 出发地/目的地
ORIGINS = [
    {"code": "NUE", "name": "纽伦堡", "note": "直接出发"},
    {"code": "FRA", "name": "法兰克福", "note": "火车约2h, ~30-50€"},
    {"code": "MUC", "name": "慕尼黑", "note": "火车约1h, ~25€"},
]
DESTINATION = "PVG"  # 上海浦东

# 搜索平台 URL 模板
SEARCH_URLS = {
    "Google Flights": (
        "https://www.google.com/travel/flights?q=Flights+from+{origin}+to+{dest}"
        "+on+{depart}+through+{return_date}&curr=EUR&tfs=CBwQAhooEgoyMDI2LTA1LTAxagwIAhIIL20vMDVicjlyDAoCEggvbS8wNmtmNhooEgoyMDI2LTA1LTE1agwIAhIIL20vMDZrZjZyDAoCEggvbS8wNWJyOUABSAFwAYIBCwj___________8BQAE"
    ),
    "Google Flights (简洁)": (
        "https://www.google.com/travel/flights/search"
        "?tfs=CBwQAhoeEgoyMDI2LTA1LTAxagcIARIDe29yaXJnB30IJBIDe2Rlc3R9Gh4SCjIwMjYtMDUtMTVqBwgBEgN7ZGVzdH1yBwgBEgN7b3JpfUABSAFwAYIBCwj___________8BQAE"
    ),
    "Skyscanner": (
        "https://www.skyscanner.net/transport/flights/{origin_lower}/{dest_lower}"
        "/{depart_sky}/{return_sky}/"
        "?adultsv2=1&cabinclass=economy&childrenv2=&ref=home&rtn=1"
        "&preferdirects=false&stops=!direct&outboundaltsenabled=true&inboundaltsenabled=true"
    ),
    "Trip.com (携程国际)": (
        "https://www.trip.com/flights/{origin}-to-shanghai/tickets-{origin_lower}-pvg"
        "?dcity={origin_lower}&acity=pvg&ddate={depart}&rdate={return_date}"
        "&flighttype=rt&class=y&lowpricesource=searchform&quantity=1"
    ),
    "Momondo": (
        "https://www.momondo.com/flight-search/{origin}-PVG/{depart}/{return_date}"
        "?sort=price_a&stops=1"
    ),
}


def format_date_skyscanner(date_str):
    """将 2026-05-01 转换为 260501 (Skyscanner 格式)"""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%y%m%d")


def generate_urls(origin, depart, return_date):
    """为指定出发地生成所有搜索平台的 URL"""
    urls = {}
    depart_sky = format_date_skyscanner(depart)
    return_sky = format_date_skyscanner(return_date)

    for name, template in SEARCH_URLS.items():
        try:
            url = template.format(
                origin=origin,
                origin_lower=origin.lower(),
                dest=DESTINATION,
                dest_lower=DESTINATION.lower(),
                depart=depart,
                return_date=return_date,
                depart_sky=depart_sky,
                return_sky=return_sky,
            )
            urls[name] = url
        except (KeyError, IndexError):
            pass
    return urls


def generate_all_urls(depart, return_date):
    """为所有出发地生成搜索链接"""
    all_urls = {}
    for origin_info in ORIGINS:
        origin = origin_info["code"]
        label = f"{origin_info['name']} ({origin}) - {origin_info['note']}"
        all_urls[label] = generate_urls(origin, depart, return_date)
    return all_urls


def save_urls_to_file(all_urls, depart, return_date, output_dir):
    """保存搜索链接到文件"""
    filepath = os.path.join(output_dir, "搜索链接.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# 机票搜索链接\n\n")
        f.write(f"- 出发日期: {depart}\n")
        f.write(f"- 返回日期: {return_date}\n")
        f.write(f"- 目的地: 上海 (PVG)\n")
        f.write(f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        for origin_label, urls in all_urls.items():
            f.write(f"## {origin_label}\n\n")
            for platform, url in urls.items():
                f.write(f"- **{platform}**: [点击搜索]({url})\n")
            f.write("\n")

        f.write("---\n\n")
        f.write("## 航司官网（手动搜索）\n\n")
        f.write("- **土耳其航空** (常有促销): https://www.turkishairlines.com/\n")
        f.write("- **芬兰航空**: https://www.finnair.com/\n")
        f.write("- **汉莎航空**: https://www.lufthansa.com/\n")
        f.write("- **中国国航**: https://www.airchina.com.cn/\n")
        f.write("- **中国东航**: https://www.ceair.com/\n")
        f.write("- **卡塔尔航空**: https://www.qatarairways.com/\n")

    print(f"✅ 搜索链接已保存到: {filepath}")
    return filepath


def open_in_chrome(all_urls):
    """使用 Selenium 在 Chrome 中打开所有搜索页面"""
    if not HAS_SELENIUM:
        print("❌ 需要安装 Selenium 才能自动打开浏览器")
        return

    print("🚀 启动 Chrome 浏览器...")

    options = Options()
    # 使用用户的 Chrome profile，保留登录状态和设置
    # options.add_argument("--user-data-dir=~/.config/google-chrome")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    try:
        from webdriver_manager.chrome import ChromeDriverManager
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
    except Exception:
        print("  尝试直接使用系统 chromedriver...")
        driver = webdriver.Chrome(options=options)

    # 优先打开 NUE 的搜索结果
    nue_label = [k for k in all_urls.keys() if "NUE" in k][0]
    nue_urls = all_urls[nue_label]

    # 先打开 Google Flights
    first_url = list(nue_urls.values())[0]
    driver.get(first_url)
    print(f"  ✅ 已打开: Google Flights (NUE)")
    time.sleep(2)

    # 其余的在新标签页打开
    for platform, url in list(nue_urls.items())[1:]:
        driver.execute_script(f"window.open('{url}', '_blank');")
        print(f"  ✅ 已打开: {platform} (NUE)")
        time.sleep(1)

    # 询问是否也打开 FRA/MUC 的搜索
    print("\n📋 NUE出发的搜索页面已全部打开。")
    print("   浏览器会保持打开状态，请在各标签页中查看结果。")

    # 也打开 FRA 和 MUC（以 Google Flights 为主）
    for label, urls in all_urls.items():
        if "NUE" in label:
            continue
        google_url = list(urls.values())[0]
        driver.execute_script(f"window.open('{google_url}', '_blank');")
        code = label.split("(")[1].split(")")[0]
        print(f"  ✅ 已打开: Google Flights ({code})")
        time.sleep(1)

    print("\n🎯 所有搜索页面已打开！请在浏览器中查看和比较价格。")
    print("   按 Enter 关闭浏览器，或手动关闭...")

    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass

    driver.quit()


def create_price_tracker(output_dir):
    """创建价格记录模板"""
    filepath = os.path.join(output_dir, "价格记录.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# 机票价格记录\n\n")
        f.write("在各平台搜索后，在下面记录找到的最佳价格：\n\n")
        f.write("| 日期 | 出发地 | 航司 | 路线 | 价格(EUR) | 中转时间 | 总时长 | 平台 | 备注 |\n")
        f.write("|------|--------|------|------|-----------|----------|--------|------|------|\n")
        f.write("| | NUE | TK | NUE→IST→PVG | | | | Google | |\n")
        f.write("| | NUE | AY | NUE→HEL→PVG | | | | Google | |\n")
        f.write("| | NUE | LH | NUE→FRA→PVG | | | | Google | |\n")
        f.write("| | FRA | CA | FRA→PVG(直飞) | | | | Trip.com | +火车€40 |\n")
        f.write("| | MUC | LH | MUC→PVG(直飞) | | | | Trip.com | +火车€25 |\n")
        f.write("\n")
    print(f"✅ 价格记录模板已保存到: {filepath}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="搜索 纽伦堡⇄上海 往返机票")
    parser.add_argument("--depart", default="2026-05-01", help="出发日期 (YYYY-MM-DD)")
    parser.add_argument("--return", dest="return_date", default="2026-05-15",
                        help="返回日期 (YYYY-MM-DD)")
    parser.add_argument("--no-browser", action="store_true",
                        help="不打开浏览器，只生成搜索链接")
    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("✈  纽伦堡 (NUE) ⇄ 上海 (PVG) 机票搜索")
    print("=" * 60)
    print(f"  出发: {args.depart}")
    print(f"  返回: {args.return_date}")
    print(f"  中转: 最多1次")
    print()

    # 生成所有搜索链接
    all_urls = generate_all_urls(args.depart, args.return_date)

    # 保存到文件
    save_urls_to_file(all_urls, args.depart, args.return_date, output_dir)
    create_price_tracker(output_dir)

    # 打印链接
    print("\n📋 搜索链接预览：")
    for origin_label, urls in all_urls.items():
        print(f"\n  [{origin_label}]")
        for platform, url in urls.items():
            print(f"    {platform}: {url[:80]}...")

    # 打开浏览器
    if not args.no_browser:
        print()
        open_in_chrome(all_urls)
    else:
        print("\n💡 使用 --no-browser 模式，请手动打开上述链接搜索。")


if __name__ == "__main__":
    main()
