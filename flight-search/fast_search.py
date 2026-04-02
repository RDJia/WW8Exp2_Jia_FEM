"""
使用 fast-flights 库搜索 纽伦堡(NUE) ⇄ 上海(PVG) 的往返机票
自动搜索多个出发地（NUE/FRA/MUC）并比较价格

用法:
  python fast_search.py
  python fast_search.py --depart 2026-05-01 --return 2026-05-15
"""

import argparse
import json
import os
import time
from datetime import datetime

from fast_flights import FlightData, Passengers, get_flights


# 搜索配置
ORIGINS = [
    {"code": "NUE", "name": "纽伦堡", "extra_cost": 0, "note": "直接出发"},
    {"code": "FRA", "name": "法兰克福", "extra_cost": 40, "note": "火车~2h, +€40"},
    {"code": "MUC", "name": "慕尼黑", "extra_cost": 25, "note": "火车~1h, +€25"},
]
DESTINATION = "PVG"


def search_flights(from_airport, to_airport, depart_date, return_date):
    """搜索往返航班"""
    print(f"  🔍 搜索 {from_airport}⇄{to_airport} ({depart_date} ~ {return_date})...")

    try:
        result = get_flights(
            flight_data=[
                FlightData(
                    date=depart_date,
                    from_airport=from_airport,
                    to_airport=to_airport,
                    max_stops=1,
                ),
                FlightData(
                    date=return_date,
                    from_airport=to_airport,
                    to_airport=from_airport,
                    max_stops=1,
                ),
            ],
            trip="round-trip",
            seat="economy",
            passengers=Passengers(adults=1),
            max_stops=1,
        )
        return result
    except Exception as e:
        print(f"    ❌ 搜索失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def format_flight(flight, rank=0):
    """格式化航班信息"""
    lines = []
    prefix = f"  #{rank}" if rank else "  "

    # 尝试读取所有可能的属性
    attrs = {}
    for attr in dir(flight):
        if not attr.startswith('_'):
            try:
                val = getattr(flight, attr)
                if not callable(val):
                    attrs[attr] = val
            except Exception:
                pass

    price = attrs.get('price', 'N/A')
    duration = attrs.get('duration', attrs.get('total_duration', 'N/A'))
    stops = attrs.get('stops', attrs.get('num_stops', 'N/A'))
    name = attrs.get('name', attrs.get('airline', attrs.get('airlines', 'N/A')))
    departure = attrs.get('departure', attrs.get('dep_time', 'N/A'))
    arrival = attrs.get('arrival', attrs.get('arr_time', 'N/A'))

    lines.append(f"{prefix} 💰 {price} | ⏱ {duration} | 🔄 中转: {stops}")
    lines.append(f"       航司: {name}")
    lines.append(f"       时间: {departure} → {arrival}")

    # 航段信息
    legs = attrs.get('legs', attrs.get('segments', []))
    if legs:
        for i, leg in enumerate(legs):
            leg_attrs = {}
            for la in dir(leg):
                if not la.startswith('_'):
                    try:
                        lv = getattr(leg, la)
                        if not callable(lv):
                            leg_attrs[la] = lv
                    except Exception:
                        pass
            leg_from = leg_attrs.get('from_airport', leg_attrs.get('from', '?'))
            leg_to = leg_attrs.get('to_airport', leg_attrs.get('to', '?'))
            leg_airline = leg_attrs.get('airline', leg_attrs.get('name', '?'))
            leg_num = leg_attrs.get('flight_no', leg_attrs.get('flight_number', ''))
            leg_dur = leg_attrs.get('duration', '')
            lines.append(f"       段{i+1}: {leg_from}→{leg_to} ({leg_airline} {leg_num}) {leg_dur}")

    return "\n".join(lines)


def dump_flight_attrs(flight, label="Flight"):
    """调试用：打印航班的所有属性"""
    print(f"  [DEBUG] {label} attributes:")
    for attr in sorted(dir(flight)):
        if not attr.startswith('_'):
            try:
                val = getattr(flight, attr)
                if not callable(val):
                    print(f"    {attr} = {repr(val)[:100]}")
            except Exception as e:
                print(f"    {attr} = <error: {e}>")


def save_results(all_results, depart, return_date, output_dir):
    """保存结果到文件"""
    filepath = os.path.join(output_dir, "搜索结果.md")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# 机票搜索结果\n\n")
        f.write(f"- 搜索时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"- 出发: {depart}\n")
        f.write(f"- 返回: {return_date}\n\n")

        for label, result in all_results.items():
            f.write(f"## {label}\n\n")
            if result is None:
                f.write("搜索失败\n\n")
                continue

            flights = getattr(result, 'flights', [])
            if not flights:
                f.write("未找到航班\n\n")
                continue

            for i, fl in enumerate(flights[:10]):
                f.write(f"```\n{format_flight(fl, i+1)}\n```\n\n")

    print(f"\n✅ 结果已保存到: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="搜索纽伦堡⇄上海机票")
    parser.add_argument("--depart", default="2026-05-01", help="出发日期 YYYY-MM-DD")
    parser.add_argument("--return", dest="return_date", default="2026-05-15", help="返回日期 YYYY-MM-DD")
    parser.add_argument("--debug", action="store_true", help="显示调试信息")
    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("✈  纽伦堡 ⇄ 上海 机票搜索 (via Google Flights)")
    print("=" * 60)
    print(f"  出发: {args.depart}")
    print(f"  返回: {args.return_date}")
    print(f"  限制: 最多1次中转")
    print()

    all_results = {}

    for origin in ORIGINS:
        code = origin["code"]
        label = f"{origin['name']} ({code}) - {origin['note']}"
        extra = origin["extra_cost"]

        print(f"\n{'─' * 50}")
        print(f"📍 {label}")
        print(f"{'─' * 50}")

        result = search_flights(code, DESTINATION, args.depart, args.return_date)

        if result:
            flights = getattr(result, 'flights', [])
            print(f"  ✅ 找到 {len(flights)} 个航班选项")

            if args.debug and flights:
                dump_flight_attrs(flights[0], "第一个航班")
                legs = getattr(flights[0], 'legs', getattr(flights[0], 'segments', []))
                if legs:
                    dump_flight_attrs(legs[0], "第一个航段")

            for i, flight in enumerate(flights[:5]):
                print()
                print(format_flight(flight, i + 1))
                if extra > 0:
                    print(f"       📌 加上火车费: +€{extra}")
        else:
            print("  ❌ 未找到结果")

        all_results[label] = result
        time.sleep(3)

    save_results(all_results, args.depart, args.return_date, output_dir)

    print("\n" + "=" * 60)
    print("📋 完成！建议同时在 Trip.com / 携程 上查看中国航司价格")
    print("=" * 60)


if __name__ == "__main__":
    main()
