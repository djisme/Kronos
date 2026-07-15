#!/usr/bin/env python3
"""尝试各种接口获取 603399 数据"""
import akshare as ak
import pandas as pd

print("=" * 60)
print("获取 603399（永杉锂业）数据 - 多接口尝试")
print("=" * 60)

# 接口1: 东方财富分钟线
print("\n1️⃣  尝试东方财富分钟线接口...")
try:
    df = ak.stock_zh_a_hist_min_em(symbol="603399", period="5")
    print(f"✅ 成功! 获取到 {len(df)} 条数据")
    print(f"   列名: {list(df.columns)}")
    df.to_csv("603399_test1.csv", index=False)
    print("   已保存到: 603399_test1.csv")
except Exception as e:
    print(f"❌ 失败: {e}")

# 接口2: 新浪分钟线
print("\n2️⃣  尝试新浪分钟线接口...")
try:
    df = ak.stock_zh_a_minute(symbol="sh603399", period="5")
    print(f"✅ 成功! 获取到 {len(df)} 条数据")
    print(f"   列名: {list(df.columns)}")
    df.to_csv("603399_test2.csv", index=False)
    print("   已保存到: 603399_test2.csv")
except Exception as e:
    print(f"❌ 失败: {e}")

# 接口3: 新浪另一种格式
print("\n3️⃣  尝试新浪另一种格式...")
try:
    df = ak.stock_zh_a_minute(symbol="603399", period="5")
    print(f"✅ 成功! 获取到 {len(df)} 条数据")
    print(f"   列名: {list(df.columns)}")
except Exception as e:
    print(f"❌ 失败: {e}")

# 接口4: 日线
print("\n4️⃣  尝试日线接口...")
try:
    df = ak.stock_zh_a_hist(symbol="603399", period="daily", start_date="20250101")
    print(f"✅ 成功! 获取到 {len(df)} 条日线数据")
    print(f"   列名: {list(df.columns)}")
    df.to_csv("603399_daily.csv", index=False)
    print("   已保存到: 603399_daily.csv")
except Exception as e:
    print(f"❌ 失败: {e}")

# 接口5: 东财另一种方式
print("\n5️⃣  尝试东方财富另一种方式...")
try:
    df = ak.stock_zh_a_spot_em()
    stock = df[df['代码'] == '603399']
    if len(stock) > 0:
        print("✅ 股票存在!")
        print(stock.iloc[0])
    else:
        print("❌ 未找到股票")
except Exception as e:
    print(f"❌ 失败: {e}")

print("\n" + "=" * 60)
print("尝试完成! 请查看上面的输出结果")
print("=" * 60)
