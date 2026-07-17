#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试akshare能获取多长时间的5分钟K线数据
目标：获取3年历史数据
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import pandas as pd
from datetime import datetime, timedelta


def test_akshare_data_range(code):
    """测试akshare能获取的数据范围"""
    
    print("=" * 70)
    print(f"📊 测试 akshare 获取 {code} 历史5分钟K线数据范围")
    print("=" * 70)
    
    try:
        import akshare as ak
        
        print("\n[1] 开始获取全部可用数据...")
        print("   正在请求akshare...")
        
        if code.startswith('6') or code.startswith('5'):
            symbol = f'sh{code}'
        else:
            symbol = f'sz{code}'
        
        import time
        start_time = time.time()
        
        df = ak.stock_zh_a_minute(symbol=symbol, period="5", adjust="qfq")
        
        elapsed = time.time() - start_time
        
        if df is not None and len(df) > 0:
            # 重命名列
            df = df.rename(columns={
                'day': 'timestamps',
                '时间': 'timestamps'
            })
            
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            
            print(f"\n✅ 获取成功！")
            print(f"   ⏱️ 耗时: {elapsed:.1f}秒")
            print(f"   📊 总条数: {len(df)} 条5分钟K线")
            print(f"   📅 最早时间: {df['timestamps'].min()}")
            print(f"   📅 最晚时间: {df['timestamps'].max()}")
            
            # 计算时间跨度
            time_span = df['timestamps'].max() - df['timestamps'].min()
            
            print(f"   ⏳ 数据跨度: {time_span.days} 天 ({time_span.days/30:.1f} 个月)")
            
            # 每年交易日约250天，每天48根5分钟K线
            # 理论每年：250 * 48 = 12,000 根
            print(f"\n📈 数据统计:")
            print(f"   平均每天K线数: {len(df) / time_span.days:.1f} 根")
            print(f"   交易日数量: 约 {len(df) // 48} 天")
            
            # 保存
            filename = f"{code}_5分钟K线_{df['timestamps'].min().strftime('%Y%m%d')}_至_{df['timestamps'].max().strftime('%Y%m%d')}.csv"
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"\n💾 已保存到: {filename}")
            
            # 预览首尾
            print(f"\n📋 数据预览（前5条）:")
            print(df.head(5).to_string())
            
            print(f"\n📋 数据预览（后5条）:")
            print(df.tail(5).to_string())
            
            print("\n" + "=" * 70)
            
            if time_span.days < 365:
                print("⚠️  重要提示：")
                print(f"   akshare 免费接口只能获取约 {time_span.days} 天（{time_span.days/30:.1f}个月）的5分钟K线。")
                print("   要获取3年数据，需要：")
                print("   1. 使用付费数据接口（如聚宽、Tushare Pro）")
                print("   2. 或者每天增量抓取，积累3年")
                print("   3. 或者改用日线数据（可以拿到3年）")
            else:
                print(f"✅ 很棒！可以获取超过1年的数据！")
            
            return df, filename
            
        else:
            print("❌ 获取失败，返回为空")
            return None, None
            
    except Exception as e:
        print(f"❌ 出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def get_daily_data_3years(code):
    """获取3年日线数据（作为替代方案）"""
    
    print("\n" + "=" * 70)
    print("📊 备选方案：获取3年日线数据")
    print("=" * 70)
    
    try:
        import akshare as ak
        
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y%m%d')
        
        print(f"\n[1] 获取 {start_date} 至 {end_date} 的日线数据...")
        
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
        
        if df is not None and len(df) > 0:
            df = df.rename(columns={
                '日期': 'timestamps',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume',
                '成交额': 'amount'
            })
            
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            
            print(f"\n✅ 获取成功！共 {len(df)} 条日线")
            print(f"   📅 时间跨度: {df['timestamps'].min()} 至 {df['timestamps'].max()}")
            
            time_span = df['timestamps'].max() - df['timestamps'].min()
            print(f"   ⏳ 数据跨度: {time_span.days} 天 ({time_span.days/365:.1f} 年)")
            
            filename = f"{code}_3年日线数据.csv"
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"   💾 已保存到: {filename}")
            
            return df, filename
            
    except Exception as e:
        print(f"❌ 出错: {e}")
        return None, None


def main():
    code = sys.argv[1] if len(sys.argv) > 1 else "603399"
    
    # 1. 先测试5分钟K线能获取多少
    df_5min, file_5min = test_akshare_data_range(code)
    
    # 2. 提供3年日线备选
    print("\n" + "=" * 70)
    choice = input("是否需要获取3年日线数据作为备选？(y/n): ").strip().lower()
    
    if choice == 'y':
        df_daily, file_daily = get_daily_data_3years(code)
        
        print("\n" + "=" * 70)
        print("📊 总结")
        print("=" * 70)
        print(f"\n5分钟K线:")
        if df_5min is not None:
            time_span = df_5min['timestamps'].max() - df_5min['timestamps'].min()
            print(f"   ✅ {len(df_5min)} 条, 覆盖{time_span.days}天")
            print(f"   📁 {file_5min}")
        else:
            print(f"   ❌ 失败")
        
        print(f"\n日线数据:")
        if df_daily is not None:
            print(f"   ✅ {len(df_daily)} 条, 覆盖3年")
            print(f"   📁 {file_daily}")
        else:
            print(f"   ❌ 失败")
    
    print("\n" + "=" * 70)
    print("💡 提示：")
    print("   免费接口通常只能获取1-2个月的5分钟K线")
    print("   如果一定要3年5分钟数据，需要使用付费金融数据API")
    print("   如：Tushare Pro、聚宽JQData、米筐RQData等")
    print("=" * 70)


if __name__ == "__main__":
    main()
