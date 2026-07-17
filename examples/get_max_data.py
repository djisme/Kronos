#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一次性获取：
1. 最大范围的5分钟K线（约2个月）
2. 3年日线数据
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def get_max_5min_kline(code):
    """获取尽可能多的5分钟K线"""
    
    print("=" * 70)
    print(f"📊 正在获取 {code} 的最大范围5分钟K线数据")
    print("=" * 70)
    
    try:
        import akshare as ak
        
        print("\n[1/2] 请求akshare 5分钟K线数据...")
        
        if code.startswith('6') or code.startswith('5'):
            symbol = f'sh{code}'
        else:
            symbol = f'sz{code}'
        
        import time
        start_time = time.time()
        
        df = ak.stock_zh_a_minute(symbol=symbol, period="5", adjust="qfq")
        
        elapsed = time.time() - start_time
        
        if df is not None and len(df) > 0:
            df = df.rename(columns={
                'day': 'timestamps',
                '时间': 'timestamps'
            })
            
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            time_span = df['timestamps'].max() - df['timestamps'].min()
            
            print(f"\n✅ 获取成功！")
            print(f"   ⏱️ 耗时: {elapsed:.1f}秒")
            print(f"   📊 总条数: {len(df)} 条5分钟K线")
            print(f"   📅 起始时间: {df['timestamps'].min()}")
            print(f"   📅 结束时间: {df['timestamps'].max()}")
            print(f"   ⏳ 数据跨度: {time_span.days} 天 ({time_span.days/30:.1f} 个月)")
            print(f"   💰 最新价格: {float(df['close'].iloc[-1]):.2f} 元")
            
            filename = f"{code}_5分钟K线_{df['timestamps'].min().strftime('%Y%m%d')}_至_{df['timestamps'].max().strftime('%Y%m%d')}.csv"
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"\n💾 已保存到: {filename}")
            
            return df, filename
            
        else:
            print("❌ 获取失败")
            return None, None
            
    except Exception as e:
        print(f"❌ 出错: {e}")
        return None, None


def get_3years_daily(code):
    """获取3年日线数据"""
    
    print("\n" + "=" * 70)
    print(f"📊 正在获取 {code} 的3年日线数据")
    print("=" * 70)
    
    try:
        import akshare as ak
        
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y%m%d')
        
        print(f"\n[2/2] 从 {start_date} 到 {end_date} 的日线...")
        
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
            
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            time_span = df['timestamps'].max() - df['timestamps'].min()
            
            print(f"\n✅ 获取成功！")
            print(f"   📊 总条数: {len(df)} 条日线")
            print(f"   📅 起始时间: {df['timestamps'].min().date()}")
            print(f"   📅 结束时间: {df['timestamps'].max().date()}")
            print(f"   ⏳ 数据跨度: {time_span.days} 天 ({time_span.days/365:.1f} 年)")
            print(f"   💰 最新价格: {float(df['close'].iloc[-1]):.2f} 元")
            
            filename = f"{code}_3年日线数据_{df['timestamps'].min().strftime('%Y%m%d')}_至_{df['timestamps'].max().strftime('%Y%m%d')}.csv"
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"\n💾 已保存到: {filename}")
            
            return df, filename
            
    except Exception as e:
        print(f"❌ 出错: {e}")
        return None, None


def main():
    code = sys.argv[1] if len(sys.argv) > 1 else "603399"
    
    # 获取5分钟K线（最大范围）
    df_5min, file_5min = get_max_5min_kline(code)
    
    # 获取3年日线
    df_daily, file_daily = get_3years_daily(code)
    
    print("\n" + "=" * 70)
    print("📊 数据获取总结")
    print("=" * 70)
    
    print(f"\n📈 5分钟K线:")
    if df_5min is not None:
        time_span = df_5min['timestamps'].max() - df_5min['timestamps'].min()
        print(f"   ✅ 成功获取 {len(df_5min)} 条")
        print(f"   📁 覆盖 {time_span.days} 天 ({time_span.days/30:.1f} 个月)")
        print(f"   💾 文件: {file_5min}")
    else:
        print(f"   ❌ 失败")
    
    print(f"\n📈 日线数据:")
    if df_daily is not None:
        time_span = df_daily['timestamps'].max() - df_daily['timestamps'].min()
        print(f"   ✅ 成功获取 {len(df_daily)} 条")
        print(f"   📁 覆盖 {time_span.days} 天 ({time_span.days/365:.1f} 年)")
        print(f"   💾 文件: {file_daily}")
    else:
        print(f"   ❌ 失败")
    
    print("\n" + "=" * 70)
    print("💡 说明:")
    print("   1. 免费接口的5分钟K线只能拿到约2个月的数据")
    print("   2. 这是目前能拿到的最大数量的真实历史5分钟K线了")
    print("   3. 足够Kronos进行预测（只需约500条就够）")
    print("   4. 日线数据可用于更长周期的趋势分析")
    print("=" * 70)


if __name__ == "__main__":
    main()
