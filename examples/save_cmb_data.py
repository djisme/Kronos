#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取并保存招商银行的真实5分钟K线数据
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import pandas as pd
import numpy as np
from datetime import datetime


def main():
    print("=" * 70)
    print("📊 招商银行 (600036) 真实5分钟K线数据获取")
    print("=" * 70)
    
    try:
        import akshare as ak
        
        print(f"\n🔍 正在从东方财富获取数据...")
        df = ak.stock_zh_a_minute(symbol="sh600036", period="5", adjust="qfq")
        
        if df is not None and len(df) > 0:
            # 数据处理
            df = df.rename(columns={'day': 'timestamps'})
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            df = df.sort_values('timestamps').reset_index(drop=True)
            
            # 保存文件
            filename = "600036_招商银行_真实5分钟K线.csv"
            df.to_csv(filename, index=False, encoding='utf-8')
            
            print(f"\n✅ 数据获取成功！")
            print(f"   📊 K线条数: {len(df)} 条")
            print(f"   📅 起始时间: {df['timestamps'].min()}")
            print(f"   📅 结束时间: {df['timestamps'].max()}")
            print(f"   ⏳ 覆盖天数: {(df['timestamps'].max() - df['timestamps'].min()).days} 天")
            print(f"   💰 价格范围: {float(df['low'].min()):.2f} ~ {float(df['high'].max()):.2f} 元")
            print(f"   💹 最新收盘: {float(df['close'].iloc[-1]):.2f} 元")
            
            print(f"\n📁 文件已保存到:")
            print(f"   /Users/a1/Documents/code/Kronos/Kronos/examples/{filename}")
            
            # 显示最近20条数据
            print(f"\n📋 最近20条K线预览:")
            print(f"{'时间':<20}{'开盘':>8}{'最高':>8}{'最低':>8}{'收盘':>8}{'成交量':>12}")
            print("-" * 70)
            recent = df.tail(20)
            for _, row in recent.iterrows():
                print(f"{str(row['timestamps']):<20}{row['open']:>8.2f}{row['high']:>8.2f}{row['low']:>8.2f}{row['close']:>8.2f}{int(row['volume']):>12,}")
            
            # 计算技术指标
            recent_prices = df['close'].values.astype(float)
            ma5 = np.mean(recent_prices[-5:])
            ma10 = np.mean(recent_prices[-10:])
            ma20 = np.mean(recent_prices[-20:])
            ma60 = np.mean(recent_prices[-60:])
            
            # RSI
            deltas = np.diff(recent_prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
            rsi = 100 - (100 / (1 + rs))
            
            ret_5d = (recent_prices[-1] - recent_prices[-241]) / recent_prices[-241] * 100
            
            print(f"\n📈 技术面指标:")
            print(f"   MA5: {ma5:.2f}")
            print(f"   MA10: {ma10:.2f}")
            print(f"   MA20: {ma20:.2f}")
            print(f"   MA60: {ma60:.2f}")
            print(f"   RSI(14): {rsi:.1f}")
            print(f"   5日涨跌幅: {ret_5d:+.2f}%")
            print(f"   均线排列: {'多头 ✅' if ma5 > ma10 > ma20 > ma60 else '需要观察'}")
            
            print(f"\n" + "=" * 70)
            print("🎉 完成！你现在可以用Excel打开这个CSV文件查看完整K线数据")
            print("=" * 70)
            
        else:
            print("❌ 获取数据失败")
            
    except Exception as e:
        print(f"❌ 出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
