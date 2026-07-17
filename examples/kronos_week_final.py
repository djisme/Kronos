#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
603399 永杉锂业 - 下周行情稳健预测
单日涨跌幅限制在±8%以内，更符合实际
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import torch


def load_data():
    print("=" * 80)
    print("📊 603399 永杉锂业 - 下周行情稳健预测")
    print("=" * 80)
    
    files = [f for f in os.listdir('.') if '5分钟K线_20260519' in f and '603399' in f]
    if not files:
        files = [f for f in os.listdir('.') if '5分钟K线' in f and '603399' in f and '真实' in f]
    
    filename = sorted(files)[-1]
    print(f"\n📂 使用最新数据: {filename}")
    
    df = pd.read_csv(filename)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    df = df.sort_values('timestamps').reset_index(drop=True)
    
    print(f"\n✅ 数据: {len(df)} 条5分钟K线")
    print(f"   📅 覆盖: {df['timestamps'].min()} 至 {df['timestamps'].max()}")
    print(f"   💰 最新价格: {float(df['close'].iloc[-1]):.2f} 元")
    
    print(f"\n📈 最近7个交易日:")
    for i in range(7):
        idx = -1 - i * 48
        if -idx > len(df): break
        row = df.iloc[idx]
        prev = df.iloc[idx-48] if -idx-48 <= len(df) else row
        change = (float(row['close']) - float(prev['close'])) / float(prev['close']) * 100
        print(f"   {row['timestamps'].strftime('%m-%d')}: {float(row['close']):.2f} 元 ({change:+.2f}%)")
    
    return df


def predict_week_robust(df):
    """稳健预测 - 单日预测，逐日累加"""
    
    print("\n" + "=" * 80)
    print("🔮 运行逐日预测模型")
    print("=" * 80)
    
    sys.path.append("../")
    from model import Kronos, KronosTokenizer, KronosPredictor
    
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=256)
    
    current_price = float(df['close'].iloc[-1])
    all_predictions = []
    current_df = df.copy()
    
    # 逐日预测
    for day in range(5):
        print(f"\n📅 预测第 {day+1} 天...")
        
        lookback = min(240, len(current_df) - 10)
        
        x_df = current_df.iloc[-lookback:][['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
        x_timestamp = current_df.iloc[-lookback:]['timestamps']
        
        # 生成当日时间戳
        future_dates = []
        last_time = current_df['timestamps'].iloc[-1]
        for i in range(48):
            t = last_time + timedelta(minutes=5 * (i + 1))
            hour = t.hour
            minute = t.minute
            if (hour == 11 and minute > 30) or (hour == 12) or (hour == 13 and minute == 0):
                t = t.replace(hour=13, minute=5, second=0, microsecond=0)
            if hour >= 15:
                t = t + timedelta(days=1)
                while t.weekday() >= 5:
                    t = t + timedelta(days=1)
                t = t.replace(hour=9, minute=35, second=0, microsecond=0)
            future_dates.append(t)
        
        y_timestamp = pd.Series(future_dates)
        
        # 预测当日
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=48,
            T=0.8,
            top_p=0.9,
            sample_count=3,
            verbose=False
        )
        
        # 限制单日涨跌幅在±8%以内
        day_open = float(pred_df['close'].iloc[0])
        day_close = float(pred_df['close'].iloc[-1])
        day_return = (day_close - day_open) / day_open * 100
        
        if abs(day_return) > 8:
            scale = 8 / abs(day_return) * 0.8
            print(f"   ⚠️  原预测涨跌幅 {day_return:+.2f}%，限制到±8%")
            pred_df['close'] = day_open + (pred_df['close'] - day_open) * scale
            pred_df['high'] = pred_df['close'] * 1.01
            pred_df['low'] = pred_df['close'] * 0.99
        
        all_predictions.append(pred_df)
        
        # 把预测结果加入历史，作为下一天预测的基础
        pred_for_next = pred_df.copy()
        pred_for_next['timestamps'] = y_timestamp
        current_df = pd.concat([current_df, pred_for_next], ignore_index=True)
        
        actual_return = (float(pred_df['close'].iloc[-1]) - float(pred_df['close'].iloc[0])) / float(pred_df['close'].iloc[0]) * 100
        print(f"   ✅ 预测涨跌: {actual_return:+.2f}%")
    
    full_pred = pd.concat(all_predictions, ignore_index=True)
    return full_pred, df


def analyze_full(pred_df, df):
    """分析预测结果"""
    
    print("\n" + "=" * 80)
    print("📊 预测结果深度分析")
    print("=" * 80)
    
    current_price = float(df['close'].iloc[-1])
    pred_close = pred_df['close'].values.astype(float)
    
    # 按日统计
    days_pred = []
    for day in range(5):
        s = day * 48
        e = (day + 1) * 48
        day_data = pred_close[s:e]
        if len(day_data) > 0:
            days_pred.append({
                'open': day_data[0],
                'high': np.max(day_data),
                'low': np.min(day_data),
                'close': day_data[-1],
                'return': (day_data[-1] - day_data[0]) / day_data[0] * 100
            })
    
    week_high = np.max(pred_close)
    week_low = np.min(pred_close)
    week_close = pred_close[-1]
    week_return = (week_close - current_price) / current_price * 100
    
    # 计算周上涨概率
    up_days = sum(1 for d in days_pred if d['return'] > 0)
    up_in_pred = sum(1 for p in pred_close if p > current_price) / len(pred_close)
    
    # 技术面
    recent = df['close'].tail(192).values.astype(float)
    ma5 = np.mean(recent[-5:])
    ma10 = np.mean(recent[-10:])
    ma20 = np.mean(recent[-20:])
    ma60 = np.mean(recent[-60:])
    
    ma_score = sum([ma5>ma10, ma10>ma20, ma20>ma60])
    volatility = np.std(recent[-96:]) / np.mean(recent[-96:]) * 100
    
    up_probability = min(70, 40 + up_days * 5 + ma_score * 5 + up_in_pred * 10)
    
    # 周趋势
    if week_return > 4 and up_probability > 60:
        trend = "上涨趋势 🟢"
    elif week_return > 0:
        trend = "震荡偏强 🟡"
    elif week_return > -4:
        trend = "震荡偏弱 ⚪"
    else:
        trend = "下跌趋势 🔴"
    
    # 关键价位
    r1, r2, r3 = current_price * 1.03, current_price * 1.06, current_price * 1.10
    s1, s2, s3 = current_price * 0.97, current_price * 0.94, current_price * 0.90
    
    day_names = ['周一', '周二', '周三', '周四', '周五']
    
    print(f"\n📍 当前技术面:")
    print(f"   最新价格: {current_price:.2f} 元")
    print(f"   MA5: {ma5:.2f}, MA10: {ma10:.2f}, MA20: {ma20:.2f}, MA60: {ma60:.2f}")
    print(f"   均线排列: {'偏多 ✅' if ma5>ma10>ma20 else '偏弱 ⚠️'}")
    print(f"   周波动率: ±{volatility:.1f}%")
    
    print(f"\n🎯 下周整体预测:")
    print(f"   周一开盘: {days_pred[0]['open']:.2f} 元 ({((days_pred[0]['open']/current_price-1)*100):+.2f}%)")
    print(f"   周内最高: {week_high:.2f} 元 (+{((week_high/current_price-1)*100):.2f}%)")
    print(f"   周内最低: {week_low:.2f} 元 ({((week_low/current_price-1)*100):.2f}%)")
    print(f"   周五收盘: {week_close:.2f} 元")
    print(f"   全周涨跌: {week_return:+.2f}%")
    print(f"   周上涨概率: {up_probability:.1f}%")
    print(f"   周趋势判断: {trend}")
    
    print(f"\n📅 每日预测详情:")
    for i, d in enumerate(days_pred):
        print(f"   {day_names[i]}: {d['open']:.2f} → {d['close']:.2f} ({d['return']:+.2f}%)")
    
    print(f"\n⚓ 关键价位:")
    print(f"   强压力: {r3:.2f} 元 (+10%)")
    print(f"   压力2: {r2:.2f} 元 (+6%)")
    print(f"   压力1: {r1:.2f} 元 (+3%)")
    print(f"   -------------------")
    print(f"   当前价: {current_price:.2f} 元")
    print(f"   -------------------")
    print(f"   支撑1: {s1:.2f} 元 (-3%)")
    print(f"   支撑2: {s2:.2f} 元 (-6%)")
    print(f"   支撑3: {s3:.2f} 元 (-10%)")
    
    print(f"\n💡 周策略建议:")
    if up_probability > 60 and week_return > 2:
        print("   ✅ 偏乐观，可分批建仓，目标看压力位2")
    elif up_probability > 50 and week_return > 0:
        print("   ⚠️ 谨慎偏多，小仓位试探，严格止损")
    elif up_probability > 40 and week_return > -2:
        print("   ⏳ 震荡市，高抛低吸，快进快出")
    else:
        print("   🔴 偏空，观望为主，不急于抄底")
    
    print(f"\n   止盈目标: {r2:.2f} ~ {r3:.2f} 元 (+6~10%)")
    print(f"   止损位置: {s2:.2f} ~ {s1:.2f} 元 (-3~-6%)")
    print(f"   仓位控制: 总仓位不超过 30%")
    print(f"   操作风格: 短线为主，快进快出")
    
    # 保存
    pred_df.to_csv("603399_永杉锂业_下周稳健预测.csv", index=False, encoding='utf-8')
    print(f"\n💾 预测结果已保存")
    
    print("\n" + "=" * 80)
    print("⚠️ 免责声明: 仅供学习研究，不构成任何投资建议！")
    print("=" * 80)


def main():
    df = load_data()
    pred_df, df = predict_week_robust(df)
    analyze_full(pred_df, df)
    print("\n🎉 预测完成！")


if __name__ == "__main__":
    main()
