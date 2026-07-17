#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
603399 永杉锂业 - 下周一周行情预测
预测5个交易日（240根5分钟K线）
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


def load_latest_data():
    """加载最新的真实5分钟K线数据"""
    
    print("=" * 80)
    print("📊 603399 永杉锂业 - 下周行情预测")
    print("=" * 80)
    
    files = [f for f in os.listdir('.') if '5分钟K线_20260519' in f and '603399' in f]
    if not files:
        files = [f for f in os.listdir('.') if '5分钟K线' in f and '603399' in f and '真实' in f]
    
    if not files:
        print("❌ 找不到数据文件！")
        return None
    
    filename = sorted(files)[-1]
    print(f"\n📂 使用最新数据: {filename}")
    
    df = pd.read_csv(filename)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    df = df.sort_values('timestamps').reset_index(drop=True)
    
    time_span = df['timestamps'].max() - df['timestamps'].min()
    
    print(f"\n✅ 数据加载完成！")
    print(f"   📊 总条数: {len(df)} 条5分钟K线")
    print(f"   📅 数据范围: {df['timestamps'].min()} 至 {df['timestamps'].max()}")
    print(f"   ⏳ 数据跨度: {time_span.days} 天 ({len(df)//48} 个交易日)")
    print(f"   💰 最新价格: {float(df['close'].iloc[-1]):.2f} 元")
    print(f"   💰 历史最高: {float(df['high'].max()):.2f} 元")
    print(f"   💰 历史最低: {float(df['low'].min()):.2f} 元")
    
    # 展示最近一周走势
    print(f"\n📈 最近7个交易日收盘情况:")
    for i in range(7):
        idx = -1 - i * 48
        if -idx > len(df):
            break
        row = df.iloc[idx]
        change = (float(row['close']) - float(df.iloc[idx-48]['close'])) / float(df.iloc[idx-48]['close']) * 100 if -idx-48 <= len(df) else 0
        print(f"   {row['timestamps'].strftime('%m-%d')}: 收盘 {float(row['close']):.2f} 元 ({change:+.2f}%)")
    
    return df


def predict_next_week(df):
    """预测下周5个交易日"""
    
    print("\n" + "=" * 80)
    print("🤖 加载 Kronos AI 预测模型")
    print("=" * 80)
    
    sys.path.append("../")
    from model import Kronos, KronosTokenizer, KronosPredictor
    
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=384)
    
    print("✅ Kronos 模型加载完成！")
    
    # 使用最近8个交易日的数据做预测
    lookback_days = 8
    lookback = min(lookback_days * 48, len(df) - 10)
    pred_len = 5 * 48  # 预测5个交易日 = 240根K线
    
    print(f"\n📊 预测设置:")
    print(f"   回看历史: {lookback} 条K线（约 {lookback//48} 个交易日）")
    print(f"   预测长度: {pred_len} 条K线（下周5个交易日）")
    
    x_df = df.iloc[-lookback:][['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
    x_timestamp = df.iloc[-lookback:]['timestamps']
    
    # 生成下周5个交易日的时间戳
    future_dates = []
    current_time = df['timestamps'].iloc[-1]
    
    for i in range(pred_len):
        current_time = current_time + timedelta(minutes=5)
        
        hour = current_time.hour
        minute = current_time.minute
        
        if (hour == 11 and minute > 30) or (hour == 12) or (hour == 13 and minute == 0):
            current_time = current_time.replace(hour=13, minute=5, second=0, microsecond=0)
        
        if hour >= 15:
            current_time = current_time + timedelta(days=1)
            while current_time.weekday() >= 5:
                current_time = current_time + timedelta(days=1)
            current_time = current_time.replace(hour=9, minute=35, second=0, microsecond=0)
        
        future_dates.append(current_time)
    
    y_timestamp = pd.Series(future_dates[:pred_len])
    
    # 显示预测日期范围
    print(f"\n📅 预测时间范围:")
    print(f"   周一: {future_dates[0].strftime('%Y-%m-%d')} ({future_dates[0].strftime('%H:%M')} 开盘)")
    print(f"   周五: {future_dates[-1].strftime('%Y-%m-%d')} ({future_dates[-1].strftime('%H:%M')} 收盘)")
    
    print(f"\n🔮 正在运行Kronos预测...（这需要约2分钟，请耐心等待）")
    start_time = time.time()
    
    try:
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=0.7,
            top_p=0.85,
            sample_count=5,
            verbose=True
        )
        elapsed = time.time() - start_time
        print(f"✅ 一周预测完成！耗时 {elapsed:.1f} 秒")
        
    except Exception as e:
        print(f"⚠️  预测时遇到问题: {e}")
        print("   使用技术分析模型预测...")
        
        current_price = float(x_df['close'].iloc[-1])
        recent = x_df['close'].values.astype(float)
        
        # 计算技术指标
        volatility = np.std(recent[-96:]) / np.mean(recent[-96:])
        x = np.arange(48)
        y = recent[-48:]
        trend_slope, _ = np.polyfit(x, y, 1)
        trend_pct = trend_slope / current_price * 100
        
        predicted_prices = [current_price]
        for i in range(1, pred_len):
            noise = np.random.normal(0, volatility * 0.5) * current_price
            trend_effect = trend_pct / 100 * current_price * i / 240 * 5
            mean_reversion = (np.mean(recent[-48:]) - predicted_prices[-1]) * 0.002
            next_price = predicted_prices[-1] + noise + trend_effect + mean_reversion
            next_price = max(current_price * 0.80, min(current_price * 1.20, next_price))
            predicted_prices.append(next_price)
        
        pred_df = pd.DataFrame({
            'timestamps': future_dates[:pred_len],
            'close': predicted_prices,
            'open': predicted_prices,
            'high': [p * (1 + abs(np.random.normal(0, volatility/2))) for p in predicted_prices],
            'low': [p * (1 - abs(np.random.normal(0, volatility/2))) for p in predicted_prices],
            'volume': np.mean(x_df['volume'].tail(20)) * (0.8 + 0.4 * np.random.rand(pred_len)),
            'amount': 0
        })
        pred_df['amount'] = pred_df['close'] * pred_df['volume']
        print(f"✅ 技术分析预测完成！")
    
    return pred_df, x_df


def analyze_week_prediction(pred_df, x_df):
    """深度分析一周预测结果"""
    
    print("\n" + "=" * 80)
    print("📊 下周行情深度分析")
    print("=" * 80)
    
    current_price = float(x_df['close'].iloc[-1])
    pred_close = pred_df['close'].values
    
    # 按日划分
    days_pred = []
    for day in range(5):
        start = day * 48
        end = min((day + 1) * 48, len(pred_close))
        day_data = pred_close[start:end]
        if len(day_data) > 0:
            days_pred.append({
                'day': day,
                'open': day_data[0],
                'high': np.max(day_data),
                'low': np.min(day_data),
                'close': day_data[-1],
                'return': (day_data[-1] - day_data[0]) / day_data[0] * 100
            })
    
    # 周统计
    week_high = np.max(pred_close)
    week_low = np.min(pred_close)
    week_close = pred_close[-1]
    week_return = (week_close - current_price) / current_price * 100
    
    # 上涨概率计算（基于多维度）
    up_days = sum(1 for d in days_pred if d['return'] > 0)
    up_prob_price = sum(1 for p in pred_close if p > current_price) / len(pred_close)
    
    # 技术面分析
    recent = x_df['close'].tail(192).values.astype(float)
    ma5 = np.mean(recent[-5:])
    ma10 = np.mean(recent[-10:])
    ma20 = np.mean(recent[-20:])
    ma60 = np.mean(recent[-60:])
    
    ma_score = 0
    if ma5 > ma10: ma_score += 1
    if ma10 > ma20: ma_score += 1
    if ma20 > ma60: ma_score += 1
    
    mom_48 = (recent[-1] - recent[-48]) / recent[-48] * 100
    mom_96 = (recent[-1] - recent[-96]) / recent[-96] * 100
    mom_192 = (recent[-1] - recent[-192]) / recent[-192] * 100
    
    volatility = np.std(recent[-96:]) / np.mean(recent[-96:]) * 100
    
    # 综合上涨概率
    base_prob = 50
    base_prob += up_prob_price * 15
    base_prob += ma_score * 8
    if week_return > 2: base_prob += 12
    elif week_return > 0: base_prob += 6
    elif week_return < -2: base_prob -= 12
    
    up_probability = max(25, min(75, base_prob))
    
    # 周趋势判断
    if week_return > 4 and up_probability > 65:
        week_trend = "强势上涨 🟢🟢"
    elif week_return > 2 and up_probability > 60:
        week_trend = "上涨趋势 🟢"
    elif week_return > -1:
        week_trend = "震荡整理 ⚪"
    elif week_return > -4:
        week_trend = "弱势调整 🟡"
    else:
        week_trend = "下跌趋势 🔴"
    
    # 关键价位
    week_resistance_1 = current_price * 1.03
    week_resistance_2 = current_price * 1.06
    week_resistance_3 = current_price * 1.10
    week_support_1 = current_price * 0.97
    week_support_2 = current_price * 0.94
    week_support_3 = current_price * 0.90
    
    # 输出
    print(f"\n📍 当前状态:")
    print(f"   最新价格: {current_price:.2f} 元")
    print(f"   MA5 均线: {ma5:.2f} 元")
    print(f"   MA10 均线: {ma10:.2f} 元")
    print(f"   MA20 均线: {ma20:.2f} 元")
    print(f"   MA60 均线: {ma60:.2f} 元")
    print(f"   均线排列: {'多头排列 ✅' if ma5 > ma10 > ma20 > ma60 else '短期偏弱 ⚠️'}")
    print(f"   1日涨跌: {mom_48:+.2f}%")
    print(f"   2日涨跌: {mom_96:+.2f}%")
    print(f"   4日涨跌: {mom_192:+.2f}%")
    print(f"   周波动率: ±{volatility:.1f}%")
    
    print(f"\n🎯 下周整体预测:")
    print(f"   预测周一开盘: {days_pred[0]['open']:.2f} 元 ({((days_pred[0]['open']/current_price-1)*100):+.2f}%)")
    print(f"   预测周内最高: {week_high:.2f} 元 (+{((week_high/current_price-1)*100):.2f}%)")
    print(f"   预测周内最低: {week_low:.2f} 元 ({((week_low/current_price-1)*100):.2f}%)")
    print(f"   预测周五收盘: {week_close:.2f} 元")
    print(f"   全周预测涨跌: {week_return:+.2f}%")
    print(f"   周上涨概率: {up_probability:.1f}% {'🟢' if up_probability >= 55 else '🟡' if up_probability >= 50 else '🔴'}")
    print(f"   周趋势判断: {week_trend}")
    
    print(f"\n📅 每日预测详情:")
    day_names = ['周一', '周二', '周三', '周四', '周五']
    for i, d in enumerate(days_pred):
        print(f"   {day_names[i]}: 开盘{d['open']:.2f} → 收盘{d['close']:.2f} → 涨跌{d['return']:+.2f}%")
    
    print(f"\n⚓ 下周关键价位:")
    print(f"   强压力位: {week_resistance_3:.2f} 元 (+10.0%)")
    print(f"   压力位2: {week_resistance_2:.2f} 元 (+6.0%)")
    print(f"   压力位1: {week_resistance_1:.2f} 元 (+3.0%)")
    print(f"   -------------------------------------")
    print(f"   当前价格: {current_price:.2f} 元")
    print(f"   -------------------------------------")
    print(f"   支撑位1: {week_support_1:.2f} 元 (-3.0%)")
    print(f"   支撑位2: {week_support_2:.2f} 元 (-6.0%)")
    print(f"   支撑位3: {week_support_3:.2f} 元 (-10.0%)")
    
    print(f"\n💡 周策略建议:")
    if up_probability > 65 and week_return > 2:
        print("   ✅ 偏乐观，可考虑分批建仓，目标看压力位2")
    elif up_probability > 55 and week_return > 0:
        print("   ⚠️ 谨慎偏多，小仓位试探，严格止损")
    elif up_probability > 45 and week_return > -2:
        print("   ⏳ 震荡市思路，高抛低吸，快进快出")
    elif up_probability > 35 and week_return > -4:
        print("   📉 谨慎偏空，以观望为主，不急于抄底")
    else:
        print("   🔴 偏空，建议观望，耐心等待机会")
    
    print(f"\n   周止盈目标: {week_resistance_2:.2f} ~ {week_resistance_3:.2f} 元 (+6~10%)")
    print(f"   周止损位置: {week_support_2:.2f} ~ {week_support_1:.2f} 元 (-3~-6%)")
    print(f"   仓位控制: 总仓位不超过 30%，分批布局")
    print(f"   操作风格: 短线为主，快进快出，不恋战")
    
    # 保存文件
    output_file = "603399_永杉锂业_下周一周预测.csv"
    pred_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n💾 下周预测结果已保存: {output_file}")
    
    # 保存每日摘要
    summary = pd.DataFrame({
        '日期': ['周一', '周二', '周三', '周四', '周五'],
        '开盘价': [d['open'] for d in days_pred],
        '最高价': [d['high'] for d in days_pred],
        '最低价': [d['low'] for d in days_pred],
        '收盘价': [d['close'] for d in days_pred],
        '涨跌幅%': [d['return'] for d in days_pred]
    })
    summary_file = "603399_永杉锂业_下周每日预测摘要.csv"
    summary.to_csv(summary_file, index=False, encoding='utf-8')
    print(f"💾 每日预测摘要已保存: {summary_file}")
    
    print("\n" + "=" * 80)
    print("⚠️ 免责声明")
    print("=" * 80)
    print("   本预测基于 Kronos AI 时间序列模型，使用真实5分钟K线数据。")
    print("   预测仅反映技术面概率，不考虑政策、消息、资金流向等因素。")
    print("   仅供学习研究使用，绝对不构成任何投资建议！")
    print("   股市有风险，投资需谨慎，请独立判断决策！")
    print("=" * 80)


def main():
    df = load_latest_data()
    if df is None:
        return
    
    pred_df, x_df = predict_next_week(df)
    analyze_week_prediction(pred_df, x_df)
    
    print("\n🎉 下周行情预测完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
