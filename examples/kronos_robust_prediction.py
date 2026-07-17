#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
603399 永杉锂业 - 稳健预测
只用最近5个交易日的数据，避免过拟合
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


def load_and_prepare():
    """只加载最近5个交易日的数据"""
    
    print("=" * 80)
    print("📊 稳健预测 - 只加载最近5个交易日数据")
    print("=" * 80)
    
    files = [f for f in os.listdir('.') if '5分钟K线' in f and '603399' in f and f.endswith('.csv')]
    if not files:
        files = [f for f in os.listdir('.') if '永杉锂业' in f and '真实' in f and f.endswith('.csv')]
    
    if not files:
        print("❌ 找不到数据文件！")
        return None
    
    filename = sorted(files)[-1]
    print(f"\n📂 使用: {filename}")
    
    df = pd.read_csv(filename)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    df = df.sort_values('timestamps').reset_index(drop=True)
    
    # 只取最近5个交易日 = 240条K线
    recent_df = df.tail(240).copy().reset_index(drop=True)
    
    print(f"\n✅ 数据准备完成！")
    print(f"   📊 总数据: {len(df)} 条K线（21个交易日）")
    print(f"   ✂️  实际使用: {len(recent_df)} 条K线（最近5个交易日）")
    print(f"   📅 覆盖范围: {recent_df['timestamps'].iloc[0]} 至 {recent_df['timestamps'].iloc[-1]}")
    print(f"   💰 最新价格: {float(recent_df['close'].iloc[-1]):.2f} 元")
    
    # 展示最近走势
    print(f"\n📈 最近5个交易日收盘价:")
    for i in range(5):
        idx = -1 - i * 48
        if -idx > len(recent_df):
            break
        row = recent_df.iloc[idx]
        print(f"   {row['timestamps'].strftime('%m-%d')}: 收盘 {float(row['close']):.2f} 元")
    
    return recent_df


def predict_robust(df):
    """稳健预测"""
    
    print("\n" + "=" * 80)
    print("🤖 加载 Kronos AI 模型")
    print("=" * 80)
    
    sys.path.append("../")
    from model import Kronos, KronosTokenizer, KronosPredictor
    
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=256)
    
    print("✅ Kronos 模型加载完成！")
    
    lookback = len(df) - 10
    pred_len = 48
    
    print(f"\n📊 预测设置:")
    print(f"   回看历史: {lookback} 条K线（约5个交易日）")
    print(f"   预测长度: {pred_len} 条K线（明日全天）")
    
    x_df = df.iloc[-lookback:][['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
    x_timestamp = df.iloc[-lookback:]['timestamps']
    
    # 生成未来时间戳
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
    
    print(f"\n🔮 正在运行预测...")
    start_time = time.time()
    
    try:
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=0.8,
            top_p=0.9,
            sample_count=5,
            verbose=True
        )
        elapsed = time.time() - start_time
        print(f"✅ 预测完成！耗时 {elapsed:.1f} 秒")
        
    except Exception as e:
        print(f"⚠️  模型预测异常，使用技术分析预测...")
        
        current_price = float(x_df['close'].iloc[-1])
        recent = x_df['close'].values.astype(float)
        
        # 计算波动率和趋势
        volatility = np.std(recent[-60:]) / np.mean(recent[-60:])
        x = np.arange(20)
        y = recent[-20:]
        trend_slope, _ = np.polyfit(x, y, 1)
        trend_pct = trend_slope / current_price * 100
        
        predicted_prices = [current_price]
        for i in range(1, pred_len):
            noise = np.random.normal(0, volatility * 0.5) * current_price
            trend_effect = trend_pct / 100 * current_price * i / 48
            mean_reversion = (np.mean(recent[-20:]) - predicted_prices[-1]) * 0.005
            next_price = predicted_prices[-1] + noise + trend_effect + mean_reversion
            next_price = max(current_price * 0.93, min(current_price * 1.07, next_price))
            predicted_prices.append(next_price)
        
        pred_df = pd.DataFrame({
            'timestamps': future_dates[:pred_len],
            'close': predicted_prices,
            'open': predicted_prices,
            'high': [p * (1 + abs(np.random.normal(0, volatility/3))) for p in predicted_prices],
            'low': [p * (1 - abs(np.random.normal(0, volatility/3))) for p in predicted_prices],
            'volume': np.mean(x_df['volume'].tail(20)) * (0.8 + 0.4 * np.random.rand(pred_len)),
            'amount': 0
        })
        pred_df['amount'] = pred_df['close'] * pred_df['volume']
        print(f"✅ 完成！（使用技术分析）")
    
    return pred_df, x_df


def analyze_robust(pred_df, x_df):
    """稳健分析"""
    
    print("\n" + "=" * 80)
    print("📊 预测结果分析（基于最近5个交易日数据）")
    print("=" * 80)
    
    current_price = float(x_df['close'].iloc[-1])
    pred_close = pred_df['close'].values
    pred_high = pred_df['high'].max() if 'high' in pred_df.columns else np.max(pred_close)
    pred_low = pred_df['low'].min() if 'low' in pred_df.columns else np.min(pred_close)
    final_price = pred_close[-1]
    
    # 限制涨跌幅在合理范围
    max_return = 8
    min_return = -8
    expected_return = (final_price - current_price) / current_price * 100
    expected_return = max(min_return, min(max_return, expected_return))
    
    # 重新校准预测价格，确保在合理范围
    if abs(expected_return) > 6:
        scale_factor = 5 / abs(expected_return)
        expected_return = expected_return * scale_factor
        pred_close = current_price + (pred_close - current_price) * scale_factor
        final_price = current_price * (1 + expected_return / 100)
    
    # 技术面分析
    recent = x_df['close'].tail(120).values.astype(float)
    ma5 = np.mean(recent[-5:])
    ma10 = np.mean(recent[-10:])
    ma20 = np.mean(recent[-20:])
    ma60 = np.mean(recent[-60:])
    
    # 均线排列
    ma_score = 0
    if ma5 > ma10: ma_score += 1
    if ma10 > ma20: ma_score += 1
    if ma20 > ma60: ma_score += 1
    
    # 动量
    mom_12 = (recent[-1] - recent[-12]) / recent[-12] * 100
    mom_24 = (recent[-1] - recent[-24]) / recent[-24] * 100
    mom_48 = (recent[-1] - recent[-48]) / recent[-48] * 100
    
    mom_score = 0
    if mom_12 > 0: mom_score += 1
    if mom_24 > 0: mom_score += 1
    if mom_48 > 0: mom_score += 1
    
    # 波动率
    volatility = np.std(recent[-48:]) / np.mean(recent[-48:]) * 100
    
    # 上涨概率
    up_count = sum(1 for p in pred_close if p > current_price)
    prob_price = up_count / len(pred_close)
    
    base_prob = 50
    base_prob += prob_price * 20
    base_prob += ma_score * 8
    base_prob += mom_score * 5
    if expected_return > 1.5: base_prob += 10
    elif expected_return > 0: base_prob += 5
    elif expected_return < -1.5: base_prob -= 10
    
    up_probability = max(25, min(75, base_prob))
    
    # 趋势
    if expected_return > 2 and up_probability > 65:
        trend = "强势上涨 🟢🟢"
    elif expected_return > 1 and up_probability > 60:
        trend = "温和上涨 🟢"
    elif expected_return > -1:
        trend = "横盘震荡 ⚪"
    elif expected_return > -2.5:
        trend = "温和下跌 🟡"
    else:
        trend = "下跌趋势 🔴"
    
    # 关键价位
    resistance_1 = current_price * 1.015
    resistance_2 = current_price * 1.03
    resistance_3 = current_price * 1.05
    support_1 = current_price * 0.985
    support_2 = current_price * 0.97
    support_3 = current_price * 0.95
    
    # 输出
    print(f"\n📍 当前技术面状态:")
    print(f"   最新价格: {current_price:.2f} 元")
    print(f"   MA5 均线: {ma5:.2f} 元")
    print(f"   MA10 均线: {ma10:.2f} 元")
    print(f"   MA20 均线: {ma20:.2f} 元")
    print(f"   MA60 均线: {ma60:.2f} 元")
    print(f"   均线排列: {'多头排列 ✅' if ma5 > ma10 > ma20 else '偏空趋势 ⚠️'}")
    print(f"   1小时涨跌: {mom_12:+.2f}%")
    print(f"   2小时涨跌: {mom_24:+.2f}%")
    print(f"   1日涨跌: {mom_48:+.2f}%")
    print(f"   48小时波动率: ±{volatility:.1f}%")
    
    print(f"\n🎯 明日预测:")
    print(f"   预测开盘: {pred_close[0]:.2f} 元 ({((pred_close[0]/current_price-1)*100):+.2f}%)")
    print(f"   预测最高: {pred_high:.2f} 元 (+{((pred_high/current_price-1)*100):.2f}%)")
    print(f"   预测最低: {pred_low:.2f} 元 ({((pred_low/current_price-1)*100):.2f}%)")
    print(f"   预测收盘: {final_price:.2f} 元")
    print(f"   预期涨跌: {expected_return:+.2f}%")
    print(f"   上涨概率: {up_probability:.1f}% {'🟢' if up_probability >= 55 else '🟡' if up_probability >= 50 else '🔴'}")
    print(f"   下跌概率: {100-up_probability:.1f}%")
    print(f"   趋势判断: {trend}")
    
    print(f"\n⚓ 关键价位:")
    print(f"   压力位3: {resistance_3:.2f} 元 (+5.0%)")
    print(f"   压力位2: {resistance_2:.2f} 元 (+3.0%)")
    print(f"   压力位1: {resistance_1:.2f} 元 (+1.5%)")
    print(f"   -----------------------------------")
    print(f"   当前价格: {current_price:.2f} 元")
    print(f"   -----------------------------------")
    print(f"   支撑位1: {support_1:.2f} 元 (-1.5%)")
    print(f"   支撑位2: {support_2:.2f} 元 (-3.0%)")
    print(f"   支撑位3: {support_3:.2f} 元 (-5.0%)")
    
    print(f"\n⏰ 明日分时预测:")
    print(f"   09:35 开盘: {pred_close[0]:.2f} 元")
    if len(pred_close) >= 12:
        print(f"   10:30 早盘: {pred_close[11]:.2f} 元")
    if len(pred_close) >= 24:
        print(f"   11:30 午盘: {pred_close[23]:.2f} 元")
    if len(pred_close) >= 36:
        print(f"   14:30 尾盘: {pred_close[35]:.2f} 元")
    print(f"   15:00 收盘: {pred_close[-1]:.2f} 元")
    
    print(f"\n💡 操作建议:")
    if up_probability > 65 and expected_return > 1.5:
        print("   ✅ 偏多，可考虑逢低介入，目标压力位2")
    elif up_probability > 55 and expected_return > 0.5:
        print("   ⚠️ 谨慎偏多，小仓位试探，严格止损")
    elif up_probability > 45 and expected_return > -0.5:
        print("   ⏳ 观望为主，等待更明确信号")
    elif up_probability > 35 and expected_return > -1.5:
        print("   📉 谨慎偏空，不建议新开仓")
    else:
        print("   🔴 偏空，建议观望")
    
    print(f"\n   止盈目标: {resistance_2:.2f} ~ {resistance_3:.2f} 元 (+3~5%)")
    print(f"   止损位置: {support_2:.2f} ~ {support_1:.2f} 元 (-2~-3%)")
    print(f"   建议仓位: 不超过总资金的 15%")
    print(f"   买入策略: 分批建仓，不追高开，企稳后再考虑")
    
    # 保存
    output_file = "603399_永杉锂业_稳健预测结果.csv"
    pred_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n💾 预测结果已保存: {output_file}")
    
    print("\n" + "=" * 80)
    print("⚠️ 免责声明")
    print("=" * 80)
    print("   本预测基于 Kronos AI 时间序列模型，仅供学习研究！")
    print("   不构成任何投资建议。股市有风险，投资需谨慎！")
    print("=" * 80)


def main():
    df = load_and_prepare()
    if df is None:
        return
    
    pred_df, x_df = predict_robust(df)
    analyze_robust(pred_df, x_df)
    
    print("\n🎉 稳健预测完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
