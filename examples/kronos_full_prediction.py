#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
603399 永杉锂业 - 全量历史数据预测
使用所有 1008 条真实5分钟K线（21个工作日）
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


def load_data(filename):
    """加载真实5分钟K线数据"""
    
    print("=" * 80)
    print("📊 加载全量真实5分钟K线数据")
    print("=" * 80)
    
    df = pd.read_csv(filename)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    df = df.sort_values('timestamps').reset_index(drop=True)
    
    time_span = df['timestamps'].max() - df['timestamps'].min()
    
    print(f"\n✅ 数据加载完成！")
    print(f"   📊 总条数: {len(df)} 条5分钟K线")
    print(f"   📅 起始时间: {df['timestamps'].min()}")
    print(f"   📅 结束时间: {df['timestamps'].max()}")
    print(f"   ⏳ 数据跨度: {time_span.days} 天 ({len(df)//48} 个交易日)")
    print(f"   💰 最新价格: {float(df['close'].iloc[-1]):.2f} 元")
    print(f"   💰 价格范围: {float(df['low'].min()):.2f} ~ {float(df['high'].max()):.2f} 元")
    
    # 分段展示
    print(f"\n📋 数据全貌（每3天展示最新一条）:")
    step = len(df) // 7
    for i in range(0, len(df), step):
        row = df.iloc[i]
        print(f"   {row['timestamps']}: 收盘 {row['close']:.2f} 元")
    print(f"   {df['timestamps'].iloc[-1]}: 收盘 {float(df['close'].iloc[-1]):.2f} 元 (最新)")
    
    return df


def run_full_prediction(df):
    """使用全量历史数据进行预测"""
    
    print("\n" + "=" * 80)
    print("🤖 加载 Kronos AI 预测模型")
    print("=" * 80)
    
    sys.path.append("../")
    from model import Kronos, KronosTokenizer, KronosPredictor
    
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
    
    print("✅ Kronos 模型加载完成！")
    
    # 使用全量数据！
    lookback = len(df) - 10
    pred_len = 48  # 预测明日全天
    
    print(f"\n📊 预测设置:")
    print(f"   回看历史: {lookback} 条K线 ({lookback//48} 个完整交易日)")
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
    
    print(f"\n🔮 正在运行Kronos全量数据预测...")
    print(f"   这会需要更长时间，请稍候...")
    start_time = time.time()
    
    try:
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=0.6,
            top_p=0.8,
            sample_count=8,
            verbose=True
        )
        elapsed = time.time() - start_time
        print(f"✅ 全量预测完成！耗时 {elapsed:.1f} 秒")
        
        return pred_df, x_df
        
    except Exception as e:
        print(f"⚠️  预测出错: {e}")
        import traceback
        traceback.print_exc()
        
        # 降级方案
        print(f"\n使用降级方案预测...")
        last_price = float(x_df['close'].iloc[-1])
        recent = x_df['close'].values.astype(float)
        
        volatility = np.std(recent[-60:]) / np.mean(recent[-60:])
        trend_slope = np.polyfit(range(20), recent[-20:], 1)[0] / last_price
        
        predicted_prices = [last_price]
        for i in range(1, pred_len):
            noise = np.random.normal(0, volatility) * last_price
            trend_effect = trend_slope * i * last_price
            mean_reversion = (np.mean(recent[-20:]) - predicted_prices[-1]) * 0.01
            next_price = predicted_prices[-1] + noise + trend_effect + mean_reversion
            next_price = max(last_price * 0.9, min(last_price * 1.1, next_price))
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
        
        print("✅ 降级预测完成！")
        return pred_df, x_df


def detailed_analysis(pred_df, x_df):
    """详细分析预测结果"""
    
    print("\n" + "=" * 80)
    print("📊 全量数据预测结果深度分析")
    print("=" * 80)
    
    current_price = float(x_df['close'].iloc[-1])
    pred_close = pred_df['close'].values
    pred_high = pred_df['high'].max() if 'high' in pred_df.columns else pred_close.max()
    pred_low = pred_df['low'].min() if 'low' in pred_df.columns else pred_close.min()
    final_price = pred_close[-1]
    
    expected_return = (final_price - current_price) / current_price * 100
    
    # 多维度计算上涨概率
    # 1. 基于预测价格分布
    up_count = sum(1 for p in pred_close if p > current_price)
    prob_price = up_count / len(pred_close)
    
    # 2. 基于技术指标
    recent = x_df['close'].tail(120).values.astype(float)
    ma5 = np.mean(recent[-5:])
    ma10 = np.mean(recent[-10:])
    ma20 = np.mean(recent[-20:])
    ma60 = np.mean(recent[-60:])
    
    # 均线排列分数
    ma_score = 0
    if ma5 > ma10: ma_score += 1
    if ma10 > ma20: ma_score += 1
    if ma20 > ma60: ma_score += 1
    
    # 3. 基于动量
    mom_12 = (recent[-1] - recent[-12]) / recent[-12] * 100  # 1小时涨跌幅
    mom_24 = (recent[-1] - recent[-24]) / recent[-24] * 100   # 2小时涨跌幅
    mom_48 = (recent[-1] - recent[-48]) / recent[-48] * 100   # 1个交易日涨跌幅
    
    # 动量分数
    mom_score = 0
    if mom_12 > 0: mom_score += 1
    if mom_24 > 0: mom_score += 1
    if mom_48 > 0: mom_score += 1
    
    # 4. 波动率
    volatility = np.std(recent[-48:]) / np.mean(recent[-48:]) * 100
    
    # 综合上涨概率
    base_prob = 50  # 基准概率
    base_prob += prob_price * 20  # 价格分布贡献
    base_prob += ma_score * 8     # 均线排列贡献
    base_prob += mom_score * 5    # 动量贡献
    if expected_return > 1: base_prob += 10
    elif expected_return > 0: base_prob += 5
    elif expected_return < -1: base_prob -= 10
    
    up_probability = max(20, min(80, base_prob))
    
    # 趋势强度判断
    if expected_return > 2.5 and up_probability > 65:
        trend = "强势上涨 🟢🟢"
    elif expected_return > 1.2 and up_probability > 60:
        trend = "温和上涨 🟢"
    elif expected_return > -0.8:
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
    
    # 输出结果
    print(f"\n📍 当前技术面状态:")
    print(f"   最新价格: {current_price:.2f} 元")
    print(f"   MA5 均线: {ma5:.2f} 元")
    print(f"   MA10 均线: {ma10:.2f} 元")
    print(f"   MA20 均线: {ma20:.2f} 元")
    print(f"   MA60 均线: {ma60:.2f} 元")
    print(f"   均线排列: {'多头排列 ✅' if ma5 > ma10 > ma20 > ma60 else '趋势偏空 ⚠️'}")
    print(f"   1小时涨跌: {mom_12:+.2f}%")
    print(f"   2小时涨跌: {mom_24:+.2f}%")
    print(f"   1日涨跌: {mom_48:+.2f}%")
    print(f"   48小时波动率: ±{volatility:.1f}%")
    
    print(f"\n🎯 明日深度预测:")
    print(f"   预测开盘: {pred_close[0]:.2f} 元 ({((pred_close[0]/current_price-1)*100):+.2f}%)")
    print(f"   预测最高: {pred_high:.2f} 元 (+{((pred_high/current_price-1)*100):.2f}%)")
    print(f"   预测最低: {pred_low:.2f} 元 ({((pred_low/current_price-1)*100):.2f}%)")
    print(f"   预测收盘: {final_price:.2f} 元")
    print(f"   预期涨跌: {expected_return:+.2f}%")
    print(f"   上涨概率: {up_probability:.1f}% {'🟢' if up_probability >= 55 else '🟡' if up_probability >= 50 else '🔴'}")
    print(f"   下跌概率: {100-up_probability:.1f}%")
    print(f"   趋势判断: {trend}")
    
    print(f"\n⚓ 关键支撑位 & 压力位:")
    print(f"   压力位3: {resistance_3:.2f} 元 (+5.0%)")
    print(f"   压力位2: {resistance_2:.2f} 元 (+3.0%)")
    print(f"   压力位1: {resistance_1:.2f} 元 (+1.5%)")
    print(f"   -----------------------------------")
    print(f"   当前价格: {current_price:.2f} 元")
    print(f"   -----------------------------------")
    print(f"   支撑位1: {support_1:.2f} 元 (-1.5%)")
    print(f"   支撑位2: {support_2:.2f} 元 (-3.0%)")
    print(f"   支撑位3: {support_3:.2f} 元 (-5.0%)")
    
    print(f"\n⏰ 明日分时关键点位预测:")
    print(f"   09:35 开盘: {pred_close[0]:.2f} 元")
    if len(pred_close) >= 12:
        print(f"   10:30 早盘: {pred_close[11]:.2f} 元")
    if len(pred_close) >= 24:
        print(f"   11:30 午盘: {pred_close[23]:.2f} 元")
    if len(pred_close) >= 36:
        print(f"   14:30 尾盘: {pred_close[35]:.2f} 元")
    print(f"   15:00 收盘: {pred_close[-1]:.2f} 元")
    
    print(f"\n💡 综合操作建议:")
    if up_probability > 65 and expected_return > 1.5:
        print("   ✅ 乐观，可考虑逢低介入，目标看压力位2")
    elif up_probability > 55 and expected_return > 0.5:
        print("   ⚠️ 谨慎偏多，小仓位试探，严格止损在支撑位2")
    elif up_probability > 45 and expected_return > -0.5:
        print("   ⏳ 观望为主，等待更明确的信号")
    elif up_probability > 35 and expected_return > -1.5:
        print("   📉 谨慎偏空，不建议新开仓，持仓考虑减仓")
    else:
        print("   🔴 偏空，建议观望，不要急于抄底")
    
    print(f"\n   止盈目标: {resistance_2:.2f} ~ {resistance_3:.2f} 元 (+3~5%)")
    print(f"   止损位置: {support_2:.2f} ~ {support_1:.2f} 元 (-2~-3%)")
    print(f"   建议仓位: 不超过总资金的 15%")
    print(f"   操作方式: 分批建仓，不追高开，企稳后再考虑")
    
    # 保存文件
    output_file = "603399_永杉锂业_全量数据深度预测.csv"
    pred_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n💾 深度预测结果已保存: {output_file}")
    
    x_df_with_type = x_df.copy()
    x_df_with_type['type'] = '历史'
    pred_df['type'] = '预测'
    combined = pd.concat([x_df_with_type, pred_df], ignore_index=True)
    combined_file = "603399_永杉锂业_全量历史+预测合并.csv"
    combined.to_csv(combined_file, index=False, encoding='utf-8')
    print(f"💾 全量历史+预测合并数据已保存: {combined_file}")
    
    print("\n" + "=" * 80)
    print("⚠️ 免责声明")
    print("=" * 80)
    print("   本预测基于 Kronos AI 时间序列模型，使用全量历史数据计算。")
    print("   仅用于学习研究，不构成任何投资建议。")
    print("   股市有风险，投资需谨慎，请独立判断决策！")
    print("=" * 80)
    
    return pred_df


def main():
    # 查找数据文件
    files = [f for f in os.listdir('.') if '5分钟K线' in f and '603399' in f and f.endswith('.csv')]
    
    if not files:
        files = [f for f in os.listdir('.') if '永杉锂业' in f and '真实' in f and f.endswith('.csv')]
    
    if not files:
        print("❌ 找不到数据文件！")
        return
    
    filename = sorted(files)[-1]
    print(f"📂 使用数据文件: {filename}")
    
    # 1. 加载全量数据
    df = load_data(filename)
    
    # 2. 运行全量预测
    pred_df, x_df = run_full_prediction(df)
    
    # 3. 深度分析
    pred_df = detailed_analysis(pred_df, x_df)
    
    print("\n🎉 全量数据深度预测完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
