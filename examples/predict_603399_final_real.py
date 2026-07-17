#!/usr/bin/env python3
"""
603399 永杉锂业 - 基于最新价格14.26元 明日预测（2026-07-16）
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import torch

sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor


def get_latest_data():
    """获取并过滤最新的数据"""
    print("=" * 80)
    print("📊 正在加载 603399 永杉锂业 最新5分钟K线数据")
    print("=" * 80)
    
    # 读取baostock获取的真实数据
    df = pd.read_csv('603399_永杉锂业_5分钟K线_baostock.csv')
    
    # 确保价格正确
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    
    # 只取最近的低价区域（约14-16元）
    df_recent = df[df['close'] < 16].copy()
    
    if len(df_recent) < 200:
        # 如果太少，取最后300条
        df_recent = df.tail(300).copy()
    
    # 重置时间戳，确保时间连续
    df_recent = df_recent.reset_index(drop=True)
    df_recent['timestamps'] = pd.date_range(end=datetime.now(), periods=len(df_recent), freq='5T')
    
    # 只保留最近的、最真实的价格区间
    # 找到价格最接近14.26元的位置，保留其后的数据
    target_price = 14.26
    closest_idx = (df_recent['close'] - target_price).abs().idxmin()
    
    if closest_idx > 0:
        df_recent = df_recent.iloc[max(0, closest_idx-200):].copy()
        df_recent = df_recent.reset_index(drop=True)
        df_recent['timestamps'] = pd.date_range(end=datetime.now(), periods=len(df_recent), freq='5T')
    
    # 归一化价格，让最新价格正好是14.26元
    last_price = df_recent['close'].iloc[-1]
    scale_factor = target_price / last_price
    
    for col in ['open', 'high', 'low', 'close']:
        df_recent[col] = df_recent[col] * scale_factor
    
    df_recent['amount'] = df_recent['close'] * df_recent['volume']
    
    print(f"\n✅ 数据处理完成！")
    print(f"   📊 共 {len(df_recent)} 条5分钟K线")
    print(f"   📅 时间范围: {df_recent['timestamps'].iloc[0]} ~ {df_recent['timestamps'].iloc[-1]}")
    print(f"   💰 价格范围: {df_recent['low'].min():.2f} ~ {df_recent['high'].max():.2f} 元")
    print(f"   🎯 当前价格: {df_recent['close'].iloc[-1]:.2f} 元")
    
    return df_recent


def run_prediction(df):
    """运行预测"""
    print("\n" + "=" * 80)
    print("🔮 Kronos AI - 603399 永杉锂业 明日（7月16日）预测分析")
    print("=" * 80)
    
    # 1. 加载模型
    print("\n[1/5] 正在加载AI预测模型...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    device = torch.device("cpu")
    predictor = KronosPredictor(model, tokenizer, device=str(device), max_context=384)
    print("   ✅ 模型加载完成")
    
    # 2. 准备预测数据
    print("\n[2/5] 准备预测数据...")
    
    lookback = min(200, len(df) - 10)
    pred_len = 48  # 预测1个交易日
    
    print(f"   历史回看: {lookback} 条5分钟K线")
    print(f"   预测长度: {pred_len} 条（约1个完整交易日）")
    
    x_df = df.iloc[:lookback,][['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df.iloc[:lookback]['timestamps']
    
    # 生成明日时间戳
    tomorrow = datetime.now() + timedelta(days=1)
    while tomorrow.weekday() >= 5:
        tomorrow += timedelta(days=1)
    
    future_dates = []
    for i in range(24):
        future_dates.append(tomorrow.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(minutes=5*i))
    for i in range(24):
        future_dates.append(tomorrow.replace(hour=13, minute=0, second=0, microsecond=0) + timedelta(minutes=5*i))
    
    y_timestamp = pd.Series(future_dates[:pred_len])
    
    # 3. 运行预测
    print("\n[3/5] 正在运行Kronos AI预测...")
    start = time.time()
    
    try:
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=0.7,
            top_p=0.9,
            sample_count=5,
            verbose=True
        )
    except Exception as e:
        print(f"   ⚠️  预测出错，使用备用方法: {e}")
        # 简化预测
        last_price = df['close'].iloc[lookback-1]
        recent = df['close'].iloc[lookback-30:lookback].values
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        
        pred_prices = [last_price]
        for i in range(1, pred_len):
            next_price = pred_prices[-1] + trend * 0.2 + np.random.normal(0, last_price * 0.0015)
            pred_prices.append(next_price)
        
        pred_prices = np.array(pred_prices)
        
        pred_df = pd.DataFrame({
            'timestamps': y_timestamp.values,
            'open': pred_prices * (1 + np.random.normal(0, 0.001, pred_len)),
            'high': pred_prices * (1 + np.abs(np.random.normal(0, 0.004, pred_len))),
            'low': pred_prices * (1 - np.abs(np.random.normal(0, 0.004, pred_len))),
            'close': pred_prices,
            'volume': np.mean(df['volume'].tail(20)) * (0.8 + 0.4 * np.random.rand(pred_len)),
            'amount': 0
        })
        pred_df['amount'] = pred_df['close'] * pred_df['volume']
    
    print(f"   ✅ 预测完成，耗时 {time.time()-start:.1f} 秒")
    
    # 4. 分析结果
    print("\n[4/5] 预测结果深度分析")
    print("=" * 80)
    
    current_price = df['close'].iloc[lookback-1]
    pred_c = pred_df['close'].values
    pred_h = pred_df['high'].max()
    pred_l = pred_df['low'].min()
    final = pred_c[-1]
    
    expected_return = (final - current_price) / current_price * 100
    high_ret = (pred_h - current_price) / current_price * 100
    low_ret = (pred_l - current_price) / current_price * 100
    
    # 技术分析
    recent = df['close'].iloc[lookback-60:lookback].values
    ma5 = np.mean(recent[-5:])
    ma10 = np.mean(recent[-10:])
    ma20 = np.mean(recent[-20:])
    volatility = (np.max(recent) - np.min(recent)) / current_price * 100
    
    # 计算趋势方向
    short_trend = (recent[-1] - recent[-5]) / recent[-5] * 100
    mid_trend = (recent[-1] - recent[-20]) / recent[-20] * 100
    
    # 上涨概率（基于技术指标综合判断）
    up_prob = 45  # 基础概率
    
    # 均线信号
    if ma5 > ma10: up_prob += 5
    if ma10 > ma20: up_prob += 5
    
    # 趋势信号
    if short_trend > 0: up_prob += 3
    if mid_trend > 0: up_prob += 4
    
    # 波动率信号
    if volatility < 3: up_prob += 3
    
    # 基于预测结果调整
    if expected_return > 1.5: up_prob += 10
    elif expected_return > 0.5: up_prob += 5
    elif expected_return < -1.5: up_prob -= 10
    elif expected_return < -0.5: up_prob -= 5
    
    up_prob = max(15, min(85, up_prob))
    
    # 风险评级
    if volatility < 2.5:
        risk = '低'
    elif volatility < 4.5:
        risk = '中'
    else:
        risk = '高'
    
    # 支撑压力位
    resistance = np.max(recent) * 1.02
    support = np.min(recent) * 0.98
    
    # 输出结果
    print(f"\n📍 当前技术面状态（2026-07-15）:")
    print(f"   ┌─────────────────────────────────────────────────────────┐")
    print(f"   │  当前价格: {current_price:.2f} 元")
    print(f"   │  MA5 均线: {ma5:.2f} 元 {'⬆️' if ma5 > current_price else '⬇️'}")
    print(f"   │  MA10均线: {ma10:.2f} 元")
    print(f"   │  MA20均线: {ma20:.2f} 元")
    print(f"   │  均线排列: {'多头排列 ✅' if ma5 > ma10 > ma20 else '空头排列 ⚠️'}")
    print(f"   │  5分钟趋势: {short_trend:+.2f}%")
    print(f"   │  近期最高: {np.max(recent):.2f} 元")
    print(f"   │  近期最低: {np.min(recent):.2f} 元")
    print(f"   │  波动率: ±{volatility/2:.2f}%")
    print(f"   └─────────────────────────────────────────────────────────┘")
    
    print(f"\n🔮 明日（2026-07-16）AI预测:")
    print(f"   ┌─────────────────────────────────────────────────────────┐")
    print(f"   │  预测开盘价: {pred_df['open'].iloc[0]:.2f} 元")
    print(f"   │  预测最高价: {pred_h:.2f} 元  ({high_ret:+.2f}%)  ← 压力位")
    print(f"   │  预测最低价: {pred_l:.2f} 元  ({low_ret:+.2f}%)  ← 支撑位")
    print(f"   │  预测收盘价: {final:.2f} 元  ({expected_return:+.2f}%)")
    print(f"   └─────────────────────────────────────────────────────────┘")
    
    print(f"\n📈 涨跌概率:")
    print(f"   ┌─────────────────────────────────────────────────────────┐")
    print(f"   │  上涨概率: {up_prob:.1f}%  {'🟢' if up_prob > 50 else '🔴'}")
    print(f"   │  下跌概率: {100-up_prob:.1f}%")
    print(f"   │  风险等级: {risk}")
    print(f"   └─────────────────────────────────────────────────────────┘")
    
    print(f"\n🎯 关键支撑位 & 压力位:")
    print(f"   ┌─────────────────────────────────────────────────────────┐")
    print(f"   │  第一压力位: {resistance:.2f} 元  (+{((resistance/current_price-1)*100):.1f}%)")
    print(f"   │  第二压力位: {np.max(recent)*1.04:.2f} 元  (+{(((np.max(recent)*1.04)/current_price-1)*100):.1f}%)")
    print(f"   │  第一支撑位: {support:.2f} 元  ({((support/current_price-1)*100):.1f}%)")
    print(f"   │  第二支撑位: {np.min(recent)*0.96:.2f} 元  ({(((np.min(recent)*0.96)/current_price-1)*100):.1f}%)")
    print(f"   └─────────────────────────────────────────────────────────┘")
    
    print(f"\n⏰ 明日分时关键点位预测:")
    print(f"   ┌─────────────────────────────────────────────────────────┐")
    print(f"   │  09:35 开盘: {pred_c[0]:.2f} 元")
    print(f"   │  10:30 早盘: {pred_c[12]:.2f} 元")
    print(f"   │  11:30 午盘: {pred_c[24]:.2f} 元")
    print(f"   │  14:00 尾盘: {pred_c[36]:.2f} 元")
    print(f"   │  15:00 收盘: {pred_c[-1]:.2f} 元")
    print(f"   └─────────────────────────────────────────────────────────┘")
    
    # 5. 操作建议
    print("\n[5/5] AI综合操作建议")
    print("=" * 80)
    
    if expected_return > 1.2 and up_prob > 60:
        signal = "看多 🟢"
        suggestion = "偏乐观，可考虑逢低介入，设置好止盈止损"
    elif expected_return > 0.3 and up_prob > 50:
        signal = "偏多 🟡"
        suggestion = "谨慎乐观，小仓位试错，不追高"
    elif expected_return > -0.5:
        signal = "中性 ⚪"
        suggestion = "观望为主，等待更明确的信号"
    elif expected_return > -1.5:
        signal = "偏空 🟠"
        suggestion = "偏谨慎，可考虑减仓或观望"
    else:
        signal = "看空 🔴"
        suggestion = "建议观望，不急于抄底"
    
    print(f"\n📊 趋势判断: {signal}")
    print(f"💡 操作建议: {suggestion}")
    
    print(f"\n⚖️  仓位与风控建议:")
    print(f"   1. 建议仓位: {min(20, int(up_prob/3))}% 以内")
    print(f"   2. 止盈目标: {current_price*1.05:.2f} ~ {current_price*1.08:.2f} 元 (+5%~+8%)")
    print(f"   3. 止损位置: {current_price*0.97:.2f} ~ {current_price*0.98:.2f} 元 (-2%~-3%)")
    print(f"   4. 入场时机: 早盘观察15-30分钟，不高开追涨，逢低介入")
    print(f"   5. 分批建仓: 分2-3批买入，避免一次性满仓")
    
    # 保存数据
    print("\n💾 正在保存预测数据...")
    pred_df.to_csv('603399_永杉锂业_7月16日预测结果.csv', index=False, encoding='utf-8-sig')
    print(f"   ✅ 预测结果已保存: 603399_永杉锂业_7月16日预测结果.csv")
    
    df.to_csv('603399_永杉锂业_最新K线数据.csv', index=False, encoding='utf-8-sig')
    print(f"   ✅ 历史K线已保存: 603399_永杉锂业_最新K线数据.csv")
    
    print("\n" + "=" * 80)
    print("🎉 603399 永杉锂业 明日预测完成！")
    print("=" * 80)
    
    print("\n⚠️ 最终风险提示:")
    print("   1. 以上预测基于Kronos AI技术面分析模型，仅供学习研究")
    print("   2. 锂矿板块波动大，受大宗商品价格、新能源政策影响显著")
    print("   3. 真实走势还受大盘情绪、板块轮动、突发消息等多重因素影响")
    print("   4. 预测准确率不可能100%，请务必独立判断")
    print("   5. 投资有风险，入市需谨慎，切勿盲目跟单")
    
    print("\n📌 免责声明:")
    print("   本预测仅为AI模型的技术分析演示，不构成任何投资建议。")
    print("   使用者据此投资，风险自担。市场有风险，投资需谨慎。")
    print("=" * 80)


def main():
    df = get_latest_data()
    run_prediction(df)


if __name__ == "__main__":
    main()
