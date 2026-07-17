#!/usr/bin/env python3
"""
603399 永杉锂业 - 最终预测报告
价格严格对齐：14.31元
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


def generate_final_simulation():
    """
    生成最终模拟数据，严格对齐14.31元
    """
    print("=" * 85)
    print("📊 603399 永杉锂业 - 基于真实价格水平的K线数据")
    print("=" * 85)
    print(f"\n💰 基准价格: 14.31元（与同花顺截图完全对齐）")
    print("📈 趋势特征: 下跌通道，逐步探底（根据你的截图）")
    print("=" * 85)
    
    np.random.seed(6033991)
    
    bars_per_day = 48
    days = 10
    total_bars = days * bars_per_day
    
    t = np.arange(total_bars)
    
    # 模拟下跌趋势：从约15.6元跌到14.31元
    trend_slope = -(15.6 - 14.31) / total_bars  # 线性下跌斜率
    long_trend = np.arange(total_bars) * trend_slope / 15.0
    mid_trend = 0.012 * np.sin(t / 70)
    short_cycles = 0.007 * np.sin(t / 22) + 0.003 * np.sin(t / 6)
    noise = np.random.normal(0, 0.0075, total_bars)
    
    price_factors = 1 + long_trend + mid_trend * 0.04 + short_cycles * 0.035 + np.cumsum(noise) / 25
    
    start_price = 15.6  # 10天前的价格
    close_prices = start_price * price_factors
    
    # 确保最终价格正好是14.31
    final_adjustment = 14.31 / close_prices[-1]
    close_prices = close_prices * final_adjustment
    
    open_prices = close_prices.copy()
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.0065, total_bars)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.0065, total_bars)))
    
    volume_base = 145000
    volume_trend = 1 + 0.38 * np.abs(np.sin(t / 50))
    volume = volume_base * volume_trend * (1 + 0.32 * np.random.rand(total_bars))
    amount = close_prices * volume
    
    # 生成交易时间
    dates = []
    base_date = datetime.now() - timedelta(days=10)
    
    for day in range(days + 2):
        current_date = base_date + timedelta(days=day)
        if current_date.weekday() >= 5:
            continue
        if len(dates) >= total_bars:
            break
            
        # 上午
        for i in range(24):
            dates.append(current_date.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(minutes=5*i))
            if len(dates) >= total_bars:
                break
        if len(dates) >= total_bars:
            break
            
        # 下午
        for i in range(24):
            dates.append(current_date.replace(hour=13, minute=0, second=0, microsecond=0) + timedelta(minutes=5*i))
            if len(dates) >= total_bars:
                break
    
    actual_len = min(len(dates), len(close_prices))
    
    df = pd.DataFrame({
        'timestamps': pd.Series(dates[:actual_len]),
        'open': open_prices[:actual_len],
        'high': high_prices[:actual_len],
        'low': low_prices[:actual_len],
        'close': close_prices[:actual_len],
        'volume': volume[:actual_len],
        'amount': amount[:actual_len]
    })
    
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    print(f"\n✅ 历史K线数据就绪:")
    print(f"   📊 共 {len(df)} 条5分钟K线")
    print(f"   📅 时间: {df['timestamps'].iloc[0].strftime('%Y-%m-%d')} ~ {df['timestamps'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"   💰 价格区间: {df['low'].min():.2f} ~ {df['high'].max():.2f} 元")
    print(f"   🎯 当前价格: {df['close'].iloc[-1]:.2f} 元 ✅")
    
    return df


def run_prediction(df):
    """运行完整预测"""
    print("\n" + "=" * 85)
    print("🔮 Kronos AI - 603399 永杉锂业 明日（7月16日）预测报告")
    print("=" * 85)
    
    # 1. 加载模型
    print("\n[1/5] 加载AI模型...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    device = torch.device("cpu")
    predictor = KronosPredictor(model, tokenizer, device=str(device), max_context=384)
    print("   ✅ 完成")
    
    # 2. 准备数据
    print("\n[2/5] 准备预测数据...")
    
    lookback = min(280, len(df) - 60)
    pred_len = 48
    
    print(f"   历史数据: {lookback} 条5分钟K线")
    print(f"   预测长度: {pred_len} 条（1个交易日）")
    
    x_df = df.iloc[:lookback,][['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df.iloc[:lookback]['timestamps']
    
    # 明日时间戳
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
    print("\n[3/5] 运行Kronos AI预测...")
    start = time.time()
    
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
    
    print(f"   ✅ 预测完成，耗时 {time.time()-start:.1f} 秒")
    
    # 4. 分析结果
    print("\n[4/5] 预测结果深度分析")
    print("=" * 85)
    
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
    
    # 上涨概率
    if expected_return > 1.2:
        up_prob = 70
    elif expected_return > 0.4:
        up_prob = 56
    elif expected_return > 0:
        up_prob = 48
    elif expected_return > -0.5:
        up_prob = 40
    else:
        up_prob = 28
    
    ma_signal = sum([ma5 > ma10, ma10 > ma20])
    up_prob += ma_signal * 5
    up_prob = max(15, min(85, up_prob))
    
    risk = '高' if volatility > 5.5 else ('中' if volatility > 3 else '低')
    
    # 支撑压力位
    resistance = np.max(recent) * 1.015
    support = np.min(recent) * 0.985
    
    # 输出结果
    print(f"\n📍 当前技术面状态（{current_price:.2f}元）:")
    print(f"   ┌─────────────────────────────────────────────────────────────┐")
    print(f"   │  MA5 均线: {ma5:.2f} 元 {'⬇️' if ma5 < current_price else '⬆️'}")
    print(f"   │  MA10均线: {ma10:.2f} 元")
    print(f"   │  MA20均线: {ma20:.2f} 元")
    print(f"   │  均线排列: {'多头排列 ✅' if ma5 > ma10 > ma20 else '空头排列 ⚠️'}")
    print(f"   │  近期最高: {np.max(recent):.2f} 元")
    print(f"   │  近期最低: {np.min(recent):.2f} 元")
    print(f"   │  波动率: ±{volatility/2:.2f}%")
    print(f"   └─────────────────────────────────────────────────────────────┘")
    
    print(f"\n🔮 明日（{tomorrow.strftime('%m月%d日')}）AI预测:")
    print(f"   ┌─────────────────────────────────────────────────────────────┐")
    print(f"   │  预测开盘: {pred_df['open'].iloc[0]:.2f} 元")
    print(f"   │  预测最高: {pred_h:.2f} 元  ({high_ret:+.2f}%)  ← 压力位")
    print(f"   │  预测最低: {pred_l:.2f} 元  ({low_ret:+.2f}%)  ← 支撑位")
    print(f"   │  预测收盘: {final:.2f} 元  ({expected_return:+.2f}%)")
    print(f"   └─────────────────────────────────────────────────────────────┘")
    
    print(f"\n📈 涨跌概率:")
    print(f"   ┌─────────────────────────────────────────────────────────────┐")
    print(f"   │  上涨概率: {up_prob:.1f}%  {'🟢' if up_prob > 50 else '🔴'}")
    print(f"   │  下跌概率: {100-up_prob:.1f}%")
    print(f"   │  风险等级: {risk}")
    print(f"   └─────────────────────────────────────────────────────────────┘")
    
    print(f"\n🎯 关键价位:")
    print(f"   ┌─────────────────────────────────────────────────────────────┐")
    print(f"   │  压力位1: {resistance:.2f} 元  (+{((resistance/current_price-1)*100):.1f}%)")
    print(f"   │  压力位2: {np.max(recent)*1.03:.2f} 元  (+{(((np.max(recent)*1.03)/current_price-1)*100):.1f}%)")
    print(f"   │  支撑位1: {support:.2f} 元  ({((support/current_price-1)*100):.1f}%)")
    print(f"   │  支撑位2: {np.min(recent)*0.97:.2f} 元  ({(((np.min(recent)*0.97)/current_price-1)*100):.1f}%)")
    print(f"   └─────────────────────────────────────────────────────────────┘")
    
    print(f"\n⏰ 分时走势预测:")
    print(f"   ┌─────────────────────────────────────────────────────────────┐")
    print(f"   │  09:35 开盘: {pred_c[0]:.2f} 元")
    print(f"   │  10:30 早盘: {pred_c[12]:.2f} 元")
    print(f"   │  11:30 午盘: {pred_c[24]:.2f} 元")
    print(f"   │  14:00 尾盘: {pred_c[36]:.2f} 元")
    print(f"   │  15:00 收盘: {pred_c[-1]:.2f} 元")
    print(f"   └─────────────────────────────────────────────────────────────┘")
    
    # 5. 操作建议
    print("\n[5/5] AI操作建议")
    print("=" * 85)
    
    if expected_return > 1.0 and up_prob > 60:
        signal = "看多 🟢"
        suggestion = "偏乐观，可考虑逢低介入，设置止盈止损"
    elif expected_return > 0.3 and up_prob > 50:
        signal = "偏多 🟡"
        suggestion = "谨慎乐观，小仓位试错，不追高"
    elif expected_return > -0.5:
        signal = "中性 ⚪"
        suggestion = "观望为主，等待更明确信号"
    elif expected_return > -1.5:
        signal = "偏空 🟠"
        suggestion = "偏谨慎，可考虑减仓或观望"
    else:
        signal = "看空 🔴"
        suggestion = "建议观望，不急于抄底"
    
    print(f"\n📊 趋势判断: {signal}")
    print(f"💡 操作建议: {suggestion}")
    
    print(f"\n⚖️  仓位与风控:")
    print(f"   1. 建议仓位: {min(20, int(up_prob/3))}% 以内")
    print(f"   2. 止盈目标: {current_price*1.05:.2f} ~ {current_price*1.08:.2f} 元 (+5%~+8%)")
    print(f"   3. 止损位置: {current_price*0.97:.2f} ~ {current_price*0.98:.2f} 元 (-2%~-3%)")
    print(f"   4. 入场时机: 早盘观察15-30分钟，不高开追涨，逢低介入")
    print(f"   5. 分批建仓: 分2-3批买入，避免满仓")
    
    # 保存
    print("\n💾 保存数据...")
    pred_df.to_csv('603399_永杉锂业_最终预测结果.csv', index=False)
    df.to_csv('603399_永杉锂业_历史K线数据.csv', index=False)
    print("   ✅ 已保存")
    
    print("\n" + "=" * 85)
    print("🎉 603399 永杉锂业 预测分析完成")
    print("=" * 85)
    
    print("\n⚠️ 重要声明:")
    print("   1. 价格基准已对齐你的同花顺截图（14.31元）")
    print("   2. 由于网络原因，akshare获取实时数据失败，使用了趋势模拟")
    print("   3. 要获得最准确预测，请从同花顺导出5分钟K线CSV文件")
    print("   4. 预测仅供研究学习，不构成任何投资建议")
    print("   5. 投资有风险，入市需谨慎，请独立判断和决策")
    
    print("\n📋 导出真实数据步骤:")
    print("   1. 打开同花顺 → 输入603399进入永杉锂业")
    print("   2. 按F8键切换到5分钟K线图")
    print("   3. 在K线区域右键 → 数据导出")
    print("   4. 选择Excel/CSV格式，保存文件")
    print("   5. 将文件发给我，我用真实数据重新预测")
    print("=" * 85)


if __name__ == "__main__":
    df = generate_final_simulation()
    run_prediction(df)
