#!/usr/bin/env python3
"""
000776 广发证券 - 今日（7月15日）行情预测 (简化版)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import torch
from datetime import datetime, timedelta

sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor


def generate_gf_data(days=8):
    """生成广发证券模拟数据"""
    print("📊 生成 000776（广发证券）历史数据...")
    
    bars_per_day = 48
    total_bars = days * bars_per_day
    
    dates = []
    current = datetime(2026, 7, 15, 10, 30) - timedelta(days=days+3)
    
    while len(dates) < total_bars:
        while current.weekday() >= 5:
            current += timedelta(days=1)
        
        morning_start = current.replace(hour=9, minute=30, second=0, microsecond=0)
        for i in range(24):
            dates.append(morning_start + timedelta(minutes=5*i))
            if len(dates) >= total_bars:
                break
        
        if len(dates) >= total_bars:
            break
            
        afternoon_start = current.replace(hour=13, minute=0, second=0, microsecond=0)
        for i in range(24):
            dates.append(afternoon_start + timedelta(minutes=5*i))
            if len(dates) >= total_bars:
                break
        
        current += timedelta(days=1)
    
    base_price = 15.20
    np.random.seed(123)
    
    t = np.arange(len(dates))
    trend = 0.00008 * t
    noise = np.random.normal(0, 0.009, len(dates))
    cycles = 0.01 * np.sin(t / 30) + 0.005 * np.sin(t / 10)
    
    returns = trend + noise + cycles * 0.06
    price_factors = 1 + np.cumsum(returns) / 28
    
    close_prices = base_price * price_factors
    open_prices = close_prices.copy()
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.0055, len(dates))))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.0055, len(dates))))
    
    volume_base = 85000
    volume = volume_base * (1 + 0.3 * np.random.rand(len(dates)) + 0.15 * cycles)
    amount = close_prices * volume
    
    df = pd.DataFrame({
        'timestamps': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume,
        'amount': amount
    })
    
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    print(f"✅ 数据生成完成! 共 {len(df)} 条 5分钟 K 线")
    print(f"   时间范围: {df['timestamps'].iloc[0]} ~ {df['timestamps'].iloc[-1]}")
    print(f"   价格范围: {df['low'].min():.2f} ~ {df['high'].max():.2f} 元")
    print(f"   当前价格: {df['close'].iloc[-1]:.2f} 元")
    
    return df


def generate_today_timestamps(start_hour=10, start_minute=35, count=40):
    """生成今天剩余交易时间的时间戳"""
    timestamps = []
    current = datetime(2026, 7, 15, start_hour, start_minute, 0)
    
    while len(timestamps) < count:
        if current.hour < 11 or (current.hour == 11 and current.minute < 30):
            timestamps.append(current)
            current += timedelta(minutes=5)
        elif 11 <= current.hour < 13:
            current = current.replace(hour=13, minute=0, second=0, microsecond=0)
        elif 13 <= current.hour < 15:
            timestamps.append(current)
            current += timedelta(minutes=5)
        else:
            break
    
    return timestamps


def plot_prediction(hist_df, pred_df, save_path='000776_广发证券_今日行情预测.png'):
    """绘制预测结果图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))
    fig.suptitle('000776 广发证券 - 7月15日 行情预测', fontsize=18, y=0.98, fontweight='bold')
    
    recent_hist = hist_df.tail(50)
    hist_range = range(len(recent_hist))
    pred_range = range(len(recent_hist)-1, len(recent_hist)+len(pred_df)-1)
    
    ax1.plot(hist_range, recent_hist['close'], label='历史价格', 
             color='blue', linewidth=2, alpha=0.85)
    ax1.plot(pred_range, pred_df['close'], label='预测价格', 
             color='red', linewidth=2.5)
    ax1.axvline(x=len(recent_hist)-1, color='green', linestyle='--', 
                alpha=0.8, linewidth=2, label='当前时刻')
    ax1.set_ylabel('价格 (元)', fontsize=13)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('价格走势', fontsize=14)

    ax2.plot(hist_range, recent_hist['volume'], label='历史成交量', 
             color='blue', linewidth=1.5, alpha=0.85)
    ax2.plot(pred_range, pred_df['volume'], label='预测成交量', 
             color='red', linewidth=2)
    ax2.axvline(x=len(recent_hist)-1, color='green', linestyle='--', alpha=0.8, linewidth=2)
    ax2.set_ylabel('成交量', fontsize=13)
    ax2.set_xlabel('5分钟K线序号', fontsize=13)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('成交量', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n🖼️  预测行情图已保存: {save_path}")
    plt.close()


def main():
    print("=" * 80)
    print("🔮 Kronos AI - 000776 广发证券 今日行情预测")
    print(f"📅 日期：2026年7月15日")
    print("⚠️  注意：使用模拟数据演示模型功能")
    print("=" * 80)

    # 1. 加载模型
    print("\n[1/5] 加载模型...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    print("✅ 模型加载完成")

    # 2. 初始化预测器
    print("\n[2/5] 初始化预测器...")
    device = torch.device("cpu")
    predictor = KronosPredictor(model, tokenizer, device=str(device), max_context=384)
    print("✅ 预测器初始化完成")

    # 3. 准备数据
    print("\n[3/5] 准备历史数据...")
    df = generate_gf_data(days=8)
    df.to_csv("000776_广发证券_历史数据.csv", index=False)

    lookback = 320  # 减少历史数据量
    pred_len = 36   # 预测3小时
    
    x_df = df.iloc[:lookback,][['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df.iloc[:lookback]['timestamps']
    future_timestamps = generate_today_timestamps(10, 35, pred_len)
    y_timestamp = pd.Series(future_timestamps)

    print(f"   历史数据: {lookback} 条")
    print(f"   预测时长: {pred_len} 条 (约3小时)")

    # 4. 开始预测
    print("\n[4/5] 开始预测今日行情...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=0.8,
        top_p=0.95,
        sample_count=3,
        verbose=True
    )
    print("✅ 预测完成!")

    # 5. 分析结果
    print("\n" + "=" * 80)
    print("📊 广发证券（000776）7月15日 行情预测分析")
    print("=" * 80)
    
    current_price = df['close'].iloc[lookback-1]
    today_data = df.iloc[lookback-12:lookback]
    today_open = today_data['open'].iloc[0]
    today_high = today_data['high'].max()
    today_low = today_data['low'].min()
    
    print(f"\n📍 当前情况:")
    print(f"   今日开盘: {today_open:.2f} 元")
    print(f"   今日最高: {today_high:.2f} 元")
    print(f"   今日最低: {today_low:.2f} 元")
    print(f"   当前价格: {current_price:.2f} 元")
    print(f"   目前涨跌: {(current_price - today_open)/today_open*100:+.2f}%")
    
    print(f"\n🔮 预测今日剩余时间走势:")
    pred_high = pred_df['high'].max()
    pred_low = pred_df['low'].min()
    pred_end = pred_df['close'].iloc[-1]
    
    print(f"   预测收盘: {pred_end:.2f} 元")
    print(f"   预测最高: {pred_high:.2f} 元")
    print(f"   预测最低: {pred_low:.2f} 元")
    
    daily_change = (pred_end - today_open) / today_open * 100
    print(f"   预测今日涨跌: {daily_change:+.2f}%")
    
    # 趋势判断
    if daily_change > 1.2:
        trend = "上涨 🟢"
        advice = "持有为主，逢高可适量止盈"
    elif daily_change > 0.3:
        trend = "小涨 🟢"
        advice = "继续持有"
    elif daily_change > -0.3:
        trend = "横盘震荡 🟡"
        advice = "观望为主，可高抛低吸"
    elif daily_change > -1.2:
        trend = "小幅调整 🔴"
        advice = "可考虑减仓部分"
    else:
        trend = "下跌 🔴"
        advice = "建议控制仓位，注意风险"
    
    print(f"\n📈 趋势判断: {trend}")
    print(f"💡 交易建议: {advice}")
    
    # 绘制图表
    plot_prediction(df.iloc[:lookback], pred_df)
    
    pred_df.to_csv("000776_广发证券_预测结果.csv")
    print(f"\n💾 预测数据已保存")
    
    print("\n" + "=" * 80)
    print("🎉 预测完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
