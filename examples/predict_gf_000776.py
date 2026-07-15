#!/usr/bin/env python3
"""
000776 广发证券 - 今日（7月15日）行情预测
预测今天的日内走势
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


def generate_gf_data(days=10):
    """生成广发证券模拟数据，截止到今天早盘"""
    print("📊 生成 000776（广发证券）历史数据...")
    
    bars_per_day = 48
    total_bars = days * bars_per_day
    
    dates = []
    current = datetime(2026, 7, 15, 10, 30) - timedelta(days=days+3)
    
    while len(dates) < total_bars:
        # 跳过周末
        while current.weekday() >= 5:
            current += timedelta(days=1)
        
        # 上午 9:30-11:30
        morning_start = current.replace(hour=9, minute=30, second=0, microsecond=0)
        for i in range(24):
            dates.append(morning_start + timedelta(minutes=5*i))
            if len(dates) >= total_bars:
                break
        
        if len(dates) >= total_bars:
            break
            
        # 下午 13:00-15:00
        afternoon_start = current.replace(hour=13, minute=0, second=0, microsecond=0)
        for i in range(24):
            dates.append(afternoon_start + timedelta(minutes=5*i))
            if len(dates) >= total_bars:
                break
        
        current += timedelta(days=1)
    
    # 生成价格数据（广发证券真实价格水平：约14-16元）
    base_price = 15.20
    np.random.seed(123)
    
    t = np.arange(len(dates))
    trend = 0.00005 * t
    noise = np.random.normal(0, 0.01, len(dates))
    cycles = 0.012 * np.sin(t / 35) + 0.006 * np.sin(t / 12)
    
    returns = trend + noise + cycles * 0.08
    price_factors = 1 + np.cumsum(returns) / 25
    
    close_prices = base_price * price_factors
    
    open_prices = close_prices.copy()
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.006, len(dates))))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.006, len(dates))))
    
    volume_base = 85000
    volume = volume_base * (1 + 0.35 * np.random.rand(len(dates)) + 0.18 * cycles)
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
    print(f"   当前价格（7/15 10:30）: {df['close'].iloc[-1]:.2f} 元")
    
    return df


def generate_today_timestamps(start_hour=10, start_minute=35, count=50):
    """生成今天剩余交易时间的时间戳"""
    timestamps = []
    current = datetime(2026, 7, 15, start_hour, start_minute, 0)
    
    while len(timestamps) < count:
        # 上午交易时间
        if current.hour < 11 or (current.hour == 11 and current.minute < 30):
            timestamps.append(current)
            current += timedelta(minutes=5)
        # 午间休市，跳到下午
        elif 11 <= current.hour < 13:
            current = current.replace(hour=13, minute=0, second=0, microsecond=0)
        # 下午交易时间
        elif 13 <= current.hour < 15:
            timestamps.append(current)
            current += timedelta(minutes=5)
        # 收盘，停止
        else:
            break
    
    return timestamps


def plot_today_prediction(hist_df, pred_df, save_path='000776_广发证券_7月15日行情预测.png'):
    """绘制今日行情预测图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))
    fig.suptitle('000776 广发证券 - 7月15日 实时行情预测', fontsize=18, y=0.98, fontweight='bold')
    
    # 只显示最近的历史数据 + 预测
    recent_hist = hist_df.tail(60)  # 最近5小时
    hist_range = range(len(recent_hist))
    pred_range = range(len(recent_hist)-1, len(recent_hist)+len(pred_df)-1)
    
    # 价格图
    ax1.plot(hist_range, recent_hist['close'], label='历史价格（截至10:30）', 
             color='blue', linewidth=2, alpha=0.85)
    ax1.plot(pred_range, pred_df['close'], label='预测价格（10:30之后）', 
             color='red', linewidth=2.5)
    ax1.axvline(x=len(recent_hist)-1, color='green', linestyle='--', alpha=0.8, 
                linewidth=2, label='当前时刻（10:30）')
    ax1.set_ylabel('价格 (元)', fontsize=13)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('价格走势预测', fontsize=14)
    
    # 添加价格标注
    current_price = recent_hist['close'].iloc[-1]
    pred_end_price = pred_df['close'].iloc[-1]
    ax1.annotate(f'现价: {current_price:.2f}', 
                xy=(len(recent_hist)-1, current_price),
                xytext=(10, 30), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=11, color='green', fontweight='bold')
    
    change_pct = (pred_end_price - current_price) / current_price * 100
    ax1.annotate(f'目标价: {pred_end_price:.2f}\n预测涨跌: {change_pct:+.2f}%', 
                xy=(len(recent_hist)+len(pred_df)-2, pred_end_price),
                xytext=(-80, -40), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=11, color='red', fontweight='bold')

    # 成交量图
    ax2.plot(hist_range, recent_hist['volume'], label='历史成交量', 
             color='blue', linewidth=1.5, alpha=0.85)
    ax2.plot(pred_range, pred_df['volume'], label='预测成交量', 
             color='red', linewidth=2)
    ax2.axvline(x=len(recent_hist)-1, color='green', linestyle='--', alpha=0.8, linewidth=2)
    ax2.set_ylabel('成交量', fontsize=13)
    ax2.set_xlabel('交易时间（5分钟K线序号）', fontsize=13)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('成交量预测', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n🖼️  预测行情图已保存到: {save_path}")
    plt.close()


def main():
    print("=" * 80)
    print("🔮 Kronos AI - 000776 广发证券 今日行情预测")
    print(f"📅 当前时间：2026年7月15日 10:30（交易中）")
    print("⚠️  注意：使用模拟数据演示模型功能")
    print("=" * 80)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"使用设备: CPU (Intel Mac 专用)")

    # 1. 加载模型
    print("\n[1/5] 正在加载模型...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    print("✅ 模型加载完成!")

    # 2. 初始化预测器
    print("\n[2/5] 初始化预测器...")
    device = torch.device("cpu")
    predictor = KronosPredictor(model, tokenizer, device=str(device), max_context=512)
    print("✅ 预测器初始化完成!")

    # 3. 准备数据（截止到今天10:30）
    print("\n[3/5] 准备历史数据（截止到今天10:30）...")
    df = generate_gf_data(days=10)
    df.to_csv("000776_广发证券_历史数据.csv", index=False)
    print("💾 历史数据已保存")

    lookback = 400  # 使用历史数据量
    pred_len = 50   # 预测今天剩余交易时间
    
    x_df = df.iloc[:lookback,][['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df.iloc[:lookback]['timestamps']
    
    # 生成今天剩余时间的时间戳
    future_timestamps = generate_today_timestamps(10, 35, pred_len)
    y_timestamp = pd.Series(future_timestamps)

    print(f"   历史数据: {lookback} 条 5分钟 K 线")
    print(f"   预测时长: {pred_len} 条（约今天剩余 4 小时）")
    print(f"   预测时间: {y_timestamp.iloc[0]} ~ {y_timestamp.iloc[-1]}")

    # 4. 开始预测
    print("\n[4/5] 开始预测广发证券今日行情...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=0.8,
        top_p=0.95,
        sample_count=5,
        verbose=True
    )
    print("✅ 今日行情预测完成!")

    # 5. 详细分析
    print("\n" + "=" * 80)
    print("📊 广发证券（000776）7月15日 行情预测分析")
    print("=" * 80)
    
    current_price = df['close'].iloc[-1]
    today_open = df[df['timestamps'].dt.day == 15]['open'].iloc[0]
    today_high = df[df['timestamps'].dt.day == 15]['high'].max()
    today_low = df[df['timestamps'].dt.day == 15]['low'].min()
    
    print(f"\n📍 当前情况（10:30）:")
    print(f"   今日开盘价: {today_open:.2f} 元")
    print(f"   今日最高价: {today_high:.2f} 元")
    print(f"   今日最低价: {today_low:.2f} 元")
    print(f"   当前价格: {current_price:.2f} 元")
    print(f"   目前涨跌: {(current_price - today_open)/today_open*100:+.2f}%")
    
    print(f"\n🔮 预测今天剩余时间走势:")
    pred_prices = pred_df['close'].values
    pred_high = pred_df['high'].max()
    pred_low = pred_df['low'].min()
    pred_end = pred_prices[-1]
    
    print(f"   预测今日收盘: {pred_end:.2f} 元")
    print(f"   预测日内最高: {pred_high:.2f} 元")
    print(f"   预测日内最低: {pred_low:.2f} 元")
    
    # 完整涨跌计算
    daily_change = (pred_end - today_open) / today_open * 100
    print(f"   预测今日涨跌: {daily_change:+.2f}%")
    
    # 趋势判断
    if daily_change > 1.5:
        trend = "大涨 🟢"
        advice = "持仓待涨，可考虑加仓"
    elif daily_change > 0.5:
        trend = "小涨 🟢"
        advice = "持有为主，高抛低吸"
    elif daily_change > -0.5:
        trend = "横盘 🟡"
        advice = "观望为主，等待方向"
    elif daily_change > -1.5:
        trend = "小跌 🔴"
        advice = "适当减仓，控制仓位"
    else:
        trend = "大跌 🔴"
        advice = "减仓或离场观望"
    
    print(f"   趋势判断: {trend}")
    
    print(f"\n💡 交易建议:")
    print(f"   {advice}")
    
    # 分时走势关键点位
    print(f"\n🎯 关键点位预测:")
    mid_price = pred_prices[len(pred_prices)//2]
    print(f"   午盘（11:30）预测价: {mid_price:.2f} 元")
    print(f"   收盘（15:00）预测价: {pred_end:.2f} 元")
    
    # 波动分析
    price_range = pred_high - pred_low
    print(f"\n📈 日内波动分析:")
    print(f"   预计波动区间: {pred_low:.2f} ~ {pred_high:.2f} 元")
    print(f"   波动幅度: ±{price_range/2:.2f} 元 ({price_range/current_price*100/2:+.1f}%)")
    
    # 绘制图表
    plot_today_prediction(df, pred_df)
    
    # 保存预测结果
    pred_df.to_csv("000776_广发证券_7月15日预测结果.csv")
    print(f"\n💾 预测数据已保存到: 000776_广发证券_7月15日预测结果.csv")
    
    print("\n" + "=" * 80)
    print("🎉 广发证券（000776）今日行情预测完成!")
    print("=" * 80)
    print("\n⚠️ 重要提示:")
    print("   1. 以上预测基于模拟数据，仅作模型功能演示")
    print("   2. 真实市场受多种因素影响，请谨慎决策")
    print("   3. 如需真实预测，请导入真实的5分钟K线数据")
    print("=" * 80)


if __name__ == "__main__":
    main()
