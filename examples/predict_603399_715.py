#!/usr/bin/env python3
"""
603399 永杉锂业 - 真实时间预测（2026年7月15日）
预测今天（7月15日）之后的价格走势
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


def generate_realistic_data(stock_code="603399", days=12):
    """
    生成真实时间的数据
    数据截止到：2026年7月15日（今天）
    """
    print(f"📊 生成 {stock_code}（永杉锂业）真实时间数据...")
    print(f"   数据截止到：2026年7月15日（今天）")
    
    # 交易时间：每个交易日 4小时 = 48条 5分钟K线
    bars_per_day = 48
    total_bars = days * bars_per_day
    
    # 生成时间戳 - 从12天前开始，截止到今天（7月15日）
    dates = []
    current = datetime(2026, 7, 15, 15, 0) - timedelta(days=days+5)  # 倒推
    
    day_count = 0
    while len(dates) < total_bars:
        # 跳过周末
        while current.weekday() >= 5:
            current += timedelta(days=1)
        
        # 上午 9:30-11:30 (24条)
        morning_start = current.replace(hour=9, minute=30, second=0, microsecond=0)
        for i in range(24):
            dates.append(morning_start + timedelta(minutes=5*i))
            if len(dates) >= total_bars:
                break
        
        if len(dates) >= total_bars:
            break
            
        # 下午 13:00-15:00 (24条)
        afternoon_start = current.replace(hour=13, minute=0, second=0, microsecond=0)
        for i in range(24):
            dates.append(afternoon_start + timedelta(minutes=5*i))
            if len(dates) >= total_bars:
                break
        
        current += timedelta(days=1)
        day_count += 1
    
    # 生成价格数据（基于永杉锂业真实价格水平：约8-9元）
    base_price = 8.50
    np.random.seed(42)
    
    t = np.arange(len(dates))
    
    # 更真实的价格模式：有涨有跌，带波动
    trend = 0.0001 * t  # 非常轻微的上升趋势
    noise = np.random.normal(0, 0.012, len(dates))
    cycles = 0.015 * np.sin(t / 40) + 0.008 * np.sin(t / 15)
    
    # 累计变化
    returns = trend + noise + cycles * 0.1
    price_factors = 1 + np.cumsum(returns) / 20
    
    close_prices = base_price * price_factors
    
    # 生成 OHLC
    open_prices = close_prices.copy()
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.007, len(dates))))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.007, len(dates))))
    
    # 成交量
    volume_base = 45000
    volume = volume_base * (1 + 0.4 * np.random.rand(len(dates)) + 0.2 * cycles)
    
    # 成交额
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
    
    # 确保 OHLC 逻辑正确
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    print(f"✅ 生成完成! 共 {len(df)} 条 5分钟 K 线")
    print(f"   时间范围: {df['timestamps'].iloc[0]} ~ {df['timestamps'].iloc[-1]}")
    print(f"   价格范围: {df['low'].min():.2f} ~ {df['high'].max():.2f} 元")
    print(f"   今日（7/15）收盘价: {df['close'].iloc[-1]:.2f} 元")
    
    return df


def generate_future_timestamps(start_time, count):
    """生成未来的时间戳（用于预测）"""
    timestamps = []
    current = start_time
    
    while len(timestamps) < count:
        # 如果是收盘时间，跳到下一个交易日
        if current.hour >= 15:
            current = current.replace(hour=9, minute=30, second=0, microsecond=0)
            current += timedelta(days=1)
            # 跳过周末
            while current.weekday() >= 5:
                current += timedelta(days=1)
            continue
        
        # 如果是午间休市，跳到下午
        if 11 < current.hour < 13 or (current.hour == 11 and current.minute >= 30):
            current = current.replace(hour=13, minute=0, second=0, microsecond=0)
            continue
        
        timestamps.append(current)
        current += timedelta(minutes=5)
    
    return timestamps


def plot_prediction(kline_df, pred_df, save_path='603399_永杉锂业_7月15日预测.png'):
    """绘制预测结果图"""
    # 设置索引对齐
    kline_df = kline_df.reset_index(drop=True)
    pred_df = pred_df.reset_index(drop=True)
    
    # 构建完整数据
    full_close = pd.concat([
        pd.Series(kline_df['close'].values, name='历史价格'),
        pd.Series(pred_df['close'].values, name='预测价格')
    ], axis=0).reset_index(drop=True)
    
    full_volume = pd.concat([
        pd.Series(kline_df['volume'].values, name='历史成交量'),
        pd.Series(pred_df['volume'].values, name='预测成交量')
    ], axis=0).reset_index(drop=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('603399 永杉锂业 - 7月15日 实时预测', fontsize=16, y=0.98, fontweight='bold')

    # 价格图
    ax1.plot(range(len(kline_df)), kline_df['close'], label='历史价格（截至7/15）', 
             color='blue', linewidth=1.5, alpha=0.8)
    ax1.plot(range(len(kline_df)-1, len(kline_df)+len(pred_df)-1), pred_df['close'], 
             label='预测价格（7/15之后）', color='red', linewidth=2)
    ax1.axvline(x=len(kline_df)-1, color='gray', linestyle='--', alpha=0.7, linewidth=2, label='今天（7/15）')
    ax1.set_ylabel('价格 (元)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('价格走势预测', fontsize=13)

    # 成交量图
    ax2.plot(range(len(kline_df)), kline_df['volume'], label='历史成交量', 
             color='blue', linewidth=1.5, alpha=0.8)
    ax2.plot(range(len(kline_df)-1, len(kline_df)+len(pred_df)-1), pred_df['volume'], 
             label='预测成交量', color='red', linewidth=2)
    ax2.axvline(x=len(kline_df)-1, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    ax2.set_ylabel('成交量', fontsize=12)
    ax2.set_xlabel('时间（5分钟K线序号）', fontsize=12)
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('成交量预测', fontsize=13)

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n🖼️  预测结果图已保存到: {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print("🔮 Kronos - 603399 永杉锂业 实时预测")
    print(f"📅 今天是：2026年7月15日")
    print("⚠️  注意：使用模拟数据演示模型功能")
    print("=" * 70)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"使用设备: CPU (Intel Mac 专用)")

    # 1. 加载模型
    print("\n[1/5] 正在加载模型和分词器...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    print("✅ 模型加载完成!")

    # 2. 初始化预测器
    print("\n[2/5] 初始化预测器...")
    device = torch.device("cpu")
    predictor = KronosPredictor(model, tokenizer, device=str(device), max_context=512)
    print("✅ 预测器初始化完成!")

    # 3. 准备数据（截止到7月15日）
    print("\n[3/5] 准备历史数据（截止到7月15日收盘）...")
    df = generate_realistic_data("603399", days=12)

    # 保存数据
    df.to_csv("603399_截止7月15日_历史数据.csv", index=False)
    print("💾 历史数据已保存到: 603399_截止7月15日_历史数据.csv")

    lookback = 480  # 使用全部历史数据
    pred_len = 96   # 预测未来 2 个交易日（96条5分钟K线）

    # 确保数据足够
    if len(df) < lookback:
        lookback = len(df)
        print(f"   调整历史数据量为: {lookback} 条")

    x_df = df.iloc[:lookback,][['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df.iloc[:lookback]['timestamps']
    
    # 生成未来的时间戳
    last_time = df.iloc[-1]['timestamps']
    future_timestamps = generate_future_timestamps(last_time + timedelta(minutes=5), pred_len)
    y_timestamp = pd.Series(future_timestamps)

    print(f"   历史数据长度: {lookback} 条")
    print(f"   预测长度: {pred_len} 条（约 2 个交易日）")
    print(f"   预测时间范围: {y_timestamp.iloc[0]} ~ {y_timestamp.iloc[-1]}")

    # 4. 开始预测
    print("\n[4/5] 开始预测 603399 永杉锂业（7月15日之后）...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=3,
        verbose=True
    )
    print("✅ 预测完成!")

    # 5. 结果分析
    print("\n[5/5] 预测结果分析...")
    print("\n📈 预测数据前5条（7月15日之后）:")
    print(pred_df[['open', 'high', 'low', 'close', 'volume']].head())
    
    print(f"\n📊 预测统计:")
    current_price = df['close'].iloc[-1]
    start_price = pred_df['close'].iloc[0]
    end_price = pred_df['close'].iloc[-1]
    change_pct = (end_price - current_price) / current_price * 100
    max_price = pred_df['close'].max()
    min_price = pred_df['close'].min()
    
    print(f"   今日（7/15）收盘价: {current_price:.2f} 元")
    print(f"   预测起始价格: {start_price:.2f} 元")
    print(f"   预测结束价格: {end_price:.2f} 元")
    print(f"   预测期间最高: {max_price:.2f} 元")
    print(f"   预测期间最低: {min_price:.2f} 元")
    print(f"   预测涨跌幅度: {change_pct:+.2f}%")
    
    # 趋势判断
    trend = "看涨 ↑" if change_pct > 2 else ("看跌 ↓" if change_pct < -2 else "震荡 →")
    strength = "强" if abs(change_pct) > 5 else ("中" if abs(change_pct) > 2 else "弱")
    print(f"   预测趋势: {trend} ({strength})")
    
    # 给出简单的交易建议
    print(f"\n💡 简单交易建议:")
    if change_pct > 3:
        print(f"   模型预测上涨，可考虑持有或逢低买入")
    elif change_pct < -3:
        print(f"   模型预测下跌，可考虑减仓或观望")
    else:
        print(f"   模型预测震荡，可考虑高抛低吸或观望")

    # 绘制并保存
    plot_prediction(df, pred_df)
    
    # 保存预测结果
    pred_df.to_csv("603399_永杉锂业_7月15日之后预测.csv")
    print("💾 预测数据已保存到: 603399_永杉锂业_7月15日之后预测.csv")
    
    print("\n" + "=" * 70)
    print("🎉 603399 永杉锂业 预测任务完成!")
    print("=" * 70)
    print("\n📅 预测时间说明:")
    print("   历史数据: 截止到 2026年7月15日（今天收盘）")
    print("   预测范围: 2026年7月15日之后的 2 个交易日")
    print(f"   预测时间点: {y_timestamp.iloc[0]} ~ {y_timestamp.iloc[-1]}")
    print("\n⚠️ 风险提示:")
    print("   1. 此预测基于模拟数据，仅供演示 Kronos 模型功能")
    print("   2. 股票投资有风险，预测结果仅供参考")
    print("   3. 真实预测请使用真实的 5分钟 K 线数据")
    print("\n🔧 使用真实数据命令:")
    print(f"   ../kronos_env/bin/python predict_stock.py --data 你的数据.csv --stock-name 永杉锂业")
    print("=" * 70)


if __name__ == "__main__":
    main()
