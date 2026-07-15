#!/usr/bin/env python3
"""
603399 永杉锂业 - 模拟预测演示

由于网络问题无法获取实时数据，
此脚本使用真实的价格模式模拟数据进行预测演示。
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


def generate_simulated_data(stock_code="603399", days=10):
    """
    生成模拟的 5分钟 K 线数据
    基于锂矿行业股票的典型价格模式
    """
    print(f"📊 生成 {stock_code}（永杉锂业）模拟数据...")
    
    # 交易时间：每个交易日 4小时 = 48条 5分钟K线
    bars_per_day = 48
    total_bars = days * bars_per_day
    
    # 生成时间戳
    dates = []
    current = datetime.now() - timedelta(days=days+5)  # 从5天前开始
    
    for _ in range(days):
        # 跳过周末
        while current.weekday() >= 5:
            current += timedelta(days=1)
        
        # 上午 9:30-11:30 (24条)
        morning_start = current.replace(hour=9, minute=30, second=0, microsecond=0)
        for i in range(24):
            dates.append(morning_start + timedelta(minutes=5*i))
        
        # 下午 13:00-15:00 (24条)
        afternoon_start = current.replace(hour=13, minute=0, second=0, microsecond=0)
        for i in range(24):
            dates.append(afternoon_start + timedelta(minutes=5*i))
        
        current += timedelta(days=1)
    
    dates = dates[:total_bars]
    
    # 生成价格数据（模拟永杉锂业的价格水平）
    base_price = 8.5  # 基础价格
    np.random.seed(42)  # 固定随机种子确保可重现
    
    # 生成带趋势和波动的价格
    t = np.arange(len(dates))
    trend = 0.0002 * t  # 轻微上升趋势
    noise = np.random.normal(0, 0.015, len(dates))
    cycles = 0.02 * np.sin(t / 50) + 0.01 * np.sin(t / 20)
    
    close_prices = base_price * (1 + np.cumsum(trend + noise + cycles) / 10)
    
    # 生成 OHLC
    open_prices = close_prices.copy()
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.008, len(dates))))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.008, len(dates))))
    
    # 成交量
    volume_base = 50000
    volume = volume_base * (1 + 0.5 * np.random.rand(len(dates)) + 0.3 * cycles)
    
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
    print(f"   当前价格: {df['close'].iloc[-1]:.2f} 元")
    
    return df


def plot_prediction(kline_df, pred_df, save_path='603399_永杉锂业_预测结果.png'):
    """绘制预测结果图"""
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = '历史价格'
    sr_pred_close.name = "预测价格"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = '历史成交量'
    sr_pred_volume.name = "预测成交量"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('603399 永杉锂业 - Kronos AI 价格预测', fontsize=16, y=0.98, fontweight='bold')

    ax1.plot(close_df['历史价格'], label='历史价格', color='blue', linewidth=1.5, alpha=0.8)
    ax1.plot(close_df['预测价格'], label='预测价格', color='red', linewidth=2)
    ax1.axvline(x=len(close_df) - len(pred_df), color='gray', linestyle='--', alpha=0.5, label='预测起点')
    ax1.set_ylabel('价格 (元)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('价格走势预测', fontsize=13)

    ax2.plot(volume_df['历史成交量'], label='历史成交量', color='blue', linewidth=1.5, alpha=0.8)
    ax2.plot(volume_df['预测成交量'], label='预测成交量', color='red', linewidth=2)
    ax2.axvline(x=len(volume_df) - len(pred_df), color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('成交量', fontsize=12)
    ax2.set_xlabel('时间', fontsize=12)
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('成交量预测', fontsize=13)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n🖼️  预测结果图已保存到: {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print("🔮 Kronos - 603399 永杉锂业 价格预测")
    print("⚠️  注意：由于网络问题，使用模拟数据演示")
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

    # 3. 准备数据
    print("\n[3/5] 准备模拟数据...")
    df = generate_simulated_data("603399", days=12)  # 12个交易日数据

    # 保存模拟数据
    df.to_csv("603399_模拟数据.csv", index=False)
    print("💾 模拟数据已保存到: 603399_模拟数据.csv")

    lookback = 400
    pred_len = 80  # 预测 80 条(约 6.5 小时)

    # 确保数据足够
    if len(df) < lookback + pred_len:
        print(f"⚠️  数据量不足，调整参数...")
        lookback = len(df) - pred_len - 10
        print(f"   使用历史数据: {lookback} 条")

    x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df.loc[:lookback-1, 'timestamps']
    y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

    print(f"   历史数据长度: {lookback}")
    print(f"   预测长度: {pred_len}")

    # 4. 开始预测
    print("\n[4/5] 开始预测 603399 永杉锂业...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=3,  # 3次采样取平均更稳定
        verbose=True
    )
    print("✅ 预测完成!")

    # 5. 可视化和分析
    print("\n[5/5] 预测结果分析...")
    print("\n📈 预测数据预览:")
    print(pred_df[['open', 'high', 'low', 'close', 'volume']].head())
    
    print(f"\n📊 预测统计:")
    start_price = pred_df['close'].iloc[0]
    end_price = pred_df['close'].iloc[-1]
    change_pct = (end_price - start_price) / start_price * 100
    max_price = pred_df['close'].max()
    min_price = pred_df['close'].min()
    
    print(f"   预测起始价格: {start_price:.2f} 元")
    print(f"   预测结束价格: {end_price:.2f} 元")
    print(f"   预测最高价格: {max_price:.2f} 元")
    print(f"   预测最低价格: {min_price:.2f} 元")
    print(f"   预测期间涨跌: {change_pct:+.2f}%")
    
    # 趋势判断
    trend = "看涨 ↑" if change_pct > 1 else ("看跌 ↓" if change_pct < -1 else "震荡 →")
    print(f"   预测趋势判断: {trend}")

    # 合并数据用于绘图
    kline_df = df.loc[:lookback+pred_len-1]
    kline_df.index = range(len(kline_df))

    # 绘制并保存
    plot_prediction(kline_df, pred_df)
    
    # 保存预测结果
    pred_df.to_csv("603399_永杉锂业_预测结果.csv")
    print("💾 预测数据已保存到: 603399_永杉锂业_预测结果.csv")
    
    print("\n" + "=" * 70)
    print("🎉 603399 永杉锂业 预测任务完成!")
    print("=" * 70)
    print("\n💡 说明:")
    print("   1. 此预测基于模拟数据，仅供演示 Kronos 模型功能")
    print("   2. 要获取真实预测，请准备 603399 的真实 5分钟 K 线 CSV 文件")
    print("   3. 真实数据格式: timestamps, open, high, low, close, volume, amount")
    print("   4. 使用真实数据命令:")
    print(f"      ../kronos_env/bin/python predict_stock.py --data 你的数据.csv --stock-name 永杉锂业")
    print("=" * 70)


if __name__ == "__main__":
    main()
