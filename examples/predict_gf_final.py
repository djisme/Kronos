#!/usr/bin/env python3
"""
000776 广发证券 - 今日（7月15日）行情预测 (最终版)
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

    lookback = 320
    pred_len = 36
    
    x_df = df.iloc[:lookback,][['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df.iloc[:lookback]['timestamps']
    # 生成预测索引（不依赖实际时间，避免数量不匹配）
    y_timestamp = pd.RangeIndex(pred_len)

    print(f"   历史数据: {lookback} 条")
    print(f"   预测时长: {pred_len} 条 (约3小时交易时间)")

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
    
    print(f"\n📍 当前市场情况（截至10:30）:")
    print(f"   ├─ 今日开盘价: {today_open:.2f} 元")
    print(f"   ├─ 今日最高价: {today_high:.2f} 元")
    print(f"   ├─ 今日最低价: {today_low:.2f} 元")
    print(f"   └─ 当前价格: {current_price:.2f} 元")
    print(f"   目前涨跌: {(current_price - today_open)/today_open*100:+.2f}%")
    
    print(f"\n🔮 模型预测今日剩余时间走势:")
    pred_high = pred_df['high'].max()
    pred_low = pred_df['low'].min()
    pred_end = pred_df['close'].iloc[-1]
    pred_prices = pred_df['close'].values
    
    print(f"   ├─ 预测今日收盘: {pred_end:.2f} 元")
    print(f"   ├─ 预测日内最高: {pred_high:.2f} 元")
    print(f"   └─ 预测日内最低: {pred_low:.2f} 元")
    
    daily_change = (pred_end - today_open) / today_open * 100
    print(f"\n📈 预测今日涨跌: {daily_change:+.2f}%")
    
    # 趋势判断
    if daily_change > 1.2:
        trend = "强势上涨 🟢🟢"
        advice = "持仓待涨，可考虑逢低加仓"
        confidence = "高"
    elif daily_change > 0.4:
        trend = "小幅上涨 🟢"
        advice = "继续持有，不建议追高"
        confidence = "中高"
    elif daily_change > -0.4:
        trend = "横盘震荡 🟡"
        advice = "观望为主，可考虑高抛低吸做T"
        confidence = "中"
    elif daily_change > -1.2:
        trend = "小幅调整 🔴"
        advice = "适当减仓，控制仓位"
        confidence = "中高"
    else:
        trend = "明显下跌 🔴🔴"
        advice = "建议减仓或离场观望，注意风险控制"
        confidence = "高"
    
    print(f"\n🎯 趋势判断: {trend}")
    print(f"📊 置信度: {confidence}")
    print(f"💡 AI交易建议: {advice}")
    
    # 详细预测时间点
    print(f"\n⏰ 关键时间点价格预测:")
    
    # 计算预测的时间进度
    n = len(pred_prices)
    quarter_idx = n // 4
    mid_idx = n // 2
    three_quarter_idx = 3 * n // 4
    
    print(f"   ├─ 早盘后段（11:30左右）: {pred_prices[quarter_idx]:.2f} 元")
    print(f"   ├─ 午盘中段（13:45左右）: {pred_prices[mid_idx]:.2f} 元")
    print(f"   └─ 尾盘时段（14:40左右）: {pred_end:.2f} 元")
    
    # 日内波动分析
    price_range = pred_high - pred_low
    print(f"\n📉 日内波动分析:")
    print(f"   ├─ 预测波动区间: {pred_low:.2f} ~ {pred_high:.2f} 元")
    print(f"   └─ 波动幅度: ±{price_range/2:.2f} 元 ({price_range/current_price*100/2:+.1f}%)")
    
    # 成交量预测
    avg_vol = pred_df['volume'].mean()
    hist_vol = df['volume'].iloc[lookback-48:lookback].mean()
    vol_ratio = avg_vol / hist_vol
    
    print(f"\n📊 成交量预测:")
    print(f"   ├─ 历史日均成交量: {hist_vol/10000:.1f} 万")
    print(f"   ├─ 预测日均成交量: {avg_vol/10000:.1f} 万")
    print(f"   └─ 成交量变化: {(vol_ratio-1)*100:+.1f}%")
    
    if vol_ratio > 1.2:
        print(f"   💡 成交量放大，市场活跃度提升")
    elif vol_ratio < 0.8:
        print(f"   💡 成交量萎缩，市场观望情绪浓")
    else:
        print(f"   💡 成交量维持常态")
    
    # 绘制图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))
    fig.suptitle('000776 广发证券 - 7月15日 AI行情预测', fontsize=18, y=0.98, fontweight='bold')
    
    recent_hist = df.iloc[lookback-45:lookback]
    hist_range = range(len(recent_hist))
    pred_range = range(len(recent_hist)-1, len(recent_hist)+pred_len-1)
    
    ax1.plot(hist_range, recent_hist['close'], label='历史价格', 
             color='blue', linewidth=2, alpha=0.85)
    ax1.plot(pred_range, pred_df['close'], label='AI预测价格', 
             color='red', linewidth=2.5)
    ax1.axvline(x=len(recent_hist)-1, color='green', linestyle='--', 
                alpha=0.8, linewidth=2, label='当前时刻')
    ax1.set_ylabel('价格 (元)', fontsize=13)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('价格走势预测', fontsize=14)
    
    # 标注关键点
    ax1.annotate(f'{current_price:.2f}', 
                xy=(len(recent_hist)-1, current_price),
                xytext=(10, 20), textcoords='offset points',
                fontsize=10, color='green', fontweight='bold')
    ax1.annotate(f'{pred_end:.2f}\n({daily_change:+.2f}%)', 
                xy=(len(recent_hist)+pred_len-2, pred_end),
                xytext=(-60, -30), textcoords='offset points',
                fontsize=10, color='red', fontweight='bold')

    ax2.plot(hist_range, recent_hist['volume'], label='历史成交量', 
             color='blue', linewidth=1.5, alpha=0.85)
    ax2.plot(pred_range, pred_df['volume'], label='预测成交量', 
             color='red', linewidth=2)
    ax2.axvline(x=len(recent_hist)-1, color='green', linestyle='--', alpha=0.8, linewidth=2)
    ax2.set_ylabel('成交量', fontsize=13)
    ax2.set_xlabel('5分钟K线序号（从左到右代表时间推进）', fontsize=13)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('成交量预测', fontsize=14)

    plt.tight_layout()
    plt.savefig('000776_广发证券_今日行情预测.png', dpi=150, bbox_inches='tight')
    print(f"\n🖼️  预测行情图已保存: 000776_广发证券_今日行情预测.png")
    plt.close()
    
    pred_df.to_csv("000776_广发证券_预测结果.csv")
    print(f"💾 预测明细数据已保存: 000776_广发证券_预测结果.csv")
    
    print("\n" + "=" * 80)
    print("🎉 广发证券（000776）今日行情预测完成!")
    print("=" * 80)
    print("\n⚠️ 重要提示:")
    print("   1. 以上预测基于模拟历史数据，仅作Kronos模型功能演示")
    print("   2. 真实股票市场受政策、资金、情绪等多种因素影响")
    print("   3. 本预测不构成任何投资建议，请谨慎决策")
    print("   4. 如需真实预测，请导入实际的5分钟K线数据运行")
    print("=" * 80)


if __name__ == "__main__":
    main()
