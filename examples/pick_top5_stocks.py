#!/usr/bin/env python3
"""
Kronos AI - 明日牛股批量预测演示
对多只股票进行建模预测，筛选上涨概率最高的标的
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

# 模拟的股票池（A股各行业龙头）
STOCK_POOL = [
    ('000001', '平安银行', '银行'),
    ('000858', '五粮液', '白酒'),
    ('300750', '宁德时代', '新能源'),
    ('600519', '贵州茅台', '白酒'),
    ('601318', '中国平安', '保险'),
    ('002594', '比亚迪', '新能源汽车'),
    ('600036', '招商银行', '银行'),
    ('002475', '立讯精密', '消费电子'),
    ('300059', '东方财富', '证券'),
    ('600900', '长江电力', '电力'),
]


def generate_stock_data(base_price, days=7, seed=42):
    """生成单只股票的模拟5分钟K线数据"""
    np.random.seed(seed)
    bars_per_day = 48
    total_bars = days * bars_per_day
    
    dates = []
    current = datetime(2026, 7, 15, 15, 0) - timedelta(days=days+2)
    
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
    
    # 生成带趋势的价格
    t = np.arange(len(dates))
    trend = np.random.normal(0.00003, 0.00008) * t  # 随机趋势
    noise = np.random.normal(0, 0.008, len(dates))
    cycles = 0.008 * np.sin(t / 25) + 0.004 * np.sin(t / 8)
    
    returns = trend + noise + cycles * 0.05
    price_factors = 1 + np.cumsum(returns) / 25
    
    close_prices = base_price * price_factors
    open_prices = close_prices.copy()
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates))))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates))))
    
    volume_base = 50000 + np.random.randint(0, 100000)
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
    
    return df


def generate_future_timestamps(n):
    """生成未来n个交易时间戳（7月16日）"""
    timestamps = []
    current = datetime(2026, 7, 16, 9, 30, 0)
    
    while len(timestamps) < n:
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
    
    return pd.Series(timestamps)


def predict_single_stock(code, name, sector, predictor, df, pred_len=24):
    """预测单只股票的明日走势"""
    lookback = min(256, len(df) - 10)
    
    x_df = df.iloc[:lookback,][['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df.iloc[:lookback]['timestamps']
    y_timestamp = generate_future_timestamps(pred_len)
    
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=0.7,
        top_p=0.9,
        sample_count=5,
        verbose=False
    )
    
    current_price = df['close'].iloc[lookback-1]
    pred_end = pred_df['close'].iloc[-1]
    pred_high = pred_df['high'].max()
    pred_low = pred_df['low'].min()
    
    expected_return = (pred_end - current_price) / current_price * 100
    max_possible_gain = (pred_high - current_price) / current_price * 100
    
    # 计算上涨概率评分（综合收益率、波动率、趋势）
    volatility = (pred_high - pred_low) / current_price * 100
    trend_score = np.mean(np.diff(pred_df['close'].values))
    if expected_return > 1.5:
        probability = min(95, 70 + expected_return * 5 - volatility * 2)
    elif expected_return > 0:
        probability = min(85, 50 + expected_return * 8)
    else:
        probability = max(10, 40 + expected_return * 5)
    
    probability = np.clip(probability, 5, 95)
    
    # 风险评级
    if volatility > 3.5:
        risk = '高'
    elif volatility > 2:
        risk = '中'
    else:
        risk = '低'
    
    return {
        'code': code,
        'name': name,
        'sector': sector,
        'current_price': current_price,
        'predicted_close': pred_end,
        'predicted_high': pred_high,
        'predicted_low': pred_low,
        'expected_return': expected_return,
        'max_gain': max_possible_gain,
        'probability': probability,
        'volatility': volatility,
        'risk': risk,
        'pred_df': pred_df
    }


def plot_top5_stocks(results, top5):
    """绘制前5名股票的预测走势图"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Kronos AI - 明日高概率上涨股票预测 (7月16日)', fontsize=16, y=0.995, fontweight='bold')
    axes = axes.flatten()
    
    colors = ['#FF4D4D', '#FF8C00', '#FFD700', '#4DA6FF', '#66CDAA']
    
    for idx, (stock, color) in enumerate(zip(top5, colors)):
        if idx >= 5:
            break
            
        ax = axes[idx]
        result = [r for r in results if r['code'] == stock['code']][0]
        pred_df = result['pred_df']
        
        # 绘制预测价格
        pred_prices = pred_df['close'].values
        ax.plot(range(len(pred_prices)), pred_prices, color=color, linewidth=2.5, label='预测价格')
        
        # 填充涨跌区域
        ax.axhline(y=stock['current_price'], color='gray', linestyle='--', alpha=0.7, label='当前价格')
        
        ax.set_title(f"{stock['name']}({stock['code']}) | 预测涨幅: +{stock['expected_return']:.2f}%", fontsize=11, fontweight='bold')
        ax.set_ylabel('价格 (元)', fontsize=9)
        ax.set_xlabel('5分钟K线序号', fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
        
        # 添加关键信息标注
        ax.annotate(f"上涨概率: {stock['probability']:.1f}%", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=9, fontweight='bold')
    
    # 隐藏最后一个子图
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig('明日Top5牛股预测.png', dpi=150, bbox_inches='tight')
    print("📊 预测对比图已保存: 明日Top5牛股预测.png")
    plt.close()


def main():
    print("=" * 90)
    print("🔮 Kronos AI - 明日（7月16日）高概率上涨股票批量筛选")
    print(f"📅 预测日期：2026年7月16日")
    print("⚠️  注意：本程序使用模拟数据演示模型功能，不构成任何投资建议")
    print("=" * 90)

    # 1. 加载模型
    print("\n[1/4] 正在加载AI模型...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    device = torch.device("cpu")
    predictor = KronosPredictor(model, tokenizer, device=str(device), max_context=384)
    print("✅ AI模型加载完成")

    # 2. 对股票池进行批量预测
    print(f"\n[2/4] 开始对 {len(STOCK_POOL)} 只股票进行明日走势预测...")
    print("-" * 90)
    
    results = []
    
    for idx, (code, name, sector) in enumerate(STOCK_POOL):
        print(f"  [{idx+1}/{len(STOCK_POOL)}] 正在分析 {name}({code})...", end='', flush=True)
        
        # 生成模拟数据
        base_price = np.random.uniform(15, 180)
        df = generate_stock_data(base_price, seed=hash(code) % 10000)
        
        # 进行预测
        result = predict_single_stock(code, name, sector, predictor, df, pred_len=24)
        results.append(result)
        
        status = "📈 看涨" if result['expected_return'] > 0 else "📉 看跌"
        print(f" 完成 | {status} {result['expected_return']:+.2f}% | 概率: {result['probability']:.1f}%")

    print("-" * 90)
    print("✅ 全部股票预测完成!")

    # 3. 筛选明日上涨概率最高的5只股票
    print("\n[3/4] 正在筛选明日上涨概率最高的5只股票...")
    
    # 按上涨概率和预期收益率综合排序
    results_sorted = sorted(results, key=lambda x: (x['probability'], x['expected_return']), reverse=True)
    
    top5 = results_sorted[:5]
    
    print("\n" + "=" * 90)
    print("🏆 明日（7月16日）上涨概率最高的5只股票 - TOP5")
    print("=" * 90)
    
    for idx, stock in enumerate(top5):
        medal = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][idx]
        
        print(f"\n{medal} 第{idx+1}名: {stock['name']}({stock['code']}) - {stock['sector']}")
        print(f"   ┌─────────────────────────────────────────────────────────────────┐")
        print(f"   │  当前价格: {stock['current_price']:.2f} 元")
        print(f"   │  预测收盘价: {stock['predicted_close']:.2f} 元")
        print(f"   │  预测最高价: {stock['predicted_high']:.2f} 元")
        print(f"   │  预测最低价: {stock['predicted_low']:.2f} 元")
        print(f"   │  预期涨幅: {stock['expected_return']:+.2f}%")
        print(f"   │  最大潜在收益: +{stock['max_gain']:.2f}%")
        print(f"   │  上涨概率: {stock['probability']:.1f}%")
        print(f"   │  波动幅度: ±{stock['volatility']/2:.2f}%")
        print(f"   │  风险等级: {stock['risk']}")
        print(f"   └─────────────────────────────────────────────────────────────────┘")

    # 4. 投资建议总结
    print("\n" + "=" * 90)
    print("💡 AI策略建议")
    print("=" * 90)
    
    avg_return = np.mean([s['expected_return'] for s in top5])
    avg_prob = np.mean([s['probability'] for s in top5])
    sectors = set([s['sector'] for s in top5])
    
    print(f"\n📊 组合统计:")
    print(f"   平均预期涨幅: {avg_return:+.2f}%")
    print(f"   平均上涨概率: {avg_prob:.1f}%")
    print(f"   涉及板块: {', '.join(sectors)}")
    
    print(f"\n🎯 配置建议:")
    print(f"   1. 建议重点关注 Top3 标的，胜率和收益率更有保障")
    print(f"   2. 单只仓位不超过 20%，做好分散配置")
    print(f"   3. 设置止盈止损：止盈+5%，止损-2%")
    print(f"   4. 早盘高开不追，等待回调企稳再考虑入场")
    
    print(f"\n⚠️ 风险提示:")
    print(f"   1. 以上预测基于模拟历史数据，仅作模型功能演示")
    print(f"   2. 真实市场受突发消息、大盘情绪、资金流向等多种因素影响")
    print(f"   3. 预测胜率不代表实际收益，投资有风险，入市需谨慎")
    print(f"   4. 如需真实预测，请导入通达信/同花顺/东方财富的5分钟K线数据")

    # 5. 绘制Top5预测对比图
    print("\n[4/4] 正在生成预测对比图...")
    plot_top5_stocks(results, top5)
    
    # 保存预测结果
    top5_df = pd.DataFrame([{k: v for k, v in s.items() if k != 'pred_df'} for s in top5])
    top5_df.to_csv('明日Top5牛股预测.csv', index=False, encoding='utf-8-sig')
    print("💾 预测结果已保存: 明日Top5牛股预测.csv")
    
    print("\n" + "=" * 90)
    print("🎉 明日牛股批量预测完成!")
    print("=" * 90)


if __name__ == "__main__":
    main()
