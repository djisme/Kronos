#!/usr/bin/env python3
"""
Kronos AI - 热门股票批量预测演示
对A股热门股票进行建模预测
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

# 热门股票池
STOCK_POOL = [
    ('300308', '中际旭创', '光模块'),
    ('300502', '新易盛', '光模块'),
    ('688256', '寒武纪', 'AI芯片'),
    ('300476', '胜宏科技', 'PCB'),
    ('300750', '宁德时代', '动力电池'),
    ('002594', '比亚迪', '新能源汽车'),
    ('688585', '上纬新材', '新材料'),
    ('300059', '东方财富', '证券'),
    ('601318', '中国平安', '保险'),
    ('601398', '工商银行', '银行'),
    ('600036', '招商银行', '银行'),
    ('600519', '贵州茅台', '白酒'),
    ('000858', '五粮液', '白酒'),
    ('002475', '立讯精密', '消费电子'),
]

# 股票基准价格（2025年真实大致范围）
STOCK_PRICES = {
    '300308': 680,
    '300502': 280,
    '688256': 420,
    '300476': 95,
    '300750': 360,
    '002594': 260,
    '688585': 135,
    '300059': 18,
    '601318': 48,
    '601398': 7.5,
    '600036': 35,
    '600519': 1750,
    '000858': 170,
    '002475': 48,
}


def generate_stock_data(code, days=8):
    """生成单只股票的模拟5分钟K线数据"""
    base_price = STOCK_PRICES.get(code, 50)
    seed = hash(code) % 10000
    np.random.seed(seed)
    
    bars_per_day = 48
    total_bars = days * bars_per_day
    
    dates = []
    current = datetime(2026, 7, 16, 9, 30) - timedelta(days=days+2)
    
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
    trend_strength = np.random.normal(0.00002, 0.00008)
    trend = trend_strength * t
    noise = np.random.normal(0, 0.009, len(dates))
    cycles = 0.009 * np.sin(t / 28) + 0.005 * np.sin(t / 9)
    
    returns = trend + noise + cycles * 0.06
    price_factors = 1 + np.cumsum(returns) / 26
    
    close_prices = base_price * price_factors
    open_prices = close_prices.copy()
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.006, len(dates))))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.006, len(dates))))
    
    volume_base = 40000 + np.random.randint(0, 80000)
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
        T=0.75,
        top_p=0.9,
        sample_count=4,
        verbose=False
    )
    
    current_price = df['close'].iloc[lookback-1]
    pred_end = pred_df['close'].iloc[-1]
    pred_high = pred_df['high'].max()
    pred_low = pred_df['low'].min()
    
    expected_return = (pred_end - current_price) / current_price * 100
    max_possible_gain = (pred_high - current_price) / current_price * 100
    
    # 计算上涨概率评分
    volatility = (pred_high - pred_low) / current_price * 100
    if expected_return > 2.0:
        probability = min(92, 72 + expected_return * 4.5 - volatility * 1.8)
    elif expected_return > 0.5:
        probability = min(82, 52 + expected_return * 7)
    elif expected_return > 0:
        probability = min(65, 45 + expected_return * 10)
    else:
        probability = max(8, 42 + expected_return * 4)
    
    probability = np.clip(probability, 5, 95)
    
    # 风险评级
    if volatility > 4.0:
        risk = '高'
    elif volatility > 2.2:
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


def plot_top_stocks(results, top_n=5):
    """绘制上涨概率最高的股票预测图"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Kronos AI - 7月16日热门股票预测对比', fontsize=17, y=0.995, fontweight='bold')
    axes = axes.flatten()
    
    colors = ['#FF3333', '#FF8C00', '#FFD700', '#4D94FF', '#66CDAA']
    
    for idx, (stock, color) in enumerate(zip(top_n, colors)):
        if idx >= 5:
            break
            
        ax = axes[idx]
        result = [r for r in results if r['code'] == stock['code']][0]
        pred_df = result['pred_df']
        
        pred_prices = pred_df['close'].values
        ax.plot(range(len(pred_prices)), pred_prices, color=color, linewidth=2.5, label='预测价格')
        ax.axhline(y=stock['current_price'], color='gray', linestyle='--', alpha=0.7, label='当前价格')
        
        ax.set_title(f"{stock['name']}({stock['code']}) | {stock['sector']}\n预期涨幅: +{stock['expected_return']:.2f}%", fontsize=11, fontweight='bold')
        ax.set_ylabel('价格 (元)', fontsize=9)
        ax.set_xlabel('5分钟K线序号', fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
        
        ax.annotate(f"上涨概率: {stock['probability']:.1f}%", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=9, fontweight='bold')
        ax.annotate(f"风险: {stock['risk']}", xy=(0.05, 0.88), xycoords='axes fraction', fontsize=8)
    
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig('热门股票明日预测对比.png', dpi=150, bbox_inches='tight')
    print("📊 预测对比图已保存: 热门股票明日预测对比.png")
    plt.close()


def main():
    print("=" * 95)
    print("🔮 Kronos AI - A股热门股票 7月16日 批量预测分析")
    print(f"📅 预测日期：2026年7月16日")
    print("⚠️  声明：基于模拟数据演示模型功能，不构成任何投资建议")
    print("=" * 95)

    # 1. 加载模型
    print("\n[1/4] 正在加载AI预测模型...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    device = torch.device("cpu")
    predictor = KronosPredictor(model, tokenizer, device=str(device), max_context=384)
    print("✅ AI模型加载完成")

    # 2. 对股票池进行批量预测
    print(f"\n[2/4] 开始对 {len(STOCK_POOL)} 只热门股票进行明日走势预测...")
    print("-" * 95)
    
    results = []
    
    for idx, (code, name, sector) in enumerate(STOCK_POOL):
        print(f"  [{idx+1:2d}/{len(STOCK_POOL)}] 分析中: {name:8s}({code})...", end='', flush=True)
        
        df = generate_stock_data(code, days=8)
        result = predict_single_stock(code, name, sector, predictor, df, pred_len=24)
        results.append(result)
        
        trend_icon = "📈" if result['expected_return'] > 0 else "📉"
        print(f" 完成 | {trend_icon} {result['expected_return']:+6.2f}% | 概率: {result['probability']:5.1f}%")

    print("-" * 95)
    print("✅ 全部股票预测完成!")

    # 3. 筛选明日上涨概率最高的股票
    print("\n[3/4] 正在筛选明日上涨概率最高的股票...")
    
    results_sorted = sorted(results, key=lambda x: (x['probability'], x['expected_return']), reverse=True)
    
    print("\n" + "=" * 95)
    print("🏆 7月16日 热门股票上涨概率排行榜 - TOP10")
    print("=" * 95)
    
    for idx, stock in enumerate(results_sorted[:10]):
        medal = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣', '6️⃣', '7️⃣', '8️⃣', '9️⃣', '🔟'][idx]
        
        print(f"\n{medal} 第{idx+1}名: {stock['name']}({stock['code']}) - {stock['sector']}")
        print(f"   ┌───────────────────────────────────────────────────────────────────────────┐")
        print(f"   │  当前价格: {stock['current_price']:8.2f} 元 | 预期涨幅: {stock['expected_return']:+6.2f}%")
        print(f"   │  上涨概率: {stock['probability']:7.1f}%        | 波动幅度: ±{stock['volatility']/2:5.2f}%")
        print(f"   │  预测收盘: {stock['predicted_close']:8.2f} 元 | 风险等级: {stock['risk']:>6s}")
        print(f"   └───────────────────────────────────────────────────────────────────────────┘")

    # 4. 策略建议
    print("\n" + "=" * 95)
    print("💡 AI策略建议")
    print("=" * 95)
    
    top5 = results_sorted[:5]
    avg_return = np.mean([s['expected_return'] for s in top5])
    avg_prob = np.mean([s['probability'] for s in top5])
    sectors = set([s['sector'] for s in top5])
    
    print(f"\n📊 组合统计（Top5）:")
    print(f"   平均预期涨幅: {avg_return:+.2f}%")
    print(f"   平均上涨概率: {avg_prob:.1f}%")
    print(f"   涉及板块: {', '.join(sectors)}")
    
    print(f"\n🎯 配置建议:")
    print(f"   1. 首选标的: {top5[0]['name']}({top5[0]['code']}) - 预期涨幅最高，风险可控")
    print(f"   2. 次选标的: {top5[1]['name']}、{top5[2]['name']} - 作为组合配置")
    print(f"   3. 仓位控制: 单只仓位不超过 20%，总仓位不超过 60%")
    print(f"   4. 止盈止损: 止盈 +5% ~ +8%，止损 -2% ~ -3%")
    print(f"   5. 入场时机: 早盘观望，确认支撑后再考虑入场，不追高开")
    
    print(f"\n⚠️ 风险提示:")
    print(f"   1. 以上预测基于模拟历史数据，仅展示 Kronos 模型功能")
    print(f"   2. 真实市场受政策、消息、资金流向等多重因素影响")
    print(f"   3. AI预测胜率 ≠ 实际投资收益，请务必独立判断")
    print(f"   4. 如需真实预测，请从同花顺/通达信导出5分钟K线数据")
    print(f"   5. 投资有风险，入市需谨慎！请对自己的资金负责")

    # 5. 绘制Top5预测对比图
    print("\n[4/4] 正在生成预测对比图...")
    plot_top_stocks(results, top5)
    
    # 保存完整预测结果
    full_df = pd.DataFrame([{k: v for k, v in s.items() if k != 'pred_df'} for s in results_sorted])
    full_df.to_csv('热门股票明日预测结果.csv', index=False, encoding='utf-8-sig')
    print("💾 完整预测结果已保存: 热门股票明日预测结果.csv")
    
    print("\n" + "=" * 95)
    print("🎉 热门股票批量预测完成!")
    print("=" * 95)


if __name__ == "__main__":
    main()
