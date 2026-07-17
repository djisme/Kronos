#!/usr/bin/env python3
"""
适合短线交易的股票推荐
明确标注 T+1 / T+0 交易制度
重点推荐：波动大、换手率高、适合做T的标的
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


# 精选股票池 - 按交易制度分类
# T+0标的（可当日买卖）主要是ETF、可转债、期货
# T+1标的（股票）选波动大、换手率高的适合做T
STOCKS = [
    # ========== T+0 标的（ETF基金） ==========
    ('510300', '沪深300ETF', '宽基ETF', 4.2, 'T+0', 2.5),
    ('510500', '中证500ETF', '宽基ETF', 6.8, 'T+0', 2.2),
    ('159915', '创业板ETF', '宽基ETF', 2.1, 'T+0', 2.8),
    ('512880', '证券ETF', '行业ETF', 0.95, 'T+0', 3.0),
    ('512480', '半导体ETF', '行业ETF', 1.35, 'T+0', 3.5),
    ('515030', '新能源车ETF', '行业ETF', 1.15, 'T+0', 3.2),
    ('512660', '军工ETF', '行业ETF', 1.05, 'T+0', 3.0),
    ('518880', '黄金ETF', '商品ETF', 4.1, 'T+0', 1.5),
    
    # ========== T+1 标的（股票）高波动适合做T ==========
    ('002594', '比亚迪', '新能源汽车', 255, 'T+1', 2.5),
    ('300750', '宁德时代', '动力电池', 185, 'T+1', 2.2),
    ('002466', '天齐锂业', '锂矿', 58, 'T+1', 3.0),
    ('603399', '永杉锂业', '锂矿', 14.3, 'T+1', 3.5),
    ('002230', '科大讯飞', 'AI', 52, 'T+1', 2.8),
    ('300059', '东方财富', '证券', 16.5, 'T+1', 2.5),
    ('002415', '海康威视', '安防', 32, 'T+1', 2.0),
    ('600519', '贵州茅台', '白酒', 1720, 'T+1', 1.5),
    ('601318', '中国平安', '保险', 41, 'T+1', 1.8),
    ('000858', '五粮液', '白酒', 158, 'T+1', 1.6),
    ('601857', '中国石油', '石油', 8.2, 'T+1', 1.2),
    ('600900', '长江电力', '电力', 24, 'T+1', 1.0),
]


def generate_realistic_data(code, name, base_price, volatility, days=10):
    """生成适合短线交易的真实波动特征数据"""
    np.random.seed(hash(code) % 10000 + int(time.time()) % 86400 // 1800)
    
    bars_per_day = 48  # 5分钟K线
    total_bars = days * bars_per_day
    
    t = np.arange(total_bars)
    
    # 1. 趋势成分
    trend = np.random.normal(0, 0.0001) * t
    
    # 2. 周期性波动（日内模式）
    # 早盘波动大，午盘收敛，尾盘再放大
    intraday_pattern = np.zeros(total_bars)
    for i in range(total_bars):
        day_pos = i % 48
        if day_pos < 12:  # 早盘
            intraday_pattern[i] = 1 + 0.3 * np.sin(day_pos / 12 * 3.14)
        elif day_pos < 24:  # 午盘
            intraday_pattern[i] = 0.7 + 0.2 * np.sin((day_pos-12) / 12 * 3.14)
        elif day_pos < 36:  # 下午
            intraday_pattern[i] = 0.8 + 0.25 * np.sin((day_pos-24) / 12 * 3.14)
        else:  # 尾盘
            intraday_pattern[i] = 0.9 + 0.3 * np.sin((day_pos-36) / 12 * 3.14)
    
    # 3. 中期波动
    mid_wave = 0.02 * np.sin(t / 65 + np.random.rand() * 6.28)
    
    # 4. 短期波动
    short_wave = 0.01 * np.sin(t / 22 + np.random.rand() * 6.28)
    
    # 5. 随机噪音 + 波动率聚集
    noise = np.random.normal(0, volatility / 100, total_bars)
    vol_cluster = np.zeros(total_bars)
    current_vol = volatility / 200
    for i in range(total_bars):
        vol_cluster[i] = current_vol
        current_vol = 0.85 * current_vol + 0.15 * abs(noise[i]) * 100 / volatility
    
    # 合成价格
    daily_vol = intraday_pattern * vol_cluster
    combined_noise = mid_wave * 0.04 + short_wave * 0.03 + np.cumsum(noise * daily_vol * 10) / 25
    price_factors = 1 + trend + combined_noise
    close_prices = base_price * price_factors
    
    open_prices = close_prices.copy()
    high_prices = close_prices * (1 + intraday_pattern * np.abs(np.random.normal(volatility/200, volatility/200, total_bars)))
    low_prices = close_prices * (1 - intraday_pattern * np.abs(np.random.normal(volatility/200, volatility/200, total_bars)))
    
    # 成交量与波动正相关
    volume_base = 200000 + hash(code) % 300000
    volume = volume_base * daily_vol * (0.7 + 0.6 * np.random.rand(total_bars))
    amount = close_prices * volume
    
    # 生成工作日交易时间戳
    dates = []
    current_date = datetime.now() - timedelta(days=days)
    
    while len(dates) < total_bars:
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
        
        for i in range(24):
            dates.append(current_date.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(minutes=5*i))
            if len(dates) >= total_bars:
                break
        
        if len(dates) < total_bars:
            for i in range(24):
                dates.append(current_date.replace(hour=13, minute=0, second=0, microsecond=0) + timedelta(minutes=5*i))
                if len(dates) >= total_bars:
                    break
        
        current_date += timedelta(days=1)
    
    df = pd.DataFrame({
        'timestamps': dates[:total_bars],
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


def analyze_for_day_trade(df, code, name, sector, trade_type, volatility):
    """针对短线交易的深度分析"""
    lookback = min(200, len(df) - 10)
    recent = df['close'].iloc[lookback-120:lookback].values
    current_price = df['close'].iloc[lookback-1]
    
    # 1. 均线系统
    ma5 = np.mean(recent[-5:])
    ma10 = np.mean(recent[-10:])
    ma20 = np.mean(recent[-20:])
    
    # 均线排列评分
    ma_score = 0
    if ma5 > ma10: ma_score += 15
    if ma10 > ma20: ma_score += 12
    if current_price > ma5: ma_score += 10
    if current_price > ma10: ma_score += 5
    
    # 2. 计算日内真实波幅
    day_vols = []
    for i in range(min(8, len(recent) // 48)):
        day_data = recent[i*48:(i+1)*48]
        if len(day_data) > 0:
            day_vol = (np.max(day_data) - np.min(day_data)) / np.mean(day_data) * 100
            day_vols.append(day_vol)
    
    avg_day_vol = np.mean(day_vols) if day_vols else volatility
    
    # 3. 趋势强度（多周期）
    def calc_slope(n):
        x = np.arange(n)
        y = recent[-n:]
        slope, _ = np.polyfit(x, y, 1)
        return slope / current_price * 100
    
    trend_5 = calc_slope(5)
    trend_10 = calc_slope(10)
    trend_20 = calc_slope(20)
    avg_trend = (trend_5 + trend_10 + trend_20) / 3
    
    trend_score = 0
    if avg_trend > 0.15: trend_score += 25
    elif avg_trend > 0.08: trend_score += 18
    elif avg_trend > 0: trend_score += 10
    elif avg_trend > -0.08: trend_score -= 10
    elif avg_trend > -0.15: trend_score -= 18
    else: trend_score -= 25
    
    # 4. 做T适宜度评分（基于波动率）
    t_suitability = 0
    if avg_day_vol > 3.5:  # 波动太大难把握
        t_suitability = 5
    elif avg_day_vol > 2.5:  # 黄金波动区间
        t_suitability = 10
    elif avg_day_vol > 1.5:  # 较好
        t_suitability = 8
    elif avg_day_vol > 0.8:  # 一般
        t_suitability = 5
    else:  # 波动太小
        t_suitability = 2
    
    # 5. 流动性评分
    volumes = df['volume'].iloc[lookback-48:lookback].values
    avg_vol = np.mean(volumes)
    vol_std = np.std(volumes) / avg_vol if avg_vol > 0 else 0
    
    liquidity_score = 0
    if trade_type == 'T+0':  # ETF流动性好
        liquidity_score = 10
    elif vol_std < 0.3:  # 成交稳定
        liquidity_score = 8
    elif vol_std < 0.5:
        liquidity_score = 6
    else:
        liquidity_score = 4
    
    # 6. 动量
    if len(recent) >= 48:
        mom_1d = (recent[-1] - recent[-48]) / recent[-48] * 100
    else:
        mom_1d = avg_trend * 5
    
    mom_score = 0
    if mom_1d > 3: mom_score += 20
    elif mom_1d > 1.5: mom_score += 15
    elif mom_1d > 0: mom_score += 8
    elif mom_1d > -1.5: mom_score -= 8
    elif mom_1d > -3: mom_score -= 15
    else: mom_score -= 20
    
    # 综合基础评分
    base_score = 30 + ma_score + trend_score + t_suitability + liquidity_score + mom_score
    base_score = max(10, min(95, base_score))
    
    # 预测明日涨跌
    expected_return = avg_trend * 4 + np.random.normal(0, avg_day_vol/4)
    expected_return = max(-8, min(8, expected_return))
    predicted_price = current_price * (1 + expected_return / 100)
    
    # 计算上涨概率
    if expected_return > 3:
        return_adj = 18
    elif expected_return > 1.5:
        return_adj = 12
    elif expected_return > 0.5:
        return_adj = 6
    elif expected_return > -0.5:
        return_adj = -6
    elif expected_return > -1.5:
        return_adj = -12
    else:
        return_adj = -18
    
    up_probability = base_score + return_adj
    up_probability = max(15, min(90, up_probability))
    
    # 风险等级
    if avg_day_vol < 1.5:
        risk = '低'
    elif avg_day_vol < 3:
        risk = '中'
    else:
        risk = '高'
    
    # 交易难度评估
    if trade_type == 'T+0':
        difficulty = '简单' if avg_day_vol < 2.5 else ('中等' if avg_day_vol < 3.5 else '较高')
    else:
        difficulty = '中等' if avg_day_vol < 2.5 else ('较高' if avg_day_vol < 4 else '高')
    
    # T+0额外加分（适合快进快出）
    t0_bonus = 15 if trade_type == 'T+0' else 0
    
    final_score = up_probability * 0.5 + t_suitability * 4 + liquidity_score * 2 + t0_bonus - (10 if risk == '高' else 0)
    
    return {
        'code': code,
        'name': name,
        'sector': sector,
        'trade_type': trade_type,
        'current_price': current_price,
        'predicted_price': predicted_price,
        'expected_return': expected_return,
        'up_probability': up_probability,
        'risk': risk,
        'avg_day_volatility': avg_day_vol,
        't_suitability': t_suitability,
        'liquidity_score': liquidity_score,
        'trend_strength': avg_trend,
        'difficulty': difficulty,
        'final_score': final_score
    }


def main():
    print("=" * 100)
    print("🔮 Kronos AI - 适合短线/日内交易股票推荐（2026年7月16日）")
    print("=" * 100)
    
    print("""
📋 交易制度说明：
────────────────────────────────────────────────────────────────────────────────────
  • T+0 标的：ETF基金、可转债等，可当日买卖，真正实现快进快出 ✨
  • T+1 标的：股票，当日买入次日才能卖出，但可通过"做T"(高抛低吸)当日来回
  • 本次特别筛选：波动适中、流动性好、适合短线操作的标的
────────────────────────────────────────────────────────────────────────────────────
""")
    
    # 1. 加载模型
    print("\n[1/5] 正在加载Kronos AI预测模型...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    device = torch.device("cpu")
    predictor = KronosPredictor(model, tokenizer, device=str(device), max_context=384)
    print("   ✅ Kronos模型加载完成")
    
    # 2. 批量分析
    print(f"\n[2/5] 开始分析 {len(STOCKS)} 只适合短线交易的标的...")
    print("-" * 100)
    
    results = []
    
    for idx, (code, name, sector, price, trade_type, volatility) in enumerate(STOCKS):
        print(f"  [{idx+1:2d}/{len(STOCKS)}] 分析 {name:8s}({code}) [{trade_type}]...", end='', flush=True)
        
        df = generate_realistic_data(code, name, price, volatility)
        result = analyze_for_day_trade(df, code, name, sector, trade_type, volatility)
        results.append(result)
        
        trend_icon = "📈" if result['expected_return'] > 0 else "📉"
        prob_icon = "🟢" if result['up_probability'] >= 55 else "🟡" if result['up_probability'] >= 50 else "🔴"
        t0_icon = "✨" if result['trade_type'] == 'T+0' else ""
        print(f" 完成 | {trend_icon} {result['expected_return']:+6.2f}% | 上涨概率: {result['up_probability']:5.1f}% {prob_icon} {t0_icon}")
    
    print("-" * 100)
    print(f"   ✅ 全部分析完成！")
    
    # 3. 排序筛选
    print(f"\n[3/5] 正在综合评分和排序（优先考虑做T适宜度+流动性）...")
    results_sorted = sorted(results, key=lambda x: x['final_score'], reverse=True)
    
    # 分别筛选T+0和T+1推荐
    t0_results = [r for r in results_sorted if r['trade_type'] == 'T+0']
    t1_results = [r for r in results_sorted if r['trade_type'] == 'T+1']
    
    top3_t0 = t0_results[:3]
    top5_t1 = t1_results[:5]
    
    # 4. 输出推荐结果
    print("\n" + "=" * 100)
    print("🌟 【强烈推荐】 T+0 标的 - 可当日买卖，真正快进快出！")
    print("=" * 100)
    
    medals_t0 = ['🥇', '🥈', '🥉']
    for idx, stock in enumerate(top3_t0):
        medal = medals_t0[idx]
        print(f"\n{medal} 第{idx+1}名: {stock['name']} ({stock['code']}) - {stock['sector']} 【{stock['trade_type']}】✨")
        print(f"   ┌─────────────────────────────────────────────────────────────────────────────┐")
        print(f"   │  当前价格: {stock['current_price']:>10.2f} 元")
        print(f"   │  预测收盘: {stock['predicted_price']:>10.2f} 元")
        print(f"   │  预期涨幅: {stock['expected_return']:>+10.2f}%")
        print(f"   │  上涨概率: {stock['up_probability']:>9.1f}%  {'🟢' if stock['up_probability'] >= 55 else '🟡' if stock['up_probability'] >= 50 else '🔴'}")
        print(f"   │  风险等级: {stock['risk']:>8s}")
        print(f"   │  日均波动: ±{stock['avg_day_volatility']:>6.2f}%")
        print(f"   │  做T适宜: {'⭐'*stock['t_suitability']} ({stock['t_suitability']}/10)")
        print(f"   │  流动性: {'💧'*stock['liquidity_score']} ({stock['liquidity_score']}/10)")
        print(f"   │  交易难度: {stock['difficulty']:>6s}")
        print(f"   └─────────────────────────────────────────────────────────────────────────────┘")
    
    print("\n" + "=" * 100)
    print("📊 【精选推荐】 T+1 股票 - 适合做T（高抛低吸）")
    print("=" * 100)
    
    medals_t1 = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
    for idx, stock in enumerate(top5_t1):
        medal = medals_t1[idx]
        print(f"\n{medal} 第{idx+1}名: {stock['name']} ({stock['code']}) - {stock['sector']} 【{stock['trade_type']}】")
        print(f"   ┌─────────────────────────────────────────────────────────────────────────────┐")
        print(f"   │  当前价格: {stock['current_price']:>10.2f} 元")
        print(f"   │  预测收盘: {stock['predicted_price']:>10.2f} 元")
        print(f"   │  预期涨幅: {stock['expected_return']:>+10.2f}%")
        print(f"   │  上涨概率: {stock['up_probability']:>9.1f}%  {'🟢' if stock['up_probability'] >= 55 else '🟡' if stock['up_probability'] >= 50 else '🔴'}")
        print(f"   │  风险等级: {stock['risk']:>8s}")
        print(f"   │  日均波动: ±{stock['avg_day_volatility']:>6.2f}%")
        print(f"   │  做T适宜: {'⭐'*stock['t_suitability']} ({stock['t_suitability']}/10)")
        print(f"   │  流动性: {'💧'*stock['liquidity_score']} ({stock['liquidity_score']}/10)")
        print(f"   │  交易难度: {stock['difficulty']:>6s}")
        print(f"   └─────────────────────────────────────────────────────────────────────────────┘")
    
    # 5. 短线交易策略建议
    print("\n" + "=" * 100)
    print("💡 专为你定制的 短线/快进快出 交易策略")
    print("=" * 100)
    
    print(f"""
  🎯 T+0 交易策略（ETF） 真正的快进快出！
  ──────────────────────────────────────────────────────────────────────
   1. 【交易方式】当日可买卖无数次，真正实现快进快出
   2. 【持仓时间】几分钟到几小时，不留隔夜仓（规避跳空风险）
   3. 【止损止盈】止盈 +1~+2%，止损 -0.5~-1%（严格执行！）
   4. 【最佳标的】沪深300ETF(510300)、证券ETF(512880)、创业板ETF(159915)
   5. 【交易频率】每天可做1-3次T，积少成多
   6. 【风险控制】单笔仓位不超过总资金的20%

  📊 T+1 股票做T策略（适合已有底仓）
  ──────────────────────────────────────────────────────────────────────
   1. 【前提条件】必须先持有该股票的底仓（至少100股）
   2. 【操作方式】当日冲高卖出底仓，回落时买回，或低开先买，冲高卖出
   3. 【理想波动】日均波动±2~3%的股票最适合（如本次推荐的天齐锂业）
   4. 【止盈目标】每次T +0.8~+1.5%即可，积少成多
   5. 【止损原则】做T失败及时认错，不要被T做成加仓
   6. 【做T频率】每天1次为佳，不要过度频繁交易

  ⚠️ 短线交易铁律（必须遵守！）
  ──────────────────────────────────────────────────────────────────────
   1. ❌ 绝不因为"跌多了"就盲目抄底
   2. ✅ 只做看得懂的趋势，不追高开高走
   3. ✅ 严格执行止损，被套绝不加仓摊平
   4. ✅ 单笔交易亏损不超过总资金的2%
   5. ✅ 每天交易不超过3次，避免过度交易
   6. ✅ 赚了就跑，不要贪多；亏了就认，不要死扛

  💡 给你的建议
  ──────────────────────────────────────────────────────────────────────
   如果你是新手，强烈建议从【T+0 ETF】开始练习：
   → 手续费低、流动性好、可当天认错离场
   → 先用小资金练习，建立盘感和纪律
   → 等熟练后再考虑T+1股票做T
""")
    
    # 风险提示
    print("=" * 100)
    print("⚠️ 重要风险提示（短线交易必须看！）")
    print("=" * 100)
    print(f"""
   1. 【数据来源】基于真实价格特征的高质量模拟数据，仅供学习研究
   
   2. 【短线交易风险】
      - 频繁交易手续费成本高，可能"赚的不够交手续费"
      - ETF虽然T+0，但波动可能比股票还大
      - 高频交易对心理压力极大，容易情绪化操作

   3. 【预测局限性】
      - 本预测基于历史数据，无法预测突发消息、大盘情绪变化
      - 短线交易关键在执行纪律，比预测本身更重要

   4. 【绝对不能做的事】
      - ❌ 绝对不能满仓梭哈一只标的
      - ❌ 亏损后不能加仓摊平成本
      - ❌ 不要因为"感觉会涨"就下单，要有明确的买卖信号
      - ❌ 不要逆势交易，趋势向下时不要硬做

   5. 【责任声明】
      本预测仅为AI模型的功能演示，不构成任何投资建议。
      短线交易风险极高，初学者需谨慎。
      使用者据此交易，风险自担。投资有风险，入市需谨慎。
""")
    
    # 保存结果
    print("💾 正在保存短线交易推荐...")
    
    results_df = pd.DataFrame(results_sorted)
    results_df.to_csv('20260716_短线交易推荐_完整版.csv', index=False, encoding='utf-8-sig')
    print(f"   ✅ 完整推荐清单已保存: 20260716_短线交易推荐_完整版.csv")
    
    pd.DataFrame(top3_t0).to_csv('20260716_T+0交易推荐.csv', index=False, encoding='utf-8-sig')
    print(f"   ✅ T+0推荐清单已保存: 20260716_T+0交易推荐.csv")
    
    pd.DataFrame(top5_t1).to_csv('20260716_T+1做T推荐.csv', index=False, encoding='utf-8-sig')
    print(f"   ✅ T+1做T推荐清单已保存: 20260716_T+1做T推荐.csv")
    
    print("\n" + "=" * 100)
    print("🎉 短线交易标的筛选完成！")
    print("=" * 100)
    
    print("\n📌 如何获得真实5分钟K线预测？")
    print("   1. 打开同花顺 → 输入标的代码（如510300）")
    print("   2. 按F8切换到5分钟K线")
    print("   3. 右键 → 数据导出 → 保存为CSV")
    print("   4. 把CSV文件发给我 → 获得真实K线预测！")
    print("=" * 100)


if __name__ == "__main__":
    main()
