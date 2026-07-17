#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
热门股票批量预测 - 筛选下周上涨概率最高的5只
覆盖多个行业板块
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime


def get_stock_data(code, name):
    """获取单只股票的真实数据"""
    try:
        import akshare as ak
        print(f"   正在获取 {code} {name}...", end=" ")
        
        # 判断是沪市还是深市
        if code.startswith('6') or code.startswith('5'):
            symbol = f'sh{code}'
        else:
            symbol = f'sz{code}'
        
        # 获取5分钟K线
        df = ak.stock_zh_a_minute(symbol=symbol, period="5", adjust="qfq")
        
        if df is not None and len(df) > 100:
            # 重命名列
            df = df.rename(columns={'day': 'timestamps'})
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            df = df.sort_values('timestamps').reset_index(drop=True)
            
            print(f"✅ 成功，{len(df)} 条K线")
            return df, True
        else:
            print(f"⚠️ 数据量不足")
            return None, False
            
    except Exception as e:
        print(f"❌ 失败: {str(e)[:30]}")
        return None, False


def analyze_stock(df, code, name, sector):
    """深度分析单只股票，计算预测和上涨概率"""
    
    if df is None or len(df) < 200:
        return None
    
    recent = df['close'].values.astype(float)
    current_price = float(df['close'].iloc[-1])
    
    # ========== 技术指标计算 ==========
    
    # 均线
    ma5 = np.mean(recent[-5:])
    ma10 = np.mean(recent[-10:])
    ma20 = np.mean(recent[-20:])
    ma60 = np.mean(recent[-60:]) if len(recent) > 60 else ma20
    
    # 均线排列评分
    ma_score = sum([ma5 > ma10, ma10 > ma20, ma20 > ma60])
    ma_trend = "多头" if ma_score >= 2 else "空头" if ma_score == 0 else "震荡"
    
    # 涨跌幅
    ret_1d = (recent[-1] - recent[-49]) / recent[-49] * 100 if len(recent) > 49 else 0
    ret_3d = (recent[-1] - recent[-145]) / recent[-145] * 100 if len(recent) > 145 else 0
    ret_5d = (recent[-1] - recent[-241]) / recent[-241] * 100 if len(recent) > 241 else 0
    
    # 波动率
    vol_5d = np.std(recent[-240:]) / np.mean(recent[-240:]) * 100 if len(recent) > 240 else 2
    
    # RSI
    def calc_rsi(prices, period=14):
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:]) if len(gains) >= period else np.mean(gains)
        avg_loss = np.mean(losses[-period:]) if len(losses) >= period else np.mean(losses)
        if avg_loss == 0:
            return 50
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    rsi = calc_rsi(recent, 14)
    
    # ========== 下周预测模型 ==========
    
    np.random.seed(hash(code) % 10000)  # 固定种子
    
    # 基础趋势评分
    trend_score = 0
    if ret_1d > 0: trend_score += 1
    if ret_3d > 0: trend_score += 1
    if ret_5d > 0: trend_score += 1
    if ma5 > ma10: trend_score += 1
    if ma10 > ma20: trend_score += 1
    
    # 超卖反弹加分
    if rsi < 35: trend_score += 2
    elif rsi < 45: trend_score += 1
    
    # 超买回调减分
    if rsi > 65: trend_score -= 2
    elif rsi > 55: trend_score -= 1
    
    # 蒙特卡洛模拟下周走势
    n_simulations = 100
    weekly_returns = []
    
    for _ in range(n_simulations):
        price = current_price
        for day in range(5):
            # 趋势效应 + 随机波动
            trend_effect = trend_score * 0.1 * (0.5 + np.random.rand())
            random_noise = np.random.normal(0, vol_5d / 2.5)
            day_return = trend_effect + random_noise
            day_return = max(-8, min(8, day_return))  # 单日限幅
            price = price * (1 + day_return / 100)
        weekly_returns.append((price - current_price) / current_price * 100)
    
    expected_return = np.mean(weekly_returns)
    up_probability = sum(1 for r in weekly_returns if r > 0) / n_simulations * 100
    
    # 风险等级
    if vol_5d < 3:
        risk = "低"
    elif vol_5d < 6:
        risk = "中"
    else:
        risk = "高"
    
    # 趋势判断
    if expected_return > 4 and up_probability > 65:
        trend = "强势上涨 🟢🟢"
    elif expected_return > 2 and up_probability > 55:
        trend = "上涨趋势 🟢"
    elif expected_return > -1:
        trend = "震荡整理 ⚪"
    elif expected_return > -4:
        trend = "弱势调整 🟡"
    else:
        trend = "下跌趋势 🔴"
    
    # 综合得分（用于排名）
    score = up_probability * 0.6 + max(0, expected_return) * 5 + (10 - min(10, vol_5d)) * 2
    
    return {
        'code': code,
        'name': name,
        'sector': sector,
        'current_price': current_price,
        'return_1d': ret_1d,
        'return_5d': ret_5d,
        'ma5': ma5,
        'ma10': ma10,
        'ma_trend': ma_trend,
        'rsi': rsi,
        'volatility': vol_5d,
        'risk': risk,
        'expected_return': expected_return,
        'up_probability': up_probability,
        'trend': trend,
        'score': score
    }


def main():
    print("=" * 80)
    print("📊 A股热门股票批量预测 - 筛选下周上涨概率Top5")
    print("=" * 80)
    
    # 股票池：覆盖各行业龙头
    stock_pool = [
        # 白酒/消费
        ('600519', '贵州茅台', '白酒/消费'),
        ('000858', '五粮液', '白酒/消费'),
        ('000333', '美的集团', '家电/消费'),
        
        # 新能源
        ('002594', '比亚迪', '新能源汽车'),
        ('300750', '宁德时代', '动力电池'),
        ('603399', '永杉锂业', '锂矿/新能源'),
        ('002460', '赣锋锂业', '锂矿/新能源'),
        
        # AI/TMT
        ('002230', '科大讯飞', 'AI/人工智能'),
        ('002415', '海康威视', 'AI/安防'),
        ('300136', '信维通信', '消费电子/TMT'),
        
        # 金融
        ('601318', '中国平安', '保险/金融'),
        ('600036', '招商银行', '银行/金融'),
        ('300059', '东方财富', '证券/金融'),
        
        # 医药
        ('600276', '恒瑞医药', '创新药/医药'),
        ('300760', '迈瑞医疗', '医疗器械/医药'),
        
        # 其他核心资产
        ('600900', '长江电力', '公用事业/电力'),
        ('601728', '中国电信', '通信运营'),
        ('601012', '隆基绿能', '光伏/新能源'),
    ]
    
    print(f"\n📋 股票池: {len(stock_pool)} 只，覆盖 8 大行业")
    
    results = []
    success_count = 0
    
    print(f"\n🔍 开始获取数据并分析...\n")
    
    for code, name, sector in stock_pool:
        df, success = get_stock_data(code, name)
        if success and df is not None:
            analysis = analyze_stock(df, code, name, sector)
            if analysis is not None:
                results.append(analysis)
                success_count += 1
    
    print(f"\n✅ 完成分析！成功获取 {success_count}/{len(stock_pool)} 只股票数据")
    
    # 按综合得分排序
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # ========== 输出结果 ==========
    print(f"\n" + "=" * 80)
    print("🏆 下周上涨概率排行榜 - TOP 10")
    print("=" * 80)
    
    print(f"\n{'排名':<4}{'代码':<8}{'名称':<10}{'行业':<15}{'当前价':<8}{'5日涨跌':<10}{'RSI':<8}{'预期涨幅':<10}{'上涨概率':<10}{'趋势':<12}")
    print("-" * 100)
    
    for i, r in enumerate(results[:10], 1):
        print(f"{i:<4}{r['code']:<8}{r['name']:<10}{r['sector']:<15}{r['current_price']:<8.2f}{r['return_5d']:>+8.2f}%  {r['rsi']:<7.1f}{r['expected_return']:>+8.2f}%  {r['up_probability']:>8.1f}%  {r['trend']}")
    
    # ========== TOP 5 深度分析 ==========
    print(f"\n" + "=" * 80)
    print("🎯 TOP 5 深度分析 & 行业趋势")
    print("=" * 80)
    
    top5 = results[:5]
    
    # 按行业分组统计
    sector_stats = {}
    for r in results:
        s = r['sector'].split('/')[0]
        if s not in sector_stats:
            sector_stats[s] = {'count': 0, 'avg_prob': 0, 'stocks': []}
        sector_stats[s]['count'] += 1
        sector_stats[s]['avg_prob'] += r['up_probability']
        sector_stats[s]['stocks'].append(r)
    
    for s in sector_stats:
        sector_stats[s]['avg_prob'] /= sector_stats[s]['count']
    
    # 输出 TOP5 详情
    for i, r in enumerate(top5, 1):
        print(f"\n{'=' * 80}")
        print(f"🥇 No.{i}: {r['name']} ({r['code']})")
        print(f"{'=' * 80}")
        print(f"   所属行业: {r['sector']}")
        print(f"   当前价格: {r['current_price']:.2f} 元")
        print(f"   5日涨跌幅: {r['return_5d']:+.2f}%")
        print(f"   均线趋势: {r['ma_trend']}")
        print(f"   RSI指标: {r['rsi']:.1f} {'(超卖)' if r['rsi'] < 35 else '(超买)' if r['rsi'] > 65 else ''}")
        print(f"   波动率: ±{r['volatility']:.1f}%")
        print(f"   风险等级: {r['risk']}")
        print(f"   预期周涨幅: {r['expected_return']:+.2f}%")
        print(f"   上涨概率: {r['up_probability']:.1f}%")
        print(f"   趋势判断: {r['trend']}")
        print(f"   综合得分: {r['score']:.1f}")
    
    # ========== 行业趋势分析 ==========
    print(f"\n" + "=" * 80)
    print("📊 行业趋势总览")
    print("=" * 80)
    
    sorted_sectors = sorted(sector_stats.items(), key=lambda x: x[1]['avg_prob'], reverse=True)
    
    print(f"\n{'排名':<4}{'行业':<15}{'股票数量':<10}{'平均上涨概率':<12}{'代表个股':<20}")
    print("-" * 70)
    
    for i, (sector, stats) in enumerate(sorted_sectors, 1):
        top_stock = sorted(stats['stocks'], key=lambda x: x['score'], reverse=True)[0]
        print(f"{i:<4}{sector:<15}{stats['count']:<10}{stats['avg_prob']:>8.1f}%    {top_stock['name']}")
    
    # ========== 配置建议 ==========
    print(f"\n" + "=" * 80)
    print("💡 配置建议")
    print("=" * 80)
    
    print(f"\n🎯 推荐关注组合:")
    for i, r in enumerate(top5, 1):
        print(f"   {i}. {r['name']} ({r['code']}) - {r['sector']}")
    
    print(f"\n📌 风险提示:")
    high_risk = [r for r in top5 if r['risk'] == '高']
    if high_risk:
        print(f"   ⚠️ 高波动风险: {', '.join([r['name'] for r in high_risk])}")
    else:
        print(f"   ✅ 组合风险整体可控")
    
    print(f"\n💰 仓位建议:")
    print(f"   单只仓位: 不超过总资金的 15~20%")
    print(f"   组合仓位: 不超过总资金的 50~60%")
    print(f"   建仓方式: 分批建仓，避免一次性满仓")
    
    print(f"\n⚠️ 免责声明:")
    print(f"   本预测基于真实历史K线数据的技术面分析，仅供学习研究！")
    print(f"   不构成任何投资建议，不考虑政策、消息、资金流向等因素！")
    print(f"   股市有风险，投资需谨慎，请独立判断决策！")
    
    print(f"\n" + "=" * 80)
    print("🎉 批量预测完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
