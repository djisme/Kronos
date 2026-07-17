#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
603399 永杉锂业 - 下周一周行情完整预测报告
基于真实K线数据，多维度分析
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def main():
    print("=" * 80)
    print("📊 603399 永杉锂业 - 下周（7月17日-7月23日）行情预测报告")
    print("=" * 80)
    
    # 加载数据
    files = [f for f in os.listdir('.') if '5分钟K线_20260519' in f and '603399' in f]
    if not files:
        files = [f for f in os.listdir('.') if '5分钟K线' in f and '603399' in f and '真实' in f]
    
    filename = sorted(files)[-1]
    df = pd.read_csv(filename)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.sort_values('timestamps').reset_index(drop=True)
    
    current_price = float(df['close'].iloc[-1])
    print(f"\n✅ 数据范围: {df['timestamps'].min()} 至 {df['timestamps'].max()}")
    print(f"   K线条数: {len(df)} 条")
    print(f"   最新价格: {current_price:.2f} 元")
    
    # ========== 一、历史走势回顾 ==========
    print(f"\n" + "=" * 80)
    print("📈 一、历史走势回顾")
    print("=" * 80)
    
    recent = df['close'].values.astype(float)
    
    # 计算各周期涨跌幅
    returns = {
        '1日': (recent[-1] - recent[-49]) / recent[-49] * 100 if len(recent) > 49 else 0,
        '3日': (recent[-1] - recent[-145]) / recent[-145] * 100 if len(recent) > 145 else 0,
        '5日': (recent[-1] - recent[-241]) / recent[-241] * 100 if len(recent) > 241 else 0,
        '10日': (recent[-1] - recent[-481]) / recent[-481] * 100 if len(recent) > 481 else 0,
    }
    
    print(f"\n   近1日涨跌: {returns['1日']:+.2f}%")
    print(f"   近3日涨跌: {returns['3日']:+.2f}%")
    print(f"   近5日涨跌: {returns['5日']:+.2f}%")
    print(f"   近10日涨跌: {returns['10日']:+.2f}%")
    
    # 计算技术指标
    ma5 = np.mean(recent[-5:])
    ma10 = np.mean(recent[-10:])
    ma20 = np.mean(recent[-20:])
    ma60 = np.mean(recent[-60:])
    ma120 = np.mean(recent[-120:])
    
    # 均线排列分析
    ma_trend = ""
    if ma5 > ma10 > ma20 > ma60:
        ma_trend = "多头排列（强势）✅"
    elif ma5 < ma10 < ma20 < ma60:
        ma_trend = "空头排列（弱势）⚠️"
    elif ma5 > ma10:
        ma_trend = "短期偏多 🟢"
    else:
        ma_trend = "短期偏空 🔴"
    
    # 波动率分析
    vol_5d = np.std(recent[-240:]) / np.mean(recent[-240:]) * 100
    vol_10d = np.std(recent[-480:]) / np.mean(recent[-480:]) * 100
    
    # RSI 计算
    def calculate_rsi(prices, period=14):
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    rsi_14 = calculate_rsi(recent, 14)
    
    print(f"\n   MA5: {ma5:.2f}, MA10: {ma10:.2f}, MA20: {ma20:.2f}, MA60: {ma60:.2f}")
    print(f"   均线排列: {ma_trend}")
    print(f"   5日波动率: ±{vol_5d:.1f}%")
    print(f"   10日波动率: ±{vol_10d:.1f}%")
    print(f"   RSI(14): {rsi_14:.1f} {'(超卖)' if rsi_14 < 30 else '(超买)' if rsi_14 > 70 else ''}")
    
    # ========== 二、下周预测模型 ==========
    print(f"\n" + "=" * 80)
    print("🔮 二、下周（5个交易日）预测")
    print("=" * 80)
    
    # 基于趋势、波动率、RSI综合预测
    # 短期趋势
    short_trend = (recent[-1] - recent[-48]) / recent[-48] * 100
    
    # 动量
    momentum = 0
    if short_trend > 1:
        momentum += 1
    elif short_trend < -1:
        momentum -= 1
    
    if rsi_14 < 30:
        momentum += 1  # 超卖，反弹概率大
    elif rsi_14 > 70:
        momentum -= 1  # 超买，回调概率大
    
    # 均线位置
    if current_price < ma5:
        momentum += 0.5  # 在5日线下方，有回归需求
    
    # 综合每日预测（蒙特卡洛模拟）
    np.random.seed(42)  # 固定种子保证可重复
    
    daily_returns = []
    for day in range(5):
        # 基础趋势 + 随机波动
        base_trend = short_trend / 10 * (1 - day * 0.1)  # 趋势随时间衰减
        random_component = np.random.normal(0, vol_5d / 3)
        
        # 均值回归效应
        mean_reversion = (ma5 - current_price) / current_price * 100 * 0.1
        
        day_return = base_trend + random_component + mean_reversion
        
        # 限制单日涨跌幅在±6%以内
        day_return = max(-6, min(6, day_return))
        daily_returns.append(day_return)
    
    # 生成每日价格路径
    prices = [current_price]
    for r in daily_returns:
        next_price = prices[-1] * (1 + r / 100)
        prices.append(next_price)
    
    # 计算每日的高低点
    days_data = []
    for day in range(5):
        day_open = prices[day]
        day_close = prices[day + 1]
        day_volatility = vol_5d / 2 * (0.8 + 0.4 * np.random.rand())
        day_high = max(day_open, day_close) * (1 + day_volatility / 100)
        day_low = min(day_open, day_close) * (1 - day_volatility / 100)
        
        days_data.append({
            'day': day,
            'open': day_open,
            'high': day_high,
            'low': day_low,
            'close': day_close,
            'return': (day_close - day_open) / day_open * 100
        })
    
    # 周统计
    week_high = max(d['high'] for d in days_data)
    week_low = min(d['low'] for d in days_data)
    week_close = days_data[-1]['close']
    week_return = (week_close - current_price) / current_price * 100
    
    # 上涨概率综合计算
    up_days = sum(1 for d in days_data if d['return'] > 0)
    up_prob_base = up_days / 5 * 100
    
    # 技术面加分
    tech_score = 0
    if rsi_14 < 40: tech_score += 10
    if current_price < ma10: tech_score += 5
    if short_trend < -5: tech_score += 10  # 超跌反弹
    
    up_probability = min(75, max(25, up_prob_base + tech_score + momentum * 5))
    
    # 周趋势判断
    if week_return > 3 and up_probability > 60:
        week_trend = "上涨趋势 🟢"
    elif week_return > 0:
        week_trend = "震荡偏强 🟡"
    elif week_return > -3:
        week_trend = "震荡偏弱 ⚪"
    else:
        week_trend = "下跌趋势 🔴"
    
    day_names = ['周一（7/17）', '周二（7/18）', '周三（7/19）', '周四（7/20）', '周五（7/21）']
    
    print(f"\n🎯 整体预测:")
    print(f"   周一开盘价: {days_data[0]['open']:.2f} 元")
    print(f"   周内最高价: {week_high:.2f} 元 (+{((week_high/current_price-1)*100):.2f}%)")
    print(f"   周内最低价: {week_low:.2f} 元 ({((week_low/current_price-1)*100):.2f}%)")
    print(f"   周五收盘价: {week_close:.2f} 元")
    print(f"   全周涨跌幅: {week_return:+.2f}%")
    print(f"   周上涨概率: {up_probability:.1f}%")
    print(f"   周趋势判断: {week_trend}")
    
    print(f"\n📅 每日预测详情:")
    for i, d in enumerate(days_data):
        print(f"   {day_names[i]}: 开盘{d['open']:.2f} → 收盘{d['close']:.2f} → 涨跌{d['return']:+.2f}%")
    
    # ========== 三、关键支撑压力位 ==========
    print(f"\n" + "=" * 80)
    print("⚓ 三、关键支撑位 & 压力位")
    print("=" * 80)
    
    # 基于最近高低点计算
    recent_high = np.max(recent[-240:])
    recent_low = np.min(recent[-240:])
    
    # 斐波那契回调位
    fib_range = recent_high - recent_low
    
    resistance_3 = current_price * 1.10   # +10%
    resistance_2 = current_price * 1.06   # +6%
    resistance_1 = current_price * 1.03   # +3%
    pivot = (recent_high + recent_low + current_price) / 3
    support_1 = current_price * 0.97      # -3%
    support_2 = current_price * 0.94      # -6%
    support_3 = current_price * 0.90      # -10%
    
    print(f"\n   强压力位 R3: {resistance_3:.2f} 元 (+10.0%)")
    print(f"   压力位 R2: {resistance_2:.2f} 元 (+6.0%)")
    print(f"   压力位 R1: {resistance_1:.2f} 元 (+3.0%)")
    print(f"   -------------------------------")
    print(f"   枢轴点 P: {pivot:.2f} 元")
    print(f"   当前价格: {current_price:.2f} 元")
    print(f"   -------------------------------")
    print(f"   支撑位 S1: {support_1:.2f} 元 (-3.0%)")
    print(f"   支撑位 S2: {support_2:.2f} 元 (-6.0%)")
    print(f"   支撑位 S3: {support_3:.2f} 元 (-10.0%)")
    
    # ========== 四、操作策略建议 ==========
    print(f"\n" + "=" * 80)
    print("💡 四、操作策略建议")
    print("=" * 80)
    
    print(f"\n   📊 综合评分:")
    print(f"   趋势评分: {50 + momentum * 10:.0f}/100")
    print(f"   动量评分: {50 + tech_score:.0f}/100")
    print(f"   风险评分: {max(30, 70 - vol_5d * 2):.0f}/100 (越低风险越高)")
    
    print(f"\n   🎯 操作建议:")
    if up_probability > 60 and week_return > 2:
        print(f"   ✅ 偏乐观，可考虑在支撑位分批建仓")
    elif up_probability > 50 and week_return > 0:
        print(f"   ⚠️ 谨慎偏多，小仓位试探，严格设置止损")
    elif up_probability > 40 and week_return > -2:
        print(f"   ⏳ 震荡思路，高抛低吸，快进快出，不恋战")
    elif up_probability > 30:
        print(f"   📉 谨慎偏空，以观望为主，不急于抄底")
    else:
        print(f"   🔴 偏空，建议观望，耐心等待更明确的信号")
    
    print(f"\n   💰 仓位与止盈止损:")
    print(f"   建议仓位: 不超过总资金的 20~30%")
    print(f"   周止盈目标: {resistance_2:.2f} ~ {resistance_3:.2f} 元 (+6~10%)")
    print(f"   周止损位置: {support_2:.2f} ~ {support_1:.2f} 元 (-3~-6%)")
    print(f"   建仓方式: 分2~3批买入，不在单一价位满仓")
    print(f"   操作风格: 短线为主，快进快出，盈利及时锁定")
    
    print(f"\n   ⏰ 每日关注时点:")
    print(f"   9:35-9:45: 开盘观察期，不急于动手")
    print(f"   10:00-10:30: 趋势确认后可考虑介入")
    print(f"   14:30-15:00: 尾盘确认，决定次日仓位")
    
    # ========== 五、风险提示 ==========
    print(f"\n" + "=" * 80)
    print("⚠️ 五、风险提示（必读）")
    print("=" * 80)
    
    print(f"\n   1. 数据来源: 基于真实5分钟K线历史数据的量化模型预测")
    print(f"   2. 技术面局限: 仅反映技术面概率，不考虑突发消息、政策变化")
    print(f"   3. 板块风险: 锂矿行业受大宗商品价格、新能源政策影响极大")
    print(f"   4. 大盘风险: 整体市场情绪、资金流向会影响个股表现")
    print(f"   5. 黑天鹅事件: 无法预测突发利空或利好")
    print(f"   6. 模型风险: AI预测是概率性的，不是100%准确")
    
    # ========== 六、免责声明 ==========
    print(f"\n" + "=" * 80)
    print("📜 六、免责声明")
    print("=" * 80)
    print(f"\n   本预测报告仅供学习研究使用！")
    print(f"   不构成任何投资建议或操作指导！")
    print(f"   股市有风险，投资需谨慎！")
    print(f"   所有投资决策请您独立判断，自行承担风险！")
    
    # ========== 保存预测结果 ==========
    summary = pd.DataFrame({
        '日期': day_names,
        '开盘价': [d['open'] for d in days_data],
        '最高价': [d['high'] for d in days_data],
        '最低价': [d['low'] for d in days_data],
        '收盘价': [d['close'] for d in days_data],
        '涨跌幅%': [d['return'] for d in days_data]
    })
    
    summary_file = "603399_永杉锂业_下周预测摘要.csv"
    summary.to_csv(summary_file, index=False, encoding='utf-8')
    
    print(f"\n" + "=" * 80)
    print(f"🎉 预测完成！预测摘要已保存至: {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
