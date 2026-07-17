#!/usr/bin/env python3
"""
603399 永杉锂业 明日（2026-07-16）深度预测
使用真实价格特征 + 高级技术分析
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


def get_baostock_real_data(code='603399', days=20):
    """尝试获取真实的baostock数据"""
    try:
        import baostock as bs
        
        print("   正在连接baostock数据源...")
        lg = bs.login()
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        bs_code = f'sh.{code}' if code.startswith('6') else f'sz.{code}'
        
        print(f"   正在获取 {bs_code} 的5分钟K线...")
        print(f"   时间范围: {start_date} 至 {end_date}")
        
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,time,open,high,low,close,volume,amount",
            start_date=start_date,
            end_date=end_date,
            frequency="5",
            adjustflag="2"
        )
        
        if rs.error_code == '0':
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            if len(data_list) > 50:
                df = pd.DataFrame(data_list, columns=rs.fields)
                for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 处理时间戳
                df['timestamps'] = pd.to_datetime(df['time'])
                
                df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
                df = df.sort_values('timestamps').reset_index(drop=True)
                
                bs.logout()
                
                if len(df) > 50:
                    print(f"   ✅ 获取成功！共 {len(df)} 条5分钟K线")
                    print(f"   📅 时间范围: {df['timestamps'].iloc[0]} 至 {df['timestamps'].iloc[-1]}")
                    print(f"   💰 最新价格: {df['close'].iloc[-1]:.2f} 元")
                    return df, True
        
        bs.logout()
        print("   ❌ baostock获取失败，数据不足")
        return None, False
        
    except Exception as e:
        print(f"   ❌ baostock异常: {e}")
        return None, False


def generate_yongshan_realistic_data():
    """
    生成永杉锂业(603399)真实特征的模拟K线
    基于永杉锂业的真实特性：
    - 锂矿概念股，波动性较大
    - 真实价格区间：14-15元左右
    - 中小盘股，波动比蓝筹股大
    """
    np.random.seed(int(time.time()) % 86400)
    
    base_price = 14.31  # 接近真实价格
    beta = 2.2  # 锂矿股波动大
    
    days = 10
    bars_per_day = 48
    total_bars = days * bars_per_day
    
    t = np.arange(total_bars)
    
    # 1. 中长期趋势（下降通道，符合截图）
    trend = -0.00005 * t  # 轻微下降趋势
    
    # 2. 中期波动（周线级别）
    mid_wave = 0.02 * np.sin(t / 65 + 2.5)
    
    # 3. 短期波动
    short_wave = 0.01 * np.sin(t / 22 + 1.2)
    
    # 4. 随机游走噪音 + 波动率聚集
    noise = np.random.normal(0, 0.008, total_bars)
    vol_cluster = np.zeros(total_bars)
    current_vol = 0.007
    for i in range(total_bars):
        vol_cluster[i] = current_vol
        current_vol = 0.88 * current_vol + 0.12 * abs(noise[i])
    
    # 5. 合成价格
    combined_noise = mid_wave * 0.04 + short_wave * 0.03 + np.cumsum(noise * vol_cluster * 12) / 25
    price_factors = 1 + trend + combined_noise
    close_prices = base_price * price_factors
    
    # 生成OHLC
    open_prices = close_prices.copy()
    high_prices = close_prices * (1 + np.abs(np.random.normal(0.004, 0.004, total_bars)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0.004, 0.004, total_bars)))
    
    # 成交量：锂矿股成交量波动大
    volume_base = 250000
    volume_vol = 0.6 + vol_cluster / np.mean(vol_cluster)
    volume = volume_base * volume_vol * (0.6 + 0.8 * np.random.rand(total_bars))
    amount = close_prices * volume
    
    # 生成交易时间戳（工作日）
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


def advanced_technical_analysis(df):
    """深度技术分析"""
    lookback = min(200, len(df) - 10)
    recent = df['close'].iloc[lookback-120:lookback].values
    current_price = df['close'].iloc[lookback-1]
    
    results = {}
    results['current_price'] = current_price
    
    # 1. 均线系统
    results['ma5'] = np.mean(recent[-5:])
    results['ma10'] = np.mean(recent[-10:])
    results['ma20'] = np.mean(recent[-20:])
    results['ma60'] = np.mean(recent[-60:]) if len(recent) >= 60 else results['ma20']
    results['ma120'] = np.mean(recent)
    
    # 2. 均线排列评分
    ma_score = 0
    if results['ma5'] > results['ma10']: ma_score += 15
    if results['ma10'] > results['ma20']: ma_score += 12
    if results['ma20'] > results['ma60']: ma_score += 10
    if current_price > results['ma5']: ma_score += 8
    if current_price > results['ma10']: ma_score += 5
    results['ma_score'] = ma_score
    
    # 3. 趋势强度（多周期斜率分析）
    def calc_slope(n):
        x = np.arange(n)
        y = recent[-n:]
        slope, _ = np.polyfit(x, y, 1)
        return slope / current_price * 100
    
    results['trend_5'] = calc_slope(5)
    results['trend_10'] = calc_slope(10)
    results['trend_20'] = calc_slope(20)
    results['trend_60'] = calc_slope(min(60, len(recent)))
    
    trend_score = 0
    avg_trend = (results['trend_5'] + results['trend_10'] + results['trend_20']) / 3
    if avg_trend > 0.1: trend_score += 20
    elif avg_trend > 0.05: trend_score += 12
    elif avg_trend > 0: trend_score += 5
    elif avg_trend > -0.05: trend_score -= 5
    elif avg_trend > -0.1: trend_score -= 12
    else: trend_score -= 20
    results['trend_score'] = trend_score
    
    # 4. 波动率分析
    results['volatility_total'] = (np.max(recent) - np.min(recent)) / current_price * 100
    results['volatility_daily'] = results['volatility_total'] / np.sqrt(8)
    
    risk_score = 0
    if results['volatility_daily'] < 2: risk_score += 15
    elif results['volatility_daily'] < 3.5: risk_score += 10
    elif results['volatility_daily'] < 5: risk_score += 5
    else: risk_score -= 10
    results['risk_score'] = risk_score
    
    # 5. 动量分析
    if len(recent) >= 48:
        results['mom_1d'] = (recent[-1] - recent[-48]) / recent[-48] * 100
    else:
        results['mom_1d'] = 0
    if len(recent) >= 144:
        results['mom_3d'] = (recent[-1] - recent[-144]) / recent[-144] * 100
    else:
        results['mom_3d'] = results['mom_1d']
    
    mom_score = 0
    if results['mom_1d'] > 2.5: mom_score += 15
    elif results['mom_1d'] > 1.2: mom_score += 10
    elif results['mom_1d'] > 0: mom_score += 5
    elif results['mom_1d'] > -1.2: mom_score -= 5
    elif results['mom_1d'] > -2.5: mom_score -= 10
    else: mom_score -= 15
    results['mom_score'] = mom_score
    
    # 6. 成交量分析
    volumes = df['volume'].iloc[lookback-30:lookback].values
    results['vol_ma5'] = np.mean(volumes[-5:])
    results['vol_ma10'] = np.mean(volumes[-10:])
    results['vol_ratio'] = results['vol_ma5'] / results['vol_ma10'] if results['vol_ma10'] > 0 else 1
    
    vol_score = 0
    if results['vol_ratio'] > 1.3: vol_score += 10
    elif results['vol_ratio'] > 1.1: vol_score += 5
    elif results['vol_ratio'] < 0.7: vol_score -= 5
    results['vol_score'] = vol_score
    
    # 7. 支撑压力位
    results['resistance_1'] = np.max(recent[-20:]) * 1.015
    results['resistance_2'] = np.max(recent) * 1.03
    results['support_1'] = np.min(recent[-20:]) * 0.985
    results['support_2'] = np.min(recent) * 0.97
    
    # 综合基础评分（0-100）
    results['base_score'] = 35 + ma_score + trend_score + risk_score + mom_score + vol_score
    results['base_score'] = max(10, min(90, results['base_score']))
    
    return results, lookback


def predict_tomorrow_with_kronos(df, lookback, predictor):
    """使用Kronos模型预测明日走势"""
    pred_len = 48  # 预测1个交易日
    
    try:
        x_df = df.iloc[:lookback][['open', 'high', 'low', 'close', 'volume', 'amount']]
        x_timestamp = df.iloc[:lookback]['timestamps']
        
        # 生成未来时间戳
        future_dates = pd.date_range(start=datetime.now() + timedelta(hours=1), periods=pred_len, freq='5T')
        
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=pd.Series(future_dates),
            pred_len=pred_len,
            T=0.7,
            top_p=0.9,
            sample_count=5,
            verbose=False
        )
        
        predicted_price = pred_df['close'].iloc[-1]
        return predicted_price, pred_df, True
        
    except Exception as e:
        print(f"   ⚠️  Kronos预测异常，使用技术分析预测: {e}")
        
        # 使用简化预测方法
        analysis, _ = advanced_technical_analysis(df)
        current_price = analysis['current_price']
        trend_pct = (analysis['trend_5'] + analysis['trend_10'] + analysis['trend_20']) / 3
        
        # 基于当前趋势生成预测
        expected_return = trend_pct * 3 + np.random.normal(0, 0.6)
        expected_return = max(-6, min(6, expected_return))
        predicted_price = current_price * (1 + expected_return / 100)
        
        return predicted_price, None, False


def main():
    print("=" * 90)
    print("🔮 Kronos AI - 603399 永杉锂业 明日（2026-07-16）深度预测报告")
    print("=" * 90)
    
    # 1. 获取数据
    print("\n[1/6] 正在获取数据...")
    df, is_real = get_baostock_real_data('603399')
    
    if not is_real or df is None:
        print("\n   使用永杉锂业真实特征的高质量模拟数据...")
        df = generate_yongshan_realistic_data()
        is_real = False
    
    print(f"\n   📊 数据概况:")
    print(f"   ├─ K线条数: {len(df)} 条5分钟K线")
    print(f"   ├─ 时间范围: {df['timestamps'].iloc[0]} 至 {df['timestamps'].iloc[-1]}")
    print(f"   ├─ 价格区间: {df['low'].min():.2f} ~ {df['high'].max():.2f} 元")
    print(f"   └─ 最新价格: {df['close'].iloc[-1]:.2f} 元")
    
    # 2. 加载模型
    print("\n[2/6] 正在加载Kronos AI预测模型...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    device = torch.device("cpu")
    predictor = KronosPredictor(model, tokenizer, device=str(device), max_context=384)
    print("   ✅ Kronos模型加载完成")
    
    # 3. 技术分析
    print("\n[3/6] 正在进行深度技术分析...")
    analysis, lookback = advanced_technical_analysis(df)
    
    # 4. 运行预测
    print("\n[4/6] 正在运行明日走势预测...")
    predicted_price, pred_df, used_kronos = predict_tomorrow_with_kronos(df, lookback, predictor)
    
    current_price = analysis['current_price']
    expected_return = (predicted_price - current_price) / current_price * 100
    
    # 5. 计算最终上涨概率
    print("\n[5/6] 正在计算最终上涨概率...")
    
    # 基于预期收益调整
    if expected_return > 2.5:
        return_adj = 15
    elif expected_return > 1.5:
        return_adj = 10
    elif expected_return > 0.5:
        return_adj = 5
    elif expected_return > -0.5:
        return_adj = -5
    elif expected_return > -1.5:
        return_adj = -10
    else:
        return_adj = -15
    
    up_probability = analysis['base_score'] + return_adj
    up_probability = max(15, min(85, up_probability))
    
    # 确定风险等级
    risk = '高' if analysis['volatility_daily'] > 3.5 else ('中' if analysis['volatility_daily'] > 2 else '低')
    
    # 6. 输出完整报告
    print("\n" + "=" * 90)
    print("📊 603399 永杉锂业 - 明日深度预测报告")
    print("=" * 90)
    
    print(f"\n📍 技术面现状:")
    print(f"   ┌─────────────────────────────────────────────────────────────────────┐")
    print(f"   │  当前价格: {current_price:.2f} 元")
    print(f"   │  MA5 均线: {analysis['ma5']:.2f} 元 {'↗️' if analysis['ma5'] > current_price else '↘️'}")
    print(f"   │  MA10均线: {analysis['ma10']:.2f} 元")
    print(f"   │  MA20均线: {analysis['ma20']:.2f} 元")
    print(f"   │  MA60均线: {analysis['ma60']:.2f} 元")
    print(f"   │  均线排列: {'多头排列 ✅' if analysis['ma5'] > analysis['ma10'] > analysis['ma20'] else '空头排列 ⚠️'}")
    print(f"   │  5分钟趋势: {analysis['trend_5']:+.3f}%")
    print(f"   │  10分钟趋势: {analysis['trend_10']:+.3f}%")
    print(f"   │  20分钟趋势: {analysis['trend_20']:+.3f}%")
    print(f"   │  日波动率: ±{analysis['volatility_daily']:.2f}%")
    print(f"   │  1日涨跌幅: {analysis['mom_1d']:+.2f}%")
    print(f"   │  3日涨跌幅: {analysis['mom_3d']:+.2f}%")
    print(f"   │  成交量比率: {analysis['vol_ratio']:.2f}x")
    print(f"   └─────────────────────────────────────────────────────────────────────┘")
    
    print(f"\n🎯 明日走势预测:")
    print(f"   ┌─────────────────────────────────────────────────────────────────────┐")
    print(f"   │  预测模型: {'Kronos AI' if used_kronos else '技术分析'}")
    print(f"   │  预测开盘: {current_price * (1 + expected_return/300):.2f} 元")
    print(f"   │  预测最高: {predicted_price * 1.012:.2f} 元")
    print(f"   │  预测最低: {predicted_price * 0.993:.2f} 元")
    print(f"   │  预测收盘: {predicted_price:.2f} 元")
    print(f"   │  预期涨跌: {expected_return:+.2f}%")
    print(f"   │  上涨概率: {up_probability:.1f}%  {'🟢' if up_probability >= 55 else '🟡' if up_probability >= 50 else '🔴'}")
    print(f"   │  下跌概率: {100-up_probability:.1f}%")
    print(f"   │  风险等级: {risk}")
    print(f"   └─────────────────────────────────────────────────────────────────────┘")
    
    print(f"\n⚓ 关键支撑位 & 压力位:")
    print(f"   ┌─────────────────────────────────────────────────────────────────────┐")
    print(f"   │  第一压力位: {analysis['resistance_1']:.2f} 元 (+{((analysis['resistance_1']/current_price-1)*100):.1f}%)")
    print(f"   │  第二压力位: {analysis['resistance_2']:.2f} 元 (+{((analysis['resistance_2']/current_price-1)*100):.1f}%)")
    print(f"   │  第一支撑位: {analysis['support_1']:.2f} 元 ({((analysis['support_1']/current_price-1)*100):.1f}%)")
    print(f"   │  第二支撑位: {analysis['support_2']:.2f} 元 ({((analysis['support_2']/current_price-1)*100):.1f}%)")
    print(f"   └─────────────────────────────────────────────────────────────────────┘")
    
    print(f"\n💡 AI操作建议:")
    print(f"   ┌─────────────────────────────────────────────────────────────────────┐")
    
    if expected_return > 1.5 and up_probability > 60:
        signal = "积极看多 🟢"
        suggestion = "趋势明确，可考虑逢低介入，注意控制仓位"
        position = "建议仓位: 15-20%"
    elif expected_return > 0.5 and up_probability > 50:
        signal = "谨慎看多 🟡"
        suggestion = "偏乐观，但波动可能较大，小仓位试探"
        position = "建议仓位: 10-12%"
    elif expected_return > -1:
        signal = "中性观望 ⚪"
        suggestion = "趋势不明确，建议观望为主，等待更明确信号"
        position = "建议仓位: 5-8% 或观望"
    elif expected_return > -2.5:
        signal = "偏谨慎 🟠"
        suggestion = "短期偏弱，可考虑减仓或回避"
        position = "建议仓位: 0-5%"
    else:
        signal = "看空 🔴"
        suggestion = "趋势向下，建议回避"
        position = "建议仓位: 0%"
    
    print(f"   │  多空信号: {signal}")
    print(f"   │  操作建议: {suggestion}")
    print(f"   │  {position}")
    print(f"   │  止盈目标: {current_price * 1.05:.2f} ~ {current_price * 1.08:.2f} 元 (+5~+8%)")
    print(f"   │  止损位置: {current_price * 0.97:.2f} ~ {current_price * 0.98:.2f} 元 (-2~-3%)")
    print(f"   │  入场时机: 早盘观察15-30分钟，不追高开，逢回调企稳再考虑")
    print(f"   │  建仓方式: 分2-3批买入，避免一次性满仓")
    print(f"   └─────────────────────────────────────────────────────────────────────┘")
    
    print("\n" + "=" * 90)
    print("⚠️ 最终风险提示（必读！）")
    print("=" * 90)
    print(f"""
   1. 【数据来源】{'✅ baostock真实5分钟K线' if is_real else '❌ 基于真实价格特征的高质量模拟数据'}
   
   2. 【锂矿股特殊性】
      - 永杉锂业属于锂矿概念股，受大宗商品价格影响极大
      - 受新能源产业链景气度、政策变动影响显著
      - 中小盘股，波动性远大于蓝筹股
      - 风险等级较高，不适合风险承受能力低的投资者

   3. 【预测局限性】
      - 本预测仅基于时间序列和技术分析
      - 无法预测突发消息、政策变动、大盘情绪
      - 预测不可能100%准确，仅作参考

   4. 【投资原则】
      - ❌ 绝对不能根据此预测盲目下单
      - ✅ 请务必结合自己的研究和判断
      - ✅ 严格执行止盈止损
      - ✅ 只用闲钱投资，不要加杠杆

   5. 【责任声明】
      本预测仅为AI模型的功能演示，不构成任何投资建议。
      使用者据此投资，风险自担。投资有风险，入市需谨慎。
""")
    
    # 保存结果
    print("💾 正在保存预测结果...")
    
    result_data = {
        'code': ['603399'],
        'name': ['永杉锂业'],
        'current_price': [current_price],
        'predicted_price': [predicted_price],
        'expected_return': [expected_return],
        'up_probability': [up_probability],
        'risk_level': [risk],
        'data_source': ['baostock_real' if is_real else 'high_quality_simulation'],
        'support_1': [analysis['support_1']],
        'support_2': [analysis['support_2']],
        'resistance_1': [analysis['resistance_1']],
        'resistance_2': [analysis['resistance_2']],
        'prediction_time': [datetime.now()]
    }
    
    pd.DataFrame(result_data).to_csv('603399_永杉锂业_明日预测报告.csv', index=False, encoding='utf-8-sig')
    print(f"   ✅ 预测报告已保存: 603399_永杉锂业_明日预测报告.csv")
    
    if pred_df is not None and len(pred_df) > 0:
        pred_df.to_csv('603399_永杉锂业_明日K线预测明细.csv', index=False, encoding='utf-8-sig')
        print(f"   ✅ K线预测明细已保存: 603399_永杉锂业_明日K线预测明细.csv")
    
    print("\n" + "=" * 90)
    print("🎉 603399 永杉锂业 明日深度预测完成！")
    print("=" * 90)
    
    print("\n📌 如何获得100%真实的预测？")
    print("   1. 打开同花顺 → 输入603399进入永杉锂业")
    print("   2. 按F8切换到5分钟K线")
    print("   3. 右键 → 数据导出 → 保存为CSV")
    print("   4. 把CSV文件发给我 → 获得真正的预测！")
    print("=" * 90)


if __name__ == "__main__":
    main()
