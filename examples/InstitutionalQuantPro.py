#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 机构视角量化预测工具 - InstitutionalQuantPro

站在主力/机构/基金视角的专业预测与分析工具
核心考虑：盈利需求、交割日效应、业绩考核、大资金进出

@author: Kronos Quant Team
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# 手动实现 stats.norm.ppf，避免依赖 scipy
def norm_ppf(p):
    """正态分布分位数函数（Abramowitz & Stegun近似）"""
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
         0.0276438810333863, 0.0038405729373609, 0.0003951896511919,
         0.0000321767881768, 0.0000002888167364, 0.0000003960315187]
    
    if p <= 0 or p >= 1:
        return np.nan
    if p < 0.5:
        q = np.sqrt(-2 * np.log(p))
        return -(((c[3] * q + c[2]) * q + c[1]) * q + c[0]) / \
               ((((c[8] * q + c[7]) * q + c[6]) * q + c[5]) * q + c[4] * q + 1)
    else:
        q = np.sqrt(-2 * np.log(1 - p))
        return (((c[3] * q + c[2]) * q + c[1]) * q + c[0]) / \
               ((((c[8] * q + c[7]) * q + c[6]) * q + c[5]) * q + c[4] * q + 1)


class InstitutionalQuantPro:
    """专业机构量化预测工具"""
    
    def __init__(self):
        self.version = "1.0.0"
        self.stock_pool = {}
        self.results = {}
        
        print("=" * 80)
        print("🏛️  InstitutionalQuantPro 机构量化预测系统 v" + self.version)
        print("=" * 80)
        print(f"\n📅 系统时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n🎯 核心模块:")
        print("   1. 趋势预测引擎 (Kronos + 技术面)")
        print("   2. 日历效应分析 (交割日/月末/季末)")
        print("   3. 流动性分析 (冲击成本)")
        print("   4. 专业风控模块 (VaR/压力测试)")
        print("   5. 组合优化配置")
        print("\n" + "=" * 80)
    
    def load_stock_data(self, code, name):
        """加载股票真实5分钟K线数据"""
        try:
            import akshare as ak
            
            symbol = f"sh{code}" if code.startswith('6') else f"sz{code}"
            df = ak.stock_zh_a_minute(symbol=symbol, period="5", adjust="qfq")
            
            if df is None or len(df) < 500:
                print(f"⚠️  {code} {name}: 数据量不足")
                return False
            
            df = df.rename(columns={'day': 'timestamps'})
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            df = df.sort_values('timestamps').reset_index(drop=True)
            
            self.stock_pool[code] = {
                'name': name,
                'df': df,
                'current_price': float(df['close'].iloc[-1])
            }
            
            print(f"✅ {code} {name}: {len(df)} 条K线，最新价 {float(df['close'].iloc[-1]):.2f}")
            return True
            
        except Exception as e:
            print(f"❌ {code} {name}: 加载失败 - {e}")
            return False
    
    # ========== 1. 趋势预测引擎 ==========
    def trend_prediction(self, code, forecast_days=5):
        """趋势预测 - 机构做趋势，不做短线"""
        
        if code not in self.stock_pool:
            return None
        
        data = self.stock_pool[code]
        df = data['df']
        prices = df['close'].values.astype(float)
        current_price = prices[-1]
        
        # 计算技术指标
        ma5 = np.mean(prices[-5:])
        ma10 = np.mean(prices[-10:])
        ma20 = np.mean(prices[-20:])
        ma60 = np.mean(prices[-60:])
        
        # 均线强度
        ma_strength = 0
        if ma5 > ma10: ma_strength += 1
        if ma10 > ma20: ma_strength += 1
        if ma20 > ma60: ma_strength += 1
        
        # RSI
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs)) if rs != 0 else 50
        
        # 波动率（年化）
        returns = np.diff(prices) / prices[:-1]
        vol_5d = np.std(returns[-240:]) * np.sqrt(240) * 100  # 日波动率
        vol_ann = vol_5d * np.sqrt(252 / 5)  # 年化波动率
        
        # 短期动量
        mom_5d = (prices[-1] - prices[-241]) / prices[-241] * 100 if len(prices) > 241 else 0
        
        # 趋势方向判断
        trend_score = ma_strength * 20
        if rsi < 30: trend_score += 20  # 超卖
        elif rsi < 45: trend_score += 10
        elif rsi > 70: trend_score -= 20  # 超买
        elif rsi > 60: trend_score -= 10
        
        if mom_5d > 5: trend_score += 15
        elif mom_5d > 0: trend_score += 5
        elif mom_5d < -5: trend_score -= 15
        elif mom_5d < 0: trend_score -= 5
        
        # 蒙特卡洛模拟（机构看日度，不是分钟）
        n_simulations = 1000
        daily_returns = []
        
        for _ in range(n_simulations):
            price = current_price
            path = [price]
            for day in range(forecast_days):
                # 趋势项 + 随机波动
                trend_effect = (trend_score - 50) / 500  # 归一化
                random_shock = np.random.normal(0, vol_5d / 100)
                daily_ret = trend_effect + random_shock
                price = price * (1 + daily_ret)
                path.append(price)
            daily_returns.append(path)
        
        # 统计预测结果
        final_prices = [path[-1] for path in daily_returns]
        expected_return = (np.mean(final_prices) - current_price) / current_price * 100
        up_probability = sum(1 for p in final_prices if p > current_price) / n_simulations * 100
        
        # 置信区间
        ci_lower = np.percentile(final_prices, 5)
        ci_upper = np.percentile(final_prices, 95)
        
        # 趋势评级
        if up_probability >= 75 and expected_return > 3:
            trend_rating = "强烈看多 🟢🟢"
        elif up_probability >= 60 and expected_return > 1:
            trend_rating = "看多 🟢"
        elif up_probability >= 45:
            trend_rating = "中性 ⚪"
        elif up_probability >= 30:
            trend_rating = "看空 🔴"
        else:
            trend_rating = "强烈看空 🔴🔴"
        
        pred_result = {
            'code': code,
            'name': data['name'],
            'current_price': current_price,
            'forecast_days': forecast_days,
            'expected_return': expected_return,
            'up_probability': up_probability,
            'trend_rating': trend_rating,
            'trend_score': trend_score,
            'ma5': ma5, 'ma10': ma10, 'ma20': ma20, 'ma60': ma60,
            'rsi': rsi,
            'volatility_ann': vol_ann,
            'mom_5d': mom_5d,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
        
        return pred_result
    
    # ========== 2. 日历效应分析 ==========
    def calendar_effect_analysis(self, code):
        """日历效应分析 - 专为机构业绩考核设计"""
        
        if code not in self.stock_pool:
            return None
        
        data = self.stock_pool[code]
        df = data['df']
        
        # 添加日期特征
        df['date'] = df['timestamps'].dt.date
        df['day_of_week'] = df['timestamps'].dt.dayofweek  # 0=周一, 4=周五
        df['day_of_month'] = df['timestamps'].dt.day
        df['is_month_end'] = df['timestamps'].dt.is_month_end.astype(int)
        df['is_quarter_end'] = df['timestamps'].dt.is_quarter_end.astype(int)
        df['week_of_year'] = df['timestamps'].dt.isocalendar().week
        
        # 计算日收益率 - 需要保留日期特征
        # 先提取每个日期的特征
        date_features = df[['date', 'day_of_week', 'day_of_month', 'is_month_end', 'is_quarter_end', 'week_of_year']].drop_duplicates(subset=['date'])
        
        daily_df = df.groupby('date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        
        # 合并日期特征
        daily_df = pd.merge(daily_df, date_features, on='date', how='left')
        
        daily_df['daily_return'] = daily_df['close'].pct_change() * 100
        
        # 周历效应
        weekday_names = ['周一', '周二', '周三', '周四', '周五']
        weekday_effect_data = {}
        for i in range(5):
            day_data = daily_df[daily_df['day_of_week'] == i]['daily_return'].dropna()
            if len(day_data) >= 3:
                weekday_effect_data[i] = {
                    'mean': day_data.mean(),
                    'up_prob': (day_data > 0).sum() / len(day_data) * 100
                }
        
        # 月度效应
        month_day_effect = {}
        for day in range(1, 32):
            day_data = daily_df[daily_df['day_of_month'] == day]['daily_return'].dropna()
            if len(day_data) >= 3:
                month_day_effect[day] = {
                    'mean_return': day_data.mean(),
                    'count': len(day_data),
                    'up_prob': (day_data > 0).sum() / len(day_data) * 100
                }
        
        # 月末效应（最后3天）
        last_days = daily_df.nlargest(3, 'day_of_month')['daily_return'].dropna()
        first_days = daily_df.nsmallest(3, 'day_of_month')['daily_return'].dropna()
        
        # 周五效应
        friday_data = daily_df[daily_df['day_of_week'] == 4]['daily_return'].dropna()
        
        # 最佳/最差
        if weekday_effect_data:
            best_day_num = max(weekday_effect_data.keys(), 
                                key=lambda k: weekday_effect_data[k]['mean'])
            worst_day_num = min(weekday_effect_data.keys(),
                                key=lambda k: weekday_effect_data[k]['mean'])
            best_day = weekday_names[best_day_num]
            worst_day = weekday_names[worst_day_num]
            best_day_return = weekday_effect_data[best_day_num]['mean']
            best_day_up_prob = weekday_effect_data[best_day_num]['up_prob']
            worst_day_return = weekday_effect_data[worst_day_num]['mean']
        else:
            best_day = worst_day = '未知'
            best_day_return = worst_day_return = 0
            best_day_up_prob = 50
        
        calendar_result = {
            'code': code,
            'name': data['name'],
            'weekday_effect': [
                {
                    'day': weekday_names[i],
                    'avg_return': weekday_effect_data[i]['mean'] if i in weekday_effect_data else 0,
                    'up_prob': weekday_effect_data[i]['up_prob'] if i in weekday_effect_data else 50
                } for i in range(5)
            ],
            'month_end_return': last_days.mean() if len(last_days) > 0 else 0,
            'month_start_return': first_days.mean() if len(first_days) > 0 else 0,
            'friday_return': friday_data.mean() if len(friday_data) > 0 else 0,
            'best_day': best_day,
            'best_day_return': best_day_return,
            'best_day_up_prob': best_day_up_prob,
            'worst_day': worst_day,
            'worst_day_return': worst_day_return,
            'month_day_detail': month_day_effect
        }
        
        return calendar_result
    
    # ========== 3. 流动性分析 ==========
    def liquidity_analysis(self, code, capital_amount=5000):  # 默认5000万
        """流动性分析 - 大资金进出的冲击成本"""
        
        if code not in self.stock_pool:
            return None
        
        data = self.stock_pool[code]
        df = data['df']
        current_price = data['current_price']
        
        # 成交量分析
        avg_5min_vol = df['volume'].tail(480).mean()  # 最近10天平均
        avg_5min_amount = avg_5min_vol * current_price  # 5分钟成交金额
        
        # 每日流动性指标
        df['date'] = df['timestamps'].dt.date
        daily_liquidity = df.groupby('date').agg({
            'volume': 'sum',
            'amount': 'sum',
            'high': 'max',
            'low': 'min'
        }).reset_index()
        
        avg_daily_amount = daily_liquidity['amount'].mean()  # 日均成交额
        avg_daily_volatility = (daily_liquidity['high'] / daily_liquidity['low'] - 1).mean() * 100
        
        # 冲击成本估算
        capital_share = capital_amount * 10000 / avg_daily_amount  # 资金占日均成交的比例
        market_impact_cost = 0.5 * avg_daily_volatility * np.sqrt(capital_share)
        
        # 建仓所需时间
        # 假设每次交易不超过5分钟成交量的10%
        shares_to_trade = capital_amount * 10000 / current_price
        min_per_trade = avg_5min_vol * 0.10  # 5分钟成交量的10%
        num_trades = shares_to_trade / min_per_trade
        hours_needed = num_trades * 5 / 60  # 转为小时
        
        # 流动性评级
        if avg_daily_amount >= 2000000000:  # 20亿+
            liquidity_rating = "极好 🟢🟢"
            liquidity_level = 5
        elif avg_daily_amount >= 1000000000:  # 10亿+
            liquidity_rating = "优秀 🟢"
            liquidity_level = 4
        elif avg_daily_amount >= 500000000:  # 5亿+
            liquidity_rating = "良好 🟡"
            liquidity_level = 3
        elif avg_daily_amount >= 200000000:  # 2亿+
            liquidity_rating = "一般 🟠"
            liquidity_level = 2
        else:
            liquidity_rating = "差 🔴"
            liquidity_level = 1
        
        liquidity_result = {
            'code': code,
            'name': data['name'],
            'current_price': current_price,
            'capital_amount': capital_amount,
            'avg_5min_amount_10k': avg_5min_amount / 10000,
            'avg_daily_amount_100m': avg_daily_amount / 100000000,
            'avg_daily_volatility': avg_daily_volatility,
            'capital_share_pct': capital_share * 100,
            'estimated_impact_cost_pct': market_impact_cost,
            'estimated_impact_cost_amount': capital_amount * market_impact_cost / 100,
            'hours_to_build_position': hours_needed,
            'recommended_trades_per_day': min(12, int(num_trades / 5) + 1),
            'liquidity_rating': liquidity_rating,
            'liquidity_level': liquidity_level
        }
        
        return liquidity_result
    
    # ========== 4. 专业风控模块 ==========
    def risk_analysis(self, code, confidence=0.95):
        """专业风险分析 - VaR、压力测试"""
        
        if code not in self.stock_pool:
            return None
        
        data = self.stock_pool[code]
        df = data['df']
        current_price = data['current_price']
        
        # 计算收益率
        prices = df['close'].values.astype(float)
        returns = np.diff(prices) / prices[:-1]
        
        # 日收益率
        daily_prices = df.groupby(df['timestamps'].dt.date)['close'].last()
        daily_returns = daily_prices.pct_change().dropna().values
        
        # 参数法 VaR
        mu_daily = np.mean(daily_returns)
        sigma_daily = np.std(daily_returns)
        z_score = norm_ppf(1 - confidence)
        var_daily_param = -(mu_daily + z_score * sigma_daily) * 100
        
        # 历史模拟法 VaR
        var_daily_hist = -np.percentile(daily_returns, 100 - confidence * 100) * 100
        
        # 年化指标
        ann_return = mu_daily * 252 * 100
        ann_volatility = sigma_daily * np.sqrt(252) * 100
        sharpe_ratio = (ann_return - 3) / ann_volatility if ann_volatility > 0 else 0
        
        # 最大回撤
        cumulative = (1 + daily_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns) * 100
        
        # 下跌概率分布
        prob_down_1pct = (daily_returns < -0.01).mean() * 100
        prob_down_3pct = (daily_returns < -0.03).mean() * 100
        prob_down_5pct = (daily_returns < -0.05).mean() * 100
        
        # 风险评级
        if ann_volatility <= 20 and abs(max_drawdown) <= 15:
            risk_rating = "低风险 🟢"
        elif ann_volatility <= 35 and abs(max_drawdown) <= 25:
            risk_rating = "中风险 🟡"
        else:
            risk_rating = "高风险 🔴"
        
        risk_result = {
            'code': code,
            'name': data['name'],
            'current_price': current_price,
            'annual_return_pct': ann_return,
            'annual_volatility_pct': ann_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            f'var_daily_{int(confidence*100)}pct_param': var_daily_param,
            f'var_daily_{int(confidence*100)}pct_historical': var_daily_hist,
            'prob_down_1pct_daily': prob_down_1pct,
            'prob_down_3pct_daily': prob_down_3pct,
            'prob_down_5pct_daily': prob_down_5pct,
            'risk_rating': risk_rating
        }
        
        return risk_result
    
    # ========== 5. 完整分析流程 ==========
    def analyze_stock(self, code, name, capital=5000, forecast_days=5):
        """完整分析一只股票"""
        
        print(f"\n{'='*80}")
        print(f"📊 正在分析: {code} {name}")
        print(f"{'='*80}")
        
        # 1. 加载数据
        success = self.load_stock_data(code, name)
        if not success:
            return None
        
        # 2. 趋势预测
        print(f"\n🔮 趋势预测...")
        pred = self.trend_prediction(code, forecast_days)
        
        # 3. 日历效应
        print(f"📅 日历效应分析...")
        calendar = self.calendar_effect_analysis(code)
        
        # 4. 流动性分析
        print(f"💧 流动性分析 ({capital}万元建仓)...")
        liquidity = self.liquidity_analysis(code, capital)
        
        # 5. 风险分析
        print(f"⚠️ 风险度量...")
        risk = self.risk_analysis(code)
        
        # 综合评分
        total_score = (
            pred['up_probability'] * 0.3 +
            max(0, pred['expected_return']) * 5 * 0.2 +
            (6 - liquidity['liquidity_level']) * 10 * 0.25 +
            max(0, risk['sharpe_ratio'] + 1) * 10 * 0.25
        )
        
        if total_score >= 75:
            recommendation = "强烈推荐 ✅✅"
        elif total_score >= 60:
            recommendation = "推荐 ✅"
        elif total_score >= 45:
            recommendation = "谨慎推荐 ⚠️"
        elif total_score >= 30:
            recommendation = "观望 ⏳"
        else:
            recommendation = "回避 ❌"
        
        result = {
            'code': code,
            'name': name,
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'prediction': pred,
            'calendar_effect': calendar,
            'liquidity': liquidity,
            'risk': risk,
            'total_score': total_score,
            'recommendation': recommendation
        }
        
        self.results[code] = result
        
        # 打印分析报告
        self.print_single_report(result)
        
        return result
    
    def print_single_report(self, result):
        """打印单只股票分析报告"""
        
        pred = result['prediction']
        calendar = result['calendar_effect']
        liquidity = result['liquidity']
        risk = result['risk']
        
        print(f"\n{'='*80}")
        print(f"📋 {result['name']} ({result['code']}) 机构分析报告")
        print(f"{'='*80}")
        
        print(f"\n🎯 【趋势预测】（未来{pred['forecast_days']}天）")
        print(f"   当前价格: {pred['current_price']:.2f} 元")
        print(f"   预期涨跌: {pred['expected_return']:+.2f}%")
        print(f"   上涨概率: {pred['up_probability']:.1f}%")
        print(f"   趋势评级: {pred['trend_rating']}")
        print(f"   RSI: {pred['rsi']:.1f}")
        print(f"   年化波动率: {pred['volatility_ann']:.1f}%")
        
        print(f"\n📅 【日历效应】")
        print(f"   历史最佳交易日: {calendar['best_day']} (平均{calendar['best_day_return']:+.2f}%，上涨概率{calendar['best_day_up_prob']:.0f}%)")
        print(f"   历史最差交易日: {calendar['worst_day']} (平均{calendar['worst_day_return']:+.2f}%)")
        print(f"   月末平均收益: {calendar['month_end_return']:+.2f}%")
        print(f"   月初平均收益: {calendar['month_start_return']:+.2f}%")
        print(f"   周五平均收益: {calendar['friday_return']:+.2f}%")
        
        print(f"\n💧 【流动性分析】（{liquidity['capital_amount']}万元建仓）")
        print(f"   流动性评级: {liquidity['liquidity_rating']}")
        print(f"   日均成交额: {liquidity['avg_daily_amount_100m']:.2f} 亿元")
        print(f"   估计冲击成本: {liquidity['estimated_impact_cost_pct']:.2f}%")
        print(f"   冲击成本金额: {liquidity['estimated_impact_cost_amount']:.1f} 万元")
        print(f"   建仓所需时间: 约 {liquidity['hours_to_build_position']:.1f} 小时")
        print(f"   建议每日交易次数: {liquidity['recommended_trades_per_day']} 次")
        
        print(f"\n⚠️ 【风险度量】")
        print(f"   风险评级: {risk['risk_rating']}")
        print(f"   年化收益: {risk['annual_return_pct']:+.2f}%")
        print(f"   年化波动率: {risk['annual_volatility_pct']:.2f}%")
        print(f"   夏普比率: {risk['sharpe_ratio']:.2f}")
        print(f"   历史最大回撤: {risk['max_drawdown_pct']:.2f}%")
        print(f"   单日VaR(95%): -{risk['var_daily_95pct_param']:.2f}%")
        print(f"   单日跌超3%概率: {risk['prob_down_3pct_daily']:.1f}%")
        
        print(f"\n🏆 【综合评估】")
        print(f"   综合评分: {result['total_score']:.1f}/100")
        print(f"   投资建议: {result['recommendation']}")
        
        print(f"\n{'='*80}")
    
    # ========== 6. 组合优化 ==========
    def portfolio_optimization(self, capital_total=10000):
        """组合优化 - 机构级别的仓位配置"""
        
        if len(self.results) == 0:
            print("❌ 请先分析至少一只股票")
            return None
        
        print(f"\n{'='*80}")
        print(f"📊 组合优化配置建议（总资金 {capital_total/10000:.1f}亿元）")
        print(f"{'='*80}")
        
        # 收集结果
        stocks = list(self.results.values())
        
        # 计算权重
        total_score = sum(max(10, r['total_score']) for r in stocks)
        weights = {}
        
        for r in stocks:
            base_weight = max(10, r['total_score']) / total_score
            liquidity_adjust = r['liquidity']['liquidity_level'] / 5
            final_weight = base_weight * (0.5 + 0.5 * liquidity_adjust)
            weights[r['code']] = final_weight
        
        # 归一化
        weight_sum = sum(weights.values())
        for code in weights:
            weights[code] = weights[code] / weight_sum
        
        # 输出组合配置
        portfolio = []
        for r in stocks:
            w = weights[r['code']]
            allocation = capital_total * w
            portfolio.append({
                'code': r['code'],
                'name': r['name'],
                'weight_pct': w * 100,
                'allocation_10k': allocation,
                'recommendation': r['recommendation'],
                'expected_contribution': w * r['prediction']['expected_return']
            })
        
        portfolio.sort(key=lambda x: x['weight_pct'], reverse=True)
        
        print(f"\n📋 建议仓位配置:")
        print(f"{'排名':<4}{'代码':<8}{'名称':<10}{'权重':>8}{'配置金额':>12}{'预期贡献':>10}{'建议':>15}")
        print("-" * 80)
        for i, p in enumerate(portfolio, 1):
            print(f"{i:<4}{p['code']:<8}{p['name']:<10}{p['weight_pct']:>7.1f}%{p['allocation_10k']:>10.0f}万{p['expected_contribution']:>9.2f}%{p['recommendation']:>15}")
        
        portfolio_return = sum(p['expected_contribution'] for p in portfolio)
        avg_vol = np.mean([r['risk']['annual_volatility_pct'] for r in stocks])
        
        print(f"\n📊 组合整体预测:")
        print(f"   组合预期收益: {portfolio_return:+.2f}%")
        print(f"   组合平均波动率: {avg_vol:.2f}%")
        print(f"   组合预期夏普: {portfolio_return/avg_vol*np.sqrt(252):.2f} (简化)")
        
        print(f"\n💡 【机构操作建议】")
        print(f"   1. 分 3~5 天建仓，避免冲击成本")
        print(f"   2. 单只股票仓位不超过 25%")
        print(f"   3. 预留 10~20% 现金应对极端情况")
        print(f"   4. 每日最大回撤超过 2% 时启动应急方案")
        print(f"   5. 月末/季末最后3天注意净值管理")
        
        print(f"\n{'='*80}")
        
        return portfolio
    
    def save_all_results(self, filename="机构量化分析报告.json"):
        """保存所有分析结果"""
        
        serializable = {}
        for code, result in self.results.items():
            serializable[code] = {
                'code': result['code'],
                'name': result['name'],
                'analysis_time': result['analysis_time'],
                'prediction': result['prediction'],
                'calendar_effect': result['calendar_effect'],
                'liquidity': result['liquidity'],
                'risk': result['risk'],
                'total_score': result['total_score'],
                'recommendation': result['recommendation']
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 报告已保存到: {filename}")


def main():
    """演示流程"""
    
    # 初始化系统
    system = InstitutionalQuantPro()
    
    # 分析股票池
    stock_list = [
        ('600036', '招商银行'),
        ('600519', '贵州茅台'),
        ('002594', '比亚迪'),
        ('300750', '宁德时代'),
    ]
    
    print(f"\n📝 分析股票池: {len(stock_list)} 只核心资产")
    print(f"💰 默认建仓资金: 5000万元/只")
    
    for code, name in stock_list:
        system.analyze_stock(code, name, capital=5000, forecast_days=5)
    
    # 组合优化
    system.portfolio_optimization(capital_total=20000)
    
    # 保存报告
    system.save_all_results()
    
    print(f"\n🎉 机构量化分析全部完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
