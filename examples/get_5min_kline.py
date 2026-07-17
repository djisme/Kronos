#!/usr/bin/env python3
"""
多数据源自动获取5分钟K线脚本
整合：akshare + baostock + tushare，确保至少能拿到数据
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def get_data_akshare(code="603399", days=30):
    """使用akshare获取数据"""
    try:
        import akshare as ak
        print("✅ akshare 已加载")
        
        # 尝试多种接口
        try:
            # 方式1: 分钟K线
            df = ak.stock_zh_a_minute(
                symbol=code,
                period="5",
                adjust="qfq"
            )
            if df is not None and len(df) > 0:
                print(f"✅ akshare 分钟K线获取成功，共 {len(df)} 条")
                return df, "akshare"
        except:
            pass
        
        try:
            # 方式2: 日线（降级方案）
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                adjust="qfq"
            )
            if df is not None and len(df) > 0:
                print(f"✅ akshare 日线获取成功，共 {len(df)} 条")
                return df, "akshare_daily"
        except:
            pass
        
    except Exception as e:
        print(f"❌ akshare 失败: {e}")
    
    return None, None


def get_data_baostock(code="603399", days=30):
    """使用baostock获取数据（更稳定）"""
    try:
        import baostock as bs
        print("✅ baostock 已加载")
        
        lg = bs.login()
        if lg.error_code != '0':
            print("❌ baostock 登录失败")
            return None, None
        
        # 计算日期范围
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # 转换股票代码格式
        bs_code = f"sh.{code}" if code.startswith('6') else f"sz.{code}"
        
        # 获取5分钟K线
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,time,open,high,low,close,volume,amount",
            start_date=start_date,
            end_date=end_date,
            frequency="5",
            adjustflag="2"
        )
        
        if rs.error_code == '0':
            df = rs.get_data()
            if len(df) > 0:
                print(f"✅ baostock 获取成功，共 {len(df)} 条5分钟K线")
                bs.logout()
                return df, "baostock"
        
        # 如果5分钟失败，尝试日线
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"
        )
        
        if rs.error_code == '0':
            df = rs.get_data()
            if len(df) > 0:
                print(f"✅ baostock 日线获取成功，共 {len(df)} 条")
                bs.logout()
                return df, "baostock_daily"
        
        bs.logout()
        
    except Exception as e:
        print(f"❌ baostock 失败: {e}")
    
    return None, None


def get_data_tushare(code="603399", days=30):
    """使用tushare获取数据（备用）"""
    try:
        import tushare as ts
        print("✅ tushare 已加载")
        
        # 尝试获取分钟数据
        try:
            pro = ts.pro_api()
            # 需要token，降级到简单方法
        except:
            pass
        
        # 使用旧版接口
        df = ts.get_k_data(code, ktype='5')
        if df is not None and len(df) > 0:
            print(f"✅ tushare 获取成功，共 {len(df)} 条")
            return df, "tushare"
            
    except Exception as e:
        print(f"❌ tushare 失败: {e}")
    
    return None, None


def generate_simulation_data(code="603399", days=10):
    """
    备用方案：生成高质量5分钟模拟数据
    基于永杉锂业真实价格特征
    """
    print("\n💡 使用备用方案：生成高质量模拟数据")
    
    np.random.seed(hash(code) % 10000)
    
    bars_per_day = 48
    total_bars = days * bars_per_day
    
    t = np.arange(total_bars)
    
    # 模拟下跌趋势（符合截图）
    start_price = 15.5
    end_price = 14.31
    trend = np.linspace(0, (end_price - start_price) / start_price, total_bars)
    
    # 叠加波动
    mid_wave = 0.012 * np.sin(t / 70)
    short_wave = 0.007 * np.sin(t / 25) + 0.003 * np.sin(t / 6)
    noise = np.random.normal(0, 0.0075, total_bars)
    
    # 综合价格
    price_factors = 1 + trend + mid_wave * 0.04 + short_wave * 0.035 + np.cumsum(noise) / 25
    close_prices = start_price * price_factors
    
    open_prices = close_prices.copy()
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.0065, total_bars)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.0065, total_bars)))
    
    volume_base = 145000
    volume_trend = 1 + 0.38 * np.abs(np.sin(t / 50))
    volume = volume_base * volume_trend * (1 + 0.32 * np.random.rand(total_bars))
    amount = close_prices * volume
    
    # 生成交易时间
    dates = []
    base_date = datetime.now() - timedelta(days=days)
    
    for day in range(days + 2):
        current_date = base_date + timedelta(days=day)
        if current_date.weekday() >= 5:
            continue
        if len(dates) >= total_bars:
            break
            
        for i in range(24):
            dates.append(current_date.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(minutes=5*i))
            if len(dates) >= total_bars:
                break
        if len(dates) >= total_bars:
            break
            
        for i in range(24):
            dates.append(current_date.replace(hour=13, minute=0, second=0, microsecond=0) + timedelta(minutes=5*i))
            if len(dates) >= total_bars:
                break
    
    actual_len = min(len(dates), len(close_prices))
    
    df = pd.DataFrame({
        'time': dates[:actual_len],
        'open': open_prices[:actual_len],
        'high': high_prices[:actual_len],
        'low': low_prices[:actual_len],
        'close': close_prices[:actual_len],
        'volume': volume[:actual_len],
        'amount': amount[:actual_len]
    })
    
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    return df, "simulation"


def process_data(df, source):
    """统一处理不同来源的数据"""
    print(f"\n🔧 正在处理 {source} 数据...")
    
    # 列名映射字典
    mappings = {
        'akshare': {
            'day': 'timestamps',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'amount': 'amount'
        },
        'akshare_daily': {
            '日期': 'timestamps',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume',
            '成交额': 'amount'
        },
        'baostock': {
            'time': 'timestamps',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'amount': 'amount'
        },
        'baostock_daily': {
            'date': 'timestamps',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'amount': 'amount'
        },
        'tushare': {
            'date': 'timestamps',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'amount': 'amount'
        },
        'simulation': {
            'time': 'timestamps',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'amount': 'amount'
        }
    }
    
    # 重命名列
    if source in mappings:
        df = df.rename(columns=mappings[source])
    
    # 尝试其他常见列名
    column_name_attempts = [
        {'日期': 'timestamps', '时间': 'timestamps', 'day': 'timestamps'},
        {'开盘价': 'open', '开盘': 'open'},
        {'最高价': 'high', '最高': 'high'},
        {'最低价': 'low', '最低': 'low'},
        {'收盘价': 'close', '收盘': 'close'},
        {'成交量': 'volume', 'vol': 'volume', '成交量(股)': 'volume'},
        {'成交额': 'amount', 'amt': 'amount'},
    ]
    
    for mapping in column_name_attempts:
        df = df.rename(columns=mapping)
    
    # 确保有必须的列
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}，列名: {list(df.columns)}")
    
    # 处理时间列
    if 'timestamps' in df.columns:
        df['timestamps'] = pd.to_datetime(df['timestamps'], errors='coerce')
    else:
        print("   ⚠️ 未找到时间列，生成5分钟间隔时间戳")
        df['timestamps'] = pd.date_range(end=datetime.now(), periods=len(df), freq='5T')
    
    # 处理成交额
    if 'amount' not in df.columns or df['amount'].iloc[0] is None:
        df['amount'] = pd.to_numeric(df['close'], errors='coerce') * pd.to_numeric(df['volume'], errors='coerce')
    
    # 确保数值列是数字类型
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 删除空值
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    
    # 只保留需要的列
    df = df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']].copy()
    
    # 按时间排序
    df = df.sort_values('timestamps').reset_index(drop=True)
    
    print(f"✅ 数据处理完成！")
    print(f"   📊 有效数据: {len(df)} 条K线")
    if len(df) > 0:
        print(f"   📅 时间范围: {df['timestamps'].iloc[0]} ~ {df['timestamps'].iloc[-1]}")
        print(f"   💰 价格范围: {df['low'].min():.2f} ~ {df['high'].max():.2f} 元")
        print(f"   🎯 当前价格: {df['close'].iloc[-1]:.2f} 元")
    
    return df


def main():
    print("=" * 80)
    print("📈 多数据源自动获取5分钟K线工具")
    print("=" * 80)
    
    # 获取命令行参数
    code = sys.argv[1] if len(sys.argv) > 1 else "603399"
    name = sys.argv[2] if len(sys.argv) > 2 else "永杉锂业"
    
    print(f"\n🎯 目标股票: {code} {name}")
    print("=" * 80)
    
    # 尝试各个数据源
    df, source = None, None
    
    # 1. 尝试 baostock（最稳定）
    print("\n[1/4] 尝试 baostock (推荐)...")
    df, source = get_data_baostock(code, days=30)
    
    # 2. 尝试 akshare
    if df is None or len(df) == 0:
        print("\n[2/4] 尝试 akshare...")
        df, source = get_data_akshare(code, days=30)
    
    # 3. 尝试 tushare
    if df is None or len(df) == 0:
        print("\n[3/4] 尝试 tushare...")
        df, source = get_data_tushare(code, days=30)
    
    # 4. 使用模拟数据
    if df is None or len(df) == 0:
        print("\n[4/4] 所有在线数据源均不可用，使用备用方案...")
        df, source = generate_simulation_data(code, days=10)
    
    # 处理数据
    df = process_data(df, source)
    
    # 保存数据
    filename = f"{code}_{name}_5分钟K线_{source}.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"\n💾 数据已保存: {filename}")
    print(f"\n✅ 数据获取完成！")
    print("=" * 80)
    
    print("\n📖 接下来可以运行:")
    print(f"   python predict_from_csv.py {filename} {name}")
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    main()
