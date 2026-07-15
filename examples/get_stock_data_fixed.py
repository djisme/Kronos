#!/usr/bin/env python3
"""
A 股数据获取工具 - 修复版

支持股票代码示例:
    上证: 600519 (贵州茅台), 600036 (招商银行)
    深证: 000001 (平安银行), 002594 (比亚迪)
    创业板: 300750 (宁德时代)
"""

import argparse
import sys
import os
import pandas as pd
from datetime import datetime

try:
    import akshare as ak
    HAS_AKSHARE = True
except ImportError:
    HAS_AKSHARE = False


def download_stock_data(stock_code, count=1000):
    """获取 A 股 5分钟 K 线数据"""
    if not HAS_AKSHARE:
        print("❌ 错误: 未安装 akshare")
        print("请先安装: pip install akshare")
        sys.exit(1)
    
    print(f"🔍 正在获取股票 {stock_code} 的 5分钟 K 线数据...")
    
    # 方法1: 尝试 stock_zh_a_minute
    try:
        print("   尝试接口1: stock_zh_a_minute...")
        df = ak.stock_zh_a_minute(
            symbol=stock_code,
            period='5',
            adjust='qfq'
        )
        if df is not None and len(df) > 0:
            print(f"   ✅ 接口1成功! 获取到 {len(df)} 条数据")
            return process_data(df, count)
    except Exception as e:
        print(f"   接口1失败: {e}")
    
    # 方法2: 尝试 get_futures_daily
    try:
        print("   尝试接口2: stock_zh_a_daily...")
        # 先获取日数据作为备选
        df = ak.stock_zh_a_daily(
            symbol=stock_code,
            adjust='qfq'
        )
        if df is not None and len(df) > 0:
            print(f"   ⚠️  获取到日线数据共 {len(df)} 条")
            print("   注意: 日线数据，模型需要5分钟数据效果更好")
            return process_data(df, min(count, len(df)))
    except Exception as e:
        print(f"   接口2失败: {e}")
    
    # 方法3: 尝试 stock_zh_a_hist
    try:
        print("   尝试接口3: stock_zh_a_hist...")
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="5",
            adjust="qfq"
        )
        if df is not None and len(df) > 0:
            print(f"   ✅ 接口3成功! 获取到 {len(df)} 条数据")
            return process_data(df, count)
    except Exception as e:
        print(f"   接口3失败: {e}")
    
    print("\n❌ 所有接口都失败了!")
    print("\n💡 建议:")
    print("   1. 检查股票代码是否正确")
    print("   2. 检查网络连接")
    print("   3. 试试其他股票代码，如 600519, 000001, 300750")
    sys.exit(1)


def process_data(df, count):
    """处理数据格式"""
    # 显示原始列名帮助调试
    print(f"   原始列名: {list(df.columns)}")
    
    # 尝试不同的列名映射
    column_mappings = [
        {'day': 'timestamps', 'open': 'open', 'high': 'high', 
         'low': 'low', 'close': 'close', 'volume': 'volume', 'amount': 'amount'},
        {'日期': 'timestamps', '开盘': 'open', '最高': 'high',
         '最低': 'low', '收盘': 'close', '成交量': 'volume', '成交额': 'amount'},
        {'time': 'timestamps', 'open': 'open', 'high': 'high',
         'low': 'low', 'close': 'close', 'volume': 'volume', 'amount': 'amount'},
    ]
    
    for mapping in column_mappings:
        try:
            df_renamed = df.rename(columns=mapping)
            if all(col in df_renamed.columns for col in ['timestamps', 'open', 'high', 'low', 'close', 'volume']):
                df = df_renamed
                break
        except:
            continue
    
    # 检查必需的列
    required = ['timestamps', 'open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"❌ 缺少必要列: {missing}")
        sys.exit(1)
    
    # 如果没有 amount 列，用 close * volume 估算
    if 'amount' not in df.columns:
        df['amount'] = df['close'] * df['volume']
    
    # 取最近 count 条
    df = df.tail(count).reset_index(drop=True)
    
    # 确保列顺序正确
    df = df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']]
    
    print(f"✅ 数据处理完成! 共 {len(df)} 条记录")
    if len(df) > 0:
        print(f"   时间范围: {df['timestamps'].iloc[0]} ~ {df['timestamps'].iloc[-1]}")
        print(f"   价格范围: {df['low'].min():.2f} ~ {df['high'].max():.2f} 元")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='A 股数据获取工具')
    parser.add_argument('--code', type=str, required=True, help='股票代码')
    parser.add_argument('--count', type=int, default=600, help='K 线数量')
    parser.add_argument('--output', type=str, default='', help='输出文件名')
    parser.add_argument('--predict', action='store_true', help='获取后自动预测')
    parser.add_argument('--pred-len', type=int, default=120, help='预测长度')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("📈 A 股数据获取工具")
    print("=" * 70)
    
    # 获取数据
    df = download_stock_data(args.code, args.count)
    
    if len(df) < 200:
        print(f"\n⚠️  数据量较少，预测效果可能不佳")
        print(f"   建议至少有 500+ 条 5分钟 K 线数据")
    
    # 保存数据
    output_file = args.output if args.output else f'{args.code}.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 数据已保存到: {output_file}")
    
    # 如果需要预测
    if args.predict:
        print("\n" + "=" * 70)
        print("🚀 开始预测...")
        print("=" * 70)
        
        lookback = min(len(df) - args.pred_len - 10, 400)
        if lookback < 100:
            print(f"❌ 数据不足! 需要至少 {100 + args.pred_len} 条才能预测")
            return
            
        os.system(f'cd {os.path.dirname(os.path.abspath(__file__))} && '
                  f'../kronos_env/bin/python predict_stock.py '
                  f'--data {output_file} '
                  f'--lookback {lookback} '
                  f'--pred-len {args.pred_len} '
                  f'--stock-name {args.code}')
    
    print("\n✅ 任务完成!")


if __name__ == "__main__":
    main()
