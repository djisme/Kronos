#!/usr/bin/env python3
"""
A 股数据获取工具 (支持 akshare)

使用方法:
    # 获取股票 600519 的 5分钟 K 线（最近 500 条）
    python get_stock_data.py --code 600519
    
    # 指定获取条数
    python get_stock_data.py --code 000001 --count 1000
    
    # 获取后直接预测
    python get_stock_data.py --code 600519 --predict
"""

import argparse
import sys
import os
import pandas as pd
from datetime import datetime, timedelta

try:
    import akshare as ak
    HAS_AKSHARE = True
except ImportError:
    HAS_AKSHARE = False


def download_stock_data(stock_code, count=1000):
    """
    获取 A 股 5分钟 K 线数据
    
    Args:
        stock_code: 股票代码，如 '600519'
        count: 获取的 K 线数量，默认 1000 条
    """
    if not HAS_AKSHARE:
        print("❌ 错误: 未安装 akshare")
        print("请先安装: pip install akshare")
        sys.exit(1)
    
    print(f"🔍 正在获取股票 {stock_code} 的 5分钟 K 线数据...")
    
    try:
        # 同花顺接口获取 5分钟 K 线
        df = ak.stock_zh_a_minute(
            symbol=stock_code,
            period='5',
            adjust='qfq'  # 前复权
        )
        
        # 取最近的 count 条数据
        df = df.tail(count).reset_index(drop=True)
        
        # 重命名列以匹配 Kronos 要求
        df = df.rename(columns={
            'day': 'timestamps',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'amount': 'amount'
        })
        
        # 确保列顺序正确
        df = df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']]
        
        print(f"✅ 获取成功! 共 {len(df)} 条数据")
        print(f"   时间范围: {df['timestamps'].iloc[0]} ~ {df['timestamps'].iloc[-1]}")
        print(f"   价格范围: {df['low'].min():.2f} ~ {df['high'].max():.2f} 元")
        
        return df
        
    except Exception as e:
        print(f"❌ 获取数据失败: {e}")
        print("\n💡 可能的原因:")
        print("   1. 股票代码不正确 (上证 60xxxx，深证 00xxxx，创业板 30xxxx)")
        print("   2. 网络连接问题")
        print("   3. akshare API 接口变更")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='A 股数据获取工具')
    parser.add_argument('--code', type=str, required=True,
                        help='股票代码 (如 600519, 000001, 300750)')
    parser.add_argument('--count', type=int, default=600,
                        help='获取 K 线数量 (默认: 600，约 25 天的 5分钟 K 线)')
    parser.add_argument('--output', type=str, default='',
                        help='输出文件名 (默认: 股票代码.csv)')
    parser.add_argument('--predict', action='store_true',
                        help='获取数据后自动运行预测')
    parser.add_argument('--pred-len', type=int, default=120,
                        help='预测长度 (默认: 120，约 10 小时)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("📈 A 股数据获取工具")
    print("=" * 70)
    
    # 获取数据
    df = download_stock_data(args.code, args.count)
    
    # 保存数据
    output_file = args.output if args.output else f'{args.code}.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 数据已保存到: {output_file}")
    
    # 如果需要预测
    if args.predict:
        print("\n" + "=" * 70)
        print("🚀 开始预测...")
        print("=" * 70)
        
        # 调用预测脚本
        lookback = min(len(df) - args.pred_len - 10, 400)  # 确保有足够数据
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
