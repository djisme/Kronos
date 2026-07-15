#!/usr/bin/env python3
"""
Kronos 股票预测 - 自定义股票版本 (Intel Mac 适配)

使用方法:
    # 使用自己的 CSV 文件
    python predict_stock.py --data 你的股票数据.csv

    # 指定历史长度和预测长度
    python predict_stock.py --data 你的股票数据.csv --lookback 400 --pred-len 120

    # 指定输出图片名称
    python predict_stock.py --data 600519.csv --output 茅台预测结果.png
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import torch
import argparse
import os

sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor


def plot_prediction(kline_df, pred_df, stock_name="股票", save_path='prediction_result.png'):
    """绘制预测结果图"""
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = '真实价格'
    sr_pred_close.name = "预测价格"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = '真实成交量'
    sr_pred_volume.name = "预测成交量"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f'{stock_name} - 价格预测结果', fontsize=16, y=0.98)

    ax1.plot(close_df['真实价格'], label='真实价格', color='blue', linewidth=1.5)
    ax1.plot(close_df['预测价格'], label='预测价格', color='red', linewidth=1.5)
    ax1.set_ylabel('价格 (元)', fontsize=12)
    ax1.legend(loc='lower left', fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(volume_df['真实成交量'], label='真实成交量', color='blue', linewidth=1.5)
    ax2.plot(volume_df['预测成交量'], label='预测成交量', color='red', linewidth=1.5)
    ax2.set_ylabel('成交量', fontsize=12)
    ax2.set_xlabel('时间', fontsize=12)
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 预测结果图已保存到: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Kronos 股票预测 (Intel Mac 版)')
    parser.add_argument('--data', type=str, default='../tests/data/regression_input.csv',
                        help='股票数据 CSV 文件路径')
    parser.add_argument('--lookback', type=int, default=400,
                        help='历史数据长度 (默认: 400)')
    parser.add_argument('--pred-len', type=int, default=120,
                        help='预测长度 (默认: 120，约 10 小时的 5 分钟 K 线)')
    parser.add_argument('--stock-name', type=str, default='',
                        help='股票名称 (用于图表标题)')
    parser.add_argument('--output', type=str, default='',
                        help='输出图片路径 (默认: 根据股票名自动生成)')
    parser.add_argument('--sample-count', type=int, default=1,
                        help='采样次数，越大越稳定但越慢 (默认: 1)')
    
    args = parser.parse_args()

    print("=" * 70)
    print("🚀 Kronos 股票预测 (Intel Mac 适配版)")
    print("=" * 70)
    print(f"📊 PyTorch 版本: {torch.__version__}")
    print(f"💻 使用设备: CPU (Intel Mac 专用)")
    print(f"📁 数据文件: {args.data}")
    print(f"⏳ 历史长度: {args.lookback} 条 K 线")
    print(f"🔮 预测长度: {args.pred_len} 条 K 线")
    print("=" * 70)

    # 1. 加载模型和分词器
    print("\n[1/5] 正在加载模型和分词器...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    print("✅ 模型加载完成!")

    # 2. 初始化预测器
    print("\n[2/5] 初始化预测器 (CPU 模式)...")
    device = torch.device("cpu")
    predictor = KronosPredictor(model, tokenizer, device=str(device), max_context=512)
    print("✅ 预测器初始化完成!")

    # 3. 准备数据
    print("\n[3/5] 准备数据...")
    if not os.path.exists(args.data):
        print(f"❌ 错误: 数据文件 {args.data} 不存在!")
        print("\n💡 提示: 请先准备你的股票 CSV 文件，需要包含以下列:")
        print("   timestamps, open, high, low, close, volume, amount")
        sys.exit(1)
    
    df = pd.read_csv(args.data)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    
    # 检查数据长度
    min_required = args.lookback + args.pred_len
    if len(df) < min_required:
        print(f"❌ 错误: 数据不足! 需要至少 {min_required} 条记录")
        print(f"   当前数据: {len(df)} 条")
        print(f"   请减小 --lookback 或 --pred-len 参数")
        sys.exit(1)
    
    print(f"✅ 数据加载完成! 共 {len(df)} 条记录")
    print(f"   时间范围: {df['timestamps'].min()} ~ {df['timestamps'].max()}")

    # 提取股票名称（用于显示）
    stock_name = args.stock_name if args.stock_name else os.path.basename(args.data).replace('.csv', '')
    
    # 准备输入数据
    x_df = df.loc[:args.lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df.loc[:args.lookback-1, 'timestamps']
    y_timestamp = df.loc[args.lookback:args.lookback+args.pred_len-1, 'timestamps']

    # 4. 开始预测
    print(f"\n[4/5] 开始预测 {stock_name}...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=args.pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=args.sample_count,
        verbose=True
    )
    print("✅ 预测完成!")

    # 5. 可视化和保存结果
    print("\n[5/5] 可视化结果...")
    print("\n📈 预测数据前5行:")
    print(pred_df[['open', 'high', 'low', 'close', 'volume']].head())
    
    # 预测统计
    print(f"\n📊 预测统计:")
    print(f"   预测起始价格: {pred_df['close'].iloc[0]:.2f} 元")
    print(f"   预测结束价格: {pred_df['close'].iloc[-1]:.2f} 元")
    print(f"   预测涨跌: {(pred_df['close'].iloc[-1] - pred_df['close'].iloc[0])/pred_df['close'].iloc[0]*100:+.2f}%")

    # Combine historical and forecasted data for plotting
    kline_df = df.loc[:args.lookback+args.pred_len-1]

    # 确定输出文件名
    output_file = args.output if args.output else f'{stock_name}_预测结果.png'
    
    # 绘制并保存
    plot_prediction(kline_df, pred_df, stock_name=stock_name, save_path=output_file)
    
    # 保存预测结果为 CSV
    csv_output = output_file.replace('.png', '.csv')
    pred_df.to_csv(csv_output)
    print(f"✅ 预测数据已保存到: {csv_output}")
    
    print("\n" + "=" * 70)
    print(f"🎉 {stock_name} 预测任务完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
