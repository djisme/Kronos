import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免 Intel Mac GUI 问题
import matplotlib.pyplot as plt
import sys
import torch
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor


def plot_prediction(kline_df, pred_df, save_path='prediction_result.png'):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax1.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax1.set_ylabel('Close Price', fontsize=14)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True)

    ax2.plot(volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax2.plot(volume_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"预测结果图已保存到: {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Kronos 股票预测示例 (Intel Mac 适配版)")
    print("=" * 60)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"使用设备: CPU (Intel Mac 专用)")
    print("=" * 60)

    # 1. Load Model and Tokenizer
    print("\n[1/5] 正在加载模型和分词器...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    print("模型加载完成!")

    # 2. Instantiate Predictor (强制使用 CPU 设备)
    print("\n[2/5] 初始化预测器 (CPU 模式)...")
    device = torch.device("cpu")  # Intel Mac 强制使用 CPU
    predictor = KronosPredictor(model, tokenizer, device=str(device), max_context=512)
    print("预测器初始化完成!")

    # 3. Prepare Data
    print("\n[3/5] 准备数据...")
    # 使用测试数据（确保 Intel Mac 可以直接运行）
    import os
    data_path = "../tests/data/regression_input.csv"
    if not os.path.exists(data_path):
        print(f"警告: 数据文件 {data_path} 不存在!")
        return
    
    df = pd.read_csv(data_path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    print(f"数据加载完成! 共 {len(df)} 条记录")

    lookback = 400
    pred_len = 120

    x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df.loc[:lookback-1, 'timestamps']
    y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

    print(f"历史数据长度: {lookback}")
    print(f"预测长度: {pred_len}")

    # 4. Make Prediction
    print("\n[4/5] 开始预测...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True
    )
    print("预测完成!")

    # 5. Visualize Results
    print("\n[5/5] 可视化结果...")
    print("\n预测数据前5行:")
    print(pred_df.head())

    # Combine historical and forecasted data for plotting
    kline_df = df.loc[:lookback+pred_len-1]

    # visualize
    plot_prediction(kline_df, pred_df)
    
    print("\n" + "=" * 60)
    print("预测任务完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
