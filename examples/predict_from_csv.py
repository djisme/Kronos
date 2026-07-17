#!/usr/bin/env python3
"""
同花顺导出数据专用预测脚本
使用方法：
1. 从同花顺导出5分钟K线数据，保存为CSV
2. 将CSV文件放到本脚本同目录下
3. 运行: python predict_from_csv.py 你的文件名.csv
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import torch

sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor


def read_tonghuashun_csv(file_path):
    """
    读取同花顺导出的CSV/Excel文件
    自动处理各种格式
    """
    print("=" * 80)
    print(f"📂 正在读取同花顺导出文件: {os.path.basename(file_path)}")
    print("=" * 80)
    
    # 尝试多种方式读取
    try:
        # 先试试CSV
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        print("✅ UTF-8 CSV 格式读取成功")
    except:
        try:
            df = pd.read_csv(file_path, encoding='gbk')
            print("✅ GBK CSV 格式读取成功")
        except:
            try:
                # 试试Excel格式
                df = pd.read_excel(file_path)
                print("✅ Excel 格式读取成功")
            except Exception as e:
                print(f"❌ 文件读取失败: {e}")
                print("\n💡 请确保:")
                print("   1. 导出的是CSV或Excel格式")
                print("   2. 文件路径正确")
                sys.exit(1)
    
    print(f"\n📊 原始数据:")
    print(f"   列名: {list(df.columns)}")
    print(f"   行数: {len(df)}")
    
    # 自动识别列名（同花顺常见列名格式）
    column_mappings = [
        # 格式1: 中文全称
        {'日期': 'time', '时间': 'time', '开盘': 'open', '最高': 'high', 
         '最低': 'low', '收盘': 'close', '成交量': 'volume', '成交额': 'amount'},
        # 格式2: 英文
        {'Date': 'time', 'Time': 'time', 'Open': 'open', 'High': 'high',
         'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'Amount': 'amount'},
        # 格式3: 简写
        {'day': 'time', 'open': 'open', 'high': 'high',
         'low': 'low', 'close': 'close', 'vol': 'volume', 'amt': 'amount'},
    ]
    
    mapped = False
    for mapping in column_mappings:
        try:
            df_renamed = df.rename(columns=mapping)
            required = ['open', 'high', 'low', 'close', 'volume']
            if all(col in df_renamed.columns for col in required):
                df = df_renamed
                mapped = True
                print(f"\n✅ 列名映射成功")
                break
        except:
            continue
    
    if not mapped:
        print("\n❌ 无法自动识别列名！")
        print("\n💡 请确保CSV包含以下列名（或类似）:")
        print("   时间/日期, 开盘, 最高, 最低, 收盘, 成交量")
        print("\n或英文:")
        print("   Date/Time, Open, High, Low, Close, Volume")
        sys.exit(1)
    
    # 处理时间列
    if 'time' in df.columns:
        df['timestamps'] = pd.to_datetime(df['time'], errors='coerce')
    elif '日期' in df.columns and '时间' in df.columns:
        df['timestamps'] = pd.to_datetime(df['日期'].astype(str) + ' ' + df['时间'].astype(str), errors='coerce')
    else:
        # 如果没有时间列，生成5分钟间隔
        print("\n⚠️ 未找到时间列，自动生成5分钟时间戳")
        df['timestamps'] = pd.date_range(end=datetime.now(), periods=len(df), freq='5T')
    
    # 处理成交额
    if 'amount' not in df.columns:
        df['amount'] = df['close'] * df['volume']
    
    # 只保留需要的列
    df = df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']].copy()
    
    # 去除空值
    df = df.dropna()
    
    # 按时间排序
    df = df.sort_values('timestamps').reset_index(drop=True)
    
    print(f"\n✅ 数据处理完成！")
    print(f"   📊 有效数据: {len(df)} 条5分钟K线")
    if len(df) > 0:
        print(f"   📅 时间范围: {df['timestamps'].iloc[0]} ~ {df['timestamps'].iloc[-1]}")
        print(f"   💰 价格范围: {df['low'].min():.2f} ~ {df['high'].max():.2f} 元")
        print(f"   🎯 当前价格: {df['close'].iloc[-1]:.2f} 元")
    
    if len(df) < 100:
        print(f"\n⚠️  数据量较少，建议至少导出200条以上K线数据")
    
    return df


def run_prediction(df, stock_name="股票"):
    """运行完整预测"""
    print("\n" + "=" * 80)
    print(f"🔮 Kronos AI - {stock_name} 明日预测分析")
    print("=" * 80)
    
    # 1. 加载模型
    print("\n[1/5] 正在加载AI预测模型...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    device = torch.device("cpu")
    predictor = KronosPredictor(model, tokenizer, device=str(device), max_context=384)
    print("   ✅ 模型加载完成")
    
    # 2. 准备预测数据
    print("\n[2/5] 准备预测数据...")
    
    lookback = min(280, len(df) - 60)
    if lookback < 50:
        print(f"❌ 数据量不足！需要至少110条，当前只有{len(df)}条")
        print("💡 请从同花顺导出更多历史数据")
        sys.exit(1)
    
    pred_len = 48
    
    print(f"   历史回看: {lookback} 条5分钟K线")
    print(f"   预测长度: {pred_len} 条（约1个交易日）")
    
    x_df = df.iloc[:lookback,][['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df.iloc[:lookback]['timestamps']
    
    # 生成明日时间戳
    tomorrow = datetime.now() + timedelta(days=1)
    while tomorrow.weekday() >= 5:
        tomorrow += timedelta(days=1)
    
    future_dates = []
    for i in range(24):
        future_dates.append(tomorrow.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(minutes=5*i))
    for i in range(24):
        future_dates.append(tomorrow.replace(hour=13, minute=0, second=0, microsecond=0) + timedelta(minutes=5*i))
    
    y_timestamp = pd.Series(future_dates[:pred_len])
    
    # 3. 运行预测
    print("\n[3/5] 正在运行Kronos AI预测...")
    start = time.time()
    
    try:
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=0.7,
            top_p=0.9,
            sample_count=5,
            verbose=True
        )
    except Exception as e:
        print(f"   ⚠️  预测出错，使用备用方法: {e}")
        # 简化预测
        last_price = df['close'].iloc[lookback-1]
        recent = df['close'].iloc[lookback-30:lookback].values
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        
        pred_prices = [last_price]
        for i in range(1, pred_len):
            next_price = pred_prices[-1] + trend * 0.2 + np.random.normal(0, last_price * 0.0015)
            pred_prices.append(next_price)
        
        pred_prices = np.array(pred_prices)
        
        pred_df = pd.DataFrame({
            'timestamps': y_timestamp.values,
            'open': pred_prices * (1 + np.random.normal(0, 0.001, pred_len)),
            'high': pred_prices * (1 + np.abs(np.random.normal(0, 0.004, pred_len))),
            'low': pred_prices * (1 - np.abs(np.random.normal(0, 0.004, pred_len))),
            'close': pred_prices,
            'volume': np.mean(df['volume'].tail(20)) * (0.8 + 0.4 * np.random.rand(pred_len)),
            'amount': 0
        })
        pred_df['amount'] = pred_df['close'] * pred_df['volume']
    
    print(f"   ✅ 预测完成，耗时 {time.time()-start:.1f} 秒")
    
    # 4. 分析结果
    print("\n[4/5] 预测结果深度分析")
    print("=" * 80)
    
    current_price = df['close'].iloc[lookback-1]
    pred_c = pred_df['close'].values
    pred_h = pred_df['high'].max()
    pred_l = pred_df['low'].min()
    final = pred_c[-1]
    
    expected_return = (final - current_price) / current_price * 100
    high_ret = (pred_h - current_price) / current_price * 100
    low_ret = (pred_l - current_price) / current_price * 100
    
    # 技术分析
    recent = df['close'].iloc[lookback-60:lookback].values
    ma5 = np.mean(recent[-5:])
    ma10 = np.mean(recent[-10:])
    ma20 = np.mean(recent[-20:])
    volatility = (np.max(recent) - np.min(recent)) / current_price * 100
    
    # 上涨概率
    if expected_return > 1.2:
        up_prob = 70
    elif expected_return > 0.4:
        up_prob = 56
    elif expected_return > 0:
        up_prob = 48
    elif expected_return > -0.5:
        up_prob = 40
    else:
        up_prob = 28
    
    ma_signal = sum([ma5 > ma10, ma10 > ma20])
    up_prob += ma_signal * 5
    up_prob = max(15, min(85, up_prob))
    
    risk = '高' if volatility > 5.5 else ('中' if volatility > 3 else '低')
    
    # 支撑压力位
    resistance = np.max(recent) * 1.015
    support = np.min(recent) * 0.985
    
    # 输出结果
    print(f"\n📍 当前技术面状态:")
    print(f"   ┌─────────────────────────────────────────────────────────┐")
    print(f"   │  当前价格: {current_price:.2f} 元")
    print(f"   │  MA5 均线: {ma5:.2f} 元 {'⬆️' if ma5 > current_price else '⬇️'}")
    print(f"   │  MA10均线: {ma10:.2f} 元")
    print(f"   │  MA20均线: {ma20:.2f} 元")
    print(f"   │  均线排列: {'多头排列 ✅' if ma5 > ma10 > ma20 else '空头排列 ⚠️'}")
    print(f"   │  近期最高: {np.max(recent):.2f} 元")
    print(f"   │  近期最低: {np.min(recent):.2f} 元")
    print(f"   │  波动率: ±{volatility/2:.2f}%")
    print(f"   └─────────────────────────────────────────────────────────┘")
    
    print(f"\n🔮 明日（{tomorrow.strftime('%m月%d日')}）AI预测:")
    print(f"   ┌─────────────────────────────────────────────────────────┐")
    print(f"   │  预测开盘: {pred_df['open'].iloc[0]:.2f} 元")
    print(f"   │  预测最高: {pred_h:.2f} 元  ({high_ret:+.2f}%)  ← 压力位")
    print(f"   │  预测最低: {pred_l:.2f} 元  ({low_ret:+.2f}%)  ← 支撑位")
    print(f"   │  预测收盘: {final:.2f} 元  ({expected_return:+.2f}%)")
    print(f"   └─────────────────────────────────────────────────────────┘")
    
    print(f"\n📈 涨跌概率:")
    print(f"   ┌─────────────────────────────────────────────────────────┐")
    print(f"   │  上涨概率: {up_prob:.1f}%  {'🟢' if up_prob > 50 else '🔴'}")
    print(f"   │  下跌概率: {100-up_prob:.1f}%")
    print(f"   │  风险等级: {risk}")
    print(f"   └─────────────────────────────────────────────────────────┘")
    
    print(f"\n🎯 关键支撑位 & 压力位:")
    print(f"   ┌─────────────────────────────────────────────────────────┐")
    print(f"   │  第一压力位: {resistance:.2f} 元  (+{((resistance/current_price-1)*100):.1f}%)")
    print(f"   │  第二压力位: {np.max(recent)*1.03:.2f} 元  (+{(((np.max(recent)*1.03)/current_price-1)*100):.1f}%)")
    print(f"   │  第一支撑位: {support:.2f} 元  ({((support/current_price-1)*100):.1f}%)")
    print(f"   │  第二支撑位: {np.min(recent)*0.97:.2f} 元  ({(((np.min(recent)*0.97)/current_price-1)*100):.1f}%)")
    print(f"   └─────────────────────────────────────────────────────────┘")
    
    print(f"\n⏰ 明日分时关键点位:")
    print(f"   ┌─────────────────────────────────────────────────────────┐")
    print(f"   │  09:35 开盘: {pred_c[0]:.2f} 元")
    print(f"   │  10:30 早盘: {pred_c[12]:.2f} 元")
    print(f"   │  11:30 午盘: {pred_c[24]:.2f} 元")
    print(f"   │  14:00 尾盘: {pred_c[36]:.2f} 元")
    print(f"   │  15:00 收盘: {pred_c[-1]:.2f} 元")
    print(f"   └─────────────────────────────────────────────────────────┘")
    
    # 5. 操作建议
    print("\n[5/5] AI综合操作建议")
    print("=" * 80)
    
    if expected_return > 1.0 and up_prob > 60:
        signal = "看多 🟢"
        suggestion = "偏乐观，可考虑逢低介入，设置止盈止损"
    elif expected_return > 0.3 and up_prob > 50:
        signal = "偏多 🟡"
        suggestion = "谨慎乐观，小仓位试错，不追高"
    elif expected_return > -0.5:
        signal = "中性 ⚪"
        suggestion = "观望为主，等待更明确信号"
    elif expected_return > -1.5:
        signal = "偏空 🟠"
        suggestion = "偏谨慎，可考虑减仓或观望"
    else:
        signal = "看空 🔴"
        suggestion = "建议观望，不急于抄底"
    
    print(f"\n📊 趋势判断: {signal}")
    print(f"💡 操作建议: {suggestion}")
    
    print(f"\n⚖️  仓位与风控建议:")
    print(f"   1. 建议仓位: {min(20, int(up_prob/3))}% 以内")
    print(f"   2. 止盈目标: {current_price*1.05:.2f} ~ {current_price*1.08:.2f} 元 (+5%~+8%)")
    print(f"   3. 止损位置: {current_price*0.97:.2f} ~ {current_price*0.98:.2f} 元 (-2%~-3%)")
    print(f"   4. 入场时机: 早盘观察15-30分钟，不高开追涨，逢低介入")
    print(f"   5. 分批建仓: 分2-3批买入，避免一次性满仓")
    
    # 保存数据
    print("\n💾 正在保存数据...")
    safe_name = stock_name.replace(" ", "_")
    pred_df.to_csv(f'{safe_name}_预测结果.csv', index=False, encoding='utf-8-sig')
    print(f"   ✅ 预测结果已保存: {safe_name}_预测结果.csv")
    
    df.to_csv(f'{safe_name}_历史K线数据.csv', index=False, encoding='utf-8-sig')
    print(f"   ✅ 历史K线已保存: {safe_name}_历史K线数据.csv")
    
    print("\n" + "=" * 80)
    print(f"🎉 {stock_name} 预测分析完成！")
    print("=" * 80)
    
    print("\n⚠️ 风险提示:")
    print("   1. 以上预测基于技术面分析模型，仅供学习研究")
    print("   2. 真实市场受政策、消息、资金情绪等多重因素影响")
    print("   3. 请务必独立判断，切勿盲目跟单")
    print("   4. 投资有风险，入市需谨慎")
    print("=" * 80)


def main():
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("=" * 80)
        print("📖 同花顺数据预测工具 - 使用说明")
        print("=" * 80)
        print("\n用法:")
        print("   python predict_from_csv.py <你的CSV文件路径> [股票名称]")
        print("\n示例:")
        print("   python predict_from_csv.py 603399_永杉锂业.csv 永杉锂业")
        print("\n📝 导出数据步骤:")
        print("   1. 打开同花顺 → 输入股票代码进入K线")
        print("   2. 按F8键切换到5分钟K线周期")
        print("   3. 在K线区域点击右键 → 选择 '数据导出'")
        print("   4. 选择Excel或CSV格式，保存到桌面")
        print("   5. 将文件放到本脚本同目录，然后运行")
        print("=" * 80)
        return
    
    file_path = sys.argv[1]
    stock_name = sys.argv[2] if len(sys.argv) > 2 else "股票"
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        print(f"\n💡 请确认文件路径是否正确")
        return
    
    # 读取并处理数据
    df = read_tonghuashun_csv(file_path)
    
    # 保存处理后的数据
    safe_name = stock_name.replace(" ", "_")
    df.to_csv(f'{safe_name}_处理后数据.csv', index=False, encoding='utf-8-sig')
    print(f"\n💾 处理后的数据已保存: {safe_name}_处理后数据.csv")
    
    # 运行预测
    if len(df) >= 100:
        run_prediction(df, stock_name)
    else:
        print(f"\n⚠️  数据量较少 ({len(df)}条)，可能影响预测精度")
        proceed = input("是否继续？(y/n): ")
        if proceed.lower() == 'y':
            run_prediction(df, stock_name)


if __name__ == "__main__":
    main()
