#!/usr/bin/env python3
"""
查看已有的股票预测结果
"""
import os
import json
from PIL import Image

# 预测数据目录
data_dir = "/Users/a1/Documents/code/Kronos/Kronos/examples/yuce"

# 股票代码和名称映射
stocks = {
    '000021': '深科技',
    '002354': '天娱数科', 
    '300207': '欣旺达',
    '600580': '卧龙电驱',
}

print("=" * 70)
print("📊 Kronos 股票预测结果查看器")
print("=" * 70)

for code, name in stocks.items():
    print(f"\n{'='*70}")
    print(f"🚀 {code} {name}")
    print('='*70)
    
    # 查看预测报告
    report_file = os.path.join(data_dir, f'{code}_comprehensive_analysis_report.json')
    if os.path.exists(report_file):
        with open(report_file, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        print(f"\n📈 基本面分析:")
        if 'fundamental_analysis' in report:
            fa = report['fundamental_analysis']
            print(f"   公司名称: {fa.get('company_name', name)}")
            print(f"   业务范围: {', '.join(fa.get('business_areas', []))}")
            print(f"   投资评级: {fa.get('investment_rating', '未知')}")
            print(f"   基本面评分: {fa.get('fundamental_score', 0):.2f}")
        
        print(f"\n🏭 行业分析:")
        if 'sector_analysis' in report:
            sa = report['sector_analysis']
            print(f"   所属行业: {sa.get('industry', '未知')}")
            print(f"   行业热度: {'🔥 热门' if sa.get('is_sector_hot') else '正常'}")
            print(f"   共振评分: {sa.get('resonance_score', 0):.2f}")
        
        print(f"\n💡 最近动态:")
        if 'fundamental_analysis' in report:
            for dev in fa.get('recent_developments', [])[:3]:
                print(f"   • {dev}")
    
    # 预测图片
    img_file = os.path.join(data_dir, f'{code}_optimized_prediction.png')
    if os.path.exists(img_file):
        print(f"\n🖼️  预测图表: {img_file}")
        # 显示图片基本信息
        try:
            with Image.open(img_file) as img:
                print(f"   图片尺寸: {img.size[0]} x {img.size[1]}")
        except:
            pass

print("\n" + "=" * 70)
print("💡 提示: 打开 PNG 文件可以查看详细的价格预测图")
print("=" * 70)
