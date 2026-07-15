#!/usr/bin/env python3
"""
Kronos 环境安装验证脚本 (Intel Mac 专用)
运行此脚本来验证所有依赖是否正确安装
"""

import sys
print("=" * 60)
print("Kronos 环境安装验证")
print("=" * 60)

# 1. 检查 Python 版本
print(f"\n1. Python 版本: {sys.version}")
if sys.version_info >= (3, 10) and sys.version_info < (3, 12):
    print("   ✓ Python 版本兼容 (3.10 - 3.11)")
else:
    print(f"   ⚠ 建议使用 Python 3.10 或 3.11，当前版本: {sys.version_info[:3]}")

# 2. 检查核心依赖
try:
    import numpy
    print(f"\n2. NumPy 版本: {numpy.__version__}")
    print("   ✓ NumPy 安装成功")
except ImportError:
    print("\n   ✗ NumPy 未安装!")
    sys.exit(1)

try:
    import pandas
    print(f"\n3. Pandas 版本: {pandas.__version__}")
    print("   ✓ Pandas 安装成功")
except ImportError:
    print("\n   ✗ Pandas 未安装!")
    sys.exit(1)

try:
    import torch
    print(f"\n4. PyTorch 版本: {torch.__version__}")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
    print(f"   MPS 可用: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
    print(f"   当前设备: CPU (Intel Mac 专用)")
    print("   ✓ PyTorch 安装成功")
except ImportError:
    print("\n   ✗ PyTorch 未安装!")
    sys.exit(1)

# 5. 检查其他依赖
dependencies = {
    'matplotlib': 'matplotlib',
    'tqdm': 'tqdm',
    'huggingface_hub': 'huggingface_hub',
    'einops': 'einops',
    'safetensors': 'safetensors',
}

print("\n5. 其他依赖检查:")
all_ok = True
for pkg, name in dependencies.items():
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f"   ✓ {name}: {version}")
    except ImportError:
        print(f"   ✗ {name}: 未安装")
        all_ok = False

if not all_ok:
    print("\n   ⚠ 部分依赖缺失，请运行: pip install -r requirements.txt")

# 6. 检查模型导入
print("\n6. Kronos 模型导入检查:")
try:
    sys.path.insert(0, '..')
    from model import Kronos, KronosTokenizer, KronosPredictor
    print("   ✓ Kronos 模型导入成功!")
except Exception as e:
    print(f"   ⚠ Kronos 导入警告: {e}")

print("\n" + "=" * 60)
print("安装验证完成!")
if all_ok:
    print("✓ 所有核心依赖已正确安装，可以运行 Kronos!")
    print("\n运行示例命令:")
    print("  cd examples")
    print("  ../kronos_env/bin/python prediction_example_intel_mac.py")
else:
    print("⚠ 请先安装缺失的依赖后再运行模型")
print("=" * 60)
