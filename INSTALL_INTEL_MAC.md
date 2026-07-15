# Kronos Intel Mac 安装指南

## 安装状态：✅ 完成

### 已安装的环境
- **Python**: 3.11.12 (Homebrew 版本)
- **虚拟环境**: `kronos_env/`
- **PyTorch**: 2.2.2 (CPU 模式，Intel Mac 专用)
- **NumPy**: 1.26.4 (兼容 PyTorch 版本)
- **其他依赖**: pandas, matplotlib, tqdm, huggingface_hub, einops, safetensors

### 使用方法

#### 1. 激活虚拟环境
```bash
source kronos_env/bin/activate
```

或者直接使用完整路径执行：
```bash
./kronos_env/bin/python your_script.py
```

#### 2. 运行 Intel Mac 适配的示例
```bash
cd examples
../kronos_env/bin/python prediction_example_intel_mac.py
```

#### 3. 验证安装
```bash
cd examples
../kronos_env/bin/python verify_installation.py
```

### Intel Mac 适配说明

由于 Intel Mac 的限制，进行了以下适配：

1. **设备设置**：强制使用 CPU 模式（即使 MPS 可用，Intel Mac 的 MPS 支持不完善）

2. **Matplotlib 后端**：使用 `Agg` 非交互式后端，避免 GUI 相关问题

3. **结果保存**：预测结果图保存为 PNG 文件，而非直接显示

### 关键文件说明

| 文件 | 说明 |
|------|------|
| `examples/prediction_example_intel_mac.py` | Intel Mac 专用预测示例 |
| `examples/verify_installation.py` | 安装验证脚本 |
| `tests/data/regression_input.csv` | 测试用股票数据 |
| `model/kronos.py` | Kronos 核心模型代码 |

### 注意事项

1. **磁盘空间**：PyTorch 等库占用较大空间，建议保持至少 5GB 可用空间

2. **性能**：Intel Mac 使用 CPU 推理，速度较慢。单条序列预测可能需要数分钟

3. **Python 版本**：不要升级到 Python 3.12+，PyTorch 对 Intel Mac 的支持在新版本中不完整

4. **NumPy 版本**：不要使用 NumPy 2.x，与 PyTorch 2.2.2 不兼容

### 常见问题

**Q: 为什么不使用 MPS？**
A: Intel Mac 的 MPS 支持不完善，容易出现崩溃或计算错误，CPU 模式更稳定。

**Q: 预测很慢怎么办？**
A: 可以减少 `sample_count` 参数（设为 1），或减小 `lookback` 历史数据长度。

**Q: 如何获取更多股票数据？**
A: 可以使用 akshare 库获取 A 股数据，参考 `examples/get_akshare_date_2024-2025_x.py`。
