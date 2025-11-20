# 序列吸引子网络 (Sequence Attractor Networks)

复现《Learning Sequence Attractors in Recurrent Networks with Hidden Neurons》中的RNN模型，并提供多序列、增量训练及模式重复等扩展功能。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-2.3+-orange.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基础使用

```python
from src import SequenceAttractorNetwork

# 创建网络
network = SequenceAttractorNetwork(N_v=50, T=30, N_h=200, eta=0.001)

# 训练
network.train(num_epochs=500, seed=42)

# 回放
xi_replayed = network.replay()

# 评估
result = network.evaluate_replay(xi_replayed)
print(f"成功率: {result['recall_accuracy']*100:.1f}%")
```

### 查看系统配置

```bash
# 检查 BLAS 配置和系统环境
python scripts/check_system.py

# 运行性能测试
python docs/optimization/compare_blas_performance.py
```

---

## 📚 文档

完整文档位于 [`docs/`](docs/) 目录：

| 文档 | 说明 |
|------|------|
| **[docs/README.md](docs/README.md)** | 📖 文档导航 |
| **[docs/optimization/QUICK_SUMMARY.md](docs/optimization/QUICK_SUMMARY.md)** | ⚡ 快速开始（3分钟） |
| **[docs/optimization/OPTIMIZATION_COMPLETE_SUMMARY.md](docs/optimization/OPTIMIZATION_COMPLETE_SUMMARY.md)** | 📊 性能优化完整总结 |
| **[docs/optimization/TROUBLESHOOTING.md](docs/optimization/TROUBLESHOOTING.md)** | 🔧 故障排除指南 |
| **[docs/refactoring/REFACTORING_SUMMARY.md](docs/refactoring/REFACTORING_SUMMARY.md)** | 🏗️ 代码重构说明 |

---

## 🎯 功能特性

### 核心模型

- **`SequenceAttractorNetwork`** - 基础单序列吸引子网络
- **`MemorySequenceAttractorNetwork`** - 多序列 + 增量记忆核心模块
- **`MultiSequenceAttractorNetwork`** - 多序列学习（兼容层）
- **`IncrementalSequenceAttractorNetwork`** - 增量学习（兼容层）
- **`PatternRepetitionNetwork`** - 支持重复模式与共享片段

### 性能优化 ⚡

✅ **跨平台 BLAS 智能配置**
- macOS: Apple Accelerate（自动）
- Linux: Intel MKL / OpenBLAS（自动）
- Windows: MKL（自动）
- GPU: CUDA 检测（待实施）

✅ **性能提升**
- 回放速度：2.33x ⭐
- 总体性能：1.09x
- 准确率：100%（无损）

详见 [性能测试报告](docs/optimization/PERFORMANCE_COMPARISON_REPORT.md)

---

## 📖 示例

### 基础示例

```bash
# 基础使用与可视化
python examples/basic_example.py

# 多序列 + 增量训练
python examples/multi_incremental_demo.py

# 模式重复网络
python examples/pattern_repetition_shared_demo.py
```

### 复现 Figure 5（拆分模式）

```bash
# 标准拆分模式
python scripts/sa_cli.py fig5-split --trials 100 --epochs 500 --show

# 使用多进程加速
python scripts/sa_cli.py fig5-split --trials 100 --epochs 500 --show --workers -1

# 注入单步重复模式
python scripts/sa_cli.py fig5-split --with-repetition --show --workers -1
```

**参数说明**:
- `--trials`: 每个采样点的尝试次数（默认 100）
- `--epochs`: 训练轮数（默认 500）
- `--show`: 显示图形
- `--workers`: 并行进程数（`-1`=自动，`0`=全部核心，`1`=串行）
- `--T-values`: 扫描的序列长度（逗号分隔）
- `--Nh-values`: 扫描的隐藏层规模（逗号分隔）

---

## 🧪 测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_base.py -v

# 查看测试覆盖率
pytest --cov=src tests/
```

---

## 📊 项目结构

```
seq_attractor/
├── README.md                    # 本文件
├── requirements.txt             # 依赖包
├── pytest.ini                   # 测试配置
├── .gitignore                   # Git 忽略规则
├── src/                         # 源代码
│   ├── core/                    # 核心网络实现
│   ├── models/                  # 扩展模型
│   ├── experiments/             # 实验脚本
│   └── utils/                   # 工具函数
├── scripts/                     # 命令行工具
│   ├── sa_cli.py                # CLI 入口
│   ├── check_system.py          # 系统检查
│   ├── benchmark.py             # 性能测试
│   └── backup_version.sh        # 版本备份
├── tests/                       # 单元测试
├── examples/                    # 使用示例
├── docs/                        # 📚 文档目录
│   ├── README.md                # 文档导航
│   ├── optimization/            # 性能优化文档
│   ├── refactoring/             # 重构说明文档
│   └── results/                 # 实验结果
├── benchmark_results/           # 性能测试结果
└── store/                       # 旧代码存档
```

---

## 🌟 性能优化

本项目已实施第一个优化（BLAS 配置），详见：
- **[优化完整总结](docs/optimization/OPTIMIZATION_COMPLETE_SUMMARY.md)**
- **[性能测试报告](docs/optimization/PERFORMANCE_COMPARISON_REPORT.md)**

### 当前优化

✅ **优化 1: 跨平台 BLAS 配置**
- 状态: 已完成
- 效果: 1.09x 总体，2.33x 回放
- 分支: `optimize/blas`

### 后续计划

📋 **优化 2: 混合精度** → 预期 1.5-2x  
📋 **优化 3: 多进程并行** → 预期 4-8x  
📋 **优化 4: GPU 加速** → 预期 10-30x  
📋 **优化 5-7: 更多优化** → 总计 20-100x  

详见 [优化计划](docs/optimization/OPTIMIZATION_PLAN.md)

---

## 🔧 环境配置

### macOS (推荐)
```bash
# 使用 Apple Accelerate 框架（自动配置）
python scripts/check_system.py
```

### Linux
```bash
# Intel CPU: 自动使用 MKL
# AMD CPU: 自动使用 OpenBLAS
pip install numpy

# 或使用 conda
conda install numpy "libblas=*=*mkl"  # Intel
conda install numpy "libblas=*=*openblas"  # AMD
```

### GPU (NVIDIA)
```bash
# 安装 CuPy（根据 CUDA 版本）
pip install cupy-cuda12x  # CUDA 12.x
pip install cupy-cuda11x  # CUDA 11.x

# 检查 GPU
python scripts/check_system.py
```

遇到问题？查看 [故障排除指南](docs/optimization/TROUBLESHOOTING.md)

---

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

### 开发流程
1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交改动 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 提交 Pull Request

### Git 工作流
详见 [版本管理指南](docs/optimization/VERSION_MANAGEMENT_SETUP.md)

---

## 📄 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

---

## 📧 联系方式

有问题或建议？

1. 查看 [文档](docs/README.md)
2. 运行 `python scripts/check_system.py` 诊断环境
3. 查看 [故障排除指南](docs/optimization/TROUBLESHOOTING.md)
4. 提交 Issue

---

## 🙏 致谢

- 原论文: *Learning Sequence Attractors in Recurrent Networks with Hidden Neurons*
- 优化灵感: NumPy/SciPy 社区
- 测试框架: pytest

---

**⭐ 如果这个项目对你有帮助，请给个 Star！**
