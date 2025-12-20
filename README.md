# 序列吸引子网络 (Sequence Attractor Networks)

复现《Learning Sequence Attractors in Recurrent Networks with Hidden Neurons》中的RNN模型，支持多序列学习、增量学习和模式重复。

## 快速开始

### 安装

```bash
git clone https://github.com/tuenzo/seq_attractor.git
cd seq_attractor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 基础使用

```python
from src import SequenceAttractorNetwork

# 创建并训练网络
network = SequenceAttractorNetwork(N_v=50, T=30, N_h=200)
network.train(num_epochs=500, seed=42)

# 回放与评估
xi_replayed = network.replay()
result = network.evaluate_replay(xi_replayed)
print(f"回放成功: {result['found_sequence']}")
```

### 多序列学习

```python
from src import MemorySequenceAttractorNetwork

network = MemorySequenceAttractorNetwork(N_v=50, T=30, N_h=200)
sequences = network.generate_multiple_sequences(num_sequences=3)
network.train(x=sequences, num_epochs=500)
network.test_all_memories()
```

## 项目结构

```
seq_attractor/
├── src/                    # 源代码
│   ├── core/              # 核心网络实现
│   ├── models/            # 扩展模型（多序列、模式重复）
│   ├── utils/             # 工具函数
│   └── experiments/       # 实验脚本
├── tests/                  # 单元测试
├── examples/               # 使用示例
├── scripts/                # CLI工具
├── docs/                   # 文档
└── output/                 # 输出目录
```

## 核心功能

| 类 | 功能 |
|---|------|
| `SequenceAttractorNetwork` | 基础单序列学习 |
| `MemorySequenceAttractorNetwork` | 多序列 + 增量学习 |
| `PatternRepetitionNetwork` | 共享模式序列 |

## 运行测试

```bash
pytest tests/ -v
```

## 运行示例

```bash
python examples/basic_example.py
python examples/multi_incremental_demo.py
```

## 文档

- [快速入门](docs/quickstart.md)
- [API 参考](docs/api.md)
- [示例说明](docs/examples.md)

## 许可证

MIT License
