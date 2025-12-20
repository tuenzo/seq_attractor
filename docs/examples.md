# 示例说明

## 示例文件

| 文件 | 说明 |
|------|------|
| `basic_example.py` | 基础使用：单序列训练、回放、评估 |
| `multi_incremental_demo.py` | 多序列训练和增量学习 |
| `pattern_repetition_shared_demo.py` | 共享模式序列网络 |
| `run_figure5_split_modes.py` | Figure 5 实验复现 |

## 运行示例

```bash
# 基础示例
python examples/basic_example.py

# 多序列示例
python examples/multi_incremental_demo.py

# 模式重复示例
python examples/pattern_repetition_shared_demo.py

# Figure 5 复现（使用多进程加速）
python examples/run_figure5_split_modes.py
```

## 快速代码示例

### 基础训练

```python
from src import SequenceAttractorNetwork

network = SequenceAttractorNetwork(N_v=50, T=30, N_h=200)
network.train(num_epochs=500, seed=42)
xi_replayed = network.replay()
result = network.evaluate_replay(xi_replayed)
print(f"成功: {result['found_sequence']}")
```

### 多序列学习

```python
from src import MemorySequenceAttractorNetwork

network = MemorySequenceAttractorNetwork(N_v=50, T=30, N_h=200)
sequences = network.generate_multiple_sequences(num_sequences=3)
network.train(x=sequences, num_epochs=500)
network.test_all_memories()
```

### 增量学习

```python
from src import MemorySequenceAttractorNetwork

network = MemorySequenceAttractorNetwork(N_v=50, T=30, N_h=200)

# 学习第一个序列
seq1 = network.generate_random_sequence(seed=100)
network.train(x=seq1, num_epochs=300)

# 增量学习新序列（保持旧记忆）
seq2 = network.generate_random_sequence(seed=200)
network.train(x=seq2, num_epochs=300, incremental=True)

network.test_all_memories()
```

