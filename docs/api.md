# API 参考

## 核心类

### SequenceAttractorNetwork

基础序列吸引子网络，实现单序列学习。

```python
from src import SequenceAttractorNetwork

network = SequenceAttractorNetwork(
    N_v=50,      # 可见神经元数量
    T=30,        # 序列长度
    N_h=200,     # 隐藏神经元数量（可选，默认3*(T-1)）
    eta=0.001,   # 学习率
    kappa=1.0,   # margin参数
    seed=42      # 随机种子（可选）
)
```

**主要方法：**

| 方法 | 说明 |
|------|------|
| `train(x, num_epochs, verbose, seed, V_only)` | 训练网络 |
| `replay(x_init, noise_level, max_steps)` | 序列回放 |
| `evaluate_replay(xi_replayed)` | 评估回放质量 |
| `test_robustness(noise_levels, num_trials)` | 测试噪声鲁棒性 |
| `generate_random_sequence(seed)` | 生成随机序列 |

---

### MemorySequenceAttractorNetwork

支持多序列和增量学习的网络。

```python
from src import MemorySequenceAttractorNetwork

network = MemorySequenceAttractorNetwork(N_v=50, T=30, N_h=200)

# 多序列训练
sequences = network.generate_multiple_sequences(num_sequences=3)
network.train(x=sequences, num_epochs=500, interleaved=True)

# 增量学习
network.train(x=new_sequence, incremental=True)
```

**额外方法：**

| 方法 | 说明 |
|------|------|
| `generate_multiple_sequences(num_sequences, seeds)` | 生成多个序列 |
| `test_all_memories(verbose)` | 测试所有已学习序列 |
| `get_memory_status()` | 获取记忆状态 |

---

### PatternRepetitionNetwork

支持共享模式的多序列网络。

```python
from src import PatternRepetitionNetwork

network = PatternRepetitionNetwork(N_v=50, T=40, N_h=250)

# 生成共享模式序列
sequences = network.generate_sequences_with_shared_patterns(
    num_sequences=2,
    pattern_config={
        'shared_sequences': [[0, 1]],
        'num_patterns': [1],
        'pattern_positions': [[(15, 15), (15, 15)]]
    }
)
```

**额外方法：**

| 方法 | 说明 |
|------|------|
| `generate_sequences_with_shared_patterns(...)` | 生成共享模式序列 |
| `analyze_pattern_structure(sequence)` | 分析序列模式结构 |
| `analyze_sequence_overlap(sequences)` | 分析序列重叠 |

---

## 工具函数

### 可视化

```python
from src import visualize_training_results, visualize_robustness

# 训练结果可视化
visualize_training_results(network, xi_replayed, eval_result, save_path="result.png")

# 鲁棒性可视化
visualize_robustness(noise_levels, scores, save_path="robustness.png")
```

### 评估

```python
from src import evaluate_replay_full_sequence

result = evaluate_replay_full_sequence(xi_replayed, target_sequence)
# result['found_sequence'] - 是否找到完整序列
# result['recall_accuracy'] - 回放准确率
```

---

## 实验工具

### Figure 5 复现

```python
from src.experiments import run_figure5_experiments, Figure5Config

config = Figure5Config(num_trials=100, num_epochs=500)
results = run_figure5_experiments(config=config, show_images=True)
```

