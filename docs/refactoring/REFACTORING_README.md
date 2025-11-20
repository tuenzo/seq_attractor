# 代码重构说明

## 重构目标

本次重构旨在优化项目代码结构，消除重复代码，建立清晰的模块化组织。

## 新的目录结构

```
seq_attractor/
├── src/                          # 源代码目录
│   ├── core/                     # 核心网络类
│   │   ├── __init__.py
│   │   └── base.py               # 基础序列吸引子网络
│   ├── models/                   # 扩展模型
│   │   ├── __init__.py
│   │   └── multi_sequence.py     # 多序列模型
│   └── utils/                    # 工具函数
│       ├── __init__.py
│       ├── visualization.py      # 可视化函数
│       └── evaluation.py        # 评估函数
├── examples/                     # 示例脚本
├── tests/                        # 测试文件
└── README.md
```

## 主要改进

### 1. 模块化组织
- **核心类** (`src/core/`): 基础网络实现，使用向量化计算优化性能
- **扩展模型** (`src/models/`): 多序列、增量学习等扩展功能
- **工具函数** (`src/utils/`): 可视化和评估等通用功能

### 2. 消除重复代码
- 统一的基础类实现
- 共享的可视化和评估函数
- 清晰的继承关系

### 3. 统一接口
通过 `src/__init__.py` 提供统一的导入接口：

```python
from src import (
    SequenceAttractorNetwork,
    MultiSequenceAttractorNetwork,
    visualize_training_results,
    visualize_robustness
)
```

## 使用示例

### 基础使用
```python
from src import SequenceAttractorNetwork, visualize_training_results

# 创建网络
network = SequenceAttractorNetwork(N_v=50, T=30, eta=0.01)

# 训练
train_results = network.train(num_epochs=300, seed=42)

# 回放
xi_replayed = network.replay()

# 评估
eval_result = network.evaluate_replay(xi_replayed)

# 可视化
visualize_training_results(network, xi_replayed, eval_result)
```

### 多序列使用
```python
from src import MultiSequenceAttractorNetwork, visualize_multi_sequence_overview

# 创建网络
network = MultiSequenceAttractorNetwork(N_v=50, T=30, eta=0.01)

# 生成多个序列
sequences = network.generate_multiple_sequences(num_sequences=3, seeds=[100, 200, 300])

# 训练
network.train(x=sequences, num_epochs=400, interleaved=True)

# 可视化
visualize_multi_sequence_overview(network)
```

## 迁移指南

### 旧代码
```python
from SequenceAttractorNetwork import SequenceAttractorNetwork, visualize_results
```

### 新代码
```python
from src import SequenceAttractorNetwork, visualize_training_results
```

注意：函数名从 `visualize_results` 改为 `visualize_training_results` 以更清晰地表达功能。

## 待完成工作

- [ ] 创建增量学习模型 (`src/models/incremental.py`)
- [ ] 创建模式重复模型 (`src/models/pattern_repetition.py`)
- [ ] 更新所有示例脚本使用新结构
- [ ] 添加单元测试
- [ ] 更新文档

## 分支信息

本次重构在 `refactor-code-structure` 分支中进行，完成后可以合并到主分支。

