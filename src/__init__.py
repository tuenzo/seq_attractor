"""
序列吸引子网络 - 重构后的统一接口
"""

# 核心类
from .core import SequenceAttractorNetwork

# 扩展模型
from .models import MultiSequenceAttractorNetwork

# 工具函数
from .utils import (
    visualize_training_results,
    visualize_robustness,
    visualize_multi_sequence_overview,
    evaluate_replay_full_sequence,
    evaluate_replay_frame_matching
)

__all__ = [
    # 核心类
    'SequenceAttractorNetwork',
    # 扩展模型
    'MultiSequenceAttractorNetwork',
    # 工具函数
    'visualize_training_results',
    'visualize_robustness',
    'visualize_multi_sequence_overview',
    'evaluate_replay_full_sequence',
    'evaluate_replay_frame_matching'
]

