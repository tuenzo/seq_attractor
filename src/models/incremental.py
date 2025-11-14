"""
================================================================
增量学习序列吸引子网络（兼容层）
================================================================

该模块保留原有导入路径，实际功能由 MemorySequenceAttractorNetwork 提供。
"""

from __future__ import annotations

from .memory import MemorySequenceAttractorNetwork


class IncrementalSequenceAttractorNetwork(MemorySequenceAttractorNetwork):
    """兼容增量学习接口的别名类，功能由 MemorySequenceAttractorNetwork 实现。"""


__all__ = ["IncrementalSequenceAttractorNetwork"]

