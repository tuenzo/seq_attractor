"""
测试配置和共享fixtures
"""

import os

os.environ.setdefault("NPY_DISABLE_MAC_OS_ACCELERATE", "1")

import pytest
import numpy as np
from src.core.base import SequenceAttractorNetwork
from src.models.memory import MemorySequenceAttractorNetwork
from src.models.multi_sequence import MultiSequenceAttractorNetwork
from src.models.incremental import IncrementalSequenceAttractorNetwork
from src.models.pattern_repetition import PatternRepetitionNetwork


@pytest.fixture
def basic_network_params():
    """基础网络参数"""
    return {
        'N_v': 20,
        'T': 10,
        'eta': 0.01,
        'kappa': 1.0
    }


@pytest.fixture
def small_network_params():
    """小型网络参数（用于快速测试）"""
    return {
        'N_v': 10,
        'T': 5,
        'eta': 0.01,
        'kappa': 1.0
    }


@pytest.fixture
def basic_network(basic_network_params):
    """基础序列吸引子网络实例"""
    return SequenceAttractorNetwork(**basic_network_params)


@pytest.fixture
def multi_sequence_network(basic_network_params):
    """多序列网络实例"""
    return MultiSequenceAttractorNetwork(**basic_network_params)


@pytest.fixture
def incremental_network(basic_network_params):
    """增量学习网络实例"""
    return IncrementalSequenceAttractorNetwork(**basic_network_params)


@pytest.fixture
def memory_network(basic_network_params):
    """统一记忆网络实例"""
    return MemorySequenceAttractorNetwork(**basic_network_params)


@pytest.fixture
def pattern_network(basic_network_params):
    """模式重复网络实例"""
    return PatternRepetitionNetwork(**basic_network_params)


@pytest.fixture
def sample_sequence(basic_network_params):
    """生成示例训练序列"""
    np.random.seed(42)
    N_v = basic_network_params['N_v']
    T = basic_network_params['T']
    
    x = np.sign(np.random.randn(T, N_v))
    x[x == 0] = 1
    
    # 确保序列是周期性的（最后一步等于第一步）
    x[T - 1, :] = x[0, :]
    
    return x


@pytest.fixture
def multiple_sequences(basic_network_params):
    """生成多个示例序列"""
    np.random.seed(42)
    N_v = basic_network_params['N_v']
    T = basic_network_params['T']
    
    sequences = []
    for seed in [100, 200, 300]:
        np.random.seed(seed)
        x = np.sign(np.random.randn(T, N_v))
        x[x == 0] = 1
        x[T - 1, :] = x[0, :]
        sequences.append(x)
    
    return sequences

