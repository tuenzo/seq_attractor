"""
模式重复序列吸引子网络测试
"""

import pytest
import numpy as np
from src.models.pattern import PatternRepetitionNetwork
from src.utils.evaluation import evaluate_replay_full_sequence


class TestPatternRepetitionNetwork:
    """模式重复序列吸引子网络测试类"""
    
    def test_initialization(self, basic_network_params):
        """测试模式重复网络初始化"""
        network = PatternRepetitionNetwork(**basic_network_params)
        
        assert network.N_v == basic_network_params['N_v']
        assert network.T == basic_network_params['T']
        assert network.pattern_info == {}
    
    def test_generate_sequences_with_shared_patterns(self, pattern_network):
        """测试生成包含共享模式的序列"""
        sequences = pattern_network.generate_sequences_with_shared_patterns(
            num_sequences=2,
            seeds=[100, 200],
            verbose=False
        )
        
        assert len(sequences) == 2
        for seq in sequences:
            assert seq.shape == (pattern_network.T, pattern_network.N_v)
            assert np.all(np.abs(seq) == 1)
            assert np.array_equal(seq[-1, :], seq[0, :])
    
    def test_generate_sequences_with_custom_patterns(self, pattern_network):
        """测试使用自定义配置生成共享模式序列"""
        shared_groups = [[0, 1]]
        patterns_per_group = [1]
        positions_per_group = [
            [
                [(3, 5)],  # 序列0的位置
                [(3, 5)]   # 序列1的位置
            ]
        ]
        
        sequences = pattern_network.generate_sequences_with_custom_patterns(
            num_sequences=2,
            shared_groups=shared_groups,
            patterns_per_group=patterns_per_group,
            positions_per_group=positions_per_group,
            seeds=[100, 200],
            verbose=False
        )
        
        assert len(sequences) == 2
        # 检查共享位置的帧是否相同
        assert np.array_equal(sequences[0][3, :], sequences[1][3, :])
        assert np.array_equal(sequences[0][4, :], sequences[1][4, :])
        assert np.array_equal(sequences[0][5, :], sequences[1][5, :])
    
    def test_analyze_pattern_structure(self, pattern_network):
        """测试模式结构分析"""
        # 生成一个普通序列
        sequence = pattern_network.generate_random_sequence(seed=42)
        
        analysis = pattern_network.analyze_pattern_structure(sequence)
        
        assert 'repetition_rate' in analysis
        assert 'unique_frames' in analysis
        assert 0 <= analysis['repetition_rate'] <= 1
        assert analysis['unique_frames'] >= 1
        assert analysis['unique_frames'] <= pattern_network.T
    
    def test_train_shared_pattern_sequences(self, pattern_network):
        """测试训练包含共享模式的序列"""
        sequences = pattern_network.generate_sequences_with_shared_patterns(
            num_sequences=2,
            seeds=[100, 200],
            verbose=False
        )
        
        result = pattern_network.train(
            x=sequences,
            num_epochs=100,
            interleaved=True,
            verbose=False
        )
        
        assert 'mu_history' in result
        assert len(pattern_network.training_sequences) == 2
    
    def test_sequence_overlap_analysis(self, pattern_network):
        """测试序列重叠分析"""
        sequences = pattern_network.generate_multiple_sequences(
            num_sequences=2,
            seeds=[100, 200],
            ensure_unique_across=True,
            verbose=False
        )
        
        overlap = pattern_network.analyze_sequence_overlap(sequences)
        
        assert 'unique_frames' in overlap
        assert 'duplicate_frames' in overlap
        assert 'overlap_rate' in overlap
        # 由于设置了 ensure_unique_across=True，应该没有重复
        assert overlap['duplicate_frames'] == 0
    
    def test_pattern_info_storage(self, pattern_network):
        """测试模式信息存储"""
        sequences = pattern_network.generate_sequences_with_shared_patterns(
            num_sequences=2,
            seeds=[100, 200],
            verbose=False
        )
        
        # 训练后应该存储模式信息
        pattern_network.train(x=sequences, num_epochs=50, verbose=False)
        
        # 检查模式信息是否被记录
        assert len(pattern_network.pattern_info) > 0
    
    def test_verify_non_shared_uniqueness(self, pattern_network):
        """测试非共享区域唯一性验证"""
        sequences = pattern_network.generate_sequences_with_shared_patterns(
            num_sequences=2,
            seeds=[100, 200],
            ensure_unique_non_shared=True,
            verbose=False
        )
        
        # 检查是否正确生成
        assert len(sequences) == 2
        assert pattern_network.pattern_info.get('ensure_unique_non_shared') == True

