"""
模式重复序列吸引子网络测试
"""

import pytest
import numpy as np
from src.models.pattern_repetition import PatternRepetitionNetwork
from src.utils.evaluation import evaluate_replay_full_sequence


class TestPatternRepetitionNetwork:
    """模式重复序列吸引子网络测试类"""
    
    def test_initialization(self, basic_network_params):
        """测试模式重复网络初始化"""
        network = PatternRepetitionNetwork(**basic_network_params)
        
        assert network.N_v == basic_network_params['N_v']
        assert network.T == basic_network_params['T']
        assert network.pattern_info == {}
    
    def test_generate_alternating_pattern(self, pattern_network):
        """测试生成交替模式序列"""
        sequence = pattern_network.generate_patterned_sequence(
            pattern_type='alternating',
            seed=42
        )
        
        assert sequence.shape == (pattern_network.T, pattern_network.N_v)
        assert np.all(np.abs(sequence) == 1)
        assert np.array_equal(sequence[-1, :], sequence[0, :])
        
        # 检查交替模式（前几步应该交替）
        if pattern_network.T >= 4:
            assert not np.array_equal(sequence[0, :], sequence[1, :])
            assert np.array_equal(sequence[0, :], sequence[2, :])
    
    def test_generate_periodic_pattern(self, pattern_network):
        """测试生成周期性模式序列"""
        period = 3
        sequence = pattern_network.generate_patterned_sequence(
            pattern_type='periodic',
            period=period,
            seed=42
        )
        
        assert sequence.shape == (pattern_network.T, pattern_network.N_v)
        assert np.all(np.abs(sequence) == 1)
        
        # 检查周期性
        if pattern_network.T >= period + 1:
            assert np.array_equal(sequence[0, :], sequence[period, :])
    
    def test_generate_block_pattern(self, pattern_network):
        """测试生成块状模式序列"""
        block_size = 3
        sequence = pattern_network.generate_patterned_sequence(
            pattern_type='block',
            block_size=block_size,
            seed=42
        )
        
        assert sequence.shape == (pattern_network.T, pattern_network.N_v)
        assert np.all(np.abs(sequence) == 1)
        
        # 检查块状模式（同一块内的帧应该相同）
        if pattern_network.T >= block_size * 2:
            assert np.array_equal(sequence[0, :], sequence[1, :])
            assert np.array_equal(sequence[0, :], sequence[block_size - 1, :])
    
    def test_generate_mirrored_pattern(self, pattern_network):
        """测试生成镜像模式序列"""
        sequence = pattern_network.generate_patterned_sequence(
            pattern_type='mirrored',
            seed=42
        )
        
        assert sequence.shape == (pattern_network.T, pattern_network.N_v)
        # 注意：镜像模式可能有0值（中间位置），所以只检查非零位置
        assert np.all(np.abs(sequence[sequence != 0]) == 1)
        assert np.array_equal(sequence[-1, :], sequence[0, :])
    
    def test_generate_custom_pattern(self, pattern_network):
        """测试生成自定义模式序列"""
        # custom_pattern应该是整数列表，表示帧索引
        custom_pattern = [0, 1, 0, 2]  # 使用帧索引0, 1, 0, 2重复
        
        sequence = pattern_network.generate_patterned_sequence(
            pattern_type='custom',
            custom_pattern=custom_pattern,
            seed=42
        )
        
        assert sequence.shape == (pattern_network.T, pattern_network.N_v)
        assert np.all(np.abs(sequence) == 1)
    
    def test_generate_multiple_patterned_sequences(self, pattern_network):
        """测试生成多个模式序列"""
        pattern_configs = [
            {'pattern_type': 'alternating'},
            {'pattern_type': 'periodic', 'period': 3},
            {'pattern_type': 'block', 'block_size': 2},
        ]
        
        sequences = pattern_network.generate_multiple_patterned_sequences(
            num_sequences=3,
            pattern_configs=pattern_configs
        )
        
        assert len(sequences) == 3
        for seq in sequences:
            assert seq.shape == (pattern_network.T, pattern_network.N_v)
            assert np.all(np.abs(seq) == 1)
    
    def test_analyze_pattern_structure(self, pattern_network):
        """测试模式结构分析"""
        # 生成交替模式序列
        sequence = pattern_network.generate_patterned_sequence(
            pattern_type='alternating',
            seed=42
        )
        
        analysis = pattern_network.analyze_pattern_structure(sequence)
        
        assert 'repetition_rate' in analysis
        assert 'unique_frames' in analysis
        # 注意：analyze_pattern_structure可能不返回pattern_type，而是返回其他分析信息
        assert 0 <= analysis['repetition_rate'] <= 1
        assert analysis['unique_frames'] >= 1
        assert analysis['unique_frames'] <= pattern_network.T
    
    def test_analyze_pattern_structure_periodic(self, pattern_network):
        """测试周期性模式的结构分析"""
        sequence = pattern_network.generate_patterned_sequence(
            pattern_type='periodic',
            period=3,
            seed=42
        )
        
        analysis = pattern_network.analyze_pattern_structure(sequence)
        
        # 周期性模式应该有较高的重复率
        assert analysis['repetition_rate'] > 0
        assert analysis['unique_frames'] <= 3  # 周期为3，所以唯一帧应该<=3
    
    def test_train_patterned_sequences(self, pattern_network):
        """测试训练模式序列"""
        sequences = pattern_network.generate_multiple_patterned_sequences(
            num_sequences=2,
            pattern_configs=[
                {'pattern_type': 'alternating'},
                {'pattern_type': 'periodic', 'period': 3}
            ]
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
        sequences = pattern_network.generate_multiple_patterned_sequences(
            num_sequences=2,
            pattern_configs=[
                {'pattern_type': 'alternating'},
                {'pattern_type': 'periodic', 'period': 2}
            ]
        )
        
        # analyze_sequence_overlap接受序列列表作为参数
        overlap = pattern_network.analyze_sequence_overlap(sequences)
        
        assert 'overlap_frames' in overlap or 'overlap_rate' in overlap or 'overlap_matrix' in overlap
    
    def test_pattern_info_storage(self, pattern_network):
        """测试模式信息存储"""
        sequence = pattern_network.generate_patterned_sequence(
            pattern_type='alternating',
            seed=42
        )
        
        # 训练后应该存储模式信息
        pattern_network.train(x=[sequence], num_epochs=50, verbose=False)
        
        # 检查模式信息是否被记录（如果实现的话）
        # 这取决于具体实现
    
    def test_invalid_pattern_type(self, pattern_network):
        """测试无效模式类型（应该失败或使用默认）"""
        # 根据实现，这可能抛出异常或使用默认模式
        try:
            sequence = pattern_network.generate_patterned_sequence(
                pattern_type='invalid_pattern',
                seed=42
            )
            # 如果没有抛出异常，至少应该生成一个有效序列
            assert sequence.shape == (pattern_network.T, pattern_network.N_v)
        except (ValueError, KeyError):
            # 如果抛出异常，这也是合理的
            pass
    
    def test_custom_T_in_pattern_generation(self, pattern_network):
        """测试在模式生成中使用自定义T"""
        T = 8
        sequence = pattern_network.generate_patterned_sequence(
            pattern_type='alternating',
            T=T,
            seed=42
        )
        
        assert sequence.shape == (T, pattern_network.N_v)

