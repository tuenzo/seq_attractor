"""
多序列吸引子网络测试
"""

import pytest
import numpy as np
from src.models.multi_sequence import MultiSequenceAttractorNetwork
from src.utils.evaluation import evaluate_replay_full_sequence


class TestMultiSequenceAttractorNetwork:
    """多序列吸引子网络测试类"""
    
    def test_initialization(self, basic_network_params):
        """测试多序列网络初始化"""
        network = MultiSequenceAttractorNetwork(**basic_network_params)
        
        assert network.N_v == basic_network_params['N_v']
        assert network.T == basic_network_params['T']
        assert network.training_sequences == []
        assert network.num_sequences == 0
    
    def test_generate_random_sequence_with_length(self, multi_sequence_network):
        """测试生成指定长度的随机序列"""
        T = 8
        sequence = multi_sequence_network.generate_random_sequence_with_length(T, seed=42)
        
        assert sequence.shape == (T, multi_sequence_network.N_v)
        assert np.all(np.abs(sequence) == 1)
        assert np.array_equal(sequence[-1, :], sequence[0, :])
    
    def test_generate_multiple_sequences(self, multi_sequence_network):
        """测试生成多个序列"""
        sequences = multi_sequence_network.generate_multiple_sequences(
            num_sequences=3, 
            seeds=[100, 200, 300]
        )
        
        assert len(sequences) == 3
        for seq in sequences:
            assert seq.shape == (multi_sequence_network.T, multi_sequence_network.N_v)
            assert np.all(np.abs(seq) == 1)
            assert np.array_equal(seq[-1, :], seq[0, :])
    
    def test_generate_multiple_sequences_unique(self, multi_sequence_network):
        """测试生成的多个序列是唯一的"""
        sequences = multi_sequence_network.generate_multiple_sequences(
            num_sequences=3,
            seeds=[100, 200, 300],
            ensure_unique_across=True
        )
        
        # 检查序列之间是否不同
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                assert not np.array_equal(sequences[i], sequences[j])
    
    def test_generate_multiple_sequences_with_custom_T(self, multi_sequence_network):
        """测试使用自定义长度生成多个序列"""
        T = 8
        sequences = multi_sequence_network.generate_multiple_sequences(
            num_sequences=2,
            T=T,
            seeds=[100, 200]
        )
        
        for seq in sequences:
            assert seq.shape == (T, multi_sequence_network.N_v)
    
    def test_train_single_sequence(self, multi_sequence_network, sample_sequence):
        """测试训练单个序列"""
        result = multi_sequence_network.train(
            x=sample_sequence, 
            num_epochs=50, 
            verbose=False
        )
        
        assert 'mu_history' in result
        assert len(multi_sequence_network.training_sequences) == 1
        assert multi_sequence_network.num_sequences == 1
    
    def test_train_multiple_sequences_interleaved(self, multi_sequence_network, multiple_sequences):
        """测试交替训练多个序列"""
        result = multi_sequence_network.train(
            x=multiple_sequences,
            num_epochs=100,
            interleaved=True,
            verbose=False
        )
        
        assert len(multi_sequence_network.training_sequences) == len(multiple_sequences)
        assert multi_sequence_network.num_sequences == len(multiple_sequences)
        assert 'mu_history' in result
    
    def test_train_multiple_sequences_batch(self, multi_sequence_network, multiple_sequences):
        """测试批量训练多个序列"""
        result = multi_sequence_network.train(
            x=multiple_sequences,
            num_epochs=100,
            interleaved=False,
            verbose=False
        )
        
        assert len(multi_sequence_network.training_sequences) == len(multiple_sequences)
        assert multi_sequence_network.num_sequences == len(multiple_sequences)
    
    def test_replay_specific_sequence(self, multi_sequence_network, multiple_sequences):
        """测试回放特定序列"""
        multi_sequence_network.train(
            x=multiple_sequences,
            num_epochs=150,
            interleaved=True,
            verbose=False
        )
        
        # 回放第一个序列（使用sequence_index参数）
        replayed = multi_sequence_network.replay(
            sequence_index=0,
            max_steps=40
        )
        
        assert replayed.shape[0] == 40
        assert replayed.shape[1] == multi_sequence_network.N_v
        
        # 评估回放质量
        evaluation = evaluate_replay_full_sequence(
            replayed, 
            multiple_sequences[0]
        )
        assert 'found_sequence' in evaluation
    
    def test_replay_all_sequences(self, multi_sequence_network, multiple_sequences):
        """测试回放所有序列"""
        multi_sequence_network.train(
            x=multiple_sequences,
            num_epochs=150,
            interleaved=True,
            verbose=False
        )
        
        # 手动回放所有序列（因为replay_all_sequences方法不存在）
        all_replays = []
        for i in range(len(multiple_sequences)):
            replay = multi_sequence_network.replay(sequence_index=i, max_steps=30)
            all_replays.append(replay)
        
        assert len(all_replays) == len(multiple_sequences)
        for replay in all_replays:
            assert replay.shape == (30, multi_sequence_network.N_v)
    
    def test_get_sequence_info(self, multi_sequence_network, multiple_sequences):
        """测试获取序列信息（通过直接访问属性）"""
        multi_sequence_network.train(
            x=multiple_sequences,
            num_epochs=100,
            interleaved=True,
            verbose=False
        )
        
        # 直接访问训练序列
        assert len(multi_sequence_network.training_sequences) == len(multiple_sequences)
        assert multi_sequence_network.training_sequences[0].shape == multiple_sequences[0].shape
        assert np.array_equal(multi_sequence_network.training_sequences[0], multiple_sequences[0])
    
    def test_sequence_uniqueness_check(self, multi_sequence_network):
        """测试序列唯一性检查"""
        sequences = multi_sequence_network.generate_multiple_sequences(
            num_sequences=3,
            seeds=[100, 200, 300],
            ensure_unique_across=True
        )
        
        # 验证序列确实不同
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                assert not np.array_equal(sequences[i], sequences[j])
    
    def test_train_empty_sequences_list(self, multi_sequence_network):
        """测试训练空序列列表（应该失败）"""
        # 空列表会导致除零错误或其他错误
        with pytest.raises((ValueError, ZeroDivisionError, AssertionError)):
            multi_sequence_network.train(x=[], num_epochs=10, verbose=False)
    
    def test_replay_invalid_sequence_idx(self, multi_sequence_network, multiple_sequences):
        """测试回放无效序列索引（应该失败）"""
        multi_sequence_network.train(
            x=multiple_sequences,
            num_epochs=50,
            verbose=False
        )
        
        with pytest.raises(AssertionError):
            multi_sequence_network.replay(sequence_index=999, max_steps=20)

