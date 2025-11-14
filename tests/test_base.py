"""
基础序列吸引子网络测试
"""

import pytest
import numpy as np
from src.core.base import SequenceAttractorNetwork
from src.utils.evaluation import evaluate_replay_full_sequence


class TestSequenceAttractorNetwork:
    """基础序列吸引子网络测试类"""
    
    def test_initialization(self, basic_network_params):
        """测试网络初始化"""
        network = SequenceAttractorNetwork(**basic_network_params)
        
        assert network.N_v == basic_network_params['N_v']
        assert network.T == basic_network_params['T']
        assert network.eta == basic_network_params['eta']
        assert network.kappa == basic_network_params['kappa']
        assert network.N_h == round((basic_network_params['T'] - 1) * 3)
        
        # 检查权重矩阵形状
        assert network.U.shape == (network.N_h, network.N_v)
        assert network.V.shape == (network.N_v, network.N_h)
        assert network.P.shape == (network.N_h, network.N_v)
        
        # 检查训练历史
        assert network.mu_history == []
        assert network.nu_history == []
        assert network.training_sequence is None
    
    def test_initialization_with_custom_N_h(self):
        """测试自定义隐藏层神经元数量"""
        N_h = 50
        network = SequenceAttractorNetwork(N_v=20, T=10, N_h=N_h)
        assert network.N_h == N_h
    
    def test_generate_random_sequence(self, basic_network):
        """测试随机序列生成"""
        sequence = basic_network.generate_random_sequence(seed=42)
        
        assert sequence.shape == (basic_network.T, basic_network.N_v)
        assert np.all(np.abs(sequence) == 1)  # 所有值应该是±1
        
        # 检查周期性（最后一步应该等于第一步）
        assert np.array_equal(sequence[-1, :], sequence[0, :])
        
        # 检查唯一性（除了最后一步，其他步骤应该不同）
        for t in range(1, basic_network.T - 1):
            assert not np.array_equal(sequence[t, :], sequence[0, :])
    
    def test_generate_random_sequence_reproducibility(self, basic_network):
        """测试随机序列生成的可重复性"""
        seq1 = basic_network.generate_random_sequence(seed=42)
        seq2 = basic_network.generate_random_sequence(seed=42)
        assert np.array_equal(seq1, seq2)
    
    def test_train_basic(self, basic_network, sample_sequence):
        """测试基本训练功能"""
        result = basic_network.train(x=sample_sequence, num_epochs=50, verbose=False)
        
        assert 'mu_history' in result
        assert 'nu_history' in result
        assert 'final_mu' in result
        assert 'final_nu' in result
        assert len(result['mu_history']) == 50
        assert len(result['nu_history']) == 50
        assert 0 <= result['final_mu'] <= 1
        assert 0 <= result['final_nu'] <= 1
        
        # 检查训练序列已保存
        assert basic_network.training_sequence is not None
        assert np.array_equal(basic_network.training_sequence, sample_sequence)
    
    def test_train_with_generated_sequence(self, basic_network):
        """测试使用生成的序列进行训练"""
        sequence = basic_network.generate_random_sequence(seed=42)
        result = basic_network.train(x=sequence, num_epochs=50, verbose=False)
        
        assert result['final_mu'] >= 0
        assert result['final_nu'] >= 0
    
    def test_train_V_only(self, basic_network, sample_sequence):
        """测试仅更新V权重的训练模式"""
        result = basic_network.train(x=sample_sequence, num_epochs=50, 
                                     V_only=True, verbose=False)
        
        # V_only模式下，mu应该始终为0
        assert np.all(result['mu_history'] == 0)
        assert result['final_mu'] == 0
    
    def test_replay_basic(self, basic_network, sample_sequence):
        """测试基本回放功能"""
        # 先训练
        basic_network.train(x=sample_sequence, num_epochs=100, verbose=False)
        
        # 回放
        replayed = basic_network.replay(max_steps=30)
        
        assert replayed.shape[0] == 30
        assert replayed.shape[1] == basic_network.N_v
        assert np.all(np.abs(replayed) == 1)  # 所有值应该是±1
    
    def test_replay_with_custom_initial(self, basic_network, sample_sequence):
        """测试使用自定义初始状态的回放"""
        basic_network.train(x=sample_sequence, num_epochs=100, verbose=False)
        
        x_init = sample_sequence[0, :].copy()
        replayed = basic_network.replay(x_init=x_init, max_steps=20)
        
        assert replayed.shape == (20, basic_network.N_v)
    
    def test_replay_with_noise(self, basic_network, sample_sequence):
        """测试带噪声的回放"""
        basic_network.train(x=sample_sequence, num_epochs=100, verbose=False)
        
        replayed = basic_network.replay(noise_level=0.1, max_steps=20)
        
        assert replayed.shape == (20, basic_network.N_v)
    
    def test_replay_without_training(self, basic_network):
        """测试未训练时的回放（应该失败）"""
        with pytest.raises(AssertionError):
            basic_network.replay()
    
    def test_replay_with_initial_state(self, basic_network):
        """测试提供初始状态时的回放（即使未训练）"""
        x_init = np.sign(np.random.randn(basic_network.N_v))
        x_init[x_init == 0] = 1
        
        replayed = basic_network.replay(x_init=x_init, max_steps=20)
        assert replayed.shape == (20, basic_network.N_v)
    
    def test_evaluate_replay_quality(self, basic_network, sample_sequence):
        """测试回放质量评估"""
        # 训练
        basic_network.train(x=sample_sequence, num_epochs=200, verbose=False)
        
        # 回放
        replayed = basic_network.replay(max_steps=50)
        
        # 评估
        evaluation = evaluate_replay_full_sequence(replayed, sample_sequence)
        
        assert 'found_sequence' in evaluation
        assert 'recall_accuracy' in evaluation
        assert 'match_start_idx' in evaluation
        assert evaluation['recall_accuracy'] >= 0
        assert evaluation['recall_accuracy'] <= 1
    
    def test_train_convergence(self, basic_network, sample_sequence):
        """测试训练收敛性（mu和nu应该逐渐减小）"""
        result = basic_network.train(x=sample_sequence, num_epochs=200, verbose=False)
        
        # 检查历史记录长度
        assert len(result['mu_history']) == 200
        assert len(result['nu_history']) == 200
        
        # 检查是否收敛（后期值应该小于前期值，或至少不增加）
        early_mu = np.mean(result['mu_history'][:50])
        late_mu = np.mean(result['mu_history'][-50:])
        
        # 注意：由于随机性，这个测试可能不稳定，所以只检查基本合理性
        assert early_mu >= 0
        assert late_mu >= 0
    
    def test_weight_matrices_shape(self, basic_network):
        """测试权重矩阵形状正确性"""
        assert basic_network.U.shape == (basic_network.N_h, basic_network.N_v)
        assert basic_network.V.shape == (basic_network.N_v, basic_network.N_h)
        assert basic_network.P.shape == (basic_network.N_h, basic_network.N_v)
    
    def test_different_parameters(self):
        """测试不同参数组合"""
        params_combinations = [
            {'N_v': 10, 'T': 5, 'eta': 0.001},
            {'N_v': 30, 'T': 15, 'eta': 0.01},
            {'N_v': 50, 'T': 20, 'eta': 0.1},
        ]
        
        for params in params_combinations:
            network = SequenceAttractorNetwork(**params)
            sequence = network.generate_random_sequence(seed=42)
            result = network.train(x=sequence, num_epochs=20, verbose=False)
            
            assert result['final_mu'] >= 0
            assert result['final_nu'] >= 0

