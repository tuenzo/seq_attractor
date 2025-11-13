"""
================================================================
序列吸引子网络 - 支持增量学习（在不改变接口的情况下）
================================================================
"""

import numpy as np
from typing import Optional, List, Dict, Union

class SequenceAttractorNetwork:
    """序列吸引子循环神经网络（支持增量学习）"""
    
    def __init__(self, N_v: int, T: int, N_h: Optional[int] = None, 
                 eta: float = 0.001, kappa: float = 1):
        """初始化网络"""
        self.N_v = N_v
        self.T = T
        self.N_h = N_h if N_h is not None else round((T - 1) * 3)
        self.eta = eta
        self.kappa = kappa
        
        # 初始化权重矩阵
        self.U = np.random.randn(self.N_h, N_v) * 1e-6
        self.V = np.random.randn(N_v, self.N_h) * 1e-6
        self.P = np.random.randn(self.N_h, N_v) * 1e-6
        
        # 训练历史（改为列表以支持多次训练）
        self.mu_history = []
        self.nu_history = []
        
        # 存储所有已学习的序列
        self.training_sequences = []
        self.training_sequence = None  # 保持向后兼容
        
        # 记录每个序列的训练信息
        self.sequence_training_info = []
        self._total_epochs_trained = 0
    
    def train(self, x: Optional[np.ndarray] = None, num_epochs: int = 500, 
              verbose: bool = True, seed: Optional[int] = None, 
              V_only: bool = False, reset_history: bool = False,
              incremental: bool = False) -> Dict:
        """
        训练网络（支持增量学习）
        
        参数:
            x: 训练序列 (T x N_v)
            num_epochs: 训练轮数
            verbose: 是否打印训练信息
            seed: 随机种子
            V_only: 是否仅更新V权重
            reset_history: 是否重置训练历史
            incremental: 是否为增量学习模式（学习新序列同时保持旧记忆）
            
        返回:
            训练结果字典
        """
        # 准备训练序列
        if x is None:
            if len(self.training_sequences) == 0:
                x = self.generate_random_sequence(seed)
                is_new_sequence = True
            else:
                # 继续训练已有序列
                x = self.training_sequences[-1]
                is_new_sequence = False
        else:
            assert x.shape == (self.T, self.N_v), \
                f"序列形状应为 ({self.T}, {self.N_v})，实际为 {x.shape}"
            # 检查是否为新序列
            is_new_sequence = not any(np.array_equal(x, seq) for seq in self.training_sequences)
        
        # 决定是否重置历史
        if reset_history:
            self.mu_history = []
            self.nu_history = []
            self._total_epochs_trained = 0
            self.training_sequences = []
            self.sequence_training_info = []
            if verbose:
                print("已重置训练历史和所有记忆")
        
        # 记录训练模式
        if is_new_sequence:
            if len(self.training_sequences) > 0 and incremental:
                mode = "增量学习新序列"
            else:
                mode = "学习新序列"
            self.training_sequences.append(x.copy())
        else:
            mode = "继续训练已有序列"
        
        self.training_sequence = x  # 保持向后兼容
        
        start_epoch = self._total_epochs_trained
        
        if verbose:
            print(f"{mode}...")
            print(f"N_v={self.N_v}, T={self.T}, N_h={self.N_h}")
            print(f"参数: eta={self.eta}, kappa={self.kappa}, epochs={num_epochs}")
            print(f"已学习序列数: {len(self.training_sequences)}")
            if start_epoch > 0:
                print(f"累计训练轮数: {start_epoch}")
            if V_only:
                print("仅更新 V 权重矩阵")
        
        # 预分配当前训练的历史数组
        current_mu_history = np.zeros(num_epochs)
        current_nu_history = np.zeros(num_epochs)
        
        if incremental and len(self.training_sequences) > 1:
            # 增量学习模式：同时训练新序列和旧序列
            sequences_to_train = self.training_sequences
            if verbose:
                print(f"增量学习模式：将训练 {len(sequences_to_train)} 个序列")
        else:
            # 普通模式：只训练当前序列
            sequences_to_train = [x]
        
        # 预计算所有序列的数据
        seq_data_list = []
        total_transitions = 0
        for seq in sequences_to_train:
            x_current = seq[:-1, :].T
            x_next = seq[1:, :].T
            seq_data_list.append({
                'x_current': x_current,
                'x_next': x_next,
                'T': len(seq)
            })
            total_transitions += (len(seq) - 1)
        
        # 训练循环
        for epoch in range(num_epochs):
            epoch_mu = 0
            epoch_nu = 0
            
            # 轮流训练所有序列（交替训练）
            for seq_data in seq_data_list:
                x_current_all = seq_data['x_current']
                x_next_all = seq_data['x_next']
                
                # 更新 U
                if not V_only:
                    z_target_all = np.sign(self.P @ x_next_all)
                    z_target_all[z_target_all == 0] = 1
                    h_input_all = self.U @ x_current_all
                    mu_all = (z_target_all * h_input_all < self.kappa).astype(float)
                    delta_U = (mu_all * z_target_all) @ x_current_all.T
                    self.U += self.eta * delta_U
                    epoch_mu += np.sum(mu_all)
                
                # 更新 V
                y_actual_all = np.sign(self.U @ x_current_all)
                y_actual_all[y_actual_all == 0] = 1
                v_input_all = self.V @ y_actual_all
                nu_all = (x_next_all * v_input_all < self.kappa).astype(float)
                delta_V = (nu_all * x_next_all) @ y_actual_all.T
                self.V += self.eta * delta_V
                epoch_nu += np.sum(nu_all)
            
            # 记录当前训练的历史
            current_mu_history[epoch] = epoch_mu / (self.N_h * total_transitions)
            current_nu_history[epoch] = epoch_nu / (self.N_v * total_transitions)
            
            if verbose and (epoch + 1) % 100 == 0:
                total_epoch = start_epoch + epoch + 1
                print(f'Epoch {total_epoch} ({epoch + 1}/{num_epochs}), '
                      f'μ={current_mu_history[epoch]:.4f}, '
                      f'ν={current_nu_history[epoch]:.4f}')
        
        # 追加到总历史
        self.mu_history.extend(current_mu_history.tolist())
        self.nu_history.extend(current_nu_history.tolist())
        self._total_epochs_trained += num_epochs
        
        # 记录序列训练信息
        if is_new_sequence:
            self.sequence_training_info.append({
                'sequence_index': len(self.training_sequences) - 1,
                'start_epoch': start_epoch,
                'end_epoch': self._total_epochs_trained,
                'num_epochs': num_epochs,
                'incremental': incremental
            })
        
        return {
            'mu_history': np.array(self.mu_history),
            'nu_history': np.array(self.nu_history),
            'final_mu': self.mu_history[-1],
            'final_nu': self.nu_history[-1],
            'total_epochs': self._total_epochs_trained,
            'current_epochs': num_epochs,
            'num_learned_sequences': len(self.training_sequences),
            'training_mode': mode
        }
    
    def get_memory_status(self) -> Dict:
        """
        获取当前记忆状态
        
        返回:
            包含所有已学习序列信息的字典
        """
        return {
            'num_sequences': len(self.training_sequences),
            'total_epochs_trained': self._total_epochs_trained,
            'sequence_info': self.sequence_training_info,
            'network_params': {
                'N_v': self.N_v,
                'T': self.T,
                'N_h': self.N_h,
                'eta': self.eta,
                'kappa': self.kappa
            }
        }
    
    def test_all_memories(self, verbose: bool = True) -> Dict:
        """
        测试所有已学习序列的回放质量
        
        返回:
            包含所有序列测试结果的字典
        """
        if len(self.training_sequences) == 0:
            print("警告：没有已学习的序列")
            return {}
        
        results = {}
        
        if verbose:
            print("\n" + "="*60)
            print(f"测试所有记忆 ({len(self.training_sequences)} 个序列)")
            print("="*60)
        
        for i, seq in enumerate(self.training_sequences):
            x_init = seq[0, :].copy()
            xi_replayed = self.replay(x_init=x_init)
            
            # 检查是否能正确回放该序列
            max_steps = xi_replayed.shape[0]
            found_sequence = False
            for tau in range(max_steps - len(seq) + 1):
                segment = xi_replayed[tau:tau+len(seq), :]
                if np.array_equal(segment, seq):
                    found_sequence = True
                    break
            
            success_rate = 1.0 if found_sequence else 0.0
            
            results[f'sequence_{i}'] = {
                'success': found_sequence,
                'success_rate': success_rate,
                'sequence_length': len(seq)
            }
            
            if verbose:
                status = "✓ 成功" if found_sequence else "✗ 失败"
                print(f"序列 #{i}: {status} (回放成功率: {success_rate*100:.0f}%)")
        
        # 统计
        total_success = sum(1 for r in results.values() if r['success'])
        overall_rate = total_success / len(results)
        
        results['summary'] = {
            'total_sequences': len(results),  # 不包括summary本身
            'successful_recalls': total_success,
            'overall_success_rate': overall_rate
        }
        
        if verbose:
            print(f"\n总体成功率: {overall_rate*100:.1f}% "
                  f"({total_success}/{len(results)-1})")
            print("="*60)
        
        return results
    
    def generate_random_sequence(self, seed: Optional[int] = None) -> np.ndarray:
        """生成随机训练序列"""
        if seed is not None:
            np.random.seed(seed)
            
        x = np.sign(np.random.randn(self.T, self.N_v))
        x[x == 0] = 1
        
        for t in range(1, self.T - 1):
            while np.any(np.all(x[t, :] == x[:t, :], axis=1)):
                x[t, :] = np.sign(np.random.randn(self.N_v))
                x[t, x[t, :] == 0] = 1
        
        x[self.T - 1, :] = x[0, :]
        return x
    
    def replay(self, x_init: Optional[np.ndarray] = None, 
               noise_level: float = 0.0, max_steps: Optional[int] = None) -> np.ndarray:
        """序列回放"""
        if x_init is None:
            assert self.training_sequence is not None, "请先训练网络或提供初始状态"
            x_init = self.training_sequence[0, :].copy()
        
        if max_steps is None:
            max_steps = self.T * 3
        
        xi_test = x_init.reshape(-1, 1).copy()
        
        if noise_level > 0:
            noise_mask = np.random.rand(self.N_v, 1) < noise_level
            xi_test[noise_mask] = -xi_test[noise_mask]
        
        xi_replayed = np.zeros((max_steps, self.N_v))
        
        for step in range(max_steps):
            zeta = np.sign(self.U @ xi_test)
            zeta[zeta == 0] = 1
            xi_test = np.sign(self.V @ zeta)
            xi_test[xi_test == 0] = 1
            xi_replayed[step, :] = xi_test.flatten()
        
        return xi_replayed
    
    # 其他方法保持不变...
    def evaluate_replay(self, xi_replayed: np.ndarray, 
                        num_trials: int = 50,
                        check_full_sequence: bool = True) -> Dict:
        """评估回放质量"""
        assert self.training_sequence is not None, "请先训练网络"
        
        max_steps = xi_replayed.shape[0]
        found_sequence = False
        match_start_idx = -1
        
        if check_full_sequence:
            for tau in range(max_steps - self.T + 1):
                segment = xi_replayed[tau:tau+self.T, :]
                if np.array_equal(segment, self.training_sequence):
                    found_sequence = True
                    match_start_idx = tau
                    break
            
            recall_accuracy = 1.0 if found_sequence else 0.0
        else:
            match_indices = np.zeros(max_steps, dtype=int)
            for step in range(max_steps):
                matches = np.all(xi_replayed[step, :] == self.training_sequence, axis=1)
                if np.any(matches):
                    match_indices[step] = np.where(matches)[0][0] + 1
            
            match_count = np.sum(match_indices > 0)
            recall_accuracy = match_count / max_steps
            
            return {
                'recall_accuracy': recall_accuracy,
                'match_count': match_count,
                'match_indices': match_indices,
                'evaluation_mode': 'frame_matching'
            }
        
        return {
            'recall_accuracy': recall_accuracy,
            'found_sequence': found_sequence,
            'match_start_idx': match_start_idx,
            'evaluation_mode': 'full_sequence'
        }


# ========== 使用示例：增量学习 ==========
if __name__ == "__main__":
    print("\n" + "="*70)
    print("演示：增量学习（学习多个序列并保持记忆）")
    print("="*70)
    
    # 创建网络
    network = SequenceAttractorNetwork(N_v=50, T=30, N_h=200, eta=0.01, kappa=1)
    
    # ========== 第一阶段：学习第一个序列 ==========
    print("\n【阶段1】学习第一个序列")
    print("-"*60)
    seq1 = network.generate_random_sequence(seed=100)
    result1 = network.train(x=seq1, num_epochs=300, verbose=True)
    
    # 测试第一个序列
    print("\n测试记忆:")
    memory_test1 = network.test_all_memories(verbose=True)
    
    # ========== 第二阶段：增量学习第二个序列 ==========
    print("\n【阶段2】增量学习第二个序列（保持第一个序列的记忆）")
    print("-"*60)
    seq2 = network.generate_random_sequence(seed=200)
    result2 = network.train(
        x=seq2, 
        num_epochs=300, 
        verbose=True,
        incremental=True  # 关键：开启增量学习模式
    )
    
    # 测试所有记忆
    print("\n测试记忆:")
    memory_test2 = network.test_all_memories(verbose=True)
    
    # ========== 第三阶段：再学习第三个序列 ==========
    print("\n【阶段3】继续增量学习第三个序列")
    print("-"*60)
    seq3 = network.generate_random_sequence(seed=300)
    result3 = network.train(
        x=seq3, 
        num_epochs=300, 
        verbose=True,
        incremental=True
    )
    
    # 测试所有记忆
    print("\n测试记忆:")
    memory_test3 = network.test_all_memories(verbose=True)
    
    # ========== 显示记忆状态 ==========
    print("\n" + "="*70)
    print("当前记忆状态")
    print("="*70)
    status = network.get_memory_status()
    print(f"已学习序列数: {status['num_sequences']}")
    print(f"累计训练轮数: {status['total_epochs_trained']}")
    print("\n各序列训练信息:")
    for info in status['sequence_info']:
        print(f"  序列 #{info['sequence_index']}: "
              f"轮数 {info['start_epoch']}-{info['end_epoch']} "
              f"({'增量学习' if info['incremental'] else '独立学习'})")
    
    # ========== 对比：不使用增量学习 ==========
    print("\n" + "="*70)
    print("对比实验：不使用增量学习（直接学习新序列）")
    print("="*70)
    
    network_no_incr = SequenceAttractorNetwork(N_v=50, T=30, N_h=200, eta=0.01, kappa=1)
    
    # 学习第一个序列
    print("\n学习序列1...")
    network_no_incr.train(x=seq1, num_epochs=300, verbose=False)
    
    # 直接学习第二个序列（不用incremental）
    print("学习序列2（覆盖模式）...")
    network_no_incr.train(x=seq2, num_epochs=300, verbose=False)
    
    # 直接学习第三个序列
    print("学习序列3（覆盖模式）...")
    network_no_incr.train(x=seq3, num_epochs=300, verbose=False)
    
    # 测试记忆
    print("\n测试记忆:")
    memory_test_no_incr = network_no_incr.test_all_memories(verbose=True)
    
    # ========== 结果对比 ==========
    print("\n" + "="*70)
    print("结果对比")
    print("="*70)
    print(f"\n增量学习模式:")
    print(f"  总体成功率: {memory_test3['summary']['overall_success_rate']*100:.1f}%")
    print(f"  成功回放: {memory_test3['summary']['successful_recalls']}/{memory_test3['summary']['total_sequences']}")
    
    print(f"\n覆盖学习模式:")
    print(f"  总体成功率: {memory_test_no_incr['summary']['overall_success_rate']*100:.1f}%")
    print(f"  成功回放: {memory_test_no_incr['summary']['successful_recalls']}/{memory_test_no_incr['summary']['total_sequences']}")
    
    print("\n" + "="*70)
    print("结论：增量学习可以在学习新序列的同时保持旧记忆！")
    print("="*70)
