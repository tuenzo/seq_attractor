"""
================================================================
序列吸引子网络 - 多序列扩展版（继承实现，修正评估方法）
在原有 SequenceAttractorNetwork 基础上通过继承添加多序列功能
保持原有代码完全不变，向后兼容
================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List, Union
import os
from datetime import datetime

# 导入原始类
from SAN_tensor_1 import SequenceAttractorNetwork


# ========== 多序列扩展类（通过继承）==========
class MultiSequenceAttractorNetwork(SequenceAttractorNetwork):
    """
    多序列吸引子网络（继承扩展版）
    在原有 SequenceAttractorNetwork 基础上添加多序列学习功能
    完全向后兼容，所有原有功能保持不变
    """
    
    def __init__(self, N_v: int, T: int, N_h: Optional[int] = None, 
                 eta: float = 0.001, kappa: float = 1):
        """
        初始化网络（调用父类构造函数）
        """
        super().__init__(N_v, T, N_h, eta, kappa)
        
        # 多序列专用属性
        self.training_sequences = []  # 存储多个序列
        self.num_sequences = 0
    
    # ... (其他方法保持不变，包括 generate_random_sequence_with_length, 
    #      generate_multiple_sequences, train, _train_multiple_sequences,
    #      _train_interleaved, _train_batch, replay)
    
    def generate_random_sequence_with_length(self, T: int, seed: Optional[int] = None) -> np.ndarray:
        """生成指定长度的随机序列（扩展方法）"""
        if seed is not None:
            np.random.seed(seed)
        x = np.sign(np.random.randn(T, self.N_v))
        x[x == 0] = 1
        for t in range(1, T - 1):
            while np.any(np.all(x[t, :] == x[:t, :], axis=1)):
                x[t, :] = np.sign(np.random.randn(self.N_v))
                x[t, x[t, :] == 0] = 1
        x[T - 1, :] = x[0, :]
        return x
    
    def generate_multiple_sequences(self, num_sequences: int, 
                                    seeds: Optional[List[int]] = None,
                                    T: Optional[int] = None) -> List[np.ndarray]:
        """生成多个随机序列（新方法）"""
        sequences = []
        if seeds is None:
            seeds = list(range(num_sequences))
        seq_length = T if T is not None else self.T
        for i, seed in enumerate(seeds[:num_sequences]):
            seq = self.generate_random_sequence_with_length(T=seq_length, seed=seed)
            sequences.append(seq)
        return sequences
    
    def train(self, x: Optional[Union[np.ndarray, List[np.ndarray]]] = None, 
              num_epochs: int = 500, 
              verbose: bool = True, 
              seed: Optional[int] = None, 
              V_only: bool = False,
              interleaved: bool = True) -> Dict:
        """训练网络（重写方法，支持单序列和多序列）"""
        if x is None or isinstance(x, np.ndarray):
            result = super().train(x, num_epochs, verbose, seed, V_only)
            if self.training_sequence is not None:
                self.training_sequences = [self.training_sequence]
                self.num_sequences = 1
            return result
        elif isinstance(x, list):
            return self._train_multiple_sequences(x, num_epochs, verbose, V_only, interleaved)
        else:
            raise ValueError("x 必须是 None、np.ndarray 或 List[np.ndarray]")
    
    def _train_multiple_sequences(self, sequences: List[np.ndarray], 
                                  num_epochs: int = 500, 
                                  V_only: bool = False,
                                  verbose: bool = True,
                                  interleaved: bool = True) -> Dict:
        """多序列训练的内部实现（新方法）"""
        for i, seq in enumerate(sequences):
            assert seq.shape[1] == self.N_v, \
                f"序列 {i} 的可见层维度应为 {self.N_v}，实际为 {seq.shape[1]}"
        
        self.training_sequences = sequences
        self.num_sequences = len(sequences)
        
        if verbose:
            print(f"开始多序列训练... N_v={self.N_v}, N_h={self.N_h}")
            print(f"参数: eta={self.eta}, kappa={self.kappa}, epochs={num_epochs}")
            print(f"序列数量: {self.num_sequences}")
            seq_lengths = [len(seq) for seq in sequences]
            print(f"序列长度: {seq_lengths}")
            print(f"训练模式: {'交替训练' if interleaved else '批量训练'}")
            if V_only:
                print("仅更新 V 权重矩阵")
        
        self.mu_history = np.zeros(num_epochs)
        self.nu_history = np.zeros(num_epochs)
        
        if interleaved:
            self._train_interleaved(sequences, num_epochs, V_only, verbose)
        else:
            self._train_batch(sequences, num_epochs, V_only, verbose)
        
        return {
            'mu_history': self.mu_history,
            'nu_history': self.nu_history,
            'final_mu': self.mu_history[-1],
            'final_nu': self.nu_history[-1],
            'num_sequences': self.num_sequences
        }
    
    def _train_interleaved(self, sequences: List[np.ndarray], 
                          num_epochs: int, V_only: bool, verbose: bool):
        """交替训练策略"""
        seq_data = []
        total_transitions = 0
        
        for seq in sequences:
            T = len(seq)
            x_current = seq[:-1, :].T
            x_next = seq[1:, :].T
            seq_data.append({
                'x_current': x_current,
                'x_next': x_next,
                'T': T
            })
            total_transitions += (T - 1)
        
        for epoch in range(num_epochs):
            epoch_mu = 0
            epoch_nu = 0
            
            for k, data in enumerate(seq_data):
                x_current_all = data['x_current']
                x_next_all = data['x_next']
                
                if not V_only:
                    z_target_all = np.sign(self.P @ x_next_all)
                    z_target_all[z_target_all == 0] = 1
                    h_input_all = self.U @ x_current_all
                    mu_all = (z_target_all * h_input_all < self.kappa).astype(float)
                    delta_U = (mu_all * z_target_all) @ x_current_all.T
                    self.U += self.eta * delta_U
                    epoch_mu += np.sum(mu_all)
                
                y_actual_all = np.sign(self.U @ x_current_all)
                y_actual_all[y_actual_all == 0] = 1
                v_input_all = self.V @ y_actual_all
                nu_all = (x_next_all * v_input_all < self.kappa).astype(float)
                delta_V = (nu_all * x_next_all) @ y_actual_all.T
                self.V += self.eta * delta_V
                epoch_nu += np.sum(nu_all)
            
            self.mu_history[epoch] = epoch_mu / (self.N_h * total_transitions)
            self.nu_history[epoch] = epoch_nu / (self.N_v * total_transitions)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, '
                      f'μ={self.mu_history[epoch]:.4f}, '
                      f'ν={self.nu_history[epoch]:.4f}')
    
    def _train_batch(self, sequences: List[np.ndarray], 
                    num_epochs: int, V_only: bool, verbose: bool):
        """批量训练策略"""
        all_x_current = []
        all_x_next = []
        
        for seq in sequences:
            all_x_current.append(seq[:-1, :].T)
            all_x_next.append(seq[1:, :].T)
        
        x_current_all = np.hstack(all_x_current)
        x_next_all = np.hstack(all_x_next)
        total_transitions = x_current_all.shape[1]
        
        for epoch in range(num_epochs):
            if not V_only:
                z_target_all = np.sign(self.P @ x_next_all)
                z_target_all[z_target_all == 0] = 1
                h_input_all = self.U @ x_current_all
                mu_all = (z_target_all * h_input_all < self.kappa).astype(float)
                delta_U = (mu_all * z_target_all) @ x_current_all.T
                self.U += self.eta * delta_U
                total_mu = np.sum(mu_all)
            else:
                total_mu = 0
            
            y_actual_all = np.sign(self.U @ x_current_all)
            y_actual_all[y_actual_all == 0] = 1
            v_input_all = self.V @ y_actual_all
            nu_all = (x_next_all * v_input_all < self.kappa).astype(float)
            delta_V = (nu_all * x_next_all) @ y_actual_all.T
            self.V += self.eta * delta_V
            total_nu = np.sum(nu_all)
            
            self.mu_history[epoch] = total_mu / (self.N_h * total_transitions)
            self.nu_history[epoch] = total_nu / (self.N_v * total_transitions)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, '
                      f'μ={self.mu_history[epoch]:.4f}, '
                      f'ν={self.nu_history[epoch]:.4f}')
    
    def replay(self, x_init: Optional[np.ndarray] = None, 
               noise_level: float = 0.0, 
               max_steps: Optional[int] = None,
               sequence_index: int = 0) -> np.ndarray:
        """序列回放（扩展方法，支持多序列）"""
        if x_init is None:
            if len(self.training_sequences) > 0:
                assert sequence_index < len(self.training_sequences), \
                    f"序列索引 {sequence_index} 超出范围"
                x_init = self.training_sequences[sequence_index][0, :].copy()
            elif self.training_sequence is not None:
                x_init = self.training_sequence[0, :].copy()
            else:
                raise AssertionError("请先训练网络或提供初始状态")
        
        return super().replay(x_init, noise_level, max_steps)
    
    # ========== 修正后的评估方法 ==========
    
    def evaluate_replay(self, xi_replayed: Optional[np.ndarray] = None,
                    sequence_index: Optional[int] = None,
                    num_trials: int = 50,
                    noise_level: float = 0.0,
                    verbose: bool = False,
                    include_frame_matching: bool = True) -> Dict:
        """
        评估回放质量（修正版 - 完整序列匹配 + 逐帧匹配可视化）
        
        参数:
            xi_replayed: 回放序列（如果为None，则进行多次试验）
            sequence_index: 与哪个训练序列比较
            num_trials: 多次试验的次数
            noise_level: 噪声水平
            verbose: 是否打印详细信息
            include_frame_matching: 是否包含逐帧匹配信息（用于可视化）
            
        返回:
            评估指标字典
        """
        if len(self.training_sequences) == 0:
            raise AssertionError("请先训练网络")
        
        # 多次试验模式（推荐）
        if xi_replayed is None:
            if sequence_index is not None:
                return self._test_sequence_recall_success_rate(
                    sequence_index=sequence_index,
                    num_trials=num_trials,
                    noise_level=noise_level,
                    verbose=verbose
                )
            else:
                # 测试所有序列
                results = {}
                for k in range(len(self.training_sequences)):
                    if verbose:
                        print(f"\n测试序列 #{k}:")
                    results[f'sequence_{k}'] = self._test_sequence_recall_success_rate(
                        sequence_index=k,
                        num_trials=num_trials,
                        noise_level=noise_level,
                        verbose=verbose
                    )
                return results
        
        # 单次评估模式（检查完整序列匹配 + 逐帧匹配）
        if sequence_index is not None:
            return self._evaluate_single_replay_with_frames(
                xi_replayed,
                self.training_sequences[sequence_index],
                sequence_index=sequence_index,
                include_frame_matching=include_frame_matching
            )
        else:
            # 与所有序列比较
            results = []
            for k, target_seq in enumerate(self.training_sequences):
                result = self._evaluate_single_replay_with_frames(
                    xi_replayed,
                    target_seq,
                    sequence_index=k,
                    include_frame_matching=include_frame_matching
                )
                results.append(result)
            
            best_idx = np.argmax([r['found_sequence'] for r in results])
            
            return {
                'best_match': results[best_idx],
                'all_matches': results,
                'best_sequence_index': best_idx
            }

    def _evaluate_single_replay_with_frames(self, xi_replayed: np.ndarray,
                                        target_sequence: np.ndarray,
                                        sequence_index: Optional[int] = None,
                                        include_frame_matching: bool = True) -> Dict:
        """
        评估单次回放（完整序列匹配 + 逐帧匹配信息）
        
        参数:
            xi_replayed: 回放序列
            target_sequence: 目标训练序列
            sequence_index: 序列索引（可选）
            include_frame_matching: 是否包含逐帧匹配信息
            
        返回:
            评估结果字典
        """
        max_steps = xi_replayed.shape[0]
        T = len(target_sequence)
        
        # 1. 检查是否包含完整的训练序列（主要评估）
        found_sequence = False
        match_start_idx = -1
        
        for tau in range(max_steps - T + 1):
            segment = xi_replayed[tau:tau+T, :]
            if np.array_equal(segment, target_sequence):
                found_sequence = True
                match_start_idx = tau
                break
        
        result = {
            'found_sequence': found_sequence,
            'recall_accuracy': 1.0 if found_sequence else 0.0,
            'match_start_idx': match_start_idx,
            'evaluation_mode': 'full_sequence_matching'
        }
        
        # 2. 逐帧匹配信息（用于可视化）
        if include_frame_matching:
            match_indices = np.zeros(max_steps, dtype=int)
            frame_match_count = 0
            
            for step in range(max_steps):
                for t in range(T):
                    if np.all(xi_replayed[step, :] == target_sequence[t, :]):
                        match_indices[step] = t + 1  # 1-indexed
                        frame_match_count += 1
                        break
            
            frame_recall_accuracy = frame_match_count / max_steps
            
            result['match_indices'] = match_indices
            result['frame_match_count'] = frame_match_count
            result['frame_recall_accuracy'] = frame_recall_accuracy
        
        if sequence_index is not None:
            result['sequence_index'] = sequence_index
        
        return result

    def _test_single_sequence_recall(self, target_sequence: np.ndarray,
                                    num_trials: int = 50,
                                    noise_level: float = 0.0,
                                    verbose: bool = False,
                                    sequence_name: str = "序列") -> Dict:
        """
        测试单个序列的回放成功率（类似 test_robustness 的方式）
        
        参数:
            target_sequence: 目标序列
            num_trials: 试验次数
            noise_level: 噪声水平
            verbose: 是否打印信息
            sequence_name: 序列名称（用于打印）
            
        返回:
            包含成功率和详细统计的字典
        """
        T = len(target_sequence)
        max_search_steps = T * 5
        
        success_count = 0
        convergence_steps = []
        trajectory = np.zeros((max_search_steps + 1, self.N_v))
        
        for trial in range(num_trials):
            # 1. 生成初始状态（加噪或无噪）
            xi_test = target_sequence[0, :].copy().reshape(-1, 1)
            
            if noise_level > 0:
                num_flips = int(noise_level * self.N_v)
                if num_flips > 0:
                    flip_indices = np.random.choice(self.N_v, num_flips, replace=False)
                    xi_test[flip_indices] = -xi_test[flip_indices]
            
            # 2. 记录演化轨迹
            trajectory[0, :] = xi_test.flatten()
            
            for step in range(max_search_steps):
                zeta = np.sign(self.U @ xi_test)
                zeta[zeta == 0] = 1
                xi_test = np.sign(self.V @ zeta)
                xi_test[xi_test == 0] = 1
                trajectory[step + 1, :] = xi_test.flatten()
            
            # 3. 检查是否成功回放完整序列
            found_sequence = False
            for tau in range(max_search_steps - T + 2):
                segment = trajectory[tau:tau+T, :]
                if np.array_equal(segment, target_sequence):
                    found_sequence = True
                    convergence_steps.append(tau)
                    break
            
            if found_sequence:
                success_count += 1
        
        success_rate = success_count / num_trials
        
        if verbose:
            print(f'{sequence_name}, 噪声水平 {noise_level:.2f}: '
                  f'成功率 {success_rate*100:.1f}% ({success_count}/{num_trials} 次成功)')
            if convergence_steps:
                print(f'  平均收敛步数: {np.mean(convergence_steps):.1f}')
                print(f'  收敛步数范围: [{np.min(convergence_steps)}, {np.max(convergence_steps)}]')
        
        return {
            'success_rate': success_rate,
            'recall_accuracy': success_rate,  # 为了向后兼容
            'success_count': success_count,
            'num_trials': num_trials,
            'noise_level': noise_level,
            'convergence_steps': convergence_steps if convergence_steps else None,
            'avg_convergence_steps': np.mean(convergence_steps) if convergence_steps else None,
            'evaluation_mode': 'multiple_trials'
        }
    
    def test_robustness(self, noise_levels: np.ndarray, 
                       num_trials: int = 50, 
                       verbose: bool = True,
                       sequence_index: int = 0) -> np.ndarray:
        """
        测试噪声鲁棒性（修正版 - 使用正确的评估方式）
        
        参数:
            noise_levels: 噪声水平数组
            num_trials: 每个噪声水平的测试次数
            verbose: 是否打印进度
            sequence_index: 测试哪个序列（仅多序列模式）
            
        返回:
            成功率数组
        """
        # 单序列模式
        if len(self.training_sequences) == 0:
            if self.training_sequence is None:
                raise AssertionError("请先训练网络")
            target_sequence = self.training_sequence
            seq_name = "单序列"
        else:
            # 多序列模式
            assert sequence_index < len(self.training_sequences), \
                f"序列索引 {sequence_index} 超出范围"
            target_sequence = self.training_sequences[sequence_index]
            seq_name = f"序列 #{sequence_index}"
        
        robustness_scores = np.zeros(len(noise_levels))
        
        for i, noise_level in enumerate(noise_levels):
            result = self._test_single_sequence_recall(
                target_sequence=target_sequence,
                num_trials=num_trials,
                noise_level=noise_level,
                verbose=False,
                sequence_name=seq_name
            )
            robustness_scores[i] = result['success_rate']
            
            if verbose:
                print(f'{seq_name}, 噪声水平 {noise_level:.2f}: '
                      f'成功率 {robustness_scores[i]*100:.1f}% '
                      f'({result["success_count"]}/{num_trials} 次成功)')
        
        return robustness_scores
    
    def test_robustness_all_sequences(self, noise_levels: np.ndarray, 
                                     num_trials: int = 50, 
                                     verbose: bool = True) -> Dict:
        """
        测试所有序列的噪声鲁棒性（新方法）
        
        返回:
            包含所有序列鲁棒性结果的字典
        """
        results = {}
        
        for k in range(len(self.training_sequences)):
            if verbose:
                print(f"\n测试序列 {k+1}/{len(self.training_sequences)}")
            
            robustness = self.test_robustness(
                noise_levels=noise_levels,
                num_trials=num_trials,
                verbose=verbose,
                sequence_index=k
            )
            
            results[f'sequence_{k}'] = robustness
        
        return results


"""
================================================================
可视化和参数扫描函数 - 完整适配版
同时支持单序列和多序列网络
================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Union


# ========== 1. 单序列可视化函数（原始版本，兼容扩展类）==========
def visualize_results(network: Union[SequenceAttractorNetwork, MultiSequenceAttractorNetwork], 
                     xi_replayed: np.ndarray,
                     eval_results: Dict,
                     save_path: Optional[str] = None,
                     title_suffix: str = "",
                     show_images: bool = False,
                     sequence_index: int = 0):
    """
    可视化训练和回放结果（支持单序列和多序列网络，适配新评估方法）
    
    参数:
        network: 网络对象（原始类或扩展类）
        xi_replayed: 回放序列
        eval_results: 评估结果
        save_path: 保存路径
        title_suffix: 标题后缀
        show_images: 是否显示图片
        sequence_index: 显示哪个序列（仅多序列模式）
    """
    fig = plt.figure(figsize=(14, 9))
    
    num_epochs = len(network.mu_history)
    max_steps = xi_replayed.shape[0]
    
    # 确定使用哪个训练序列
    is_multi = isinstance(network, MultiSequenceAttractorNetwork) and len(network.training_sequences) > 0
    
    if is_multi:
        training_seq = network.training_sequences[sequence_index]
        num_seq = len(network.training_sequences)
    else:
        training_seq = network.training_sequence
        num_seq = 1
    
    # 提取评估结果（适配新格式）
    if 'evaluation_mode' in eval_results:
        # 新的评估方法
        if eval_results['evaluation_mode'] == 'full_sequence_matching':
            # 单次评估模式 - 生成匹配索引用于可视化
            recall_acc = eval_results['recall_accuracy']
            match_start = eval_results.get('match_start_idx', -1)
            
            # 构造 match_indices 用于可视化
            match_indices = np.zeros(max_steps, dtype=int)
            if match_start >= 0:
                T = len(training_seq)
                for i in range(match_start, min(match_start + T, max_steps)):
                    match_indices[i] = (i - match_start) + 1
            
        elif eval_results['evaluation_mode'] == 'multiple_trials':
            # 多次试验模式 - 无法提供详细的匹配索引
            recall_acc = eval_results['recall_accuracy']
            match_indices = np.zeros(max_steps, dtype=int)
            # 显示平均成功率而非逐帧匹配
            
    else:
        # 旧的评估方法（向后兼容）
        if 'recall_accuracy' in eval_results:
            match_indices = eval_results.get('match_indices', np.zeros(max_steps, dtype=int))
            recall_acc = eval_results['recall_accuracy']
        elif 'best_match' in eval_results:
            match_indices = eval_results['best_match'].get('match_indices', np.zeros(max_steps, dtype=int))
            recall_acc = eval_results['best_match']['recall_accuracy']
        else:
            raise ValueError("评估结果格式不正确")
    
    # 子图1: 隐藏层训练误差
    ax1 = plt.subplot(3, 3, 1)
    plt.plot(range(1, num_epochs + 1), network.mu_history, 'b-', linewidth=2)
    plt.xlabel('训练轮数')
    plt.ylabel('平均误差 μ')
    plt.title('隐藏层训练误差')
    plt.grid(True)
    
    # 子图2: 可见层训练误差
    ax2 = plt.subplot(3, 3, 2)
    plt.plot(range(1, num_epochs + 1), network.nu_history, 'r-', linewidth=2)
    plt.xlabel('训练轮数')
    plt.ylabel('平均误差 ν')
    plt.title('可见层训练误差')
    plt.grid(True)
    
    # 子图3: 双误差对比
    ax3 = plt.subplot(3, 3, 3)
    plt.plot(range(1, num_epochs + 1), network.mu_history, 'b-', 
             linewidth=1.5, label='μ (隐藏层)')
    plt.plot(range(1, num_epochs + 1), network.nu_history, 'r-', 
             linewidth=1.5, label='ν (可见层)')
    plt.xlabel('训练轮数')
    plt.ylabel('平均误差')
    if num_seq > 1:
        plt.title(f'误差收敛曲线 ({num_seq}个序列)')
    else:
        plt.title('误差收敛曲线')
    plt.legend()
    plt.grid(True)
    
    # 子图4: 训练序列
    ax4 = plt.subplot(3, 3, 4)
    plt.imshow(training_seq.T, cmap='gray', 
               aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('时间步')
    plt.ylabel('神经元索引')
    if num_seq > 1:
        plt.title(f'训练序列 #{sequence_index+1}')
    else:
        plt.title('训练序列（可见层）')
    
    # 子图5: 回放序列
    ax5 = plt.subplot(3, 3, 5)
    plt.imshow(xi_replayed.T, cmap='gray', aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('时间步')
    plt.ylabel('神经元索引')
    plt.title('回放序列（可见层）')
    
    # 子图6: U权重矩阵
    ax6 = plt.subplot(3, 3, 6)
    im6 = plt.imshow(network.U, cmap='jet', aspect='auto', interpolation='nearest')
    plt.colorbar(im6)
    plt.xlabel('可见神经元')
    plt.ylabel('隐藏神经元')
    plt.title('权重矩阵 U')
    
    # 子图7: V权重矩阵
    ax7 = plt.subplot(3, 3, 7)
    im7 = plt.imshow(network.V, cmap='jet', aspect='auto', interpolation='nearest')
    plt.colorbar(im7)
    plt.xlabel('隐藏神经元')
    plt.ylabel('可见神经元')
    plt.title('权重矩阵 V')
    
    # 子图8: P固定投影矩阵
    ax8 = plt.subplot(3, 3, 8)
    im8 = plt.imshow(network.P, cmap='jet', aspect='auto', interpolation='nearest')
    plt.colorbar(im8)
    plt.xlabel('可见神经元')
    plt.ylabel('隐藏神经元')
    plt.title('固定投影矩阵 P')
    
    # 子图9: 序列匹配追踪（适配新评估方法）
    # 子图9: 序列匹配追踪（增强版）  
    ax9 = plt.subplot(3, 3, 9)  
    
    if 'evaluation_mode' in eval_results and \
       eval_results['evaluation_mode'] == 'full_sequence_matching':  
        
        # 绘制逐帧匹配情况  
        if 'match_indices' in eval_results:  
            match_indices = eval_results['match_indices']  
            max_steps = len(match_indices)  
            
            plt.plot(range(1, max_steps + 1), match_indices, 'o',   
                     markersize=4, alpha=0.5, color='gray', label='逐帧匹配')  
            
            # 如果找到完整序列，高亮显示  
            if eval_results['found_sequence']:  
                match_start = eval_results['match_start_idx']  
                T = len(training_seq)  
                
                # 绘制完整匹配区域  
                complete_match_x = range(match_start + 1, match_start + T + 1)  
                complete_match_y = range(1, T + 1)  
                plt.plot(complete_match_x, complete_match_y, 'o-',   
                         linewidth=2, markersize=6, color='green',   
                         label='完整序列匹配')  
                
                # 添加阴影区域  
                plt.axvspan(match_start + 1, match_start + T,   
                           alpha=0.2, color='green')  
                
                title_text = f'序列匹配追踪\n' \
                           f'完整匹配: ✓ (位置={match_start})\n' \
                           f'逐帧准确率: {eval_results.get("frame_recall_accuracy", 0)*100:.1f}%'  
            else:  
                title_text = f'序列匹配追踪\n' \
                           f'完整匹配: ✗\n' \
                           f'逐帧准确率: {eval_results.get("frame_recall_accuracy", 0)*100:.1f}%'  
            
            plt.xlabel('回放时间步')  
            plt.ylabel('匹配的训练序列位置')  
            plt.title(title_text, fontsize=10)  
            plt.ylim([0, len(training_seq) + 1])  
            plt.legend(loc='best', fontsize=8)  
            plt.grid(True, alpha=0.3)  
        else:
            plt.text(0.5, 0.5, '未找到完整序列匹配', 
                    ha='center', va='center', transform=ax9.transAxes,
                    fontsize=12, color='red')
            title_text = '序列匹配追踪\n(完整序列匹配失败)'
        plt.xlabel('回放时间步')
        plt.ylabel('训练序列位置')
        plt.title(title_text)
        plt.ylim([0, len(training_seq) + 1])
        
    elif 'evaluation_mode' in eval_results and eval_results['evaluation_mode'] == 'multiple_trials':
        # 多次试验模式 - 显示统计信息
        success_rate = eval_results['recall_accuracy'] * 100
        ax9.axis('off')
        info_text = f"多次试验评估结果\n\n"
        info_text += f"成功率: {success_rate:.1f}%\n"
        info_text += f"试验次数: {eval_results.get('num_trials', 'N/A')}\n"
        info_text += f"成功次数: {eval_results.get('success_count', 'N/A')}\n"
        if eval_results.get('avg_convergence_steps') is not None:
            info_text += f"平均收敛步数: {eval_results['avg_convergence_steps']:.1f}\n"
        plt.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
    else:
        # 旧的评估方法（向后兼容）
        plt.plot(range(1, max_steps + 1), match_indices, 'o-', 
                 linewidth=1.5, markersize=6)
        plt.xlabel('回放时间步')
        plt.ylabel('匹配的训练序列索引')
        plt.title(f'序列匹配追踪 (准确率: {recall_acc*100:.1f}%)')
        plt.ylim([0, len(training_seq) + 1])
    
    plt.grid(True, alpha=0.3)
    
    # 主标题
    main_title = f'序列吸引子网络训练与回放{title_suffix}'
    if 'evaluation_mode' in eval_results:
        eval_mode_text = {
            'full_sequence_matching': '完整序列匹配',
            'multiple_trials': '多次试验统计'
        }.get(eval_results['evaluation_mode'], '')
        main_title += f'\n评估方式: {eval_mode_text}'
    
    plt.suptitle(main_title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
    if show_images:
        plt.show()
    else:
        plt.close()


# ========== 2. 鲁棒性可视化函数（通用版本）==========
def visualize_robustness(noise_levels: np.ndarray, 
                        robustness_scores: Union[np.ndarray, Dict],
                        save_path: Optional[str] = None,
                        title_suffix: str = "",
                        show_images: bool = False,
                        labels: Optional[List[str]] = None):
    """
    可视化噪声鲁棒性测试结果（支持单序列和多序列）
    
    参数:
        noise_levels: 噪声水平数组
        robustness_scores: 
            - np.ndarray: 单序列的鲁棒性分数
            - Dict: 多序列的鲁棒性分数字典
        save_path: 保存路径
        title_suffix: 标题后缀
        show_images: 是否显示图片
        labels: 自定义标签（可选）
    """
    plt.figure(figsize=(10, 6))
    
    if isinstance(robustness_scores, np.ndarray):
        # 单序列模式
        plt.plot(noise_levels * 100, robustness_scores * 100, '-o',
                 linewidth=2.5, markersize=8, color='#A23B72',
                 label='单序列')
        title = f'序列吸引子的噪声鲁棒性{title_suffix}'
    
    elif isinstance(robustness_scores, dict):
        # 多序列模式
        colors = plt.cm.tab10(np.linspace(0, 1, len(robustness_scores)))
        
        for i, (seq_name, scores) in enumerate(robustness_scores.items()):
            if labels is not None and i < len(labels):
                label = labels[i]
            else:
                seq_idx = int(seq_name.split('_')[1])
                label = f'序列 #{seq_idx+1}'
            
            plt.plot(noise_levels * 100, scores * 100, '-o',
                    linewidth=2, markersize=6, 
                    color=colors[i],
                    label=label)
        
        title = f'多序列噪声鲁棒性对比{title_suffix}'
    
    else:
        raise ValueError("robustness_scores 必须是 np.ndarray 或 Dict")
    
    plt.xlabel('噪声水平 (%)', fontsize=12)
    plt.ylabel('恢复到原序列的成功率 (%)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
    if show_images:
        plt.show()
    else:
        plt.close()


# ========== 3. 参数扫描函数（扩展版本）==========
def parameter_sweep(param_name: str,
                   param_values: np.ndarray,
                   base_config: Dict,
                   mode: str = 'single',
                   sequences: Optional[List[np.ndarray]] = None,
                   num_trials: int = 5,
                   verbose: bool = True) -> Dict:
    """
    参数扫描实验（支持单序列和多序列）
    
    参数:
        param_name: 要扫描的参数名称 ('eta', 'kappa', 'N_h', 等)
        param_values: 参数值数组
        base_config: 基础配置字典
            例如: {'N_v': 50, 'T': 30, 'eta': 0.01, 'kappa': 1, 'num_epochs': 300}
        mode: 'single' 或 'multi'
        sequences: 训练序列（多序列模式必需）
        num_trials: 每个参数值重复次数
        verbose: 是否打印进度
        
    返回:
        结果字典，包含所有实验数据
    """
    results = {
        'param_name': param_name,
        'param_values': param_values,
        'final_mu': np.zeros((len(param_values), num_trials)),
        'final_nu': np.zeros((len(param_values), num_trials)),
        'recall_accuracy': np.zeros((len(param_values), num_trials)),
        'mode': mode
    }
    
    if mode == 'multi':
        results['per_sequence_accuracy'] = {}
    
    for i, param_val in enumerate(param_values):
        if verbose:
            print(f"\n{'='*60}")
            print(f"测试 {param_name} = {param_val} ({i+1}/{len(param_values)})")
            print(f"{'='*60}")
        
        for trial in range(num_trials):
            if verbose:
                print(f"  试验 {trial+1}/{num_trials}...")
            
            # 更新配置
            config = base_config.copy()
            config[param_name] = param_val
            
            # 创建网络
            if mode == 'single':
                network = SequenceAttractorNetwork(
                    N_v=config['N_v'],
                    T=config['T'],
                    N_h=config.get('N_h', None),
                    eta=config.get('eta', 0.001),
                    kappa=config.get('kappa', 1)
                )
                
                # 训练
                network.train(
                    num_epochs=config.get('num_epochs', 500),
                    seed=trial,
                    verbose=False
                )
                
                # 回放测试
                xi_replayed = network.replay()
                eval_result = network.evaluate_replay(xi_replayed)
                
                results['final_mu'][i, trial] = network.mu_history[-1]
                results['final_nu'][i, trial] = network.nu_history[-1]
                results['recall_accuracy'][i, trial] = eval_result['recall_accuracy']
            
            elif mode == 'multi':
                assert sequences is not None, "多序列模式需要提供序列"
                
                network = MultiSequenceAttractorNetwork(
                    N_v=config['N_v'],
                    T=config['T'],
                    N_h=config.get('N_h', None),
                    eta=config.get('eta', 0.001),
                    kappa=config.get('kappa', 1)
                )
                
                # 训练
                network.train(
                    x=sequences,
                    num_epochs=config.get('num_epochs', 500),
                    verbose=False,
                    interleaved=config.get('interleaved', True)
                )
                
                results['final_mu'][i, trial] = network.mu_history[-1]
                results['final_nu'][i, trial] = network.nu_history[-1]
                
                # 测试所有序列
                seq_accuracies = []
                for k in range(len(sequences)):
                    xi_replayed = network.replay(sequence_index=k)
                    eval_result = network.evaluate_replay(xi_replayed, sequence_index=k)
                    seq_accuracies.append(eval_result['recall_accuracy'])
                
                # 平均准确率
                results['recall_accuracy'][i, trial] = np.mean(seq_accuracies)
                
                # 记录每个序列的准确率
                for k, acc in enumerate(seq_accuracies):
                    key = f'sequence_{k}'
                    if key not in results['per_sequence_accuracy']:
                        results['per_sequence_accuracy'][key] = np.zeros((len(param_values), num_trials))
                    results['per_sequence_accuracy'][key][i, trial] = acc
    
    # 计算统计量
    results['final_mu_mean'] = np.mean(results['final_mu'], axis=1)
    results['final_mu_std'] = np.std(results['final_mu'], axis=1)
    results['final_nu_mean'] = np.mean(results['final_nu'], axis=1)
    results['final_nu_std'] = np.std(results['final_nu'], axis=1)
    results['recall_accuracy_mean'] = np.mean(results['recall_accuracy'], axis=1)
    results['recall_accuracy_std'] = np.std(results['recall_accuracy'], axis=1)
    
    if verbose:
        print(f"\n{'='*60}")
        print("参数扫描完成！")
        print(f"{'='*60}")
    
    return results


def visualize_parameter_sweep(results: Dict,
                              save_path: Optional[str] = None,
                              show_images: bool = False):
    """
    可视化参数扫描结果
    
    参数:
        results: parameter_sweep 返回的结果字典
        save_path: 保存路径
        show_images: 是否显示图片
    """
    param_name = results['param_name']
    param_values = results['param_values']
    mode = results['mode']
    
    fig = plt.figure(figsize=(14, 10))
    
    # 子图1: 训练误差 μ
    ax1 = plt.subplot(2, 2, 1)
    plt.errorbar(param_values, results['final_mu_mean'], 
                 yerr=results['final_mu_std'],
                 fmt='o-', linewidth=2, markersize=8, capsize=5, color='blue')
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel('最终 μ 误差', fontsize=12)
    plt.title('隐藏层训练误差 vs ' + param_name, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 子图2: 训练误差 ν
    ax2 = plt.subplot(2, 2, 2)
    plt.errorbar(param_values, results['final_nu_mean'], 
                 yerr=results['final_nu_std'],
                 fmt='o-', linewidth=2, markersize=8, capsize=5, color='red')
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel('最终 ν 误差', fontsize=12)
    plt.title('可见层训练误差 vs ' + param_name, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 子图3: 回放准确率
    ax3 = plt.subplot(2, 2, 3)
    plt.errorbar(param_values, results['recall_accuracy_mean'] * 100, 
                 yerr=results['recall_accuracy_std'] * 100,
                 fmt='o-', linewidth=2, markersize=8, capsize=5, color='green')
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel('回放准确率 (%)', fontsize=12)
    if mode == 'multi':
        plt.title('平均回放准确率 vs ' + param_name, fontweight='bold')
    else:
        plt.title('回放准确率 vs ' + param_name, fontweight='bold')
    plt.ylim([0, 105])
    plt.grid(True, alpha=0.3)
    
    # 子图4: 每个序列的准确率（多序列模式）或总结（单序列模式）
    ax4 = plt.subplot(2, 2, 4)
    if mode == 'multi' and 'per_sequence_accuracy' in results:
        colors = plt.cm.tab10(np.linspace(0, 1, len(results['per_sequence_accuracy'])))
        for i, (seq_name, accuracies) in enumerate(results['per_sequence_accuracy'].items()):
            seq_idx = int(seq_name.split('_')[1])
            mean_acc = np.mean(accuracies, axis=1) * 100
            std_acc = np.std(accuracies, axis=1) * 100
            plt.errorbar(param_values, mean_acc, yerr=std_acc,
                        fmt='o-', linewidth=2, markersize=6, capsize=3,
                        color=colors[i], label=f'序列 #{seq_idx+1}')
        plt.xlabel(param_name, fontsize=12)
        plt.ylabel('回放准确率 (%)', fontsize=12)
        plt.title('各序列回放准确率 vs ' + param_name, fontweight='bold')
        plt.ylim([0, 105])
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        # 单序列模式：显示汇总表格
        plt.axis('off')
        summary_text = f"参数扫描汇总\n\n"
        summary_text += f"参数: {param_name}\n"
        summary_text += f"范围: {param_values[0]:.4f} - {param_values[-1]:.4f}\n\n"
        summary_text += f"最佳回放准确率:\n"
        best_idx = np.argmax(results['recall_accuracy_mean'])
        summary_text += f"  {param_name} = {param_values[best_idx]:.4f}\n"
        summary_text += f"  准确率 = {results['recall_accuracy_mean'][best_idx]*100:.1f}%\n\n"
        summary_text += f"最小训练误差:\n"
        best_mu_idx = np.argmin(results['final_mu_mean'])
        summary_text += f"  μ: {param_name} = {param_values[best_mu_idx]:.4f}\n"
        summary_text += f"      μ = {results['final_mu_mean'][best_mu_idx]:.4f}\n"
        best_nu_idx = np.argmin(results['final_nu_mean'])
        summary_text += f"  ν: {param_name} = {param_values[best_nu_idx]:.4f}\n"
        summary_text += f"      ν = {results['final_nu_mean'][best_nu_idx]:.4f}"
        
        plt.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f'参数扫描结果: {param_name} ({mode} 模式)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
    if show_images:
        plt.show()
    else:
        plt.close()
# ========== 可视化函数（扩展版）==========
def visualize_multi_sequence_results(network: MultiSequenceAttractorNetwork,
                                    save_path: Optional[str] = None,
                                    title_suffix: str = "",
                                    show_images: bool = False):
    """
    可视化多序列学习结果
    """
    K = len(network.training_sequences)
    
    if K == 0:
        print("警告：没有训练序列可以可视化")
        return
    
    # 动态计算子图布局
    n_cols = min(K, 3)
    n_rows = (K + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows + 3))
    
    # 顶部：训练误差曲线
    ax_top = plt.subplot(n_rows + 1, 1, 1)
    num_epochs = len(network.mu_history)
    plt.plot(range(1, num_epochs + 1), network.mu_history, 'b-', 
             linewidth=1.5, label='μ (隐藏层)')
    plt.plot(range(1, num_epochs + 1), network.nu_history, 'r-', 
             linewidth=1.5, label='ν (可见层)')
    plt.xlabel('训练轮数')
    plt.ylabel('平均误差')
    plt.title(f'多序列训练误差收敛 ({K}个序列)', fontweight='bold')
    plt.legend()
    plt.grid(True)
    
    # 为每个序列创建回放测试
    for k in range(K):
        xi_replayed = network.replay(sequence_index=k, 
                                     max_steps=network.training_sequences[k].shape[0] * 2)
        eval_result = network.evaluate_replay(xi_replayed, sequence_index=k)
        
        # 训练序列
        ax_train = plt.subplot(n_rows + 1, n_cols * 2, n_cols * 2 + k * 2 + 1)
        plt.imshow(network.training_sequences[k].T, cmap='gray', 
                   aspect='auto', interpolation='nearest')
        plt.title(f'序列 #{k+1} (训练)', fontsize=10)
        plt.xlabel('时间步')
        plt.ylabel('神经元')
        
        # 回放序列
        ax_replay = plt.subplot(n_rows + 1, n_cols * 2, n_cols * 2 + k * 2 + 2)
        plt.imshow(xi_replayed.T, cmap='gray', 
                   aspect='auto', interpolation='nearest')
        plt.title(f'序列 #{k+1} (回放, 准确率={eval_result["recall_accuracy"]*100:.0f}%)', 
                  fontsize=10)
        plt.xlabel('时间步')
        plt.ylabel('神经元')
    
    plt.suptitle(f'多序列学习结果{title_suffix}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"多序列结果图已保存: {save_path}")
    if show_images:
        plt.show()
    else:
        plt.close()


def visualize_multi_sequence_robustness(noise_levels: np.ndarray,
                                       robustness_results: Dict,
                                       save_path: Optional[str] = None,
                                       title_suffix: str = "",
                                       show_images: bool = False):
    """
    可视化多序列的鲁棒性测试结果
    """
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(robustness_results)))
    
    for i, (seq_name, scores) in enumerate(robustness_results.items()):
        seq_idx = int(seq_name.split('_')[1])
        plt.plot(noise_levels * 100, scores * 100, '-o',
                linewidth=2, markersize=6, 
                color=colors[i],
                label=f'序列 #{seq_idx+1}')
    
    plt.xlabel('噪声水平 (%)', fontsize=12)
    plt.ylabel('恢复到原序列的成功率 (%)', fontsize=12)
    plt.title(f'多序列噪声鲁棒性对比{title_suffix}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"多序列鲁棒性图已保存: {save_path}")
    if show_images:
        plt.show()
    else:
        plt.close()

# ========== 使用示例（演示新的评估方式）==========
if __name__ == "__main__":
    import os
    os.makedirs("examples_corrected", exist_ok=True)
    
    print("\n" + "="*70)
    print("演示修正后的评估方法")
    print("="*70)
    
    # 创建网络并训练
    network = MultiSequenceAttractorNetwork(N_v=50, T=30, N_h=200, eta=0.01)
    sequences = network.generate_multiple_sequences(num_sequences=3, seeds=[10, 20, 30])
    
    print("\n训练网络...")
    network.train(x=sequences, num_epochs=400, verbose=True, interleaved=True)
    
    # 方式1: 正确的评估方式 - 多次试验（推荐）
    print("\n" + "="*70)
    print("方式1: 多次试验评估（正确方式）")
    print("="*70)
    
    for k in range(len(sequences)):
        print(f"\n测试序列 #{k}:")
        
        # 无噪声测试
        result = network.evaluate_replay(
            xi_replayed=None,  # 触发多次试验模式
            sequence_index=k,
            num_trials=50,
            noise_level=0.0,
            verbose=True
        )
        
        print(f"  成功率: {result['success_rate']*100:.1f}%")
        if result['avg_convergence_steps'] is not None:
            print(f"  平均收敛步数: {result['avg_convergence_steps']:.1f}")
    
    # 方式2: 单次回放评估（使用完整序列匹配）
    print("\n" + "="*70)
    print("方式2: 单次回放评估（完整序列匹配）")
    print("="*70)
    
    for k in range(len(sequences)):
        xi_replayed = network.replay(sequence_index=k, max_steps=sequences[k].shape[0] * 3)
        result = network.evaluate_replay(
            xi_replayed=xi_replayed,
            sequence_index=k
        )
        
        status = "✓ 成功" if result['found_sequence'] else "✗ 失败"
        print(f"序列 #{k}: {status}")
        if result['found_sequence']:
            print(f"  匹配起始位置: {result['match_start_idx']}")
    
    # 方式3: 鲁棒性测试（不同噪声水平）
    print("\n" + "="*70)
    print("方式3: 鲁棒性测试（修正后）")
    print("="*70)
    
    noise_levels = np.arange(0, 0.3, 0.05)
    
    print("\n测试所有序列的鲁棒性:")
    robustness_results = network.test_robustness_all_sequences(
        noise_levels=noise_levels,
        num_trials=50,
        verbose=True
    )
    
    # 可视化鲁棒性结果
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(robustness_results)))
    
    for i, (seq_name, scores) in enumerate(robustness_results.items()):
        seq_idx = int(seq_name.split('_')[1])
        plt.plot(noise_levels * 100, scores * 100, '-o',
                linewidth=2, markersize=6, 
                color=colors[i],
                label=f'序列 #{seq_idx+1}')
    
    plt.xlabel('噪声水平 (%)', fontsize=12)
    plt.ylabel('完整序列回放成功率 (%)', fontsize=12)
    plt.title('修正后的鲁棒性评估（完整序列匹配）', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])
    plt.tight_layout()
    plt.savefig("examples_corrected/robustness_corrected.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ 已保存: examples_corrected/robustness_corrected.png")
    
    # 对比新旧评估方式
    print("\n" + "="*70)
    print("对比: 新旧评估方式的差异")
    print("="*70)
    
    print("\n旧方法（逐帧匹配）的问题:")
    print("  ✗ 只统计有多少帧匹配训练序列中的某一帧")
    print("  ✗ 不检查是否形成完整的连续序列")
    print("  ✗ 可能高估回放能力")
    
    print("\n新方法（完整序列匹配）的优势:")
    print("  ✓ 检查回放序列中是否包含完整的训练序列")
    print("  ✓ 通过多次试验统计成功概率")
    print("  ✓ 更准确地反映网络的序列记忆能力")
    print("  ✓ 与 test_robustness 的评估方式一致")
    
    print("\n" + "="*70)
    print("演示完成！")
    print("="*70)
