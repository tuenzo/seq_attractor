"""
================================================================
多序列吸引子网络
支持同时学习多个序列
================================================================
"""

import numpy as np
from typing import Optional, Dict, List, Union
from ..core.base import SequenceAttractorNetwork
from ..utils.evaluation import evaluate_replay_full_sequence


class MultiSequenceAttractorNetwork(SequenceAttractorNetwork):
    """
    多序列吸引子网络
    在基础网络基础上添加多序列学习功能
    """
    
    def __init__(self, N_v: int, T: int, N_h: Optional[int] = None, 
                 eta: float = 0.001, kappa: float = 1):
        """初始化网络"""
        super().__init__(N_v, T, N_h, eta, kappa)
        
        # 多序列专用属性
        self.training_sequences = []
        self.num_sequences = 0
    
    def generate_random_sequence_with_length(self, T: int, seed: Optional[int] = None) -> np.ndarray:
        """生成指定长度的随机序列"""
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
                                    T: Optional[int] = None,
                                    ensure_unique_across: bool = True,
                                    max_attempts: int = 1000) -> List[np.ndarray]:
        """生成多个随机序列，支持跨序列唯一性检查"""
        sequences = []
        if seeds is None:
            seeds = list(range(num_sequences))
        
        seq_length = T if T is not None else self.T
        
        if not ensure_unique_across:
            for i, seed in enumerate(seeds[:num_sequences]):
                seq = self.generate_random_sequence_with_length(T=seq_length, seed=seed)
                sequences.append(seq)
            return sequences
        
        # 确保跨序列唯一性
        print(f"生成 {num_sequences} 个序列，确保跨序列唯一性...")
        all_used_frames = []
        
        for seq_idx, seed in enumerate(seeds[:num_sequences]):
            if seed is not None:
                np.random.seed(seed)
            
            print(f"  正在生成序列 #{seq_idx+1}...", end=" ")
            seq = np.zeros((seq_length, self.N_v))
            
            for t in range(seq_length - 1):
                attempts = 0
                while attempts < max_attempts:
                    candidate_frame = np.sign(np.random.randn(self.N_v))
                    candidate_frame[candidate_frame == 0] = 1
                    
                    # 检查序列内重复
                    is_unique_within = True
                    for prev_t in range(t):
                        if np.array_equal(candidate_frame, seq[prev_t, :]):
                            is_unique_within = False
                            break
                    
                    if not is_unique_within:
                        attempts += 1
                        continue
                    
                    # 检查跨序列重复
                    is_unique_across = True
                    for used_frame in all_used_frames:
                        if np.array_equal(candidate_frame, used_frame):
                            is_unique_across = False
                            break
                    
                    if is_unique_across:
                        seq[t, :] = candidate_frame
                        all_used_frames.append(candidate_frame.copy())
                        break
                    
                    attempts += 1
                
                if attempts >= max_attempts:
                    print(f"\n警告: 序列 #{seq_idx+1} 位置 {t} 无法生成唯一帧")
                    seq[t, :] = candidate_frame
            
            seq[seq_length - 1, :] = seq[0, :]
            sequences.append(seq)
            print("完成")
        
        print("所有序列生成完毕\n")
        return sequences
    
    def train(self, x: Optional[Union[np.ndarray, List[np.ndarray]]] = None, 
              num_epochs: int = 500, 
              verbose: bool = True, 
              seed: Optional[int] = None, 
              V_only: bool = False,
              interleaved: bool = True) -> Dict:
        """训练网络（支持单序列和多序列）"""
        if x is None or isinstance(x, np.ndarray):
            result = super().train(x=x, num_epochs=num_epochs, verbose=verbose, 
                                  seed=seed, V_only=V_only)
            if self.training_sequence is not None:
                self.training_sequences = [self.training_sequence]
                self.num_sequences = 1
            return result
        elif isinstance(x, list):
            return self._train_multiple_sequences(
                sequences=x, num_epochs=num_epochs, V_only=V_only, 
                verbose=verbose, interleaved=interleaved
            )
        else:
            raise ValueError("x 必须是 None、np.ndarray 或 List[np.ndarray]")
    
    def _train_multiple_sequences(self, sequences: List[np.ndarray], 
                                  num_epochs: int, 
                                  V_only: bool,
                                  verbose: bool,
                                  interleaved: bool) -> Dict:
        """多序列训练的内部实现"""
        for i, seq in enumerate(sequences):
            assert seq.shape[1] == self.N_v, \
                f"序列 {i} 的可见层维度应为 {self.N_v}，实际为 {seq.shape[1]}"
        
        self.training_sequences = sequences
        self.num_sequences = len(sequences)
        
        if verbose:
            print(f"开始多序列训练... N_v={self.N_v}, N_h={self.N_h}")
            print(f"参数: eta={self.eta}, kappa={self.kappa}, epochs={num_epochs}")
            print(f"序列数量: {self.num_sequences}")
            print(f"训练模式: {'交替训练' if interleaved else '批量训练'}")
        
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
        """序列回放（支持多序列）"""
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
    
    def evaluate_replay(self, xi_replayed: Optional[np.ndarray] = None,
                    sequence_index: Optional[int] = None,
                    num_trials: int = 50,
                    noise_level: float = 0.0,
                    verbose: bool = False) -> Dict:
        """评估回放质量"""
        if len(self.training_sequences) == 0:
            raise AssertionError("请先训练网络")
        
        # 多次试验模式
        if xi_replayed is None:
            if sequence_index is not None:
                return self._test_sequence_recall(
                    sequence_index=sequence_index,
                    num_trials=num_trials,
                    noise_level=noise_level,
                    verbose=verbose
                )
            else:
                results = {}
                for k in range(len(self.training_sequences)):
                    if verbose:
                        print(f"\n测试序列 #{k}:")
                    results[f'sequence_{k}'] = self._test_sequence_recall(
                        sequence_index=k,
                        num_trials=num_trials,
                        noise_level=noise_level,
                        verbose=verbose
                    )
                return results
        
        # 单次评估模式
        if sequence_index is not None:
            return evaluate_replay_full_sequence(
                xi_replayed,
                self.training_sequences[sequence_index]
            )
        else:
            results = []
            for k, target_seq in enumerate(self.training_sequences):
                result = evaluate_replay_full_sequence(xi_replayed, target_seq)
                result['sequence_index'] = k
                results.append(result)
            
            best_idx = np.argmax([r['found_sequence'] for r in results])
            return {
                'best_match': results[best_idx],
                'all_matches': results,
                'best_sequence_index': best_idx
            }
    
    def _test_sequence_recall(self, sequence_index: int,
                             num_trials: int = 50,
                             noise_level: float = 0.0,
                             verbose: bool = False) -> Dict:
        """测试单个序列的回放成功率"""
        assert sequence_index < len(self.training_sequences), \
            f"序列索引 {sequence_index} 超出范围"
        
        target_sequence = self.training_sequences[sequence_index]
        T = len(target_sequence)
        max_search_steps = T * 5
        
        success_count = 0
        convergence_steps = []
        
        for trial in range(num_trials):
            xi_test = target_sequence[0, :].copy().reshape(-1, 1)
            
            if noise_level > 0:
                num_flips = int(noise_level * self.N_v)
                if num_flips > 0:
                    flip_indices = np.random.choice(self.N_v, num_flips, replace=False)
                    xi_test[flip_indices] = -xi_test[flip_indices]
            
            trajectory = [xi_test.flatten().copy()]
            for step in range(max_search_steps):
                zeta = np.sign(self.U @ xi_test)
                zeta[zeta == 0] = 1
                xi_test = np.sign(self.V @ zeta)
                xi_test[xi_test == 0] = 1
                trajectory.append(xi_test.flatten().copy())
            
            found_sequence = False
            for tau in range(max_search_steps - T + 2):
                segment = np.array(trajectory[tau:tau+T])
                if np.array_equal(segment, target_sequence):
                    found_sequence = True
                    convergence_steps.append(tau)
                    break
            
            if found_sequence:
                success_count += 1
        
        success_rate = success_count / num_trials
        
        if verbose:
            print(f'序列 #{sequence_index}, 噪声水平 {noise_level:.2f}: '
                  f'成功率 {success_rate*100:.1f}% ({success_count}/{num_trials} 次成功)')
            if convergence_steps:
                print(f'  平均收敛步数: {np.mean(convergence_steps):.1f}')
        
        return {
            'success_rate': success_rate,
            'recall_accuracy': success_rate,
            'success_count': success_count,
            'num_trials': num_trials,
            'noise_level': noise_level,
            'sequence_index': sequence_index,
            'convergence_steps': convergence_steps if convergence_steps else None,
            'avg_convergence_steps': np.mean(convergence_steps) if convergence_steps else None,
            'evaluation_mode': 'multiple_trials'
        }
    
    def test_robustness(self, noise_levels: np.ndarray, 
                       num_trials: int = 50, 
                       verbose: bool = True,
                       sequence_index: int = 0) -> np.ndarray:
        """测试噪声鲁棒性"""
        assert sequence_index < len(self.training_sequences), \
            f"序列索引 {sequence_index} 超出范围"
        
        robustness_scores = np.zeros(len(noise_levels))
        
        for i, noise_level in enumerate(noise_levels):
            result = self._test_sequence_recall(
                sequence_index=sequence_index,
                num_trials=num_trials,
                noise_level=noise_level,
                verbose=False
            )
            robustness_scores[i] = result['success_rate']
            
            if verbose:
                print(f'序列 #{sequence_index}, 噪声水平 {noise_level:.2f}: '
                      f'成功率 {robustness_scores[i]*100:.1f}%')
        
        return robustness_scores

