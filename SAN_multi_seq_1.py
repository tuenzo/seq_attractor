"""
================================================================
序列吸引子网络 - 多序列扩展版（继承实现）
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
        
    def generate_random_sequence_with_length(self, T: int, seed: Optional[int] = None) -> np.ndarray:
        """
        生成指定长度的随机序列（扩展方法）
        
        参数:
            T: 序列长度
            seed: 随机种子
            
        返回:
            x: T x N_v 的二值序列
        """
        if seed is not None:
            np.random.seed(seed)
            
        x = np.sign(np.random.randn(T, self.N_v))
        x[x == 0] = 1
        
        # 确保序列中没有重复（除了首尾）
        for t in range(1, T - 1):
            while np.any(np.all(x[t, :] == x[:t, :], axis=1)):
                x[t, :] = np.sign(np.random.randn(self.N_v))
                x[t, x[t, :] == 0] = 1
        
        x[T - 1, :] = x[0, :]  # 周期性
        
        return x
    
    def generate_multiple_sequences(self, num_sequences: int, 
                                    seeds: Optional[List[int]] = None,
                                    T: Optional[int] = None) -> List[np.ndarray]:
        """
        生成多个随机序列（新方法）
        
        参数:
            num_sequences: 序列数量
            seeds: 随机种子列表（可选）
            T: 序列长度（可选，默认使用self.T）
            
        返回:
            序列列表
        """
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
        """
        训练网络（重写方法，支持单序列和多序列）
        
        参数:
            x: 训练序列
                - None: 自动生成单个随机序列（调用父类方法）
                - np.ndarray: 单个序列（调用父类方法）
                - List[np.ndarray]: 多个序列（使用多序列训练）
            num_epochs: 训练轮数
            verbose: 是否打印训练信息
            seed: 随机种子
            V_only: 是否仅更新V权重
            interleaved: 多序列训练模式（True=交替训练，False=批量训练）
            
        返回:
            训练结果字典
        """
        # 判断训练模式
        if x is None or isinstance(x, np.ndarray):
            # 单序列模式：直接调用父类方法
            result = super().train(x, num_epochs, verbose, seed, V_only)
            
            # 同步到多序列存储
            if self.training_sequence is not None:
                self.training_sequences = [self.training_sequence]
                self.num_sequences = 1
            
            return result
        
        elif isinstance(x, list):
            # 多序列模式：使用新的训练逻辑
            return self._train_multiple_sequences(x, num_epochs, verbose, V_only, interleaved)
        
        else:
            raise ValueError("x 必须是 None、np.ndarray 或 List[np.ndarray]")
    
    def _train_multiple_sequences(self, sequences: List[np.ndarray], 
                                  num_epochs: int, 
                                  V_only: bool,
                                  verbose: bool,
                                  interleaved: bool) -> Dict:
        """
        多序列训练的内部实现（新方法）
        """
        # 验证序列
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
        
        # 预分配历史数组
        self.mu_history = np.zeros(num_epochs)
        self.nu_history = np.zeros(num_epochs)
        
        # 选择训练策略
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
        """
        交替训练策略（论文方法）
        每个epoch内轮流训练所有序列
        """
        # 预计算所有序列的数据
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
        
        # 训练循环
        for epoch in range(num_epochs):
            epoch_mu = 0
            epoch_nu = 0
            
            # 轮流处理每个序列
            for k, data in enumerate(seq_data):
                x_current_all = data['x_current']
                x_next_all = data['x_next']
                
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
            
            # 记录历史
            self.mu_history[epoch] = epoch_mu / (self.N_h * total_transitions)
            self.nu_history[epoch] = epoch_nu / (self.N_v * total_transitions)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, '
                      f'μ={self.mu_history[epoch]:.4f}, '
                      f'ν={self.nu_history[epoch]:.4f}')
    
    def _train_batch(self, sequences: List[np.ndarray], 
                    num_epochs: int, V_only: bool, verbose: bool):
        """
        批量训练策略
        将所有序列的转换合并成一个大批次
        """
        # 合并所有序列
        all_x_current = []
        all_x_next = []
        
        for seq in sequences:
            all_x_current.append(seq[:-1, :].T)
            all_x_next.append(seq[1:, :].T)
        
        x_current_all = np.hstack(all_x_current)
        x_next_all = np.hstack(all_x_next)
        total_transitions = x_current_all.shape[1]
        
        # 训练循环
        for epoch in range(num_epochs):
            # 更新 U
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
            
            # 更新 V
            y_actual_all = np.sign(self.U @ x_current_all)
            y_actual_all[y_actual_all == 0] = 1
            
            v_input_all = self.V @ y_actual_all
            nu_all = (x_next_all * v_input_all < self.kappa).astype(float)
            
            delta_V = (nu_all * x_next_all) @ y_actual_all.T
            self.V += self.eta * delta_V
            
            total_nu = np.sum(nu_all)
            
            # 记录历史
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
        """
        序列回放（扩展方法，支持多序列）
        
        参数:
            x_init: 初始状态
            noise_level: 噪声水平
            max_steps: 最大回放步数
            sequence_index: 使用哪个训练序列的初始状态（仅多序列模式）
            
        返回:
            回放序列
        """
        if x_init is None:
            if len(self.training_sequences) > 0:
                # 多序列模式
                assert sequence_index < len(self.training_sequences), \
                    f"序列索引 {sequence_index} 超出范围"
                x_init = self.training_sequences[sequence_index][0, :].copy()
            elif self.training_sequence is not None:
                # 单序列模式（父类）
                x_init = self.training_sequence[0, :].copy()
            else:
                raise AssertionError("请先训练网络或提供初始状态")
        
        # 调用父类的replay方法
        return super().replay(x_init, noise_level, max_steps)
    
    def evaluate_replay(self, xi_replayed: np.ndarray, 
                       sequence_index: Optional[int] = None) -> Dict:
        """
        评估回放质量（扩展方法，支持多序列）
        
        参数:
            xi_replayed: 回放序列
            sequence_index: 与哪个训练序列比较（None表示与所有序列比较）
            
        返回:
            评估指标字典
        """
        if len(self.training_sequences) == 0:
            # 单序列模式（父类）
            return super().evaluate_replay(xi_replayed)
        
        max_steps = xi_replayed.shape[0]
        
        if sequence_index is not None:
            # 与指定序列比较
            target_seq = self.training_sequences[sequence_index]
            match_indices = np.zeros(max_steps, dtype=int)
            
            for step in range(max_steps):
                matches = np.all(xi_replayed[step, :] == target_seq, axis=1)
                if np.any(matches):
                    match_indices[step] = np.where(matches)[0][0] + 1
            
            match_count = np.sum(match_indices > 0)
            recall_accuracy = match_count / max_steps
            
            return {
                'recall_accuracy': recall_accuracy,
                'match_count': match_count,
                'match_indices': match_indices,
                'sequence_index': sequence_index
            }
        else:
            # 与所有序列比较
            results = []
            for k, target_seq in enumerate(self.training_sequences):
                match_indices = np.zeros(max_steps, dtype=int)
                
                for step in range(max_steps):
                    matches = np.all(xi_replayed[step, :] == target_seq, axis=1)
                    if np.any(matches):
                        match_indices[step] = np.where(matches)[0][0] + 1
                
                match_count = np.sum(match_indices > 0)
                recall_accuracy = match_count / max_steps
                
                results.append({
                    'recall_accuracy': recall_accuracy,
                    'match_count': match_count,
                    'match_indices': match_indices,
                    'sequence_index': k
                })
            
            # 找到最佳匹配
            best_idx = np.argmax([r['recall_accuracy'] for r in results])
            
            return {
                'best_match': results[best_idx],
                'all_matches': results,
                'best_sequence_index': best_idx
            }
    
    def test_robustness(self, noise_levels: np.ndarray, 
                       num_trials: int = 50, 
                       verbose: bool = True,
                       sequence_index: int = 0) -> np.ndarray:
        """
        测试噪声鲁棒性（扩展方法，支持多序列）
        
        参数:
            noise_levels: 噪声水平数组
            num_trials: 每个噪声水平的测试次数
            verbose: 是否打印进度
            sequence_index: 测试哪个序列（仅多序列模式）
            
        返回:
            成功率数组
        """
        if len(self.training_sequences) == 0:
            # 单序列模式（父类）
            return super().test_robustness(noise_levels, num_trials, verbose)
        
        # 多序列模式
        assert sequence_index < len(self.training_sequences), \
            f"序列索引 {sequence_index} 超出范围"
        
        target_sequence = self.training_sequences[sequence_index]
        T = len(target_sequence)
        
        robustness_scores = np.zeros(len(noise_levels))
        max_search_steps = T * 5
        trajectory = np.zeros((max_search_steps + 1, self.N_v))
        
        for i, noise_level in enumerate(noise_levels):
            success_count = 0
            
            for trial in range(num_trials):
                # 生成加噪初始状态
                xi_noisy = target_sequence[0, :].copy().reshape(-1, 1)
                num_flips = int(noise_level * self.N_v)
                if num_flips > 0:
                    flip_indices = np.random.choice(self.N_v, num_flips, replace=False)
                    xi_noisy[flip_indices] = -xi_noisy[flip_indices]
                
                # 演化轨迹
                trajectory[0, :] = xi_noisy.flatten()
                
                for step in range(max_search_steps):
                    zeta = np.sign(self.U @ xi_noisy)
                    zeta[zeta == 0] = 1
                    xi_noisy = np.sign(self.V @ zeta)
                    xi_noisy[xi_noisy == 0] = 1
                    trajectory[step + 1, :] = xi_noisy.flatten()
                
                # 检查匹配
                found_sequence = False
                for tau in range(max_search_steps - T + 2):
                    segment = trajectory[tau:tau+T, :]
                    if np.array_equal(segment, target_sequence):
                        found_sequence = True
                        break
                
                if found_sequence:
                    success_count += 1
            
            robustness_scores[i] = success_count / num_trials
            
            if verbose:
                print(f'序列 {sequence_index}, 噪声水平 {noise_level:.2f}: '
                      f'成功率 {robustness_scores[i]*100:.1f}%')
        
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


# ========== 使用示例 ==========
if __name__ == "__main__":
    import time
    
    print("\n" + "="*70)
    print("示例1: 使用原始类（单序列）- 向后兼容测试")
    print("="*70)
    
    # 使用原始类，功能完全不变
    original_network = SequenceAttractorNetwork(N_v=50, T=30, eta=0.01)
    original_network.train(num_epochs=200, seed=42, verbose=True)
    
    xi_replayed = original_network.replay()
    eval_result = original_network.evaluate_replay(xi_replayed)
    print(f"原始类回放准确率: {eval_result['recall_accuracy']*100:.1f}%")
    
    
    print("\n" + "="*70)
    print("示例2: 使用扩展类（单序列）- 向后兼容测试")
    print("="*70)
    
    # 扩展类也支持单序列，完全兼容
    extended_network_single = MultiSequenceAttractorNetwork(N_v=50, T=30, eta=0.01)
    extended_network_single.train(num_epochs=200, seed=42, verbose=True)
    
    xi_replayed = extended_network_single.replay()
    eval_result = extended_network_single.evaluate_replay(xi_replayed)
    print(f"扩展类（单序列）回放准确率: {eval_result['recall_accuracy']*100:.1f}%")
    
    
    print("\n" + "="*70)
    print("示例3: 使用扩展类（多序列）- 新功能")
    print("="*70)
    
    # 创建扩展类
    multi_network = MultiSequenceAttractorNetwork(N_v=50, T=30, N_h=200, eta=0.01)
    
    # 生成多个序列
    sequences = multi_network.generate_multiple_sequences(num_sequences=3, seeds=[10, 20, 30])
    print(f"生成了 {len(sequences)} 个序列")
    
    # 多序列训练（交替模式）
    start_time = time.time()
    train_results = multi_network.train(
        x=sequences,
        num_epochs=400,
        verbose=True,
        interleaved=True
    )
    train_time = time.time() - start_time
    
    print(f"\n训练完成，耗时: {train_time:.2f} 秒")
    print(f"最终 μ 误差: {train_results['final_mu']:.4f}")
    print(f"最终 ν 误差: {train_results['final_nu']:.4f}")
    
    # 测试每个序列的回放
    print("\n回放测试:")
    for k in range(len(sequences)):
        xi_replayed = multi_network.replay(sequence_index=k)
        eval_result = multi_network.evaluate_replay(xi_replayed, sequence_index=k)
        print(f"  序列 #{k+1}: 准确率 {eval_result['recall_accuracy']*100:.1f}%")
    
    # 可视化多序列结果
    visualize_multi_sequence_results(
        multi_network,
        save_path="multi_sequence_inheritance_demo.png",
        title_suffix="\n(继承实现, 3个序列)",
        show_images=True
    )
    
    
    print("\n" + "="*70)
    print("示例4: 多序列鲁棒性测试")
    print("="*70)
    
    noise_levels = np.arange(0, 0.3, 0.05)
    robustness_results = multi_network.test_robustness_all_sequences(
        noise_levels=noise_levels,
        num_trials=30,
        verbose=True
    )
    
    # 可视化
    visualize_multi_sequence_robustness(
        noise_levels=noise_levels,
        robustness_results=robustness_results,
        save_path="multi_sequence_robustness_inheritance.png",
        title_suffix="\n(继承实现)",
        show_images=True
    )
    
    
    print("\n" + "="*70)
    print("示例5: 对比批量训练 vs 交替训练")
    print("="*70)
    
    test_sequences = multi_network.generate_multiple_sequences(3, seeds=[100, 200, 300])
    
    # 交替训练
    net_interleaved = MultiSequenceAttractorNetwork(N_v=50, T=30, N_h=200, eta=0.01)
    start = time.time()
    net_interleaved.train(x=test_sequences, num_epochs=300, verbose=False, interleaved=True)
    time_interleaved = time.time() - start
    
    # 批量训练
    net_batch = MultiSequenceAttractorNetwork(N_v=50, T=30, N_h=200, eta=0.01)
    start = time.time()
    net_batch.train(x=test_sequences, num_epochs=300, verbose=False, interleaved=False)
    time_batch = time.time() - start
    
    print(f"训练时间对比:")
    print(f"  交替训练: {time_interleaved:.2f} 秒")
    print(f"  批量训练: {time_batch:.2f} 秒")
    
    print(f"\n回放准确率对比:")
    for k in range(len(test_sequences)):
        replay_int = net_interleaved.replay(sequence_index=k)
        eval_int = net_interleaved.evaluate_replay(replay_int, sequence_index=k)
        
        replay_bat = net_batch.replay(sequence_index=k)
        eval_bat = net_batch.evaluate_replay(replay_bat, sequence_index=k)
        
        print(f"  序列 #{k+1}:")
        print(f"    交替训练: {eval_int['recall_accuracy']*100:.1f}%")
        print(f"    批量训练: {eval_bat['recall_accuracy']*100:.1f}%")
    
    
    print("\n" + "="*70)
    print("所有示例完成！")
    print("="*70)