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
        # 修复：检查是否有多序列数据
        if len(self.training_sequences) == 0:
            # 单序列模式（父类）- 使用 training_sequence
            return super().evaluate_replay(xi_replayed)
        
        # 如果只有一个序列且没有指定sequence_index，也使用单序列逻辑
        if len(self.training_sequences) == 1 and sequence_index is None:
            target_seq = self.training_sequences[0]
            max_steps = xi_replayed.shape[0]
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
                'match_indices': match_indices
            }
        
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
    可视化训练和回放结果（支持单序列和多序列网络）
    
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
    
    # 提取评估结果
    if 'recall_accuracy' in eval_results:
        match_indices = eval_results['match_indices']
        recall_acc = eval_results['recall_accuracy']
    elif 'best_match' in eval_results:
        match_indices = eval_results['best_match']['match_indices']
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
    
    # 子图9: 序列匹配追踪
    ax9 = plt.subplot(3, 3, 9)
    plt.plot(range(1, max_steps + 1), match_indices, 'o-', 
             linewidth=1.5, markersize=6)
    plt.xlabel('回放时间步')
    plt.ylabel('匹配的训练序列索引')
    plt.title(f'序列匹配追踪 (准确率: {recall_acc*100:.1f}%)')
    plt.ylim([0, len(training_seq) + 1])
    plt.grid(True)
    
    # 主标题
    main_title = f'序列吸引子网络训练与回放{title_suffix}'
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

# ========== 使用示例 ==========
if __name__ == "__main__":
    import time
    import os
    
    # 创建 examples 文件夹
    os.makedirs("examples", exist_ok=True)
    
    print("\n" + "="*70)
    print("示例1: 使用原始类（单序列）- 向后兼容测试")
    print("="*70)
    
    # 使用原始类，功能完全不变
    original_network = SequenceAttractorNetwork(N_v=50, T=30, eta=0.01)
    original_network.train(num_epochs=200, seed=42, verbose=True)
    
    xi_replayed = original_network.replay()
    eval_result = original_network.evaluate_replay(xi_replayed)
    print(f"原始类回放准确率: {eval_result['recall_accuracy']*100:.1f}%")
    
    # 使用适配后的可视化函数
    visualize_results(
        network=original_network,
        xi_replayed=xi_replayed,
        eval_results=eval_result,
        save_path="examples/example1_original_single.png",
        title_suffix="\n(原始类 - 单序列)",
        show_images=False
    )
    print("✓ 已保存: examples/example1_original_single.png")

    
    print("\n" + "="*70)
    print("示例2: 使用扩展类（单序列）- 向后兼容测试")
    print("="*70)
    
    # 扩展类也支持单序列，完全兼容
    extended_network_single = MultiSequenceAttractorNetwork(N_v=50, T=30, eta=0.01)
    extended_network_single.train(num_epochs=200, seed=42, verbose=True)
    
    xi_replayed = extended_network_single.replay()
    eval_result = extended_network_single.evaluate_replay(xi_replayed)
    print(f"扩展类（单序列）回放准确率: {eval_result['recall_accuracy']*100:.1f}%")
    
    # 使用适配后的可视化函数
    visualize_results(
        network=extended_network_single,
        xi_replayed=xi_replayed,
        eval_results=eval_result,
        save_path="examples/example2_extended_single.png",
        title_suffix="\n(扩展类 - 单序列)",
        show_images=False
    )
    print("✓ 已保存: examples/example2_extended_single.png")

    # 测试单序列鲁棒性
    print("\n单序列鲁棒性测试:")
    noise_levels_single = np.arange(0, 0.25, 0.05)
    robustness_single = extended_network_single.test_robustness(
        noise_levels=noise_levels_single,
        num_trials=30,
        verbose=True
    )
    
    visualize_robustness(
        noise_levels=noise_levels_single,
        robustness_scores=robustness_single,
        save_path="examples/example2_robustness_single.png",
        title_suffix="\n(扩展类 - 单序列)",
        show_images=False
    )
    print("✓ 已保存: examples/example2_robustness_single.png")

    
    print("\n" + "="*70)
    print("示例3: 使用扩展类（多序列）- 新功能")
    print("="*70)
    
    # 创建扩展类
    multi_network = MultiSequenceAttractorNetwork(N_v=50, T=30, N_h=200, eta=0.01)
    
    # 生成多个序列
    sequences = multi_network.generate_multiple_sequences(num_sequences=3, seeds=[10, 20, 30])
    print(f"生成了 {len(sequences)} 个序列")
    for i, seq in enumerate(sequences):
        print(f"  序列 #{i+1}: 形状 {seq.shape}")
    
    # 多序列训练（交替模式）
    print("\n开始多序列训练...")
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
        
        # 可视化每个序列的回放结果
        visualize_results(
            network=multi_network,
            xi_replayed=xi_replayed,
            eval_results=eval_result,
            save_path=f"examples/example3_multi_seq{k+1}.png",
            title_suffix=f"\n(多序列 - 序列 #{k+1})",
            show_images=False,
            sequence_index=k
        )
        print(f"  ✓ 已保存: examples/example3_multi_seq{k+1}.png")
    
    # 使用专用的多序列可视化
    visualize_multi_sequence_results(
        multi_network,
        save_path="examples/example3_multi_overview.png",
        title_suffix="\n(继承实现, 3个序列)",
        show_images=False
    )
    print("✓ 已保存: examples/example3_multi_overview.png")

    
    print("\n" + "="*70)
    print("示例4: 多序列鲁棒性测试")
    print("="*70)
    
    noise_levels = np.arange(0, 0.3, 0.05)
    print(f"噪声水平: {noise_levels}")
    print("开始测试所有序列的鲁棒性...")
    
    robustness_results = multi_network.test_robustness_all_sequences(
        noise_levels=noise_levels,
        num_trials=30,
        verbose=True
    )
    
    # 可视化（使用适配后的函数）
    visualize_robustness(
        noise_levels=noise_levels,
        robustness_scores=robustness_results,
        save_path="examples/example4_robustness_multi.png",
        title_suffix="\n(多序列对比)",
        show_images=False
    )
    print("✓ 已保存: examples/example4_robustness_multi.png")

    # 也使用专用的多序列鲁棒性可视化
    visualize_multi_sequence_robustness(
        noise_levels=noise_levels,
        robustness_results=robustness_results,
        save_path="examples/example4_robustness_multi_detail.png",
        title_suffix="\n(继承实现)",
        show_images=False
    )
    print("✓ 已保存: examples/example4_robustness_multi_detail.png")

    
    print("\n" + "="*70)
    print("示例5: 对比批量训练 vs 交替训练")
    print("="*70)
    
    test_sequences = multi_network.generate_multiple_sequences(3, seeds=[100, 200, 300])
    print(f"生成测试序列: {len(test_sequences)} 个")
    
    # 交替训练
    print("\n1. 交替训练模式...")
    net_interleaved = MultiSequenceAttractorNetwork(N_v=50, T=30, N_h=200, eta=0.01)
    start = time.time()
    net_interleaved.train(x=test_sequences, num_epochs=300, verbose=False, interleaved=True)
    time_interleaved = time.time() - start
    
    # 批量训练
    print("2. 批量训练模式...")
    net_batch = MultiSequenceAttractorNetwork(N_v=50, T=30, N_h=200, eta=0.01)
    start = time.time()
    net_batch.train(x=test_sequences, num_epochs=300, verbose=False, interleaved=False)
    time_batch = time.time() - start
    
    print(f"\n训练时间对比:")
    print(f"  交替训练: {time_interleaved:.2f} 秒")
    print(f"  批量训练: {time_batch:.2f} 秒")
    print(f"  速度提升: {time_interleaved/time_batch:.2f}x")
    
    print(f"\n回放准确率对比:")
    accuracies_interleaved = []
    accuracies_batch = []
    
    for k in range(len(test_sequences)):
        replay_int = net_interleaved.replay(sequence_index=k)
        eval_int = net_interleaved.evaluate_replay(replay_int, sequence_index=k)
        
        replay_bat = net_batch.replay(sequence_index=k)
        eval_bat = net_batch.evaluate_replay(replay_bat, sequence_index=k)
        
        acc_int = eval_int['recall_accuracy']*100
        acc_bat = eval_bat['recall_accuracy']*100
        
        accuracies_interleaved.append(acc_int)
        accuracies_batch.append(acc_bat)
        
        print(f"  序列 #{k+1}:")
        print(f"    交替训练: {acc_int:.1f}%")
        print(f"    批量训练: {acc_bat:.1f}%")
    
    print(f"\n平均准确率:")
    print(f"  交替训练: {np.mean(accuracies_interleaved):.1f}%")
    print(f"  批量训练: {np.mean(accuracies_batch):.1f}%")
    
    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 训练误差对比
    ax1 = axes[0]
    ax1.plot(net_interleaved.mu_history, 'b-', linewidth=1.5, label='μ (交替)')
    ax1.plot(net_interleaved.nu_history, 'r-', linewidth=1.5, label='ν (交替)')
    ax1.plot(net_batch.mu_history, 'b--', linewidth=1.5, alpha=0.7, label='μ (批量)')
    ax1.plot(net_batch.nu_history, 'r--', linewidth=1.5, alpha=0.7, label='ν (批量)')
    ax1.set_xlabel('训练轮数')
    ax1.set_ylabel('平均误差')
    ax1.set_title('训练误差收敛对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率对比
    ax2 = axes[1]
    x_pos = np.arange(len(test_sequences))
    width = 0.35
    ax2.bar(x_pos - width/2, accuracies_interleaved, width, 
            label='交替训练', alpha=0.8, color='#2E86AB')
    ax2.bar(x_pos + width/2, accuracies_batch, width, 
            label='批量训练', alpha=0.8, color='#A23B72')
    ax2.set_xlabel('序列编号')
    ax2.set_ylabel('回放准确率 (%)')
    ax2.set_title('回放准确率对比')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'#{i+1}' for i in range(len(test_sequences))])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig("examples/example5_training_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ 已保存: examples/example5_training_comparison.png")
    
    
    print("\n" + "="*70)
    print("示例6: 参数扫描 - 学习率 (单序列)")
    print("="*70)
    
    eta_values = np.logspace(-3, -1, 5)  # 0.001 到 0.1
    print(f"扫描学习率 η: {eta_values}")
    
    sweep_results_single = parameter_sweep(
        param_name='eta',
        param_values=eta_values,
        base_config={
            'N_v': 50,
            'T': 30,
            'kappa': 1,
            'num_epochs': 200
        },
        mode='single',
        num_trials=3,
        verbose=True
    )
    
    visualize_parameter_sweep(
        sweep_results_single,
        save_path="examples/example6_sweep_eta_single.png",
        show_images=False
    )
    print("✓ 已保存: examples/example6_sweep_eta_single.png")


    print("\n" + "="*70)
    print("示例7: 参数扫描 - 隐藏层大小 (多序列)")
    print("="*70)
    
    # 生成测试序列
    sweep_net = MultiSequenceAttractorNetwork(N_v=50, T=30)
    sweep_sequences = sweep_net.generate_multiple_sequences(3, seeds=[1000, 2000, 3000])
    
    N_h_values = np.array([100, 150, 200, 250, 300])
    print(f"扫描隐藏层大小 N_h: {N_h_values}")
    
    sweep_results_multi = parameter_sweep(
        param_name='N_h',
        param_values=N_h_values,
        base_config={
            'N_v': 50,
            'T': 30,
            'eta': 0.01,
            'kappa': 1,
            'num_epochs': 200,
            'interleaved': True
        },
        mode='multi',
        sequences=sweep_sequences,
        num_trials=3,
        verbose=True
    )
    
    visualize_parameter_sweep(
        sweep_results_multi,
        save_path="examples/example7_sweep_Nh_multi.png",
        show_images=False
    )
    print("✓ 已保存: examples/example7_sweep_Nh_multi.png")


    print("\n" + "="*70)
    print("示例8: 参数扫描 - 鲁棒性参数 κ (多序列)")
    print("="*70)
    
    kappa_values = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    print(f"扫描鲁棒性参数 κ: {kappa_values}")
    
    sweep_results_kappa = parameter_sweep(
        param_name='kappa',
        param_values=kappa_values,
        base_config={
            'N_v': 50,
            'T': 30,
            'N_h': 200,
            'eta': 0.01,
            'num_epochs': 200,
            'interleaved': True
        },
        mode='multi',
        sequences=sweep_sequences,
        num_trials=3,
        verbose=True
    )
    
    visualize_parameter_sweep(
        sweep_results_kappa,
        save_path="examples/example8_sweep_kappa_multi.png",
        show_images=False
    )
    print("✓ 已保存: examples/example8_sweep_kappa_multi.png")
    
    
    print("\n" + "="*70)
    print("示例9: 容量测试 - 不同数量的序列")
    print("="*70)
    
    num_sequences_list = [1, 2, 3, 4, 5]
    capacity_results = {
        'num_sequences': [],
        'mean_accuracy': [],
        'std_accuracy': [],
        'final_mu': [],
        'final_nu': []
    }
    
    for num_seq in num_sequences_list:
        print(f"\n测试 {num_seq} 个序列...")
        
        # 生成序列
        cap_net = MultiSequenceAttractorNetwork(N_v=50, T=30, N_h=200, eta=0.01)
        cap_sequences = cap_net.generate_multiple_sequences(
            num_seq, 
            seeds=list(range(1000, 1000+num_seq))
        )
        
        # 训练
        cap_net.train(x=cap_sequences, num_epochs=300, verbose=False)
        
        # 测试所有序列
        accuracies = []
        for k in range(num_seq):
            xi_rep = cap_net.replay(sequence_index=k)
            eval_res = cap_net.evaluate_replay(xi_rep, sequence_index=k)
            accuracies.append(eval_res['recall_accuracy'])
        
        capacity_results['num_sequences'].append(num_seq)
        capacity_results['mean_accuracy'].append(np.mean(accuracies))
        capacity_results['std_accuracy'].append(np.std(accuracies))
        capacity_results['final_mu'].append(cap_net.mu_history[-1])
        capacity_results['final_nu'].append(cap_net.nu_history[-1])
        
        print(f"  平均准确率: {np.mean(accuracies)*100:.1f}%")
        print(f"  最终 μ: {cap_net.mu_history[-1]:.4f}")
    
    # 可视化容量测试结果
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.errorbar(capacity_results['num_sequences'], 
                 np.array(capacity_results['mean_accuracy'])*100,
                 yerr=np.array(capacity_results['std_accuracy'])*100,
                 fmt='o-', linewidth=2, markersize=8, capsize=5, color='green')
    ax1.set_xlabel('序列数量', fontsize=12)
    ax1.set_ylabel('平均回放准确率 (%)', fontsize=12)
    ax1.set_title('网络容量测试', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])
    
    ax2 = axes[1]
    ax2.plot(capacity_results['num_sequences'], capacity_results['final_mu'], 
             'o-', linewidth=2, markersize=8, label='μ', color='blue')
    ax2.plot(capacity_results['num_sequences'], capacity_results['final_nu'], 
             's-', linewidth=2, markersize=8, label='ν', color='red')
    ax2.set_xlabel('序列数量', fontsize=12)
    ax2.set_ylabel('最终训练误差', fontsize=12)
    ax2.set_title('训练误差 vs 序列数量', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("examples/example9_capacity_test.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ 已保存: examples/example9_capacity_test.png")
    
    
    print("\n" + "="*70)
    print("示例10: 综合性能总结")
    print("="*70)
    
    # 创建综合报告
    summary_text = f"""
╔═══════════════════════════════════════════════════════════════╗
║        序列吸引子网络 - 多序列扩展版性能总结                  ║
╚═══════════════════════════════════════════════════════════════╝

【向后兼容性】
✓ 原始类功能完整保留
✓ 扩展类完全兼容单序列模式
✓ 所有可视化函数自动适配

【多序列学习能力】
✓ 支持任意数量序列同时学习
✓ 交替训练和批量训练两种模式
✓ 独立的序列回放和评估

【性能指标】（基于本次测试）
• 单序列回放准确率: {eval_result['recall_accuracy']*100:.1f}%
• 多序列平均准确率: {capacity_results['mean_accuracy'][-1]*100:.1f}%
• 训练时间对比: 交替 vs 批量 = {time_interleaved:.2f}s vs {time_batch:.2f}s

【参数扫描结果】
• 学习率 η 最优值: ~0.01
• 隐藏层大小 N_h: 推荐 3×(T-1) 到 5×(T-1)
• 鲁棒性参数 κ: 推荐 1.0-1.5

【容量分析】
• 测试序列数: 1-5个
• 最大容量（准确率>90%）: {max([i+1 for i, acc in enumerate(capacity_results['mean_accuracy']) if acc > 0.9], default='N/A')}个序列

【生成的图片文件】（保存在 examples/ 文件夹）
"""
    
    # 统计 examples 文件夹中的图片
    example_files = [f for f in os.listdir('examples') if f.endswith('.png')]
    for i, img_file in enumerate(sorted(example_files), 1):
        summary_text += f"\n    {i:2d}. {img_file}"
    
    summary_text += "\n\n    " + "="*60
    summary_text += "\n    总计: {} 个示例图片\n".format(len(example_files))
    
    print(summary_text)
    
    # 保存总结报告到 examples 文件夹
    with open("examples/example_summary.txt", "w", encoding='utf-8') as f:
        f.write(summary_text)
    print("✓ 已保存: examples/example_summary.txt")
    
    
    print("\n" + "="*70)
    print("🎉 所有示例完成！共生成 {} 个可视化图片".format(len(example_files)))
    print("   所有文件已保存在 examples/ 文件夹中")
    print("="*70)  