"""
================================================================
序列吸引子网络 - 模式复现类（Pattern Reproduction Network）
整合多序列可视化功能，提供完整的训练和评估接口
================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List, Union
import os
from datetime import datetime

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 导入基础类
from SAN_incremental import SequenceAttractorNetwork


class PatternReproductionNetwork(SequenceAttractorNetwork):
    """
    模式复现网络类
    
    功能特性：
    1. 支持单序列和多序列学习
    2. 支持增量学习（学习新序列同时保持旧记忆）
    3. 集成完整的可视化功能
    4. 提供序列重叠分析
    5. 支持鲁棒性测试
    """
    
    def __init__(self, N_v: int, T: int, N_h: Optional[int] = None, 
                 eta: float = 0.001, kappa: float = 1):
        """
        初始化网络
        
        参数:
            N_v: 可见神经元数量
            T: 序列长度
            N_h: 隐藏神经元数量（默认为3*(T-1)）
            eta: 学习率
            kappa: 鲁棒性参数
        """
        super().__init__(N_v, T, N_h, eta, kappa)
        
        # 多序列管理
        self.training_sequences = []  # 存储所有训练序列
        self.num_sequences = 0
        
        # 增量学习相关
        self._total_epochs_trained = 0
        self.sequence_training_info = []
        
    # ========== 序列生成方法 ==========
    
    def generate_random_sequence_with_length(self, T: int, seed: Optional[int] = None) -> np.ndarray:
        """
        生成指定长度的随机序列
        
        参数:
            T: 序列长度
            seed: 随机种子
            
        返回:
            T x N_v 的二值序列
        """
        if seed is not None:
            np.random.seed(seed)
        
        x = np.sign(np.random.randn(T, self.N_v))
        x[x == 0] = 1
        
        # 确保序列内部无重复
        for t in range(1, T - 1):
            while np.any(np.all(x[t, :] == x[:t, :], axis=1)):
                x[t, :] = np.sign(np.random.randn(self.N_v))
                x[t, x[t, :] == 0] = 1
        
        x[T - 1, :] = x[0, :]  # 周期性
        return x
    
    def generate_multiple_sequences(self, num_sequences: int, 
                                    seeds: Optional[List[int]] = None,
                                    T: Optional[int] = None,
                                    ensure_unique_across: bool = True,
                                    max_attempts: int = 1000) -> List[np.ndarray]:
        """
        生成多个随机序列
        
        参数:
            num_sequences: 序列数量
            seeds: 随机种子列表
            T: 序列长度（默认使用self.T）
            ensure_unique_across: 是否确保跨序列唯一性
            max_attempts: 生成唯一帧的最大尝试次数
            
        返回:
            序列列表
        """
        sequences = []
        if seeds is None:
            seeds = list(range(num_sequences))
        
        seq_length = T if T is not None else self.T
        
        if not ensure_unique_across:
            # 不检查跨序列重叠
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
            
            seq[seq_length - 1, :] = seq[0, :]  # 周期性
            sequences.append(seq)
            print("完成")
        
        print("所有序列生成完毕\n")
        return sequences
    
    # ========== 序列分析方法 ==========
    
    def analyze_sequence_overlap(self, sequences: List[np.ndarray]) -> Dict:
        """
        分析多个序列之间的重叠情况
        
        参数:
            sequences: 序列列表
            
        返回:
            重叠分析结果字典
        """
        num_sequences = len(sequences)
        if num_sequences == 0:
            return {"error": "没有序列"}
        
        seq_length = sequences[0].shape[0]
        
        overlap_info = {
            'total_frames': num_sequences * (seq_length - 1),
            'unique_frames': 0,
            'duplicate_frames': 0,
            'overlap_details': []
        }
        
        # 收集所有帧
        all_frames = []
        frame_sources = []
        
        for seq_idx, seq in enumerate(sequences):
            for t in range(seq_length - 1):
                frame = seq[t, :]
                all_frames.append(frame)
                frame_sources.append((seq_idx, t))
        
        # 检查重复
        checked_indices = set()
        
        for i in range(len(all_frames)):
            if i in checked_indices:
                continue
            
            duplicates = []
            for j in range(i + 1, len(all_frames)):
                if np.array_equal(all_frames[i], all_frames[j]):
                    duplicates.append(j)
                    checked_indices.add(j)
            
            if duplicates:
                overlap_info['duplicate_frames'] += 1 + len(duplicates)
                overlap_info['overlap_details'].append({
                    'frame_index': i,
                    'source': frame_sources[i],
                    'duplicates': [frame_sources[j] for j in duplicates]
                })
            else:
                overlap_info['unique_frames'] += 1
        
        overlap_info['overlap_rate'] = overlap_info['duplicate_frames'] / overlap_info['total_frames']
        return overlap_info
    
    def print_overlap_analysis(self, sequences: List[np.ndarray]):
        """
        打印序列重叠分析报告
        
        参数:
            sequences: 序列列表
        """
        analysis = self.analyze_sequence_overlap(sequences)
        
        print("\n" + "="*60)
        print("序列重叠分析报告")
        print("="*60)
        print(f"序列数量: {len(sequences)}")
        print(f"总帧数: {analysis['total_frames']}")
        print(f"唯一帧数: {analysis['unique_frames']}")
        print(f"重复帧数: {analysis['duplicate_frames']}")
        print(f"重叠率: {analysis['overlap_rate']*100:.2f}%")
        
        if analysis['overlap_details']:
            print("\n重叠详情:")
            for i, detail in enumerate(analysis['overlap_details'][:10]):
                seq_idx, pos = detail['source']
                print(f"  [{i+1}] 序列 #{seq_idx} 位置 {pos} 与以下位置重叠:")
                for dup_seq_idx, dup_pos in detail['duplicates']:
                    print(f"      - 序列 #{dup_seq_idx} 位置 {dup_pos}")
            
            if len(analysis['overlap_details']) > 10:
                print(f"  ... 还有 {len(analysis['overlap_details']) - 10} 处重叠未显示")
        else:
            print("\n✓ 没有发现重叠")
        
        print("="*60 + "\n")
    
    # ========== 在 PatternReproductionNetwork 类中添加以下方法 ==========
    # 添加到序列生成方法部分（在 generate_multiple_sequences 之后）

    def generate_patterned_sequence(self, pattern_type: str = 'alternating', 
                                    T: Optional[int] = None,
                                    seed: Optional[int] = None,
                                    **kwargs) -> np.ndarray:
        """
        生成具有特定重复模式的序列
        
        参数:
            pattern_type: 模式类型
                - 'alternating': 交替模式（A-B-A-B...）
                - 'periodic': 周期性重复（A-B-C-A-B-C...）
                - 'block': 块状重复（A-A-A-B-B-B...）
                - 'mirrored': 镜像模式（A-B-C-C-B-A...）
                - 'custom': 自定义重复模式
            T: 序列长度（默认使用self.T）
            seed: 随机种子
            **kwargs: 额外参数
                - period: 周期长度（用于periodic）
                - block_size: 块大小（用于block）
                - custom_pattern: 自定义模式列表（用于custom）
            
        返回:
            T x N_v 的序列
        """
        if seed is not None:
            np.random.seed(seed)
        
        seq_length = T if T is not None else self.T
        x = np.zeros((seq_length, self.N_v))
        
        if pattern_type == 'alternating':
            # 交替模式：两帧交替
            frame_A = np.sign(np.random.randn(self.N_v))
            frame_A[frame_A == 0] = 1
            frame_B = np.sign(np.random.randn(self.N_v))
            frame_B[frame_B == 0] = 1
            
            for t in range(seq_length - 1):
                x[t, :] = frame_A if t % 2 == 0 else frame_B
            x[seq_length - 1, :] = x[0, :]  # 周期性
            
        elif pattern_type == 'periodic':
            # 周期性重复：多帧循环
            period = kwargs.get('period', 3)
            base_frames = []
            for _ in range(period):
                frame = np.sign(np.random.randn(self.N_v))
                frame[frame == 0] = 1
                base_frames.append(frame)
            
            for t in range(seq_length - 1):
                x[t, :] = base_frames[t % period]
            x[seq_length - 1, :] = x[0, :]
            
        elif pattern_type == 'block':
            # 块状重复：连续相同帧形成块
            block_size = kwargs.get('block_size', 3)
            num_blocks = (seq_length - 1 + block_size - 1) // block_size
            
            block_frames = []
            for _ in range(num_blocks):
                frame = np.sign(np.random.randn(self.N_v))
                frame[frame == 0] = 1
                block_frames.append(frame)
            
            for t in range(seq_length - 1):
                block_idx = t // block_size
                x[t, :] = block_frames[block_idx]
            x[seq_length - 1, :] = x[0, :]
            
        elif pattern_type == 'mirrored':
            # 镜像模式：正向+反向
            half_length = (seq_length - 1) // 2
            unique_frames = []
            
            for _ in range(half_length):
                frame = np.sign(np.random.randn(self.N_v))
                frame[frame == 0] = 1
                unique_frames.append(frame)
            
            # 正向
            for t in range(half_length):
                x[t, :] = unique_frames[t]
            
            # 反向镜像
            for t in range(half_length, seq_length - 1):
                mirror_idx = seq_length - 2 - t
                if mirror_idx < len(unique_frames):
                    x[t, :] = unique_frames[mirror_idx]
            
            x[seq_length - 1, :] = x[0, :]
            
        elif pattern_type == 'custom':
            # 自定义重复模式
            custom_pattern = kwargs.get('custom_pattern', [0, 1, 0, 2])
            unique_frames_count = max(custom_pattern) + 1
            
            # 生成唯一帧
            unique_frames = []
            for _ in range(unique_frames_count):
                frame = np.sign(np.random.randn(self.N_v))
                frame[frame == 0] = 1
                unique_frames.append(frame)
            
            # 按模式填充
            for t in range(seq_length - 1):
                pattern_idx = t % len(custom_pattern)
                frame_idx = custom_pattern[pattern_idx]
                x[t, :] = unique_frames[frame_idx]
            
            x[seq_length - 1, :] = x[0, :]
        
        else:
            raise ValueError(f"未知的模式类型: {pattern_type}")
        
        return x


    def generate_multiple_patterned_sequences(self, num_sequences: int,
                                            pattern_configs: List[Dict],
                                            seeds: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        生成多个具有不同模式的序列
        
        参数:
            num_sequences: 序列数量
            pattern_configs: 模式配置列表，每个元素是一个字典，包含:
                - pattern_type: 模式类型
                - 其他参数（period, block_size等）
            seeds: 随机种子列表
            
        返回:
            序列列表
        """
        sequences = []
        if seeds is None:
            seeds = list(range(num_sequences))
        
        for i in range(num_sequences):
            config = pattern_configs[i] if i < len(pattern_configs) else pattern_configs[0]
            seq = self.generate_patterned_sequence(
                pattern_type=config.get('pattern_type', 'alternating'),
                T=config.get('T', self.T),
                seed=seeds[i],
                **{k: v for k, v in config.items() if k not in ['pattern_type', 'T']}
            )
            sequences.append(seq)
        
        return sequences


    def analyze_pattern_structure(self, sequence: np.ndarray) -> Dict:
        """
        分析序列的重复模式结构
        
        参数:
            sequence: 序列数组
            
        返回:
            模式分析结果
        """
        T = sequence.shape[0]
        
        # 检测重复帧
        repetition_map = {}
        for t in range(T - 1):
            frame_tuple = tuple(sequence[t, :])
            if frame_tuple in repetition_map:
                repetition_map[frame_tuple].append(t)
            else:
                repetition_map[frame_tuple] = [t]
        
        # 统计
        unique_frames = len(repetition_map)
        max_repetitions = max(len(positions) for positions in repetition_map.values())
        
        # 检测周期性
        detected_period = None
        for period in range(2, T // 2):
            is_periodic = True
            for t in range(T - 1 - period):
                if not np.array_equal(sequence[t, :], sequence[t + period, :]):
                    is_periodic = False
                    break
            if is_periodic:
                detected_period = period
                break
        
        return {
            'total_frames': T - 1,
            'unique_frames': unique_frames,
            'max_repetitions': max_repetitions,
            'repetition_rate': 1 - (unique_frames / (T - 1)),
            'detected_period': detected_period,
            'frame_positions': repetition_map
        }


    def visualize_pattern_comparison(self, sequences: List[np.ndarray],
                                    pattern_names: List[str],
                                    save_path: Optional[str] = None,
                                    show_images: bool = False):
        """
        可视化多个模式序列的对比
        
        参数:
            sequences: 序列列表
            pattern_names: 模式名称列表
            save_path: 保存路径
            show_images: 是否显示图片
        """
        num_sequences = len(sequences)
        fig, axes = plt.subplots(num_sequences, 1, figsize=(12, 3 * num_sequences))
        
        if num_sequences == 1:
            axes = [axes]
        
        for i, (seq, name) in enumerate(zip(sequences, pattern_names)):
            # 分析模式
            analysis = self.analyze_pattern_structure(seq)
            
            # 绘制序列
            axes[i].imshow(seq.T, cmap='gray', aspect='auto', interpolation='nearest')
            axes[i].set_xlabel('时间步')
            axes[i].set_ylabel('神经元')
            
            # 标题包含分析信息
            title = f'{name}\n'
            title += f'唯一帧: {analysis["unique_frames"]}/{analysis["total_frames"]} | '
            title += f'重复率: {analysis["repetition_rate"]*100:.1f}%'
            if analysis['detected_period']:
                title += f' | 周期: {analysis["detected_period"]}'
            
            axes[i].set_title(title, fontsize=10)
        
        plt.suptitle('不同重复模式序列对比', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"模式对比图已保存: {save_path}")
        if show_images:
            plt.show()
        else:
            plt.close()


    # ========== 训练方法 ==========
    
    def train(self, x: Optional[Union[np.ndarray, List[np.ndarray]]] = None, 
              num_epochs: int = 500, 
              verbose: bool = True, 
              seed: Optional[int] = None, 
              V_only: bool = False,
              interleaved: bool = True,
              incremental: bool = False) -> Dict:
        """
        训练网络（支持单序列、多序列和增量学习）
        
        参数:
            x: 训练序列
                - None: 自动生成单个随机序列
                - np.ndarray: 单个序列
                - List[np.ndarray]: 多个序列
            num_epochs: 训练轮数
            verbose: 是否打印训练信息
            seed: 随机种子
            V_only: 是否仅更新V权重
            interleaved: 多序列训练模式（True=交替训练）
            incremental: 是否为增量学习模式
            
        返回:
            训练结果字典
        """
        # 单序列模式
        if x is None or isinstance(x, np.ndarray):
            result = super().train(x=x, num_epochs=num_epochs, verbose=verbose, 
                                  seed=seed, V_only=V_only)
            if self.training_sequence is not None:
                self.training_sequences = [self.training_sequence]
                self.num_sequences = 1
            return result
        
        # 多序列模式
        elif isinstance(x, list):
            return self._train_multiple_sequences(
                sequences=x, 
                num_epochs=num_epochs, 
                V_only=V_only, 
                verbose=verbose, 
                interleaved=interleaved,
                incremental=incremental
            )
        else:
            raise ValueError("x 必须是 None、np.ndarray 或 List[np.ndarray]")
    
    def _train_multiple_sequences(self, sequences: List[np.ndarray], 
                                  num_epochs: int, 
                                  V_only: bool,
                                  verbose: bool,
                                  interleaved: bool,
                                  incremental: bool = False) -> Dict:
        """
        多序列训练的内部实现
        
        参数:
            sequences: 序列列表
            num_epochs: 训练轮数
            V_only: 是否仅更新V
            verbose: 是否打印信息
            interleaved: 是否交替训练
            incremental: 是否增量学习
            
        返回:
            训练结果字典
        """
        # 验证序列
        for i, seq in enumerate(sequences):
            assert seq.shape[1] == self.N_v, \
                f"序列 {i} 的可见层维度应为 {self.N_v}，实际为 {seq.shape[1]}"
        
        # 更新序列记录
        if incremental:
            # 增量学习：添加新序列，保留旧序列
            new_sequences = [seq for seq in sequences 
                           if not any(np.array_equal(seq, old_seq) 
                                    for old_seq in self.training_sequences)]
            self.training_sequences.extend(new_sequences)
            sequences_to_train = self.training_sequences  # 训练所有序列
            mode = "增量学习"
        else:
            # 普通模式：替换所有序列
            self.training_sequences = sequences
            sequences_to_train = sequences
            mode = "多序列训练"
        
        self.num_sequences = len(self.training_sequences)
        
        if verbose:
            print(f"开始{mode}... N_v={self.N_v}, N_h={self.N_h}")
            print(f"参数: eta={self.eta}, kappa={self.kappa}, epochs={num_epochs}")
            print(f"序列数量: {self.num_sequences}")
            seq_lengths = [len(seq) for seq in sequences_to_train]
            print(f"序列长度: {seq_lengths}")
            print(f"训练模式: {'交替训练' if interleaved else '批量训练'}")
            if V_only:
                print("仅更新 V 权重矩阵")
        
        # 预分配历史数组
        self.mu_history = np.zeros(num_epochs)
        self.nu_history = np.zeros(num_epochs)
        
        # 选择训练策略
        if interleaved:
            self._train_interleaved(sequences_to_train, num_epochs, V_only, verbose)
        else:
            self._train_batch(sequences_to_train, num_epochs, V_only, verbose)
        
        return {
            'mu_history': self.mu_history,
            'nu_history': self.nu_history,
            'final_mu': self.mu_history[-1],
            'final_nu': self.nu_history[-1],
            'num_sequences': self.num_sequences,
            'training_mode': mode
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
    
    # ========== 回放和评估方法 ==========
    
    def replay(self, x_init: Optional[np.ndarray] = None, 
               noise_level: float = 0.0, 
               max_steps: Optional[int] = None,
               sequence_index: int = 0) -> np.ndarray:
        """
        序列回放
        
        参数:
            x_init: 初始状态
            noise_level: 噪声水平
            max_steps: 最大回放步数
            sequence_index: 使用哪个训练序列的初始状态
            
        返回:
            回放序列
        """
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
        
        """
        评估回放质量（完整序列匹配）
        
        参数:
            xi_replayed: 回放序列（如果为None，则进行多次试验）
            sequence_index: 与哪个训练序列比较
            num_trials: 多次试验的次数
            noise_level: 噪声水平
            verbose: 是否打印详细信息
            
        返回:
            评估指标字典
        """

        if len(self.training_sequences) == 0:  
            raise AssertionError("请先训练网络")  
        
        # 自动处理：如果只有一个序列且未指定索引，默认使用索引0  
        if len(self.training_sequences) == 1 and sequence_index is None:  
            sequence_index = 0 

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
            return self._evaluate_single_replay(
                xi_replayed,
                self.training_sequences[sequence_index],
                sequence_index=sequence_index
            )
        else:
            results = []
            for k, target_seq in enumerate(self.training_sequences):
                result = self._evaluate_single_replay(xi_replayed, target_seq, k)
                results.append(result)
            
            best_idx = np.argmax([r['found_sequence'] for r in results])
            return {
                'best_match': results[best_idx],
                'all_matches': results,
                'best_sequence_index': best_idx
            }
    
    def _evaluate_single_replay(self, xi_replayed: np.ndarray,
                               target_sequence: np.ndarray,
                               sequence_index: Optional[int] = None) -> Dict:
        """评估单次回放（完整序列匹配）"""
        max_steps = xi_replayed.shape[0]
        T = len(target_sequence)
        
        # 检查完整序列匹配
        found_sequence = False
        match_start_idx = -1
        
        for tau in range(max_steps - T + 1):
            segment = xi_replayed[tau:tau+T, :]
            if np.array_equal(segment, target_sequence):
                found_sequence = True
                match_start_idx = tau
                break
        
        # 逐帧匹配（用于可视化）
        match_indices = np.zeros(max_steps, dtype=int)
        frame_match_count = 0
        
        for step in range(max_steps):
            for t in range(T):
                if np.all(xi_replayed[step, :] == target_sequence[t, :]):
                    match_indices[step] = t + 1
                    frame_match_count += 1
                    break
        
        frame_recall_accuracy = frame_match_count / max_steps
        
        result = {
            'found_sequence': found_sequence,
            'recall_accuracy': 1.0 if found_sequence else 0.0,
            'match_start_idx': match_start_idx,
            'match_indices': match_indices,
            'frame_match_count': frame_match_count,
            'frame_recall_accuracy': frame_recall_accuracy,
            'evaluation_mode': 'full_sequence_matching'
        }
        
        if sequence_index is not None:
            result['sequence_index'] = sequence_index
        
        return result
    
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
        trajectory = np.zeros((max_search_steps + 1, self.N_v))
        
        for trial in range(num_trials):
            xi_test = target_sequence[0, :].copy().reshape(-1, 1)
            
            if noise_level > 0:
                num_flips = int(noise_level * self.N_v)
                if num_flips > 0:
                    flip_indices = np.random.choice(self.N_v, num_flips, replace=False)
                    xi_test[flip_indices] = -xi_test[flip_indices]
            
            trajectory[0, :] = xi_test.flatten()
            
            for step in range(max_search_steps):
                zeta = np.sign(self.U @ xi_test)
                zeta[zeta == 0] = 1
                xi_test = np.sign(self.V @ zeta)
                xi_test[xi_test == 0] = 1
                trajectory[step + 1, :] = xi_test.flatten()
            
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
    
    # ========== 鲁棒性测试方法 ==========
    
    def test_robustness(self, noise_levels: np.ndarray, 
                       num_trials: int = 50, 
                       verbose: bool = True,
                       sequence_index: int = 0) -> np.ndarray:
        """
        测试噪声鲁棒性
        
        参数:
            noise_levels: 噪声水平数组
            num_trials: 每个噪声水平的测试次数
            verbose: 是否打印进度
            sequence_index: 测试哪个序列
            
        返回:
            成功率数组
        """
        if len(self.training_sequences) == 0:
            if self.training_sequence is None:
                raise AssertionError("请先训练网络")
            target_sequence = self.training_sequence
            seq_name = "单序列"
        else:
            assert sequence_index < len(self.training_sequences), \
                f"序列索引 {sequence_index} 超出范围"
            target_sequence = self.training_sequences[sequence_index]
            seq_name = f"序列 #{sequence_index}"
        
        robustness_scores = np.zeros(len(noise_levels))
        
        for i, noise_level in enumerate(noise_levels):
            result = self._test_sequence_recall(
                sequence_index=sequence_index if len(self.training_sequences) > 0 else 0,
                num_trials=num_trials,
                noise_level=noise_level,
                verbose=False
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
        测试所有序列的噪声鲁棒性
        
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
    
    # ========== 记忆管理方法 ==========
    
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
            
            eval_result = self._evaluate_single_replay(xi_replayed, seq, i)
            
            results[f'sequence_{i}'] = {
                'success': eval_result['found_sequence'],
                'success_rate': eval_result['recall_accuracy'],
                'sequence_length': len(seq)
            }
            
            if verbose:
                status = "✓ 成功" if eval_result['found_sequence'] else "✗ 失败"
                print(f"序列 #{i}: {status} (回放成功率: {eval_result['recall_accuracy']*100:.0f}%)")
        
        # 统计
        total_success = sum(1 for r in results.values() if r['success'])
        overall_rate = total_success / len(results)
        
        results['summary'] = {
            'total_sequences': len(results),
            'successful_recalls': total_success,
            'overall_success_rate': overall_rate
        }
        
        if verbose:
            print(f"\n总体成功率: {overall_rate*100:.1f}% "
                  f"({total_success}/{len(results)-1})")
            print("="*60)
        
        return results
    
    # ========== 可视化方法 ==========
    
    def visualize_training_results(self, xi_replayed: np.ndarray,
                                   eval_results: Dict,
                                   save_path: Optional[str] = None,
                                   title_suffix: str = "",
                                   show_images: bool = False,
                                   sequence_index: int = 0):
        """
        可视化训练和回放结果（9宫格布局）
        
        参数:
            xi_replayed: 回放序列
            eval_results: 评估结果
            save_path: 保存路径
            title_suffix: 标题后缀
            show_images: 是否显示图片
            sequence_index: 显示哪个序列
        """
        fig = plt.figure(figsize=(14, 9))
        
        num_epochs = len(self.mu_history)
        max_steps = xi_replayed.shape[0]
        
        # 确定训练序列
        is_multi = len(self.training_sequences) > 0
        if is_multi:
            training_seq = self.training_sequences[sequence_index]
            num_seq = len(self.training_sequences)
        else:
            training_seq = self.training_sequence
            num_seq = 1
        
        # 提取评估结果
        if 'evaluation_mode' in eval_results:
            if eval_results['evaluation_mode'] == 'full_sequence_matching':
                recall_acc = eval_results['recall_accuracy']
                match_indices = eval_results.get('match_indices', np.zeros(max_steps, dtype=int))
            elif eval_results['evaluation_mode'] == 'multiple_trials':
                recall_acc = eval_results['recall_accuracy']
                match_indices = np.zeros(max_steps, dtype=int)
        else:
            if 'recall_accuracy' in eval_results:
                match_indices = eval_results.get('match_indices', np.zeros(max_steps, dtype=int))
                recall_acc = eval_results['recall_accuracy']
            elif 'best_match' in eval_results:
                match_indices = eval_results['best_match'].get('match_indices', np.zeros(max_steps, dtype=int))
                recall_acc = eval_results['best_match']['recall_accuracy']
        
        # 子图1: 隐藏层训练误差
        ax1 = plt.subplot(3, 3, 1)
        plt.plot(range(1, num_epochs + 1), self.mu_history, 'b-', linewidth=2)
        plt.xlabel('训练轮数')
        plt.ylabel('平均误差 μ')
        plt.title('隐藏层训练误差')
        plt.grid(True, alpha=0.3)
        
        # 子图2: 可见层训练误差
        ax2 = plt.subplot(3, 3, 2)
        plt.plot(range(1, num_epochs + 1), self.nu_history, 'r-', linewidth=2)
        plt.xlabel('训练轮数')
        plt.ylabel('平均误差 ν')
        plt.title('可见层训练误差')
        plt.grid(True, alpha=0.3)
        
        # 子图3: 双误差对比
        ax3 = plt.subplot(3, 3, 3)
        plt.plot(range(1, num_epochs + 1), self.mu_history, 'b-', 
                 linewidth=1.5, label='μ (隐藏层)')
        plt.plot(range(1, num_epochs + 1), self.nu_history, 'r-', 
                 linewidth=1.5, label='ν (可见层)')
        plt.xlabel('训练轮数')
        plt.ylabel('平均误差')
        if num_seq > 1:
            plt.title(f'误差收敛曲线 ({num_seq}个序列)')
        else:
            plt.title('误差收敛曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图4: 训练序列
        ax4 = plt.subplot(3, 3, 4)
        plt.imshow(training_seq.T, cmap='gray', aspect='auto', interpolation='nearest')
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
        im6 = plt.imshow(self.U, cmap='jet', aspect='auto', interpolation='nearest')
        plt.colorbar(im6)
        plt.xlabel('可见神经元')
        plt.ylabel('隐藏神经元')
        plt.title('权重矩阵 U')
        
        # 子图7: V权重矩阵
        ax7 = plt.subplot(3, 3, 7)
        im7 = plt.imshow(self.V, cmap='jet', aspect='auto', interpolation='nearest')
        plt.colorbar(im7)
        plt.xlabel('隐藏神经元')
        plt.ylabel('可见神经元')
        plt.title('权重矩阵 V')
        
        # 子图8: P固定投影矩阵
        ax8 = plt.subplot(3, 3, 8)
        im8 = plt.imshow(self.P, cmap='jet', aspect='auto', interpolation='nearest')
        plt.colorbar(im8)
        plt.xlabel('可见神经元')
        plt.ylabel('隐藏神经元')
        plt.title('固定投影矩阵 P')
        
        # 子图9: 序列匹配追踪
        ax9 = plt.subplot(3, 3, 9)
        
        if 'evaluation_mode' in eval_results and \
           eval_results['evaluation_mode'] == 'full_sequence_matching':
            
            if 'match_indices' in eval_results:
                match_indices = eval_results['match_indices']
                max_steps = len(match_indices)
                
                plt.plot(range(1, max_steps + 1), match_indices, 'o', 
                        markersize=4, alpha=0.5, color='gray', label='逐帧匹配')
                
                if eval_results['found_sequence']:
                    match_start = eval_results['match_start_idx']
                    T = len(training_seq)
                    
                    complete_match_x = range(match_start + 1, match_start + T + 1)
                    complete_match_y = range(1, T + 1)
                    plt.plot(complete_match_x, complete_match_y, 'o-', 
                            linewidth=2, markersize=6, color='green', 
                            label='完整序列匹配')
                    
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
        
        elif 'evaluation_mode' in eval_results and eval_results['evaluation_mode'] == 'multiple_trials':
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
            plt.plot(range(1, max_steps + 1), match_indices, 'o-', 
                     linewidth=1.5, markersize=6)
            plt.xlabel('回放时间步')
            plt.ylabel('匹配的训练序列索引')
            plt.title(f'序列匹配追踪 (准确率: {recall_acc*100:.1f}%)')
            plt.ylim([0, len(training_seq) + 1])
        
        plt.grid(True, alpha=0.3)
        
        # 主标题
        main_title = f'模式复现网络训练与回放{title_suffix}'
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
    
    def visualize_multi_sequence_overview(self, save_path: Optional[str] = None,
                                         title_suffix: str = "",
                                         show_images: bool = False):
        """
        可视化多序列学习概览
        
        参数:
            save_path: 保存路径
            title_suffix: 标题后缀
            show_images: 是否显示图片
        """
        K = len(self.training_sequences)
        
        if K == 0:
            print("警告：没有训练序列可以可视化")
            return
        
        # 动态计算子图布局
        n_cols = min(K, 3)
        n_rows = (K + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows + 3))
        
        # 顶部：训练误差曲线
        ax_top = plt.subplot(n_rows + 1, 1, 1)
        num_epochs = len(self.mu_history)
        plt.plot(range(1, num_epochs + 1), self.mu_history, 'b-', 
                 linewidth=1.5, label='μ (隐藏层)')
        plt.plot(range(1, num_epochs + 1), self.nu_history, 'r-', 
                 linewidth=1.5, label='ν (可见层)')
        plt.xlabel('训练轮数')
        plt.ylabel('平均误差')
        plt.title(f'多序列训练误差收敛 ({K}个序列)', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 为每个序列创建回放测试
        for k in range(K):
            xi_replayed = self.replay(sequence_index=k, 
                                     max_steps=self.training_sequences[k].shape[0] * 2)
            eval_result = self.evaluate_replay(xi_replayed, sequence_index=k)
            
            # 训练序列
            ax_train = plt.subplot(n_rows + 1, n_cols * 2, n_cols * 2 + k * 2 + 1)
            plt.imshow(self.training_sequences[k].T, cmap='gray', 
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
        
        plt.suptitle(f'多序列学习结果概览{title_suffix}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"多序列概览图已保存: {save_path}")
        if show_images:
            plt.show()
        else:
            plt.close()
    
    def visualize_robustness_results(self, noise_levels: np.ndarray,
                                    robustness_results: Union[np.ndarray, Dict],
                                    save_path: Optional[str] = None,
                                    title_suffix: str = "",
                                    show_images: bool = False,
                                    labels: Optional[List[str]] = None):
        """
        可视化鲁棒性测试结果
        
        参数:
            noise_levels: 噪声水平数组
            robustness_results: 鲁棒性分数（单序列用数组，多序列用字典）
            save_path: 保存路径
            title_suffix: 标题后缀
            show_images: 是否显示图片
            labels: 自定义标签
        """
        plt.figure(figsize=(10, 6))
        
        if isinstance(robustness_results, np.ndarray):
            # 单序列模式
            plt.plot(noise_levels * 100, robustness_results * 100, '-o',
                     linewidth=2.5, markersize=8, color='#A23B72',
                     label='单序列')
            title = f'噪声鲁棒性测试{title_suffix}'
        
        elif isinstance(robustness_results, dict):
            # 多序列模式
            colors = plt.cm.tab10(np.linspace(0, 1, len(robustness_results)))
            
            for i, (seq_name, scores) in enumerate(robustness_results.items()):
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
            raise ValueError("robustness_results 必须是 np.ndarray 或 Dict")
        
        plt.xlabel('噪声水平 (%)', fontsize=12)
        plt.ylabel('完整序列回放成功率 (%)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 105])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"鲁棒性图已保存: {save_path}")
        if show_images:
            plt.show()
        else:
            plt.close()


# ========== 使用示例 ==========
if __name__ == "__main__":
    import os
    os.makedirs("ptrnrep_examples", exist_ok=True)
    
    print("\n" + "="*70)
    print("模式复现网络 (Pattern Reproduction Network) 示例")
    print("="*70)
    
    # ========== 示例1: 单序列学习 ==========
    print("\n【示例1】单序列学习")
    print("-"*60)
    
    network1 = PatternReproductionNetwork(N_v=50, T=30, N_h=200, eta=0.01)
    network1.train(num_epochs=300, seed=42, verbose=True)
    
    # 回放测试
    xi_replayed1 = network1.replay()
    eval_result1 = network1.evaluate_replay(xi_replayed1)
    
    print(f"\n回放成功: {eval_result1['found_sequence']}")
    print(f"准确率: {eval_result1['recall_accuracy']*100:.1f}%")
    
    # 可视化
    network1.visualize_training_results(
        xi_replayed=xi_replayed1,
        eval_results=eval_result1,
        save_path="ptrnrep_examples/example1_single_sequence.png",
        title_suffix="\n(单序列学习)",
        show_images=False
    )
    print("✓ 已保存: ptrnrep_examples/example1_single_sequence.png")
    
    # ========== 示例2: 多序列学习 ==========
    print("\n【示例2】多序列学习")
    print("-"*60)
    
    network2 = PatternReproductionNetwork(N_v=50, T=30, N_h=200, eta=0.01)
    sequences = network2.generate_multiple_sequences(
        num_sequences=3, 
        seeds=[100, 200, 300],
        ensure_unique_across=True
    )
    
    # 分析序列重叠
    network2.print_overlap_analysis(sequences)
    
    # 训练
    network2.train(x=sequences, num_epochs=400, verbose=True, interleaved=True)
    
    # 测试每个序列
    print("\n测试所有序列:")
    for k in range(len(sequences)):
        xi_replayed2 = network2.replay(sequence_index=k)
        eval_result2 = network2.evaluate_replay(xi_replayed2, sequence_index=k)
        
        status = "✓ 成功" if eval_result2['found_sequence'] else "✗ 失败"
        print(f"序列 #{k}: {status} (准确率: {eval_result2['recall_accuracy']*100:.1f}%)")
        
        # 可视化单个序列
        network2.visualize_training_results(
            xi_replayed=xi_replayed2,
            eval_results=eval_result2,
            save_path=f"ptrnrep_examples/example2_multi_seq{k}.png",
            title_suffix=f"\n(多序列学习 - 序列 #{k+1})",
            show_images=False,
            sequence_index=k
        )
        print(f"✓ 已保存: ptrnrep_examples/example2_multi_seq{k}.png")
    
    # 多序列概览
    network2.visualize_multi_sequence_overview(
        save_path="ptrnrep_examples/example2_multi_overview.png",
        title_suffix="\n(3个序列)",
        show_images=False
    )
    print("✓ 已保存: ptrnrep_examples/example2_multi_overview.png")
    
    # ========== 示例3: 鲁棒性测试 ==========
    print("\n【示例3】鲁棒性测试")
    print("-"*60)
    
    noise_levels = np.arange(0, 0.3, 0.05)
    
    # 测试所有序列的鲁棒性
    robustness_results = network2.test_robustness_all_sequences(
        noise_levels=noise_levels,
        num_trials=50,
        verbose=True
    )
    
    # 可视化鲁棒性结果
    network2.visualize_robustness_results(
        noise_levels=noise_levels,
        robustness_results=robustness_results,
        save_path="ptrnrep_examples/example3_robustness.png",
        title_suffix="\n(多序列鲁棒性)",
        show_images=False
    )
    print("✓ 已保存: ptrnrep_examples/example3_robustness.png")
    
    # ========== 示例4: 增量学习 ==========
    print("\n【示例4】增量学习")
    print("-"*60)
    
    network3 = PatternReproductionNetwork(N_v=50, T=30, N_h=200, eta=0.01)
    
    # 学习第一个序列
    print("\n学习序列1...")
    seq1 = network3.generate_random_sequence_with_length(T=30, seed=1000)
    network3.train(x=[seq1], num_epochs=300, verbose=False)
    
    # 测试记忆
    memory_test1 = network3.test_all_memories(verbose=True)
    
    # 增量学习第二个序列
    print("\n增量学习序列2（保持序列1的记忆）...")
    seq2 = network3.generate_random_sequence_with_length(T=30, seed=2000)
    network3.train(x=[seq2], num_epochs=300, verbose=False, incremental=True)
    
    # 测试记忆
    memory_test2 = network3.test_all_memories(verbose=True)
    
    # 再增量学习第三个序列
    print("\n增量学习序列3...")
    seq3 = network3.generate_random_sequence_with_length(T=30, seed=3000)
    network3.train(x=[seq3], num_epochs=300, verbose=False, incremental=True)
    
    # 测试记忆
    memory_test3 = network3.test_all_memories(verbose=True)
    
    # 可视化增量学习结果
    network3.visualize_multi_sequence_overview(
        save_path="ptrnrep_examples/example4_incremental.png",
        title_suffix="\n(增量学习 - 3个序列)",
        show_images=False
    )
    print("✓ 已保存: ptrnrep_examples/example4_incremental.png")
    

    # ========== 添加到 if __name__ == "__main__" 部分 ==========
# 放在示例4之后

    # ========== 示例5: 自定义重复模式测试 ==========
    print("\n【示例5】自定义重复模式测试")
    print("-"*60)
    
    network4 = PatternReproductionNetwork(N_v=50, T=40, N_h=250, eta=0.01)
    
    # 配置不同的模式
    pattern_configs = [
        {'pattern_type': 'alternating'},  # 交替模式
        {'pattern_type': 'periodic', 'period': 4},  # 4帧周期
        {'pattern_type': 'block', 'block_size': 5},  # 块大小5
        {'pattern_type': 'mirrored'},  # 镜像模式
        {'pattern_type': 'custom', 'custom_pattern': [0, 1, 2, 1, 0, 3]}  # 自定义
    ]
    
    pattern_names = [
        '交替模式 (A-B-A-B...)',
        '周期模式 (period=4)',
        '块状模式 (block_size=5)',
        '镜像模式 (A-B-C...C-B-A)',
        '自定义模式 [0,1,2,1,0,3]'
    ]
    
    # 生成模式序列
    print("\n生成5种不同重复模式的序列...")
    patterned_sequences = network4.generate_multiple_patterned_sequences(
        num_sequences=5,
        pattern_configs=pattern_configs,
        seeds=[5000, 5001, 5002, 5003, 5004]
    )
    
    # 分析每个序列的模式结构
    print("\n模式结构分析:")
    for i, (seq, name) in enumerate(zip(patterned_sequences, pattern_names)):
        analysis = network4.analyze_pattern_structure(seq)
        print(f"\n  {name}:")
        print(f"    唯一帧数: {analysis['unique_frames']}/{analysis['total_frames']}")
        print(f"    重复率: {analysis['repetition_rate']*100:.1f}%")
        print(f"    最大重复次数: {analysis['max_repetitions']}")
        if analysis['detected_period']:
            print(f"    检测到周期: {analysis['detected_period']}")
    
    # 可视化模式对比
    network4.visualize_pattern_comparison(
        sequences=patterned_sequences,
        pattern_names=pattern_names,
        save_path="ptrnrep_examples/example5_pattern_comparison.png",
        show_images=False
    )
    print("\n✓ 已保存: ptrnrep_examples/example5_pattern_comparison.png")
    
    # 训练多模式序列
    print("\n训练多模式序列网络...")
    network4.train(x=patterned_sequences, num_epochs=500, verbose=False, interleaved=True)
    
    # 测试每种模式的回放
    print("\n测试各模式序列回放:")
    pattern_recall_results = []
    for k, name in enumerate(pattern_names):
        xi_replayed = network4.replay(sequence_index=k, max_steps=network4.T * 2)
        eval_result = network4.evaluate_replay(xi_replayed, sequence_index=k)
        pattern_recall_results.append(eval_result['recall_accuracy'])
        
        status = "✓ 成功" if eval_result['found_sequence'] else "✗ 失败"
        print(f"  {name}: {status} (准确率: {eval_result['recall_accuracy']*100:.1f}%)")
        
        # 可视化单个模式
        network4.visualize_training_results(
            xi_replayed=xi_replayed,
            eval_results=eval_result,
            save_path=f"ptrnrep_examples/example5_pattern_{k}.png",
            title_suffix=f"\n({name})",
            show_images=False,
            sequence_index=k
        )
    
    # 可视化所有模式的训练结果
    network4.visualize_multi_sequence_overview(
        save_path="ptrnrep_examples/example5_patterns_overview.png",
        title_suffix="\n(5种重复模式)",
        show_images=False
    )
    print("✓ 已保存: ptrnrep_examples/example5_patterns_overview.png")
    
    # 绘制模式回放准确率对比图
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    bars = ax.bar(range(len(pattern_names)), 
                   [r*100 for r in pattern_recall_results],
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('模式类型', fontsize=12)
    ax.set_ylabel('回放准确率 (%)', fontsize=12)
    ax.set_title('不同重复模式的回放性能对比', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(pattern_names)))
    ax.set_xticklabels([name.split('(')[0].strip() for name in pattern_names], 
                        rotation=15, ha='right')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, pattern_recall_results)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("ptrnrep_examples/example5_pattern_performance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: ptrnrep_examples/example5_pattern_performance.png")
    
    # 测试模式鲁棒性
    print("\n测试模式序列的噪声鲁棒性...")
    noise_levels = np.arange(0, 0.25, 0.05)
    pattern_robustness = network4.test_robustness_all_sequences(
        noise_levels=noise_levels,
        num_trials=30,
        verbose=False
    )
    
    network4.visualize_robustness_results(
        noise_levels=noise_levels,
        robustness_results=pattern_robustness,
        save_path="ptrnrep_examples/example5_pattern_robustness.png",
        title_suffix="\n(不同重复模式)",
        show_images=False,
        labels=[name.split('(')[0].strip() for name in pattern_names]
    )
    print("✓ 已保存: ptrnrep_examples/example5_pattern_robustness.png")
    
    print("\n" + "="*70)
    print("模式测试完成！生成了以下文件：")
    print("  - example5_pattern_comparison.png (模式可视化)")
    print("  - example5_pattern_*.png (各模式训练详情)")
    print("  - example5_patterns_overview.png (所有模式概览)")
    print("  - example5_pattern_performance.png (性能对比)")
    print("  - example5_pattern_robustness.png (鲁棒性对比)")
    print("="*70)

    print("\n" + "="*70)
    print("所有示例完成！")
    print("生成的文件保存在 ptrnrep_examples/ 文件夹中")
    print("="*70)
