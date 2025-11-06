"""
================================================================
序列吸引子网络 - 模式重复扩展版
支持多序列间共享重复模式（修正评估方法）
================================================================
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from SAN_multi_seq_1 import MultiSequenceAttractorNetwork, visualize_results, visualize_multi_sequence_results
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class PatternRepeatingSequenceNetwork(MultiSequenceAttractorNetwork):
    """
    模式重复序列网络（继承扩展版）
    在 MultiSequenceAttractorNetwork 基础上添加序列间模式重复功能
    """
    
    def __init__(self, N_v: int, T: int, N_h: Optional[int] = None, 
                 eta: float = 0.001, kappa: float = 1):
        """初始化网络（调用父类构造函数）"""
        super().__init__(N_v, T, N_h, eta, kappa)
        # 存储模式重复信息
        self.pattern_info = {}
    
    # ========== 保持原有的模式生成方法不变 ==========
    
    def generate_sequences_with_shared_patterns(
        self,
        num_sequences: int,
        pattern_config: Optional[Dict] = None,
        seeds: Optional[List[int]] = None,
        T: Optional[int] = None
    ) -> List[np.ndarray]:
        """生成包含共享模式的多个序列"""
        seq_length = T if T is not None else self.T
        
        if seeds is None:
            seeds = list(range(num_sequences))
        
        # 默认配置
        if pattern_config is None:
            pattern_config = {
                'shared_sequences': [[0, 1]],
                'num_patterns': [1],
                'pattern_positions': [
                    [
                        (seq_length // 2, seq_length // 2),
                        (seq_length // 2, seq_length // 2)
                    ]
                ]
            }
        
        # 验证配置
        self._validate_pattern_config(pattern_config, num_sequences, seq_length)
        
        # 生成基础序列
        sequences = []
        for i in range(num_sequences):
            seq = self.generate_random_sequence_with_length(T=seq_length, seed=seeds[i])
            sequences.append(seq)
        
        # 应用共享模式
        self._apply_shared_patterns(sequences, pattern_config, seq_length)
        
        # 保存模式信息
        self.pattern_info = {
            'config': pattern_config,
            'num_sequences': num_sequences,
            'sequence_length': seq_length
        }
        
        return sequences
    
    def _validate_pattern_config(self, config: Dict, num_sequences: int, seq_length: int):
        """验证模式配置的有效性"""
        shared_sequences = config.get('shared_sequences', [])
        num_patterns = config.get('num_patterns', [])
        pattern_positions = config.get('pattern_positions', [])
        
        if not (len(shared_sequences) == len(num_patterns) == len(pattern_positions)):
            raise ValueError("shared_sequences, num_patterns, pattern_positions 长度必须一致")
        
        for group in shared_sequences:
            for seq_idx in group:
                if seq_idx >= num_sequences:
                    raise ValueError(f"序列索引 {seq_idx} 超出范围 [0, {num_sequences-1}]")
        
        for group_idx, positions_group in enumerate(pattern_positions):
            n_patterns = num_patterns[group_idx]
            n_seqs_in_group = len(shared_sequences[group_idx])
            
            if len(positions_group) != n_seqs_in_group:
                raise ValueError(
                    f"第 {group_idx} 组：位置列表长度 ({len(positions_group)}) "
                    f"必须等于该组序列数量 ({n_seqs_in_group})"
                )
            
            for seq_positions in positions_group:
                if len(seq_positions) != n_patterns * 2:
                    raise ValueError(
                        f"位置元组数量错误：预期 {n_patterns} 个模式需要 {n_patterns*2} 个位置值"
                    )
                
                for pos in seq_positions:
                    if not (0 <= pos < seq_length):
                        raise ValueError(f"位置 {pos} 超出序列长度范围 [0, {seq_length-1}]")
    
    def _apply_shared_patterns(self, sequences: List[np.ndarray], 
                               config: Dict, seq_length: int):
        """应用共享模式到序列"""
        shared_sequences = config['shared_sequences']
        num_patterns = config['num_patterns']
        pattern_positions = config['pattern_positions']
        
        for group_idx, seq_group in enumerate(shared_sequences):
            n_patterns = num_patterns[group_idx]
            positions_group = pattern_positions[group_idx]
            
            shared_patterns = self._generate_shared_patterns(n_patterns, self.N_v)
            
            for seq_idx_in_group, global_seq_idx in enumerate(seq_group):
                positions = positions_group[seq_idx_in_group]
                
                for pattern_idx in range(n_patterns):
                    start_pos = positions[pattern_idx * 2]
                    end_pos = positions[pattern_idx * 2 + 1]
                    
                    if start_pos > end_pos:
                        raise ValueError(f"起始位置 {start_pos} 不能大于结束位置 {end_pos}")
                    
                    pattern = shared_patterns[pattern_idx]
                    pattern_length = end_pos - start_pos + 1
                    
                    if pattern_length > len(pattern):
                        repeated_pattern = np.tile(pattern, (pattern_length // len(pattern) + 1, 1))
                        pattern = repeated_pattern[:pattern_length, :]
                    else:
                        pattern = pattern[:pattern_length, :]
                    
                    if end_pos < seq_length - 1:
                        sequences[global_seq_idx][start_pos:end_pos+1, :] = pattern
                    else:
                        sequences[global_seq_idx][start_pos:seq_length-1, :] = pattern[:seq_length-1-start_pos, :]
                        sequences[global_seq_idx][-1, :] = sequences[global_seq_idx][0, :]
    
    def _generate_shared_patterns(self, num_patterns: int, N_v: int) -> List[np.ndarray]:
        """生成共享模式列表"""
        patterns = []
        for _ in range(num_patterns):
            pattern = np.sign(np.random.randn(1, N_v))
            pattern[pattern == 0] = 1
            patterns.append(pattern)
        return patterns
    
    def generate_sequences_with_custom_patterns(
        self,
        num_sequences: int,
        shared_groups: List[List[int]],
        patterns_per_group: List[int],
        positions_per_group: List[List[List[Tuple[int, int]]]],
        seeds: Optional[List[int]] = None,
        T: Optional[int] = None
    ) -> List[np.ndarray]:
        """使用更直观的参数生成包含共享模式的序列"""
        pattern_positions_flat = []
        for group_positions in positions_per_group:
            group_flat = []
            for seq_positions in group_positions:
                flat_positions = []
                for start, end in seq_positions:
                    flat_positions.extend([start, end])
                group_flat.append(tuple(flat_positions))
            pattern_positions_flat.append(group_flat)
        
        pattern_config = {
            'shared_sequences': shared_groups,
            'num_patterns': patterns_per_group,
            'pattern_positions': pattern_positions_flat
        }
        
        return self.generate_sequences_with_shared_patterns(
            num_sequences=num_sequences,
            pattern_config=pattern_config,
            seeds=seeds,
            T=T
        )
    
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

    def _test_sequence_recall_success_rate(self, sequence_index: int,
                                          num_trials: int = 50,
                                          noise_level: float = 0.0,
                                          verbose: bool = False) -> Dict:
        """
        测试单个序列的回放成功率（类似 test_robustness 的评估方式）
        """
        assert sequence_index < len(self.training_sequences), \
            f"序列索引 {sequence_index} 超出范围"
        
        target_sequence = self.training_sequences[sequence_index]
        T = len(target_sequence)
        max_search_steps = T * 5
        
        success_count = 0
        convergence_steps = []
        trajectory = np.zeros((max_search_steps + 1, self.N_v))
        
        for trial in range(num_trials):
            # 1. 生成初始状态
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
            print(f'序列 #{sequence_index}, 噪声水平 {noise_level:.2f}: '
                  f'成功率 {success_rate*100:.1f}% ({success_count}/{num_trials} 次成功)')
            if convergence_steps:
                print(f'  平均收敛步数: {np.mean(convergence_steps):.1f}')
                print(f'  收敛步数范围: [{np.min(convergence_steps)}, {np.max(convergence_steps)}]')
        
        return {
            'success_rate': success_rate,
            'recall_accuracy': success_rate,  # 向后兼容
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
        """
        测试噪声鲁棒性（使用正确的评估方式）
        """
        assert sequence_index < len(self.training_sequences), \
            f"序列索引 {sequence_index} 超出范围"
        
        robustness_scores = np.zeros(len(noise_levels))
        
        for i, noise_level in enumerate(noise_levels):
            result = self._test_sequence_recall_success_rate(
                sequence_index=sequence_index,
                num_trials=num_trials,
                noise_level=noise_level,
                verbose=False
            )
            robustness_scores[i] = result['success_rate']
            
            if verbose:
                print(f'序列 #{sequence_index}, 噪声水平 {noise_level:.2f}: '
                      f'成功率 {robustness_scores[i]*100:.1f}% '
                      f'({result["success_count"]}/{num_trials} 次成功)')
        
        return robustness_scores
    
    # ========== 保持原有的可视化方法 ==========
    
    def visualize_pattern_info(self, save_path: Optional[str] = None, 
                              show_images: bool = True):
        """可视化模式重复信息"""
        if not self.training_sequences or not self.pattern_info:
            print("警告：没有训练序列或模式信息")
            return
        
        config = self.pattern_info['config']
        num_sequences = self.pattern_info['num_sequences']
        seq_length = self.pattern_info['sequence_length']
        
        fig, axes = plt.subplots(num_sequences, 1, figsize=(12, 3 * num_sequences))
        if num_sequences == 1:
            axes = [axes]
        
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for seq_idx, seq in enumerate(self.training_sequences):
            ax = axes[seq_idx]
            ax.imshow(seq.T, cmap='gray', aspect='auto', interpolation='nearest')
            ax.set_ylabel('神经元索引')
            ax.set_xlabel('时间步')
            ax.set_title(f'序列 #{seq_idx}')
            
            for group_idx, seq_group in enumerate(config['shared_sequences']):
                if seq_idx in seq_group:
                    seq_idx_in_group = seq_group.index(seq_idx)
                    positions = config['pattern_positions'][group_idx][seq_idx_in_group]
                    n_patterns = config['num_patterns'][group_idx]
                    
                    for pattern_idx in range(n_patterns):
                        start_pos = positions[pattern_idx * 2]
                        end_pos = positions[pattern_idx * 2 + 1]
                        
                        rect = mpatches.Rectangle(
                            (start_pos - 0.5, -0.5),
                            end_pos - start_pos + 1,
                            self.N_v,
                            linewidth=3,
                            edgecolor=colors[group_idx % 10],
                            facecolor='none',
                            linestyle='--'
                        )
                        ax.add_patch(rect)
                        
                        ax.text(
                            (start_pos + end_pos) / 2,
                            self.N_v + 1,
                            f'组{group_idx}-模式{pattern_idx}',
                            ha='center',
                            va='bottom',
                            color=colors[group_idx % 10],
                            fontweight='bold',
                            fontsize=9
                        )
        
        plt.suptitle('序列中的共享模式标注', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图片已保存: {save_path}")
        
        if show_images:
            plt.show()
        else:
            plt.close()
    
    def get_pattern_overlap_report(self) -> str:
        """生成模式重叠报告"""
        if not self.pattern_info:
            return "没有可用的模式信息"
        
        config = self.pattern_info['config']
        report = []
        report.append("=" * 60)
        report.append("模式重复配置报告")
        report.append("=" * 60)
        report.append(f"序列总数: {self.pattern_info['num_sequences']}")
        report.append(f"序列长度: {self.pattern_info['sequence_length']}")
        report.append("")
        
        for group_idx, seq_group in enumerate(config['shared_sequences']):
            report.append(f"共享组 {group_idx}:")
            report.append(f"  参与序列: {seq_group}")
            report.append(f"  共享模式数: {config['num_patterns'][group_idx]}")
            report.append("  模式位置:")
            
            for seq_idx_in_group, global_seq_idx in enumerate(seq_group):
                positions = config['pattern_positions'][group_idx][seq_idx_in_group]
                n_patterns = config['num_patterns'][group_idx]
                
                pos_str = []
                for pattern_idx in range(n_patterns):
                    start = positions[pattern_idx * 2]
                    end = positions[pattern_idx * 2 + 1]
                    pos_str.append(f"模式{pattern_idx}:[{start},{end}]")
                
                report.append(f"    序列 #{global_seq_idx}: {', '.join(pos_str)}")
            report.append("")
        
        report.append("=" * 60)
        return "\n".join(report)


# ========== 使用示例 ==========
if __name__ == "__main__":
    import os
    os.makedirs("pattern_examples", exist_ok=True)
    
    print("\n" + "="*70)
    print("示例1: 默认配置（使用修正后的评估方法）")
    print("="*70)
    
    network1 = PatternRepeatingSequenceNetwork(N_v=50, T=30, eta=0.01)
    
    sequences1 = network1.generate_sequences_with_shared_patterns(
        num_sequences=3,
        seeds=[100, 200, 300]
    )
    
    print(f"生成了 {len(sequences1)} 个序列")
    print(network1.get_pattern_overlap_report())
    
    network1.train(x=sequences1, num_epochs=300, verbose=True)
    
    network1.visualize_pattern_info(
        save_path="pattern_examples/example1_default_pattern.png",
        show_images=False
    )
    
    # 方式1: 多次试验评估（推荐）
    print("\n=== 方式1: 多次试验评估（正确方式）===")
    for k in range(len(sequences1)):
        eval_result = network1.evaluate_replay(
            xi_replayed=None,  # 触发多次试验
            sequence_index=k,
            num_trials=50,
            noise_level=0.0,
            verbose=True
        )
    
    # 方式2: 单次回放可视化
    print("\n=== 方式2: 单次回放可视化 ===")
    for k in range(len(sequences1)):
        xi_replayed = network1.replay(sequence_index=k)
        eval_result = network1.evaluate_replay(xi_replayed, sequence_index=k)
        status = "✓ 成功" if eval_result['found_sequence'] else "✗ 失败"
        print(f"序列 #{k}: {status}")
        visualize_results(network1, xi_replayed, eval_result, 
                         save_path=f'pattern_examples/example1_replay_{k}.png',
                         show_images=False)
    
    visualize_multi_sequence_results(network1, 
                                     save_path='pattern_examples/example1_all_sequences.png',
                                     title_suffix="\n(修正评估方法)",
                                     show_images=False)
    
    print("\n" + "="*70)
    print("所有示例完成！")
    print("="*70)
