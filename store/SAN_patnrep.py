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
    确保非共享区域不重叠  
    """  
    
    def __init__(self, N_v: int, T: int, N_h: Optional[int] = None,   
                 eta: float = 0.001, kappa: float = 1):  
        """初始化网络（调用父类构造函数）"""  
        super().__init__(N_v, T, N_h, eta, kappa)  
        # 存储模式重复信息  
        self.pattern_info = {}  
    
    def generate_sequences_with_shared_patterns(  
        self,  
        num_sequences: int,  
        pattern_config: Optional[Dict] = None,  
        seeds: Optional[List[int]] = None,  
        T: Optional[int] = None,  
        ensure_unique_non_shared: bool = True,  
        max_attempts: int = 1000  
    ) -> List[np.ndarray]:  
        """  
        生成包含共享模式的多个序列（确保非共享区域唯一）  
        
        参数:  
            num_sequences: 序列数量  
            pattern_config: 模式配置字典  
            seeds: 随机种子列表（可选）  
            T: 序列长度（可选，默认使用self.T）  
            ensure_unique_non_shared: 是否确保非共享区域也不重叠（默认True）  
            max_attempts: 生成唯一帧的最大尝试次数  
            
        返回:  
            序列列表  
        """  
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
        
        print(f"生成 {num_sequences} 个序列，确保非共享区域唯一...")  
        
        # 1. 识别所有共享位置  
        shared_positions = self._get_all_shared_positions(pattern_config, num_sequences, seq_length)  
        
        # 2. 生成共享模式  
        shared_patterns = self._generate_all_shared_patterns(pattern_config)  
        
        # 3. 初始化序列  
        sequences = []  
        all_used_frames = []  # 存储所有已使用的非共享帧  
        
        for seq_idx in range(num_sequences):  
            if seeds[seq_idx] is not None:  
                np.random.seed(seeds[seq_idx])  
            
            print(f"  正在生成序列 #{seq_idx}...", end=" ")  
            
            # 初始化序列  
            seq = np.zeros((seq_length, self.N_v))  
            shared_pos_set = shared_positions.get(seq_idx, set())  
            
            # 生成非共享位置的帧  
            for t in range(seq_length - 1):  # 最后一帧单独处理（周期性）  
                if t in shared_pos_set:  
                    # 共享位置，先跳过，稍后填充  
                    continue  
                
                # 生成唯一的非共享帧  
                attempts = 0  
                while attempts < max_attempts:  
                    # 生成候选帧  
                    candidate_frame = np.sign(np.random.randn(self.N_v))  
                    candidate_frame[candidate_frame == 0] = 1  
                    
                    # 检查与当前序列内部已生成帧的重叠  
                    is_unique_within = True  
                    for prev_t in range(t):  
                        if prev_t not in shared_pos_set:  # 只与非共享位置比较  
                            if np.array_equal(candidate_frame, seq[prev_t, :]):  
                                is_unique_within = False  
                                break  
                    
                    if not is_unique_within:  
                        attempts += 1  
                        continue  
                    
                    # 检查与所有其他序列的非共享帧的重叠  
                    is_unique_across = True  
                    if ensure_unique_non_shared:  
                        for used_frame, used_seq_idx, used_pos in all_used_frames:  
                            if np.array_equal(candidate_frame, used_frame):  
                                is_unique_across = False  
                                break  
                    
                    if is_unique_across:  
                        # 找到唯一帧  
                        seq[t, :] = candidate_frame  
                        all_used_frames.append((candidate_frame.copy(), seq_idx, t))  
                        break  
                    
                    attempts += 1  
                
                if attempts >= max_attempts:  
                    print(f"\n警告: 序列 #{seq_idx} 位置 {t} 无法生成唯一帧（尝试{max_attempts}次）")  
                    # 使用当前候选帧（可能重复）  
                    seq[t, :] = candidate_frame  
                    all_used_frames.append((candidate_frame.copy(), seq_idx, t))  
            
            # 处理最后一帧（周期性：与第一帧相同）  
            if (seq_length - 1) not in shared_pos_set:  
                seq[seq_length - 1, :] = seq[0, :]  
            
            sequences.append(seq)  
            print("完成")  
        
        # 4. 应用共享模式  
        print("  应用共享模式...")  
        self._apply_shared_patterns_from_generated(  
            sequences, pattern_config, shared_patterns, seq_length  
        )  
        
        # 5. 保存模式信息  
        self.pattern_info = {  
            'config': pattern_config,  
            'num_sequences': num_sequences,  
            'sequence_length': seq_length,  
            'shared_positions': shared_positions,  
            'ensure_unique_non_shared': ensure_unique_non_shared  
        }  
        
        print("所有序列生成完毕\n")  
        
        # 6. 验证并报告  
        if ensure_unique_non_shared:  
            self._verify_uniqueness(sequences, shared_positions)  
        
        return sequences  
    
    def _get_all_shared_positions(self, config: Dict, num_sequences: int, seq_length: int) -> Dict:  
        """  
        获取所有共享位置的映射  
        
        返回:  
            字典，键为序列索引，值为该序列中所有共享位置的集合  
        """  
        shared_positions = {i: set() for i in range(num_sequences)}  
        
        for group_idx, seq_group in enumerate(config['shared_sequences']):  
            positions_group = config['pattern_positions'][group_idx]  
            n_patterns = config['num_patterns'][group_idx]  
            
            for seq_idx_in_group, global_seq_idx in enumerate(seq_group):  
                positions = positions_group[seq_idx_in_group]  
                
                for pattern_idx in range(n_patterns):  
                    start_pos = positions[pattern_idx * 2]  
                    end_pos = positions[pattern_idx * 2 + 1]  
                    
                    # 记录该序列的共享位置  
                    for pos in range(start_pos, end_pos + 1):  
                        if pos < seq_length - 1:  # 不包括最后一帧（周期性）  
                            shared_positions[global_seq_idx].add(pos)  
        
        return shared_positions  
    
    def _generate_all_shared_patterns(self, config: Dict) -> Dict:  
        """  
        预先生成所有共享模式  
        
        返回:  
            字典，键为 (group_idx, pattern_idx)，值为模式数组  
        """  
        shared_patterns = {}  
        
        for group_idx, seq_group in enumerate(config['shared_sequences']):  
            n_patterns = config['num_patterns'][group_idx]  
            
            for pattern_idx in range(n_patterns):  
                # 生成单个时间步的随机模式  
                pattern = np.sign(np.random.randn(1, self.N_v))  
                pattern[pattern == 0] = 1  
                shared_patterns[(group_idx, pattern_idx)] = pattern  
        
        return shared_patterns  
    
    def _apply_shared_patterns_from_generated(self, sequences: List[np.ndarray],   
                                             config: Dict,   
                                             shared_patterns: Dict,  
                                             seq_length: int):  
        """  
        将预生成的共享模式应用到序列的指定位置  
        """  
        shared_sequences = config['shared_sequences']  
        num_patterns = config['num_patterns']  
        pattern_positions = config['pattern_positions']  
        
        for group_idx, seq_group in enumerate(shared_sequences):  
            n_patterns = num_patterns[group_idx]  
            positions_group = pattern_positions[group_idx]  
            
            # 将模式插入到每个序列的指定位置  
            for seq_idx_in_group, global_seq_idx in enumerate(seq_group):  
                positions = positions_group[seq_idx_in_group]  
                
                for pattern_idx in range(n_patterns):  
                    start_pos = positions[pattern_idx * 2]  
                    end_pos = positions[pattern_idx * 2 + 1]  
                    
                    if start_pos > end_pos:  
                        raise ValueError(f"起始位置 {start_pos} 不能大于结束位置 {end_pos}")  
                    
                    # 获取共享模式  
                    pattern = shared_patterns[(group_idx, pattern_idx)]  
                    pattern_length = end_pos - start_pos + 1  
                    
                    # 如果需要，重复模式以填充长度  
                    if pattern_length > len(pattern):  
                        repeated_pattern = np.tile(pattern, (pattern_length // len(pattern) + 1, 1))  
                        pattern_to_insert = repeated_pattern[:pattern_length, :]  
                    else:  
                        pattern_to_insert = pattern[:pattern_length, :]  
                    
                    # 插入模式（保持序列首尾相同的周期性约束）  
                    if end_pos < seq_length - 1:  
                        sequences[global_seq_idx][start_pos:end_pos+1, :] = pattern_to_insert  
                    else:  
                        # 如果包含最后一帧，需要确保与第一帧相同  
                        sequences[global_seq_idx][start_pos:seq_length-1, :] = \
                            pattern_to_insert[:seq_length-1-start_pos, :]  
                        sequences[global_seq_idx][-1, :] = sequences[global_seq_idx][0, :]  
    
    def _verify_uniqueness(self, sequences: List[np.ndarray],   
                          shared_positions: Dict):  
        """  
        验证非共享区域的唯一性  
        """  
        print("\n验证非共享区域唯一性...")  
        
        num_sequences = len(sequences)  
        seq_length = sequences[0].shape[0]  
        
        # 收集所有非共享帧  
        non_shared_frames = []  
        
        for seq_idx, seq in enumerate(sequences):  
            shared_pos_set = shared_positions.get(seq_idx, set())  
            
            for t in range(seq_length - 1):  
                if t not in shared_pos_set:  
                    frame = seq[t, :]  
                    non_shared_frames.append((frame, seq_idx, t))  
        
        # 检查重复  
        duplicates_found = []  
        for i in range(len(non_shared_frames)):  
            frame_i, seq_i, pos_i = non_shared_frames[i]  
            for j in range(i + 1, len(non_shared_frames)):  
                frame_j, seq_j, pos_j = non_shared_frames[j]  
                if np.array_equal(frame_i, frame_j):  
                    duplicates_found.append(  
                        (seq_i, pos_i, seq_j, pos_j)  
                    )  
        
        if duplicates_found:  
            print(f"⚠️  发现 {len(duplicates_found)} 处非共享区域重复:")  
            for seq_i, pos_i, seq_j, pos_j in duplicates_found[:5]:  # 只显示前5个  
                print(f"    序列 #{seq_i} 位置 {pos_i} 与 序列 #{seq_j} 位置 {pos_j} 重复")  
            if len(duplicates_found) > 5:  
                print(f"    ... 还有 {len(duplicates_found) - 5} 处重复")  
        else:  
            print("✓ 非共享区域完全唯一")  
        
        # 统计共享区域  
        total_shared = sum(len(pos_set) for pos_set in shared_positions.values())  
        total_frames = num_sequences * (seq_length - 1)  
        total_non_shared = total_frames - total_shared  
        
        print(f"\n统计信息:")  
        print(f"  总帧数: {total_frames}")  
        print(f"  共享帧数: {total_shared} ({total_shared/total_frames*100:.1f}%)")  
        print(f"  非共享帧数: {total_non_shared} ({total_non_shared/total_frames*100:.1f}%)")  
        print(f"  非共享重复数: {len(duplicates_found)}")  
    
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
    
    def generate_sequences_with_custom_patterns(  
        self,  
        num_sequences: int,  
        shared_groups: List[List[int]],  
        patterns_per_group: List[int],  
        positions_per_group: List[List[List[Tuple[int, int]]]],  
        seeds: Optional[List[int]] = None,  
        T: Optional[int] = None,  
        ensure_unique_non_shared: bool = True  
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
            T=T,  
            ensure_unique_non_shared=ensure_unique_non_shared  
        )  

    # ========== 评估方法（保持不变）==========  
    
    def _test_sequence_recall_success_rate(self, sequence_index: int,  
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
    
    def test_robustness(self, noise_levels: np.ndarray,   
                       num_trials: int = 50,   
                       verbose: bool = True,  
                       sequence_index: int = 0) -> np.ndarray:  
        """测试噪声鲁棒性"""  
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
    
    # ========== 可视化方法（保持不变）==========  
    
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
        report.append(f"非共享区域唯一性: {'已确保' if self.pattern_info.get('ensure_unique_non_shared', False) else '未确保'}")  
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

    network1.train(x=sequences1, num_epochs=300, verbose=True, V_only=False, interleaved=True)

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
    visualize_multi_sequence_results(network1, 
                                    save_path='pattern_examples/example1_mutitest1_all_sequences.png',
                                    title_suffix="\n(修正评估方法)",
                                    show_images=False)
    
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
                                     save_path='pattern_examples/example1_sigletestsum_all_sequences.png',
                                     title_suffix="\n(修正评估方法)",
                                     show_images=False)
    
    print("\n" + "="*70)
    print("示例2: 自定义配置")
    print("="*70)
    network2 = PatternRepeatingSequenceNetwork(N_v=60, T=40, eta=0.005)
    # 自定义配置  
    sequences2 = network2.generate_sequences_with_custom_patterns(  
        num_sequences=5,  
        shared_groups=[[0, 1, 2], [3, 4]],  # 两组共享  
        patterns_per_group=[2, 1],           # 第一组2个模式，第二组1个模式  
        positions_per_group=[  
            [  # 第一组的位置  
                [(10, 12), (25, 27)],  # 序列0: 模式在位置10-12和25-27  
                [(15, 17), (30, 32)],  # 序列1: 模式在位置15-17和30-32  
                [(8, 10), (20, 22)]    # 序列2: 模式在位置8-10和20-22  
            ],  
            [  # 第二组的位置  
                [(18, 20)],  # 序列3: 模式在位置18-20  
                [(22, 24)]   # 序列4: 模式在位置22-24  
            ]  
        ],  
        seeds=[1000, 2000, 3000, 4000, 5000]  
    )  
    
    print(f"生成了 {len(sequences2)} 个序列")  
    print(network2.get_pattern_overlap_report())  
    
    # 训练  
    network2.train(x=sequences2, num_epochs=400, verbose=True)  
    
    # 可视化  
    network2.visualize_pattern_info(  
        save_path="pattern_examples/example2_custom_pattern.png",  
        show_images=False  
    )  
    
    # 测试回放  
    print("\n回放测试:")  
    for k in range(len(sequences2)):  
        xi_replayed = network2.replay(sequence_index=k)  
        eval_result = network2.evaluate_replay(xi_replayed, sequence_index=k)  
        print(f"序列 #{k}: 回放准确率 {eval_result['recall_accuracy']*100:.1f}%")  
        visualize_results(network2, xi_replayed, eval_result,  
                         save_path=f'pattern_examples/example2_replay_{k}.png',
                         show_images=False)

    print("\n" + "="*70)  
    print("示例3: 复杂配置（多组多模式，不同位置）")  
    print("="*70)  
    
    network3 = PatternRepeatingSequenceNetwork(N_v=60, T=50, N_h=250, eta=0.01)  
    
    sequences3 = network3.generate_sequences_with_custom_patterns(  
        num_sequences=6,  
        shared_groups=[[0, 1], [2, 3, 4], [5]],  # 三组  
        patterns_per_group=[3, 2, 0],             # 不同数量的模式  
        positions_per_group=[  
            [  # 组1: 序列0,1共享3个模式  
                [(5, 7), (20, 22), (40, 42)],  
                [(10, 12), (25, 27), (45, 47)]  
            ],  
            [  # 组2: 序列2,3,4共享2个模式  
                [(8, 10), (30, 32)],  
                [(12, 14), (35, 37)],  
                [(15, 17), (38, 40)]  
            ],  
            [  # 组3: 序列5没有共享模式  
                []  
            ]  
        ],  
        seeds=list(range(6000, 6006))  
    )  
    
    print(f"生成了 {len(sequences3)} 个序列")  
    print(network3.get_pattern_overlap_report())  
    
    # 训练  
    network3.train(x=sequences3, num_epochs=500, verbose=True)  
    # network3.train(x=sequences3[5], num_epochs=100, verbose=True)  
    # 可视化  
    network3.visualize_pattern_info(  
        save_path="pattern_examples/example3_complex_pattern.png",  
        show_images=False  
    )  
    
    # 回放测试  
    print("\n回放测试:")  
    accuracies = []  
    for k in range(len(sequences3)):  
        xi_replayed = network3.replay(sequence_index=k)  
        eval_result = network3.evaluate_replay(xi_replayed, sequence_index=k)  
        acc = eval_result['recall_accuracy']  
        accuracies.append(acc)  
        print(f"序列 #{k}: 回放准确率 {acc*100:.1f}%")  
        visualize_results(network3, xi_replayed, eval_result,  
                         save_path=f'pattern_examples/example3_replay_{k}.png',
                         show_images=False)
    # visualize_multi_sequence_results(network3,  
    #                                  save_path='pattern_examples/example3_all_replays.png',  
    #                                  title_suffix="\n(修正评估方法)",  
    #                                  show_images=False)


    print(f"\n平均回放准确率: {np.mean(accuracies)*100:.1f}%")  
    print("\n" + "="*70)
    print("所有示例完成！")
    print("="*70)
