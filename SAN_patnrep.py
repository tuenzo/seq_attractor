"""
================================================================
序列吸引子网络 - 模式重复扩展版
支持多序列间共享重复模式
================================================================
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from SAN_multi_seq_1 import MultiSequenceAttractorNetwork, visualize_results,visualize_multi_sequence_results


class PatternRepeatingSequenceNetwork(MultiSequenceAttractorNetwork):
    """
    模式重复序列网络（继承扩展版）
    在 MultiSequenceAttractorNetwork 基础上添加序列间模式重复功能
    """
    
    def __init__(self, N_v: int, T: int, N_h: Optional[int] = None, 
                 eta: float = 0.001, kappa: float = 1):
        """
        初始化网络（调用父类构造函数）
        """
        super().__init__(N_v, T, N_h, eta, kappa)
        # 存储模式重复信息
        self.pattern_info = {}
    
    def generate_sequences_with_shared_patterns(
        self,
        num_sequences: int,
        pattern_config: Optional[Dict] = None,
        seeds: Optional[List[int]] = None,
        T: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        生成包含共享模式的多个序列
        
        参数:
            num_sequences: 序列数量
            pattern_config: 模式配置字典，格式为：
                {
                    'shared_sequences': List[List[int]],  # 哪些序列共享模式，如 [[0,1], [2,3]]
                    'num_patterns': List[int],            # 每组共享几个模式
                    'pattern_positions': List[List[Tuple[int, int]]]  # 模式在各序列中的位置
                }
                默认配置：前两个序列在中间位置共享1个模式
            seeds: 随机种子列表
            T: 序列长度（默认使用self.T）
        
        返回:
            序列列表
        """
        seq_length = T if T is not None else self.T
        
        if seeds is None:
            seeds = list(range(num_sequences))
        
        # 默认配置：前两个序列共享1个模式，位置在中间
        if pattern_config is None:
            pattern_config = {
                'shared_sequences': [[0, 1]],  # 序列0和序列1共享
                'num_patterns': [1],            # 共享1个模式
                'pattern_positions': [
                    [
                        (seq_length // 2, seq_length // 2),  # 序列0的中间位置
                        (seq_length // 2, seq_length // 2)   # 序列1的中间位置
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
        
        # 检查各项长度一致
        if not (len(shared_sequences) == len(num_patterns) == len(pattern_positions)):
            raise ValueError("shared_sequences, num_patterns, pattern_positions 长度必须一致")
        
        # 检查序列索引有效性
        for group in shared_sequences:
            for seq_idx in group:
                if seq_idx >= num_sequences:
                    raise ValueError(f"序列索引 {seq_idx} 超出范围 [0, {num_sequences-1}]")
        
        # 检查位置有效性
        for group_idx, positions_group in enumerate(pattern_positions):
            n_patterns = num_patterns[group_idx]
            n_seqs_in_group = len(shared_sequences[group_idx])
            
            if len(positions_group) != n_seqs_in_group:
                raise ValueError(
                    f"第 {group_idx} 组：位置列表长度 ({len(positions_group)}) "
                    f"必须等于该组序列数量 ({n_seqs_in_group})"
                )
            
            for seq_positions in positions_group:
                if len(seq_positions) != n_patterns * 2:  # 每个模式需要起始和结束位置
                    raise ValueError(
                        f"位置元组数量错误：预期 {n_patterns} 个模式需要 {n_patterns*2} 个位置值"
                    )
                
                # 检查位置范围
                for pos in seq_positions:
                    if not (0 <= pos < seq_length):
                        raise ValueError(f"位置 {pos} 超出序列长度范围 [0, {seq_length-1}]")
    
    def _apply_shared_patterns(self, sequences: List[np.ndarray], 
                               config: Dict, seq_length: int):
        """
        应用共享模式到序列
        
        参数:
            sequences: 序列列表（会被就地修改）
            config: 模式配置
            seq_length: 序列长度
        """
        shared_sequences = config['shared_sequences']
        num_patterns = config['num_patterns']
        pattern_positions = config['pattern_positions']
        
        # 遍历每组共享配置
        for group_idx, seq_group in enumerate(shared_sequences):
            n_patterns = num_patterns[group_idx]
            positions_group = pattern_positions[group_idx]
            
            # 为这组序列生成共享模式
            shared_patterns = self._generate_shared_patterns(n_patterns, self.N_v)
            
            # 将模式插入到每个序列的指定位置
            for seq_idx_in_group, global_seq_idx in enumerate(seq_group):
                positions = positions_group[seq_idx_in_group]
                
                # 解析位置（每个模式需要两个值：start, end）
                for pattern_idx in range(n_patterns):
                    start_pos = positions[pattern_idx * 2]
                    end_pos = positions[pattern_idx * 2 + 1]
                    
                    if start_pos > end_pos:
                        raise ValueError(f"起始位置 {start_pos} 不能大于结束位置 {end_pos}")
                    
                    # 插入模式
                    pattern = shared_patterns[pattern_idx]
                    pattern_length = end_pos - start_pos + 1
                    
                    # 如果模式长度不够，重复模式
                    if pattern_length > len(pattern):
                        repeated_pattern = np.tile(pattern, (pattern_length // len(pattern) + 1, 1))
                        pattern = repeated_pattern[:pattern_length, :]
                    else:
                        pattern = pattern[:pattern_length, :]
                    
                    # 应用模式（保持序列首尾相同的周期性约束）
                    if end_pos < seq_length - 1:  # 不覆盖最后一帧（周期性）
                        sequences[global_seq_idx][start_pos:end_pos+1, :] = pattern
                    else:
                        # 如果包含最后一帧，需要确保与第一帧相同
                        sequences[global_seq_idx][start_pos:seq_length-1, :] = pattern[:seq_length-1-start_pos, :]
                        sequences[global_seq_idx][-1, :] = sequences[global_seq_idx][0, :]
    
    def _generate_shared_patterns(self, num_patterns: int, N_v: int) -> List[np.ndarray]:
        """
        生成共享模式列表
        
        参数:
            num_patterns: 模式数量
            N_v: 可见层神经元数量
        
        返回:
            模式列表，每个模式是一个 (pattern_length, N_v) 的数组
        """
        patterns = []
        for _ in range(num_patterns):
            # 生成单个时间步的随机模式
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
        """
        使用更直观的参数生成包含共享模式的序列
        
        参数:
            num_sequences: 序列总数
            shared_groups: 共享组列表，如 [[0,1,2], [3,4]] 表示序列0,1,2共享，3,4共享
            patterns_per_group: 每组共享的模式数量，如 [2, 1]
            positions_per_group: 每组中每个序列的模式位置
                格式：[
                    [  # 第一组
                        [(start1, end1), (start2, end2)],  # 序列0的2个模式位置
                        [(start1, end1), (start2, end2)],  # 序列1的2个模式位置
                        [(start1, end1), (start2, end2)]   # 序列2的2个模式位置
                    ],
                    [  # 第二组
                        [(start1, end1)],  # 序列3的1个模式位置
                        [(start1, end1)]   # 序列4的1个模式位置
                    ]
                ]
            seeds: 随机种子
            T: 序列长度
        
        返回:
            序列列表
        """
        # 转换为标准配置格式
        pattern_positions_flat = []
        for group_positions in positions_per_group:
            group_flat = []
            for seq_positions in group_positions:
                # 将 [(start1, end1), (start2, end2)] 展平为 [start1, end1, start2, end2]
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
    
    def visualize_pattern_info(self, save_path: Optional[str] = None, 
                              show_images: bool = True):
        """
        可视化模式重复信息
        
        参数:
            save_path: 保存路径
            show_images: 是否显示图像
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
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
        
        # 绘制每个序列
        for seq_idx, seq in enumerate(self.training_sequences):
            ax = axes[seq_idx]
            ax.imshow(seq.T, cmap='gray', aspect='auto', interpolation='nearest')
            ax.set_ylabel('神经元索引')
            ax.set_xlabel('时间步')
            ax.set_title(f'序列 #{seq_idx}')
            
            # 标记共享模式区域
            for group_idx, seq_group in enumerate(config['shared_sequences']):
                if seq_idx in seq_group:
                    # 找到该序列在组内的索引
                    seq_idx_in_group = seq_group.index(seq_idx)
                    positions = config['pattern_positions'][group_idx][seq_idx_in_group]
                    n_patterns = config['num_patterns'][group_idx]
                    
                    # 为每个模式绘制边框
                    for pattern_idx in range(n_patterns):
                        start_pos = positions[pattern_idx * 2]
                        end_pos = positions[pattern_idx * 2 + 1]
                        
                        # 绘制矩形框
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
                        
                        # 添加标签
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
        """
        生成模式重叠报告
        
        返回:
            格式化的报告字符串
        """
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
    print("示例1: 默认配置（前两个序列中间共享1个模式）")
    print("="*70)
    
    network1 = PatternRepeatingSequenceNetwork(N_v=50, T=30, eta=0.01)
    
    # 使用默认配置生成序列
    sequences1 = network1.generate_sequences_with_shared_patterns(
        num_sequences=3,
        seeds=[100, 200, 300]
    )
    
    print(f"生成了 {len(sequences1)} 个序列")
    print(network1.get_pattern_overlap_report())
    
    # 训练网络
    network1.train(x=sequences1, num_epochs=300, verbose=True)
    
    # 可视化模式信息
    network1.visualize_pattern_info(
        save_path="pattern_examples/example1_default_pattern.png",
        show_images=False
    )
    
    # 测试回放
    for k in range(len(sequences1)):
        xi_replayed = network1.replay(sequence_index=k)
        eval_result = network1.evaluate_replay(xi_replayed, sequence_index=k)
        print(f"序列 #{k}: 回放准确率 {eval_result['recall_accuracy']*100:.1f}%")
        visualize_results(network1, xi_replayed, eval_result, save_path=f'pattern_examples/example1_replay_{k}.png')
    visualize_multi_sequence_results(network1, 
                                     save_path='pattern_examples/example1_all_sequences.png',
                                     title_suffix="\n(三个序列回放结果汇总)",
                                     show_images=False
                                     )
    
    
    print("\n" + "="*70)
    print("示例2: 自定义配置（序列0,1,2共享2个模式，序列3,4共享1个模式）")
    print("="*70)
    
    network2 = PatternRepeatingSequenceNetwork(N_v=50, T=40, eta=0.01)
    
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
        visualize_results(network1, xi_replayed, eval_result, save_path=f'pattern_examples/example1_replay_{k}.png')
    visualize_multi_sequence_results(network2, 
                                     save_path='pattern_examples/example2_all_sequences.png',
                                     title_suffix="\n(五个序列回放结果汇总)",
                                     show_images=False
                                     )
    
    
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
        visualize_results(network3, xi_replayed, eval_result, save_path=f'pattern_examples/example3_replay_{k}.png')
    visualize_multi_sequence_results(network3, 
                                     save_path='pattern_examples/example3_all_sequences.png',
                                     title_suffix="\n(六个序列回放结果汇总)",
                                     show_images=False
                                     )

    print(f"\n平均回放准确率: {np.mean(accuracies)*100:.1f}%")
    
    print("\n" + "="*70)
    print("所有示例完成！结果已保存到 pattern_examples/ 文件夹")
    print("="*70)
