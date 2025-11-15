"""
================================================================
模式重复序列吸引子网络
支持学习具有重复模式的序列，并提供模式分析功能
================================================================
"""

import numpy as np
from typing import Optional, Dict, List, Union, Tuple
from .memory import MemorySequenceAttractorNetwork


class PatternRepetitionNetwork(MemorySequenceAttractorNetwork):
    """
    模式重复序列吸引子网络
    在多序列网络基础上添加模式重复分析和生成功能
    """
    
    def __init__(self, N_v: int, T: int, N_h: Optional[int] = None, 
                 eta: float = 0.001, kappa: float = 1, seed: Optional[int] = None):
        """初始化网络"""
        super().__init__(N_v, T, N_h, eta, kappa, seed=seed)
        
        # 模式重复专用属性
        self.pattern_info = {}  # 存储模式信息
    
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
    
    def generate_sequences_with_shared_patterns(
        self,
        num_sequences: int,
        pattern_config: Optional[Dict] = None,
        seeds: Optional[List[int]] = None,
        T: Optional[int] = None,
        ensure_unique_non_shared: bool = True,
        max_attempts: int = 1000,
        verbose: bool = True
    ) -> List[np.ndarray]:
        """
        生成包含共享模式的多个序列（支持非共享区域唯一性约束）
        """
        seq_length = T if T is not None else self.T
        if seeds is None:
            seeds = list(range(num_sequences))
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
        self._validate_pattern_config(pattern_config, num_sequences, seq_length)
        shared_positions = self._get_all_shared_positions(pattern_config, num_sequences, seq_length)
        shared_patterns = self._generate_all_shared_patterns(pattern_config)
        sequences = []
        all_used_frames: List[Tuple[np.ndarray, int, int]] = []
        if verbose:
            print(f"生成 {num_sequences} 个序列，确保非共享区域唯一...")
        for seq_idx in range(num_sequences):
            seed = seeds[seq_idx] if seq_idx < len(seeds) else None
            if seed is not None:
                np.random.seed(seed)
            if verbose:
                print(f"  正在生成序列 #{seq_idx}...", end=" ")
            seq = np.zeros((seq_length, self.N_v))
            shared_pos_set = shared_positions.get(seq_idx, set())
            for t in range(seq_length - 1):
                if t in shared_pos_set:
                    continue
                attempts = 0
                candidate_frame = None
                while attempts < max_attempts:
                    candidate_frame = np.sign(np.random.randn(self.N_v))
                    candidate_frame[candidate_frame == 0] = 1
                    is_unique_within = True
                    for prev_t in range(t):
                        if prev_t in shared_pos_set:
                            continue
                        if np.array_equal(candidate_frame, seq[prev_t, :]):
                            is_unique_within = False
                            break
                    if not is_unique_within:
                        attempts += 1
                        continue
                    is_unique_across = True
                    if ensure_unique_non_shared:
                        for used_frame, _, _ in all_used_frames:
                            if np.array_equal(candidate_frame, used_frame):
                                is_unique_across = False
                                break
                    if is_unique_across:
                        seq[t, :] = candidate_frame
                        all_used_frames.append((candidate_frame.copy(), seq_idx, t))
                        break
                    attempts += 1
                if attempts >= max_attempts and candidate_frame is not None:
                    if verbose:
                        print(f"\n警告: 序列 #{seq_idx} 位置 {t} 无法生成唯一帧（尝试 {max_attempts} 次）")
                    seq[t, :] = candidate_frame
                    all_used_frames.append((candidate_frame.copy(), seq_idx, t))
            if (seq_length - 1) not in shared_pos_set:
                seq[seq_length - 1, :] = seq[0, :]
            sequences.append(seq)
            if verbose:
                print("完成")
        if verbose:
            print("  应用共享模式...")
        self._apply_shared_patterns_from_generated(sequences, pattern_config, shared_patterns, seq_length)
        self.pattern_info = {
            'config': pattern_config,
            'num_sequences': num_sequences,
            'sequence_length': seq_length,
            'shared_positions': shared_positions,
            'ensure_unique_non_shared': ensure_unique_non_shared
        }
        if ensure_unique_non_shared:
            self._verify_non_shared_uniqueness(sequences, shared_positions, verbose=verbose)
        if verbose:
            print("所有序列生成完毕\n")
        return sequences
    
    def generate_sequences_with_custom_patterns(
        self,
        num_sequences: int,
        shared_groups: List[List[int]],
        patterns_per_group: List[int],
        positions_per_group: List[List[List[Tuple[int, int]]]],
        seeds: Optional[List[int]] = None,
        T: Optional[int] = None,
        ensure_unique_non_shared: bool = True,
        verbose: bool = True
    ) -> List[np.ndarray]:
        """
        使用直观配置生成包含共享模式的序列
        """
        pattern_positions_flat: List[List[Tuple[int, ...]]] = []
        for group_positions in positions_per_group:
            group_flat: List[Tuple[int, ...]] = []
            for seq_positions in group_positions:
                flat_positions: List[int] = []
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
            ensure_unique_non_shared=ensure_unique_non_shared,
            verbose=verbose
        )
    
    def _validate_pattern_config(self, config: Dict, num_sequences: int, seq_length: int):
        """验证共享模式配置的有效性"""
        shared_sequences = config.get('shared_sequences', [])
        num_patterns = config.get('num_patterns', [])
        pattern_positions = config.get('pattern_positions', [])
        if not (len(shared_sequences) == len(num_patterns) == len(pattern_positions)):
            raise ValueError("shared_sequences, num_patterns, pattern_positions 长度必须一致")
        for group in shared_sequences:
            for seq_idx in group:
                if seq_idx >= num_sequences:
                    raise ValueError(f"序列索引 {seq_idx} 超出范围 [0, {num_sequences - 1}]")
        for group_idx, positions_group in enumerate(pattern_positions):
            n_patterns = num_patterns[group_idx]
            n_seqs_in_group = len(shared_sequences[group_idx])
            if len(positions_group) != n_seqs_in_group:
                raise ValueError(
                    f"第 {group_idx} 组位置数量 ({len(positions_group)}) 与序列数量 ({n_seqs_in_group}) 不匹配"
                )
            for seq_positions in positions_group:
                if len(seq_positions) != n_patterns * 2:
                    raise ValueError(
                        f"第 {group_idx} 组位置参数错误，需提供 {n_patterns * 2} 个位置值"
                    )
                for pos in seq_positions:
                    if not (0 <= pos < seq_length):
                        raise ValueError(f"位置 {pos} 超出有效范围 [0, {seq_length - 1}]")
    
    def _get_all_shared_positions(
        self,
        config: Dict,
        num_sequences: int,
        seq_length: int
    ) -> Dict[int, set]:
        """计算每个序列的共享位置集合"""
        shared_positions: Dict[int, set] = {i: set() for i in range(num_sequences)}
        for group_idx, seq_group in enumerate(config['shared_sequences']):
            positions_group = config['pattern_positions'][group_idx]
            n_patterns = config['num_patterns'][group_idx]
            for seq_idx_in_group, global_seq_idx in enumerate(seq_group):
                positions = positions_group[seq_idx_in_group]
                for pattern_idx in range(n_patterns):
                    start = positions[pattern_idx * 2]
                    end = positions[pattern_idx * 2 + 1]
                    for pos in range(start, end + 1):
                        if pos < seq_length - 1:
                            shared_positions[global_seq_idx].add(pos)
        return shared_positions
    
    def _generate_all_shared_patterns(self, config: Dict) -> Dict[Tuple[int, int], np.ndarray]:
        """为所有共享模式生成基础帧"""
        shared_patterns: Dict[Tuple[int, int], np.ndarray] = {}
        for group_idx, _ in enumerate(config['shared_sequences']):
            n_patterns = config['num_patterns'][group_idx]
            for pattern_idx in range(n_patterns):
                pattern = np.sign(np.random.randn(1, self.N_v))
                pattern[pattern == 0] = 1
                shared_patterns[(group_idx, pattern_idx)] = pattern
        return shared_patterns
    
    def _apply_shared_patterns_from_generated(
        self,
        sequences: List[np.ndarray],
        config: Dict,
        shared_patterns: Dict[Tuple[int, int], np.ndarray],
        seq_length: int
    ):
        """将共享模式写入指定序列位置"""
        shared_sequences = config['shared_sequences']
        pattern_positions = config['pattern_positions']
        for group_idx, seq_group in enumerate(shared_sequences):
            n_patterns = config['num_patterns'][group_idx]
            positions_group = pattern_positions[group_idx]
            for seq_idx_in_group, global_seq_idx in enumerate(seq_group):
                positions = positions_group[seq_idx_in_group]
                for pattern_idx in range(n_patterns):
                    start = positions[pattern_idx * 2]
                    end = positions[pattern_idx * 2 + 1]
                    if start > end:
                        raise ValueError(f"起始位置 {start} 不能大于结束位置 {end}")
                    pattern = shared_patterns[(group_idx, pattern_idx)]
                    pattern_length = end - start + 1
                    if pattern_length > len(pattern):
                        repeats = pattern_length // len(pattern) + 1
                        pattern_to_insert = np.tile(pattern, (repeats, 1))[:pattern_length, :]
                    else:
                        pattern_to_insert = pattern[:pattern_length, :]
                    if end < seq_length - 1:
                        sequences[global_seq_idx][start:end + 1, :] = pattern_to_insert
                    else:
                        valid_length = seq_length - 1 - start
                        if valid_length > 0:
                            sequences[global_seq_idx][start:seq_length - 1, :] = pattern_to_insert[:valid_length, :]
                        sequences[global_seq_idx][-1, :] = sequences[global_seq_idx][0, :]
    
    def _verify_non_shared_uniqueness(
        self,
        sequences: List[np.ndarray],
        shared_positions: Dict[int, set],
        verbose: bool = True
    ):
        """验证非共享区域是否满足唯一性"""
        if verbose:
            print("\n验证非共享区域唯一性...")
        non_shared_frames: List[Tuple[np.ndarray, int, int]] = []
        for seq_idx, seq in enumerate(sequences):
            shared_pos_set = shared_positions.get(seq_idx, set())
            for t in range(seq.shape[0] - 1):
                if t not in shared_pos_set:
                    non_shared_frames.append((seq[t, :], seq_idx, t))
        duplicates_found: List[Tuple[int, int, int, int]] = []
        for i in range(len(non_shared_frames)):
            frame_i, seq_i, pos_i = non_shared_frames[i]
            for j in range(i + 1, len(non_shared_frames)):
                frame_j, seq_j, pos_j = non_shared_frames[j]
                if np.array_equal(frame_i, frame_j):
                    duplicates_found.append((seq_i, pos_i, seq_j, pos_j))
        if verbose:
            if duplicates_found:
                print(f"⚠️  发现 {len(duplicates_found)} 处非共享区域重复:")
                for seq_i, pos_i, seq_j, pos_j in duplicates_found[:5]:
                    print(f"    序列 #{seq_i} 位置 {pos_i} 与 序列 #{seq_j} 位置 {pos_j} 重复")
                if len(duplicates_found) > 5:
                    print(f"    ... 还有 {len(duplicates_found) - 5} 处重复")
            else:
                print("✓ 非共享区域完全唯一")
            total_shared = sum(len(pos_set) for pos_set in shared_positions.values())
            seq_length = sequences[0].shape[0]
            total_frames = len(sequences) * (seq_length - 1)
            total_non_shared = total_frames - total_shared
            print("\n统计信息:")
            print(f"  总帧数: {total_frames}")
            print(f"  共享帧数: {total_shared} ({(total_shared / total_frames * 100) if total_frames > 0 else 0:.1f}%)")
            print(f"  非共享帧数: {total_non_shared} ({(total_non_shared / total_frames * 100) if total_frames > 0 else 0:.1f}%)")
            print(f"  非共享重复数: {len(duplicates_found)}")
    
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
        max_repetitions = max(len(positions) for positions in repetition_map.values()) if repetition_map else 0
        
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
            'repetition_rate': 1 - (unique_frames / (T - 1)) if (T - 1) > 0 else 0.0,
            'detected_period': detected_period,
            'frame_positions': repetition_map
        }
    
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
        
        overlap_info['overlap_rate'] = overlap_info['duplicate_frames'] / overlap_info['total_frames'] if overlap_info['total_frames'] > 0 else 0.0
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

    def get_pattern_overlap_report(self) -> str:
        """
        根据已保存的 pattern_info 生成共享模式报告
        """
        if not self.pattern_info:
            return "没有可用的模式信息"
        config = self.pattern_info['config']
        lines = []
        lines.append("=" * 60)
        lines.append("模式重复配置报告")
        lines.append("=" * 60)
        lines.append(f"序列总数: {self.pattern_info.get('num_sequences')}")
        lines.append(f"序列长度: {self.pattern_info.get('sequence_length')}")
        lines.append(f"非共享区域唯一性: {'已确保' if self.pattern_info.get('ensure_unique_non_shared') else '未确保'}")
        lines.append("")
        for group_idx, seq_group in enumerate(config.get('shared_sequences', [])):
            lines.append(f"共享组 {group_idx}:")
            lines.append(f"  参与序列: {seq_group}")
            lines.append(f"  共享模式数: {config['num_patterns'][group_idx]}")
            lines.append("  模式位置:")
            positions_group = config['pattern_positions'][group_idx]
            n_patterns = config['num_patterns'][group_idx]
            for seq_idx_in_group, global_seq_idx in enumerate(seq_group):
                positions = positions_group[seq_idx_in_group]
                pos_desc = []
                for pattern_idx in range(n_patterns):
                    start = positions[pattern_idx * 2]
                    end = positions[pattern_idx * 2 + 1]
                    pos_desc.append(f"模式{pattern_idx}:[{start},{end}]")
                lines.append(f"    序列 #{global_seq_idx}: {', '.join(pos_desc)}")
            lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def visualize_pattern_info(self, save_path: Optional[str] = None, show_images: bool = True):
        """
        可视化共享模式的时间位置分布
        """
        if not self.training_sequences or not self.pattern_info:
            print("警告：没有训练序列或模式信息")
            return
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError as exc:
            raise ImportError("需要 matplotlib 才能使用可视化功能") from exc
        config = self.pattern_info['config']
        num_sequences = self.pattern_info['num_sequences']
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

