"""
================================================================
评估工具函数
提供序列回放质量的评估方法
================================================================
"""

import numpy as np
from typing import Optional, Dict


def evaluate_replay_full_sequence(xi_replayed: np.ndarray,
                                  target_sequence: np.ndarray,
                                  include_frame_matching: bool = True) -> Dict:
    """
    评估回放质量（完整序列匹配）
    
    参数:
        xi_replayed: 回放序列
        target_sequence: 目标训练序列
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
    
    return result


def evaluate_replay_frame_matching(xi_replayed: np.ndarray,
                                   target_sequence: np.ndarray) -> Dict:
    """
    评估回放质量（逐帧匹配方式，向后兼容）
    
    参数:
        xi_replayed: 回放序列
        target_sequence: 目标训练序列
        
    返回:
        评估结果字典
    """
    max_steps = xi_replayed.shape[0]
    T = len(target_sequence)
    
    match_indices = np.zeros(max_steps, dtype=int)
    
    for step in range(max_steps):
        for t in range(T):
            if np.all(xi_replayed[step, :] == target_sequence[t, :]):
                match_indices[step] = t + 1
                break
    
    match_count = np.sum(match_indices > 0)
    recall_accuracy = match_count / max_steps
    
    return {
        'recall_accuracy': recall_accuracy,
        'match_count': match_count,
        'match_indices': match_indices,
        'evaluation_mode': 'frame_matching'
    }

