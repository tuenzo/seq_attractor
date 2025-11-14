"""
工具函数测试
"""

import pytest
import numpy as np
from src.utils.evaluation import (
    evaluate_replay_full_sequence,
    evaluate_replay_frame_matching
)


class TestEvaluationFunctions:
    """评估函数测试类"""
    
    def test_evaluate_replay_full_sequence_perfect_match(self):
        """测试完整序列匹配评估（完美匹配）"""
        T, N_v = 5, 10
        target_sequence = np.sign(np.random.randn(T, N_v))
        target_sequence[target_sequence == 0] = 1
        
        # 创建完美匹配的回放序列
        max_steps = 20
        xi_replayed = np.zeros((max_steps, N_v))
        xi_replayed[5:5+T, :] = target_sequence  # 在中间插入目标序列
        
        evaluation = evaluate_replay_full_sequence(xi_replayed, target_sequence)
        
        assert evaluation['found_sequence'] == True
        assert evaluation['recall_accuracy'] == 1.0
        assert evaluation['match_start_idx'] == 5
        assert evaluation['evaluation_mode'] == 'full_sequence_matching'
    
    def test_evaluate_replay_full_sequence_no_match(self):
        """测试完整序列匹配评估（无匹配）"""
        T, N_v = 5, 10
        target_sequence = np.sign(np.random.randn(T, N_v))
        target_sequence[target_sequence == 0] = 1
        
        # 创建完全不匹配的回放序列
        max_steps = 20
        xi_replayed = np.sign(np.random.randn(max_steps, N_v))
        xi_replayed[xi_replayed == 0] = 1
        
        evaluation = evaluate_replay_full_sequence(xi_replayed, target_sequence)
        
        assert evaluation['found_sequence'] == False
        assert evaluation['recall_accuracy'] == 0.0
        assert evaluation['match_start_idx'] == -1
    
    def test_evaluate_replay_full_sequence_with_frame_matching(self):
        """测试包含逐帧匹配信息的评估"""
        T, N_v = 5, 10
        target_sequence = np.sign(np.random.randn(T, N_v))
        target_sequence[target_sequence == 0] = 1
        
        max_steps = 20
        xi_replayed = np.zeros((max_steps, N_v))
        xi_replayed[5:5+T, :] = target_sequence
        
        evaluation = evaluate_replay_full_sequence(
            xi_replayed, 
            target_sequence,
            include_frame_matching=True
        )
        
        assert 'match_indices' in evaluation
        assert 'frame_match_count' in evaluation
        assert 'frame_recall_accuracy' in evaluation
        assert len(evaluation['match_indices']) == max_steps
        assert evaluation['frame_recall_accuracy'] >= 0
        assert evaluation['frame_recall_accuracy'] <= 1
    
    def test_evaluate_replay_full_sequence_without_frame_matching(self):
        """测试不包含逐帧匹配信息的评估"""
        T, N_v = 5, 10
        target_sequence = np.sign(np.random.randn(T, N_v))
        target_sequence[target_sequence == 0] = 1
        
        max_steps = 20
        xi_replayed = np.zeros((max_steps, N_v))
        xi_replayed[5:5+T, :] = target_sequence
        
        evaluation = evaluate_replay_full_sequence(
            xi_replayed,
            target_sequence,
            include_frame_matching=False
        )
        
        assert 'match_indices' not in evaluation
        assert 'found_sequence' in evaluation
    
    def test_evaluate_replay_frame_matching(self):
        """测试逐帧匹配评估"""
        T, N_v = 5, 10
        target_sequence = np.sign(np.random.randn(T, N_v))
        target_sequence[target_sequence == 0] = 1
        
        max_steps = 20
        xi_replayed = np.zeros((max_steps, N_v))
        # 在多个位置插入目标序列的帧
        for i in range(0, max_steps, 3):
            if i + T <= max_steps:
                xi_replayed[i:i+T, :] = target_sequence
        
        evaluation = evaluate_replay_frame_matching(xi_replayed, target_sequence)
        
        assert 'recall_accuracy' in evaluation
        assert 'match_count' in evaluation
        assert 'match_indices' in evaluation
        assert 'evaluation_mode' in evaluation
        assert evaluation['evaluation_mode'] == 'frame_matching'
        assert 0 <= evaluation['recall_accuracy'] <= 1
        assert evaluation['match_count'] >= 0
        assert len(evaluation['match_indices']) == max_steps
    
    def test_evaluate_replay_frame_matching_no_matches(self):
        """测试逐帧匹配评估（无匹配）"""
        T, N_v = 5, 10
        target_sequence = np.sign(np.random.randn(T, N_v))
        target_sequence[target_sequence == 0] = 1
        
        max_steps = 20
        xi_replayed = np.sign(np.random.randn(max_steps, N_v))
        xi_replayed[xi_replayed == 0] = 1
        
        evaluation = evaluate_replay_frame_matching(xi_replayed, target_sequence)
        
        assert evaluation['recall_accuracy'] == 0.0
        assert evaluation['match_count'] == 0
        assert np.all(evaluation['match_indices'] == 0)
    
    def test_evaluate_replay_full_sequence_partial_match(self):
        """测试部分匹配的情况"""
        T, N_v = 5, 10
        target_sequence = np.sign(np.random.randn(T, N_v))
        target_sequence[target_sequence == 0] = 1
        
        max_steps = 20
        xi_replayed = np.sign(np.random.randn(max_steps, N_v))
        xi_replayed[xi_replayed == 0] = 1
        
        # 只在部分位置匹配
        xi_replayed[10:12, :] = target_sequence[0:2, :]
        
        evaluation = evaluate_replay_full_sequence(xi_replayed, target_sequence)
        
        # 部分匹配不应该被识别为完整序列匹配
        assert evaluation['found_sequence'] == False
        assert evaluation['recall_accuracy'] == 0.0
    
    def test_evaluate_replay_full_sequence_multiple_matches(self):
        """测试多个匹配的情况（应该返回第一个）"""
        T, N_v = 5, 10
        target_sequence = np.sign(np.random.randn(T, N_v))
        target_sequence[target_sequence == 0] = 1
        
        max_steps = 30
        xi_replayed = np.zeros((max_steps, N_v))
        # 在多个位置插入完整序列
        xi_replayed[5:5+T, :] = target_sequence
        xi_replayed[15:15+T, :] = target_sequence
        
        evaluation = evaluate_replay_full_sequence(xi_replayed, target_sequence)
        
        assert evaluation['found_sequence'] == True
        assert evaluation['match_start_idx'] == 5  # 应该返回第一个匹配位置
    
    def test_evaluate_replay_sequence_shorter_than_target(self):
        """测试回放序列比目标序列短的情况"""
        T, N_v = 10, 10
        target_sequence = np.sign(np.random.randn(T, N_v))
        target_sequence[target_sequence == 0] = 1
        
        max_steps = 5  # 比目标序列短
        xi_replayed = np.sign(np.random.randn(max_steps, N_v))
        xi_replayed[xi_replayed == 0] = 1
        
        evaluation = evaluate_replay_full_sequence(xi_replayed, target_sequence)
        
        # 应该无法找到完整序列
        assert evaluation['found_sequence'] == False
        assert evaluation['recall_accuracy'] == 0.0
    
    def test_evaluate_replay_different_dimensions(self):
        """测试不同维度的情况"""
        T, N_v = 5, 10
        target_sequence = np.sign(np.random.randn(T, N_v))
        target_sequence[target_sequence == 0] = 1
        
        max_steps = 20
        xi_replayed = np.zeros((max_steps, N_v))
        xi_replayed[5:5+T, :] = target_sequence
        
        evaluation = evaluate_replay_full_sequence(xi_replayed, target_sequence)
        
        assert evaluation['found_sequence'] == True

