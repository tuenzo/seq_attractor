"""
基础使用示例
演示如何使用重构后的代码结构
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import (
    SequenceAttractorNetwork,
    MultiSequenceAttractorNetwork,
    IncrementalSequenceAttractorNetwork,
    PatternRepetitionNetwork,
    visualize_training_results,
    visualize_robustness,
    visualize_multi_sequence_overview
)
import numpy as np


def example_basic_usage():
    """示例1: 基础单序列学习"""
    print("\n" + "="*60)
    print("示例1: 基础单序列学习")
    print("="*60)
    
    # 创建网络
    network = SequenceAttractorNetwork(N_v=50, T=30, N_h=200, eta=0.01, kappa=1)
    
    # 训练
    print("\n开始训练...")
    train_results = network.train(num_epochs=300, seed=42, verbose=True)
    
    # 回放
    print("\n测试回放...")
    xi_replayed = network.replay()
    
    # 评估
    eval_result = network.evaluate_replay(xi_replayed)
    print(f"\n回放结果:")
    print(f"  找到完整序列: {eval_result.get('found_sequence', False)}")
    print(f"  准确率: {eval_result['recall_accuracy']*100:.1f}%")
    
    # 可视化
    visualize_training_results(
        network, 
        xi_replayed, 
        eval_result,
        save_path="basic_example_result.png",
        show_images=False
    )
    print("\n✓ 结果已保存到 basic_example_result.png")


def example_multi_sequence():
    """示例2: 多序列学习"""
    print("\n" + "="*60)
    print("示例2: 多序列学习")
    print("="*60)
    
    # 创建网络
    network = MultiSequenceAttractorNetwork(N_v=50, T=30, N_h=200, eta=0.01)
    
    # 生成多个序列
    print("\n生成多个序列...")
    sequences = network.generate_multiple_sequences(
        num_sequences=3, 
        seeds=[100, 200, 300],
        ensure_unique_across=True
    )
    
    # 训练
    print("\n开始多序列训练...")
    train_results = network.train(
        x=sequences, 
        num_epochs=400, 
        verbose=True, 
        interleaved=True
    )
    
    # 测试每个序列
    print("\n测试各序列回放:")
    for k in range(len(sequences)):
        xi_replayed = network.replay(sequence_index=k)
        eval_result = network.evaluate_replay(xi_replayed, sequence_index=k)
        
        status = "✓ 成功" if eval_result.get('found_sequence', False) else "✗ 失败"
        print(f"  序列 #{k}: {status} (准确率: {eval_result['recall_accuracy']*100:.1f}%)")
        
        # 可视化单个序列
        visualize_training_results(
            network,
            xi_replayed,
            eval_result,
            save_path=f"multi_seq_example_{k}.png",
            show_images=False,
            sequence_index=k
        )
    
    print("\n✓ 结果已保存")


def example_robustness():
    """示例3: 鲁棒性测试"""
    print("\n" + "="*60)
    print("示例3: 噪声鲁棒性测试")
    print("="*60)
    
    # 创建并训练网络
    network = SequenceAttractorNetwork(N_v=50, T=30, N_h=200, eta=0.01)
    network.train(num_epochs=300, seed=42, verbose=False)
    
    # 测试鲁棒性
    print("\n进行鲁棒性测试...")
    noise_levels = np.arange(0, 0.3, 0.05)
    robustness_scores = network.test_robustness(
        noise_levels=noise_levels,
        num_trials=50,
        verbose=True
    )
    
    # 可视化
    visualize_robustness(
        noise_levels,
        robustness_scores,
        save_path="robustness_example.png",
        show_images=False
    )
    print("\n✓ 鲁棒性结果已保存到 robustness_example.png")


def example_incremental_learning():
    """示例4: 增量学习"""
    print("\n" + "="*60)
    print("示例4: 增量学习")
    print("="*60)
    
    # 创建网络
    network = IncrementalSequenceAttractorNetwork(N_v=50, T=30, N_h=200, eta=0.01)
    
    # 第一阶段：学习第一个序列
    print("\n【阶段1】学习第一个序列")
    seq1 = network.generate_random_sequence(seed=100)
    network.train(x=seq1, num_epochs=300, verbose=True)
    
    # 测试第一个序列
    print("\n测试记忆:")
    memory_test1 = network.test_all_memories(verbose=True)
    
    # 第二阶段：增量学习第二个序列
    print("\n【阶段2】增量学习第二个序列（保持第一个序列的记忆）")
    seq2 = network.generate_random_sequence(seed=200)
    network.train(
        x=seq2, 
        num_epochs=300, 
        verbose=True,
        incremental=True  # 关键：开启增量学习模式
    )
    
    # 测试所有记忆
    print("\n测试记忆:")
    memory_test2 = network.test_all_memories(verbose=True)
    
    # 第三阶段：再学习第三个序列
    print("\n【阶段3】继续增量学习第三个序列")
    seq3 = network.generate_random_sequence(seed=300)
    network.train(
        x=seq3, 
        num_epochs=300, 
        verbose=True,
        incremental=True
    )
    
    # 测试所有记忆
    print("\n测试记忆:")
    memory_test3 = network.test_all_memories(verbose=True)
    
    # 显示记忆状态
    print("\n" + "="*60)
    print("当前记忆状态")
    print("="*60)
    status = network.get_memory_status()
    print(f"已学习序列数: {status['num_sequences']}")
    print(f"累计训练轮数: {status['total_epochs_trained']}")
    print("\n各序列训练信息:")
    for info in status['sequence_info']:
        print(f"  序列 #{info['sequence_index']}: "
              f"轮数 {info['start_epoch']}-{info['end_epoch']} "
              f"({'增量学习' if info['incremental'] else '独立学习'})")
    
    print("\n✓ 增量学习示例完成")


def example_pattern_repetition():
    """示例5: 模式重复"""
    print("\n" + "="*60)
    print("示例5: 模式重复网络")
    print("="*60)
    
    # 创建网络
    network = PatternRepetitionNetwork(N_v=50, T=40, N_h=250, eta=0.01)
    
    # 配置不同的模式
    pattern_configs = [
        {'pattern_type': 'alternating'},  # 交替模式
        {'pattern_type': 'periodic', 'period': 4},  # 4帧周期
        {'pattern_type': 'block', 'block_size': 5},  # 块大小5
        {'pattern_type': 'mirrored'},  # 镜像模式
    ]
    
    pattern_names = [
        '交替模式 (A-B-A-B...)',
        '周期模式 (period=4)',
        '块状模式 (block_size=5)',
        '镜像模式 (A-B-C...C-B-A)',
    ]
    
    # 生成模式序列
    print("\n生成4种不同重复模式的序列...")
    patterned_sequences = network.generate_multiple_patterned_sequences(
        num_sequences=4,
        pattern_configs=pattern_configs,
        seeds=[5000, 5001, 5002, 5003]
    )
    
    # 分析每个序列的模式结构
    print("\n模式结构分析:")
    for i, (seq, name) in enumerate(zip(patterned_sequences, pattern_names)):
        analysis = network.analyze_pattern_structure(seq)
        print(f"\n  {name}:")
        print(f"    唯一帧数: {analysis['unique_frames']}/{analysis['total_frames']}")
        print(f"    重复率: {analysis['repetition_rate']*100:.1f}%")
        print(f"    最大重复次数: {analysis['max_repetitions']}")
        if analysis['detected_period']:
            print(f"    检测到周期: {analysis['detected_period']}")
    
    # 分析序列重叠
    network.print_overlap_analysis(patterned_sequences)
    
    # 训练多模式序列
    print("\n训练多模式序列网络...")
    network.train(x=patterned_sequences, num_epochs=400, verbose=True, interleaved=True)
    
    # 测试每种模式的回放
    print("\n测试各模式序列回放:")
    for k, name in enumerate(pattern_names):
        xi_replayed = network.replay(sequence_index=k, max_steps=network.T * 2)
        eval_result = network.evaluate_replay(xi_replayed, sequence_index=k)
        
        status = "✓ 成功" if eval_result.get('found_sequence', False) else "✗ 失败"
        print(f"  {name}: {status} (准确率: {eval_result['recall_accuracy']*100:.1f}%)")
        
        # 可视化单个模式
        visualize_training_results(
            network,
            xi_replayed,
            eval_result,
            save_path=f"pattern_example_{k}.png",
            show_images=False,
            sequence_index=k
        )
    
    # 多序列概览
    visualize_multi_sequence_overview(
        network,
        save_path="pattern_overview.png",
        show_images=False
    )
    print("\n✓ 模式重复示例完成")


if __name__ == "__main__":
    # 运行所有示例
    example_basic_usage()
    example_multi_sequence()
    example_robustness()
    example_incremental_learning()
    example_pattern_repetition()
    
    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)

