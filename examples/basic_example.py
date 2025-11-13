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
    visualize_training_results,
    visualize_robustness
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


if __name__ == "__main__":
    # 运行所有示例
    example_basic_usage()
    example_multi_sequence()
    example_robustness()
    
    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)

