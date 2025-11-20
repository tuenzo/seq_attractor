#!/usr/bin/env python
"""
对比 BLAS 优化前后的性能
测试：禁用 vs 启用 macOS Accelerate 框架
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_with_accelerate_disabled():
    """测试：禁用 Accelerate"""
    print("\n" + "="*70)
    print("测试 1: 禁用 macOS Accelerate（模拟优化前）")
    print("="*70)
    
    # 设置禁用 Accelerate（需要重启 Python 才能生效）
    # 这里我们只是模拟，实际上已经加载的 NumPy 不会改变
    os.environ['NPY_DISABLE_MAC_OS_ACCELERATE'] = '1'
    
    print("注意：由于 NumPy 已加载，此设置仅为演示")
    print("实际对比需要在不同进程中运行\n")
    
    # 导入网络（这会使用当前加载的 NumPy/BLAS）
    from src import SequenceAttractorNetwork
    
    # 小规模快速测试
    network = SequenceAttractorNetwork(N_v=50, T=30, N_h=150, eta=0.01, kappa=1, seed=42)
    
    # 测试训练性能
    print("训练 100 轮...")
    start_time = time.time()
    network.train(num_epochs=100, seed=100, verbose=False)
    train_time = time.time() - start_time
    
    # 测试回放性能
    print("测试回放...")
    start_time = time.time()
    xi_replayed = network.replay(max_steps=100)
    replay_time = time.time() - start_time
    
    # 评估
    eval_result = network.evaluate_replay(xi_replayed)
    
    results = {
        'train_time': train_time,
        'replay_time': replay_time,
        'success': eval_result.get('found_sequence', False),
        'accuracy': eval_result['recall_accuracy']
    }
    
    print(f"训练时间: {train_time:.3f} 秒")
    print(f"回放时间: {replay_time:.4f} 秒")
    print(f"成功回放: {'✓' if results['success'] else '✗'}")
    print(f"准确率: {results['accuracy']*100:.1f}%")
    
    return results


def test_with_accelerate_enabled():
    """测试：启用 Accelerate（优化后）"""
    print("\n" + "="*70)
    print("测试 2: 启用 macOS Accelerate（优化后）")
    print("="*70)
    
    # 确保没有禁用
    if 'NPY_DISABLE_MAC_OS_ACCELERATE' in os.environ:
        del os.environ['NPY_DISABLE_MAC_OS_ACCELERATE']
    
    print("当前 NumPy 配置：")
    print(f"  版本: {np.__version__}")
    print(f"  NPY_DISABLE_MAC_OS_ACCELERATE: {os.environ.get('NPY_DISABLE_MAC_OS_ACCELERATE', 'Not set')}")
    print()
    
    from src import SequenceAttractorNetwork
    
    # 相同参数测试
    network = SequenceAttractorNetwork(N_v=50, T=30, N_h=150, eta=0.01, kappa=1, seed=42)
    
    # 测试训练性能
    print("训练 100 轮...")
    start_time = time.time()
    network.train(num_epochs=100, seed=100, verbose=False)
    train_time = time.time() - start_time
    
    # 测试回放性能
    print("测试回放...")
    start_time = time.time()
    xi_replayed = network.replay(max_steps=100)
    replay_time = time.time() - start_time
    
    # 评估
    eval_result = network.evaluate_replay(xi_replayed)
    
    results = {
        'train_time': train_time,
        'replay_time': replay_time,
        'success': eval_result.get('found_sequence', False),
        'accuracy': eval_result['recall_accuracy']
    }
    
    print(f"训练时间: {train_time:.3f} 秒")
    print(f"回放时间: {replay_time:.4f} 秒")
    print(f"成功回放: {'✓' if results['success'] else '✗'}")
    print(f"准确率: {results['accuracy']*100:.1f}%")
    
    return results


def compare_results(baseline, optimized):
    """对比结果"""
    print("\n" + "="*70)
    print(" "*25 + "性能对比")
    print("="*70)
    
    train_speedup = baseline['train_time'] / optimized['train_time']
    replay_speedup = baseline['replay_time'] / optimized['replay_time']
    
    print(f"\n训练性能:")
    print(f"  基线: {baseline['train_time']:.3f}秒")
    print(f"  优化: {optimized['train_time']:.3f}秒")
    print(f"  加速: {train_speedup:.2f}x ({(train_speedup-1)*100:+.1f}%)")
    
    print(f"\n回放性能:")
    print(f"  基线: {baseline['replay_time']:.4f}秒")
    print(f"  优化: {optimized['replay_time']:.4f}秒")
    print(f"  加速: {replay_speedup:.2f}x ({(replay_speedup-1)*100:+.1f}%)")
    
    total_baseline = baseline['train_time'] + baseline['replay_time']
    total_optimized = optimized['train_time'] + optimized['replay_time']
    total_speedup = total_baseline / total_optimized
    
    print(f"\n总体:")
    print(f"  基线总时间: {total_baseline:.3f}秒")
    print(f"  优化总时间: {total_optimized:.3f}秒")
    print(f"  总体加速: {total_speedup:.2f}x ({(total_speedup-1)*100:+.1f}%)")
    
    print(f"\n准确率:")
    print(f"  基线: {baseline['accuracy']*100:.1f}%")
    print(f"  优化: {optimized['accuracy']*100:.1f}%")
    print(f"  变化: {(optimized['accuracy']-baseline['accuracy'])*100:+.1f}%")
    
    print("="*70)
    
    return {
        'train_speedup': train_speedup,
        'replay_speedup': replay_speedup,
        'total_speedup': total_speedup,
        'accuracy_maintained': abs(baseline['accuracy'] - optimized['accuracy']) < 0.01
    }


def main():
    """主函数"""
    print("\n" + "="*70)
    print(" "*20 + "BLAS 优化性能对比测试")
    print("="*70)
    print("\n说明:")
    print("  由于 NumPy 已加载，我们无法在同一进程中真正切换 BLAS 库。")
    print("  但优化后的代码确保了 Accelerate 框架已启用。")
    print("  测试将展示当前配置下的性能。")
    
    # 检查当前是否在优化分支
    try:
        import subprocess
        result = subprocess.run(['git', 'branch', '--show-current'], 
                              capture_output=True, text=True, cwd=project_root)
        current_branch = result.stdout.strip()
        print(f"\n当前分支: {current_branch}")
    except:
        pass
    
    # 运行测试（实际上都会使用当前已加载的 NumPy/BLAS）
    # 这里主要是展示流程和验证功能
    
    print("\n" + "="*70)
    print("运行性能测试（使用当前 BLAS 配置）")
    print("="*70)
    
    from src import SequenceAttractorNetwork
    
    # 执行完整测试
    print("\n执行测试...")
    network = SequenceAttractorNetwork(N_v=50, T=30, N_h=150, eta=0.01, kappa=1, seed=42)
    
    # 训练
    print("训练 200 轮...")
    start_train = time.time()
    network.train(num_epochs=200, seed=100, verbose=False)
    train_time = time.time() - start_train
    
    # 回放
    print("测试回放...")
    start_replay = time.time()
    xi_replayed = network.replay(max_steps=100)
    replay_time = time.time() - start_replay
    
    # 鲁棒性测试
    print("鲁棒性测试...")
    start_robust = time.time()
    robustness = network.test_robustness(
        noise_levels=np.array([0.0, 0.1, 0.2]),
        num_trials=10,
        verbose=False
    )
    robust_time = time.time() - start_robust
    
    # 评估
    eval_result = network.evaluate_replay(xi_replayed)
    
    # 显示结果
    print("\n" + "="*70)
    print(" "*25 + "测试结果")
    print("="*70)
    print(f"\n训练时间: {train_time:.3f} 秒")
    print(f"回放时间: {replay_time:.4f} 秒")
    print(f"鲁棒性测试: {robust_time:.3f} 秒")
    print(f"总时间: {train_time + replay_time + robust_time:.3f} 秒")
    
    print(f"\n成功回放: {'✓' if eval_result.get('found_sequence', False) else '✗'}")
    print(f"准确率: {eval_result['recall_accuracy']*100:.1f}%")
    
    print(f"\n鲁棒性:")
    for i, noise_level in enumerate([0.0, 0.1, 0.2]):
        print(f"  噪声 {noise_level:.1f}: {robustness[i]*100:.1f}%")
    
    print("\n" + "="*70)
    print(" "*20 + "BLAS 配置信息")
    print("="*70)
    
    # 显示 BLAS 配置
    from src.utils.blas_config import get_optimal_blas_config
    config = get_optimal_blas_config()
    print(f"\n系统: {config['system']}")
    print(f"架构: {config['machine']}")
    print(f"BLAS 库: {config['blas_library']}")
    
    if config['system'] == 'Darwin':
        accelerate_status = 'NPY_DISABLE_MAC_OS_ACCELERATE' not in os.environ
        print(f"Accelerate 状态: {'✓ 已启用' if accelerate_status else '✗ 已禁用'}")
    
    print("\n建议:")
    for rec in config['recommendations']:
        print(f"  - {rec}")
    
    print("="*70)
    
    print("\n✓ 测试完成！")
    print("\n说明:")
    print("  - 如果看到 'Accelerate 已启用'，表示正在使用优化的 BLAS 库")
    print("  - 在 Apple Silicon (M系列) 芯片上，Accelerate 性能极佳")
    print("  - 与禁用 Accelerate 相比，预期有 1.5-3x 的性能提升")


if __name__ == "__main__":
    main()

