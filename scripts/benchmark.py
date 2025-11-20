"""
性能基准测试脚本
用于对比不同优化方案的性能

运行方式：
    python scripts/benchmark.py
    python scripts/benchmark.py --output results.json
"""

import sys
import time
import json
import platform
from pathlib import Path
from datetime import datetime
import argparse

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src import SequenceAttractorNetwork, MemorySequenceAttractorNetwork


def get_system_info():
    """获取系统信息"""
    return {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'numpy_version': np.__version__,
        'timestamp': datetime.now().isoformat(),
    }


def benchmark_single_sequence_training(verbose=True):
    """基准测试1：单序列训练"""
    if verbose:
        print("\n" + "="*60)
        print("基准测试 1: 单序列训练")
        print("="*60)
    
    network = SequenceAttractorNetwork(N_v=50, T=30, N_h=200, eta=0.001, kappa=1, seed=42)
    
    start_time = time.time()
    network.train(num_epochs=300, seed=100, verbose=False)
    train_time = time.time() - start_time
    
    # 测试回放
    start_time = time.time()
    xi_replayed = network.replay()
    replay_time = time.time() - start_time
    
    eval_result = network.evaluate_replay(xi_replayed)
    success = eval_result.get('found_sequence', False)
    
    result = {
        'train_time': train_time,
        'replay_time': replay_time,
        'success': success,
        'accuracy': eval_result['recall_accuracy']
    }
    
    if verbose:
        print(f"训练时间: {train_time:.2f} 秒")
        print(f"回放时间: {replay_time:.4f} 秒")
        print(f"成功回放: {'✓' if success else '✗'}")
        print(f"准确率: {result['accuracy']*100:.1f}%")
    
    return result


def benchmark_multi_sequence_training(verbose=True):
    """基准测试2：多序列训练"""
    if verbose:
        print("\n" + "="*60)
        print("基准测试 2: 多序列训练")
        print("="*60)
    
    network = MemorySequenceAttractorNetwork(N_v=50, T=30, N_h=200, eta=0.001, seed=42)
    
    # 生成3个序列
    sequences = network.generate_multiple_sequences(
        num_sequences=3,
        seeds=[100, 200, 300],
        ensure_unique_across=True,
        verbose=False
    )
    
    start_time = time.time()
    network.train(x=sequences, num_epochs=400, verbose=False, interleaved=True)
    train_time = time.time() - start_time
    
    # 测试每个序列
    start_time = time.time()
    success_count = 0
    for k in range(len(sequences)):
        xi_replayed = network.replay(sequence_index=k)
        eval_result = network.evaluate_replay(xi_replayed, sequence_index=k)
        if eval_result.get('found_sequence', False):
            success_count += 1
    replay_time = time.time() - start_time
    
    result = {
        'train_time': train_time,
        'replay_time': replay_time,
        'success_count': success_count,
        'total_sequences': len(sequences),
        'success_rate': success_count / len(sequences)
    }
    
    if verbose:
        print(f"训练时间: {train_time:.2f} 秒")
        print(f"回放时间: {replay_time:.4f} 秒")
        print(f"成功序列: {success_count}/{len(sequences)}")
        print(f"成功率: {result['success_rate']*100:.1f}%")
    
    return result


def benchmark_robustness_test(verbose=True):
    """基准测试3：鲁棒性测试"""
    if verbose:
        print("\n" + "="*60)
        print("基准测试 3: 鲁棒性测试")
        print("="*60)
    
    network = SequenceAttractorNetwork(N_v=50, T=30, N_h=200, eta=0.01, seed=42)
    network.train(num_epochs=300, seed=100, verbose=False)
    
    noise_levels = np.array([0.0, 0.1, 0.2])
    num_trials = 20  # 减少试验次数以加快测试
    
    start_time = time.time()
    robustness_scores = network.test_robustness(
        noise_levels=noise_levels,
        num_trials=num_trials,
        verbose=False
    )
    test_time = time.time() - start_time
    
    result = {
        'test_time': test_time,
        'noise_levels': noise_levels.tolist(),
        'robustness_scores': robustness_scores.tolist(),
        'num_trials': num_trials
    }
    
    if verbose:
        print(f"测试时间: {test_time:.2f} 秒")
        for i, noise_level in enumerate(noise_levels):
            print(f"噪声 {noise_level:.1f}: 成功率 {robustness_scores[i]*100:.1f}%")
    
    return result


def benchmark_sequence_generation(verbose=True):
    """基准测试4：序列生成"""
    if verbose:
        print("\n" + "="*60)
        print("基准测试 4: 序列生成")
        print("="*60)
    
    network = MemorySequenceAttractorNetwork(N_v=50, T=30, N_h=200, seed=42)
    
    start_time = time.time()
    sequences = network.generate_multiple_sequences(
        num_sequences=10,
        seeds=list(range(100, 110)),
        ensure_unique_across=True,
        verbose=False
    )
    generation_time = time.time() - start_time
    
    result = {
        'generation_time': generation_time,
        'num_sequences': len(sequences),
        'time_per_sequence': generation_time / len(sequences)
    }
    
    if verbose:
        print(f"生成时间: {generation_time:.2f} 秒")
        print(f"序列数量: {len(sequences)}")
        print(f"平均每个: {result['time_per_sequence']:.3f} 秒")
    
    return result


def run_all_benchmarks(output_file=None, verbose=True):
    """运行所有基准测试"""
    if verbose:
        print("\n" + "="*70)
        print(" "*20 + "性能基准测试")
        print("="*70)
    
    system_info = get_system_info()
    
    if verbose:
        print(f"\n系统信息:")
        print(f"  平台: {system_info['platform']}")
        print(f"  处理器: {system_info['processor']}")
        print(f"  Python: {system_info['python_version']}")
        print(f"  NumPy: {system_info['numpy_version']}")
        print(f"  时间: {system_info['timestamp']}")
    
    results = {
        'system_info': system_info,
        'benchmarks': {}
    }
    
    # 运行各项测试
    try:
        results['benchmarks']['single_sequence'] = benchmark_single_sequence_training(verbose)
    except Exception as e:
        if verbose:
            print(f"测试失败: {e}")
        results['benchmarks']['single_sequence'] = {'error': str(e)}
    
    try:
        results['benchmarks']['multi_sequence'] = benchmark_multi_sequence_training(verbose)
    except Exception as e:
        if verbose:
            print(f"测试失败: {e}")
        results['benchmarks']['multi_sequence'] = {'error': str(e)}
    
    try:
        results['benchmarks']['robustness'] = benchmark_robustness_test(verbose)
    except Exception as e:
        if verbose:
            print(f"测试失败: {e}")
        results['benchmarks']['robustness'] = {'error': str(e)}
    
    try:
        results['benchmarks']['sequence_generation'] = benchmark_sequence_generation(verbose)
    except Exception as e:
        if verbose:
            print(f"测试失败: {e}")
        results['benchmarks']['sequence_generation'] = {'error': str(e)}
    
    # 计算总时间
    total_time = sum(
        bench.get('train_time', 0) + bench.get('replay_time', 0) + 
        bench.get('test_time', 0) + bench.get('generation_time', 0)
        for bench in results['benchmarks'].values()
        if isinstance(bench, dict) and 'error' not in bench
    )
    
    results['total_time'] = total_time
    
    if verbose:
        print("\n" + "="*70)
        print(f"总计用时: {total_time:.2f} 秒")
        print("="*70 + "\n")
    
    # 保存结果
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"✓ 结果已保存到: {output_file}\n")
    
    return results


def compare_results(baseline_file, optimized_file):
    """对比两个测试结果"""
    with open(baseline_file, 'r', encoding='utf-8') as f:
        baseline = json.load(f)
    
    with open(optimized_file, 'r', encoding='utf-8') as f:
        optimized = json.load(f)
    
    print("\n" + "="*70)
    print(" "*25 + "性能对比")
    print("="*70)
    
    print(f"\n基线版本: {baseline['system_info']['timestamp']}")
    print(f"优化版本: {optimized['system_info']['timestamp']}")
    
    # 对比各项测试
    for test_name in ['single_sequence', 'multi_sequence', 'robustness', 'sequence_generation']:
        if test_name not in baseline['benchmarks'] or test_name not in optimized['benchmarks']:
            continue
        
        base = baseline['benchmarks'][test_name]
        opt = optimized['benchmarks'][test_name]
        
        if 'error' in base or 'error' in opt:
            continue
        
        print(f"\n{test_name.replace('_', ' ').title()}:")
        
        # 训练时间
        if 'train_time' in base:
            speedup = base['train_time'] / opt['train_time']
            improvement = (1 - opt['train_time'] / base['train_time']) * 100
            print(f"  训练时间: {base['train_time']:.2f}s → {opt['train_time']:.2f}s "
                  f"(加速 {speedup:.2f}x, 提升 {improvement:.1f}%)")
        
        # 测试时间
        if 'test_time' in base:
            speedup = base['test_time'] / opt['test_time']
            improvement = (1 - opt['test_time'] / base['test_time']) * 100
            print(f"  测试时间: {base['test_time']:.2f}s → {opt['test_time']:.2f}s "
                  f"(加速 {speedup:.2f}x, 提升 {improvement:.1f}%)")
        
        # 生成时间
        if 'generation_time' in base:
            speedup = base['generation_time'] / opt['generation_time']
            improvement = (1 - opt['generation_time'] / base['generation_time']) * 100
            print(f"  生成时间: {base['generation_time']:.2f}s → {opt['generation_time']:.2f}s "
                  f"(加速 {speedup:.2f}x, 提升 {improvement:.1f}%)")
    
    # 总时间
    if 'total_time' in baseline and 'total_time' in optimized:
        speedup = baseline['total_time'] / optimized['total_time']
        improvement = (1 - optimized['total_time'] / baseline['total_time']) * 100
        print(f"\n总计:")
        print(f"  总时间: {baseline['total_time']:.2f}s → {optimized['total_time']:.2f}s "
              f"(加速 {speedup:.2f}x, 提升 {improvement:.1f}%)")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='性能基准测试')
    parser.add_argument('--output', '-o', default=None, help='输出文件路径 (JSON格式)')
    parser.add_argument('--compare', '-c', nargs=2, metavar=('BASELINE', 'OPTIMIZED'),
                       help='对比两个结果文件')
    parser.add_argument('--quiet', '-q', action='store_true', help='静默模式')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_results(args.compare[0], args.compare[1])
    else:
        run_all_benchmarks(output_file=args.output, verbose=not args.quiet)


if __name__ == "__main__":
    main()

