#!/usr/bin/env python3
"""
基线版本性能测试脚本
适用于 baseline-before-optimization 标签

这个脚本可以在基线版本运行，生成性能数据用于后续对比。
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
import socket

# 确保项目根目录在路径中
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src import SequenceAttractorNetwork
    import numpy as np
except ImportError as e:
    print(f"错误: 无法导入模块: {e}")
    print(f"请确保在项目根目录运行此脚本")
    sys.exit(1)

def run_benchmark():
    """运行性能基准测试"""
    print("=" * 60)
    print("基线版本性能测试")
    print("=" * 60)
    print()
    
    # 测试参数
    N_v = 50
    T = 30
    N_h = 200
    eta = 0.001
    num_epochs = 200
    
    print(f"测试参数:")
    print(f"  N_v = {N_v}")
    print(f"  T = {T}")
    print(f"  N_h = {N_h}")
    print(f"  eta = {eta}")
    print(f" 训练轮数 = {num_epochs}")
    print()
    
    # 创建结果目录
    results_dir = project_root / "test_reports"
    results_dir.mkdir(exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hostname = socket.gethostname().split('.')[0]
    json_file = results_dir / f"baseline_before_optimize_{hostname}_{timestamp}.json"
    
    results = {
        "system_info": {
            "platform": sys.platform,
            "python_version": sys.version.split()[0],
            "timestamp": datetime.now().isoformat(),
            "hostname": hostname
        },
        "test_params": {
            "N_v": N_v,
            "T": T,
            "N_h": N_h,
            "eta": eta,
            "num_epochs": num_epochs
        }
    }
    
    # 测试 1: 单序列训练
    print("测试 1: 单序列训练...")
    network = SequenceAttractorNetwork(N_v=N_v, T=T, N_h=N_h, eta=eta)
    
    start_time = time.time()
    train_results = network.train(num_epochs=num_epochs, seed=42, verbose=False)
    training_time = time.time() - start_time
    
    print(f"  ✓ 训练完成: {training_time:.2f}s")
    results["training_time"] = training_time
    results["train_mu_final"] = train_results.get('mu_history', [0])[-1] if train_results else 0
    
    # 测试 2: 回放
    print("测试 2: 回放测试...")
    replay_times = []
    for _ in range(100):
        start_time = time.time()
        replayed = network.replay(max_steps=100)
        replay_times.append(time.time() - start_time)
    
    replay_time = np.mean(replay_times)
    print(f"  ✓ 回放完成: {replay_time*1000:.2f}ms (平均)")
    results["replay_time"] = replay_time
    
    # 测试 3: 准确率评估
    print("测试 3: 准确率评估...")
    replayed = network.replay(max_steps=100)
    eval_result = network.evaluate_replay(replayed)
    accuracy = eval_result.get('recall_accuracy', 0.0)
    print(f"  ✓ 准确率: {accuracy*100:.1f}%")
    results["replay_accuracy"] = accuracy
    results["found_sequence"] = eval_result.get('found_sequence', False)
    
    # 测试 4: 鲁棒性测试（简化版）
    print("测试 4: 鲁棒性测试（简化版）...")
    noise_levels = [0.0, 0.1, 0.2]
    robustness_scores = []
    robustness_start = time.time()
    
    for noise_level in noise_levels:
        success_count = 0
        trials = 20  # 减少试验次数以加快速度
        
        for _ in range(trials):
            # 添加噪声
            noisy_start = network.xi[0].copy()
            noise = np.random.randn(*noisy_start.shape) * noise_level
            noisy_start = np.clip(noisy_start + noise, -1, 1)
            
            # 回放
            replayed = network.replay(max_steps=100, initial_state=noisy_start)
            eval_result = network.evaluate_replay(replayed)
            
            if eval_result.get('found_sequence', False):
                success_count += 1
        
        score = success_count / trials
        robustness_scores.append(score)
        print(f"  噪声 {noise_level:.1f}: {score*100:.1f}% ({success_count}/{trials})")
    
    robustness_test_time = time.time() - robustness_start
    print(f"  ✓ 鲁棒性测试完成: {robustness_test_time:.2f}s")
    results["robustness_test_time"] = robustness_test_time
    results["robustness_scores"] = robustness_scores
    
    # 计算总时间
    total_time = training_time + robustness_test_time
    results["total_time"] = total_time
    
    # 保存结果
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print("=" * 60)
    print("测试完成")
    print("=" * 60)
    print()
    print("性能指标:")
    print(f"  训练时间: {training_time:.2f}s")
    print(f"  回放时间: {replay_time*1000:.2f}ms")
    print(f"  鲁棒性测试: {robustness_test_time:.2f}s")
    print(f"  总计: {total_time:.2f}s")
    print(f"  准确率: {accuracy*100:.1f}%")
    print()
    print(f"结果已保存: {json_file}")
    print()
    print("下一步:")
    print("  1. 将此文件复制为基线文件:")
    print(f"     cp {json_file} test_reports/baseline_before_optimize.json")
    print("  2. 切换回优化分支（如 optimize/blas）")
    print("  3. 运行对比测试:")
    print("     python scripts/auto_test.py --compare --baseline-file test_reports/baseline_before_optimize.json")
    
    return json_file

if __name__ == "__main__":
    try:
        run_benchmark()
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

