"""
================================================================
序列吸引子网络 - Figure 5 复现（并行加速版）
对比只训练V权重和同时训练UV权重的效果
分图(a)：固定N_v=100, N_h=500，扫描序列长度T
分图(b)：固定N_v=100, T=70，扫描隐藏层神经元数量N_h

性能优化：使用多进程并行计算多个trials
资源管理：默认使用75%的CPU核心，保留25%给系统
================================================================
"""

from SAN_tensor_1 import SequenceAttractorNetwork
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import math
from typing import Optional

# 尝试导入tqdm
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  未安装 tqdm，将不显示进度条。建议: pip install tqdm\n")


def get_optimal_n_jobs(n_jobs=-1, reserve_ratio=0.25):
    """
    获取最优的进程数，默认保留25%的CPU核心给系统
    
    参数:
        n_jobs: 
            -1: 自动使用75%的核心
            0: 使用所有核心（不推荐）
            正整数: 使用指定数量的核心
        reserve_ratio: 保留给系统的CPU比例（0-1之间）
    
    返回:
        实际使用的进程数
    """
    total_cores = cpu_count()
    
    if n_jobs == -1:
        # 默认：使用75%的核心，至少1个
        n_cores = max(1, int(total_cores * (1 - reserve_ratio)))
    elif n_jobs == 0:
        # 使用所有核心
        n_cores = total_cores
    else:
        # 使用指定数量，但不超过总核心数
        n_cores = min(n_jobs, total_cores)
    
    return n_cores


def plot_figure5(results_v_only, results_uv, 
                param_name: str, 
                param_values: np.ndarray,
                save_path: str = None,
                show_plot: bool = True):
    """
    绘制Figure 5风格的对比图
    
    参数:
        results_v_only: 只训练V的结果列表
        results_uv: 训练U+V的结果列表
        param_name: 参数名称 ('T' 或 'N_h')
        param_values: 参数值列表
        save_path: 保存路径
        show_plot: 是否显示图像
    """
    # 从results中提取数据
    success_rate_v_only = [result['recall_accuracy'] for result in results_v_only]
    success_rate_uv = [result['recall_accuracy'] for result in results_uv]
    
    # 创建图形
    plt.figure(figsize=(8, 6))
    
    # 绘制两条曲线
    plt.plot(param_values, np.array(success_rate_v_only) * 100, 
             'o-', linewidth=2, markersize=8, 
             label='Only training V', color='#E74C3C')
    
    plt.plot(param_values, np.array(success_rate_uv) * 100, 
             's-', linewidth=2, markersize=8,
             label='Training both U and V', color='#3498DB')
    
    # 设置标签和标题
    if param_name == 'T':
        plt.xlabel('Sequence Length (T)', fontsize=14, fontweight='bold')
        title = f'(a) Fixed N=100, M=500'
    elif param_name == 'N_h':
        plt.xlabel('Number of Hidden Neurons (M)', fontsize=14, fontweight='bold')
        title = f'(b) Fixed N=100, T=70'
    else:
        plt.xlabel(param_name, fontsize=14)
        title = f'Comparison: {param_name}'
    
    plt.ylabel('Successful Retrievals (%)', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 设置网格和图例
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, loc='best', framealpha=0.9)
    
    # 设置y轴范围
    plt.ylim([-5, 105])
    plt.yticks(np.arange(0, 101, 20))
    
    # 美化
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n图像已保存至: {save_path}")
    
    # 显示
    if show_plot:
        plt.show()
    else:
        plt.close()


def single_trial_task(trial_params):
    """
    单个trial的任务函数（用于并行）
    
    参数:
        trial_params: 包含所有必要参数的字典
    
    返回:
        是否成功 (bool)
    """
    N_v = trial_params['N_v']
    N_h = trial_params['N_h']
    T = trial_params['T']
    eta = trial_params['eta']
    kappa = trial_params['kappa']
    num_epochs = trial_params['num_epochs']
    noise_level = trial_params['noise_level']
    V_only = trial_params['V_only']
    seed = trial_params['seed']
    
    # 创建网络
    network = SequenceAttractorNetwork(
        N_v=N_v,
        N_h=N_h,
        T=T,
        eta=eta,
        kappa=kappa
    )
    
    # 训练
    network.train(
        x=None,
        num_epochs=num_epochs,
        verbose=False,
        seed=seed,
        V_only=V_only
    )
    
    # 测试鲁棒性
    robustness = network.test_robustness(
        noise_levels=np.array([noise_level]),
        num_trials=1,
        verbose=False
    )
    
    # 返回是否成功
    return robustness[0] > 0.5


def run_parallel_trials(base_params, num_trials, V_only=False, n_jobs=-1):
    """
    并行运行多个trials
    
    参数:
        base_params: 基础参数字典
        num_trials: 试验次数
        V_only: 是否只训练V
        n_jobs: 并行进程数
            -1: 使用75%的CPU核心（推荐）
            0: 使用所有CPU核心
            正整数: 使用指定数量的核心
    
    返回:
        success_rate: 成功率
        success_count: 成功次数
    """
    # 准备所有trial的参数
    trial_params_list = []
    for trial in range(num_trials):
        params = base_params.copy()
        params['V_only'] = V_only
        params['seed'] = trial
        trial_params_list.append(params)
    
    # 获取最优进程数
    n_cores = get_optimal_n_jobs(n_jobs)
    n_cores = min(n_cores, num_trials)  # 不超过trial数量
    
    # 并行执行
    with Pool(processes=n_cores) as pool:
        if TQDM_AVAILABLE:
            # 使用进度条
            results = list(tqdm(
                pool.imap(single_trial_task, trial_params_list),
                total=num_trials,
                desc=f"{'V-only' if V_only else 'U+V'} trials",
                ncols=80
            ))
        else:
            # 不使用进度条
            results = list(pool.imap(single_trial_task, trial_params_list))
            print(f"  完成 {num_trials} 次试验 (使用{n_cores}核)")
    
    # 计算成功率
    success_count = sum(results)
    success_rate = success_count / num_trials
    
    return success_rate, success_count


def run_figure5_experiments_parallel(num_trials: int = 100,
                                    noise_num: int = 10,
                                    num_epochs: int = 500,
                                    T_values: Optional[np.ndarray] = None,
                                    N_h_values: Optional[np.ndarray] = None,
                                    output_dir: str = "./figure5_results",
                                    show_images: bool = False,
                                    n_jobs: int = -1):
    """
    运行Figure 5的完整实验（并行版本）
    
    参数:
        num_trials: 每个配置的随机试验次数
        noise_num: 固定翻转的神经元数量（论文使用10）
        num_epochs: 训练轮数
        output_dir: 输出目录
        show_images: 是否显示中间图像
        n_jobs: 并行进程数
            -1: 使用75%的CPU核心（推荐，默认）
            0: 使用所有CPU核心
            正整数: 使用指定数量的核心
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig5_dir = os.path.join(output_dir, f"figure5_{timestamp}")
    os.makedirs(fig5_dir, exist_ok=True)
    
    # 获取CPU核心信息
    total_cores = cpu_count()
    n_cores = get_optimal_n_jobs(n_jobs)
    reserved_cores = total_cores - n_cores
    
    print("\n" + "="*70)
    print("开始 Figure 5 实验（并行加速版）")
    print(f"每个配置运行 {num_trials} 次独立试验")
    print(f"固定翻转 {noise_num} 个神经元")
    print(f"CPU配置: 总共{total_cores}核，使用{n_cores}核，保留{reserved_cores}核给系统")
    print(f"预计加速比: ~{n_cores}x")
    print("="*70)
    
    # ============================================================
    # Figure 5(a): 扫描序列长度T
    # ============================================================
    print("\n" + "="*70)
    print("Figure 5(a): 扫描序列长度T (固定 N=100, M=500)")
    print("="*70)
    
    base_params_a = {
        'N_v': 100,
        'N_h': 500,
        'eta': 0.01,
        'kappa': 1.0,
        'num_epochs': num_epochs
    }

    if T_values is None:
        T_values = np.linspace(10, 140, 8, dtype=int)

    print("\n--- 只训练 V ---")
    results_v_only_a = []
    for i, T in enumerate(T_values):
        print(f"\n[{i+1}/{len(T_values)}] T = {T}")
        
        params = base_params_a.copy()
        params['T'] = T
        params['noise_level'] = noise_num / params['N_v']
        
        success_rate, success_count = run_parallel_trials(
            params, num_trials, V_only=True, n_jobs=n_jobs
        )
        
        results_v_only_a.append({
            'T': T,
            'recall_accuracy': success_rate,
            'N_v': params['N_v'],
            'N_h': params['N_h']
        })
        print(f"  成功率: {success_rate*100:.1f}% ({success_count}/{num_trials})")
    
    print("\n--- 训练 U+V ---")
    results_uv_a = []
    for i, T in enumerate(T_values):
        print(f"\n[{i+1}/{len(T_values)}] T = {T}")
        
        params = base_params_a.copy()
        params['T'] = T
        params['noise_level'] = noise_num / params['N_v']
        
        success_rate, success_count = run_parallel_trials(
            params, num_trials, V_only=False, n_jobs=n_jobs
        )
        
        results_uv_a.append({
            'T': T,
            'recall_accuracy': success_rate,
            'N_v': params['N_v'],
            'N_h': params['N_h']
        })
        print(f"  成功率: {success_rate*100:.1f}% ({success_count}/{num_trials})")
    
    # 绘制Figure 5(a)
    plot_figure5(results_v_only_a, results_uv_a, 
                param_name='T',
                param_values=T_values,
                save_path=os.path.join(fig5_dir, "figure5a.png"),
                show_plot=show_images)
    
    # ============================================================
    # Figure 5(b): 扫描隐藏层神经元数量N_h
    # ============================================================
    print("\n" + "="*70)
    print("Figure 5(b): 扫描隐藏层大小M (固定 N=100, T=70)")
    print("="*70)
    
    base_params_b = {
        'N_v': 100,
        'T': 70,
        'eta': 0.01,
        'kappa': 1.0,
        'num_epochs': num_epochs
    }
    
    if N_h_values is None:
        N_h_values = np.linspace(100, 1000, 6, dtype=int)

    noise_level_b = noise_num / base_params_b['N_v']
    
    print("\n--- 只训练 V ---")
    results_v_only_b = []
    for i, N_h in enumerate(N_h_values):
        print(f"\n[{i+1}/{len(N_h_values)}] M = {N_h}")
        
        params = base_params_b.copy()
        params['N_h'] = N_h
        params['noise_level'] = noise_level_b
        
        success_rate, success_count = run_parallel_trials(
            params, num_trials, V_only=True, n_jobs=n_jobs
        )
        
        results_v_only_b.append({
            'N_h': N_h,
            'recall_accuracy': success_rate,
            'N_v': params['N_v'],
            'T': params['T']
        })
        print(f"  成功率: {success_rate*100:.1f}% ({success_count}/{num_trials})")
    
    print("\n--- 训练 U+V ---")
    results_uv_b = []
    for i, N_h in enumerate(N_h_values):
        print(f"\n[{i+1}/{len(N_h_values)}] M = {N_h}")
        
        params = base_params_b.copy()
        params['N_h'] = N_h
        params['noise_level'] = noise_level_b
        
        success_rate, success_count = run_parallel_trials(
            params, num_trials, V_only=False, n_jobs=n_jobs
        )
        
        results_uv_b.append({
            'N_h': N_h,
            'recall_accuracy': success_rate,
            'N_v': params['N_v'],
            'T': params['T']
        })
        print(f"  成功率: {success_rate*100:.1f}% ({success_count}/{num_trials})")
    
    # 绘制Figure 5(b)
    plot_figure5(results_v_only_b, results_uv_b, 
                param_name='N_h',
                param_values=N_h_values,
                save_path=os.path.join(fig5_dir, "figure5b.png"),
                show_plot=show_images)
    
    # ============================================================
    # 保存数据
    # ============================================================
    summary_path = os.path.join(fig5_dir, "results_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Figure 5 实验结果汇总（并行计算版）\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"实验参数: num_trials={num_trials}, noise_num={noise_num}, num_epochs={num_epochs}\n")
        f.write(f"并行设置: 总共{total_cores}核，使用{n_cores}核，保留{reserved_cores}核\n\n")
        
        f.write("(a) 扫描序列长度T (N=100, M=500)\n")
        f.write("-"*80 + "\n")
        f.write(f"{'T':<8} {'V-only (%)':<15} {'U+V (%)':<15} {'Improvement':<15}\n")
        f.write("-"*80 + "\n")
        for i, T in enumerate(T_values):
            v_only = results_v_only_a[i]['recall_accuracy']*100
            uv = results_uv_a[i]['recall_accuracy']*100
            improvement = uv - v_only
            f.write(f"{T:<8} {v_only:<15.1f} {uv:<15.1f} {improvement:+.1f}\n")
        
        f.write("\n\n(b) 扫描隐藏层大小M (N=100, T=70)\n")
        f.write("-"*80 + "\n")
        f.write(f"{'M':<8} {'V-only (%)':<15} {'U+V (%)':<15} {'Improvement':<15}\n")
        f.write("-"*80 + "\n")
        for i, N_h in enumerate(N_h_values):
            v_only = results_v_only_b[i]['recall_accuracy']*100
            uv = results_uv_b[i]['recall_accuracy']*100
            improvement = uv - v_only
            f.write(f"{N_h:<8} {v_only:<15.1f} {uv:<15.1f} {improvement:+.1f}\n")
    
    print(f"\n{'='*70}")
    print(f"实验完成! 结果保存在: {fig5_dir}")
    print(f"{'='*70}\n")
    
    return results_v_only_a, results_uv_a, results_v_only_b, results_uv_b


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    # 快速测试（少量试验）
    print("开始快速测试模式（并行）...")
    T_values = np.linspace(10, 140, 14, dtype=int)
    N_h_values = np.linspace(100, 1000, 10, dtype=int)
    results_test = run_figure5_experiments_parallel(
        num_trials=50,       # 快速测试用10次
        noise_num=10,
        num_epochs=500,
        T_values=T_values,
        N_h_values=N_h_values,
        output_dir="./figure5_results",
        show_images=True,
        n_jobs=-1            # 使用75%的CPU核心（推荐）
    )
    
    # 如需完整实验（耗时较长，但比串行快很多），取消下面的注释
    # print("\n开始完整实验（并行）...")
    # results_full = run_figure5_experiments_parallel(
    #     num_trials=100,      # 论文使用100次试验
    #     noise_num=10,
    #     num_epochs=500,
    #     output_dir="./figure5_results",
    #     show_images=True,
    #     n_jobs=-1            # 使用75%的CPU核心
    # )
    
    print("\n实验完成!")


