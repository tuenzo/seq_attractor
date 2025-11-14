"""
================================================================
序列吸引子网络 - Figure 5 复现
对比只训练V权重和同时训练UV权重的效果
分图(a)：固定N_v=100, N_h=500，扫描序列长度T
分图(b)：固定N_v=100, T=70，扫描隐藏层神经元数量N_h
================================================================
"""

from SequenceAttractorNetwork import SequenceAttractorNetwork
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

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


def run_figure5_experiments(num_trials: int = 100,
                           noise_num: int = 10,
                           num_epochs: int = 500,
                           output_dir: str = "./figure5_results",
                           show_images: bool = False):
    """
    运行Figure 5的完整实验
    
    参数:
        num_trials: 每个配置的随机试验次数
        noise_num: 固定翻转的神经元数量（论文使用10）
        num_epochs: 训练轮数
        output_dir: 输出目录
        show_images: 是否显示中间图像
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig5_dir = os.path.join(output_dir, f"figure5_{timestamp}")
    os.makedirs(fig5_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("开始 Figure 5 实验")
    print(f"每个配置运行 {num_trials} 次独立试验")
    print(f"固定翻转 {noise_num} 个神经元")
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
        'kappa': 1.0
    }
    
    T_values = np.linspace(10, 140, 7, dtype=int)
    
    print("\n--- 只训练 V ---")
    results_v_only_a = []
    for i, T in enumerate(T_values):
        print(f"\n[{i+1}/{len(T_values)}] T = {T}")
        
        success_count = 0
        for trial in range(num_trials):
            # 创建网络（注意：不传入num_epochs）
            network = SequenceAttractorNetwork(
                N_v=base_params_a['N_v'],
                N_h=base_params_a['N_h'],
                T=T,  # 当前扫描的T值
                eta=base_params_a['eta'],
                kappa=base_params_a['kappa']
            )
            
            # 训练（num_epochs在这里传入）
            network.train(x=None, 
                         num_epochs=num_epochs, 
                         verbose=False, 
                         seed=trial, 
                         V_only=True)
            
            # 测试鲁棒性
            noise_level = noise_num / base_params_a['N_v']
            robustness = network.test_robustness(
                noise_levels=np.array([noise_level]),
                num_trials=1,
                verbose=False
            )
            
            if robustness[0] > 0.5:  # 成功
                success_count += 1
        
        success_rate = success_count / num_trials
        results_v_only_a.append({
            'T': T,
            'recall_accuracy': success_rate,
            'N_v': base_params_a['N_v'],
            'N_h': base_params_a['N_h']
        })
        print(f"成功率: {success_rate*100:.1f}% ({success_count}/{num_trials})")
    
    print("\n--- 训练 U+V ---")
    results_uv_a = []
    for i, T in enumerate(T_values):
        print(f"\n[{i+1}/{len(T_values)}] T = {T}")
        
        success_count = 0
        for trial in range(num_trials):
            network = SequenceAttractorNetwork(
                N_v=base_params_a['N_v'],
                N_h=base_params_a['N_h'],
                T=T,
                eta=base_params_a['eta'],
                kappa=base_params_a['kappa']
            )
            
            network.train(x=None, 
                         num_epochs=num_epochs, 
                         verbose=False, 
                         seed=trial, 
                         V_only=False)
            
            noise_level = noise_num / base_params_a['N_v']
            robustness = network.test_robustness(
                noise_levels=np.array([noise_level]),
                num_trials=1,
                verbose=False
            )
            
            if robustness[0] > 0.5:  # 成功
                success_count += 1
        
        success_rate = success_count / num_trials
        results_uv_a.append({
            'T': T,
            'recall_accuracy': success_rate,
            'N_v': base_params_a['N_v'],
            'N_h': base_params_a['N_h']
        })
        print(f"成功率: {success_rate*100:.1f}% ({success_count}/{num_trials})")
    
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
        'kappa': 1.0
    }

    N_h_values = np.linspace(100, 1000, 5, dtype=int)
    noise_level_b = noise_num / base_params_b['N_v']
    
    print("\n--- 只训练 V ---")
    results_v_only_b = []
    for i, N_h in enumerate(N_h_values):
        print(f"\n[{i+1}/{len(N_h_values)}] M = {N_h}")
        
        success_count = 0
        for trial in range(num_trials):
            network = SequenceAttractorNetwork(
                N_v=base_params_b['N_v'],
                N_h=N_h,  # 当前扫描的N_h值
                T=base_params_b['T'],
                eta=base_params_b['eta'],
                kappa=base_params_b['kappa']
            )
            
            network.train(x=None, 
                         num_epochs=num_epochs, 
                         verbose=False, 
                         seed=trial, 
                         V_only=True)
            
            robustness = network.test_robustness(
                noise_levels=np.array([noise_level_b]),
                num_trials=1,
                verbose=False
            )
            
            if robustness[0] > 0.5:
                success_count += 1
        
        success_rate = success_count / num_trials
        results_v_only_b.append({
            'N_h': N_h,
            'recall_accuracy': success_rate,
            'N_v': base_params_b['N_v'],
            'T': base_params_b['T']
        })
        print(f"成功率: {success_rate*100:.1f}% ({success_count}/{num_trials})")
    
    print("\n--- 训练 U+V ---")
    results_uv_b = []
    for i, N_h in enumerate(N_h_values):
        print(f"\n[{i+1}/{len(N_h_values)}] M = {N_h}")
        
        success_count = 0
        for trial in range(num_trials):
            network = SequenceAttractorNetwork(
                N_v=base_params_b['N_v'],
                N_h=N_h,
                T=base_params_b['T'],
                eta=base_params_b['eta'],
                kappa=base_params_b['kappa']
            )
            
            network.train(x=None, 
                         num_epochs=num_epochs, 
                         verbose=False, 
                         seed=trial, 
                         V_only=False)
            
            robustness = network.test_robustness(
                noise_levels=np.array([noise_level_b]),
                num_trials=1,
                verbose=False
            )
            
            if robustness[0] > 0.5:
                success_count += 1
        
        success_rate = success_count / num_trials
        results_uv_b.append({
            'N_h': N_h,
            'recall_accuracy': success_rate,
            'N_v': base_params_b['N_v'],
            'T': base_params_b['T']
        })
        print(f"成功率: {success_rate*100:.1f}% ({success_count}/{num_trials})")
    
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
        f.write("Figure 5 实验结果汇总\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"实验参数: num_trials={num_trials}, noise_num={noise_num}, num_epochs={num_epochs}\n\n")
        
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
    print("开始快速测试模式...")
    results_test = run_figure5_experiments(
        num_trials=10,       # 快速测试用10次
        noise_num=10,
        num_epochs=300,
        output_dir="./figure5_results",
        show_images=True
    )
    
    # 如需完整实验（耗时很长），取消下面的注释
    # print("\n开始完整实验...")
    # results_full = run_figure5_experiments(
    #     num_trials=100,      # 论文使用100次试验
    #     noise_num=10,
    #     num_epochs=500,
    #     output_dir="./figure5_results",
    #     show_images=True
    # )
    
    print("\n实验完成!")


