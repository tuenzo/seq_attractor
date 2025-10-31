"""
================================================================
序列吸引子网络 - 模块化版本
支持自定义序列和参数扫描
================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List
import os
from datetime import datetime

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10


class SequenceAttractorNetwork:
    """序列吸引子循环神经网络"""
    
    def __init__(self, N_v: int, T: int, N_h: Optional[int] = None, 
                 eta: float = 0.001, kappa: float = 1):
        """
        初始化网络
        
        参数:
            N_v: 可见神经元数量
            T: 序列长度
            N_h: 隐藏神经元数量（默认为3*(T-1)）
            eta: 学习率
            kappa: 鲁棒性参数（margin）
        """
        self.N_v = N_v
        self.T = T
        self.N_h = N_h if N_h is not None else round((T - 1) * 3)
        self.eta = eta
        self.kappa = kappa
        
        # 初始化权重矩阵
        self.U = np.random.randn(self.N_h, N_v) * 1e-6
        self.V = np.random.randn(N_v, self.N_h) * 1e-6
        self.P = np.random.randn(self.N_h, N_v) / np.sqrt(N_v)
        
        # 训练历史
        self.mu_history = []
        self.nu_history = []
        self.training_sequence = None
        
    def generate_random_sequence(self, seed: Optional[int] = None) -> np.ndarray:
        """
        生成随机训练序列
        
        参数:
            seed: 随机种子
            
        返回:
            x: T x N_v 的二值序列
        """
        if seed is not None:
            np.random.seed(seed)
            
        x = np.sign(np.random.randn(self.T, self.N_v))
        x[x == 0] = 1
        
        # 确保序列中没有重复（除了首尾）
        for t in range(1, self.T - 1):
            while np.any(np.all(x[t, :] == x[:t, :], axis=1)):
                x[t, :] = np.sign(np.random.randn(self.N_v))
                x[t, x[t, :] == 0] = 1
        
        x[self.T - 1, :] = x[0, :]  # 周期性
        
        return x
    
    def train(self, x: Optional[np.ndarray] = None, num_epochs: int = 500, 
              verbose: bool = True, seed: Optional[int] = None, V_only: bool = False) -> Dict:
        """
        训练网络
        
        参数:
            x: 训练序列 (T x N_v)，如果为None则自动生成
            num_epochs: 训练轮数
            verbose: 是否打印训练信息
            seed: 随机种子（仅在x为None时使用）
            
        返回:
            训练结果字典
        """
        # 准备训练序列
        if x is None:
            x = self.generate_random_sequence(seed)
        else:
            assert x.shape == (self.T, self.N_v), \
                f"序列形状应为 ({self.T}, {self.N_v})，实际为 {x.shape}"
        
        self.training_sequence = x
        
        if verbose:
            print(f"开始训练... N_v={self.N_v}, T={self.T}, N_h={self.N_h}")
            print(f"参数: eta={self.eta}, kappa={self.kappa}, epochs={num_epochs}")
            if V_only:
                print("仅更新 V 权重矩阵")


        
        self.mu_history = np.zeros(num_epochs)
        self.nu_history = np.zeros(num_epochs)
        
        for epoch in range(num_epochs):
            total_mu = 0
            total_nu = 0
            
            for t in range(self.T - 1):
                x_current = x[t, :].reshape(-1, 1)
                x_next = x[(t + 1) % self.T, :].reshape(-1, 1)
                
                if V_only:
                    mu=np.zeros((self.N_h, 1))
                else:
                # 更新U
                    z_target = np.sign(self.P @ x_next)
                    z_target[z_target == 0] = 1
                    h_input = self.U @ x_current
                    mu = (z_target * h_input < self.kappa).astype(float)
                    for i in range(self.N_h):
                        if mu[i, 0] > 0:
                            self.U[i, :] += self.eta * mu[i, 0] * z_target[i, 0] * x_current.flatten()
                
                # 更新V
                y_actual = np.sign(self.U @ x_current)
                y_actual[y_actual == 0] = 1
                v_input = self.V @ y_actual
                nu = (x_next * v_input < self.kappa).astype(float)
                
                for j in range(self.N_v):
                    if nu[j, 0] > 0:
                        self.V[j, :] += self.eta * nu[j, 0] * x_next[j, 0] * y_actual.flatten()
                
                total_mu += np.sum(mu)
                total_nu += np.sum(nu)
            
            self.mu_history[epoch] = total_mu / (self.N_h * (self.T - 1))
            self.nu_history[epoch] = total_nu / (self.N_v * (self.T - 1))
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, '
                      f'μ={self.mu_history[epoch]:.4f}, '
                      f'ν={self.nu_history[epoch]:.4f}')
        
        return {
            'mu_history': self.mu_history,
            'nu_history': self.nu_history,
            'final_mu': self.mu_history[-1],
            'final_nu': self.nu_history[-1]
        }
    
    def replay(self, x_init: Optional[np.ndarray] = None, 
               noise_level: float = 0.0, max_steps: Optional[int] = None) -> np.ndarray:
        """
        序列回放
        
        参数:
            x_init: 初始状态 (N_v,)，如果为None则使用训练序列的第一个状态
            noise_level: 噪声水平 (0.0-1.0)
            max_steps: 最大回放步数，默认为2*T
            
        返回:
            回放序列 (max_steps x N_v)
        """
        if x_init is None:
            assert self.training_sequence is not None, "请先训练网络或提供初始状态"
            x_init = self.training_sequence[0, :].copy()
        
        if max_steps is None:
            max_steps = self.T * 3
        
        xi_test = x_init.reshape(-1, 1).copy()
        
        # 添加噪声
        if noise_level > 0:
            noise_mask = np.random.rand(self.N_v, 1) < noise_level
            xi_test[noise_mask] = -xi_test[noise_mask]
        
        xi_replayed = np.zeros((max_steps, self.N_v))
        
        for step in range(max_steps):
            zeta = np.sign(self.U @ xi_test)
            zeta[zeta == 0] = 1
            xi_test = np.sign(self.V @ zeta)
            xi_test[xi_test == 0] = 1
            xi_replayed[step, :] = xi_test.flatten()
        
        return xi_replayed
    
    def evaluate_replay(self, xi_replayed: np.ndarray) -> Dict:
        """
        评估回放质量
        
        参数:
            xi_replayed: 回放序列
            
        返回:
            评估指标字典
        """
        assert self.training_sequence is not None, "请先训练网络"
        
        max_steps = xi_replayed.shape[0]
        match_count = 0
        match_indices = np.zeros(max_steps, dtype=int)
        
        for step in range(max_steps):
            for t in range(self.T):
                if np.all(xi_replayed[step, :] == self.training_sequence[t, :]):
                    match_count += 1
                    match_indices[step] = t + 1
                    break
        
        recall_accuracy = match_count / max_steps
        
        return {
            'recall_accuracy': recall_accuracy,
            'match_count': match_count,
            'match_indices': match_indices
        }
    
    def test_robustness(self, noise_levels: np.ndarray, 
                       num_trials: int = 50, verbose: bool = True) -> np.ndarray:
        """
        测试噪声鲁棒性
        
        参数:
            noise_levels: 噪声水平数组
            num_trials: 每个噪声水平的测试次数
            verbose: 是否打印进度
            
        返回:
            成功率数组
        """
        assert self.training_sequence is not None, "请先训练网络"
        
        robustness_scores = np.zeros(len(noise_levels))
        stability_threshold = 3
        
        for i, noise_level in enumerate(noise_levels):
            success_count = 0
            
            for trial in range(num_trials):
                xi_noisy = self.training_sequence[0, :].copy().reshape(-1, 1)
                noise_mask = np.random.rand(self.N_v, 1) < noise_level
                xi_noisy[noise_mask] = -xi_noisy[noise_mask]
                
                stable_steps = 0
                converged = False
                
                for step in range(self.T * 5):
                    xi_prev = xi_noisy.copy()
                    
                    zeta = np.sign(self.U @ xi_noisy)
                    zeta[zeta == 0] = 1
                    xi_noisy = np.sign(self.V @ zeta)
                    xi_noisy[xi_noisy == 0] = 1
                    
                    if np.all(xi_noisy.flatten() == self.training_sequence[0, :]):
                        converged = True
                        break
                    
                    if np.array_equal(xi_noisy, xi_prev):
                        stable_steps += 1
                    else:
                        stable_steps = 0
                    
                    if stable_steps >= stability_threshold:
                        break
                
                if converged:
                    success_count += 1
            
            robustness_scores[i] = success_count / num_trials
            
            if verbose:
                print(f'噪声水平 {noise_level:.2f}: 成功率 {robustness_scores[i]*100:.1f}%')
        
        return robustness_scores


def visualize_results(network: SequenceAttractorNetwork, 
                     xi_replayed: np.ndarray,
                     eval_results: Dict,
                     save_path: Optional[str] = None,
                     title_suffix: str = "",
                     show_images: bool = False):
    """
    可视化训练和回放结果
    
    参数:
        network: 训练好的网络
        xi_replayed: 回放序列
        eval_results: 评估结果
        save_path: 保存路径
        title_suffix: 标题后缀（用于参数标注）
    """
    fig = plt.figure(figsize=(14, 9))
    
    num_epochs = len(network.mu_history)
    max_steps = xi_replayed.shape[0]
    
    # 子图1: 隐藏层训练误差
    ax1 = plt.subplot(3, 3, 1)
    plt.plot(range(1, num_epochs + 1), network.mu_history, 'b-', linewidth=2)
    plt.xlabel('训练轮数')
    plt.ylabel('平均误差 μ')
    plt.title('隐藏层训练误差')
    plt.grid(True)
    
    # 子图2: 可见层训练误差
    ax2 = plt.subplot(3, 3, 2)
    plt.plot(range(1, num_epochs + 1), network.nu_history, 'r-', linewidth=2)
    plt.xlabel('训练轮数')
    plt.ylabel('平均误差 ν')
    plt.title('可见层训练误差')
    plt.grid(True)
    
    # 子图3: 双误差对比
    ax3 = plt.subplot(3, 3, 3)
    plt.plot(range(1, num_epochs + 1), network.mu_history, 'b-', 
             linewidth=1.5, label='μ (隐藏层)')
    plt.plot(range(1, num_epochs + 1), network.nu_history, 'r-', 
             linewidth=1.5, label='ν (可见层)')
    plt.xlabel('训练轮数')
    plt.ylabel('平均误差')
    plt.title('误差收敛曲线')
    plt.legend()
    plt.grid(True)
    
    # 子图4: 训练序列
    ax4 = plt.subplot(3, 3, 4)
    plt.imshow(network.training_sequence.T, cmap='gray', 
               aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('时间步')
    plt.ylabel('神经元索引')
    plt.title('训练序列（可见层）')
    
    # 子图5: 回放序列
    ax5 = plt.subplot(3, 3, 5)
    plt.imshow(xi_replayed.T, cmap='gray', aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('时间步')
    plt.ylabel('神经元索引')
    plt.title('回放序列（可见层）')
    
    # 子图6: U权重矩阵
    ax6 = plt.subplot(3, 3, 6)
    im6 = plt.imshow(network.U, cmap='jet', aspect='auto', interpolation='nearest')
    plt.colorbar(im6)
    plt.xlabel('可见神经元')
    plt.ylabel('隐藏神经元')
    plt.title('权重矩阵 U')
    
    # 子图7: V权重矩阵
    ax7 = plt.subplot(3, 3, 7)
    im7 = plt.imshow(network.V, cmap='jet', aspect='auto', interpolation='nearest')
    plt.colorbar(im7)
    plt.xlabel('隐藏神经元')
    plt.ylabel('可见神经元')
    plt.title('权重矩阵 V')
    
    # 子图8: P固定投影矩阵
    ax8 = plt.subplot(3, 3, 8)
    im8 = plt.imshow(network.P, cmap='jet', aspect='auto', interpolation='nearest')
    plt.colorbar(im8)
    plt.xlabel('可见神经元')
    plt.ylabel('隐藏神经元')
    plt.title('固定投影矩阵 P')
    
    # 子图9: 序列匹配追踪
    ax9 = plt.subplot(3, 3, 9)
    match_indices = eval_results['match_indices']
    plt.plot(range(1, max_steps + 1), match_indices, 'o-', 
             linewidth=1.5, markersize=6)
    plt.xlabel('回放时间步')
    plt.ylabel('匹配的训练序列索引')
    plt.title(f'序列匹配追踪 (准确率: {eval_results["recall_accuracy"]*100:.1f}%)')
    plt.ylim([0, network.T + 1])
    plt.grid(True)
    
    # 主标题
    main_title = f'序列吸引子网络训练与回放{title_suffix}'
    plt.suptitle(main_title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
    if show_images:
        plt.show()


def visualize_robustness(noise_levels: np.ndarray, 
                        robustness_scores: np.ndarray,
                        save_path: Optional[str] = None,
                        title_suffix: str = "",
                        show_images: bool = False):
    """
    可视化噪声鲁棒性测试结果
    
    参数:
        noise_levels: 噪声水平数组
        robustness_scores: 成功率数组
        save_path: 保存路径
        title_suffix: 标题后缀
    """
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels * 100, robustness_scores * 100, '-o',
             linewidth=2.5, markersize=8, color='#A23B72')
    plt.xlabel('噪声水平 (%)', fontsize=12)
    plt.ylabel('恢复到原序列的成功率 (%)', fontsize=12)
    
    title = f'序列吸引子的噪声鲁棒性{title_suffix}'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
    if show_images:
        plt.show()

def visualize_robustness_2(noise_levels: np.ndarray, 
                        robustness_scores: np.ndarray,
                        save_path: Optional[str] = None,
                        title_suffix: str = "",
                        show_images: bool = False):
    """
    可视化噪声鲁棒性测试结果
    
    参数:
        noise_levels: 噪声水平数组
        robustness_scores: 成功率数组
        save_path: 保存路径
        title_suffix: 标题后缀
    """
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels * 100, robustness_scores * 100, '-o',
             linewidth=2.5, markersize=8, color='#A23B72')
    plt.xlabel('噪声水平 (%)', fontsize=12)
    plt.ylabel('恢复到原序列的成功率 (%)', fontsize=12)
    
    title = f'序列吸引子的噪声鲁棒性{title_suffix}'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
    if show_images:
        plt.show()


def parameter_sweep(param_name: str, 
                   param_values: List,
                   base_params: Dict,
                   output_dir: str = "./param_sweep_results",
                   test_robustness: bool = True,
                   custom_sequence: Optional[np.ndarray] = None,
                   V_only: bool = False,
                   show_images: bool = False) -> List[Dict]:
    """
    参数扫描函数
    
    参数:
        param_name: 要扫描的参数名称 ('N_h', 'eta', 'kappa', 'T', 'N_v', 'num_epochs')
        param_values: 参数值列表
        base_params: 基础参数字典
        output_dir: 输出目录
        test_robustness: 是否测试鲁棒性
        custom_sequence: 自定义训练序列
        V_only: 是否仅更新V矩阵
    """
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(output_dir, f"{param_name}_sweep_{timestamp}")
    os.makedirs(sweep_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"开始参数扫描: {param_name}")
    print(f"参数值: {param_values}")
    print(f"输出目录: {sweep_dir}")
    print(f"{'='*60}\n")
    
    results_summary = []
    
    for i, value in enumerate(param_values):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(param_values)}] 测试 {param_name} = {value}")
        print(f"{'='*60}")
        
        # 更新参数
        current_params = base_params.copy()
        
        # 特殊处理训练轮数
        num_epochs = current_params.pop('num_epochs', 500)
        if param_name == 'num_epochs':
            num_epochs = value
        elif param_name in current_params:
            current_params[param_name] = value
        
        # 创建网络
        network = SequenceAttractorNetwork(**current_params)
        
        # 准备训练序列
        train_sequence = None
        if custom_sequence is not None:
            # 如果提供了自定义序列，需要确保尺寸匹配
            if custom_sequence.shape == (network.T, network.N_v):
                train_sequence = custom_sequence
            else:
                print(f"警告: 自定义序列尺寸不匹配，使用随机生成序列")
        
        # 训练
        train_results = network.train(x=train_sequence, num_epochs=num_epochs, 
                                     verbose=True, seed=42, V_only=V_only)
        
        # 回放
        xi_replayed = network.replay()
        eval_results = network.evaluate_replay(xi_replayed)
        
        # 生成标题后缀
        title_suffix = f"\n({param_name}={value}, η={network.eta}, κ={network.kappa}, " \
                      f"N_v={network.N_v}, T={network.T}, N_h={network.N_h})"
        
        # 可视化并保存
        filename_base = f"{param_name}_{value}"
        save_path = os.path.join(sweep_dir, f"{filename_base}_training.png")
        visualize_results(network, xi_replayed, eval_results, 
                         save_path=save_path, title_suffix=title_suffix, show_images=show_images)
        
        # 鲁棒性测试
        max_noise_tolerance = 0.0
        if test_robustness:
            print("\n进行噪声鲁棒性测试...")
            noise_levels = np.arange(0, 0.8, 0.05)
            robustness_scores = network.test_robustness(noise_levels, 
                                                        num_trials=50, 
                                                        verbose=False)
            
            # 找到最大噪声容忍度
            tolerance_idx = np.where(robustness_scores > 0.5)[0]
            if len(tolerance_idx) > 0:
                max_noise_tolerance = noise_levels[tolerance_idx[-1]]
            
            robustness_path = os.path.join(sweep_dir, f"{filename_base}_robustness.png")
            visualize_robustness(noise_levels, robustness_scores, 
                               save_path=robustness_path, 
                               title_suffix=title_suffix,
                               show_images=show_images)
        
        # 记录结果
        result_entry = {
            param_name: value,
            'final_mu': train_results['final_mu'],
            'final_nu': train_results['final_nu'],
            'recall_accuracy': eval_results['recall_accuracy'],
            'max_noise_tolerance': max_noise_tolerance,
            'N_v': network.N_v,
            'T': network.T,
            'N_h': network.N_h,
            'eta': network.eta,
            'kappa': network.kappa
        }
        results_summary.append(result_entry)
        
        print(f"\n结果: μ={train_results['final_mu']:.4f}, "
              f"ν={train_results['final_nu']:.4f}, "
              f"准确率={eval_results['recall_accuracy']*100:.1f}%, "
              f"噪声容忍度={max_noise_tolerance*100:.0f}%")
    
    # 保存汇总结果
    summary_path = os.path.join(sweep_dir, "results_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"参数扫描结果汇总 - {param_name}\n")
        f.write(f"{'='*80}\n\n")
        
        for i, result in enumerate(results_summary):
            f.write(f"[{i+1}] {param_name} = {result[param_name]}\n")
            f.write(f"  最终误差 (μ): {result['final_mu']:.6f}\n")
            f.write(f"  最终误差 (ν): {result['final_nu']:.6f}\n")
            f.write(f"  回放准确率: {result['recall_accuracy']*100:.2f}%\n")
            f.write(f"  最大噪声容忍度: {result['max_noise_tolerance']*100:.0f}%\n")
            f.write(f"  网络参数: N_v={result['N_v']}, T={result['T']}, N_h={result['N_h']}\n")
            f.write(f"  学习参数: η={result['eta']}, κ={result['kappa']}\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"\n{'='*60}")
    print(f"参数扫描完成！")
    print(f"结果已保存至: {sweep_dir}")
    print(f"汇总文件: {summary_path}")
    print(f"{'='*60}\n")
    
    return results_summary



# ========== 使用示例 ==========
if __name__ == "__main__":
    
    # 示例1: 基本使用（自动生成序列）
    print("\n" + "="*60)
    print("示例1: 基本使用")
    print("="*60)
    
    network = SequenceAttractorNetwork(N_v=20, T=10, eta=0.01, kappa=1)
    train_results = network.train(num_epochs=500, seed=42)
    
    xi_replayed = network.replay()
    eval_results = network.evaluate_replay(xi_replayed)
    
    print(f"\n训练完成:")
    print(f"  最终 μ 误差: {train_results['final_mu']:.4f}")
    print(f"  最终 ν 误差: {train_results['final_nu']:.4f}")
    print(f"  回放准确率: {eval_results['recall_accuracy']*100:.2f}%")
    
    visualize_results(network, xi_replayed, eval_results)
    
    # 鲁棒性测试
    noise_levels = np.arange(0, 0.8, 0.05)
    robustness_scores = network.test_robustness(noise_levels)
    visualize_robustness(noise_levels, robustness_scores)
    
    
    # 示例2: 使用自定义序列
    print("\n" + "="*60)
    print("示例2: 使用自定义序列")
    print("="*60)
    
    # 创建一个简单的周期性序列
    T_custom = 10
    N_v_custom = 15
    custom_seq = np.zeros((T_custom, N_v_custom))
    for t in range(T_custom - 1):
        custom_seq[t, :] = np.sin(2 * np.pi * t / T_custom) > 0
    custom_seq = 2 * custom_seq - 1  # 转换为 {-1, +1}
    custom_seq[T_custom - 1, :] = custom_seq[0, :]  # 周期性
    
    network2 = SequenceAttractorNetwork(N_v=N_v_custom, T=T_custom)
    train_results2 = network2.train(x=custom_seq, num_epochs=300)
    
    xi_replayed2 = network2.replay()
    eval_results2 = network2.evaluate_replay(xi_replayed2)
    
    visualize_results(network2, xi_replayed2, eval_results2, 
                     title_suffix="\n(使用自定义序列)")
    
    
    # 示例3: 参数扫描 - 学习率
    print("\n" + "="*60)
    print("示例3: 参数扫描 - 学习率")
    print("="*60)
    
    base_params = {
        'N_v': 20,
        'T': 10,
        'N_h': None,  # 自动计算
        'kappa': 1,
        'num_epochs': 300
    }
    
    eta_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    
    results = parameter_sweep(
        param_name='eta',
        param_values=eta_values,
        base_params=base_params,
        output_dir="./sweep_results",
        test_robustness=True
    )
    
    
    # 示例4: 参数扫描 - 隐藏层神经元数量
    print("\n" + "="*60)
    print("示例4: 参数扫描 - 隐藏层数量")
    print("="*60)
    
    T_base = 10
    N_h_values = [int((T_base-1) * factor) for factor in [1, 2, 3, 4, 5]]
    
    base_params2 = {
        'N_v': 20,
        'T': T_base,
        'eta': 0.01,
        'kappa': 1,
        'num_epochs': 300
    }
    
    results2 = parameter_sweep(
        param_name='N_h',
        param_values=N_h_values,
        base_params=base_params2,
        output_dir="./sweep_results",
        test_robustness=True
    )
    
    print("\n所有示例运行完成！")
