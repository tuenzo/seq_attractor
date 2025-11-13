"""
================================================================
序列吸引子网络 - 基础类
核心网络实现，使用向量化计算优化性能
================================================================
"""

import numpy as np
from typing import Optional, Dict


class SequenceAttractorNetwork:
    """序列吸引子循环神经网络（基础类）"""
    
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
        训练网络（向量化权重更新）
        
        参数:
            x: 训练序列 (T x N_v)，如果为None则自动生成
            num_epochs: 训练轮数
            verbose: 是否打印训练信息
            seed: 随机种子（仅在x为None时使用）
            V_only: 是否仅更新V权重
            
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
        
        # 预分配历史数组
        self.mu_history = np.zeros(num_epochs)
        self.nu_history = np.zeros(num_epochs)
        
        # 预计算所有时间步的输入输出对
        x_current_all = x[:-1, :].T  # (N_v, T-1)
        x_next_all = x[1:, :].T      # (N_v, T-1)
        
        for epoch in range(num_epochs):
            # ===== 更新 U（向量化版本）=====
            if not V_only:
                # 计算所有时间步的目标隐藏层
                z_target_all = np.sign(self.P @ x_next_all)  # (N_h, T-1)
                z_target_all[z_target_all == 0] = 1
                
                # 计算所有时间步的隐藏层输入
                h_input_all = self.U @ x_current_all  # (N_h, T-1)
                
                # 计算mu（是否需要更新）
                mu_all = (z_target_all * h_input_all < self.kappa).astype(float)  # (N_h, T-1)
                
                # 向量化权重更新
                delta_U = (mu_all * z_target_all) @ x_current_all.T  # (N_h, N_v)
                self.U += self.eta * delta_U
                
                total_mu = np.sum(mu_all)
            else:
                total_mu = 0
            
            # ===== 更新 V（向量化版本）=====
            # 计算所有时间步的实际隐藏层输出
            y_actual_all = np.sign(self.U @ x_current_all)  # (N_h, T-1)
            y_actual_all[y_actual_all == 0] = 1
            
            # 计算所有时间步的可见层输入
            v_input_all = self.V @ y_actual_all  # (N_v, T-1)
            
            # 计算nu（是否需要更新）
            nu_all = (x_next_all * v_input_all < self.kappa).astype(float)  # (N_v, T-1)
            
            # 向量化权重更新
            delta_V = (nu_all * x_next_all) @ y_actual_all.T  # (N_v, N_h)
            self.V += self.eta * delta_V
            
            total_nu = np.sum(nu_all)
            
            # 记录历史
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
            max_steps: 最大回放步数，默认为3*T
            
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
        
        # 预分配回放序列
        xi_replayed = np.zeros((max_steps, self.N_v))
        
        for step in range(max_steps):
            zeta = np.sign(self.U @ xi_test)
            zeta[zeta == 0] = 1
            xi_test = np.sign(self.V @ zeta)
            xi_test[xi_test == 0] = 1
            xi_replayed[step, :] = xi_test.flatten()
        
        return xi_replayed
    
    def test_recall_success_rate(self, num_trials: int = 50, 
                                noise_level: float = 0.0,
                                verbose: bool = True) -> Dict:
        """
        测试序列回放成功率
        
        参数:
            num_trials: 测试次数
            noise_level: 噪声水平（默认0.0表示无噪声）
            verbose: 是否打印信息
            
        返回:
            包含成功率和详细统计的字典
        """
        assert self.training_sequence is not None, "请先训练网络"
        
        success_count = 0
        max_search_steps = self.T * 5
        convergence_steps = []
        
        for trial in range(num_trials):
            # 1. 生成初始状态（加噪或无噪）
            xi_test = self.training_sequence[0, :].copy().reshape(-1, 1)
            
            if noise_level > 0:
                num_flips = int(noise_level * self.N_v)
                if num_flips > 0:
                    flip_indices = np.random.choice(self.N_v, num_flips, replace=False)
                    xi_test[flip_indices] = -xi_test[flip_indices]
            
            # 2. 记录演化轨迹
            trajectory = [xi_test.flatten().copy()]
            for step in range(max_search_steps):
                zeta = np.sign(self.U @ xi_test)
                zeta[zeta == 0] = 1
                xi_test = np.sign(self.V @ zeta)
                xi_test[xi_test == 0] = 1
                trajectory.append(xi_test.flatten().copy())
            
            # 3. 检查是否成功回放完整序列
            found_sequence = False
            for tau in range(max_search_steps - self.T + 2):
                segment = np.array(trajectory[tau:tau+self.T])
                if np.array_equal(segment, self.training_sequence):
                    found_sequence = True
                    convergence_steps.append(tau)
                    break
            
            if found_sequence:
                success_count += 1
        
        success_rate = success_count / num_trials
        
        if verbose:
            print(f'噪声水平 {noise_level:.2f}: 成功率 {success_rate*100:.1f}% '
                f'({success_count}/{num_trials} 次成功)')
            if convergence_steps:
                print(f'  平均收敛步数: {np.mean(convergence_steps):.1f}')
                print(f'  收敛步数范围: [{np.min(convergence_steps)}, {np.max(convergence_steps)}]')
        
        return {
            'success_rate': success_rate,
            'success_count': success_count,
            'num_trials': num_trials,
            'noise_level': noise_level,
            'convergence_steps': convergence_steps if convergence_steps else None,
            'avg_convergence_steps': np.mean(convergence_steps) if convergence_steps else None
        }
    
    def test_robustness(self, noise_levels: np.ndarray, 
                    num_trials: int = 50, 
                    verbose: bool = True) -> np.ndarray:
        """
        测试噪声鲁棒性
        
        参数:
            noise_levels: 噪声水平数组
            num_trials: 每个噪声水平的测试次数
            verbose: 是否打印进度
            
        返回:
            成功率数组
        """
        robustness_scores = np.zeros(len(noise_levels))
        
        for i, noise_level in enumerate(noise_levels):
            result = self.test_recall_success_rate(
                num_trials=num_trials,
                noise_level=noise_level,
                verbose=verbose
            )
            robustness_scores[i] = result['success_rate']
        
        return robustness_scores

