"""
================================================================
序列吸引子网络 - 图五扫描 对比只训练V权重和同时训练UV权重的效果
分图一：固定visual神经元和hidden神经元，扫描T序列长度

支持自定义序列和参数扫描
================================================================
"""

from  SequenceAttractorNetwork import SequenceAttractorNetwork, parameter_sweep, visualize_results, visualize_robustness
import numpy as np


# 参数扫描
base_params = {
    'N_v': 100,
    'T': 70,
    'N_h': 100,  # 自动计算
    'kappa': 1,
    'num_epochs': 500
}
noise_num = 10
noise_level = noise_num/base_params['T']
N_h=np.linspace(100, 1000, 10, dtype=int)
parameter_sweep(param_name = 'N_h', param_values = N_h, base_params = base_params,V_only=True, show_images=False)