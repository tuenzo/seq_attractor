"""
================================================================
序列吸引子网络 - 图五扫描 对比只训练V权重和同时训练UV权重的效果
分图一：固定visual神经元和hidden神经元，扫描T序列长度
分图二：固定visual神经元和序列长度，扫描hidden神经元数量
支持自定义序列和参数扫描
================================================================
"""

from  SequenceAttractorNetwork import SequenceAttractorNetwork, parameter_sweep, visualize_results, visualize_robustness
import numpy as np


# figure 5a: Scan over T with fixed N_v(N=100) and N_h(M=500)

base_params = {
    'N_v': 100,
    'T': 70,
    'N_h': 500,
    'kappa': 1,
    'num_epochs': 500
}

noise_num = 10
noise_level = noise_num/base_params['T']

T=np.linspace(10, 150, 10, dtype=int)

sweep_result=parameter_sweep(param_name = 'T', param_values = T, base_params = base_params,V_only=False, show_images=False)

