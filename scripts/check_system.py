#!/usr/bin/env python
"""
系统环境检查工具

检查并显示：
- 操作系统和硬件信息
- Python 和 NumPy 版本
- BLAS 库配置
- GPU 和 CUDA 支持
- 推荐的计算后端

使用：
    python scripts/check_system.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import print_system_info, get_optimal_blas_config, get_recommended_backend


def main():
    """主函数"""
    # 显示完整系统信息
    print_system_info()
    
    # 显示 BLAS 配置详情
    from src.utils.blas_config import configure_optimal_blas
    config = configure_optimal_blas(verbose=True)
    
    # 检查潜在问题
    print("\n" + "="*70)
    print(" "*25 + "环境检查")
    print("="*70)
    
    issues = []
    recommendations = []
    
    # 检查 NumPy 版本
    try:
        import numpy as np
        np_version = tuple(map(int, np.__version__.split('.')[:2]))
        if np_version < (1, 20):
            issues.append(f"NumPy 版本较旧 ({np.__version__})，建议升级到 1.20+")
    except:
        issues.append("无法检测 NumPy 版本")
    
    # 检查 BLAS 配置
    if config['blas_library'] == 'Reference BLAS':
        issues.append("使用参考 BLAS 实现，性能较低")
        recommendations.append("安装优化的 BLAS 库：pip install numpy[mkl] 或使用 conda")
    
    # 检查 macOS Accelerate
    if config['system'] == 'Darwin':
        import os
        if 'NPY_DISABLE_MAC_OS_ACCELERATE' in os.environ:
            issues.append("macOS Accelerate 框架被禁用")
            recommendations.append("删除 NPY_DISABLE_MAC_OS_ACCELERATE 环境变量")
        else:
            print("✓ macOS Accelerate 框架已启用")
    
    # 检查 GPU
    from src.utils.blas_config import check_gpu_availability
    gpu_info = check_gpu_availability()
    
    if gpu_info['cuda_available'] and not gpu_info['cupy_available']:
        recommendations.append("检测到 GPU 但未安装 CuPy，建议安装以获得 GPU 加速：pip install cupy-cuda12x")
    elif not gpu_info['cuda_available']:
        print("ℹ  未检测到 NVIDIA GPU（CPU 模式）")
    else:
        print(f"✓ GPU 加速可用：{gpu_info['gpu_count']} 个 GPU")
    
    # 显示问题和建议
    if issues:
        print("\n⚠️  发现问题：")
        for issue in issues:
            print(f"  - {issue}")
    
    if recommendations:
        print("\n💡 建议：")
        for rec in recommendations:
            print(f"  - {rec}")
    
    if not issues and not recommendations:
        print("\n✓ 环境配置良好，无需额外优化！")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

