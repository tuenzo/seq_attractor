"""
================================================================
BLAS 库跨平台智能配置
自动检测并使用最佳的 BLAS/LAPACK 实现
================================================================
"""

import os
import sys
import platform
import warnings


def get_optimal_blas_config():
    """
    检测并返回当前平台的最佳 BLAS 配置
    
    返回:
        dict: 包含 BLAS 库信息和推荐配置
    """
    system = platform.system()
    machine = platform.machine()
    
    config = {
        'system': system,
        'machine': machine,
        'blas_library': 'unknown',
        'recommendations': []
    }
    
    # macOS - 使用 Accelerate 框架
    if system == 'Darwin':
        config['blas_library'] = 'Accelerate'
        config['recommendations'] = [
            'macOS 使用 Apple Accelerate 框架（已优化）',
            'M1/M2/M3 芯片性能极佳',
            '无需额外配置'
        ]
        # 确保不禁用 Accelerate
        if 'NPY_DISABLE_MAC_OS_ACCELERATE' in os.environ:
            del os.environ['NPY_DISABLE_MAC_OS_ACCELERATE']
    
    # Linux - 检测可用的 BLAS 库
    elif system == 'Linux':
        # 尝试检测 Intel MKL
        try:
            import numpy as np
            np_config = np.__config__.show()
            if np_config and 'mkl' in str(np_config).lower():
                config['blas_library'] = 'Intel MKL'
                config['recommendations'] = [
                    '检测到 Intel MKL（Intel CPU 最佳）',
                    '建议设置: export MKL_NUM_THREADS=auto'
                ]
            elif 'openblas' in str(np_config).lower():
                config['blas_library'] = 'OpenBLAS'
                config['recommendations'] = [
                    '检测到 OpenBLAS（通用高性能）',
                    '建议设置: export OPENBLAS_NUM_THREADS=auto'
                ]
            else:
                config['blas_library'] = 'Reference BLAS'
                config['recommendations'] = [
                    '使用参考实现 BLAS（性能较低）',
                    '建议安装: pip install numpy[mkl] 或 conda install numpy "libblas=*=*openblas"'
                ]
        except:
            config['blas_library'] = 'Unknown'
    
    # Windows
    elif system == 'Windows':
        config['blas_library'] = 'MKL (likely)'
        config['recommendations'] = [
            'Windows 通常使用 Intel MKL',
            '建议使用 Anaconda 发行版'
        ]
    
    return config


def configure_optimal_blas(verbose=False):
    """
    配置最佳的 BLAS 库并设置线程数
    
    参数:
        verbose: 是否打印配置信息
    """
    config = get_optimal_blas_config()
    
    # macOS Accelerate 特殊处理
    if config['system'] == 'Darwin':
        # 确保启用 Accelerate
        if 'NPY_DISABLE_MAC_OS_ACCELERATE' in os.environ:
            del os.environ['NPY_DISABLE_MAC_OS_ACCELERATE']
        
        # 设置 vecLib 线程数（如果需要）
        if 'VECLIB_MAXIMUM_THREADS' not in os.environ:
            # 默认使用所有核心
            os.environ['VECLIB_MAXIMUM_THREADS'] = str(os.cpu_count() or 1)
    
    # Linux MKL 配置
    elif config['system'] == 'Linux' and 'MKL' in config['blas_library']:
        if 'MKL_NUM_THREADS' not in os.environ:
            os.environ['MKL_NUM_THREADS'] = str(os.cpu_count() or 1)
    
    # Linux OpenBLAS 配置
    elif config['system'] == 'Linux' and 'OpenBLAS' in config['blas_library']:
        if 'OPENBLAS_NUM_THREADS' not in os.environ:
            os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count() or 1)
    
    if verbose:
        print("="*60)
        print("BLAS 库配置")
        print("="*60)
        print(f"系统: {config['system']}")
        print(f"架构: {config['machine']}")
        print(f"BLAS 库: {config['blas_library']}")
        print("\n建议:")
        for rec in config['recommendations']:
            print(f"  - {rec}")
        print("="*60)
    
    return config


def check_gpu_availability():
    """
    检测是否有可用的 GPU 及 CUDA 支持
    
    返回:
        dict: GPU 信息
    """
    gpu_info = {
        'cuda_available': False,
        'gpu_count': 0,
        'gpu_names': [],
        'cuda_version': None,
        'cupy_available': False,
        'pytorch_available': False
    }
    
    # 检测 CUDA/NVIDIA GPU
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            gpu_info['cuda_available'] = True
            gpu_info['gpu_names'] = [name.strip() for name in result.stdout.strip().split('\n') if name.strip()]
            gpu_info['gpu_count'] = len(gpu_info['gpu_names'])
    except:
        pass
    
    # 检测 CuPy
    try:
        import cupy
        gpu_info['cupy_available'] = True
        gpu_info['cuda_version'] = cupy.cuda.runtime.runtimeGetVersion()
    except ImportError:
        pass
    
    # 检测 PyTorch
    try:
        import torch
        gpu_info['pytorch_available'] = True
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            if not gpu_info['gpu_count']:
                gpu_info['gpu_count'] = torch.cuda.device_count()
    except ImportError:
        pass
    
    return gpu_info


def get_recommended_backend(verbose=False):
    """
    根据硬件环境推荐最佳计算后端
    
    返回:
        str: 推荐的后端 ('numpy', 'cupy', 'pytorch')
    """
    gpu_info = check_gpu_availability()
    blas_config = get_optimal_blas_config()
    
    # 优先使用 GPU
    if gpu_info['cuda_available'] and gpu_info['cupy_available']:
        backend = 'cupy'
        reason = f"检测到 {gpu_info['gpu_count']} 个 GPU 和 CuPy"
    elif gpu_info['cuda_available'] and gpu_info['pytorch_available']:
        backend = 'pytorch'
        reason = f"检测到 {gpu_info['gpu_count']} 个 GPU 和 PyTorch"
    else:
        backend = 'numpy'
        reason = f"使用 CPU + {blas_config['blas_library']}"
    
    if verbose:
        print(f"\n推荐后端: {backend}")
        print(f"原因: {reason}")
        
        if gpu_info['cuda_available']:
            print(f"\nGPU 信息:")
            for i, name in enumerate(gpu_info['gpu_names']):
                print(f"  GPU {i}: {name}")
            if gpu_info['cuda_version']:
                print(f"  CUDA 版本: {gpu_info['cuda_version']}")
    
    return backend


def print_system_info():
    """打印完整的系统和计算环境信息"""
    print("\n" + "="*70)
    print(" "*25 + "系统环境信息")
    print("="*70)
    
    # 基本信息
    print(f"\n系统: {platform.system()} {platform.release()}")
    print(f"架构: {platform.machine()}")
    print(f"处理器: {platform.processor()}")
    print(f"CPU 核心: {os.cpu_count()}")
    print(f"Python: {sys.version.split()[0]}")
    
    # NumPy 和 BLAS
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
        blas_config = get_optimal_blas_config()
        print(f"BLAS 库: {blas_config['blas_library']}")
    except ImportError:
        print("NumPy: 未安装")
    
    # GPU 信息
    gpu_info = check_gpu_availability()
    print(f"\nGPU 支持: {'是' if gpu_info['cuda_available'] else '否'}")
    if gpu_info['cuda_available']:
        print(f"GPU 数量: {gpu_info['gpu_count']}")
        for i, name in enumerate(gpu_info['gpu_names']):
            print(f"  GPU {i}: {name}")
        if gpu_info['cuda_version']:
            print(f"CUDA 版本: {gpu_info['cuda_version']}")
    
    # 可用库
    print(f"\nCuPy: {'已安装' if gpu_info['cupy_available'] else '未安装'}")
    print(f"PyTorch: {'已安装' if gpu_info['pytorch_available'] else '未安装'}")
    
    # 推荐配置
    backend = get_recommended_backend()
    print(f"\n推荐后端: {backend.upper()}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # 命令行工具：显示系统信息
    print_system_info()
    configure_optimal_blas(verbose=True)

