#!/usr/bin/env python3
"""
Windows 环境检查脚本
快速诊断 Windows 环境问题

使用方法:
    python check_windows.py
"""

import sys
import os
from pathlib import Path

def main():
    print("=" * 60)
    print("Windows 环境检查")
    print("=" * 60)
    print()

    # Python 版本
    print("【Python 信息】")
    print(f"  版本: {sys.version}")
    print(f"  路径: {sys.executable}")
    print(f"  平台: {sys.platform}")
    print()

    # 当前目录
    print("【目录信息】")
    cwd = os.getcwd()
    print(f"  当前目录: {cwd}")
    print(f"  项目根目录: {Path.cwd()}")
    print()

    # 检查关键文件
    print("【关键文件检查】")
    files_to_check = [
        'src/__init__.py',
        'src/core/base.py',
        'scripts/auto_test.py',
        'scripts/benchmark.py',
        'requirements.txt',
        'README.md'
    ]

    all_exist = True
    for file in files_to_check:
        file_path = Path(file)
        exists = file_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        print()
        print("  ⚠ 警告: 某些文件不存在")
        print("  请确保在项目根目录运行此脚本")
        print(f"  当前目录: {cwd}")
        print()
    print()

    # Python 路径
    print("【Python 路径】")
    print("  sys.path 前 5 项:")
    for i, path in enumerate(sys.path[:5], 1):
        print(f"    {i}. {path}")
    print()

    # 检查项目根目录是否在 Python 路径中
    project_root = str(Path.cwd())
    if project_root not in sys.path:
        print(f"  ⚠ 警告: 项目根目录不在 Python 路径中")
        print(f"  添加到路径: {project_root}")
        sys.path.insert(0, project_root)
    else:
        print(f"  ✓ 项目根目录已在 Python 路径中")
    print()

    # 检查模块导入
    print("【模块导入检查】")
    
    # NumPy
    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy: {e}")
    
    # Matplotlib
    try:
        import matplotlib
        print(f"  ✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"  ✗ Matplotlib: {e}")
    
    # SciPy
    try:
        import scipy
        print(f"  ✓ SciPy {scipy.__version__}")
    except ImportError as e:
        print(f"  ✗ SciPy: {e}")
    
    # pytest
    try:
        import pytest
        print(f"  ✓ pytest {pytest.__version__}")
    except ImportError as e:
        print(f"  ✗ pytest: {e}")
    
    print()
    
    # 项目模块
    print("【项目模块检查】")
    try:
        from src import SequenceAttractorNetwork
        print(f"  ✓ src.SequenceAttractorNetwork")
    except ImportError as e:
        print(f"  ✗ src.SequenceAttractorNetwork: {e}")
        print()
        print("  调试信息:")
        print(f"    当前目录: {os.getcwd()}")
        print(f"    src/ 存在: {Path('src').exists()}")
        print(f"    src/__init__.py 存在: {Path('src/__init__.py').exists()}")
        if Path('src').exists():
            print(f"    src/ 内容: {list(Path('src').glob('*'))[:5]}")
    
    try:
        from src import MemorySequenceAttractorNetwork
        print(f"  ✓ src.MemorySequenceAttractorNetwork")
    except ImportError as e:
        print(f"  ✗ src.MemorySequenceAttractorNetwork: {e}")
    
    print()

    # 编码检查
    print("【编码设置】")
    print(f"  默认编码: {sys.getdefaultencoding()}")
    print(f"  文件系统编码: {sys.getfilesystemencoding()}")
    print(f"  stdout 编码: {sys.stdout.encoding}")
    print(f"  stderr 编码: {sys.stderr.encoding}")
    
    env_encoding = os.environ.get('PYTHONIOENCODING')
    if env_encoding:
        print(f"  PYTHONIOENCODING: {env_encoding}")
    else:
        print(f"  PYTHONIOENCODING: 未设置")
        print(f"  建议: 设置为 utf-8")
    print()

    # 环境变量
    print("【环境变量】")
    pythonpath = os.environ.get('PYTHONPATH')
    if pythonpath:
        print(f"  PYTHONPATH: {pythonpath}")
    else:
        print(f"  PYTHONPATH: 未设置")
    print()

    # 总结
    print("=" * 60)
    print("检查总结")
    print("=" * 60)
    
    issues = []
    suggestions = []
    
    if not all_exist:
        issues.append("某些关键文件不存在")
        suggestions.append("确保在项目根目录 (seq_attractor/) 运行脚本")
    
    if project_root not in sys.path:
        issues.append("项目根目录不在 Python 路径中")
        suggestions.append("设置 PYTHONPATH 或在项目根目录运行")
    
    try:
        import numpy
    except ImportError:
        issues.append("NumPy 未安装")
        suggestions.append("运行: pip install -r requirements.txt")
    
    try:
        from src import SequenceAttractorNetwork
    except ImportError:
        issues.append("无法导入项目模块")
        suggestions.append("确保在项目根目录且安装了所有依赖")
    
    if not env_encoding:
        issues.append("PYTHONIOENCODING 未设置")
        suggestions.append("设置环境变量: set PYTHONIOENCODING=utf-8 (CMD) 或 export PYTHONIOENCODING=utf-8 (Bash)")
    
    if issues:
        print("\n发现问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        print("\n建议修复:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
    else:
        print("\n✓ 所有检查通过！环境配置正确。")
        print("\n可以运行:")
        print("  python scripts/auto_test.py --quick")
    
    print()
    print("=" * 60)

if __name__ == "__main__":
    main()

