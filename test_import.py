#!/usr/bin/env python
"""简单的导入测试"""

print("Step 1: 导入 sys 和 os...")
import sys
import os
print("✓ OK")

print("\nStep 2: 检查环境变量...")
print(f"NPY_DISABLE_MAC_OS_ACCELERATE = {os.environ.get('NPY_DISABLE_MAC_OS_ACCELERATE', 'Not set')}")
print("✓ OK")

print("\nStep 3: 导入 numpy...")
try:
    import numpy as np
    print(f"✓ NumPy {np.__version__} 导入成功")
except Exception as e:
    print(f"✗ NumPy 导入失败: {e}")
    sys.exit(1)

print("\nStep 4: 测试简单的 numpy 操作...")
try:
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = a + b
    print(f"✓ 简单运算成功: {c}")
except Exception as e:
    print(f"✗ NumPy 运算失败: {e}")
    sys.exit(1)

print("\nStep 5: 测试矩阵乘法...")
try:
    A = np.random.randn(10, 10)
    B = np.random.randn(10, 10)
    C = A @ B
    print(f"✓ 矩阵乘法成功: shape={C.shape}")
except Exception as e:
    print(f"✗ 矩阵乘法失败: {e}")
    sys.exit(1)

print("\n所有测试通过！")

