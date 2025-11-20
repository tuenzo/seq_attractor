# 性能优化快速开始指南

## 🚀 快速开始

### 1. 运行基线测试（当前性能）

```bash
# 运行基准测试并保存结果
python scripts/benchmark.py --output benchmark_results/baseline.json
```

### 2. 切换到优化分支

```bash
# 查看所有分支
git branch -a

# 切换到某个优化分支（例如：BLAS 优化）
git checkout optimize/blas

# 或创建新的优化分支
git checkout -b optimize/my-optimization
```

### 3. 实施优化并测试

```bash
# 修改代码...

# 运行测试确保功能正常
pytest tests/

# 运行性能测试
python scripts/benchmark.py --output benchmark_results/my-optimization.json
```

### 4. 对比性能

```bash
# 对比基线和优化后的性能
python scripts/benchmark.py --compare \
    benchmark_results/baseline.json \
    benchmark_results/my-optimization.json
```

### 5. 提交优化

```bash
# 如果优化有效，提交改动
git add .
git commit -m "optimize: 描述你的优化"

# 如果优化无效，回退到基线
git checkout baseline-before-optimization
```

---

## 📋 优化分支列表

### 已规划的优化分支

| 分支名 | 优化内容 | 预期加速 | 难度 | 状态 |
|--------|---------|---------|------|------|
| `optimize/blas` | 优化 BLAS 配置 | 1.5-3x | ⭐ | 待实施 |
| `optimize/mixed-precision` | 混合精度计算 | 1.5-2x | ⭐ | 待实施 |
| `optimize/multiprocessing` | 多进程并行 | 4-8x | ⭐⭐ | 待实施 |
| `optimize/gpu-cupy` | CuPy GPU 加速 | 10-30x | ⭐⭐ | 待实施 |
| `optimize/numba` | Numba JIT 编译 | 5-10x | ⭐⭐⭐ | 待实施 |
| `optimize/algorithm` | 算法优化 | 2-5x | ⭐⭐⭐ | 待实施 |
| `optimize/all` | 集成所有优化 | 20-100x | ⭐⭐⭐⭐ | 待实施 |

---

## 🔧 常用命令

### 分支管理

```bash
# 查看当前分支
git branch

# 查看所有分支（包括远程）
git branch -a

# 创建新分支
git checkout -b optimize/my-optimization

# 切换分支
git checkout optimize/blas

# 删除分支
git branch -d optimize/failed-optimization

# 合并分支到主分支
git checkout refactor-code-structure
git merge optimize/blas
```

### 标签管理

```bash
# 查看所有标签
git tag

# 回退到基线版本
git checkout baseline-before-optimization

# 创建新标签
git tag -a v1.0-optimized -m "优化后的稳定版本"
```

### 对比版本

```bash
# 对比两个分支的差异
git diff optimize/blas optimize/numba

# 查看某个文件在不同分支的差异
git diff optimize/blas:src/core/base.py optimize/numba:src/core/base.py
```

---

## 📊 性能测试工作流

### 完整测试流程

```bash
# 1. 确保在基线版本
git checkout baseline-before-optimization

# 2. 运行基线测试
python scripts/benchmark.py --output benchmark_results/baseline.json

# 3. 切换到优化分支
git checkout optimize/blas

# 4. 运行优化版本测试
python scripts/benchmark.py --output benchmark_results/blas.json

# 5. 对比结果
python scripts/benchmark.py --compare \
    benchmark_results/baseline.json \
    benchmark_results/blas.json

# 6. 如果满意，合并到主分支
git checkout refactor-code-structure
git merge optimize/blas
```

### 批量测试多个优化

```bash
# 创建测试脚本
cat > test_all_optimizations.sh << 'EOF'
#!/bin/bash

# 基线测试
git checkout baseline-before-optimization
python scripts/benchmark.py -o benchmark_results/baseline.json -q

# 测试各个优化分支
for branch in optimize/blas optimize/mixed-precision optimize/multiprocessing; do
    echo "Testing $branch..."
    git checkout $branch
    result_file="benchmark_results/$(basename $branch).json"
    python scripts/benchmark.py -o "$result_file" -q
    
    # 对比结果
    python scripts/benchmark.py -c benchmark_results/baseline.json "$result_file"
done

# 回到主分支
git checkout refactor-code-structure
EOF

chmod +x test_all_optimizations.sh
./test_all_optimizations.sh
```

---

## 💡 最佳实践

### 1. 每次优化前先测试基线

```bash
# 确保测试环境一致
git checkout baseline-before-optimization
python scripts/benchmark.py -o benchmark_results/baseline_$(date +%Y%m%d).json
```

### 2. 小步迭代，频繁测试

- ✅ 每完成一个小优化就测试
- ✅ 确保功能测试通过：`pytest tests/`
- ✅ 确保性能有提升
- ❌ 不要一次改动太多

### 3. 记录每次优化

在分支的 commit message 中详细记录：
```
optimize: 启用 macOS Accelerate 框架

- 删除 NPY_DISABLE_MAC_OS_ACCELERATE 环境变量设置
- 允许 NumPy 使用 Apple 优化的 BLAS 库
- 预期矩阵运算提速 1.5-3x

测试结果:
- 训练时间: 45.2s → 18.3s (2.47x)
- 测试时间: 12.8s → 5.6s (2.29x)
- 准确率: 100% → 100% (无损)

硬件: MacBook Pro M1, macOS 14.6
```

### 4. 保持功能不变

每次优化后必须确保：
- ✅ 所有测试通过：`pytest tests/`
- ✅ 准确率不下降
- ✅ API 保持兼容

---

## ⚠️ 注意事项

1. **不要在主分支直接修改**
   - 所有优化都在 `optimize/*` 分支进行
   - 测试验证后再合并到主分支

2. **基线标签不可修改**
   - `baseline-before-optimization` 标签永久保留
   - 用于性能对比参考

3. **测试环境要一致**
   - 同一台机器
   - 关闭其他占用资源的程序
   - 多次测试取平均值

4. **GPU 优化需要特殊处理**
   - 确保有 NVIDIA GPU
   - 安装 CUDA 和 cuDNN
   - CPU fallback 机制

---

## 🎯 下一步

1. 运行基线测试：`python scripts/benchmark.py -o benchmark_results/baseline.json`
2. 选择第一个优化方案（推荐从 `optimize/blas` 开始）
3. 切换到优化分支：`git checkout -b optimize/blas`
4. 开始优化！

