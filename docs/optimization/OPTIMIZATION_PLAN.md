# 性能优化计划

## 版本管理策略
本项目使用 Git 分支来管理不同的优化方案，便于版本对比和回退。

## 分支结构
```
refactor-code-structure (当前稳定版本)
├── optimize/baseline (优化前基线)
├── optimize/blas (优化 BLAS 配置)
├── optimize/mixed-precision (混合精度)
├── optimize/multiprocessing (多进程并行)
├── optimize/gpu-cupy (CuPy GPU 加速)
├── optimize/numba (Numba JIT)
├── optimize/algorithm (算法优化)
└── optimize/all (集成所有优化)
```

## 优化方案列表

### 阶段 1：零代码/最小改动优化（立即见效）
1. **optimize/blas** - 优化 BLAS 库配置
   - 删除禁用 Accelerate 的代码
   - 预期加速：1.5-3x
   - 改动：1行代码

2. **optimize/mixed-precision** - 混合精度计算
   - 使用 float32 替代 float64
   - 预期加速：1.5-2x
   - 改动：最小

3. **optimize/multiprocessing** - 多进程并行
   - 并行化鲁棒性测试
   - 预期加速：4-8x（测试场景）
   - 改动：中等

### 阶段 2：GPU 加速（如有 GPU）
4. **optimize/gpu-cupy** - CuPy GPU 加速
   - NumPy → CuPy
   - 预期加速：10-30x
   - 改动：中等

### 阶段 3：深度优化
5. **optimize/numba** - Numba JIT 编译
   - JIT 编译热点函数
   - 预期加速：5-10x
   - 改动：中等

6. **optimize/algorithm** - 算法优化
   - 序列生成优化
   - 早停机制
   - 预期加速：2-5x
   - 改动：中等

### 阶段 4：集成
7. **optimize/all** - 集成所有优化
   - 组合所有有效优化
   - 预期总加速：20-100x
   - 改动：大

## 使用方法

### 切换到某个优化分支
```bash
git checkout optimize/blas
```

### 查看当前分支
```bash
git branch
```

### 对比两个版本的性能
```bash
# 在分支 A 运行测试，记录结果
python benchmark.py > results_A.txt

# 切换到分支 B
git checkout optimize/blas

# 运行测试，记录结果
python benchmark.py > results_B.txt

# 对比结果
diff results_A.txt results_B.txt
```

### 合并优化到主分支
```bash
git checkout refactor-code-structure
git merge optimize/blas
```

## 性能测试基准

每个优化分支都应运行以下基准测试：
1. 单序列训练 (N_v=50, T=30, epochs=300)
2. 多序列训练 (3个序列, epochs=400)
3. 鲁棒性测试 (50次试验)
4. 内存占用

## 记录模板
```
优化方案：XXX
测试时间：YYYY-MM-DD
硬件环境：CPU/GPU型号

性能对比：
- 训练时间：XX秒 → XX秒 (加速 X.Xx)
- 测试时间：XX秒 → XX秒 (加速 X.Xx)
- 内存占用：XX MB → XX MB
- 准确率：XX% → XX% (应保持不变)

结论：✓ 采纳 / ✗ 放弃
```
