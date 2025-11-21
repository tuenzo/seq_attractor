# 跨平台部署和测试指南

本指南帮助你在不同环境（Mac、Linux CPU、NVIDIA GPU）上部署和测试优化后的代码。

---

## 🚀 快速部署

### 1. 发布分支到远程

```bash
# 在 Mac 上推送优化分支
cd "/Users/zhao/Library/Mobile Documents/com~apple~CloudDocs/study&research/coding projects/VScode/hopf_networks/seq_attractor"

# 推送 optimize/blas 分支
git push -u origin optimize/blas

# 如果需要推送标签
git push origin --tags
```

### 2. 在其他机器克隆

```bash
# CPU 服务器或 GPU 机器
git clone https://github.com/tuenzo/seq_attractor.git
cd seq_attractor

# 切换到优化分支
git checkout optimize/blas

# 查看分支
git branch -a
```

---

## 💻 纯 CPU 环境部署

### Linux (Intel/AMD CPU)

#### 环境准备

```bash
# 1. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 检查系统配置
python scripts/check_system.py
```

**预期输出**：
```
系统: Linux
BLAS 库: Intel MKL 或 OpenBLAS
推荐后端: NUMPY
```

#### 性能测试

```bash
# 运行完整性能测试
python scripts/benchmark.py -o benchmark_results/cpu_test.json

# 运行 BLAS 对比测试
python docs/optimization/compare_blas_performance.py

# 运行单元测试
pytest tests/ -v
```

#### Intel CPU 优化（可选）

如果使用 Intel CPU，建议安装 MKL：

```bash
# 使用 conda
conda create -n seq_attractor python=3.10 numpy "libblas=*=*mkl"
conda activate seq_attractor
pip install -r requirements.txt

# 或使用 pip
pip uninstall numpy
pip install numpy intel-mkl

# 设置线程数
export MKL_NUM_THREADS=$(nproc)
```

#### AMD CPU 优化（可选）

```bash
# 使用 OpenBLAS
conda create -n seq_attractor python=3.10 numpy "libblas=*=*openblas"
conda activate seq_attractor
pip install -r requirements.txt

# 设置线程数
export OPENBLAS_NUM_THREADS=$(nproc)
```

---

## 🎮 NVIDIA GPU 环境部署

### 环境检查

```bash
# 检查 NVIDIA GPU
nvidia-smi

# 检查 CUDA 版本
nvcc --version
```

### 基础安装

```bash
# 1. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装基础依赖
pip install -r requirements.txt

# 3. 检查系统
python scripts/check_system.py
```

**预期输出（GPU 未启用）**：
```
系统: Linux
BLAS 库: OpenBLAS 或 MKL
GPU 支持: 是
GPU 数量: 1
GPU 0: NVIDIA GeForce RTX 3090
CuPy: 未安装
推荐后端: NUMPY (CuPy 未安装)
```

### GPU 加速安装（当前分支）

虽然当前 `optimize/blas` 分支还未实现 GPU 加速，但你可以安装 CuPy 为后续做准备：

```bash
# 根据 CUDA 版本安装 CuPy

# CUDA 11.x
pip install cupy-cuda11x

# CUDA 12.x
pip install cupy-cuda12x

# 验证安装
python -c "import cupy; print('CuPy OK'); a = cupy.array([1,2,3]); print(a)"

# 再次检查系统
python scripts/check_system.py
```

**预期输出（GPU 就绪）**：
```
系统: Linux
BLAS 库: OpenBLAS 或 MKL
GPU 支持: 是
GPU 数量: 1
GPU 0: NVIDIA GeForce RTX 3090
CUDA 版本: 12000
CuPy: 已安装
推荐后端: CUPY ⭐
```

### GPU 性能测试（当前分支）

```bash
# 当前分支使用 CPU + 优化的 BLAS
python scripts/benchmark.py -o benchmark_results/gpu_cpu_test.json

# 运行测试
pytest tests/ -v

# BLAS 对比
python docs/optimization/compare_blas_performance.py
```

**注意**：当前分支 (`optimize/blas`) 还未实现 GPU 加速，但已集成 GPU 检测。GPU 加速将在 `optimize/gpu-cupy` 分支实现。

---

## 📊 性能测试对比

### 测试脚本

在每个环境运行相同的测试：

```bash
# 1. 系统信息
python scripts/check_system.py > results_system_info.txt

# 2. 性能基准测试
python scripts/benchmark.py -o benchmark_results/$(hostname)_test.json

# 3. BLAS 对比测试
python docs/optimization/compare_blas_performance.py > results_blas_comparison.txt

# 4. 单元测试
pytest tests/ -v > results_pytest.txt
```

### 收集结果

```bash
# 打包所有结果
tar -czf test_results_$(hostname)_$(date +%Y%m%d).tar.gz \
    benchmark_results/*.json \
    results_*.txt

# 下载到本地对比
```

---

## 🔍 预期性能对比

基于优化分支 (`optimize/blas`)：

| 环境 | BLAS 库 | 预期加速 | 说明 |
|------|---------|---------|------|
| **Mac (M系列)** | Accelerate | 1.5-3x | 已测试 ✅ |
| **Linux (Intel)** | Intel MKL | 2-4x | 待测试 |
| **Linux (AMD)** | OpenBLAS | 1.5-2.5x | 待测试 |
| **NVIDIA GPU** | CUDA + MKL | 同 CPU | GPU 加速待后续分支 |

**注意**：
- 当前分支的优化主要在 CPU 矩阵运算（BLAS）
- GPU 加速将在 `optimize/gpu-cupy` 分支实现（预期 10-30x）
- 大规模问题（N_v>100, T>50）效果更明显

---

## 🧪 测试清单

在每个环境完成以下测试：

### 功能测试
- [ ] 系统检查通过 (`check_system.py`)
- [ ] 所有单元测试通过 (`pytest tests/`)
- [ ] 基础示例运行正常 (`examples/basic_example.py`)
- [ ] 准确率保持 100%

### 性能测试
- [ ] 基准测试完成 (`benchmark.py`)
- [ ] BLAS 对比测试完成 (`compare_blas_performance.py`)
- [ ] 记录训练时间、回放时间
- [ ] 记录内存占用

### 环境信息
- [ ] 记录系统信息（OS、CPU、GPU）
- [ ] 记录 Python、NumPy 版本
- [ ] 记录 BLAS 库类型
- [ ] 记录 CUDA 版本（如有）

---

## 📝 报告模板

创建测试报告：

```markdown
# 性能测试报告

## 环境信息
- 系统: Linux Ubuntu 22.04
- CPU: Intel Xeon E5-2690 v4 @ 2.60GHz (28核)
- GPU: NVIDIA RTX 3090 (24GB)
- Python: 3.10.12
- NumPy: 2.3.5
- BLAS: Intel MKL
- CUDA: 12.1

## 性能测试结果

| 测试项 | 时间 | 对比基线 | 加速比 |
|--------|------|---------|--------|
| 训练 (200轮) | 10.5s | 28.3s | 2.70x |
| 回放 (100步) | 0.4ms | 1.2ms | 3.00x |
| 鲁棒性测试 | 25.1s | 68.7s | 2.74x |
| 总计 | 35.6s | 98.2s | 2.76x |

## 功能验证
- ✅ 所有单元测试通过
- ✅ 准确率 100%（无损）
- ✅ 跨序列学习正常
- ✅ 增量学习正常

## 结论
Intel MKL 在该服务器上性能优异，比基线提升 2.76x。
```

---

## 🔧 故障排除

### 问题 1: BLAS 库未正确配置

**症状**：性能没有提升

**解决**：
```bash
# 检查 NumPy 配置
python -c "import numpy as np; np.show_config()"

# 重新安装 NumPy
pip uninstall numpy
pip install numpy --no-cache-dir
```

### 问题 2: GPU 未检测到

**症状**：`check_system.py` 显示 GPU 支持: 否

**解决**：
```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 CUDA
nvcc --version

# 安装 CuPy
pip install cupy-cuda12x  # 根据 CUDA 版本
```

### 问题 3: 导入错误

**症状**：`ModuleNotFoundError`

**解决**：
```bash
# 确保在项目根目录
cd seq_attractor

# 确保虚拟环境激活
source .venv/bin/activate

# 重新安装依赖
pip install -r requirements.txt

# 检查 Python 路径
python -c "import sys; print(sys.path)"
```

### 问题 4: 权限问题

**症状**：`PermissionError`

**解决**：
```bash
# 修复权限
chmod +x scripts/*.py
chmod +x scripts/*.sh

# 或使用 python 运行
python scripts/check_system.py
```

---

## 📞 获取帮助

测试遇到问题？

1. 查看 [故障排除指南](docs/optimization/TROUBLESHOOTING.md)
2. 运行诊断：`python scripts/check_system.py`
3. 查看日志文件
4. 提交 Issue 到 GitHub

---

## 🎯 下一步

### 当前分支测试完成后

1. **对比性能数据**
   - 收集各环境的测试结果
   - 生成对比报告
   - 验证跨平台兼容性

2. **准备 GPU 优化分支**
   ```bash
   # 基于当前分支创建 GPU 分支
   git checkout optimize/blas
   git checkout -b optimize/gpu-cupy
   
   # 实施 GPU 优化
   # 预期加速 10-30x
   ```

3. **继续其他优化**
   - 混合精度 (`optimize/mixed-precision`)
   - 多进程并行 (`optimize/multiprocessing`)
   - 算法优化 (`optimize/algorithm`)

---

## 📚 相关文档

- [优化计划](docs/optimization/OPTIMIZATION_PLAN.md)
- [快速开始](docs/optimization/QUICK_SUMMARY.md)
- [性能报告](docs/optimization/PERFORMANCE_COMPARISON_REPORT.md)
- [故障排除](docs/optimization/TROUBLESHOOTING.md)

---

**祝测试顺利！如有问题随时查看文档或提交 Issue。** 🚀

