# 快速测试指南

适用于在 CPU 服务器或 GPU 机器上快速开始测试。

---

## 🚀 5 分钟快速开始

### 第 1 步：克隆代码

```bash
# 克隆仓库
git clone https://github.com/tuenzo/seq_attractor.git
cd seq_attractor

# 切换到优化分支
git checkout optimize/blas

# 查看当前分支
git branch
# 应该显示: * optimize/blas
```

### 第 2 步：环境设置

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# 升级 pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

### 第 3 步：系统检查

```bash
# 检查环境配置
python scripts/check_system.py
```

**预期输出示例：**

<details>
<summary>CPU 服务器（Intel）</summary>

```
=== 系统环境检测 ===

系统信息:
  系统: Linux
  发行版: Ubuntu 22.04.3 LTS
  架构: x86_64
  处理器: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz

Python 环境:
  Python 版本: 3.10.12
  NumPy 版本: 2.3.5

BLAS 配置:
  ✅ BLAS 库: Intel MKL
  线程数: 28
  
推荐配置: NUMPY
```
</details>

<details>
<summary>GPU 机器（NVIDIA）</summary>

```
=== 系统环境检测 ===

系统信息:
  系统: Linux
  发行版: Ubuntu 22.04.3 LTS
  架构: x86_64
  处理器: Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz

Python 环境:
  Python 版本: 3.10.12
  NumPy 版本: 2.3.5

BLAS 配置:
  ✅ BLAS 库: OpenBLAS

GPU 信息:
  ✅ GPU 支持: 是
  GPU 数量: 2
  GPU 0: NVIDIA RTX 3090 (24GB)
  GPU 1: NVIDIA RTX 3090 (24GB)
  CUDA 版本: 12.1
  ⚠️  CuPy: 未安装

推荐配置: NUMPY (当前分支使用 CPU)
提示: 可安装 CuPy 为 GPU 优化做准备: pip install cupy-cuda12x
```
</details>

### 第 4 步：运行测试

```bash
# 快速功能测试
python examples/basic_example.py

# 完整单元测试
pytest tests/ -v

# 性能基准测试
python scripts/benchmark.py -o benchmark_results/test_$(hostname).json
```

### 第 5 步：查看结果

```bash
# 查看基准测试结果
cat benchmark_results/test_$(hostname).json

# 或使用 Python 美化输出
python -c "import json; print(json.dumps(json.load(open('benchmark_results/test_$(hostname).json')), indent=2, ensure_ascii=False))"
```

---

## 📊 详细性能测试

如果需要更全面的测试：

```bash
# 1. BLAS 对比测试（需要切换到基线对比）
python docs/optimization/compare_blas_performance.py

# 2. 大规模测试
python scripts/benchmark.py \
    --n_v 100 \
    --n_trials 5 \
    -o benchmark_results/large_scale_test.json

# 3. 鲁棒性测试
pytest tests/test_multi_sequence.py -v
pytest tests/test_pattern_repetition.py -v
```

---

## 🎮 GPU 准备（可选）

如果你在 GPU 机器上并希望为未来的 GPU 优化做准备：

```bash
# 检查 CUDA 版本
nvcc --version

# 安装 CuPy（根据 CUDA 版本选择）
pip install cupy-cuda11x  # CUDA 11.x
# 或
pip install cupy-cuda12x  # CUDA 12.x

# 验证 CuPy
python -c "import cupy as cp; print('CuPy OK'); a = cp.array([1,2,3]); print(a)"

# 再次检查系统
python scripts/check_system.py
```

**注意**：当前 `optimize/blas` 分支还未使用 GPU，但安装 CuPy 可以为后续 GPU 优化分支做准备。

---

## 📋 测试清单

- [ ] ✅ 代码克隆成功
- [ ] ✅ 虚拟环境创建
- [ ] ✅ 依赖安装完成
- [ ] ✅ 系统检查通过
- [ ] ✅ 基础示例运行
- [ ] ✅ 单元测试通过
- [ ] ✅ 基准测试完成
- [ ] 📊 结果已保存

---

## 🔍 预期性能（相对基线）

| 环境 | 训练加速 | 回放加速 | 总体加速 |
|------|---------|---------|---------|
| Mac M2 (已测试) | ~2.7x | ~3.0x | ~2.8x |
| Intel CPU + MKL | ~2-4x | ~2-4x | ~2-4x |
| AMD CPU + OpenBLAS | ~1.5-2.5x | ~1.5-2.5x | ~1.5-2.5x |
| GPU (当前分支) | 同 CPU | 同 CPU | 同 CPU |

**注意**：GPU 加速将在 `optimize/gpu-cupy` 分支实现，预期 10-30x 加速。

---

## 📝 保存测试结果

创建结果目录并保存：

```bash
# 创建结果目录
mkdir -p test_results_$(hostname)_$(date +%Y%m%d)
cd test_results_$(hostname)_$(date +%Y%m%d)

# 保存系统信息
python ../scripts/check_system.py > system_info.txt

# 保存基准测试
cp ../benchmark_results/*.json .

# 保存单元测试结果
pytest ../tests/ -v > pytest_results.txt

# 打包
cd ..
tar -czf test_results_$(hostname)_$(date +%Y%m%d).tar.gz test_results_$(hostname)_$(date +%Y%m%d)/

echo "结果已保存到: test_results_$(hostname)_$(date +%Y%m%d).tar.gz"
```

---

## ❓ 遇到问题？

### NumPy 导入失败

```bash
# 重新安装 NumPy
pip uninstall -y numpy
pip install numpy --no-cache-dir
```

### BLAS 未优化

```bash
# 检查 NumPy 配置
python -c "import numpy as np; np.show_config()"

# 如果是 Intel CPU，安装 MKL
pip uninstall numpy
pip install numpy intel-mkl
```

### GPU 未检测到

```bash
# 检查 NVIDIA 驱动和 CUDA
nvidia-smi
nvcc --version

# 安装 NVIDIA 驱动（如需要）
# Ubuntu: sudo apt install nvidia-driver-xxx
```

### 权限问题

```bash
# 给脚本添加执行权限
chmod +x scripts/*.py
chmod +x scripts/*.sh

# 或直接用 python 运行
python scripts/check_system.py
```

---

## 📚 更多信息

- **完整部署指南**: [docs/DEPLOYMENT_TESTING_GUIDE.md](DEPLOYMENT_TESTING_GUIDE.md)
- **优化详情**: [docs/optimization/OPTIMIZATION_PLAN.md](optimization/OPTIMIZATION_PLAN.md)
- **性能报告**: [docs/optimization/PERFORMANCE_COMPARISON_REPORT.md](optimization/PERFORMANCE_COMPARISON_REPORT.md)
- **故障排除**: [docs/optimization/TROUBLESHOOTING.md](optimization/TROUBLESHOOTING.md)

---

## 🎯 测试完成后

1. **收集结果文件**
   - `benchmark_results/*.json`
   - `system_info.txt`
   - `pytest_results.txt`

2. **对比不同环境**
   - Mac 本地结果
   - CPU 服务器结果
   - GPU 机器结果

3. **反馈和改进**
   - 性能是否符合预期？
   - 是否有兼容性问题？
   - 是否需要进一步优化？

---

**测试愉快！有问题随时查看文档或提 Issue。** 🚀

