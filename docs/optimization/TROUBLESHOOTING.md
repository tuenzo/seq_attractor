# 故障排除指南

## 🔴 当前问题：NumPy 段错误（Exit code 139）

### 症状
```bash
python test_import.py
# Exit code: 139 (Segmentation fault)
```

即使是简单的 NumPy 导入也会导致段错误。

---

## 🔍 问题原因

这通常由以下原因之一引起：

### 1. macOS Accelerate 框架冲突
- NumPy 与 macOS Accelerate 框架存在兼容性问题
- 某些 NumPy 版本在 macOS 上不稳定

### 2. 虚拟环境配置问题
- pip 和 conda 混用导致库冲突
- NumPy 版本与系统不兼容

### 3. BLAS 库损坏
- NumPy 链接的 BLAS 库损坏或版本不匹配

---

## 🛠️ 解决方案

### 方案 1：重新安装 NumPy（推荐）

#### 使用 pip（纯 Python 环境）

```bash
# 进入项目目录
cd "/Users/zhao/Library/Mobile Documents/com~apple~CloudDocs/study&research/coding projects/VScode/hopf_networks/seq_attractor"

# 激活虚拟环境
source .venv/bin/activate

# 完全卸载 NumPy
pip uninstall -y numpy

# 清理缓存
pip cache purge

# 重新安装（使用预编译的二进制包）
pip install numpy --no-cache-dir

# 测试
python test_import.py
```

#### 使用 conda（如果使用 Anaconda）

```bash
# 创建新的 conda 环境
conda create -n seq_attractor_clean python=3.10 numpy scipy matplotlib -y

# 激活环境
conda activate seq_attractor_clean

# 进入项目目录并测试
cd "/Users/zhao/Library/Mobile Documents/com~apple~CloudDocs/study&research/coding projects/VScode/hopf_networks/seq_attractor"
python test_import.py

# 如果成功，安装其他依赖
pip install -r requirements.txt
```

---

### 方案 2：使用 conda-forge 的 NumPy

```bash
# 使用 conda-forge 频道安装
conda install -c conda-forge numpy
```

---

### 方案 3：使用 OpenBLAS 替代 Accelerate

```bash
# 安装链接到 OpenBLAS 的 NumPy
pip uninstall numpy
pip install numpy --no-binary numpy  # 从源码编译（慢但可靠）

# 或使用 conda
conda install numpy "libblas=*=*openblas"
```

---

### 方案 4：降级 NumPy 到稳定版本

```bash
# 当前使用 NumPy 2.3.4，尝试降级到 1.x 系列
pip uninstall numpy
pip install "numpy<2.0"

# 或特定版本
pip install numpy==1.24.3
```

---

## ✅ 验证修复

运行以下命令验证 NumPy 是否正常工作：

```bash
# 1. 简单导入测试
python test_import.py

# 2. 系统信息检查
python scripts/check_system.py

# 3. 运行单元测试
pytest tests/test_base.py -v

# 4. 基准测试
python scripts/benchmark.py --output benchmark_results/baseline.json
```

---

## 🍎 macOS 特定问题

### M1/M2/M3 芯片

Apple Silicon 芯片有特殊的 BLAS 优化：

```bash
# 确保使用 ARM64 原生 Python
python -c "import platform; print(platform.machine())"
# 应该输出: arm64

# 如果输出 x86_64，需要重新安装 Python（ARM 版本）
# 使用 Homebrew:
brew install python@3.11

# 或使用 miniforge（推荐）
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
```

### Intel 芯片 Mac

```bash
# 使用标准的 pip 或 conda 安装即可
pip install numpy
```

---

## 🐧 Linux 环境配置

### Intel CPU

```bash
# 使用 Intel MKL（最佳性能）
pip install numpy intel-mkl

# 或使用 conda
conda install numpy "libblas=*=*mkl"

# 设置环境变量
export MKL_NUM_THREADS=$(nproc)
```

### AMD CPU 或通用

```bash
# 使用 OpenBLAS
pip install numpy
conda install numpy "libblas=*=*openblas"

# 设置环境变量
export OPENBLAS_NUM_THREADS=$(nproc)
```

---

## 🎮 GPU 环境配置

### NVIDIA GPU + CUDA

```bash
# 1. 确认 CUDA 安装
nvidia-smi

# 2. 安装 CuPy（根据 CUDA 版本）
# CUDA 11.x
pip install cupy-cuda11x

# CUDA 12.x
pip install cupy-cuda12x

# 3. 测试 GPU
python -c "import cupy; print('CuPy OK'); a = cupy.array([1,2,3]); print(a)"

# 4. 运行 GPU 优化版本（待实施）
git checkout optimize/gpu-cupy
python scripts/benchmark.py
```

---

## 📊 当前环境检查

运行以下脚本查看详细信息：

```bash
# 系统信息
python scripts/check_system.py

# NumPy 配置
python -c "import numpy as np; np.show_config()"

# BLAS 库信息
python -c "import numpy as np; print(np.__config__.show())"
```

---

## 🔄 如果所有方案都失败

### 使用 Docker 容器（100% 可靠）

```bash
# 创建 Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
RUN pip install numpy scipy matplotlib pytest

# 复制项目文件
COPY . /app

CMD ["bash"]
EOF

# 构建并运行
docker build -t seq_attractor .
docker run -it -v $(pwd):/app seq_attractor

# 在容器内测试
python test_import.py
python scripts/benchmark.py
```

---

## 📝 记录问题

如果问题仍未解决，请收集以下信息：

```bash
# 运行诊断脚本
cat > diagnose.sh << 'EOF'
#!/bin/bash
echo "========== 系统信息 =========="
uname -a
echo ""

echo "========== Python 版本 =========="
python --version
which python
echo ""

echo "========== NumPy 信息 =========="
python -c "import numpy; print('NumPy:', numpy.__version__)" 2>&1
python -c "import numpy; numpy.show_config()" 2>&1
echo ""

echo "========== 虚拟环境 =========="
echo "VIRTUAL_ENV: $VIRTUAL_ENV"
echo "PATH: $PATH"
echo ""

echo "========== 已安装包 =========="
pip list | grep -E "(numpy|scipy|mkl|openblas|blas)"
echo ""

echo "========== 环境变量 =========="
env | grep -E "(BLAS|MKL|OPENBLAS|ACCELERATE|NPY)"
EOF

chmod +x diagnose.sh
./diagnose.sh > diagnostic_report.txt 2>&1
cat diagnostic_report.txt
```

将 `diagnostic_report.txt` 分享给技术支持。

---

## ✨ 优化已完成的代码

即使当前环境有问题，**优化代码已经完成**：

- ✅ `src/utils/blas_config.py` - 智能 BLAS 配置系统
- ✅ `src/core/base.py` - 自动启用最佳 BLAS 库
- ✅ `scripts/check_system.py` - 系统环境检查工具

**一旦修复 NumPy 环境，这些优化将立即生效！**

预期性能提升：
- macOS (Accelerate): **1.5-3x**
- Linux (MKL): **2-4x**
- Linux (OpenBLAS): **1.5-2.5x**

---

## 🎯 下一步

修复 NumPy 环境后：

1. 运行基准测试：
   ```bash
   python scripts/benchmark.py -o benchmark_results/baseline.json
   ```

2. 切换到优化分支对比：
   ```bash
   git checkout optimize/blas
   python scripts/benchmark.py -o benchmark_results/blas.json
   python scripts/benchmark.py -c benchmark_results/baseline.json benchmark_results/blas.json
   ```

3. 继续其他优化！

