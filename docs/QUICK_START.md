# 快速开始指南（适用所有平台）

一键运行测试并生成报告，适用于 Linux、macOS、Windows。

---

## 🚀 一键测试（推荐）

### 方法 1: Python 脚本（跨平台）

```bash
# 克隆代码
git clone https://github.com/tuenzo/seq_attractor.git
cd seq_attractor
git checkout optimize/blas

# 创建虚拟环境并安装依赖
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# Windows: .venv\Scripts\activate

pip install -r requirements.txt

# 运行自动化测试
python scripts/auto_test.py
```

**快速模式**（跳过性能测试，适合快速验证）：
```bash
python scripts/auto_test.py --quick
```

### 方法 2: Bash 脚本（Linux/Mac）

```bash
# 克隆代码
git clone https://github.com/tuenzo/seq_attractor.git
cd seq_attractor
git checkout optimize/blas

# 运行自动化测试（会自动创建环境、安装依赖）
bash scripts/auto_test.sh
```

---

## 📊 测试报告

测试完成后会生成：

1. **Markdown 报告**: `test_reports/test_report_<hostname>_<timestamp>.md`
   - 包含所有测试结果
   - 系统环境信息
   - 性能指标

2. **JSON 性能数据**: `test_reports/benchmark_<hostname>_<timestamp>.json`
   - 详细的性能测试数据
   - 可用于后续分析对比

**查看报告**：
```bash
# 查看最新报告
ls -lt test_reports/*.md | head -1 | xargs cat

# 或直接打开
cat test_reports/test_report_*.md
```

---

## 🔧 常见问题与解决方案

### 问题 1: `git` 命令不存在

**症状**：
```
bash: git: command not found
```

**解决方案**：
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install git

# CentOS/RHEL
sudo yum install git

# macOS
brew install git
# 或安装 Xcode Command Line Tools
xcode-select --install
```

**替代方案**（直接下载代码）：
```bash
# 下载 ZIP 文件
curl -L https://github.com/tuenzo/seq_attractor/archive/refs/heads/optimize/blas.zip -o seq_attractor.zip
unzip seq_attractor.zip
cd seq_attractor-optimize-blas
```

### 问题 2: `python3` 或 `pip` 不存在

**症状**：
```
bash: python3: command not found
```

**解决方案**：
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install python3 python3-pip python3-venv

# CentOS/RHEL 7
sudo yum install python3 python3-pip

# CentOS/RHEL 8+
sudo dnf install python3 python3-pip

# macOS
brew install python3
```

**检查 Python 版本**：
```bash
python3 --version  # 应该 >= 3.8
pip3 --version
```

### 问题 3: 虚拟环境创建失败

**症状**：
```
The virtual environment was not created successfully
```

**解决方案**：
```bash
# 安装 venv 模块
# Ubuntu/Debian
sudo apt install python3-venv

# 或使用 virtualenv
pip3 install virtualenv
python3 -m virtualenv .venv
```

### 问题 4: NumPy/Matplotlib 安装失败

**症状**：
```
ERROR: Failed building wheel for numpy
```

**解决方案**：

**方法 A - 安装系统依赖**：
```bash
# Ubuntu/Debian
sudo apt install python3-dev build-essential

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel

# macOS
xcode-select --install
```

**方法 B - 使用预编译版本**：
```bash
pip install numpy --only-binary=:all:
pip install matplotlib --only-binary=:all:
```

**方法 C - 使用 conda**：
```bash
# 下载并安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 创建环境
conda create -n seq_attractor python=3.10 numpy matplotlib scipy pytest -y
conda activate seq_attractor

# 安装其余依赖
pip install -r requirements.txt
```

### 问题 5: 权限问题（Permission denied）

**症状**：
```
Permission denied: './scripts/auto_test.sh'
```

**解决方案**：
```bash
# 添加执行权限
chmod +x scripts/auto_test.sh
chmod +x scripts/*.py

# 重新运行
bash scripts/auto_test.sh
```

### 问题 6: 网络问题（无法访问 GitHub）

**症状**：
```
fatal: unable to access 'https://github.com/...'
```

**解决方案**：

**方法 A - 使用代理**：
```bash
# 设置 HTTP 代理
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port

# Git 代理
git config --global http.proxy http://your-proxy:port
git config --global https.proxy http://your-proxy:port
```

**方法 B - 使用 SSH**：
```bash
git clone git@github.com:tuenzo/seq_attractor.git
```

**方法 C - 手动下载**：
1. 访问 https://github.com/tuenzo/seq_attractor
2. 点击 "Code" -> "Download ZIP"
3. 解压到本地

### 问题 7: BLAS 库相关警告

**症状**：
```
WARNING: BLAS library not optimized
```

**这不是错误！** 代码会自动选择最优的 BLAS 库。

**验证 BLAS 配置**：
```bash
python scripts/check_system.py
```

**手动优化（可选）**：
```bash
# Intel CPU
pip install intel-mkl

# AMD CPU
conda install numpy "libblas=*=*openblas"
```

### 问题 8: 测试运行缓慢

**症状**：测试运行超过 10 分钟

**原因**：性能测试需要训练网络，在性能较弱的机器上可能较慢。

**解决方案**：
```bash
# 使用快速模式（跳过性能测试）
python scripts/auto_test.py --quick

# 或者只运行功能测试
python -c "
from src import SequenceAttractorNetwork
net = SequenceAttractorNetwork(N_v=10, T=5, N_h=20, eta=0.01)
net.train(num_epochs=10, seed=42, verbose=False)
replayed = net.replay(max_steps=10)
print('✓ 功能测试通过')
"
```

### 问题 9: GPU 未检测到（在 GPU 机器上）

**症状**：
```
GPU 支持: 否
```

**解决方案**：
```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 如果没有输出，安装驱动
# Ubuntu
sudo apt install nvidia-driver-<version>

# 验证 CUDA
nvcc --version

# 安装 CuPy（可选，为未来 GPU 优化做准备）
pip install cupy-cuda12x  # 根据 CUDA 版本
```

**注意**：当前分支（`optimize/blas`）主要优化 CPU，GPU 加速将在后续分支实现。

### 问题 10: Matplotlib 字体警告

**症状**：
```
Matplotlib is building the font cache
```

**这是正常的！** 首次运行时 Matplotlib 会构建字体缓存，需要几秒钟。

**如果想加速**：
```bash
# 清理缓存后重建
rm -rf ~/.cache/matplotlib
python -c "import matplotlib.pyplot"
```

---

## 📝 最小测试脚本

如果自动化脚本无法运行，可以使用这个最小测试脚本：

```python
# 保存为 minimal_test.py
import sys
print("Python 版本:", sys.version)

try:
    import numpy as np
    print("✓ NumPy:", np.__version__)
    
    from src import SequenceAttractorNetwork
    print("✓ 导入成功")
    
    net = SequenceAttractorNetwork(N_v=10, T=5, N_h=20, eta=0.01)
    net.train(num_epochs=10, seed=42, verbose=False)
    replayed = net.replay(max_steps=10)
    print("✓ 训练和回放成功")
    
    eval_result = net.evaluate_replay(replayed)
    print(f"✓ 准确率: {eval_result['recall_accuracy']*100:.1f}%")
    
    print("\n所有测试通过！ ✓")
    
except Exception as e:
    print(f"\n✗ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
```

**运行**：
```bash
python minimal_test.py
```

---

## 🎯 测试清单

使用这个清单确保测试完整：

- [ ] 代码克隆成功
- [ ] Python 3.8+ 已安装
- [ ] 虚拟环境创建并激活
- [ ] 依赖安装完成（`pip install -r requirements.txt`）
- [ ] 系统检测通过（`python scripts/check_system.py`）
- [ ] 功能测试通过
- [ ] 性能测试完成（或使用 `--quick` 跳过）
- [ ] 报告已生成

---

## 💡 快速诊断命令

遇到问题？运行这些命令诊断：

```bash
# 1. 检查 Python
python3 --version
which python3

# 2. 检查虚拟环境
source .venv/bin/activate  # 或 .venv\Scripts\activate (Windows)
which python
python --version

# 3. 检查依赖
pip list | grep -E "numpy|matplotlib|scipy|pytest"

# 4. 检查导入
python -c "from src import SequenceAttractorNetwork; print('✓ 导入成功')"

# 5. 检查 NumPy 配置
python -c "import numpy as np; np.show_config()"

# 6. 运行最小测试
python -c "
from src import SequenceAttractorNetwork
net = SequenceAttractorNetwork(N_v=5, T=3, N_h=10, eta=0.01)
net.train(num_epochs=5, seed=1, verbose=False)
print('✓ 基础功能正常')
"
```

---

## 📞 获取帮助

1. **查看完整文档**
   - [部署测试指南](DEPLOYMENT_TESTING_GUIDE.md)
   - [故障排除](optimization/TROUBLESHOOTING.md)

2. **运行诊断**
   ```bash
   python scripts/check_system.py
   ```

3. **提交 Issue**
   - GitHub: https://github.com/tuenzo/seq_attractor/issues
   - 附上错误信息和系统信息

---

## 🎉 测试成功后

恭喜！你的环境已经配置完成。

**下一步**：

1. **查看测试报告**
   ```bash
   cat test_reports/test_report_*.md
   ```

2. **对比性能**
   - 将报告发送给项目维护者
   - 对比不同环境的性能数据

3. **继续优化**
   - 当前分支：BLAS 优化（CPU）
   - 后续分支：GPU 加速、混合精度等

---

**祝测试顺利！有问题随时查看文档或提 Issue。** 🚀

