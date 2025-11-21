# Windows 环境配置指南

本指南帮助你在 Windows 上正确配置和运行测试。

---

## 🔧 常见问题和解决方案

### 问题 1: UnicodeDecodeError (GBK 编码错误)

**错误信息**:
```
UnicodeDecodeError: 'gbk' codec can't decode byte 0xae
```

**原因**: Windows 默认使用 GBK 编码，而代码输出包含 UTF-8 字符。

**解决方案已内置**: 最新版本的脚本已经自动处理了编码问题。如果仍然遇到，请更新代码：

```bash
git pull origin optimize/blas
```

### 问题 2: No module named 'src'

**错误信息**:
```
ModuleNotFoundError: No module named 'src'
```

**原因**: Python 找不到项目模块。

**解决方案**:

#### 方法 A: 确保在项目根目录运行

```bash
# 检查当前目录
pwd  # Git Bash
cd   # CMD

# 应该在 seq_attractor 目录下
# 如果不是，切换到正确目录
cd /e/myMD/CodingProject/Seq_Att/seq_attractor
```

#### 方法 B: 设置 PYTHONPATH

**Git Bash / PowerShell**:
```bash
export PYTHONPATH="${PYTHONPATH}:."  # Git Bash
$env:PYTHONPATH="$PWD"  # PowerShell
python scripts/auto_test.py
```

**CMD**:
```cmd
set PYTHONPATH=%CD%
python scripts/auto_test.py
```

#### 方法 C: 使用绝对路径运行

```bash
cd /e/myMD/CodingProject/Seq_Att/seq_attractor
python -c "import sys; sys.path.insert(0, '.'); exec(open('scripts/auto_test.py').read())"
```

### 问题 3: 虚拟环境问题

**症状**: 依赖包找不到或版本不对

**解决方案**:

```bash
# 1. 创建虚拟环境
python -m venv .venv

# 2. 激活虚拟环境
# Git Bash
source .venv/Scripts/activate

# CMD
.venv\Scripts\activate.bat

# PowerShell
.venv\Scripts\Activate.ps1

# 3. 升级 pip
python -m pip install --upgrade pip

# 4. 安装依赖
pip install -r requirements.txt

# 5. 验证安装
pip list | grep -E "numpy|matplotlib"
```

### 问题 4: PowerShell 执行策略错误

**错误信息**:
```
无法加载文件，因为在此系统上禁止运行脚本
```

**解决方案**:

```powershell
# 临时允许（仅当前会话）
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 或使用 Git Bash / CMD
```

---

## 🚀 推荐工作流（Windows）

### 使用 Git Bash（推荐）

Git Bash 提供类似 Linux 的环境，兼容性最好：

```bash
# 1. 打开 Git Bash

# 2. 进入项目目录
cd /e/myMD/CodingProject/Seq_Att/seq_attractor

# 3. 创建并激活虚拟环境
python -m venv .venv
source .venv/Scripts/activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 运行测试
python scripts/auto_test.py

# 或快速模式
python scripts/auto_test.py --quick
```

### 使用 Anaconda Prompt

如果使用 Anaconda：

```bash
# 1. 打开 Anaconda Prompt

# 2. 创建环境
conda create -n seq_attractor python=3.10 numpy matplotlib scipy pytest -y
conda activate seq_attractor

# 3. 进入项目目录
cd E:\myMD\CodingProject\Seq_Att\seq_attractor

# 4. 安装其他依赖
pip install -r requirements.txt

# 5. 运行测试
python scripts/auto_test.py
```

### 使用 CMD

```cmd
# 1. 打开 CMD

# 2. 进入项目目录
cd E:\myMD\CodingProject\Seq_Att\seq_attractor

# 3. 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate.bat

# 4. 安装依赖
pip install -r requirements.txt

# 5. 设置路径
set PYTHONPATH=%CD%

# 6. 运行测试
python scripts\auto_test.py
```

---

## ✅ 验证环境配置

运行这个快速检查脚本：

```python
# 保存为 check_windows.py
import sys
import os
from pathlib import Path

print("=== Windows 环境检查 ===\n")

# Python 版本
print(f"Python 版本: {sys.version}")
print(f"Python 路径: {sys.executable}\n")

# 当前目录
print(f"当前目录: {os.getcwd()}")
print(f"项目根目录: {Path.cwd()}\n")

# 检查关键文件
files_to_check = [
    'src/__init__.py',
    'scripts/auto_test.py',
    'requirements.txt',
    'README.md'
]

print("关键文件检查:")
for file in files_to_check:
    exists = Path(file).exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {file}")

print()

# 检查导入
print("模块导入检查:")
try:
    # 添加当前目录到路径
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    
    import numpy as np
    print(f"  ✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"  ✗ NumPy: {e}")

try:
    import matplotlib
    print(f"  ✓ Matplotlib {matplotlib.__version__}")
except ImportError as e:
    print(f"  ✗ Matplotlib: {e}")

try:
    from src import SequenceAttractorNetwork
    print(f"  ✓ src.SequenceAttractorNetwork")
except ImportError as e:
    print(f"  ✗ src: {e}")

print("\n=== 检查完成 ===")
```

运行检查：
```bash
python check_windows.py
```

---

## 🐛 调试技巧

### 1. 详细错误信息

```python
import traceback
try:
    # 你的代码
    from src import SequenceAttractorNetwork
except Exception as e:
    print(traceback.format_exc())
```

### 2. 检查 Python 路径

```python
import sys
print("\n".join(sys.path))
```

### 3. 检查当前目录

```python
import os
from pathlib import Path
print(f"当前目录: {os.getcwd()}")
print(f"目录内容: {list(Path('.').glob('*'))}")
```

### 4. 强制使用 UTF-8

在脚本开头添加：
```python
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
```

---

## 📝 Windows 特定配置

### 设置默认编码为 UTF-8

**方法 1: 环境变量**
```bash
# 在系统环境变量中添加
PYTHONIOENCODING=utf-8
```

**方法 2: 注册表**（Windows 10 1903+）

1. Win + R 运行 `regedit`
2. 导航到 `HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Command Processor`
3. 新建 DWORD 值 `AutoRun`，设置为 `chcp 65001`

### 长路径支持

如果路径超过 260 字符：

1. Win + R 运行 `gpedit.msc`
2. 导航到：计算机配置 > 管理模板 > 系统 > 文件系统
3. 启用"启用 Win32 长路径"

---

## 🎯 完整安装步骤（Windows）

### 使用 Git Bash（推荐）

```bash
# 1. 克隆代码
git clone https://github.com/tuenzo/seq_attractor.git
cd seq_attractor
git checkout optimize/blas

# 2. 创建虚拟环境
python -m venv .venv
source .venv/Scripts/activate

# 3. 升级 pip
python -m pip install --upgrade pip

# 4. 安装依赖
pip install -r requirements.txt

# 5. 验证安装
python check_windows.py  # 使用上面的检查脚本

# 6. 运行测试
python scripts/auto_test.py --quick

# 7. 如果成功，运行完整测试
python scripts/auto_test.py
```

### 使用 Anaconda

```bash
# 1. 创建环境
conda create -n seq_attractor python=3.10 -y
conda activate seq_attractor

# 2. 克隆代码
cd E:\myMD\CodingProject\Seq_Att
git clone https://github.com/tuenzo/seq_attractor.git
cd seq_attractor
git checkout optimize/blas

# 3. 安装依赖
conda install numpy matplotlib scipy pytest -y
pip install -r requirements.txt

# 4. 运行测试
python scripts/auto_test.py --quick
```

---

## 📞 获取帮助

如果仍然遇到问题：

1. **生成诊断报告**
   ```bash
   python check_windows.py > windows_diagnostic.txt
   ```

2. **查看详细错误**
   ```bash
   python scripts/auto_test.py 2>&1 | tee test_output.txt
   ```

3. **检查依赖版本**
   ```bash
   pip list > requirements_installed.txt
   ```

4. **提交 Issue**
   - 附上 `windows_diagnostic.txt`
   - 附上 `test_output.txt`
   - 说明 Windows 版本和 Python 版本

---

## ✨ 快速解决方案（一键修复）

创建一个启动脚本 `run_test.bat`：

```batch
@echo off
echo === 序列吸引子网络测试 ===
echo.

REM 设置编码
chcp 65001 > nul

REM 设置 Python 路径
set PYTHONPATH=%CD%
set PYTHONIOENCODING=utf-8

REM 激活虚拟环境
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo 创建虚拟环境...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo 安装依赖...
    pip install -r requirements.txt
)

REM 运行测试
echo.
echo 运行测试...
python scripts\auto_test.py %*

pause
```

使用方法：
```bash
# 双击运行，或在 CMD 中：
run_test.bat

# 快速模式
run_test.bat --quick

# 对比基线
run_test.bat --compare
```

---

**提示**: 推荐使用 Git Bash 以获得最佳体验，它提供了类似 Linux 的环境，兼容性问题更少。

