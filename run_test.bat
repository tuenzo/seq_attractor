@echo off
REM 序列吸引子网络自动化测试启动脚本 (Windows)
REM 使用方法: 
REM   run_test.bat              - 完整测试
REM   run_test.bat --quick      - 快速测试
REM   run_test.bat --compare    - 对比基线

echo ========================================
echo   序列吸引子网络测试
echo ========================================
echo.

REM 设置 UTF-8 编码
chcp 65001 > nul

REM 设置 Python 环境变量
set PYTHONPATH=%CD%
set PYTHONIOENCODING=utf-8

REM 检查虚拟环境
if exist .venv\Scripts\activate.bat (
    echo [INFO] 激活虚拟环境...
    call .venv\Scripts\activate.bat
) else (
    echo [INFO] 虚拟环境不存在，正在创建...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] 创建虚拟环境失败
        echo [INFO] 请确保已安装 Python 3.8+
        pause
        exit /b 1
    )
    
    call .venv\Scripts\activate.bat
    
    echo [INFO] 升级 pip...
    python -m pip install --upgrade pip > nul 2>&1
    
    echo [INFO] 安装依赖...
    pip install -r requirements.txt
    
    if errorlevel 1 (
        echo [ERROR] 安装依赖失败
        pause
        exit /b 1
    )
    
    echo [SUCCESS] 环境配置完成
)

echo.
echo [INFO] 运行测试...
echo.

REM 运行测试脚本
python scripts\auto_test.py %*

if errorlevel 1 (
    echo.
    echo [ERROR] 测试失败，请查看上方错误信息
    echo.
    echo 常见问题:
    echo   1. 确保在项目根目录运行此脚本
    echo   2. 运行 python check_windows.py 检查环境
    echo   3. 查看 docs/WINDOWS_SETUP.md 获取帮助
) else (
    echo.
    echo [SUCCESS] 测试完成
)

echo.
pause

