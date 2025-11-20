#!/usr/bin/env python3
"""
自动化测试脚本 (Python版本)
适用于所有平台（Linux, macOS, Windows）

使用方法:
    python scripts/auto_test.py
    python scripts/auto_test.py --quick  # 快速测试（跳过性能测试）
"""

import os
import sys
import json
import subprocess
import platform
import socket
from datetime import datetime
from pathlib import Path
import argparse

# 颜色定义（跨平台）
class Colors:
    if platform.system() != 'Windows':
        RED = '\033[0;31m'
        GREEN = '\033[0;32m'
        YELLOW = '\033[1;33m'
        BLUE = '\033[0;34m'
        NC = '\033[0m'
    else:
        RED = GREEN = YELLOW = BLUE = NC = ''

def log_info(msg, report_file=None):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}")
    if report_file:
        report_file.write(f"[INFO] {msg}\n")

def log_success(msg, report_file=None):
    print(f"{Colors.GREEN}[✓]{Colors.NC} {msg}")
    if report_file:
        report_file.write(f"[✓] {msg}\n")

def log_warning(msg, report_file=None):
    print(f"{Colors.YELLOW}[⚠]{Colors.NC} {msg}")
    if report_file:
        report_file.write(f"[⚠] {msg}\n")

def log_error(msg, report_file=None):
    print(f"{Colors.RED}[✗]{Colors.NC} {msg}")
    if report_file:
        report_file.write(f"[✗] {msg}\n")

def log_section(title, report_file=None):
    print(f"\n{Colors.BLUE}{title}{Colors.NC}\n")
    if report_file:
        report_file.write(f"\n## {title}\n\n")

def run_command(cmd, check=True, capture_output=True):
    """运行命令并返回结果"""
    try:
        if isinstance(cmd, str):
            result = subprocess.run(
                cmd, 
                shell=True, 
                check=check, 
                capture_output=capture_output,
                text=True
            )
        else:
            result = subprocess.run(
                cmd, 
                check=check, 
                capture_output=capture_output,
                text=True
            )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout if e.stdout else "", e.stderr if e.stderr else ""
    except Exception as e:
        return False, "", str(e)

def check_python_env(report_file):
    """检查 Python 环境"""
    log_section("1. Python 环境检查", report_file)
    
    # Python 版本
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    log_success(f"Python: {python_version}", report_file)
    report_file.write(f"- Python: {python_version}\n")
    
    # 检查必要的包
    try:
        import numpy
        log_success(f"NumPy: {numpy.__version__}", report_file)
        report_file.write(f"- NumPy: {numpy.__version__} ✓\n")
    except ImportError:
        log_warning("NumPy 未安装，尝试安装...", report_file)
        report_file.write("- NumPy: 正在安装...\n")
        success, _, _ = run_command([sys.executable, "-m", "pip", "install", "numpy", "-q"])
        if success:
            import numpy
            log_success(f"NumPy 安装成功: {numpy.__version__}", report_file)
        else:
            log_error("NumPy 安装失败", report_file)
            return False
    
    try:
        import matplotlib
        log_success(f"Matplotlib: {matplotlib.__version__}", report_file)
        report_file.write(f"- Matplotlib: {matplotlib.__version__} ✓\n")
    except ImportError:
        log_warning("Matplotlib 未安装", report_file)
        report_file.write("- Matplotlib: 未安装 ⚠\n")
    
    return True

def check_system(report_file):
    """系统环境检测"""
    log_section("2. 系统环境检测", report_file)
    
    log_info("运行系统检测脚本...", report_file)
    
    # 检查 check_system.py 是否存在
    check_system_path = Path("scripts/check_system.py")
    if not check_system_path.exists():
        log_warning("check_system.py 不存在，跳过系统检测", report_file)
        return True
    
    success, stdout, stderr = run_command([sys.executable, "scripts/check_system.py"])
    
    if success:
        log_success("系统检测完成", report_file)
        report_file.write("```\n")
        report_file.write(stdout)
        report_file.write("\n```\n\n")
        return True
    else:
        log_error("系统检测失败", report_file)
        report_file.write("```\n")
        report_file.write(stderr if stderr else stdout)
        report_file.write("\n```\n\n")
        return False

def run_functional_tests(report_file):
    """运行功能测试"""
    log_section("3. 功能测试", report_file)
    
    log_info("运行基础功能测试...", report_file)
    
    try:
        # 测试导入
        from src import (
            SequenceAttractorNetwork, 
            MemorySequenceAttractorNetwork
        )
        import numpy as np
        
        # 测试1: 单序列训练
        print("测试1: 单序列训练和回放...", end=" ")
        net1 = SequenceAttractorNetwork(N_v=10, T=5, N_h=20, eta=0.01)
        net1.train(num_epochs=50, seed=42, verbose=False)
        replayed1 = net1.replay(max_steps=10)
        assert replayed1.shape == (10, 10), "回放形状错误"
        print("✓")
        
        # 测试2: 多序列训练
        print("测试2: 多序列训练...", end=" ")
        net2 = MemorySequenceAttractorNetwork(N_v=10, T=5, N_h=20, eta=0.01)
        sequences = net2.generate_multiple_sequences(num_sequences=2, seeds=[1,2])
        net2.train(x=sequences, num_epochs=50, verbose=False, interleaved=True)
        replayed2 = net2.replay(sequence_index=0, max_steps=10)
        assert replayed2.shape == (10, 10), "回放形状错误"
        print("✓")
        
        # 测试3: 准确率评估
        print("测试3: 准确率评估...", end=" ")
        eval_result = net1.evaluate_replay(replayed1)
        assert 'recall_accuracy' in eval_result, "评估结果缺少准确率"
        assert 0 <= eval_result['recall_accuracy'] <= 1, "准确率范围错误"
        accuracy = eval_result['recall_accuracy'] * 100
        print(f"✓ (准确率: {accuracy:.1f}%)")
        
        log_success("所有功能测试通过", report_file)
        report_file.write("- 状态: 通过 ✓\n")
        report_file.write(f"- 单序列训练: ✓\n")
        report_file.write(f"- 多序列训练: ✓\n")
        report_file.write(f"- 准确率评估: ✓ ({accuracy:.1f}%)\n\n")
        return True
        
    except Exception as e:
        log_error(f"功能测试失败: {e}", report_file)
        report_file.write(f"- 状态: 失败 ✗\n")
        report_file.write(f"- 错误: {e}\n\n")
        return False

def run_benchmark(report_file, json_file):
    """运行性能基准测试"""
    log_section("4. 性能基准测试", report_file)
    
    log_info("运行性能测试（这可能需要几分钟）...", report_file)
    
    # 检查 benchmark.py 是否存在
    benchmark_path = Path("scripts/benchmark.py")
    if not benchmark_path.exists():
        log_warning("benchmark.py 不存在，跳过性能测试", report_file)
        return True
    
    success, stdout, stderr = run_command(
        [sys.executable, "scripts/benchmark.py", "-o", json_file]
    )
    
    if success:
        log_success("性能测试完成", report_file)
        report_file.write("- 状态: 完成 ✓\n")
        report_file.write(f"- 结果文件: `{Path(json_file).name}`\n\n")
        
        # 解析 JSON 结果
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            report_file.write("### 性能指标\n\n")
            report_file.write("| 测试项 | 时间 | 说明 |\n")
            report_file.write("|--------|------|------|\n")
            
            if "training_time" in data:
                report_file.write(f"| 训练 (200轮) | {data['training_time']:.2f}s | - |\n")
            if "replay_time" in data:
                report_file.write(f"| 回放 (100步) | {data['replay_time']*1000:.2f}ms | - |\n")
            if "robustness_test_time" in data:
                report_file.write(f"| 鲁棒性测试 | {data['robustness_test_time']:.2f}s | - |\n")
            if "total_time" in data:
                report_file.write(f"| **总计** | **{data['total_time']:.2f}s** | - |\n")
            
            report_file.write("\n### 准确率\n\n")
            if "replay_accuracy" in data:
                report_file.write(f"- 回放准确率: {data['replay_accuracy']*100:.1f}%\n")
            if "robustness_scores" in data and len(data['robustness_scores']) > 0:
                report_file.write(f"- 无噪声成功率: {data['robustness_scores'][0]*100:.1f}%\n")
            
            report_file.write("\n")
            
            log_info(f"训练时间: {data.get('training_time', 'N/A'):.2f}s", report_file)
            log_info(f"回放时间: {data.get('replay_time', 'N/A')*1000:.2f}ms", report_file)
            
        except Exception as e:
            log_warning(f"无法解析性能数据: {e}", report_file)
        
        return True
    else:
        log_warning("性能测试未能完成", report_file)
        report_file.write("- 状态: 未完成 ⚠\n")
        report_file.write("```\n")
        report_file.write(stderr if stderr else stdout)
        report_file.write("\n```\n\n")
        return False

def generate_summary(report_file, results):
    """生成总结"""
    log_section("5. 测试总结", report_file)
    
    report_file.write("\n---\n\n")
    report_file.write("## 测试结果汇总\n\n")
    
    for key, value in results.items():
        symbol = "✓" if value else "✗"
        report_file.write(f"- {symbol} {key}\n")
    
    report_file.write("\n---\n\n")
    report_file.write(f"**测试完成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def main():
    parser = argparse.ArgumentParser(description='自动化测试脚本')
    parser.add_argument('--quick', action='store_true', help='快速测试（跳过性能测试）')
    args = parser.parse_args()
    
    # 切换到项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    # 创建报告目录
    report_dir = project_root / "test_reports"
    report_dir.mkdir(exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hostname = socket.gethostname().split('.')[0]
    report_file_path = report_dir / f"test_report_{hostname}_{timestamp}.md"
    json_file_path = report_dir / f"benchmark_{hostname}_{timestamp}.json"
    
    print(f"\n{Colors.BLUE}========================================{Colors.NC}")
    print(f"{Colors.BLUE}  自动化测试脚本 (Python版){Colors.NC}")
    print(f"{Colors.BLUE}========================================{Colors.NC}\n")
    
    # 初始化报告
    with open(report_file_path, 'w', encoding='utf-8') as rf:
        rf.write("# 性能测试报告\n\n")
        rf.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        rf.write(f"**主机名**: {hostname}\n")
        rf.write(f"**系统**: {platform.system()} {platform.release()}\n")
        rf.write(f"**Python**: {sys.version.split()[0]}\n")
        
        # 获取 Git 信息
        try:
            success, branch, _ = run_command("git branch --show-current", check=False)
            if success:
                rf.write(f"**分支**: {branch.strip()}\n")
            success, commit, _ = run_command("git log --oneline -1", check=False)
            if success:
                rf.write(f"**提交**: {commit.strip()}\n")
        except:
            pass
        
        rf.write("\n---\n\n")
        
        # 执行各项测试
        results = {}
        
        # Python 环境检查
        results['Python 环境检查'] = check_python_env(rf)
        
        # 系统检测
        results['系统环境检测'] = check_system(rf)
        
        # 功能测试
        results['功能测试'] = run_functional_tests(rf)
        
        # 性能测试（可选）
        if not args.quick:
            results['性能基准测试'] = run_benchmark(rf, str(json_file_path))
        else:
            log_info("跳过性能测试（快速模式）", rf)
        
        # 生成总结
        generate_summary(rf, results)
    
    # 显示报告位置
    print(f"\n{Colors.GREEN}========================================{Colors.NC}")
    print(f"{Colors.GREEN}测试完成{Colors.NC}")
    print(f"{Colors.GREEN}========================================{Colors.NC}\n")
    print(f"报告位置: {Colors.BLUE}{report_file_path}{Colors.NC}")
    if not args.quick:
        print(f"性能数据: {Colors.BLUE}{json_file_path}{Colors.NC}")
    print(f"\n查看报告:\n  cat \"{report_file_path}\"\n")
    
    # 返回测试结果
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()

