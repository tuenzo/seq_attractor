#!/usr/bin/env python3
"""
自动化测试脚本 (Python版本) - 增强版
适用于所有平台（Linux, macOS, Windows）

功能：
- 运行完整测试套件
- 生成性能测试报告
- 自动对比基线性能
- 支持保存/加载基线

使用方法:
    python scripts/auto_test.py                    # 完整测试
    python scripts/auto_test.py --quick            # 快速测试（跳过性能测试）
    python scripts/auto_test.py --save-baseline    # 保存当前结果为基线
    python scripts/auto_test.py --compare          # 与基线对比
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
        CYAN = '\033[0;36m'
        BOLD = '\033[1m'
        NC = '\033[0m'
    else:
        RED = GREEN = YELLOW = BLUE = CYAN = BOLD = NC = ''

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
    print(f"\n{Colors.BOLD}{Colors.BLUE}{title}{Colors.NC}\n")
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
        return True, None
    
    success, stdout, stderr = run_command(
        [sys.executable, "scripts/benchmark.py", "-o", json_file]
    )
    
    if success:
        log_success("性能测试完成", report_file)
        report_file.write("- 状态: 完成 ✓\n")
        report_file.write(f"- 结果文件: `{Path(json_file).name}`\n\n")
        
        # 解析 JSON 结果
        perf_data = None
        try:
            with open(json_file, 'r') as f:
                perf_data = json.load(f)
            
            report_file.write("### 性能指标\n\n")
            report_file.write("| 测试项 | 时间 | 说明 |\n")
            report_file.write("|--------|------|------|\n")
            
            # 安全获取并格式化数值
            def safe_format(value, multiplier=1, unit='s', decimals=2):
                """安全格式化数值，处理可能的字符串或 None 值"""
                try:
                    if value is None or value == 'N/A':
                        return 'N/A'
                    num_value = float(value) * multiplier
                    return f"{num_value:.{decimals}f}{unit}"
                except (ValueError, TypeError):
                    return str(value)
            
            if "training_time" in perf_data:
                time_str = safe_format(perf_data['training_time'], 1, 's', 2)
                report_file.write(f"| 训练 (200轮) | {time_str} | - |\n")
            if "replay_time" in perf_data:
                time_str = safe_format(perf_data['replay_time'], 1000, 'ms', 2)
                report_file.write(f"| 回放 (100步) | {time_str} | - |\n")
            if "robustness_test_time" in perf_data:
                time_str = safe_format(perf_data['robustness_test_time'], 1, 's', 2)
                report_file.write(f"| 鲁棒性测试 | {time_str} | - |\n")
            if "total_time" in perf_data:
                time_str = safe_format(perf_data['total_time'], 1, 's', 2)
                report_file.write(f"| **总计** | **{time_str}** | - |\n")
            
            report_file.write("\n### 准确率\n\n")
            if "replay_accuracy" in perf_data:
                acc_str = safe_format(perf_data['replay_accuracy'], 100, '%', 1)
                report_file.write(f"- 回放准确率: {acc_str}\n")
            if "robustness_scores" in perf_data and len(perf_data['robustness_scores']) > 0:
                score_str = safe_format(perf_data['robustness_scores'][0], 100, '%', 1)
                report_file.write(f"- 无噪声成功率: {score_str}\n")
            
            report_file.write("\n")
            
            # 控制台输出也使用安全格式化
            training_str = safe_format(perf_data.get('training_time'), 1, 's', 2)
            replay_str = safe_format(perf_data.get('replay_time'), 1000, 'ms', 2)
            log_info(f"训练时间: {training_str}", report_file)
            log_info(f"回放时间: {replay_str}", report_file)
            
        except Exception as e:
            log_warning(f"无法解析性能数据: {e}", report_file)
        
        return True, perf_data
    else:
        log_warning("性能测试未能完成", report_file)
        report_file.write("- 状态: 未完成 ⚠\n")
        report_file.write("```\n")
        report_file.write(stderr if stderr else stdout)
        report_file.write("\n```\n\n")
        return False, None

def save_baseline(perf_data, baseline_file):
    """保存基线数据"""
    if perf_data is None:
        log_error("没有性能数据可以保存为基线")
        return False
    
    try:
        baseline_data = {
            'timestamp': datetime.now().isoformat(),
            'hostname': socket.gethostname().split('.')[0],
            'system': platform.system(),
            'python': sys.version.split()[0],
            'performance': perf_data
        }
        
        # 获取 Git 信息
        try:
            success, branch, _ = run_command("git branch --show-current", check=False)
            if success:
                baseline_data['git_branch'] = branch.strip()
            success, commit, _ = run_command("git log --oneline -1", check=False)
            if success:
                baseline_data['git_commit'] = commit.strip()
        except:
            pass
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        
        log_success(f"基线已保存: {baseline_file}")
        return True
    except Exception as e:
        log_error(f"保存基线失败: {e}")
        return False

def load_baseline(baseline_file):
    """加载基线数据"""
    try:
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        log_success(f"基线已加载: {baseline_file}")
        return baseline_data
    except FileNotFoundError:
        log_warning(f"基线文件不存在: {baseline_file}")
        return None
    except Exception as e:
        log_error(f"加载基线失败: {e}")
        return None

def compare_with_baseline(current_perf, baseline_data, report_file):
    """对比当前性能与基线"""
    log_section("5. 基线对比", report_file)
    
    if baseline_data is None:
        log_warning("没有基线数据可供对比", report_file)
        report_file.write("- 状态: 无基线 ⚠\n\n")
        report_file.write("> 提示: 使用 `--save-baseline` 保存当前结果为基线\n\n")
        return
    
    baseline_perf = baseline_data.get('performance', {})
    
    # 显示基线信息
    report_file.write("### 基线信息\n\n")
    report_file.write(f"- 保存时间: {baseline_data.get('timestamp', 'N/A')}\n")
    report_file.write(f"- 主机: {baseline_data.get('hostname', 'N/A')}\n")
    report_file.write(f"- 系统: {baseline_data.get('system', 'N/A')}\n")
    report_file.write(f"- 分支: {baseline_data.get('git_branch', 'N/A')}\n")
    report_file.write(f"- 提交: {baseline_data.get('git_commit', 'N/A')}\n\n")
    
    # 性能对比
    report_file.write("### 性能对比\n\n")
    report_file.write("| 测试项 | 基线 | 当前 | 变化 | 加速比 | 状态 |\n")
    report_file.write("|--------|------|------|------|--------|------|\n")
    
    metrics = [
        ('training_time', '训练时间', 's', False),
        ('replay_time', '回放时间', 'ms', False, 1000),
        ('robustness_test_time', '鲁棒性测试', 's', False),
        ('total_time', '总计时间', 's', False),
        ('replay_accuracy', '回放准确率', '%', True, 100),
    ]
    
    comparisons = []
    
    for metric_key, metric_name, unit, higher_better, *multiplier in metrics:
        mult = multiplier[0] if multiplier else 1
        
        if metric_key not in baseline_perf or metric_key not in current_perf:
            continue
        
        # 安全转换为数值
        try:
            baseline_val = float(baseline_perf[metric_key]) * mult
            current_val = float(current_perf[metric_key]) * mult
        except (ValueError, TypeError):
            log_warning(f"无法对比 {metric_name}: 数据格式错误", report_file)
            continue
        
        # 计算变化
        if baseline_val != 0:
            change_pct = ((current_val - baseline_val) / baseline_val) * 100
            speedup = baseline_val / current_val if current_val != 0 else float('inf')
        else:
            change_pct = 0
            speedup = 1.0
        
        # 判断状态
        if higher_better:
            # 准确率类指标，越高越好
            if change_pct > 1:
                status = "✓ 提升"
                status_color = Colors.GREEN
            elif change_pct < -1:
                status = "✗ 下降"
                status_color = Colors.RED
            else:
                status = "- 持平"
                status_color = Colors.YELLOW
        else:
            # 时间类指标，越低越好
            if change_pct < -5:
                status = "✓ 加速"
                status_color = Colors.GREEN
            elif change_pct > 5:
                status = "✗ 变慢"
                status_color = Colors.RED
            else:
                status = "- 持平"
                status_color = Colors.YELLOW
        
        # 格式化输出
        baseline_str = f"{baseline_val:.2f}{unit}"
        current_str = f"{current_val:.2f}{unit}"
        change_str = f"{change_pct:+.1f}%"
        speedup_str = f"{speedup:.2f}x" if not higher_better else "-"
        
        report_file.write(f"| {metric_name} | {baseline_str} | {current_str} | {change_str} | {speedup_str} | {status} |\n")
        
        # 控制台输出
        print(f"{status_color}{status}{Colors.NC} {metric_name}: {baseline_str} -> {current_str} ({change_str})")
        
        comparisons.append({
            'metric': metric_name,
            'baseline': baseline_val,
            'current': current_val,
            'change_pct': change_pct,
            'speedup': speedup,
            'status': status,
            'higher_better': higher_better
        })
    
    report_file.write("\n")
    
    # 总结
    report_file.write("### 对比总结\n\n")
    
    improvements = [c for c in comparisons if "提升" in c['status'] or "加速" in c['status']]
    regressions = [c for c in comparisons if "下降" in c['status'] or "变慢" in c['status']]
    
    if improvements:
        report_file.write("**改进项**:\n")
        for c in improvements:
            report_file.write(f"- {c['metric']}: {abs(c['change_pct']):.1f}% {'提升' if c['higher_better'] else '加速'}\n")
        report_file.write("\n")
    
    if regressions:
        report_file.write("**退化项**:\n")
        for c in regressions:
            report_file.write(f"- {c['metric']}: {abs(c['change_pct']):.1f}% {'下降' if c['higher_better'] else '变慢'}\n")
        report_file.write("\n")
    
    if not improvements and not regressions:
        report_file.write("性能与基线基本持平，无明显变化。\n\n")
    
    # 控制台总结
    print(f"\n{Colors.BOLD}对比总结:{Colors.NC}")
    print(f"  改进项: {Colors.GREEN}{len(improvements)}{Colors.NC}")
    print(f"  退化项: {Colors.RED}{len(regressions)}{Colors.NC}")
    print(f"  持平项: {Colors.YELLOW}{len(comparisons) - len(improvements) - len(regressions)}{Colors.NC}")

def generate_summary(report_file, results, has_comparison=False):
    """生成总结"""
    log_section("6. 测试总结" if not has_comparison else "7. 测试总结", report_file)
    
    report_file.write("\n---\n\n")
    report_file.write("## 测试结果汇总\n\n")
    
    for key, value in results.items():
        symbol = "✓" if value else "✗"
        report_file.write(f"- {symbol} {key}\n")
    
    report_file.write("\n---\n\n")
    report_file.write(f"**测试完成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def main():
    parser = argparse.ArgumentParser(description='自动化测试脚本 - 增强版')
    parser.add_argument('--quick', action='store_true', help='快速测试（跳过性能测试）')
    parser.add_argument('--save-baseline', action='store_true', help='保存当前测试结果为基线')
    parser.add_argument('--compare', action='store_true', help='与基线对比')
    parser.add_argument('--baseline-file', default='test_reports/baseline.json', help='基线文件路径')
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
    baseline_file_path = Path(args.baseline_file)
    
    print(f"\n{Colors.BOLD}{Colors.BLUE}========================================{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.BLUE}  自动化测试脚本 (增强版){Colors.NC}")
    print(f"{Colors.BOLD}{Colors.BLUE}========================================{Colors.NC}\n")
    
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
        perf_data = None
        if not args.quick:
            success, perf_data = run_benchmark(rf, str(json_file_path))
            results['性能基准测试'] = success
        else:
            log_info("跳过性能测试（快速模式）", rf)
        
        # 基线对比
        if args.compare and perf_data:
            baseline_data = load_baseline(baseline_file_path)
            compare_with_baseline(perf_data, baseline_data, rf)
        
        # 保存基线
        if args.save_baseline and perf_data:
            if save_baseline(perf_data, baseline_file_path):
                log_success(f"✓ 基线已保存到: {baseline_file_path}")
        
        # 生成总结
        generate_summary(rf, results, has_comparison=args.compare)
    
    # 显示报告位置
    print(f"\n{Colors.BOLD}{Colors.GREEN}========================================{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.GREEN}测试完成{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.GREEN}========================================{Colors.NC}\n")
    print(f"报告位置: {Colors.CYAN}{report_file_path}{Colors.NC}")
    if not args.quick:
        print(f"性能数据: {Colors.CYAN}{json_file_path}{Colors.NC}")
    if args.save_baseline:
        print(f"基线文件: {Colors.CYAN}{baseline_file_path}{Colors.NC}")
    print(f"\n查看报告:\n  cat \"{report_file_path}\"\n")
    
    # 返回测试结果
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
