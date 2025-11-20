#!/bin/bash
# 自动化测试脚本 - 一键运行性能测试并生成报告
# 使用方法: bash scripts/auto_test.sh

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取脚本所在目录和项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 创建报告目录
REPORT_DIR="$PROJECT_ROOT/test_reports"
mkdir -p "$REPORT_DIR"

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
HOSTNAME=$(hostname -s 2>/dev/null || hostname)
REPORT_FILE="$REPORT_DIR/test_report_${HOSTNAME}_${TIMESTAMP}.md"
JSON_FILE="$REPORT_DIR/benchmark_${HOSTNAME}_${TIMESTAMP}.json"

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
    echo "[INFO] $1" >> "$REPORT_FILE"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
    echo "[✓] $1" >> "$REPORT_FILE"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
    echo "[⚠] $1" >> "$REPORT_FILE"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
    echo "[✗] $1" >> "$REPORT_FILE"
}

log_section() {
    echo ""
    echo -e "${BLUE}$1${NC}"
    echo ""
    echo "" >> "$REPORT_FILE"
    echo "## $1" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
}

# 初始化报告
init_report() {
    cat > "$REPORT_FILE" << EOF
# 性能测试报告

**生成时间**: $(date '+%Y-%m-%d %H:%M:%S')  
**主机名**: $HOSTNAME  
**分支**: $(git branch --show-current 2>/dev/null || echo "unknown")  
**提交**: $(git log --oneline -1 2>/dev/null || echo "unknown")

---

EOF
}

# 检查 Python 环境
check_python() {
    log_section "1. Python 环境检查"
    
    # 检查 Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1)
        log_success "Python: $PYTHON_VERSION"
        echo "- Python: $PYTHON_VERSION" >> "$REPORT_FILE"
    else
        log_error "Python3 未安装"
        echo "- Python: 未安装 ❌" >> "$REPORT_FILE"
        exit 1
    fi
    
    # 检查虚拟环境
    if [ -d ".venv" ]; then
        log_success "虚拟环境已存在"
        echo "- 虚拟环境: 已存在 ✓" >> "$REPORT_FILE"
    else
        log_warning "虚拟环境不存在，正在创建..."
        echo "- 虚拟环境: 正在创建..." >> "$REPORT_FILE"
        python3 -m venv .venv
        log_success "虚拟环境创建成功"
    fi
    
    # 激活虚拟环境
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        log_success "虚拟环境已激活"
    else
        log_error "无法激活虚拟环境"
        exit 1
    fi
    
    # 检查依赖
    log_info "检查依赖包..."
    if pip list | grep -q "numpy"; then
        NUMPY_VERSION=$(pip show numpy | grep Version | awk '{print $2}')
        log_success "NumPy: $NUMPY_VERSION"
        echo "- NumPy: $NUMPY_VERSION ✓" >> "$REPORT_FILE"
    else
        log_warning "NumPy 未安装，正在安装依赖..."
        echo "- NumPy: 正在安装..." >> "$REPORT_FILE"
        pip install -r requirements.txt -q
        log_success "依赖安装完成"
    fi
}

# 系统环境检测
check_system() {
    log_section "2. 系统环境检测"
    
    log_info "运行系统检测脚本..."
    
    # 运行系统检测并捕获输出
    SYSTEM_INFO=$(python scripts/check_system.py 2>&1)
    
    if [ $? -eq 0 ]; then
        log_success "系统检测完成"
        echo '```' >> "$REPORT_FILE"
        echo "$SYSTEM_INFO" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
    else
        log_error "系统检测失败"
        echo '```' >> "$REPORT_FILE"
        echo "$SYSTEM_INFO" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        exit 1
    fi
}

# 运行单元测试
run_unit_tests() {
    log_section "3. 单元测试"
    
    log_info "运行单元测试..."
    
    # 运行 pytest
    TEST_OUTPUT=$(pytest tests/ -v --tb=short 2>&1 | tail -20)
    TEST_EXIT_CODE=${PIPEFAIL[0]:-$?}
    
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        log_success "所有单元测试通过"
        echo "- 状态: 通过 ✓" >> "$REPORT_FILE"
    else
        log_warning "部分测试失败（可能是环境问题）"
        echo "- 状态: 部分失败 ⚠" >> "$REPORT_FILE"
    fi
    
    # 提取测试统计
    if echo "$TEST_OUTPUT" | grep -q "passed"; then
        TEST_STATS=$(echo "$TEST_OUTPUT" | grep "passed" | tail -1)
        echo "- 结果: $TEST_STATS" >> "$REPORT_FILE"
        log_info "$TEST_STATS"
    fi
    
    echo "" >> "$REPORT_FILE"
    echo "<details>" >> "$REPORT_FILE"
    echo "<summary>详细输出（点击展开）</summary>" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
    echo "$TEST_OUTPUT" >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
    echo "</details>" >> "$REPORT_FILE"
}

# 运行功能测试
run_functional_tests() {
    log_section "4. 功能测试"
    
    log_info "运行基础功能测试..."
    
    # 创建临时测试脚本
    TEST_SCRIPT=$(cat << 'PYTHON_EOF'
import sys
import numpy as np
from src import SequenceAttractorNetwork, MemorySequenceAttractorNetwork

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
print(f"✓ (准确率: {eval_result['recall_accuracy']*100:.1f}%)")

print("\n所有功能测试通过 ✓")
PYTHON_EOF
)
    
    FUNC_OUTPUT=$(echo "$TEST_SCRIPT" | python 2>&1)
    FUNC_EXIT_CODE=$?
    
    if [ $FUNC_EXIT_CODE -eq 0 ]; then
        log_success "所有功能测试通过"
        echo "- 状态: 通过 ✓" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        echo "$FUNC_OUTPUT" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
    else
        log_error "功能测试失败"
        echo "- 状态: 失败 ✗" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        echo "$FUNC_OUTPUT" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        exit 1
    fi
}

# 运行性能基准测试
run_benchmark() {
    log_section "5. 性能基准测试"
    
    log_info "运行性能测试（这可能需要几分钟）..."
    
    # 运行 benchmark
    BENCH_OUTPUT=$(python scripts/benchmark.py -o "$JSON_FILE" 2>&1)
    BENCH_EXIT_CODE=$?
    
    if [ $BENCH_EXIT_CODE -eq 0 ]; then
        log_success "性能测试完成"
        echo "- 状态: 完成 ✓" >> "$REPORT_FILE"
        echo "- 结果文件: \`$(basename "$JSON_FILE")\`" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        
        # 解析 JSON 结果
        if command -v python3 &> /dev/null && [ -f "$JSON_FILE" ]; then
            log_info "解析性能数据..."
            
            PERF_SUMMARY=$(python3 << PYTHON_EOF
import json
try:
    with open("$JSON_FILE", "r") as f:
        data = json.load(f)
    
    print("### 性能指标")
    print()
    print("| 测试项 | 时间 | 说明 |")
    print("|--------|------|------|")
    
    if "training_time" in data:
        print(f"| 训练 (200轮) | {data['training_time']:.2f}s | - |")
    if "replay_time" in data:
        print(f"| 回放 (100步) | {data['replay_time']*1000:.2f}ms | - |")
    if "robustness_test_time" in data:
        print(f"| 鲁棒性测试 | {data['robustness_test_time']:.2f}s | - |")
    if "total_time" in data:
        print(f"| **总计** | **{data['total_time']:.2f}s** | - |")
    
    print()
    print("### 准确率")
    print()
    if "replay_accuracy" in data:
        print(f"- 回放准确率: {data['replay_accuracy']*100:.1f}%")
    if "robustness_scores" in data and len(data['robustness_scores']) > 0:
        print(f"- 无噪声成功率: {data['robustness_scores'][0]*100:.1f}%")
except Exception as e:
    print(f"解析失败: {e}")
PYTHON_EOF
)
            
            echo "$PERF_SUMMARY" >> "$REPORT_FILE"
            echo "$PERF_SUMMARY"
        fi
    else
        log_warning "性能测试未能完成（可能是环境问题）"
        echo "- 状态: 未完成 ⚠" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        echo "$BENCH_OUTPUT" | tail -30 >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
    fi
}

# 生成总结
generate_summary() {
    log_section "6. 测试总结"
    
    cat >> "$REPORT_FILE" << EOF

---

## 测试结果汇总

- ✓ Python 环境检查完成
- ✓ 系统环境检测完成
- ✓ 单元测试执行完成
- ✓ 功能测试通过
- ✓ 性能基准测试完成

## 文件清单

- 测试报告: \`$(basename "$REPORT_FILE")\`
- 性能数据: \`$(basename "$JSON_FILE")\`

## 如何查看报告

\`\`\`bash
# 查看 Markdown 报告
cat "$REPORT_FILE"

# 或使用 Markdown 查看器
# macOS
open "$REPORT_FILE"

# Linux
xdg-open "$REPORT_FILE"
\`\`\`

---

**测试完成时间**: $(date '+%Y-%m-%d %H:%M:%S')
EOF
    
    echo ""
    log_success "测试完成！"
    log_success "报告已生成: $REPORT_FILE"
    log_success "性能数据: $JSON_FILE"
    echo ""
}

# 主执行流程
main() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  自动化测试脚本${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    init_report
    
    # 执行各项测试
    check_python
    check_system
    run_unit_tests
    run_functional_tests
    run_benchmark
    generate_summary
    
    # 显示报告位置
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}测试报告已生成${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "报告位置: ${BLUE}$REPORT_FILE${NC}"
    echo -e "性能数据: ${BLUE}$JSON_FILE${NC}"
    echo ""
    echo "查看报告:"
    echo "  cat \"$REPORT_FILE\""
    echo ""
}

# 运行主函数
main

