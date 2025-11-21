#!/bin/bash
# 自动对比优化前后的性能
# 使用方法: bash compare_optimization.sh

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=== 性能对比：优化前 vs 优化后 ===${NC}"
echo ""

# 1. 保存当前分支
CURRENT_BRANCH=$(git branch --show-current)
echo -e "${YELLOW}当前分支: ${CURRENT_BRANCH}${NC}"
echo ""

# 2. 检查基线标签是否存在
if git rev-parse --verify baseline-before-optimization >/dev/null 2>&1; then
    BASELINE_REF="baseline-before-optimization"
    echo -e "${GREEN}✓ 找到基线标签: baseline-before-optimization${NC}"
elif git rev-parse --verify main >/dev/null 2>&1; then
    BASELINE_REF="main"
    echo -e "${GREEN}✓ 使用 main 分支作为基线${NC}"
else
    echo -e "${YELLOW}⚠ 未找到基线标签或 main 分支${NC}"
    echo "请手动指定基线版本"
    exit 1
fi

echo ""
echo -e "${BLUE}步骤 1: 切换到基线版本...${NC}"
git checkout "$BASELINE_REF"

echo ""
echo -e "${BLUE}步骤 2: 运行基线测试...${NC}"
echo "这可能需要几分钟..."
python scripts/auto_test.py --save-baseline \
    --baseline-file test_reports/baseline_before_blas.json

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠ 基线测试失败，但继续执行...${NC}"
fi

echo ""
echo -e "${BLUE}步骤 3: 切换回优化分支...${NC}"
git checkout "$CURRENT_BRANCH"

echo ""
echo -e "${BLUE}步骤 4: 运行优化版本测试并对比...${NC}"
python scripts/auto_test.py --compare \
    --baseline-file test_reports/baseline_before_blas.json

echo ""
echo -e "${GREEN}=== 对比完成 ===${NC}"
echo ""
echo "查看对比报告:"
echo "  cat test_reports/test_report_*.md | grep -A 30 '基线对比'"
echo ""
echo "或查看最新报告:"
LATEST_REPORT=$(ls -t test_reports/test_report_*.md | head -1)
echo "  cat $LATEST_REPORT"

