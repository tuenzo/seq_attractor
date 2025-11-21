#!/bin/bash
# 创建基线性能数据
# 在 baseline-before-optimization 标签上运行此脚本

set -e

echo "=========================================="
echo "  创建基线性能数据"
echo "=========================================="
echo ""

# 检查当前分支/标签
CURRENT_REF=$(git describe --tags --exact-match 2>/dev/null || git branch --show-current)
echo "当前版本: $CURRENT_REF"
echo ""

# 检查是否在基线版本
if [ "$CURRENT_REF" != "baseline-before-optimization" ]; then
    echo "⚠ 警告: 当前不在 baseline-before-optimization 标签"
    read -p "是否切换到基线标签? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git checkout baseline-before-optimization
    else
        echo "取消操作"
        exit 1
    fi
fi

# 检查脚本是否存在
if [ ! -f "scripts/baseline_test.py" ]; then
    echo "错误: scripts/baseline_test.py 不存在"
    echo "请确保在项目根目录运行此脚本"
    exit 1
fi

# 运行基线测试
echo "运行基线性能测试..."
echo "这可能需要几分钟..."
echo ""

python scripts/baseline_test.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "基线测试完成"
    echo "=========================================="
    echo ""
    echo "下一步:"
    echo "  1. 找到生成的 JSON 文件（见上方输出）"
    echo "  2. 复制为基线文件:"
    echo "     cp test_reports/baseline_before_blas_*.json test_reports/baseline_before_blas.json"
    echo "  3. 切换回优化分支:"
    echo "     git checkout optimize/blas"
    echo "  4. 运行对比:"
    echo "     python scripts/auto_test.py --compare --baseline-file test_reports/baseline_before_blas.json"
else
    echo ""
    echo "基线测试失败，请检查错误信息"
    exit 1
fi

