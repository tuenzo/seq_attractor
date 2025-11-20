#!/bin/bash
# 
# 项目版本打包备份脚本
# 用途：为不熟悉 Git 的用户提供简单的文件备份方案
#
# 使用方式：
#   ./scripts/backup_version.sh                    # 创建时间戳备份
#   ./scripts/backup_version.sh baseline           # 创建命名备份
#   ./scripts/backup_version.sh blas-optimized     # 创建优化版本备份
#

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR" || exit 1

# 备份目录
BACKUP_ROOT="${PROJECT_DIR}/../seq_attractor_backups"
mkdir -p "$BACKUP_ROOT"

# 备份名称
if [ -n "$1" ]; then
    BACKUP_NAME="$1"
else
    BACKUP_NAME="auto"
fi

# 时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 备份文件名
BACKUP_FILE="${BACKUP_ROOT}/${BACKUP_NAME}_${TIMESTAMP}.tar.gz"

echo "========================================"
echo "项目版本打包备份"
echo "========================================"
echo ""
echo "项目目录: $PROJECT_DIR"
echo "备份目录: $BACKUP_ROOT"
echo "备份名称: ${BACKUP_NAME}_${TIMESTAMP}"
echo ""

# 排除的文件和目录
EXCLUDE_PATTERNS=(
    '.git'
    '__pycache__'
    '.pytest_cache'
    '.venv'
    '*.pyc'
    '.DS_Store'
    'figure5_results/*'
    'figure5_results_local_exapmles/*'
    'benchmark_results/*.json'
    '*.png'
)

# 构建排除参数
EXCLUDE_ARGS=()
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    EXCLUDE_ARGS+=("--exclude=$pattern")
done

# 显示将要排除的内容
echo "排除内容:"
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    echo "  - $pattern"
done
echo ""

# 创建备份
echo "正在创建备份..."
tar -czf "$BACKUP_FILE" "${EXCLUDE_ARGS[@]}" -C "$(dirname "$PROJECT_DIR")" "$(basename "$PROJECT_DIR")"

if [ $? -eq 0 ]; then
    # 获取文件大小
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        SIZE=$(ls -lh "$BACKUP_FILE" | awk '{print $5}')
    else
        # Linux
        SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    fi
    
    echo ""
    echo "✓ 备份创建成功！"
    echo ""
    echo "备份文件: $BACKUP_FILE"
    echo "文件大小: $SIZE"
    echo ""
    
    # 列出所有备份
    echo "========================================"
    echo "所有备份文件:"
    echo "========================================"
    ls -lth "$BACKUP_ROOT"/*.tar.gz 2>/dev/null | head -10 | awk '{printf "  %s %s %s  %s\n", $6, $7, $8, $9}'
    
    echo ""
    echo "使用以下命令恢复备份:"
    echo "  tar -xzf \"$BACKUP_FILE\" -C /tmp/restore_test"
    echo ""
else
    echo ""
    echo "✗ 备份失败！"
    exit 1
fi

