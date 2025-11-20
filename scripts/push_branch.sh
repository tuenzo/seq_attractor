#!/bin/bash
# 推送优化分支到远程仓库

set -e

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== 推送优化分支到远程 ===${NC}\n"

# 切换到项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo -e "${YELLOW}当前工作目录:${NC} $PROJECT_ROOT"
echo

# 检查工作区状态
echo -e "${BLUE}检查工作区状态...${NC}"
if ! git diff-index --quiet HEAD --; then
    echo -e "${YELLOW}警告: 工作区有未提交的更改${NC}"
    echo "请先提交或暂存更改"
    git status --short
    exit 1
fi
echo -e "${GREEN}✓ 工作区干净${NC}\n"

# 显示当前分支
CURRENT_BRANCH=$(git branch --show-current)
echo -e "${BLUE}当前分支:${NC} $CURRENT_BRANCH"
echo

# 显示最近的提交
echo -e "${BLUE}最近的提交:${NC}"
git log --oneline --graph -5
echo

# 推送分支
echo -e "${BLUE}推送分支到远程...${NC}"
git push -u origin "$CURRENT_BRANCH"
echo -e "${GREEN}✓ 分支推送成功${NC}\n"

# 推送标签（如果有）
TAGS=$(git tag)
if [ -n "$TAGS" ]; then
    echo -e "${BLUE}检测到标签:${NC}"
    echo "$TAGS"
    echo
    read -p "是否推送所有标签? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git push origin --tags
        echo -e "${GREEN}✓ 标签推送成功${NC}\n"
    fi
fi

# 显示远程信息
echo -e "${BLUE}远程分支状态:${NC}"
git branch -vv
echo

# 显示远程仓库地址
REMOTE_URL=$(git remote get-url origin)
echo -e "${GREEN}=== 推送完成 ===${NC}"
echo -e "${BLUE}远程仓库:${NC} $REMOTE_URL"
echo -e "${BLUE}分支:${NC} $CURRENT_BRANCH"
echo
echo -e "${YELLOW}提示:${NC} 现在可以在其他机器上克隆和测试了"
echo
echo "# 在其他机器上运行:"
echo "git clone $REMOTE_URL"
echo "cd seq_attractor"
echo "git checkout $CURRENT_BRANCH"
echo "python scripts/check_system.py"
echo "python scripts/benchmark.py"

