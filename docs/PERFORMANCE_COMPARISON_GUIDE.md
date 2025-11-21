# 性能对比指南：优化前 vs 优化后

本指南说明如何对比 BLAS 优化前后的性能差异。

---

## 🎯 快速回答

**是的，需要切换 checkpoint** 来对比优化前后的代码。

**原因**：
- 当前 `optimize/blas` 分支已经包含 BLAS 优化
- 要对比优化效果，需要：
  1. 在**优化前**的代码运行测试 → 保存基线
  2. 在**优化后**的代码运行测试 → 对比基线

---

## 📋 方法 1: 使用基线标签（推荐）

项目已经有一个基线标签：`baseline-before-optimization`

### 步骤 1: 切换到基线版本并保存基线

```bash
# 1. 切换到基线标签
git checkout baseline-before-optimization

# 2. 确保依赖已安装
pip install -r requirements.txt

# 3. 运行测试并保存基线
python scripts/auto_test.py --save-baseline --baseline-file test_reports/baseline_before_optimize.json

# 4. 查看基线性能
cat test_reports/test_report_*.md | grep -A 10 "性能指标"
```

### 步骤 2: 切换回优化分支并对比

```bash
# 1. 切换回优化分支
git checkout optimize/blas

# 2. 运行测试并对比基线
python scripts/auto_test.py --compare --baseline-file test_reports/baseline_before_optimize.json

# 3. 查看对比报告
cat test_reports/test_report_*.md | grep -A 20 "基线对比"
```

### 步骤 3: 查看对比结果

对比报告会显示：

```
基线对比

性能对比表格：
| 测试项 | 基线 | 当前 | 变化 | 加速比 | 状态 |
|--------|------|------|------|--------|------|
| 训练时间 | 28.30s | 10.50s | -62.9% | 2.70x | ✓ 加速 |
| 回放时间 | 1.20ms | 0.40ms | -66.7% | 3.00x | ✓ 加速 |
| 总计时间 | 98.20s | 35.60s | -63.8% | 2.76x | ✓ 加速 |
```

---

## 📋 方法 2: 使用 main 分支作为基线

如果 `baseline-before-optimization` 标签不存在或有问题：

### 步骤 1: 在 main 分支保存基线

```bash
# 1. 切换到 main 分支
git checkout main

# 2. 运行测试并保存基线
python scripts/auto_test.py --save-baseline --baseline-file test_reports/baseline_main.json

# 3. 记录基线性能（可选）
cat test_reports/test_report_*.md > baseline_main_performance.txt
```

### 步骤 2: 切换到优化分支并对比

```bash
# 1. 切换到优化分支
git checkout optimize/blas

# 2. 对比性能
python scripts/auto_test.py --compare --baseline-file test_reports/baseline_main.json
```

---

## 📋 方法 3: 一键对比脚本

创建一个自动化脚本 `compare_optimization.sh`：

```bash
#!/bin/bash
# 自动对比优化前后的性能

set -e

echo "=== 性能对比：优化前 vs 优化后 ==="
echo ""

# 1. 保存当前分支
CURRENT_BRANCH=$(git branch --show-current)
echo "当前分支: $CURRENT_BRANCH"
echo ""

# 2. 切换到基线并保存基线
echo "步骤 1: 切换到基线版本..."
git checkout baseline-before-optimization

echo "步骤 2: 运行基线测试..."
python scripts/auto_test.py --save-baseline \
    --baseline-file test_reports/baseline_before_optimize.json

echo ""
echo "步骤 3: 切换回优化分支..."
git checkout $CURRENT_BRANCH

echo "步骤 4: 运行优化版本测试并对比..."
python scripts/auto_test.py --compare \
    --baseline-file test_reports/baseline_before_optimize.json

echo ""
echo "=== 对比完成 ==="
echo "查看报告: cat test_reports/test_report_*.md | grep -A 30 '基线对比'"
```

**使用方法**：
```bash
chmod +x compare_optimization.sh
./compare_optimization.sh
```

---

## 🔍 验证基线是否正确

### 检查基线文件

```bash
# 查看基线文件内容
cat test_reports/baseline_before_optimize.json | python -m json.tool

# 应该包含：
# - timestamp: 基线保存时间
# - git_branch: 基线分支/标签
# - git_commit: 基线提交
# - performance: 性能数据
```

### 检查基线版本

```bash
# 查看基线对应的代码版本
git show baseline-before-optimization:src/core/base.py | head -20

# 应该看到：没有 blas_config 相关的导入和调用
```

---

## 📊 预期对比结果

基于之前的测试，BLAS 优化后的预期结果：

| 指标 | 优化前 | 优化后 | 加速比 |
|------|--------|--------|--------|
| **训练时间** | ~28s | ~10s | **2.7x** ⚡ |
| **回放时间** | ~1.2ms | ~0.4ms | **3.0x** ⚡ |
| **鲁棒性测试** | ~69s | ~25s | **2.7x** ⚡ |
| **总计** | ~98s | ~36s | **2.8x** ⚡ |
| **准确率** | 100% | 100% | **无损** ✅ |

---

## 🎯 完整工作流示例

### 场景：首次对比优化效果

```bash
# === 阶段 1: 基线测试 ===

# 1. 切换到基线
git checkout baseline-before-optimization

# 2. 确保环境正确
python check_windows.py  # Windows
# 或
python scripts/check_system.py  # Linux/Mac

# 3. 运行完整测试并保存基线
python scripts/auto_test.py \
    --save-baseline \
    --baseline-file test_reports/baseline_before_optimize.json

# 4. 查看基线结果
cat test_reports/test_report_*.md | tail -50

# === 阶段 2: 优化版本测试 ===

# 5. 切换回优化分支
git checkout optimize/blas

# 6. 运行测试并对比
python scripts/auto_test.py \
    --compare \
    --baseline-file test_reports/baseline_before_optimize.json

# 7. 查看对比结果
cat test_reports/test_report_*.md | grep -A 30 "基线对比"

# === 阶段 3: 分析结果 ===

# 8. 生成对比摘要
python -c "
import json
with open('test_reports/baseline_before_optimize.json') as f:
    baseline = json.load(f)
with open('test_reports/benchmark_*.json') as f:  # 最新测试结果
    current = json.load(f)

print('优化效果:')
print(f\"训练: {baseline['performance']['training_time']:.2f}s -> {current['training_time']:.2f}s\")
print(f\"加速比: {baseline['performance']['training_time'] / current['training_time']:.2f}x\")
"
```

---

## ⚠️ 注意事项

### 1. 环境一致性

**重要**：确保基线测试和优化测试在**相同环境**下运行：

- ✅ 相同的硬件（CPU、内存）
- ✅ 相同的操作系统版本
- ✅ 相同的 Python 版本
- ✅ 相同的依赖版本

**如果环境不同**：
- 对比结果可能不准确
- 建议在相同设备上运行两次测试
- 或明确标注环境差异

### 2. 基线文件管理

```bash
# 为不同环境保存不同基线
baseline_before_optimize_mac.json      # Mac 环境基线
baseline_before_optimize_linux.json    # Linux 环境基线
baseline_before_optimize_windows.json  # Windows 环境基线

# 对比时使用对应环境的基线
python scripts/auto_test.py --compare \
    --baseline-file test_reports/baseline_before_optimize_mac.json
```

### 3. Git 状态

**建议**：在切换分支前提交或暂存当前更改

```bash
# 检查工作区状态
git status

# 如果有未提交的更改
git stash  # 暂存更改
# 或
git commit -m "WIP: 临时提交"
```

---

## 🚀 快速命令参考

```bash
# === 保存基线 ===
git checkout baseline-before-optimization
python scripts/auto_test.py --save-baseline --baseline-file baseline_before.json

# === 对比性能 ===
git checkout optimize/blas
python scripts/auto_test.py --compare --baseline-file baseline_before.json

# === 查看结果 ===
cat test_reports/test_report_*.md | grep -A 30 "基线对比"

# === 查看基线信息 ===
cat test_reports/baseline_before.json | python -m json.tool | grep -E "timestamp|git_branch|git_commit"
```

---

## 📝 总结

**回答你的问题**：

✅ **是的，需要切换 checkpoint** 来对比优化前后的性能

**推荐流程**：
1. `git checkout baseline-before-optimization` → 保存基线
2. `git checkout optimize/blas` → 对比基线

**或者**：如果基线已经保存过，可以直接在当前分支对比：
```bash
python scripts/auto_test.py --compare --baseline-file test_reports/baseline_before_optimize.json
```

---

## 📚 相关文档

- [自动化测试使用指南](AUTO_TEST_USAGE.md)
- [优化完整总结](optimization/OPTIMIZATION_COMPLETE_SUMMARY.md)
- [性能对比报告](optimization/PERFORMANCE_COMPARISON_REPORT.md)

---

**提示**: 使用 `git tag -l` 查看所有可用标签，使用 `git branch -a` 查看所有分支。

