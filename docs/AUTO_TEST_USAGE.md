# 自动化测试脚本使用指南

本指南介绍如何使用增强版自动化测试脚本，包括基线保存和性能对比功能。

---

## 🎯 功能特性

### 核心功能
- ✅ 完整的环境检查（Python、依赖、系统配置）
- ✅ 自动化功能测试（单序列、多序列、准确率）
- ✅ 性能基准测试（训练、回放、鲁棒性）
- ✅ 生成详细的 Markdown 报告
- ✅ 保存 JSON 格式的性能数据

### 增强功能
- 🆕 **保存基线**: 保存当前性能为参考基线
- 🆕 **性能对比**: 自动对比当前性能与基线
- 🆕 **加速比计算**: 显示相对基线的加速比
- 🆕 **变化分析**: 标识性能提升或退化

---

## 📖 使用方法

### 1. 基础测试（完整模式）

运行完整的测试套件，包括性能测试：

```bash
python scripts/auto_test.py
```

**输出**：
- `test_reports/test_report_<hostname>_<timestamp>.md` - 测试报告
- `test_reports/benchmark_<hostname>_<timestamp>.json` - 性能数据

### 2. 快速测试模式

跳过性能测试，仅验证功能：

```bash
python scripts/auto_test.py --quick
```

**适用场景**：
- 快速验证环境配置
- 功能回归测试
- CI/CD 快速检查

### 3. 保存基线

将当前测试结果保存为基线，用于后续对比：

```bash
# 完整测试并保存基线
python scripts/auto_test.py --save-baseline

# 使用自定义基线文件路径
python scripts/auto_test.py --save-baseline --baseline-file my_baseline.json
```

**基线文件内容**：
```json
{
  "timestamp": "2025-11-20T19:45:00",
  "hostname": "my-server",
  "system": "Linux",
  "python": "3.10.12",
  "git_branch": "optimize/blas",
  "git_commit": "2d65ae1 feat: 添加一键自动化测试脚本",
  "performance": {
    "training_time": 10.5,
    "replay_time": 0.0004,
    "robustness_test_time": 25.1,
    "total_time": 35.6,
    "replay_accuracy": 1.0,
    "robustness_scores": [1.0, 1.0, 0.98, ...]
  }
}
```

### 4. 性能对比

对比当前性能与已保存的基线：

```bash
# 运行测试并对比基线
python scripts/auto_test.py --compare

# 使用自定义基线文件
python scripts/auto_test.py --compare --baseline-file my_baseline.json
```

**对比输出示例**：
```
基线对比

性能对比表格：
| 测试项 | 基线 | 当前 | 变化 | 加速比 | 状态 |
|--------|------|------|------|--------|------|
| 训练时间 | 28.30s | 10.50s | -62.9% | 2.70x | ✓ 加速 |
| 回放时间 | 1.20ms | 0.40ms | -66.7% | 3.00x | ✓ 加速 |
| 鲁棒性测试 | 68.70s | 25.10s | -63.5% | 2.74x | ✓ 加速 |
| 总计时间 | 98.20s | 35.60s | -63.8% | 2.76x | ✓ 加速 |
| 回放准确率 | 100.0% | 100.0% | +0.0% | - | - 持平 |

对比总结:
  改进项: 4
  退化项: 0
  持平项: 1
```

### 5. 组合使用

可以组合多个选项：

```bash
# 完整测试 + 对比 + 保存新基线
python scripts/auto_test.py --compare --save-baseline

# 这个流程会：
# 1. 加载旧基线
# 2. 运行完整测试
# 3. 与旧基线对比
# 4. 保存新基线
```

---

## 🔄 典型工作流

### 工作流 1: 建立初始基线

在优化之前，建立性能基线：

```bash
# 1. 切换到基线分支
git checkout baseline-before-optimization

# 2. 运行测试并保存基线
python scripts/auto_test.py --save-baseline --baseline-file test_reports/baseline_before_optimization.json

# 3. 查看基线报告
cat test_reports/test_report_*.md
```

### 工作流 2: 测试优化效果

优化代码后，对比性能：

```bash
# 1. 切换到优化分支
git checkout optimize/blas

# 2. 运行测试并对比基线
python scripts/auto_test.py --compare --baseline-file test_reports/baseline_before_optimization.json

# 3. 查看对比报告
cat test_reports/test_report_*.md | grep -A 20 "基线对比"
```

### 工作流 3: 跨环境测试

在不同设备测试并对比：

```bash
# Mac 上建立基线
python scripts/auto_test.py --save-baseline --baseline-file test_reports/baseline_mac.json

# 将基线文件复制到其他设备
scp test_reports/baseline_mac.json user@linux-server:~/seq_attractor/test_reports/

# Linux 服务器上对比
python scripts/auto_test.py --compare --baseline-file test_reports/baseline_mac.json
```

### 工作流 4: CI/CD 集成

```bash
# CI 环境快速验证
python scripts/auto_test.py --quick

# 每日性能测试（对比前一天）
python scripts/auto_test.py --compare --baseline-file test_reports/baseline_$(date -d yesterday +%Y%m%d).json
```

---

## 📊 报告解读

### 性能指标说明

| 指标 | 说明 | 期望方向 |
|------|------|----------|
| **训练时间** | 完成 200 轮训练的时间 | 越低越好 |
| **回放时间** | 回放 100 步的平均时间 | 越低越好 |
| **鲁棒性测试** | 多噪声水平测试时间 | 越低越好 |
| **总计时间** | 所有测试的总时间 | 越低越好 |
| **回放准确率** | 回放的准确度 | 越高越好（应保持 100%）|

### 状态标识

- **✓ 加速 / 提升**: 性能改进 > 5%
- **✗ 变慢 / 下降**: 性能退化 > 5%
- **- 持平**: 性能变化在 ±5% 之间

### 加速比计算

```
加速比 = 基线时间 / 当前时间

示例:
基线训练时间: 28.3s
当前训练时间: 10.5s
加速比: 28.3 / 10.5 = 2.70x
```

---

## 🎯 实战示例

### 示例 1: 验证 BLAS 优化效果

```bash
# 步骤 1: 在优化前保存基线
git checkout main
python scripts/auto_test.py --save-baseline \
    --baseline-file test_reports/baseline_no_blas.json

# 步骤 2: 切换到优化分支
git checkout optimize/blas

# 步骤 3: 运行对比测试
python scripts/auto_test.py --compare \
    --baseline-file test_reports/baseline_no_blas.json

# 预期结果: 2-3x 加速
```

### 示例 2: 对比不同环境

```bash
# Mac M2 环境
python scripts/auto_test.py --save-baseline \
    --baseline-file test_reports/baseline_mac_m2.json

# Intel CPU 环境
python scripts/auto_test.py --compare \
    --baseline-file test_reports/baseline_mac_m2.json \
    --save-baseline \
    --baseline-file test_reports/baseline_intel_cpu.json

# 分析不同硬件的性能差异
```

### 示例 3: 回归测试

```bash
# 每次代码修改后运行
python scripts/auto_test.py --compare \
    --baseline-file test_reports/baseline_stable.json

# 如果所有测试通过且性能无退化，更新基线
python scripts/auto_test.py --save-baseline \
    --baseline-file test_reports/baseline_stable.json
```

---

## 🔧 故障排除

### 问题 1: 基线文件不存在

**错误**:
```
[⚠] 基线文件不存在: test_reports/baseline.json
```

**解决**:
```bash
# 先创建基线
python scripts/auto_test.py --save-baseline
```

### 问题 2: 基线与当前环境不匹配

**症状**: 对比结果显示巨大差异（>10x）

**原因**: 基线可能是在不同硬件或配置下生成的

**建议**:
- 在相同环境下重新生成基线
- 或使用 `--baseline-file` 指定正确的基线文件

### 问题 3: 性能数据不完整

**错误**:
```
[⚠] 无法解析性能数据
```

**解决**:
```bash
# 检查 JSON 文件是否有效
python -m json.tool test_reports/benchmark_*.json

# 重新运行性能测试
python scripts/auto_test.py
```

### 问题 4: 权限问题

**错误**:
```
Permission denied: test_reports/baseline.json
```

**解决**:
```bash
# 检查目录权限
chmod 755 test_reports/
chmod 644 test_reports/*.json

# 或使用其他位置
python scripts/auto_test.py --baseline-file ~/baseline.json
```

---

## 📁 文件说明

### 生成的文件

```
test_reports/
├── baseline.json                          # 默认基线文件
├── test_report_<hostname>_<timestamp>.md  # 测试报告
└── benchmark_<hostname>_<timestamp>.json  # 性能数据
```

### 基线文件管理

**命名建议**:
```
baseline_<分支>_<环境>.json

示例:
- baseline_main_mac_m2.json
- baseline_optimize_blas_linux_intel.json
- baseline_gpu_rtx3090.json
```

**版本控制**:
```bash
# 将基线文件加入版本控制（可选）
git add test_reports/baseline_*.json
git commit -m "chore: 添加性能基线"

# 或忽略基线文件
echo "test_reports/baseline*.json" >> .gitignore
```

---

## 💡 最佳实践

### 1. 基线管理策略

**建议**: 为不同场景维护多个基线

```bash
# 环境基线
baseline_mac_m2.json
baseline_linux_intel_mkl.json
baseline_linux_amd_openblas.json
baseline_gpu_rtx3090.json

# 分支基线
baseline_main.json
baseline_optimize_blas.json
baseline_optimize_gpu.json

# 时间基线
baseline_2025_01_stable.json
baseline_2025_02_stable.json
```

### 2. 测试频率

- **功能开发**: 每次提交后运行 `--quick`
- **性能优化**: 每次优化后运行完整测试并对比
- **发布前**: 运行完整测试套件
- **定期**: 每周/每月运行一次基准测试

### 3. 报告管理

```bash
# 保留重要报告
mkdir -p test_reports/archive/
mv test_reports/test_report_important_*.md test_reports/archive/

# 清理旧报告（保留最近 10 个）
ls -t test_reports/test_report_*.md | tail -n +11 | xargs rm -f
```

### 4. 自动化集成

**Git Hook 示例** (`.git/hooks/pre-push`):
```bash
#!/bin/bash
echo "运行快速测试..."
python scripts/auto_test.py --quick
if [ $? -ne 0 ]; then
    echo "测试失败，推送已取消"
    exit 1
fi
```

---

## 🚀 下一步

测试完成后：

1. **查看报告**
   ```bash
   cat test_reports/test_report_*.md
   ```

2. **分析性能**
   - 对比基线找出瓶颈
   - 识别优化机会

3. **分享结果**
   - 将报告提交到 GitHub Issue
   - 更新优化文档

4. **继续优化**
   - 根据对比结果进行针对性优化
   - 保存新基线以追踪进展

---

## 📚 相关文档

- [快速开始指南](QUICK_START.md)
- [跨平台部署](DEPLOYMENT_TESTING_GUIDE.md)
- [优化计划](optimization/OPTIMIZATION_PLAN.md)
- [故障排除](optimization/TROUBLESHOOTING.md)

---

**祝测试顺利！如有问题请查看文档或提 Issue。** 🎉

