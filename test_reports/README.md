# 测试报告目录

此目录用于存储自动化测试脚本生成的报告和数据文件。

---

## 📁 文件类型

### 1. 测试报告 (`.md`)
- **格式**: `test_report_<hostname>_<timestamp>.md`
- **内容**: 完整的测试结果、系统信息、性能数据、基线对比
- **示例**: `test_report_myserver_20251120_193000.md`

### 2. 性能数据 (`.json`)
- **格式**: `benchmark_<hostname>_<timestamp>.json`
- **内容**: 详细的性能测试数据（JSON 格式）
- **示例**: `benchmark_myserver_20251120_193000.json`

### 3. 基线文件 (`.json`)
- **格式**: `baseline.json` 或自定义名称
- **内容**: 用于性能对比的参考基线
- **示例**: `baseline_mac_m2.json`, `baseline_before_optimization.json`

---

## 🚀 快速开始

### 运行测试

```bash
# 完整测试
python scripts/auto_test.py

# 快速测试（跳过性能测试）
python scripts/auto_test.py --quick
```

### 保存基线

```bash
# 保存默认基线
python scripts/auto_test.py --save-baseline

# 保存到自定义文件
python scripts/auto_test.py --save-baseline --baseline-file baseline_my_env.json
```

### 对比基线

```bash
# 与默认基线对比
python scripts/auto_test.py --compare

# 与自定义基线对比
python scripts/auto_test.py --compare --baseline-file baseline_my_env.json
```

---

## 📊 报告示例

### 测试报告内容

测试报告包含以下部分：

1. **系统信息**: 主机名、系统、Python版本、Git信息
2. **环境检查**: Python依赖、BLAS配置、GPU支持
3. **功能测试**: 单序列、多序列、准确率验证
4. **性能测试**: 训练时间、回放时间、鲁棒性测试
5. **基线对比**: 与参考基线的性能对比（如使用 `--compare`）
6. **测试总结**: 所有测试结果汇总

### 性能数据格式

```json
{
  "timestamp": "2025-11-20T19:30:00",
  "training_time": 10.5,
  "replay_time": 0.0004,
  "robustness_test_time": 25.1,
  "total_time": 35.6,
  "replay_accuracy": 1.0,
  "robustness_scores": [1.0, 1.0, 0.98, 0.85, 0.72, 0.64]
}
```

---

## 🔄 典型工作流

### 场景 1: 建立性能基线

```bash
# 在优化前运行测试并保存基线
git checkout main
python scripts/auto_test.py --save-baseline --baseline-file baseline_before_opt.json
```

### 场景 2: 验证优化效果

```bash
# 切换到优化分支并对比
git checkout optimize/blas
python scripts/auto_test.py --compare --baseline-file baseline_before_opt.json
```

### 场景 3: 跨环境对比

```bash
# Mac 上生成基线
python scripts/auto_test.py --save-baseline --baseline-file baseline_mac.json

# 复制到 Linux 服务器
scp test_reports/baseline_mac.json user@server:~/seq_attractor/test_reports/

# Linux 上对比
python scripts/auto_test.py --compare --baseline-file baseline_mac.json
```

---

## 📋 文件管理

### 推荐的目录结构

```
test_reports/
├── README.md                              # 本文件
├── baseline.json                          # 默认基线
├── baseline_mac_m2.json                   # Mac M2 环境基线
├── baseline_linux_intel.json              # Linux Intel 环境基线
├── baseline_before_optimization.json      # 优化前基线
├── test_report_mac_20251120_193000.md     # 测试报告
├── benchmark_mac_20251120_193000.json     # 性能数据
└── archive/                               # 历史报告归档
    └── ...
```

### 清理旧文件

```bash
# 保留最近 10 个报告
ls -t test_reports/test_report_*.md | tail -n +11 | xargs rm -f

# 归档重要报告
mkdir -p test_reports/archive/
mv test_reports/test_report_important_*.md test_reports/archive/
```

---

## 📖 查看报告

### 命令行查看

```bash
# 查看最新报告
ls -t test_reports/test_report_*.md | head -1 | xargs cat

# 查看基线对比部分
ls -t test_reports/test_report_*.md | head -1 | xargs grep -A 20 "基线对比"

# 查看性能数据
cat test_reports/benchmark_*.json | python -m json.tool
```

### 使用 Markdown 查看器

```bash
# macOS
open test_reports/test_report_*.md

# Linux
xdg-open test_reports/test_report_*.md

# 使用 VS Code
code test_reports/test_report_*.md
```

---

## 🎯 最佳实践

### 1. 基线命名规范

使用描述性的名称：

```
baseline_<环境>_<配置>.json

示例:
- baseline_mac_m2_accelerate.json
- baseline_linux_intel_mkl.json
- baseline_gpu_rtx3090.json
- baseline_main_before_opt.json
```

### 2. 定期测试

- 每次代码修改后：运行 `--quick`
- 每次性能优化后：运行完整测试并对比
- 每周：运行一次基准测试
- 发布前：运行完整测试套件

### 3. 版本控制

**推荐**: 将重要的基线文件加入版本控制

```bash
# 添加基线文件
git add test_reports/baseline_*.json
git commit -m "chore: 添加性能基线"
```

**或**: 使用 `.gitignore` 忽略所有测试报告

```gitignore
# .gitignore
test_reports/*.md
test_reports/benchmark_*.json
```

### 4. 文档化

为每个基线创建说明文档：

```markdown
# baseline_mac_m2.json

**创建时间**: 2025-11-20
**环境**: Mac M2, 16GB RAM
**系统**: macOS 14.5
**Python**: 3.10.12
**分支**: optimize/blas
**提交**: 2d65ae1

**性能指标**:
- 训练时间: 10.5s
- 回放时间: 0.4ms
- 总计: 35.6s
- 准确率: 100%
```

---

## 📚 相关文档

- [自动化测试使用指南](../docs/AUTO_TEST_USAGE.md)
- [快速开始](../docs/QUICK_START.md)
- [部署测试指南](../docs/DEPLOYMENT_TESTING_GUIDE.md)

---

**提示**: 使用 `python scripts/auto_test.py --help` 查看所有可用选项。

