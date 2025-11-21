# 基线版本测试指南

由于 `baseline-before-optimization` 标签创建时还没有 `auto_test.py` 脚本，本指南说明如何在基线版本运行测试。

---

## 🎯 问题说明

**问题**: `baseline-before-optimization` 标签中没有 `auto_test.py` 脚本

**原因**: 该标签是在添加自动化测试脚本之前创建的

**解决方案**: 使用专门的基线测试脚本 `baseline_test.py`

---

## 🚀 快速开始

### 方法 1: 使用自动化脚本（推荐）

```bash
# 1. 切换到基线标签
git checkout baseline-before-optimization

# 2. 运行基线测试脚本
bash scripts/create_baseline.sh

# 3. 复制生成的 JSON 文件为基线
cp test_reports/baseline_before_optimize_*.json test_reports/baseline_before_optimize.json

# 4. 切换回优化分支并对比
git checkout optimize/blas
python scripts/auto_test.py --compare --baseline-file test_reports/baseline_before_optimize.json
```

### 方法 2: 手动运行

```bash
# 1. 切换到基线标签
git checkout baseline-before-optimization

# 2. 运行基线测试
python scripts/baseline_test.py

# 3. 查看生成的 JSON 文件路径（脚本会输出）

# 4. 复制为基线文件
cp test_reports/baseline_before_optimize_*.json test_reports/baseline_before_optimize.json

# 5. 切换回优化分支
git checkout optimize/blas

# 6. 运行对比
python scripts/auto_test.py --compare --baseline-file test_reports/baseline_before_optimize.json
```

---

## 📋 详细步骤

### 步骤 1: 切换到基线版本

```bash
# 查看当前分支
git branch --show-current

# 切换到基线标签
git checkout baseline-before-optimization

# 确认切换成功
git describe --tags --exact-match
# 应该输出: baseline-before-optimization
```

### 步骤 2: 运行基线测试

```bash
# 确保在项目根目录
pwd
# 应该在: .../seq_attractor

# 运行基线测试脚本
python scripts/baseline_test.py
```

**脚本会**：
- 运行性能测试（训练、回放、鲁棒性）
- 生成 JSON 格式的性能数据
- 显示测试结果
- 输出保存的文件路径

**输出示例**：
```
============================================================
基线版本性能测试
============================================================

测试参数:
  N_v = 50
  T = 30
  N_h = 200
  eta = 0.001
  训练轮数 = 200

测试 1: 单序列训练...
  ✓ 训练完成: 28.30s
测试 2: 回放测试...
  ✓ 回放完成: 1.20ms (平均)
测试 3: 准确率评估...
  ✓ 准确率: 100.0%
测试 4: 鲁棒性测试（简化版）...
  噪声 0.0: 100.0% (20/20)
  噪声 0.1: 95.0% (19/20)
  噪声 0.2: 80.0% (16/20)
  ✓ 鲁棒性测试完成: 68.70s

============================================================
测试完成
============================================================

性能指标:
  训练时间: 28.30s
  回放时间: 1.20ms
  鲁棒性测试: 68.70s
  总计: 97.00s
  准确率: 100.0%

结果已保存: test_reports/baseline_before_optimize_hostname_20251120_123456.json
```

### 步骤 3: 保存基线文件

```bash
# 找到最新生成的基线文件
LATEST_BASELINE=$(ls -t test_reports/baseline_before_optimize_*.json | head -1)

# 复制为标准基线文件名
cp "$LATEST_BASELINE" test_reports/baseline_before_optimize.json

# 验证
cat test_reports/baseline_before_optimize.json | python -m json.tool | head -20
```

### 步骤 4: 切换回优化分支并对比

```bash
# 切换回优化分支
git checkout optimize/blas

# 运行对比测试
python scripts/auto_test.py --compare \
    --baseline-file test_reports/baseline_before_optimize.json

# 查看对比结果
cat test_reports/test_report_*.md | grep -A 30 "基线对比"
```

---

## 🔍 基线测试脚本说明

### `scripts/baseline_test.py`

**功能**：
- 运行与 `auto_test.py` 兼容的性能测试
- 生成相同格式的 JSON 数据
- 可以在基线版本运行（不依赖新脚本）

**测试内容**：
1. **单序列训练**: 200 轮训练，测量时间
2. **回放测试**: 100 次回放，计算平均时间
3. **准确率评估**: 验证回放准确率
4. **鲁棒性测试**: 简化版（3 个噪声水平，每个 20 次试验）

**输出格式**：
```json
{
  "system_info": {
    "platform": "darwin",
    "python_version": "3.11.7",
    "timestamp": "2025-11-20T12:34:56",
    "hostname": "my-computer"
  },
  "test_params": {
    "N_v": 50,
    "T": 30,
    "N_h": 200,
    "eta": 0.001,
    "num_epochs": 200
  },
  "training_time": 28.30,
  "replay_time": 0.0012,
  "replay_accuracy": 1.0,
  "robustness_test_time": 68.70,
  "robustness_scores": [1.0, 0.95, 0.80],
  "total_time": 97.00
}
```

---

## ⚠️ 注意事项

### 1. 环境一致性

**重要**: 确保基线测试和优化测试在**相同环境**下运行：

- ✅ 相同的硬件（CPU、内存）
- ✅ 相同的操作系统版本
- ✅ 相同的 Python 版本
- ✅ 相同的依赖版本

**如果环境不同**：
- 对比结果可能不准确
- 建议在相同设备上运行两次测试
- 或明确标注环境差异

### 2. 基线文件位置

基线文件应该保存在 `test_reports/` 目录下，文件名建议：

```
baseline_before_optimize.json              # 默认名称（通用，适用于所有优化阶段）
baseline_before_optimize_mac.json          # Mac 环境
baseline_before_optimize_linux.json         # Linux 环境
baseline_before_optimize_windows.json       # Windows 环境
```

### 3. Git 状态

**建议**: 在切换分支前提交或暂存当前更改

```bash
# 检查工作区状态
git status

# 如果有未提交的更改
git stash  # 暂存更改
# 或
git commit -m "WIP: 临时提交"
```

---

## 🐛 故障排除

### 问题 1: 找不到 baseline_test.py

**错误**: `scripts/baseline_test.py 不存在`

**解决**:
```bash
# 确保在项目根目录
pwd

# 检查文件是否存在
ls scripts/baseline_test.py

# 如果不存在，从优化分支复制
git checkout optimize/blas
git checkout baseline-before-optimization -- scripts/baseline_test.py
```

### 问题 2: 导入错误

**错误**: `No module named 'src'`

**解决**:
```bash
# 确保在项目根目录
cd /path/to/seq_attractor

# 设置 Python 路径
export PYTHONPATH=$PWD  # Linux/Mac
# 或
set PYTHONPATH=%CD%      # Windows CMD

# 重新运行
python scripts/baseline_test.py
```

### 问题 3: 依赖缺失

**错误**: `No module named 'numpy'`

**解决**:
```bash
# 安装依赖
pip install -r requirements.txt

# 或使用虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

---

## 📊 完整工作流示例

```bash
# === 阶段 1: 基线测试 ===

# 1. 切换到基线
git checkout baseline-before-optimization

# 2. 运行基线测试
python scripts/baseline_test.py

# 3. 保存基线文件
cp test_reports/baseline_before_optimize_*.json \
   test_reports/baseline_before_optimize.json

# 4. 验证基线文件
cat test_reports/baseline_before_optimize.json | python -m json.tool

# === 阶段 2: 优化版本测试 ===

# 5. 切换回优化分支
git checkout optimize/blas

# 6. 运行对比测试
python scripts/auto_test.py --compare \
    --baseline-file test_reports/baseline_before_optimize.json

# 7. 查看对比结果
cat test_reports/test_report_*.md | grep -A 30 "基线对比"
```

---

## 🎯 快速命令参考

```bash
# === 创建基线 ===
git checkout baseline-before-optimization
python scripts/baseline_test.py
cp test_reports/baseline_before_optimize_*.json test_reports/baseline_before_optimize.json

# === 对比性能 ===
git checkout optimize/blas
python scripts/auto_test.py --compare --baseline-file test_reports/baseline_before_optimize.json

# === 查看结果 ===
cat test_reports/test_report_*.md | grep -A 30 "基线对比"
```

---

## 📚 相关文档

- [性能对比指南](PERFORMANCE_COMPARISON_GUIDE.md)
- [自动化测试使用指南](AUTO_TEST_USAGE.md)
- [优化完整总结](optimization/OPTIMIZATION_COMPLETE_SUMMARY.md)

---

**提示**: 如果基线文件已经存在，可以直接使用，无需重新运行基线测试。

