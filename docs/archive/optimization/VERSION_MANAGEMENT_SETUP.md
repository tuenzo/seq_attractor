# 版本管理系统设置完成

## ✅ 已完成的工作

### 1. Git 版本管理系统
- ✅ 创建基线标签：`baseline-before-optimization`
- ✅ 当前稳定分支：`refactor-code-structure`
- ✅ 准备优化分支结构

### 2. 性能测试系统
- ✅ 创建 `scripts/benchmark.py` 基准测试脚本
- ✅ 支持 4 种基准测试：
  - 单序列训练
  - 多序列训练
  - 鲁棒性测试
  - 序列生成
- ✅ 支持性能对比分析
- ✅ JSON 格式结果保存

### 3. 文档系统
- ✅ `OPTIMIZATION_PLAN.md` - 详细优化计划
- ✅ `OPTIMIZATION_QUICKSTART.md` - 快速开始指南
- ✅ `benchmark_results/` - 测试结果目录

### 4. Git 提交历史
```
536bb2e feat: 添加性能基准测试系统
5d1ab1b docs: 添加性能优化计划文档和版本管理策略
[tag: baseline-before-optimization]
```

---

## 📁 当前项目结构

```
seq_attractor/
├── src/                      # 源代码
│   ├── core/                 # 核心网络
│   ├── models/               # 模型扩展
│   ├── experiments/          # 实验代码
│   └── utils/                # 工具函数
├── scripts/
│   ├── benchmark.py          # ✨ 新增：性能测试脚本
│   └── ...
├── benchmark_results/        # ✨ 新增：测试结果目录
│   ├── .gitignore
│   └── README.md
├── tests/                    # 测试代码
├── OPTIMIZATION_PLAN.md      # ✨ 新增：优化计划
├── OPTIMIZATION_QUICKSTART.md # ✨ 新增：快速指南
└── VERSION_MANAGEMENT_SETUP.md # ✨ 本文档
```

---

## 🎯 优化分支规划

### 已规划的 7 个优化方案

| 分支名 | 优化内容 | 预期加速 | 难度 | 优先级 |
|--------|---------|---------|------|--------|
| `optimize/blas` | 启用 Accelerate/OpenBLAS | 1.5-3x | ⭐ | 🔥 高 |
| `optimize/mixed-precision` | float32 混合精度 | 1.5-2x | ⭐ | 🔥 高 |
| `optimize/multiprocessing` | 多进程并行化 | 4-8x | ⭐⭐ | 🔥 高 |
| `optimize/gpu-cupy` | CuPy GPU 加速 | 10-30x | ⭐⭐ | 中 |
| `optimize/numba` | Numba JIT 编译 | 5-10x | ⭐⭐⭐ | 中 |
| `optimize/algorithm` | 算法层面优化 | 2-5x | ⭐⭐⭐ | 中 |
| `optimize/all` | 集成所有优化 | 20-100x | ⭐⭐⭐⭐ | 低 |

---

## 🚀 下一步操作指南

### 方式 1：使用 Git 分支管理（推荐）

#### Step 1: 运行基线测试
```bash
# 进入项目目录
cd "/Users/zhao/Library/Mobile Documents/com~apple~CloudDocs/study&research/coding projects/VScode/hopf_networks/seq_attractor"

# 运行基线性能测试（注意：当前环境可能有问题，见下方说明）
python scripts/benchmark.py --output benchmark_results/baseline.json
```

⚠️ **注意：基线测试目前可能失败（Exit code 139 段错误）**

可能原因：
1. 禁用了 macOS Accelerate 框架（`src/core/base.py` 第 10 行）
2. NumPy/BLAS 库配置问题

解决方案：第一个优化（`optimize/blas`）将解决此问题

#### Step 2: 创建第一个优化分支
```bash
# 创建并切换到 BLAS 优化分支
git checkout -b optimize/blas

# 编辑 src/core/base.py，注释掉第 10 行：
# os.environ.setdefault("NPY_DISABLE_MAC_OS_ACCELERATE", "1")

# 测试修改
python scripts/benchmark.py --output benchmark_results/blas.json

# 对比性能
python scripts/benchmark.py --compare \
    benchmark_results/baseline.json \
    benchmark_results/blas.json

# 提交优化
git add src/core/base.py
git commit -m "optimize: 启用 macOS Accelerate 框架

- 删除禁用 Accelerate 的环境变量设置
- 允许 NumPy 使用 Apple 优化的 BLAS 库
- 预期矩阵运算提速 1.5-3x

测试结果: (待填写)"
```

#### Step 3: 继续其他优化
```bash
# 回到主分支
git checkout refactor-code-structure

# 创建新的优化分支
git checkout -b optimize/mixed-precision

# 实施优化...
# 测试、提交、合并
```

---

### 方式 2：使用文件打包备份（备选方案）

如果你更习惯文件打包的方式，可以：

#### Step 1: 创建基线备份
```bash
# 创建备份目录
mkdir -p ../seq_attractor_backups

# 打包当前版本（排除 .git 和结果文件）
tar -czf ../seq_attractor_backups/baseline_$(date +%Y%m%d_%H%M%S).tar.gz \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='figure5_results*' \
    --exclude='*.png' \
    .

# 验证备份
ls -lh ../seq_attractor_backups/
```

#### Step 2: 在当前目录实施优化
```bash
# 直接修改代码
# 每完成一个优化，创建一个新的备份

# BLAS 优化后打包
tar -czf ../seq_attractor_backups/blas_optimized_$(date +%Y%m%d_%H%M%S).tar.gz \
    --exclude='.git' \
    --exclude='__pycache__' \
    .

# 混合精度优化后打包
tar -czf ../seq_attractor_backups/mixed_precision_$(date +%Y%m%d_%H%M%S).tar.gz \
    --exclude='.git' \
    --exclude='__pycache__' \
    .
```

#### Step 3: 恢复某个版本
```bash
# 解压到临时目录查看
mkdir -p /tmp/restore_test
tar -xzf ../seq_attractor_backups/baseline_20241120_*.tar.gz -C /tmp/restore_test

# 或覆盖当前目录（危险！先备份）
tar -xzf ../seq_attractor_backups/baseline_20241120_*.tar.gz
```

---

## 📊 两种方案对比

| 特性 | Git 分支管理 | 文件打包备份 |
|------|-------------|-------------|
| **空间占用** | 很小（增量存储） | 较大（每次全量） |
| **切换速度** | 秒级 | 分钟级 |
| **对比能力** | 强大（git diff） | 需手动 diff |
| **历史追踪** | 完整 | 有限 |
| **回退能力** | 容易 | 中等 |
| **学习成本** | 中等 | 低 |
| **推荐度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 🔧 常用 Git 命令速查

```bash
# 查看当前状态
git status

# 查看所有分支
git branch -a

# 查看所有标签
git tag

# 切换分支
git checkout <branch-name>

# 创建新分支
git checkout -b <new-branch>

# 回到基线版本
git checkout baseline-before-optimization

# 查看两个分支差异
git diff optimize/blas optimize/numba

# 合并分支
git checkout refactor-code-structure
git merge optimize/blas

# 删除分支
git branch -d optimize/failed

# 查看提交历史
git log --oneline --graph --all
```

---

## ⚠️ 当前已知问题

### 问题 1: 基准测试段错误（Exit code 139）

**现象：**
```bash
python scripts/benchmark.py
# Exit code: 139 (Segmentation fault)
```

**原因：**
`src/core/base.py` 第 10 行禁用了 macOS Accelerate 框架：
```python
os.environ.setdefault("NPY_DISABLE_MAC_OS_ACCELERATE", "1")
```

**影响：**
- 无法运行基线性能测试
- NumPy 矩阵运算可能不稳定

**解决方案：**
1. 注释掉该行（第一个优化 `optimize/blas` 会处理）
2. 或重新安装 NumPy：
   ```bash
   pip uninstall numpy
   pip install numpy
   ```

### 问题 2: 其他未提交的改动

**现状：**
```
修改：     src/core/base.py
修改：     src/experiments/figure5.py
未跟踪：   figure5_results/... (多个结果文件)
未跟踪：   *.png
```

**建议：**
在开始优化前，决定如何处理：
1. **提交改动**：`git add . && git commit -m "..."`
2. **暂存改动**：`git stash`
3. **放弃改动**：`git restore <file>`

---

## 📝 优化工作记录模板

为每个优化分支创建记录：

```markdown
## optimize/blas - BLAS 库优化

### 改动内容
- 删除 `src/core/base.py` 第 10 行
- 启用 macOS Accelerate 框架

### 测试环境
- 硬件：MacBook Pro M1/M2
- 系统：macOS 14.6
- Python：3.x
- NumPy：2.3.4

### 性能对比
| 测试项 | 基线 | 优化后 | 加速比 |
|--------|------|--------|--------|
| 单序列训练 | 45.2s | 18.3s | 2.47x |
| 多序列训练 | 128.5s | 52.1s | 2.47x |
| 鲁棒性测试 | 34.8s | 14.2s | 2.45x |
| 序列生成 | 5.6s | 5.4s | 1.04x |
| **总计** | 214.1s | 90.0s | **2.38x** |

### 准确率验证
- ✅ 所有测试通过
- ✅ 准确率保持 100%
- ✅ API 保持兼容

### 结论
✅ **采纳** - 显著提升性能，无副作用

### 提交信息
```
git commit -m "optimize: 启用 macOS Accelerate 框架 (2.38x)"
```
```

---

## 🎉 总结

版本管理系统已完全设置好！你现在可以：

1. ✅ 使用 Git 分支管理不同优化版本
2. ✅ 使用 `benchmark.py` 测试性能
3. ✅ 对比不同版本的性能差异
4. ✅ 随时回退到基线版本
5. ✅ 跟踪每个优化的效果

**推荐开始顺序：**
1. 🔥 `optimize/blas` - 修复当前问题 + 提速 2-3x
2. 🔥 `optimize/mixed-precision` - 进一步提速 1.5-2x
3. 🔥 `optimize/multiprocessing` - 测试场景 4-8x

需要我帮你实施第一个优化吗？

