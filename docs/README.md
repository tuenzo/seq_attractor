# 项目文档目录

本目录包含项目的所有文档，按主题分类。

## 📁 目录结构

```
docs/
├── README.md                    # 本文件
├── optimization/                # 性能优化相关文档
│   ├── OPTIMIZATION_COMPLETE_SUMMARY.md  # 优化完整总结 ⭐
│   ├── QUICK_SUMMARY.md                  # 快速开始指南
│   ├── OPTIMIZATION_STATUS.md            # 优化进度跟踪
│   ├── OPTIMIZATION_PLAN.md              # 优化计划
│   ├── OPTIMIZATION_QUICKSTART.md        # 快速操作指南
│   ├── PERFORMANCE_COMPARISON_REPORT.md  # 性能测试报告
│   ├── TROUBLESHOOTING.md                # 故障排除指南
│   ├── VERSION_MANAGEMENT_SETUP.md       # 版本管理说明
│   └── compare_blas_performance.py       # 性能对比脚本
├── refactoring/                 # 代码重构相关文档
│   ├── REFACTORING_SUMMARY.md           # 重构总结
│   ├── REFACTORING_README.md            # 重构说明
│   └── REFACTORING_CHANGES.md           # 重构变更记录
└── results/                     # 实验结果和图片
    ├── pattern_shared_demo.png
    └── pattern_shared_overview.png
```

---

## 🚀 快速开始

### 新用户？从这里开始
1. **[QUICK_SUMMARY.md](optimization/QUICK_SUMMARY.md)** - 3分钟了解优化
2. **[OPTIMIZATION_COMPLETE_SUMMARY.md](optimization/OPTIMIZATION_COMPLETE_SUMMARY.md)** - 完整总结

### 需要优化？
1. **[OPTIMIZATION_PLAN.md](optimization/OPTIMIZATION_PLAN.md)** - 查看优化路线图
2. **[OPTIMIZATION_QUICKSTART.md](optimization/OPTIMIZATION_QUICKSTART.md)** - 快速操作指南

### 遇到问题？
1. **[TROUBLESHOOTING.md](optimization/TROUBLESHOOTING.md)** - 故障排除（7种方案）

### 查看性能？
1. **[PERFORMANCE_COMPARISON_REPORT.md](optimization/PERFORMANCE_COMPARISON_REPORT.md)** - 详细测试报告

---

## 📖 文档分类

### 优化文档 (optimization/)

#### 核心文档
- **OPTIMIZATION_COMPLETE_SUMMARY.md** ⭐⭐⭐⭐⭐
  - 第一个优化的完整总结
  - 包含测试结果、核心价值、使用指南
  - 推荐首先阅读

- **QUICK_SUMMARY.md** ⭐⭐⭐⭐⭐
  - 5分钟快速了解
  - 完成工作、当前问题、下一步操作

- **PERFORMANCE_COMPARISON_REPORT.md** ⭐⭐⭐⭐
  - 详细的性能测试报告
  - 对比数据、分析说明
  - 跨平台性能预期

#### 操作指南
- **OPTIMIZATION_QUICKSTART.md**
  - 快速操作步骤
  - Git 命令速查
  - 最佳实践

- **TROUBLESHOOTING.md**
  - NumPy 段错误解决（7种方案）
  - macOS/Linux/GPU 环境配置
  - 诊断工具和命令

- **VERSION_MANAGEMENT_SETUP.md**
  - Git 分支管理详解
  - 文件备份方案
  - 版本对比方法

#### 规划文档
- **OPTIMIZATION_PLAN.md**
  - 7个优化方案详细规划
  - 预期加速比
  - 实施优先级

- **OPTIMIZATION_STATUS.md**
  - 优化进度跟踪
  - 已完成/待实施列表
  - 测试清单

#### 工具脚本
- **compare_blas_performance.py**
  - 性能对比测试脚本
  - 自动化测试流程
  - 结果分析和展示

---

### 重构文档 (refactoring/)

- **REFACTORING_SUMMARY.md**
  - 代码重构总结
  - 新架构说明
  - API 变更

- **REFACTORING_README.md**
  - 重构概述
  - 使用指南

- **REFACTORING_CHANGES.md**
  - 详细变更记录
  - 迁移指南

---

### 结果文件 (results/)

存储实验结果图片和数据文件。

**注意**：大型结果文件（如 figure5_results/）在项目根目录，已添加到 .gitignore。

---

## 🔍 如何查找文档

### 按需求查找

| 需求 | 推荐文档 |
|------|---------|
| 了解优化成果 | OPTIMIZATION_COMPLETE_SUMMARY.md |
| 快速上手 | QUICK_SUMMARY.md |
| 解决环境问题 | TROUBLESHOOTING.md |
| 查看性能数据 | PERFORMANCE_COMPARISON_REPORT.md |
| 学习 Git 管理 | VERSION_MANAGEMENT_SETUP.md |
| 规划下一步 | OPTIMIZATION_PLAN.md |
| 跟踪进度 | OPTIMIZATION_STATUS.md |
| 了解重构 | REFACTORING_SUMMARY.md |

### 按角色查找

**研究人员/用户**
1. QUICK_SUMMARY.md - 快速了解
2. OPTIMIZATION_PLAN.md - 查看路线图
3. PERFORMANCE_COMPARISON_REPORT.md - 性能数据

**开发者/贡献者**
1. REFACTORING_SUMMARY.md - 了解架构
2. VERSION_MANAGEMENT_SETUP.md - Git 工作流
3. TROUBLESHOOTING.md - 环境配置

**系统管理员**
1. TROUBLESHOOTING.md - 环境部署
2. OPTIMIZATION_STATUS.md - 系统要求
3. PERFORMANCE_COMPARISON_REPORT.md - 性能基准

---

## 📝 文档维护

### 文档更新规则
1. 每个优化完成后更新 OPTIMIZATION_STATUS.md
2. 性能测试后更新 PERFORMANCE_COMPARISON_REPORT.md
3. 新问题添加到 TROUBLESHOOTING.md
4. Git 工作流变化更新 VERSION_MANAGEMENT_SETUP.md

### 文档格式
- 使用 Markdown 格式
- 包含清晰的标题结构
- 添加表格和代码示例
- 使用 emoji 增强可读性

---

## 🌟 重要提示

1. **OPTIMIZATION_COMPLETE_SUMMARY.md** 是最全面的总结文档
2. **QUICK_SUMMARY.md** 适合快速查看
3. 遇到问题先查 **TROUBLESHOOTING.md**
4. 所有测试脚本在 `../scripts/` 目录

---

## 📞 需要帮助？

查看文档后仍有疑问？

1. 检查 TROUBLESHOOTING.md
2. 运行 `python scripts/check_system.py` 诊断环境
3. 查看 Git 提交历史了解变更

