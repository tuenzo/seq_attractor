# 📚 项目文档导航

欢迎！这里是序列吸引子网络项目的完整文档中心。

---

## 🚀 快速开始

### 新用户？从这里开始

1. **[快速开始指南](QUICK_START.md)** ⭐⭐⭐⭐⭐
   - 一键测试和部署
   - 所有平台（Linux、macOS、Windows）
   - 常见问题解决方案
   - **推荐首先阅读**

2. **[基线测试指南](BASELINE_TESTING.md)**
   - 如何在基线版本运行测试
   - 创建性能基线
   - 用于后续优化对比

---

## 📊 性能测试与对比

### 自动化测试

1. **[自动化测试使用指南](AUTO_TEST_USAGE.md)** ⭐⭐⭐⭐
   - 完整测试流程
   - 基线保存和性能对比
   - 详细使用示例

2. **[性能对比指南](PERFORMANCE_COMPARISON_GUIDE.md)** ⭐⭐⭐⭐
   - 如何对比优化前后性能
   - 使用基线标签
   - 跨环境对比

3. **[基线测试指南](BASELINE_TESTING.md)**
   - 基线版本测试方法
   - 生成基线数据

---

## 🔧 故障排除

1. **[故障排除指南](optimization/TROUBLESHOOTING.md)** ⭐⭐⭐⭐⭐
   - 常见问题解决方案
   - 环境配置问题
   - 诊断工具和命令
   - **遇到问题先看这里**

---

## ⚡ 性能优化

### 优化文档

1. **[优化完整总结](optimization/OPTIMIZATION_COMPLETE_SUMMARY.md)** ⭐⭐⭐⭐⭐
   - 第一个优化（BLAS）的完整总结
   - 性能测试结果
   - 核心价值和使用指南
   - **推荐阅读**

2. **[优化计划](optimization/OPTIMIZATION_PLAN.md)** ⭐⭐⭐⭐
   - 7 个优化阶段详细规划
   - 预期加速比
   - 实施优先级

---

## 📁 文档结构

```
docs/
├── README.md                          # 本文件 - 文档导航
├── QUICK_START.md                     # ⭐ 快速开始（整合版）
├── BASELINE_TESTING.md                # 基线测试指南
├── AUTO_TEST_USAGE.md                 # 自动化测试使用指南
├── PERFORMANCE_COMPARISON_GUIDE.md    # 性能对比指南
│
├── optimization/                      # 性能优化文档
│   ├── OPTIMIZATION_COMPLETE_SUMMARY.md  # ⭐ 优化完整总结
│   ├── OPTIMIZATION_PLAN.md              # 优化计划
│   └── TROUBLESHOOTING.md                # ⭐ 故障排除指南
│
├── archive/                           # 归档文档
│   ├── optimization/                 # 已归档的优化文档
│   └── refactoring/                  # 已归档的重构文档
│
└── results/                           # 实验结果
    └── *.png                          # 实验图片
```

---

## 🎯 按需求查找文档

| 需求 | 推荐文档 | 优先级 |
|------|---------|--------|
| **快速上手** | [QUICK_START.md](QUICK_START.md) | ⭐⭐⭐⭐⭐ |
| **遇到问题** | [TROUBLESHOOTING.md](optimization/TROUBLESHOOTING.md) | ⭐⭐⭐⭐⭐ |
| **了解优化成果** | [OPTIMIZATION_COMPLETE_SUMMARY.md](optimization/OPTIMIZATION_COMPLETE_SUMMARY.md) | ⭐⭐⭐⭐⭐ |
| **运行测试** | [AUTO_TEST_USAGE.md](AUTO_TEST_USAGE.md) | ⭐⭐⭐⭐ |
| **对比性能** | [PERFORMANCE_COMPARISON_GUIDE.md](PERFORMANCE_COMPARISON_GUIDE.md) | ⭐⭐⭐⭐ |
| **查看优化计划** | [OPTIMIZATION_PLAN.md](optimization/OPTIMIZATION_PLAN.md) | ⭐⭐⭐⭐ |
| **基线测试** | [BASELINE_TESTING.md](BASELINE_TESTING.md) | ⭐⭐⭐ |

---

## 🔍 按角色查找

### 新用户 / 研究人员

1. [QUICK_START.md](QUICK_START.md) - 快速开始
2. [OPTIMIZATION_COMPLETE_SUMMARY.md](optimization/OPTIMIZATION_COMPLETE_SUMMARY.md) - 了解优化成果
3. [OPTIMIZATION_PLAN.md](optimization/OPTIMIZATION_PLAN.md) - 查看优化路线图

### 开发者 / 贡献者

1. [AUTO_TEST_USAGE.md](AUTO_TEST_USAGE.md) - 自动化测试
2. [PERFORMANCE_COMPARISON_GUIDE.md](PERFORMANCE_COMPARISON_GUIDE.md) - 性能对比
3. [BASELINE_TESTING.md](BASELINE_TESTING.md) - 基线测试

### 系统管理员

1. [QUICK_START.md](QUICK_START.md) - 部署指南
2. [TROUBLESHOOTING.md](optimization/TROUBLESHOOTING.md) - 环境配置
3. [AUTO_TEST_USAGE.md](AUTO_TEST_USAGE.md) - 测试流程

---

## 📝 文档更新说明

### 活跃文档（当前维护）

- ✅ `QUICK_START.md` - 整合了所有快速开始、部署、Windows 设置内容
- ✅ `AUTO_TEST_USAGE.md` - 自动化测试完整指南
- ✅ `PERFORMANCE_COMPARISON_GUIDE.md` - 性能对比指南
- ✅ `BASELINE_TESTING.md` - 基线测试指南
- ✅ `optimization/OPTIMIZATION_COMPLETE_SUMMARY.md` - 优化总结
- ✅ `optimization/OPTIMIZATION_PLAN.md` - 优化计划
- ✅ `optimization/TROUBLESHOOTING.md` - 故障排除

### 归档文档（历史参考）

已归档到 `docs/archive/` 目录的文档：
- `DEPLOYMENT_TESTING_GUIDE.md` → 内容已整合到 `QUICK_START.md`
- `QUICK_TEST_GUIDE.md` → 内容已整合到 `QUICK_START.md`
- `WINDOWS_SETUP.md` → 内容已整合到 `QUICK_START.md`
- `READY_FOR_DEPLOYMENT.md` → 临时文档，已归档
- `optimization/QUICK_SUMMARY.md` → 内容已整合到 `OPTIMIZATION_COMPLETE_SUMMARY.md`
- `optimization/OPTIMIZATION_QUICKSTART.md` → 内容已整合到 `QUICK_START.md`
- `optimization/OPTIMIZATION_STATUS.md` → 状态文档，已归档
- `optimization/VERSION_MANAGEMENT_SETUP.md` → 内容已整合到其他文档
- `optimization/PERFORMANCE_COMPARISON_REPORT.md` → 历史报告，已归档
- `refactoring/*` → 重构文档，已归档

---

## 🆘 需要帮助？

### 1. 查看文档

- 遇到问题 → [TROUBLESHOOTING.md](optimization/TROUBLESHOOTING.md)
- 快速开始 → [QUICK_START.md](QUICK_START.md)
- 测试问题 → [AUTO_TEST_USAGE.md](AUTO_TEST_USAGE.md)

### 2. 运行诊断

```bash
# 系统环境检查
python scripts/check_system.py

# Windows 环境检查
python check_windows.py
```

### 3. 提交 Issue

- GitHub: https://github.com/tuenzo/seq_attractor/issues
- 附上错误信息和系统信息

---

## 📌 重要提示

1. **首次使用**：先阅读 [QUICK_START.md](QUICK_START.md)
2. **遇到问题**：先查看 [TROUBLESHOOTING.md](optimization/TROUBLESHOOTING.md)
3. **运行测试**：使用 `python scripts/auto_test.py`
4. **对比性能**：使用 `--compare` 选项

---

**最后更新**: 2025-11-20  
**维护状态**: 活跃维护中 ✅
