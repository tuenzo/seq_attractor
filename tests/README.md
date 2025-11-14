# 测试套件说明

本目录包含序列吸引子网络的完整测试套件。

## 测试结构

```
tests/
├── __init__.py              # 测试包初始化
├── conftest.py              # pytest配置和共享fixtures
├── test_base.py             # 基础网络测试
├── test_multi_sequence.py   # 多序列模型测试
├── test_incremental.py      # 增量学习模型测试
├── test_pattern_repetition.py # 模式重复模型测试
└── test_utils.py            # 工具函数测试
```

## 运行测试

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行所有测试

```bash
pytest
```

### 运行特定测试文件

```bash
# 运行基础网络测试
pytest tests/test_base.py

# 运行多序列模型测试
pytest tests/test_multi_sequence.py

# 运行增量学习测试
pytest tests/test_incremental.py

# 运行模式重复测试
pytest tests/test_pattern_repetition.py

# 运行工具函数测试
pytest tests/test_utils.py
```

### 运行特定测试类或函数

```bash
# 运行特定测试类
pytest tests/test_base.py::TestSequenceAttractorNetwork

# 运行特定测试函数
pytest tests/test_base.py::TestSequenceAttractorNetwork::test_initialization
```

### 带覆盖率的测试

```bash
# 运行测试并生成覆盖率报告
pytest --cov=src --cov-report=html

# 查看覆盖率报告
open htmlcov/index.html  # macOS
# 或
xdg-open htmlcov/index.html  # Linux
```

### 详细输出

```bash
# 显示详细输出
pytest -v

# 显示最详细的输出
pytest -vv

# 显示打印语句输出
pytest -s
```

## 测试覆盖范围

### 基础网络 (`test_base.py`)
- ✅ 网络初始化
- ✅ 随机序列生成
- ✅ 训练功能
- ✅ 序列回放
- ✅ 回放质量评估
- ✅ 不同参数组合

### 多序列模型 (`test_multi_sequence.py`)
- ✅ 多序列网络初始化
- ✅ 多序列生成
- ✅ 交替训练和批量训练
- ✅ 序列回放
- ✅ 序列信息查询

### 增量学习模型 (`test_incremental.py`)
- ✅ 增量学习初始化
- ✅ 增量学习功能
- ✅ 记忆保持
- ✅ 记忆状态查询
- ✅ 记忆测试功能

### 模式重复模型 (`test_pattern_repetition.py`)
- ✅ 模式生成（交替、周期、块状、镜像等）
- ✅ 模式结构分析
- ✅ 序列重叠分析
- ✅ 模式序列训练

### 工具函数 (`test_utils.py`)
- ✅ 完整序列匹配评估
- ✅ 逐帧匹配评估
- ✅ 各种边界情况

## 编写新测试

### 添加新测试函数

在相应的测试文件中添加新的测试函数：

```python
def test_new_feature(self, basic_network):
    """测试新功能"""
    # 测试代码
    assert condition
```

### 使用共享Fixtures

在 `conftest.py` 中定义的fixtures可以在所有测试文件中使用：

- `basic_network_params`: 基础网络参数
- `basic_network`: 基础网络实例
- `multi_sequence_network`: 多序列网络实例
- `incremental_network`: 增量学习网络实例
- `pattern_network`: 模式重复网络实例
- `sample_sequence`: 示例训练序列
- `multiple_sequences`: 多个示例序列

### 测试最佳实践

1. **测试命名**: 使用描述性的测试函数名，以 `test_` 开头
2. **独立性**: 每个测试应该独立，不依赖其他测试的执行顺序
3. **可重复性**: 使用固定种子确保测试可重复
4. **边界情况**: 测试正常情况、边界情况和错误情况
5. **断言清晰**: 使用清晰的断言消息

## 持续集成

测试可以在CI/CD流程中运行，确保代码质量。

## 故障排除

### 导入错误

如果遇到导入错误，确保：
1. 项目根目录在Python路径中
2. `src` 目录结构正确
3. 所有 `__init__.py` 文件存在

### 测试失败

如果测试失败：
1. 检查错误消息和堆栈跟踪
2. 确认测试环境正确
3. 检查随机种子是否影响结果
4. 某些测试可能因为网络容量限制而不稳定（这是正常的）

