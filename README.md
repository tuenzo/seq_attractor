复现《Learning Sequence Attractors in Recurrent Networks with Hidden Neurons》中的RNN模型，并提供多序列、增量训练及模式重复等扩展功能。

## 环境依赖

```bash
pip install -r requirements.txt
```

## 可用模型

- `SequenceAttractorNetwork`：基础单序列吸引子网络；
- `MemorySequenceAttractorNetwork`：统一的多序列 + 增量记忆核心模块；
- `MultiSequenceAttractorNetwork` / `IncrementalSequenceAttractorNetwork`：兼容层，继承自统一核心，保留原有接口；
- `PatternRepetitionNetwork`：构建在统一核心之上，支持具有重复模式与共享片段的序列建模。

## 示例脚本

项目在 `examples/` 目录中提供了若干示例，可直接运行体验核心功能：

```bash
# 基础使用与可视化
python examples/basic_example.py

# 多序列 + 增量训练组合流程
python examples/multi_incremental_demo.py

# 模式重复网络的高级用法（共享模式、唯一性约束、报告/可视化导出）
python examples/pattern_repetition_shared_demo.py
```

运行模式重复示例后，输出的图像文件包括：

- `pattern_shared_demo.png`：共享模式在时间维度上的可视化；
- `pattern_shared_overview.png`：训练后各序列的整体回放概览。

## 命令行用法（Figure 5 拆分模式）

提供了简洁的 CLI 入口，复现实验图5的“拆分模式”：
- 图(a)：仅训练 V，固定隐藏层 `M=500`，扫描序列长度 `T`
- 图(b)：训练 U+V，固定 `T=70`，扫描隐藏层规模 `M`

```bash
# 在项目根目录下运行
python scripts/sa_cli.py fig5-split --trials 100 --epochs 500 --show
```

常用参数：
- `--trials`：每个采样点的尝试次数（默认 100）
- `--epochs`：每次训练的轮数（默认 500）
- `--show`：显示绘图窗口
- `--out`：输出目录（默认写入内部目录）
- `--no-timestamp`：不创建带时间戳的子目录
- `--T-values`：扫描的 `T` 列表，逗号分隔，如 `10,30,50,70`
- `--Nh-values`：扫描的 `M(N_h)` 列表，如 `100,325,550,775,1000`
- `--with-repetition`：在训练序列中注入“单步重复”
- `--repeat-pos`：重复发生的位置（默认在中点）

示例：
```bash
# 标准拆分模式（无重复）
python scripts/sa_cli.py fig5-split --trials 100 --epochs 500 --show

# 在序列中注入单步重复（默认在中点）
python scripts/sa_cli.py fig5-split --with-repetition --show

# 自定义扫描范围
python scripts/sa_cli.py fig5-split --T-values 10,30,50,70,110 --Nh-values 100,325,550,775,1000 --show
```

## 测试

```bash
pytest
```
