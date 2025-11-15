"""
运行 Figure 5 多序列学习实验：
- 支持一次性学习多个序列
- 支持在序列之间设置共享模式（重复模式）
- 默认情况下使用2个序列，如果启用共享模式，默认在中间位置有一个模式重复
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.experiments.figure5 import Figure5Config, run_figure5_experiments_split_modes  # noqa: E402


def main() -> None:
    # 示例1: 学习一个序列，使用默认参数
    cfg1 = Figure5Config(
        num_trials=10,
        noise_num=10,
        num_epochs=100,
        T_values=(10, 30, 50),
        N_h_values=(100, 325, 550),
        num_sequences=1,
        with_shared_patterns=False,
    )
    run_figure5_experiments_split_modes(
        config=cfg1,
        output_dir="figure5_results",
        create_timestamp_dir=True,
        show_images=False,
        use_progress=True,
        workers=1,
    )


    # 示例2：默认多序列学习（2个序列，无共享模式）
    print("=" * 80)
    print("示例2：默认多序列学习（2个序列，无共享模式）")
    print("=" * 80)
    cfg2 = Figure5Config(
        num_trials=10,  # 减少试验次数以便快速测试
        noise_num=10,
        num_epochs=100,  # 减少训练轮数以便快速测试
        T_values=(10, 30, 50),
        N_h_values=(100, 325, 550),
        num_sequences=None,  # None表示默认2个序列
        with_shared_patterns=False,
    )
    run_figure5_experiments_split_modes(
        config=cfg2,
        output_dir="figure5_results",
        create_timestamp_dir=True,
        show_images=False,
        use_progress=True,
        workers=1,
    )
    
    # 示例3：多序列学习，启用共享模式（默认中间位置重复）
    print("\n" + "=" * 80)
    print("示例3：多序列学习，启用共享模式（默认中间位置重复）")
    print("=" * 80)
    cfg3 = Figure5Config(
        num_trials=10,
        noise_num=10,
        num_epochs=100,
        T_values=(10, 30, 50),
        N_h_values=(100, 325, 550),
        num_sequences=2,  # 2个序列
        with_shared_patterns=True,  # 启用共享模式
        shared_pattern_positions=None,  # None表示默认中间位置
    )
    run_figure5_experiments_split_modes(
        config=cfg3,
        output_dir="figure5_results",
        create_timestamp_dir=True,
        show_images=False,
        use_progress=True,
        workers=1,
    )
    
    # 示例4：多序列学习，自定义共享模式位置
    print("\n" + "=" * 80)
    print("示例4：多序列学习，自定义共享模式位置")
    print("=" * 80)
    # 自定义共享模式位置：序列0和序列1都在位置5-6有共享模式
    custom_positions = [
        [(5, 6)],  # 序列0的共享模式位置
        [(5, 6)],  # 序列1的共享模式位置
    ]
    cfg4 = Figure5Config(
        num_trials=10,
        noise_num=10,
        num_epochs=100,
        T_values=(10, 30, 50),
        N_h_values=(100, 325, 550),
        num_sequences=2,
        with_shared_patterns=True,
        shared_pattern_positions=custom_positions,
    )

    run_figure5_experiments_split_modes(
        config=cfg4,
        output_dir="figure5_results",
        create_timestamp_dir=True,
        show_images=False,
        use_progress=True,
        workers=1,
    )
    
    # 示例5：学习3个序列，前两个序列有共享模式
    print("\n" + "=" * 80)
    print("示例5：学习3个序列，前两个序列有共享模式")
    print("=" * 80)
    cfg5 = Figure5Config(
        num_trials=10,
        noise_num=10,
        num_epochs=100,
        T_values=(10, 30, 50),
        N_h_values=(100, 325, 550),
        num_sequences=3,
        with_shared_patterns=True,
        shared_pattern_positions=None,  # 默认：前两个序列中间位置有共享模式
    )
    run_figure5_experiments_split_modes(
        config=cfg5,
        output_dir="figure5_results",
        create_timestamp_dir=True,
        show_images=False,
        use_progress=True,
        workers=1,
    )


if __name__ == "__main__":
    main()

