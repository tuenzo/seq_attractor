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
    print("\n" + "=" * 80)
    print("实验内容：学习3个序列，前两个序列有共享模式")
    print("=" * 80)
    custom_positions = [
        [(5, 6)],  # 序列0的共享模式位置
        [(5, 6)],  # 序列1的共享模式位置
    ]
    cfg5 = Figure5Config(
        num_trials=100,
        noise_num=10,
        num_epochs=500,
        T_values=np.linspace(10, 140, 14, dtype=int),
        N_h_values=np.linspace(100, 1000, 10, dtype=int),
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
        workers=-1,
    )

if __name__ == "__main__":
    main()