"""
运行 Figure 5 多序列学习实验：
- 支持一次性学习多个序列
- 支持在序列之间设置共享模式（重复模式）
- 默认情况下使用2个序列，如果启用共享模式，默认在中间位置有一个模式重复
"""

import os
import sys
import numpy as np
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.experiments.figure5 import Figure5Config, run_figure5_experiments_split_modes  # noqa: E402

def main() -> None:
    start_time = time.time()
    print("\n" + "=" * 80)
    print("实验内容：学习n个序列，前两个序列有共享模式")
    print("=" * 80)
    custom_positions = [
        [(4, 4)],  # 序列0的共享模式位置 (start, end)
        [(4, 4)],  # 序列1的共享模式位置 (start, end)
    ]
    cfg5 = Figure5Config(
        num_trials=100,
        noise_num=10,
        num_epochs=500,
        T_values=np.linspace(5, 55, 6, dtype=int),
        N_h_values=np.linspace(100, 1000, 5, dtype=int),
        num_sequences=5,
    )
    results = run_figure5_experiments_split_modes(
        config=cfg5,
        base_params_a={"N_v": 100, "N_h": 500, "eta": 0.001, "kappa": 1.0},
        base_params_b={"N_v": 100, "T": 25, "eta": 0.001, "kappa": 1.0},
        output_dir="figure5_results_local_exapmles",
        create_timestamp_dir=True,
        show_images=False,
        use_progress=True,
        workers=-1, 
        with_shared_patterns=False,
        shared_pattern_positions=None,
    )
    time_cost = time.time() - start_time
    print("实验完成，实验耗时：%.2f 秒" % time_cost)
    print("实验结果保存在 figure5_results 目录下，文件名以 timestamp 结尾")


if __name__ == "__main__":
    main()