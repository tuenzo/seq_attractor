"""
运行 Figure 5 拆分模式对比：
- 图(a)：固定 N_h=500，扫描 T，对比“仅训练 V”与“训练 U+V”
- 图(b)：固定 T=70，扫描 N_h，对比“仅训练 V”与“训练 U+V”
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.experiments.figure5 import Figure5Config, run_figure5_experiments_split_modes  # noqa: E402


def main() -> None:
    cfg = Figure5Config(
        num_trials=100,
        noise_num=10,
        num_epochs=500,
        T_values=np.linspace(10, 140, 14, dtype=int),
        N_h_values=np.linspace(100, 1000, 10, dtype=int),
    )
    run_figure5_experiments_split_modes(
        config=cfg,
        output_dir=None,
        create_timestamp_dir=True,
        show_images=True,
        use_progress=True,
        workers=-1,
    )


if __name__ == "__main__":
    main()


