"""
运行 Figure 5 拆分模式对比（重复序列版本）：
- 在训练序列中引入单步重复（默认在中点）
- 图(a)：仅训练 V，固定 N_h=500 扫描 T
- 图(b)：训练 U+V，固定 T=70 扫描 N_h
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.experiments.figure5 import Figure5Config, run_figure5_experiments_split_modes  # noqa: E402


def main() -> None:
    cfg = Figure5Config(
        num_trials=100,
        noise_num=10,
        num_epochs=500,
        T_values=(10, 30, 50, 70, 90, 110, 140),
        N_h_values=(100, 325, 550, 775, 1000),
    )
    run_figure5_experiments_split_modes(
        config=cfg,
        output_dir=None,
        create_timestamp_dir=True,
        show_images=True,
        with_repetition=True,
        repetition_position=None,  # None=默认中点
    )


if __name__ == "__main__":
    main()


