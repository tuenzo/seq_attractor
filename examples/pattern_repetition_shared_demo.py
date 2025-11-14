"""
共享模式与唯一性约束示例
========================

该脚本演示 `PatternRepetitionNetwork` 的高级功能，包括：

1. 通过直观配置生成跨序列共享的模式片段；
2. 对非共享区域启用全局唯一性约束，避免训练数据泄漏；
3. 输出模式重叠分析与配置报告，并保存可视化示意图。

运行方式::

    python examples/pattern_repetition_shared_demo.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import (  # noqa: E402
    PatternRepetitionNetwork,
    visualize_multi_sequence_overview,
)


@dataclass(frozen=True)
class DemoConfig:
    """示例配置参数。"""

    N_v: int = 64
    T: int = 36
    N_h: int = 256
    eta: float = 0.01
    kappa: float = 1.0
    num_sequences: int = 3
    num_epochs: int = 250
    seeds: Tuple[int, ...] = (2025, 2026, 2027)


def build_custom_pattern_spec(
    sequence_length: int,
) -> Tuple[List[List[int]], List[int], List[List[List[Tuple[int, int]]]]]:
    """
    构造共享模式配置。

    返回:
        shared_groups: 每个元素表示一组需要共享相同模式的序列索引。
        patterns_per_group: 对应组中包含的模式数量。
        positions_per_group: 每个组中每个序列的模式起止位置 (包含终点)。
    """
    if sequence_length < 30:
        raise ValueError("示例配置假定序列长度>=30")

    shared_groups = [
        [0, 1],  # 序列0与1共享一段中心模式
        [1, 2],  # 序列1与2共享两段尾部模式
    ]
    patterns_per_group = [
        1,  # 每组的模式数量
        2,
    ]
    positions_per_group = [
        [
            [(6, 11)],  # 序列0中共享模式的起止位置 (包含终点)
            [(6, 11)],  # 序列1中共享模式的起止位置
        ],
        [
            [(16, 20), (24, 27)],  # 序列1的两段共享模式
            [(16, 20), (24, 27)],  # 序列2的两段共享模式
        ],
    ]
    return shared_groups, patterns_per_group, positions_per_group


def summarize_sequence_statistics(
    network: PatternRepetitionNetwork,
    sequences: Sequence[np.ndarray],
) -> None:
    """打印单个序列的模式结构统计。"""
    print("\n=== 单序列模式统计 ===")
    for idx, sequence in enumerate(sequences):
        analysis = network.analyze_pattern_structure(sequence)
        detected_period = analysis.get("detected_period")
        print(f"- 序列 #{idx}:")
        print(f"    总帧数 (不含循环帧) : {analysis['total_frames']}")
        print(f"    唯一帧数           : {analysis['unique_frames']}")
        print(f"    最大重复次数       : {analysis['max_repetitions']}")
        print(f"    重复率             : {analysis['repetition_rate'] * 100:.1f}%")
        if detected_period:
            print(f"    检测到周期         : {detected_period}")


def main() -> None:
    cfg = DemoConfig()
    np.random.seed(2025)

    network = PatternRepetitionNetwork(
        N_v=cfg.N_v,
        T=cfg.T,
        N_h=cfg.N_h,
        eta=cfg.eta,
        kappa=cfg.kappa,
    )

    shared_groups, patterns_per_group, positions_per_group = build_custom_pattern_spec(cfg.T)

    print("=== 生成包含共享模式的训练序列 ===")
    sequences = network.generate_sequences_with_custom_patterns(
        num_sequences=cfg.num_sequences,
        shared_groups=shared_groups,
        patterns_per_group=patterns_per_group,
        positions_per_group=positions_per_group,
        seeds=list(cfg.seeds),
        T=cfg.T,
        ensure_unique_non_shared=True,
    )

    summarize_sequence_statistics(network, sequences)
    network.print_overlap_analysis(sequences)

    print("=== 模式配置验证报告 ===")
    print(network.get_pattern_overlap_report())

    print("=== 开始训练 ===")
    network.train(
        x=list(sequences),
        num_epochs=cfg.num_epochs,
        verbose=False,
        interleaved=True,
    )

    print("=== 训练回放准确率 ===")
    for idx in range(cfg.num_sequences):
        xi_replayed = network.replay(sequence_index=idx, max_steps=cfg.T * 2)
        evaluation = network.evaluate_replay(xi_replayed, sequence_index=idx)
        status = "✓ 成功" if evaluation.get("found_sequence", False) else "✗ 失败"
        accuracy = evaluation.get("recall_accuracy", 0.0)
        print(f"- 序列 #{idx}: {status}，准确率 = {accuracy * 100:.1f}%")

    print("=== 保存可视化结果 ===")
    network.visualize_pattern_info(
        save_path="pattern_shared_demo.png",
        show_images=False,
    )
    visualize_multi_sequence_overview(
        network,
        save_path="pattern_shared_overview.png",
        show_images=False,
    )
    print("✓ 已生成 pattern_shared_demo.png 与 pattern_shared_overview.png")

    print("\n=== 示例完成 ===")


if __name__ == "__main__":
    main()


