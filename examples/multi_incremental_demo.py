"""
多序列与增量训练综合示例
==================================

该脚本演示统一的 `MemorySequenceAttractorNetwork` 在不同记忆策略下的用法：
1. 生成多个互不重复的训练序列，并进行跨序列重复性检查；
2. 使用多序列模式一次性学习所有序列；
3. 使用增量模式在单次训练中回顾已有记忆；
4. 逐个增量添加序列，展示累积学习效果。

运行方式::

    python examples/multi_incremental_demo.py
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import MemorySequenceAttractorNetwork  # noqa: E402  (路径插入需在导入前)


@dataclass(frozen=True)
class DemoConfig:
    """示例配置参数。"""

    N_v: int = 60
    T: int = 32
    N_h: int = 240
    eta: float = 0.01
    kappa: float = 1.0
    num_sequences: int = 3
    num_epochs_multi: int = 200
    num_epochs_incremental_first: int = 180
    num_epochs_incremental_additional: int = 120
    seeds: Tuple[int, ...] = (7, 19, 37)


def check_no_duplicate_frames(sequences: Sequence[np.ndarray]) -> List[Tuple[int, int, Tuple[int, ...]]]:
    """
    检查所有序列（忽略每序列的最后一帧）是否存在重复帧。

    返回重复帧列表，元素格式为 (序列索引, 帧索引, 帧内容)。
    若列表为空，表示所有帧唯一。
    """
    seen_frames: Dict[Tuple[int, ...], Tuple[int, int]] = {}
    duplicates: List[Tuple[int, int, Tuple[int, ...]]] = []

    for seq_idx, sequence in enumerate(sequences):
        for frame_idx in range(sequence.shape[0] - 1):  # 忽略最后一帧（等于首帧）
            frame_tuple = tuple(sequence[frame_idx].astype(int).tolist())
            if frame_tuple in seen_frames:
                duplicates.append((seq_idx, frame_idx, frame_tuple))
            else:
                seen_frames[frame_tuple] = (seq_idx, frame_idx)

    return duplicates


def summarize_recall_accuracy(network, sequence_count: int, title: str) -> None:
    """打印每个序列的回放准确率。"""
    print(f"\n[{title}] 回放准确率概览:")
    for idx in range(sequence_count):
        xi_replayed = network.replay(sequence_index=idx, max_steps=network.T * 3)
        evaluation = network.evaluate_replay(xi_replayed, sequence_index=idx)
        accuracy = evaluation.get("recall_accuracy", 0.0)
        status = "✓ 成功" if evaluation.get("found_sequence", False) else "✗ 失败"
        print(f"  序列 #{idx}: {status}，准确率 = {accuracy * 100:.1f}%")


def build_network(cfg: DemoConfig) -> MemorySequenceAttractorNetwork:
    """创建统一的记忆网络实例。"""
    return MemorySequenceAttractorNetwork(
        N_v=cfg.N_v,
        T=cfg.T,
        N_h=cfg.N_h,
        eta=cfg.eta,
        kappa=cfg.kappa,
    )


def run_multi_sequence_training(cfg: DemoConfig, sequences: Sequence[np.ndarray]) -> MemorySequenceAttractorNetwork:
    """使用多序列网络一次性训练所有序列。"""
    print("\n=== 多序列联合训练 ===")
    multi_net = build_network(cfg)

    multi_net.train(
        x=list(sequences),
        num_epochs=cfg.num_epochs_multi,
        verbose=False,
        interleaved=True,
    )

    summarize_recall_accuracy(multi_net, len(sequences), "多序列训练")
    return multi_net


def run_incremental_batch_training(cfg: DemoConfig, sequences: Sequence[np.ndarray]) -> MemorySequenceAttractorNetwork:
    """
    使用增量网络在一次训练调用中学习所有序列。

    这里先将序列注册到网络的记忆，再通过一次增量训练迭代所有序列。
    """
    print("\n=== 增量训练：一次性学习所有序列 ===")
    incremental_net = build_network(cfg)

    # 先注册所有序列（num_epochs=0 仅用于记录，不进行权重更新）
    for seq in sequences:
        incremental_net.train(x=seq, num_epochs=0, verbose=False)

    # 一次性训练所有已注册序列
    incremental_net.train(
        x=None,
        num_epochs=cfg.num_epochs_incremental_first,
        verbose=False,
        incremental=True,
    )

    summarize_recall_accuracy(incremental_net, len(sequences), "增量训练（一次性）")
    return incremental_net


def run_incremental_sequential_training(cfg: DemoConfig, sequences: Sequence[np.ndarray]) -> MemorySequenceAttractorNetwork:
    """
    使用增量网络逐个新增序列，并在每次添加后进行训练。
    """
    print("\n=== 增量训练：逐个添加序列 ===")
    incremental_net = build_network(cfg)

    # 先学习第一个序列（基础记忆）
    incremental_net.train(
        x=sequences[0],
        num_epochs=cfg.num_epochs_incremental_first,
        verbose=False,
    )

    # 依次增量加入后续序列
    for seq in sequences[1:]:
        incremental_net.train(
            x=seq,
            num_epochs=cfg.num_epochs_incremental_additional,
            verbose=False,
            incremental=True,
        )

    summarize_recall_accuracy(incremental_net, len(sequences), "增量训练（逐个添加）")
    return incremental_net


def main() -> None:
    cfg = DemoConfig()
    np.random.seed(1234)

    print("=== 序列生成与重复性检查 ===")
    generator = build_network(cfg)

    sequences = generator.generate_multiple_sequences(
        num_sequences=cfg.num_sequences,
        seeds=list(cfg.seeds),
        ensure_unique_across=True,
    )

    duplicates = check_no_duplicate_frames(sequences)
    if duplicates:
        raise RuntimeError(f"检测到重复帧: {duplicates}")

    print(f"生成序列数量: {len(sequences)}，每个序列长度: {cfg.T}")
    print("重复性检查: ✓ 通过（跨序列所有帧均唯一）")

    run_multi_sequence_training(cfg, sequences)
    run_incremental_batch_training(cfg, sequences)
    run_incremental_sequential_training(cfg, sequences)

    print("\n=== 示例流程完成 ===")


if __name__ == "__main__":
    main()

