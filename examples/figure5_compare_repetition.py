"""
图 5 对比实验：无重复 vs 引入单步共享模式
==========================================

该脚本比较两组实验：
1) 无重复：两个训练序列在所有时间步都不同（通过生成器的跨序列唯一性约束保证）。
2) 有重复：在相同时间步（默认为中点）让两个序列共享同一帧（将序列2的该帧替换为序列1对应帧）。

评价指标：
- 对每个序列做回放并计算 recall_accuracy，用于粗略衡量记忆与回放质量。

运行方式::

    python examples/figure5_compare_repetition.py
    # 可选参数
    python examples/figure5_compare_repetition.py --shared-pos 1 --epochs 200
"""

import os
import sys
import argparse
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import MemorySequenceAttractorNetwork  # noqa: E402  (路径插入需在导入前)


@dataclass(frozen=True)
class ExpConfig:
    """实验配置。"""

    N_v: int = 60
    T: int = 32
    N_h: int = 240
    eta: float = 0.01
    kappa: float = 1.0
    epochs: int = 220
    seeds: Tuple[int, int] = (11, 23)
    shared_pos: int = -1  # -1 表示默认使用中点


def build_network(cfg: ExpConfig) -> MemorySequenceAttractorNetwork:
    """创建统一记忆网络实例。"""
    return MemorySequenceAttractorNetwork(
        N_v=cfg.N_v,
        T=cfg.T,
        N_h=cfg.N_h,
        eta=cfg.eta,
        kappa=cfg.kappa,
    )


def generate_distinct_pair(cfg: ExpConfig) -> List[np.ndarray]:
    """
    生成两个互不重复的训练序列（跨序列所有帧唯一，忽略最后回环帧）。
    """
    generator = build_network(cfg)
    sequences = generator.generate_multiple_sequences(
        num_sequences=2,
        seeds=list(cfg.seeds),
        ensure_unique_across=True,
    )
    return [sequences[0].copy(), sequences[1].copy()]


def make_shared_frame_pair(sequences: Sequence[np.ndarray], shared_pos: int, T: int) -> List[np.ndarray]:
    """
    在给定时间步 shared_pos 让两个序列共享同一帧：
    - 将 seq2[shared_pos] 用 seq1[shared_pos] 覆盖
    """
    seq1 = sequences[0].copy()
    seq2 = sequences[1].copy()

    # 规整共享位置
    if shared_pos < 0 or shared_pos >= T:
        shared_pos = T // 2

    # 避免覆盖最后一帧（通常等于首帧作为回环）
    if shared_pos == T - 1:
        shared_pos = T // 2

    seq2[shared_pos] = seq1[shared_pos]
    return [seq1, seq2]


def train_and_summarize(cfg: ExpConfig, title: str, sequences: Sequence[np.ndarray]) -> Tuple[MemorySequenceAttractorNetwork, List[float]]:
    """
    训练网络并返回每个序列的回放准确率。
    """
    print(f"\n=== {title} ===")
    net = build_network(cfg)
    net.train(x=list(sequences), num_epochs=cfg.epochs, verbose=False, interleaved=True)

    accuracies: List[float] = []
    for idx in range(len(sequences)):
        xi_replayed = net.replay(sequence_index=idx, max_steps=net.T * 3)
        eval_res = net.evaluate_replay(xi_replayed, sequence_index=idx)
        acc = float(eval_res.get("recall_accuracy", 0.0))
        ok = eval_res.get("found_sequence", False)
        accuracies.append(acc)
        status = "✓ 成功" if ok else "✗ 失败"
        print(f"  序列 #{idx}: {status}，recall_accuracy = {acc * 100:.2f}%")

    return net, accuracies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="图5对比实验：无重复 vs 单步共享")
    parser.add_argument("--epochs", type=int, default=220, help="训练轮数")
    parser.add_argument("--shared-pos", type=int, default=-1, help="共享帧位置（默认中点）")
    parser.add_argument("--seed", type=int, default=2025, help="全局随机种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    cfg = ExpConfig(epochs=args.epochs, shared_pos=args.shared_pos)

    # 实验 A：无重复
    distinct_pair = generate_distinct_pair(cfg)
    _, acc_no_shared = train_and_summarize(cfg, "实验 A：无重复（两序列所有帧互异）", distinct_pair)

    # 实验 B：引入单步共享（两个序列在同一时间步共享一帧）
    shared_pair = make_shared_frame_pair(distinct_pair, cfg.shared_pos, cfg.T)
    _, acc_with_shared = train_and_summarize(cfg, "实验 B：引入单步共享（同一时间步相同帧）", shared_pair)

    # 简要对比汇总
    print("\n=== 汇总对比（recall_accuracy） ===")
    for i in range(2):
        a = acc_no_shared[i] * 100.0
        b = acc_with_shared[i] * 100.0
        diff = b - a
        print(f"序列 #{i}: 无重复 = {a:.2f}%, 共享 = {b:.2f}%  (Δ = {diff:+.2f}%)")

    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()


