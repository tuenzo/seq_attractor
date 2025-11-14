"""
================================================================
记忆增强序列吸引子网络
统一支持多序列学习与增量学习
================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..core.base import SequenceAttractorNetwork
from ..utils.evaluation import evaluate_replay_full_sequence


@dataclass(frozen=True)
class TrainingSummary:
    """单次训练过程的统计信息。"""

    mu_history: np.ndarray
    nu_history: np.ndarray
    final_mu: Optional[float]
    final_nu: Optional[float]


class MemorySequenceAttractorNetwork(SequenceAttractorNetwork):
    """
    记忆增强序列吸引子网络。

    - 支持一次性学习多个序列（多序列学习）；
    - 支持在保留既有记忆的前提下增量加入新序列（增量学习）；
    - 与基础类保持向后兼容（单序列训练、回放、评估接口一致）。
    """

    def __init__(
        self,
        N_v: int,
        T: int,
        N_h: Optional[int] = None,
        eta: float = 0.001,
        kappa: float = 1,
    ) -> None:
        super().__init__(N_v=N_v, T=T, N_h=N_h, eta=eta, kappa=kappa)

        self.training_sequences: List[np.ndarray] = []
        self.sequence_training_info: List[Dict] = []
        self._total_epochs_trained: int = 0

    # ------------------------------------------------------------------
    # 序列生成相关
    # ------------------------------------------------------------------
    def generate_random_sequence_with_length(
        self,
        T: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """生成指定长度的随机周期序列，并确保非末尾帧唯一。"""
        if seed is not None:
            np.random.seed(seed)

        x = np.sign(np.random.randn(T, self.N_v))
        x[x == 0] = 1

        for t in range(1, T - 1):
            while np.any(np.all(x[t, :] == x[:t, :], axis=1)):
                x[t, :] = np.sign(np.random.randn(self.N_v))
                x[t, x[t, :] == 0] = 1

        x[T - 1, :] = x[0, :]
        return x

    def generate_multiple_sequences(
        self,
        num_sequences: int,
        seeds: Optional[List[int]] = None,
        T: Optional[int] = None,
        ensure_unique_across: bool = True,
        max_attempts: int = 1000,
    ) -> List[np.ndarray]:
        """生成多个随机序列，可选跨序列唯一性约束。"""
        sequences: List[np.ndarray] = []
        if seeds is None:
            seeds = list(range(num_sequences))

        seq_length = T if T is not None else self.T

        if not ensure_unique_across:
            for seed in seeds[:num_sequences]:
                seq = self.generate_random_sequence_with_length(seq_length, seed)
                sequences.append(seq)
            return sequences

        print(f"生成 {num_sequences} 个序列，确保跨序列唯一性...")
        all_used_frames: List[np.ndarray] = []

        for seq_idx, seed in enumerate(seeds[:num_sequences]):
            if seed is not None:
                np.random.seed(seed)

            print(f"  正在生成序列 #{seq_idx + 1}...", end=" ")
            seq = np.zeros((seq_length, self.N_v))

            for t in range(seq_length - 1):
                attempts = 0
                candidate_frame = None
                while attempts < max_attempts:
                    candidate_frame = np.sign(np.random.randn(self.N_v))
                    candidate_frame[candidate_frame == 0] = 1

                    is_unique_within = not np.any(
                        np.all(candidate_frame == seq[:t, :], axis=1)
                    )
                    if not is_unique_within:
                        attempts += 1
                        continue

                    is_unique_across = all(
                        not np.array_equal(candidate_frame, used)
                        for used in all_used_frames
                    )
                    if is_unique_across:
                        seq[t, :] = candidate_frame
                        all_used_frames.append(candidate_frame.copy())
                        break

                    attempts += 1

                if attempts >= max_attempts and candidate_frame is not None:
                    print(
                        f"\n警告: 序列 #{seq_idx + 1} 位置 {t} 无法生成唯一帧"
                        f"（尝试 {max_attempts} 次）"
                    )
                    seq[t, :] = candidate_frame  # type: ignore[assignment]

            seq[seq_length - 1, :] = seq[0, :]
            sequences.append(seq)
            print("完成")

        print("所有序列生成完毕\n")
        return sequences

    # ------------------------------------------------------------------
    # 训练主流程
    # ------------------------------------------------------------------
    def train(
        self,
        x: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
        num_epochs: int = 500,
        verbose: bool = True,
        seed: Optional[int] = None,
        V_only: bool = False,
        interleaved: bool = True,
        incremental: bool = False,
        reset_history: bool = False,
    ) -> Dict:
        """
        训练网络，统一处理以下模式：
        - 单序列训练（向后兼容基础类）；
        - 一次性多序列训练；
        - 增量式训练（持续保留旧序列）。
        """
        if num_epochs < 0:
            raise ValueError("num_epochs 必须为非负整数")

        if reset_history:
            self._reset_memory(verbose=verbose)

        if isinstance(x, Sequence) and not isinstance(x, np.ndarray):
            return self._train_with_sequence_list(
                sequences=x,
                num_epochs=num_epochs,
                verbose=verbose,
                V_only=V_only,
                interleaved=interleaved,
            )

        return self._train_single_sequence(
            sequence=x,
            num_epochs=num_epochs,
            verbose=verbose,
            seed=seed,
            V_only=V_only,
            interleaved=interleaved,
            incremental=incremental,
        )

    # ------------------------------------------------------------------
    # 回放与评估
    # ------------------------------------------------------------------
    def replay(
        self,
        x_init: Optional[np.ndarray] = None,
        noise_level: float = 0.0,
        max_steps: Optional[int] = None,
        sequence_index: int = 0,
    ) -> np.ndarray:
        """从指定序列索引回放，默认使用第一个已学习的序列。"""
        if x_init is None:
            if self.training_sequences:
                assert (
                    0 <= sequence_index < len(self.training_sequences)
                ), f"序列索引 {sequence_index} 超出范围"
                x_init = self.training_sequences[sequence_index][0, :].copy()
            elif self.training_sequence is not None:
                x_init = self.training_sequence[0, :].copy()
            else:
                raise AssertionError("请先训练网络或提供初始状态")

        return super().replay(x_init=x_init, noise_level=noise_level, max_steps=max_steps)

    def evaluate_replay(
        self,
        xi_replayed: Optional[np.ndarray] = None,
        sequence_index: Optional[int] = None,
        num_trials: int = 50,
        noise_level: float = 0.0,
        verbose: bool = False,
    ) -> Dict:
        """评估回放质量，支持单次或多次试验模式。"""
        if not self.training_sequences:
            raise AssertionError("请先训练网络")

        if xi_replayed is None:
            if sequence_index is not None:
                return self._test_sequence_recall(
                    sequence_index=sequence_index,
                    num_trials=num_trials,
                    noise_level=noise_level,
                    verbose=verbose,
                )

            results: Dict[str, Dict] = {}
            for idx in range(len(self.training_sequences)):
                if verbose:
                    print(f"\n测试序列 #{idx}:")
                results[f"sequence_{idx}"] = self._test_sequence_recall(
                    sequence_index=idx,
                    num_trials=num_trials,
                    noise_level=noise_level,
                    verbose=verbose,
                )
            return results

        if sequence_index is not None:
            return evaluate_replay_full_sequence(
                xi_replayed,
                self.training_sequences[sequence_index],
            )

        matches = []
        for idx, target in enumerate(self.training_sequences):
            match = evaluate_replay_full_sequence(xi_replayed, target)
            match["sequence_index"] = idx
            matches.append(match)

        best_idx = int(np.argmax([m.get("found_sequence", False) for m in matches]))
        return {
            "best_match": matches[best_idx],
            "all_matches": matches,
            "best_sequence_index": best_idx,
        }

    def test_robustness(
        self,
        noise_levels: np.ndarray,
        num_trials: int = 50,
        verbose: bool = True,
        sequence_index: int = 0,
    ) -> np.ndarray:
        """测试噪声鲁棒性。"""
        assert (
            0 <= sequence_index < len(self.training_sequences)
        ), f"序列索引 {sequence_index} 超出范围"

        robustness_scores = np.zeros(len(noise_levels))
        for i, noise_level in enumerate(noise_levels):
            result = self._test_sequence_recall(
                sequence_index=sequence_index,
                num_trials=num_trials,
                noise_level=noise_level,
                verbose=verbose,
            )
            robustness_scores[i] = result["success_rate"]
        return robustness_scores

    # ------------------------------------------------------------------
    # 状态检索
    # ------------------------------------------------------------------
    @property
    def num_sequences(self) -> int:
        """当前已存储的序列数量。"""
        return len(self.training_sequences)

    def get_memory_status(self) -> Dict:
        """返回当前记忆状态。"""
        return {
            "num_sequences": self.num_sequences,
            "total_epochs_trained": self._total_epochs_trained,
            "sequence_info": self.sequence_training_info,
            "network_params": {
                "N_v": self.N_v,
                "T": self.T,
                "N_h": self.N_h,
                "eta": self.eta,
                "kappa": self.kappa,
            },
        }

    def test_all_memories(self, verbose: bool = True) -> Dict:
        """测试所有已学习序列的回放质量。"""
        if not self.training_sequences:
            if verbose:
                print("警告：没有已学习的序列")
            return {}

        results: Dict[str, Dict] = {}
        if verbose:
            print("\n" + "=" * 60)
            print(f"测试所有记忆 ({self.num_sequences} 个序列)")
            print("=" * 60)

        for idx, seq in enumerate(self.training_sequences):
            xi_replayed = self.replay(
                x_init=seq[0, :].copy(),
                sequence_index=idx,
                max_steps=self.T * 3,
            )
            found = any(
                np.array_equal(
                    xi_replayed[start : start + len(seq), :],  # noqa: E203
                    seq,
                )
                for start in range(max(1, xi_replayed.shape[0] - len(seq) + 1))
            )

            success_rate = 1.0 if found else 0.0
            results[f"sequence_{idx}"] = {
                "success": found,
                "success_rate": success_rate,
                "sequence_length": len(seq),
            }

            if verbose:
                status = "✓ 成功" if found else "✗ 失败"
                print(f"序列 #{idx}: {status} (回放成功率: {success_rate * 100:.0f}%)")

        total_success = sum(1 for r in results.values() if r["success"])
        overall_rate = total_success / self.num_sequences if self.num_sequences else 0.0
        results["summary"] = {
            "total_sequences": self.num_sequences,
            "successful_recalls": total_success,
            "overall_success_rate": overall_rate,
        }

        if verbose:
            print(f"\n总体成功率: {overall_rate * 100:.1f}% "
                  f"({total_success}/{self.num_sequences})")
            print("=" * 60)

        return results

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _train_with_sequence_list(
        self,
        sequences: Sequence[np.ndarray],
        num_epochs: int,
        verbose: bool,
        V_only: bool,
        interleaved: bool,
    ) -> Dict:
        if len(sequences) == 0:
            raise ValueError("多序列训练至少需要一个序列")

        validated = [self._ensure_sequence_shape(seq) for seq in sequences]
        self.training_sequences = [seq.copy() for seq in validated]
        self.training_sequence = self.training_sequences[0]

        if verbose:
            print(f"开始多序列训练... N_v={self.N_v}, N_h={self.N_h}")
            print(f"参数: eta={self.eta}, kappa={self.kappa}, epochs={num_epochs}")
            print(f"序列数量: {self.num_sequences}")
            print(f"训练模式: {'交替训练' if interleaved else '批量训练'}")

        summary = self._train_sequences(
            sequences=self.training_sequences,
            num_epochs=num_epochs,
            V_only=V_only,
            verbose=verbose,
            interleaved=interleaved,
        )

        self._append_history(summary.mu_history, summary.nu_history)
        start_epoch = self._total_epochs_trained
        self._total_epochs_trained += num_epochs

        for idx in range(self.num_sequences):
            self.sequence_training_info.append(
                {
                    "sequence_index": idx,
                    "start_epoch": start_epoch,
                    "end_epoch": self._total_epochs_trained,
                    "num_epochs": num_epochs,
                    "incremental": False,
                    "training_mode": "multi_sequence",
                }
            )

        return {
            "mu_history": summary.mu_history,
            "nu_history": summary.nu_history,
            "final_mu": summary.final_mu,
            "final_nu": summary.final_nu,
            "num_sequences": self.num_sequences,
            "num_learned_sequences": self.num_sequences,
            "total_epochs": self._total_epochs_trained,
            "current_epochs": num_epochs,
            "training_mode": "multi_sequence",
        }

    def _train_single_sequence(
        self,
        sequence: Optional[np.ndarray],
        num_epochs: int,
        verbose: bool,
        seed: Optional[int],
        V_only: bool,
        interleaved: bool,
        incremental: bool,
    ) -> Dict:
        if sequence is None:
            if not self.training_sequences:
                sequence = self.generate_random_sequence(seed)
                is_new_sequence = True
            else:
                sequence = self.training_sequences[-1]
                is_new_sequence = False
        else:
            sequence = self._ensure_sequence_shape(sequence)
            is_new_sequence = not any(
                np.array_equal(sequence, seq) for seq in self.training_sequences
            )

        if is_new_sequence:
            self.training_sequences.append(sequence.copy())
            sequence_index = self.num_sequences - 1
            mode = (
                "增量学习新序列"
                if incremental and self.num_sequences > 1
                else "学习新序列"
            )
        else:
            sequence_index = next(
                idx
                for idx, seq in enumerate(self.training_sequences)
                if np.array_equal(seq, sequence)
            )
            mode = "继续训练已有序列" if not incremental else "增量继续训练"

        self.training_sequence = sequence
        sequences_to_train: Sequence[np.ndarray]
        if incremental and self.num_sequences > 1:
            sequences_to_train = self.training_sequences
        else:
            sequences_to_train = [sequence]

        if verbose:
            print(f"{mode}...")
            print(f"N_v={self.N_v}, T={self.T}, N_h={self.N_h}")
            print(f"参数: eta={self.eta}, kappa={self.kappa}, epochs={num_epochs}")
            print(f"已学习序列数: {self.num_sequences}")
            if self._total_epochs_trained > 0:
                print(f"累计训练轮数: {self._total_epochs_trained}")
            if V_only:
                print("仅更新 V 权重矩阵")
            if incremental and self.num_sequences > 1:
                print(f"增量学习模式：将训练 {len(sequences_to_train)} 个序列")

        if num_epochs == 0:
            mu_history = np.zeros(0)
            nu_history = np.zeros(0)
        elif len(sequences_to_train) == 1 and not incremental:
            base_result = super().train(
                x=sequence,
                num_epochs=num_epochs,
                verbose=verbose,
                seed=seed,
                V_only=V_only,
            )
            mu_history = np.array(base_result["mu_history"])
            nu_history = np.array(base_result["nu_history"])
        else:
            summary = self._train_sequences(
                sequences=sequences_to_train,
                num_epochs=num_epochs,
                V_only=V_only,
                verbose=verbose,
                interleaved=interleaved,
            )
            mu_history = summary.mu_history
            nu_history = summary.nu_history

        self._append_history(mu_history, nu_history)
        start_epoch = self._total_epochs_trained
        self._total_epochs_trained += num_epochs

        if is_new_sequence:
            self.sequence_training_info.append(
                {
                    "sequence_index": sequence_index,
                    "start_epoch": start_epoch,
                    "end_epoch": self._total_epochs_trained,
                    "num_epochs": num_epochs,
                    "incremental": incremental,
                    "training_mode": "incremental" if incremental else "single_sequence",
                }
            )

        return {
            "mu_history": mu_history,
            "nu_history": nu_history,
            "final_mu": mu_history[-1] if mu_history.size > 0 else None,
            "final_nu": nu_history[-1] if nu_history.size > 0 else None,
            "total_epochs": self._total_epochs_trained,
            "current_epochs": num_epochs,
            "num_learned_sequences": self.num_sequences,
            "num_sequences": self.num_sequences,
            "training_mode": mode,
        }

    def _train_sequences(
        self,
        sequences: Sequence[np.ndarray],
        num_epochs: int,
        V_only: bool,
        verbose: bool,
        interleaved: bool,
    ) -> TrainingSummary:
        if num_epochs == 0:
            return TrainingSummary(
                mu_history=np.zeros(0),
                nu_history=np.zeros(0),
                final_mu=None,
                final_nu=None,
            )

        if interleaved:
            mu_history, nu_history = self._train_sequences_interleaved(
                sequences=sequences,
                num_epochs=num_epochs,
                V_only=V_only,
                verbose=verbose,
            )
        else:
            mu_history, nu_history = self._train_sequences_batch(
                sequences=sequences,
                num_epochs=num_epochs,
                V_only=V_only,
                verbose=verbose,
            )

        return TrainingSummary(
            mu_history=mu_history,
            nu_history=nu_history,
            final_mu=mu_history[-1] if mu_history.size > 0 else None,
            final_nu=nu_history[-1] if nu_history.size > 0 else None,
        )

    def _train_sequences_interleaved(
        self,
        sequences: Sequence[np.ndarray],
        num_epochs: int,
        V_only: bool,
        verbose: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        seq_data = []
        total_transitions = 0
        for seq in sequences:
            x_current = seq[:-1, :].T
            x_next = seq[1:, :].T
            seq_data.append(
                {
                    "x_current": x_current,
                    "x_next": x_next,
                    "T": len(seq),
                }
            )
            total_transitions += (len(seq) - 1)

        mu_history = np.zeros(num_epochs)
        nu_history = np.zeros(num_epochs)

        for epoch in range(num_epochs):
            epoch_mu = 0.0
            epoch_nu = 0.0

            for data in seq_data:
                x_current_all = data["x_current"]
                x_next_all = data["x_next"]

                if not V_only:
                    z_target_all = np.sign(self.P @ x_next_all)
                    z_target_all[z_target_all == 0] = 1
                    h_input_all = self.U @ x_current_all
                    mu_all = (z_target_all * h_input_all < self.kappa).astype(float)
                    delta_U = (mu_all * z_target_all) @ x_current_all.T
                    self.U += self.eta * delta_U
                    epoch_mu += float(np.sum(mu_all))
                else:
                    mu_all = np.zeros_like(self.U @ x_current_all)

                y_actual_all = np.sign(self.U @ x_current_all)
                y_actual_all[y_actual_all == 0] = 1
                v_input_all = self.V @ y_actual_all
                nu_all = (x_next_all * v_input_all < self.kappa).astype(float)
                delta_V = (nu_all * x_next_all) @ y_actual_all.T
                self.V += self.eta * delta_V
                epoch_nu += float(np.sum(nu_all))

            mu_history[epoch] = epoch_mu / (self.N_h * total_transitions)
            nu_history[epoch] = epoch_nu / (self.N_v * total_transitions)

            if verbose and (epoch + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, "
                    f"μ={mu_history[epoch]:.4f}, ν={nu_history[epoch]:.4f}"
                )

        return mu_history, nu_history

    def _train_sequences_batch(
        self,
        sequences: Sequence[np.ndarray],
        num_epochs: int,
        V_only: bool,
        verbose: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        all_x_current = []
        all_x_next = []

        for seq in sequences:
            all_x_current.append(seq[:-1, :].T)
            all_x_next.append(seq[1:, :].T)

        x_current_all = np.hstack(all_x_current)
        x_next_all = np.hstack(all_x_next)
        total_transitions = x_current_all.shape[1]

        mu_history = np.zeros(num_epochs)
        nu_history = np.zeros(num_epochs)

        for epoch in range(num_epochs):
            if not V_only:
                z_target_all = np.sign(self.P @ x_next_all)
                z_target_all[z_target_all == 0] = 1
                h_input_all = self.U @ x_current_all
                mu_all = (z_target_all * h_input_all < self.kappa).astype(float)
                delta_U = (mu_all * z_target_all) @ x_current_all.T
                self.U += self.eta * delta_U
                total_mu = float(np.sum(mu_all))
            else:
                total_mu = 0.0

            y_actual_all = np.sign(self.U @ x_current_all)
            y_actual_all[y_actual_all == 0] = 1
            v_input_all = self.V @ y_actual_all
            nu_all = (x_next_all * v_input_all < self.kappa).astype(float)
            delta_V = (nu_all * x_next_all) @ y_actual_all.T
            self.V += self.eta * delta_V
            total_nu = float(np.sum(nu_all))

            mu_history[epoch] = total_mu / (self.N_h * total_transitions)
            nu_history[epoch] = total_nu / (self.N_v * total_transitions)

            if verbose and (epoch + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, "
                    f"μ={mu_history[epoch]:.4f}, ν={nu_history[epoch]:.4f}"
                )

        return mu_history, nu_history

    def _append_history(self, mu_history: np.ndarray, nu_history: np.ndarray) -> None:
        if hasattr(self.mu_history, "tolist"):
            self.mu_history = list(self.mu_history)  # type: ignore[assignment]
        if hasattr(self.nu_history, "tolist"):
            self.nu_history = list(self.nu_history)  # type: ignore[assignment]

        if not isinstance(self.mu_history, list):
            self.mu_history = []  # type: ignore[assignment]
        if not isinstance(self.nu_history, list):
            self.nu_history = []  # type: ignore[assignment]

        self.mu_history.extend(mu_history.tolist())
        self.nu_history.extend(nu_history.tolist())

    def _ensure_sequence_shape(self, sequence: np.ndarray) -> np.ndarray:
        assert sequence.shape[1] == self.N_v, (
            f"序列可见层维度应为 {self.N_v}，实际为 {sequence.shape[1]}"
        )
        if sequence.shape[0] != self.T:
            raise AssertionError(
                f"序列长度应为 {self.T}，实际为 {sequence.shape[0]}"
            )
        return sequence

    def _reset_memory(self, verbose: bool) -> None:
        self.mu_history = []
        self.nu_history = []
        self.training_sequences = []
        self.sequence_training_info = []
        self._total_epochs_trained = 0
        self.training_sequence = None
        if verbose:
            print("已重置训练历史和所有记忆")

    def _test_sequence_recall(
        self,
        sequence_index: int,
        num_trials: int = 50,
        noise_level: float = 0.0,
        verbose: bool = False,
    ) -> Dict:
        assert 0 <= sequence_index < self.num_sequences, (
            f"序列索引 {sequence_index} 超出范围"
        )

        target_sequence = self.training_sequences[sequence_index]
        T = len(target_sequence)
        max_search_steps = T * 5

        success_count = 0
        convergence_steps: List[int] = []

        for _ in range(num_trials):
            xi_test = target_sequence[0, :].copy().reshape(-1, 1)

            if noise_level > 0:
                num_flips = int(noise_level * self.N_v)
                if num_flips > 0:
                    flip_indices = np.random.choice(self.N_v, num_flips, replace=False)
                    xi_test[flip_indices] = -xi_test[flip_indices]

            trajectory = [xi_test.flatten().copy()]
            for _ in range(max_search_steps):
                zeta = np.sign(self.U @ xi_test)
                zeta[zeta == 0] = 1
                xi_test = np.sign(self.V @ zeta)
                xi_test[xi_test == 0] = 1
                trajectory.append(xi_test.flatten().copy())

            found_sequence = False
            for tau in range(max_search_steps - T + 2):
                segment = np.array(trajectory[tau : tau + T])
                if np.array_equal(segment, target_sequence):
                    found_sequence = True
                    convergence_steps.append(tau)
                    break

            if found_sequence:
                success_count += 1

        success_rate = success_count / num_trials
        if verbose:
            print(
                f"序列 #{sequence_index}, 噪声水平 {noise_level:.2f}: "
                f"成功率 {success_rate * 100:.1f}% "
                f"({success_count}/{num_trials} 次成功)"
            )
            if convergence_steps:
                print(f"  平均收敛步数: {np.mean(convergence_steps):.1f}")

        return {
            "success_rate": success_rate,
            "recall_accuracy": success_rate,
            "success_count": success_count,
            "num_trials": num_trials,
            "noise_level": noise_level,
            "sequence_index": sequence_index,
            "convergence_steps": convergence_steps if convergence_steps else None,
            "avg_convergence_steps": np.mean(convergence_steps)
            if convergence_steps
            else None,
            "evaluation_mode": "multiple_trials",
        }


