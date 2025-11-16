"""Figure 5 reproduction experiments for the sequence attractor network."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import math

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None

from ..models.pattern_repetition import PatternRepetitionNetwork


@dataclass(frozen=True)
class Figure5Config:
    """Configuration container for the Figure 5 experiments."""

    num_trials: int = 100
    noise_num: int = 10
    num_epochs: int = 500
    T_values: Sequence[int] = (10, 30, 50, 70, 90, 110, 140)
    N_h_values: Sequence[int] = (100, 325, 550, 775, 1000)
    # 多序列学习相关参数
    num_sequences: Optional[int] = None  # None表示使用默认值2
    # 重复模式相关参数
    with_shared_patterns: bool = False  # 是否使用共享模式
    shared_pattern_positions: Optional[List[List[Tuple[int, int]]]] = None  # 共享模式位置，None表示默认中间位置

    def noise_level(self, N_v: int) -> float:
        return self.noise_num / float(N_v)
    
    def get_num_sequences(self) -> int:
        """获取序列数量，默认值为2"""
        return self.num_sequences if self.num_sequences is not None else 2


def _ensure_output_dir(output_dir: Optional[Path], *, create_timestamp: bool) -> Path:
    base_dir = Path(output_dir) if output_dir is not None else Path("figure5_results")
    if create_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = base_dir / f"figure5_{timestamp}"
    else:
        target_dir = base_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _create_progress_bar(total: int, *, enabled: bool, desc: str):
    if not enabled or total <= 1 or tqdm is None:
        return None
    return tqdm(total=total, desc=desc, leave=False, unit="cfg")


def _resolve_worker_count(requested: Optional[int], reserve_ratio: float = 0.25) -> int:
    """
    Translate user input into an actual worker count.

    Semantics:
    - None or <= 1 → run sequentially.
    - -1 → use (1 - reserve_ratio) of available CPUs (default 75%, at least 1).
    - 0 → use all available CPUs.
    - N > 1 → use min(N, cpu_count).
    
    Args:
        requested: Requested worker count
        reserve_ratio: Ratio of CPUs to reserve for system (default 0.25 = 25%)
    """

    if requested is None or requested < -1:
        return 1

    total = cpu_count() or 1
    if requested == -1:
        return max(1, int(total * (1 - reserve_ratio)))
    if requested == 0:
        return total
    if requested < -1:
        raise ValueError(f"Invalid worker count: {requested}")
    return max(1, min(requested, total))


def _single_trial_task(trial_params: Dict) -> bool:
    """
    Single trial task function for parallel execution.
    
    Args:
        trial_params: Dictionary containing all necessary parameters:
            - params: Network parameters (N_v, N_h, T, eta, kappa)
            - num_epochs: Number of training epochs
            - noise_level: Noise level for robustness testing
            - v_only: Whether to train only V weights
            - num_sequences: Number of sequences to learn (None means default 2)
            - with_shared_patterns: Whether to use shared patterns between sequences
            - shared_pattern_positions: Shared pattern positions
            - seed: Random seed for this trial
    
    Returns:
        bool: True if robustness test passed (robustness > 0.5), False otherwise
    """
    params = trial_params["params"]
    num_epochs = trial_params["num_epochs"]
    noise_level = trial_params["noise_level"]
    v_only = trial_params["v_only"]
    seed = trial_params["seed"]
    
    # 多序列学习相关参数
    num_sequences = trial_params.get("num_sequences")
    with_shared_patterns = trial_params.get("with_shared_patterns", False)
    shared_pattern_positions = trial_params.get("shared_pattern_positions")
    
    # Each trial already has a unique seed (trial index), so we can use it directly
    # No need to use process ID since each task has a unique seed parameter
    # This ensures reproducibility and avoids conflicts in parallel execution
    trial_seed = seed
    
    # Create network with seed for weight initialization
    # Use a different seed offset for weight initialization to avoid conflicts
    weight_init_seed = (trial_seed * 1000 + 10000) % (2**31)
    
    # 使用多序列学习（PatternRepetitionNetwork）
    network = PatternRepetitionNetwork(**params, seed=weight_init_seed)
    
    # 确定序列数量（如果为None，默认使用2个序列）
    actual_num_sequences = num_sequences if num_sequences is not None else 2
    
    # 生成多序列（直接使用已创建的 network 实例）
    if not with_shared_patterns:
        # 不使用共享模式，生成多个唯一序列
        x_train_list = network.generate_multiple_sequences(
            num_sequences=actual_num_sequences,
            seeds=[trial_seed + i for i in range(actual_num_sequences)],
            T=params["T"],
            ensure_unique_across=True,
            verbose=False,
        )
    else:
        # 使用共享模式
        T = params["T"]
        
        # 确定共享模式位置
        if shared_pattern_positions is None:
            # 默认：两个序列，中间位置有一个模式重复
            if actual_num_sequences < 2:
                actual_num_sequences = 2
            mid_pos = T // 2
            start_pos = mid_pos
            end_pos = mid_pos
            
            # 构建共享组：序列0和1共享中间模式
            shared_groups = [[0, 1]] if actual_num_sequences >= 2 else []
            patterns_per_group = [1]
            positions_per_group = [
                [
                    [(start_pos, end_pos)],  # 序列0的位置
                    [(start_pos, end_pos)],  # 序列1的位置
                ]
            ]
            
            # 如果序列数大于2，其他序列不共享
            if actual_num_sequences > 2:
                # 为其他序列生成唯一序列
                all_sequences = []
                # 先为前两个序列生成共享模式
                shared_seqs = network.generate_sequences_with_custom_patterns(
                    num_sequences=2,
                    shared_groups=shared_groups,
                    patterns_per_group=patterns_per_group,
                    positions_per_group=positions_per_group,
                    seeds=[trial_seed, trial_seed + 1],
                    T=T,
                    ensure_unique_non_shared=True,
                    verbose=False,
                )
                all_sequences.extend(shared_seqs)
                
                # 为其他序列生成唯一序列
                for i in range(2, actual_num_sequences):
                    unique_seqs = network.generate_multiple_sequences(
                        num_sequences=1,
                        seeds=[trial_seed + i],
                        T=T,
                        ensure_unique_across=True,
                        verbose=False,
                    )
                    all_sequences.extend(unique_seqs)
                
                x_train_list = all_sequences
            else:
                # 正好两个序列，使用共享模式
                x_train_list = network.generate_sequences_with_custom_patterns(
                    num_sequences=actual_num_sequences,
                    shared_groups=shared_groups,
                    patterns_per_group=patterns_per_group,
                    positions_per_group=positions_per_group,
                    seeds=[trial_seed + i for i in range(actual_num_sequences)],
                    T=T,
                    ensure_unique_non_shared=True,
                    verbose=False,
                )
        else:
            # 使用用户指定的共享模式位置
            # 需要将用户格式转换为generate_sequences_with_custom_patterns需要的格式
            # 用户格式: [[(start1, end1), ...], [(start2, end2), ...], ...]
            # 需要转换为: shared_groups, patterns_per_group, positions_per_group
            
            # 简化处理：假设所有序列共享相同的模式位置（可以后续扩展）
            # 这里我们假设用户想要所有序列对之间共享模式
            # 为了简化，我们假设前两个序列共享第一个模式组
            if actual_num_sequences < 2:
                actual_num_sequences = 2
            
            # 解析共享模式位置
            # 假设shared_pattern_positions格式为每个序列的模式位置列表
            if len(shared_pattern_positions) < 2:
                # 如果只提供了一个序列的位置，复制给第二个序列
                shared_pattern_positions = [
                    shared_pattern_positions[0],
                    shared_pattern_positions[0] if len(shared_pattern_positions) > 0 else [(T // 2, T // 2)],
                ]
            
            # 构建共享组配置
            shared_groups = [[0, 1]]
            patterns_per_group = [len(shared_pattern_positions[0])]
            positions_per_group = [
                [
                    shared_pattern_positions[0],  # 序列0的位置
                    shared_pattern_positions[1] if len(shared_pattern_positions) > 1 else shared_pattern_positions[0],  # 序列1的位置
                ]
            ]
            
            # 如果序列数大于2，其他序列不共享
            if actual_num_sequences > 2:
                all_sequences = []
                shared_seqs = network.generate_sequences_with_custom_patterns(
                    num_sequences=2,
                    shared_groups=shared_groups,
                    patterns_per_group=patterns_per_group,
                    positions_per_group=positions_per_group,
                    seeds=[trial_seed, trial_seed + 1],
                    T=T,
                    ensure_unique_non_shared=True,
                    verbose=False,
                )
                all_sequences.extend(shared_seqs)
                
                for i in range(2, actual_num_sequences):
                    unique_seqs = network.generate_multiple_sequences(
                        num_sequences=1,
                        seeds=[trial_seed + i],
                        T=T,
                        ensure_unique_across=True,
                        verbose=False,
                    )
                    all_sequences.extend(unique_seqs)
                
                x_train_list = all_sequences
            else:
                x_train_list = network.generate_sequences_with_custom_patterns(
                    num_sequences=actual_num_sequences,
                    shared_groups=shared_groups,
                    patterns_per_group=patterns_per_group,
                    positions_per_group=positions_per_group,
                    seeds=[trial_seed + i for i in range(actual_num_sequences)],
                    T=T,
                    ensure_unique_non_shared=True,
                    verbose=False,
                )
    
    # 训练多序列
    network.train(
        x=x_train_list,
        num_epochs=num_epochs,
        verbose=False,
        V_only=v_only,
        interleaved=True,
    )
    
    # 测试所有序列的鲁棒性，计算平均成功率
    # 分别测试每个序列，然后计算总成功率
    num_learned_sequences = len(network.training_sequences)
    if num_learned_sequences == 0:
        # 如果没有序列，返回False
        return False
    
    success_rates = []
    for seq_idx in range(num_learned_sequences):
        robustness = network.test_robustness(
            noise_levels=np.array([noise_level]),
            num_trials=1,
            verbose=False,
            sequence_index=seq_idx,
        )
        success_rates.append(robustness[0])
    
    # 计算所有序列的平均成功率
    avg_success_rate = np.mean(success_rates)
    robustness = np.array([avg_success_rate])
    
    return bool(robustness[0] > 0.5)


def _run_trials(
    params: Dict,
    *,
    cfg: Figure5Config,
    v_only: bool,
    num_workers: int,
    show_trial_progress: bool = False,
    num_sequences: Optional[int] = None,
    with_shared_patterns: bool = False,
    shared_pattern_positions: Optional[List[List[Tuple[int, int]]]] = None,
) -> float:
    """
    Run multiple trials in parallel or sequentially.
    
    Uses multiprocessing.Pool with imap() for better performance than ProcessPoolExecutor.
    Pre-computes all trial parameters to avoid serialization overhead.
    
    Args:
        params: Network parameters
        cfg: Figure5Config configuration
        v_only: Whether to train only V weights
        num_workers: Number of parallel workers
        show_trial_progress: Whether to show trial progress
        num_sequences: Number of sequences to learn (None means use cfg default)
        with_shared_patterns: Whether to use shared patterns between sequences
        shared_pattern_positions: Shared pattern positions
    """
    noise_level = cfg.noise_level(params["N_v"])
    
    # 确定序列数量（优先使用参数，否则使用配置）
    actual_num_sequences = num_sequences if num_sequences is not None else cfg.get_num_sequences()
    actual_with_shared_patterns = with_shared_patterns
    actual_shared_pattern_positions = shared_pattern_positions if actual_with_shared_patterns else None
    
    # Pre-compute all trial parameters to avoid serialization overhead
    trial_params_list = []
    for trial in range(cfg.num_trials):
        trial_params = {
            "params": params,
            "num_epochs": cfg.num_epochs,
            "noise_level": noise_level,
            "v_only": v_only,
            "seed": trial,
            "num_sequences": actual_num_sequences,
            "with_shared_patterns": actual_with_shared_patterns,
            "shared_pattern_positions": actual_shared_pattern_positions,
        }
        trial_params_list.append(trial_params)

    if num_workers <= 1:
        # Sequential execution
        success_count = 0
        trial_iter = trial_params_list
        if show_trial_progress and tqdm is not None:
            trial_iter = tqdm(
                trial_iter,
                total=cfg.num_trials,
                desc=f"{'V-only' if v_only else 'U+V'} trials",
                leave=False,
                ncols=80,
            )
        for trial_params in trial_iter:
            if _single_trial_task(trial_params):
                success_count += 1
        return success_count / cfg.num_trials

    # Parallel execution using multiprocessing.Pool
    # Limit worker count to not exceed number of trials
    num_workers = min(num_workers, cfg.num_trials)

    with Pool(processes=num_workers) as pool:
        # Use imap() for immediate processing and better progress tracking
        trial_iter = pool.imap(_single_trial_task, trial_params_list)
        if show_trial_progress and tqdm is not None:
            trial_iter = tqdm(
                trial_iter,
                total=cfg.num_trials,
                desc=f"{'V-only' if v_only else 'U+V'} trials",
                leave=False,
                ncols=80,
            )
        results = list(trial_iter)
        success_count = sum(1 for result in results if result)

    return success_count / cfg.num_trials


def plot_figure5(
    results_v_only: Sequence[Dict],
    results_uv: Sequence[Dict],
    param_name: str,
    param_values: Iterable[int],
    *,
    save_path: Optional[Path] = None,
    show_plot: bool = True,
    title_suffix: str = "",
) -> None:
    """Plot the Figure 5 comparison between V-only and U+V training."""

    success_rate_v_only = [result["recall_accuracy"] for result in results_v_only]
    success_rate_uv = [result["recall_accuracy"] for result in results_uv]

    plt.figure(figsize=(8, 6))

    plt.plot(
        list(param_values),
        np.array(success_rate_v_only) * 100,
        "o-",
        linewidth=2,
        markersize=8,
        label="Only training V",
        color="#E74C3C",
    )

    plt.plot(
        list(param_values),
        np.array(success_rate_uv) * 100,
        "s-",
        linewidth=2,
        markersize=8,
        label="Training both U and V",
        color="#3498DB",
    )

    if param_name == "T":
        plt.xlabel("Sequence Length (T)", fontsize=14, fontweight="bold")
        title = "(a) Fixed N=100, M=500"
    elif param_name == "N_h":
        plt.xlabel("Number of Hidden Neurons (M)", fontsize=14, fontweight="bold")
        title = "(b) Fixed N=100, T=70"
    else:
        plt.xlabel(param_name, fontsize=14)
        title = f"Comparison: {param_name}"

    if title_suffix:
        title = f"{title}{title_suffix}"

    plt.ylabel("Successful Retrievals (%)", fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=12, loc="best", framealpha=0.9)
    plt.ylim([-5, 105])
    plt.yticks(np.arange(0, 101, 20))
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()


def _run_single_configuration(
    base_params: Dict,
    varying_key: str,
    varying_values: Sequence[int],
    *,
    config: Figure5Config,
    v_only: bool,
    progress: Optional[Callable[[int, int, int], None]] = None,
    use_progress: bool = False,
    progress_label: Optional[str] = None,
    num_workers: int = 1,
) -> List[Dict]:
    results: List[Dict] = []
    total = len(varying_values)
    bar_desc = progress_label or f"{'V-only' if v_only else 'U+V'}: {varying_key}"
    bar = _create_progress_bar(total, enabled=use_progress, desc=bar_desc)

    try:
        for idx, value in enumerate(varying_values):
            params = dict(base_params)
            params[varying_key] = value
            success_rate = _run_trials(
                params,
                cfg=config,
                v_only=v_only,
                num_workers=num_workers,
                show_trial_progress=use_progress,
            )
            results.append(
                {
                    varying_key: value,
                    "recall_accuracy": success_rate,
                    **{k: v for k, v in params.items() if k != varying_key},
                }
            )

            if progress is not None:
                progress(idx + 1, total, value)
            if bar is not None:
                bar.update(1)
                bar.set_postfix({varying_key: value}, refresh=False)
    finally:
        if bar is not None:
            bar.close()

    return results


def run_figure5_experiments(
    *,
    config: Figure5Config | None = None,
    output_dir: Optional[Path] = None,
    create_timestamp_dir: bool = True,
    show_images: bool = False,
    progress_callback: Optional[Callable[[str, int, int, int], None]] = None,
    use_progress: bool = False,
    workers: Optional[int] = None,
    base_params_a: Optional[Dict] = None,
    base_params_b: Optional[Dict] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], Path]:
    """Execute the Figure 5 experiments with optional multiprocessing.
    
    Args:
        config: Configuration object for experiments
        output_dir: Output directory for results
        create_timestamp_dir: Whether to create timestamped subdirectory
        show_images: Whether to display plots
        progress_callback: Optional callback for progress updates
        use_progress: Whether to show tqdm progress bars
        workers: Number of parallel workers (None/<=1 for sequential, -1 for auto, 0 for all CPUs, N for N workers)
        base_params_a: Base parameters for experiment (a) - scanning T (default: {"N_v": 100, "N_h": 500, "eta": 0.001, "kappa": 1.0})
        base_params_b: Base parameters for experiment (b) - scanning N_h (default: {"N_v": 100, "T": 70, "eta": 0.001, "kappa": 1.0})
    """

    cfg = config or Figure5Config()
    output_path = _ensure_output_dir(output_dir, create_timestamp=create_timestamp_dir)
    worker_count = _resolve_worker_count(workers)

    def report(stage: str, current: int, total: int, value: int) -> None:
        if progress_callback is not None:
            progress_callback(stage, current, total, value)

    if base_params_a is None:
        base_params_a = {"N_v": 100, "N_h": 500, "eta": 0.001, "kappa": 1.0}
    if base_params_b is None:
        base_params_b = {"N_v": 100, "T": 70, "eta": 0.001, "kappa": 1.0}

    results_v_only_a = _run_single_configuration(
        {**base_params_a},
        "T",
        cfg.T_values,
        config=cfg,
        v_only=True,
        progress=None
        if progress_callback is None
        else lambda current, total, value: report("V_only_T", current, total, value),
        use_progress=use_progress,
        progress_label="V-only: scan T",
        num_workers=worker_count,
    )

    results_uv_a = _run_single_configuration(
        {**base_params_a},
        "T",
        cfg.T_values,
        config=cfg,
        v_only=False,
        progress=None
        if progress_callback is None
        else lambda current, total, value: report("UV_T", current, total, value),
        use_progress=use_progress,
        progress_label="U+V: scan T",
        num_workers=worker_count,
    )

    results_v_only_b = _run_single_configuration(
        {**base_params_b},
        "N_h",
        cfg.N_h_values,
        config=cfg,
        v_only=True,
        progress=None
        if progress_callback is None
        else lambda current, total, value: report("V_only_N_h", current, total, value),
        use_progress=use_progress,
        progress_label="V-only: scan N_h",
        num_workers=worker_count,
    )

    results_uv_b = _run_single_configuration(
        {**base_params_b},
        "N_h",
        cfg.N_h_values,
        config=cfg,
        v_only=False,
        progress=None
        if progress_callback is None
        else lambda current, total, value: report("UV_N_h", current, total, value),
        use_progress=use_progress,
        progress_label="U+V: scan N_h",
        num_workers=worker_count,
    )

    plot_figure5(
        results_v_only_a,
        results_uv_a,
        param_name="T",
        param_values=cfg.T_values,
        save_path=output_path / "figure5a.png",
        show_plot=show_images,
    )

    plot_figure5(
        results_v_only_b,
        results_uv_b,
        param_name="N_h",
        param_values=cfg.N_h_values,
        save_path=output_path / "figure5b.png",
        show_plot=show_images,
    )

    summary_path = output_path / "results_summary.txt"
    with summary_path.open("w", encoding="utf-8") as file:
        file.write("Figure 5 实验结果汇总\n")
        file.write("=" * 80 + "\n\n")
        file.write(
            f"实验参数: num_trials={cfg.num_trials}, noise_num={cfg.noise_num}, num_epochs={cfg.num_epochs}\n\n"
        )
        file.write("(a) 扫描序列长度T (N=100, M=500)\n")
        file.write("-" * 80 + "\n")
        file.write(f"{'T':<8} {'V-only (%)':<15} {'U+V (%)':<15} {'Improvement':<15}\n")
        file.write("-" * 80 + "\n")
        for idx, T_value in enumerate(cfg.T_values):
            v_only = results_v_only_a[idx]["recall_accuracy"] * 100
            uv = results_uv_a[idx]["recall_accuracy"] * 100
            improvement = uv - v_only
            file.write(f"{T_value:<8} {v_only:<15.1f} {uv:<15.1f} {improvement:+.1f}\n")

        file.write("\n\n(b) 扫描隐藏层大小M (N=100, T=70)\n")
        file.write("-" * 80 + "\n")
        file.write(f"{'M':<8} {'V-only (%)':<15} {'U+V (%)':<15} {'Improvement':<15}\n")
        file.write("-" * 80 + "\n")
        for idx, N_h_value in enumerate(cfg.N_h_values):
            v_only = results_v_only_b[idx]["recall_accuracy"] * 100
            uv = results_uv_b[idx]["recall_accuracy"] * 100
            improvement = uv - v_only
            file.write(f"{N_h_value:<8} {v_only:<15.1f} {uv:<15.1f} {improvement:+.1f}\n")

    return results_v_only_a, results_uv_a, results_v_only_b, results_uv_b, output_path


# ========= 拆分模式版本：每个子图对比仅训练 V 与训练 U+V =========


def _run_split_configuration(
    base_params: Dict,
    varying_key: str,
    varying_values: Sequence[int],
    *,
    cfg: Figure5Config,
    use_progress: bool = False,
    progress_label: Optional[str] = None,
    num_workers: int = 1,
    num_sequences: Optional[int] = None,
    with_shared_patterns: bool = False,
    shared_pattern_positions: Optional[List[List[Tuple[int, int]]]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    results_v_only: List[Dict] = []
    results_uv: List[Dict] = []
    total = len(varying_values)
    desc = progress_label or f"Split scan: {varying_key}"
    bar = _create_progress_bar(total, enabled=use_progress, desc=desc)

    try:
        for value in varying_values:
            params = dict(base_params)
            params[varying_key] = int(value)

            success_v = _run_trials(
                params,
                cfg=cfg,
                v_only=True,
                num_workers=num_workers,
                show_trial_progress=use_progress,
                num_sequences=num_sequences,
                with_shared_patterns=with_shared_patterns,
                shared_pattern_positions=shared_pattern_positions,
            )
            success_uv = _run_trials(
                params,
                cfg=cfg,
                v_only=False,
                num_workers=num_workers,
                show_trial_progress=use_progress,
                num_sequences=num_sequences,
                with_shared_patterns=with_shared_patterns,
                shared_pattern_positions=shared_pattern_positions,
            )

            common_metadata = {
                varying_key: params[varying_key],
                "recall_accuracy": 0.0,  # placeholder to overwrite below
                "N_v": params["N_v"],
                "num_sequences": num_sequences if num_sequences is not None else cfg.get_num_sequences(),
                "with_shared_patterns": with_shared_patterns,
            }
            if "T" in params:
                common_metadata["T"] = params["T"]
            if "N_h" in params:
                common_metadata["N_h"] = params["N_h"]

            results_v_only.append({**common_metadata, "recall_accuracy": success_v})
            results_uv.append({**common_metadata, "recall_accuracy": success_uv})

            if bar is not None:
                bar.update(1)
                bar.set_postfix({varying_key: params[varying_key]}, refresh=False)
    finally:
        if bar is not None:
            bar.close()

    return results_v_only, results_uv

def run_figure5_experiments_split_modes(
    *,
    config: Figure5Config | None = None,
    output_dir: Optional[Path] = None,
    create_timestamp_dir: bool = True,
    show_images: bool = False,
    use_progress: bool = False,
    workers: Optional[int] = None,
    num_sequences: Optional[int] = None,
    with_shared_patterns: bool = False,
    shared_pattern_positions: Optional[List[List[Tuple[int, int]]]] = None,
    base_params_a: Optional[Dict] = None,
    base_params_b: Optional[Dict] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], Path]:
    """
    拆分模式对比：
    - 图(a)：固定 N_h=500，扫描 T，比较仅训练 V 与训练 U+V
    - 图(b)：固定 T=70，扫描 N_h，比较仅训练 V 与训练 U+V
    
    可选参数：
    - num_sequences: 序列数量（None表示使用config默认值2）
    - with_shared_patterns: 是否使用共享模式
    - shared_pattern_positions: 共享模式位置（None表示默认中间位置）
    - use_progress: 是否显示 tqdm 进度条
    - base_params_a: Base parameters for experiment (a) - scanning T (default: {"N_v": 100, "N_h": 500, "eta": 0.001, "kappa": 1.0})
    - base_params_b: Base parameters for experiment (b) - scanning N_h (default: {"N_v": 100, "T": 70, "eta": 0.001, "kappa": 1.0})
    
    返回：(results_v_only_T_scan, results_uv_T_scan, results_v_only_Nh_scan, results_uv_Nh_scan, output_path)
    """
    cfg = config or Figure5Config()
    output_path = _ensure_output_dir(output_dir, create_timestamp=create_timestamp_dir)
    worker_count = _resolve_worker_count(workers)
    print(f"worker_count: {worker_count}")
    
    # 确定多序列参数（优先使用函数参数，否则使用配置）
    # 如果函数参数和配置都为None，默认使用2个序列
    actual_num_sequences = num_sequences if num_sequences is not None else (cfg.num_sequences if cfg.num_sequences is not None else 2)
    actual_with_shared_patterns = with_shared_patterns
    # 如果 with_shared_patterns=True 且 shared_pattern_positions=None，保持 None
    # 让 _single_trial_task 中的逻辑处理默认中间位置
    # 只有当用户明确提供了 shared_pattern_positions 时才使用它
    actual_shared_pattern_positions = shared_pattern_positions

    if base_params_a is None:
        base_params_a = {"N_v": 100, "N_h": 500, "eta": 0.001, "kappa": 1.0}
    if base_params_b is None:
        base_params_b = {"N_v": 100, "T": 70, "eta": 0.001, "kappa": 1.0}

    results_v_only_T_scan, results_uv_T_scan = _run_split_configuration(
        base_params_a,
        "T",
        cfg.T_values,
        cfg=cfg,
        use_progress=use_progress,
        progress_label="Split scan: T",
        num_workers=worker_count,
        num_sequences=actual_num_sequences,
        with_shared_patterns=actual_with_shared_patterns,
        shared_pattern_positions=actual_shared_pattern_positions,
    )

    results_v_only_Nh_scan, results_uv_Nh_scan = _run_split_configuration(
        base_params_b,
        "N_h",
        cfg.N_h_values,
        cfg=cfg,
        use_progress=use_progress,
        progress_label="Split scan: N_h",
        num_workers=worker_count,
        num_sequences=actual_num_sequences,
        with_shared_patterns=actual_with_shared_patterns,
        shared_pattern_positions=actual_shared_pattern_positions,
    )

    # 构建标题后缀
    title_parts = []
    if actual_with_shared_patterns:
        title_parts.append(f"with {actual_num_sequences if actual_num_sequences is not None else cfg.get_num_sequences()} sequences (shared patterns)")
    elif actual_num_sequences is not None and actual_num_sequences > 1:
        title_parts.append(f"with {actual_num_sequences} sequences")
    title_suffix = " (" + ", ".join(title_parts) + ")" if title_parts else ""
    
    # 构建文件名后缀
    filename_suffix = ""
    if actual_with_shared_patterns:
        filename_suffix = "_multi_shared"
    elif actual_num_sequences is not None and actual_num_sequences > 1:
        filename_suffix = f"_multi_{actual_num_sequences}"

    plot_figure5(
        results_v_only_T_scan,
        results_uv_T_scan,
        param_name="T",
        param_values=cfg.T_values,
        save_path=output_path / f"figure5a_split{filename_suffix}.png",
        show_plot=show_images,
        title_suffix=title_suffix,
    )

    plot_figure5(
        results_v_only_Nh_scan,
        results_uv_Nh_scan,
        param_name="N_h",
        param_values=cfg.N_h_values,
        save_path=output_path / f"figure5b_split{filename_suffix}.png",
        show_plot=show_images,
        title_suffix=title_suffix,
    )

    summary = output_path / f"results_split_summary{filename_suffix}.txt"
    with summary.open("w", encoding="utf-8") as f:
        f.write("Figure 5 拆分模式对比\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"参数: num_trials={cfg.num_trials}, noise_num={cfg.noise_num}, num_epochs={cfg.num_epochs}\n")
        f.write(f"num_sequences={actual_num_sequences if actual_num_sequences is not None else cfg.get_num_sequences()}\n")
        f.write(f"with_shared_patterns={actual_with_shared_patterns}\n")
        if actual_shared_pattern_positions:
            f.write(f"shared_pattern_positions={actual_shared_pattern_positions}\n")
        f.write("\n")

        f.write("(a) 扫描 T，比较 V-only vs U+V (N=100, M=500)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'T':<8} {'V-only (%)':<15} {'U+V (%)':<15} {'Improvement':<15}\n")
        f.write("-" * 80 + "\n")
        for idx, T_value in enumerate(cfg.T_values):
            v_only = results_v_only_T_scan[idx]["recall_accuracy"] * 100
            uv = results_uv_T_scan[idx]["recall_accuracy"] * 100
            f.write(f"{T_value:<8} {v_only:<15.1f} {uv:<15.1f} {uv - v_only:+.1f}\n")

        f.write("\n(b) 扫描 N_h，比较 V-only vs U+V (N=100, T=70)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'M':<8} {'V-only (%)':<15} {'U+V (%)':<15} {'Improvement':<15}\n")
        f.write("-" * 80 + "\n")
        for idx, N_h_value in enumerate(cfg.N_h_values):
            v_only = results_v_only_Nh_scan[idx]["recall_accuracy"] * 100
            uv = results_uv_Nh_scan[idx]["recall_accuracy"] * 100
            f.write(f"{N_h_value:<8} {v_only:<15.1f} {uv:<15.1f} {uv - v_only:+.1f}\n")

    return (
        results_v_only_T_scan,
        results_uv_T_scan,
        results_v_only_Nh_scan,
        results_uv_Nh_scan,
        output_path,
    )
