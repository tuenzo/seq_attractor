"""Figure 5 reproduction experiments for the sequence attractor network."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from ..core import SequenceAttractorNetwork


@dataclass(frozen=True)
class Figure5Config:
    """Configuration container for the Figure 5 experiments."""

    num_trials: int = 100
    noise_num: int = 10
    num_epochs: int = 500
    T_values: Sequence[int] = (10, 30, 50, 70, 90, 110, 140)
    N_h_values: Sequence[int] = (100, 325, 550, 775, 1000)

    def noise_level(self, N_v: int) -> float:
        return self.noise_num / float(N_v)


def _ensure_output_dir(output_dir: Optional[Path], *, create_timestamp: bool) -> Path:
    base_dir = Path(output_dir) if output_dir is not None else Path("figure5_results")
    if create_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = base_dir / f"figure5_{timestamp}"
    else:
        target_dir = base_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def plot_figure5(
    results_v_only: Sequence[Dict],
    results_uv: Sequence[Dict],
    param_name: str,
    param_values: Iterable[int],
    *,
    save_path: Optional[Path] = None,
    show_plot: bool = True,
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
) -> List[Dict]:
    results: List[Dict] = []
    total = len(varying_values)

    for idx, value in enumerate(varying_values):
        params = dict(base_params)
        params[varying_key] = value
        success_count = 0

        for trial in range(config.num_trials):
            network = SequenceAttractorNetwork(**params)
            network.train(
                x=None,
                num_epochs=config.num_epochs,
                verbose=False,
                seed=trial,
                V_only=v_only,
            )

            noise_level = config.noise_level(params["N_v"])
            robustness = network.test_robustness(
                noise_levels=np.array([noise_level]),
                num_trials=1,
                verbose=False,
            )

            if robustness[0] > 0.5:
                success_count += 1

        success_rate = success_count / config.num_trials
        results.append(
            {
                varying_key: value,
                "recall_accuracy": success_rate,
                **{k: v for k, v in params.items() if k != varying_key},
            }
        )

        if progress is not None:
            progress(idx + 1, total, value)

    return results


def run_figure5_experiments(
    *,
    config: Figure5Config | None = None,
    output_dir: Optional[Path] = None,
    create_timestamp_dir: bool = True,
    show_images: bool = False,
    progress_callback: Optional[Callable[[str, int, int, int], None]] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], Path]:
    """Execute the Figure 5 experiments and optionally store artefacts."""

    cfg = config or Figure5Config()
    output_path = _ensure_output_dir(output_dir, create_timestamp=create_timestamp_dir)

    def report(stage: str, current: int, total: int, value: int) -> None:
        if progress_callback is not None:
            progress_callback(stage, current, total, value)

    base_params_a = {"N_v": 100, "N_h": 500, "eta": 0.01, "kappa": 1.0}
    base_params_b = {"N_v": 100, "T": 70, "eta": 0.01, "kappa": 1.0}

    results_v_only_a = _run_single_configuration(
        {**base_params_a},
        "T",
        cfg.T_values,
        config=cfg,
        v_only=True,
        progress=None if progress_callback is None else lambda current, total, value: report("V_only_T", current, total, value),
    )

    results_uv_a = _run_single_configuration(
        {**base_params_a},
        "T",
        cfg.T_values,
        config=cfg,
        v_only=False,
        progress=None if progress_callback is None else lambda current, total, value: report("UV_T", current, total, value),
    )

    results_v_only_b = _run_single_configuration(
        {**base_params_b},
        "N_h",
        cfg.N_h_values,
        config=cfg,
        v_only=True,
        progress=None if progress_callback is None else lambda current, total, value: report("V_only_N_h", current, total, value),
    )

    results_uv_b = _run_single_configuration(
        {**base_params_b},
        "N_h",
        cfg.N_h_values,
        config=cfg,
        v_only=False,
        progress=None if progress_callback is None else lambda current, total, value: report("UV_N_h", current, total, value),
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


# ========= 新增：拆分模式版本（符合“V-only 扫 T；UV 扫 N_h”对比） =========

def _generate_sequence_with_single_repetition(T: int, N_v: int, seed: int, repeat_pos: Optional[int] = None) -> np.ndarray:
    """
    生成一个基础随机序列，并在 repeat_pos 位置引入单步重复（复制首帧），最后一帧仍等于首帧。
    """
    rng = np.random.RandomState(seed)
    x = np.sign(rng.randn(T, N_v))
    x[x == 0] = 1
    # 确保中间没有与首帧重复（简化保证唯一）
    for t in range(1, T - 1):
        while np.all(x[t, :] == x[0, :]):
            x[t, :] = np.sign(rng.randn(N_v))
            x[t, x[t, :] == 0] = 1
    # 应用单步重复
    pos = repeat_pos if (repeat_pos is not None and 0 <= repeat_pos < T - 1) else (T // 2)
    if pos == T - 1:
        pos = T // 2
    x[pos, :] = x[0, :]
    # 闭环
    x[T - 1, :] = x[0, :]
    return x


def run_figure5_experiments_split_modes(
    *,
    config: Figure5Config | None = None,
    output_dir: Optional[Path] = None,
    create_timestamp_dir: bool = True,
    show_images: bool = False,
    with_repetition: bool = False,
    repetition_position: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict], Path]:
    """
    拆分模式对比：
    - 图(a)：仅训练 V，固定 N_h=500，扫描 T
    - 图(b)：训练 U+V，固定 T=70，扫描 N_h
    可选：with_repetition=True 时，在训练序列中引入一个单步重复。
    返回：(results_v_only_T_scan, results_uv_Nh_scan, output_path)
    """
    cfg = config or Figure5Config()
    output_path = _ensure_output_dir(output_dir, create_timestamp=create_timestamp_dir)

    base_params_a = {"N_v": 100, "N_h": 500, "eta": 0.01, "kappa": 1.0}
    base_params_b = {"N_v": 100, "T": 70, "eta": 0.01, "kappa": 1.0}

    results_v_only_T_scan: List[Dict] = []
    # 图(a)：V-only 扫描 T
    for T_value in cfg.T_values:
        params = {**base_params_a, "T": int(T_value)}
        success_count = 0
        for trial in range(cfg.num_trials):
            net = SequenceAttractorNetwork(**params)
            if with_repetition:
                x_train = _generate_sequence_with_single_repetition(
                    T=params["T"], N_v=params["N_v"], seed=trial, repeat_pos=repetition_position
                )
            else:
                x_train = None
            net.train(
                x=x_train,
                num_epochs=cfg.num_epochs,
                verbose=False,
                seed=None if x_train is not None else trial,
                V_only=True,
            )
            noise_level = cfg.noise_level(params["N_v"])
            robustness = net.test_robustness(
                noise_levels=np.array([noise_level]),
                num_trials=1,
                verbose=False,
            )
            if robustness[0] > 0.5:
                success_count += 1
        success_rate = success_count / cfg.num_trials
        results_v_only_T_scan.append(
            {
                "T": params["T"],
                "recall_accuracy": success_rate,
                "N_v": params["N_v"],
                "N_h": params["N_h"],
                "with_repetition": with_repetition,
            }
        )

    results_uv_Nh_scan: List[Dict] = []
    # 图(b)：U+V 扫描 N_h
    for N_h_value in cfg.N_h_values:
        params = {**base_params_b, "N_h": int(N_h_value)}
        success_count = 0
        for trial in range(cfg.num_trials):
            net = SequenceAttractorNetwork(**params)
            if with_repetition:
                x_train = _generate_sequence_with_single_repetition(
                    T=params["T"], N_v=params["N_v"], seed=trial, repeat_pos=repetition_position
                )
            else:
                x_train = None
            net.train(
                x=x_train,
                num_epochs=cfg.num_epochs,
                verbose=False,
                seed=None if x_train is not None else trial,
                V_only=False,
            )
            noise_level = cfg.noise_level(params["N_v"])
            robustness = net.test_robustness(
                noise_levels=np.array([noise_level]),
                num_trials=1,
                verbose=False,
            )
            if robustness[0] > 0.5:
                success_count += 1
        success_rate = success_count / cfg.num_trials
        results_uv_Nh_scan.append(
            {
                "N_h": params["N_h"],
                "recall_accuracy": success_rate,
                "N_v": params["N_v"],
                "T": params["T"],
                "with_repetition": with_repetition,
            }
        )

    # 绘制两张图：分别只有一条曲线
    plt.figure(figsize=(8, 6))
    plt.plot(
        list(cfg.T_values),
        np.array([r["recall_accuracy"] for r in results_v_only_T_scan]) * 100.0,
        "o-",
        linewidth=2,
        markersize=8,
        label="V-only (scan T)",
        color="#E74C3C",
    )
    plt.xlabel("Sequence Length (T)", fontsize=14, fontweight="bold")
    plt.ylabel("Successful Retrievals (%)", fontsize=14, fontweight="bold")
    title_suffix = " with single-step repetition" if with_repetition else ""
    plt.title(f"(a) Fixed N=100, M=500{title_suffix}", fontsize=16, fontweight="bold", pad=20)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.ylim([-5, 105])
    plt.yticks(np.arange(0, 101, 20))
    plt.tight_layout()
    plt.savefig(output_path / ("figure5a_split.png" if not with_repetition else "figure5a_split_repetition.png"),
                dpi=300, bbox_inches="tight")
    if show_images:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(
        list(cfg.N_h_values),
        np.array([r["recall_accuracy"] for r in results_uv_Nh_scan]) * 100.0,
        "s-",
        linewidth=2,
        markersize=8,
        label="U+V (scan N_h)",
        color="#3498DB",
    )
    plt.xlabel("Number of Hidden Neurons (M)", fontsize=14, fontweight="bold")
    plt.ylabel("Successful Retrievals (%)", fontsize=14, fontweight="bold")
    plt.title(f"(b) Fixed N=100, T=70{title_suffix}", fontsize=16, fontweight="bold", pad=20)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.ylim([-5, 105])
    plt.yticks(np.arange(0, 101, 20))
    plt.tight_layout()
    plt.savefig(output_path / ("figure5b_split.png" if not with_repetition else "figure5b_split_repetition.png"),
                dpi=300, bbox_inches="tight")
    if show_images:
        plt.show()
    else:
        plt.close()

    # 写摘要
    summary = output_path / ("results_split_summary.txt" if not with_repetition else "results_split_summary_repetition.txt")
    with summary.open("w", encoding="utf-8") as f:
        f.write("Figure 5 拆分模式对比\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"参数: num_trials={cfg.num_trials}, noise_num={cfg.noise_num}, num_epochs={cfg.num_epochs}\n")
        f.write(f"with_repetition={with_repetition}, repetition_position={repetition_position}\n\n")
        f.write("(a) V-only 扫描 T (N=100, M=500)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'T':<8} {'V-only (%)':<15}\n")
        f.write("-" * 80 + "\n")
        for r in results_v_only_T_scan:
            f.write(f"{r['T']:<8} {r['recall_accuracy']*100:<15.1f}\n")
        f.write("\n(b) U+V 扫描 M (N=100, T=70)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'M':<8} {'U+V (%)':<15}\n")
        f.write("-" * 80 + "\n")
        for r in results_uv_Nh_scan:
            f.write(f"{r['N_h']:<8} {r['recall_accuracy']*100:<15.1f}\n")

    return results_v_only_T_scan, results_uv_Nh_scan, output_path
