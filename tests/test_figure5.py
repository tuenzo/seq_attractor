import numpy as np
import pytest

from src.experiments.figure5 import Figure5Config, plot_figure5, run_figure5_experiments


class TestFigure5Experiments:
    def test_run_figure5_experiments_minimal(self, tmp_path):
        config = Figure5Config(
            num_trials=1,
            noise_num=1,
            num_epochs=1,
            T_values=(10,),
            N_h_values=(50,),
        )

        output_dir = tmp_path / "figure5"
        (
            results_v_only_a,
            results_uv_a,
            results_v_only_b,
            results_uv_b,
            output_path,
        ) = run_figure5_experiments(
            config=config,
            output_dir=output_dir,
            create_timestamp_dir=False,
            show_images=False,
        )

        assert len(results_v_only_a) == 1
        assert len(results_uv_a) == 1
        assert len(results_v_only_b) == 1
        assert len(results_uv_b) == 1
        assert output_path.exists()
        assert (output_path / "figure5a.png").exists()
        assert (output_path / "figure5b.png").exists()
        assert (output_path / "results_summary.txt").exists()

    def test_plot_figure5_creates_file(self, tmp_path):
        results_v_only = [{"recall_accuracy": 0.5}, {"recall_accuracy": 0.75}]
        results_uv = [{"recall_accuracy": 0.6}, {"recall_accuracy": 0.9}]
        param_values = np.array([10, 20])

        output_file = tmp_path / "plot.png"
        plot_figure5(
            results_v_only,
            results_uv,
            param_name="T",
            param_values=param_values,
            save_path=output_file,
            show_plot=False,
        )

        assert output_file.exists()
