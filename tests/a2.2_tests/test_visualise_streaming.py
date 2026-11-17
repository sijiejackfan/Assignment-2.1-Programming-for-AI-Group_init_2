import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from numcompute_stream.visualise import (
    plot_metric_over_time, compare_models, plot_predictions_vs_ground_truth
)


history = [
    {"chunk": 1, "accuracy": None},
    {"chunk": 2, "accuracy": 0.75},
    {"chunk": 3, "accuracy": 0.8},
]


def test_plot_metric_over_time(tmp_path):
    path = tmp_path / "accuracy.png"
    fig = plot_metric_over_time(history, save_path=path)
    assert path.exists()
    assert len(fig.axes) == 1
    plt.close(fig)


def test_compare_models(tmp_path):
    path = tmp_path / "comparison.png"
    fig = compare_models({"A": history, "B": history}, save_path=path)
    assert path.exists()
    assert len(fig.axes[0].lines) == 2
    plt.close(fig)


def test_plot_predictions_vs_ground_truth(tmp_path):
    path = tmp_path / "predictions.png"
    fig = plot_predictions_vs_ground_truth([0, 1, 1], [0, 0, 1], save_path=path)
    assert path.exists()
    assert len(fig.axes[0].lines) == 2
    plt.close(fig)


def test_visualise_invalid_input():
    with pytest.raises(ValueError):
        plot_metric_over_time([])
    with pytest.raises(ValueError):
        plot_predictions_vs_ground_truth([0, 1], [0])
