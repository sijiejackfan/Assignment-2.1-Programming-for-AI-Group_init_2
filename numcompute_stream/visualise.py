'''
### `visualise.py`
- Provide reusable plotting functions using `matplotlib`
- Must be usable across scripts, demos, or pipeline logs
- Required plots:
  - `plot_metric_over_time(metric_values, title, ylabel)`: Plot a metric (e.g., accuracy) across chunks
  - `compare_models(metric1, metric2, labels)`: Compare two models on streaming metrics (e.g., trees with different `buffer_size` values)
  - `plot_predictions_vs_ground_truth(y_true, y_pred)`: Visualise predictions vs. actuals on latest chunk
- Should support options for saving to file or inline display
Source: 
https://matplotlib.org/stable/users/explain/quick_start.html
https://matplotlib.org/stable/gallery/index.html
'''

import numpy as np
import matplotlib.pyplot as plt


def plot_metric_over_time(history, metric="accuracy", save_path=None):
    """
    Plot a streaming metric across data chunks.
    """
    if not history:
        raise ValueError("History cannot be empty")

    chunks = [record["chunk"] for record in history]

    values = [
        np.nan if record[metric] is None else record[metric]
        for record in history
    ]

    fig, ax = plt.subplots()

    ax.plot(chunks, values, marker="o")

    ax.set_xlabel("Chunk")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Streaming " + metric.replace("_", " ").title())

    ax.grid(True, alpha=0.3)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def compare_models(histories, metric="accuracy", save_path=None):
    """
    Compare streaming metrics from multiple models.
    """
    if not histories:
        raise ValueError("Histories cannot be empty")

    fig, ax = plt.subplots()

    for name, history in histories.items():
        chunks = [record["chunk"] for record in history]

        values = [
            np.nan if record[metric] is None else record[metric]
            for record in history
        ]

        ax.plot(chunks, values, marker="o", label=name)

    ax.set_xlabel("Chunk")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Model Comparison")

    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_predictions_vs_ground_truth(
    y_true, y_pred, save_path=None
):
    """
    Compare predicted and actual binary class labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")

    if y_true.shape != y_pred.shape:
        raise ValueError("Inputs must have the same shape")

    if y_true.size == 0:
        raise ValueError("Inputs cannot be empty")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("Labels must be 0 or 1")

    if not np.all(np.isin(y_pred, [0, 1])):
        raise ValueError("Predictions must be 0 or 1")

    samples = np.arange(len(y_true))

    fig, ax = plt.subplots()

    ax.plot(
        samples, y_true,
        marker="o",
        label="Ground Truth"
    )

    ax.plot(
        samples, y_pred,
        marker="x",
        linestyle="--",
        label="Predictions"
    )

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Class Label")
    ax.set_title("Predictions vs Ground Truth")

    ax.set_yticks([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig
