import numpy as np

from numcompute_stream.metrics import StreamingClassificationMetrics


def test_streaming_confusion_matrix():
    metrics = StreamingClassificationMetrics()
    metrics.update([0, 1, 1, 0], [0, 1, 0, 1])
    result = metrics.result()
    assert result["confusion_matrix"].tolist() == [[1, 1], [1, 1]]
    assert np.isclose(result["accuracy"], 0.5)


def test_streaming_cumulative_metrics():
    metrics = StreamingClassificationMetrics()
    metrics.update([0, 1], [0, 1])
    metrics.update([0, 1], [1, 1])
    result = metrics.result()
    assert np.isclose(result["accuracy"], 0.75)
    assert np.isclose(result["precision"], 2 / 3)
    assert np.isclose(result["recall"], 1.0)


def test_streaming_metrics_reset():
    metrics = StreamingClassificationMetrics()
    metrics.update([0, 1], [0, 0])
    metrics.reset()
    metrics.update([0, 1], [0, 1])
    assert np.isclose(metrics.result()["accuracy"], 1.0)
