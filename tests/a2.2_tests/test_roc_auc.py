
import numpy as np
import pytest

from numcompute_stream.metrics import roc_curve, auc, roc_auc_score
from numcompute_stream.metrics import StreamingClassificationMetrics


def test_auc_perfect():
    result = roc_auc_score(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9]
    )
    assert result == 1.0


def test_auc_reversed():
    result = roc_auc_score(
        [0, 0, 1, 1],
        [0.9, 0.8, 0.2, 0.1]
    )
    assert result == 0.0


def test_auc_mixed():
    result = roc_auc_score(
        [0, 0, 1, 1],
        [0.1, 0.4, 0.35, 0.8]
    )
    assert np.isclose(result, 0.75)


def test_auc_ties():
    result = roc_auc_score(
        [0, 1, 0, 1],
        [0.5, 0.5, 0.5, 0.5]
    )
    assert result == 0.5


def test_roc_curve_endpoints():
    fpr, tpr, thresholds = roc_curve(
        [0, 1], [0.2, 0.8]
    )

    assert fpr.tolist() == [0, 0, 1]
    assert tpr.tolist() == [0, 1, 1]
    assert len(thresholds) == 3


def test_auc_one_class():
    with pytest.raises(ValueError):
        roc_auc_score([1, 1], [0.8, 0.5])


def test_auc_bad_input():
    with pytest.raises(ValueError):
        roc_auc_score([0, 1], [0.1, np.nan])


def test_streaming_auc():
    metrics = StreamingClassificationMetrics()
    metrics.update([0, 0], [0, 1], [0.1, 0.4])
    metrics.update([1, 1], [0, 1], [0.35, 0.8])
    assert np.isclose(metrics.result()["auc"], 0.75)


def test_streaming_auc_one_class():
    metrics = StreamingClassificationMetrics()
    metrics.update([0, 0], [0, 0], [0.1, 0.2])
    assert metrics.result()["auc"] is None


def test_streaming_auc_reset():
    metrics = StreamingClassificationMetrics()
    metrics.update([0, 1], [0, 1], [0.2, 0.9])
    metrics.reset()
    assert metrics.result()["auc"] is None


def test_streaming_auc_without_scores():
    metrics = StreamingClassificationMetrics()
    metrics.update([0, 1], [0, 1])
    result = metrics.result()
    assert result["auc"] is None
    assert result["accuracy"] == 1.0


def test_streaming_auc_invalid_score_keeps_state():
    metrics = StreamingClassificationMetrics()
    metrics.update([0, 1], [0, 1], [0.1, 0.9])
    with pytest.raises(ValueError):
        metrics.update([0, 1], [0, 1], [0.2, np.nan])
    assert metrics.matrix.sum() == 2
    assert metrics.result()["auc"] == 1.0
