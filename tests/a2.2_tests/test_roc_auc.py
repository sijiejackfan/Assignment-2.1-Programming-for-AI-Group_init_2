
import numpy as np
import pytest

from numcompute_stream.metrics import roc_curve, auc, roc_auc_score


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
