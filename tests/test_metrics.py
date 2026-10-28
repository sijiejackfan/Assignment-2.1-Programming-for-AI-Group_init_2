
import numpy as np
from numcompute.metrics import (
    accuracy, precision, recall, f1, confusion_matrix, mse
)


def test_accuracy():
    assert accuracy([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0


def test_accuracy_with_errors():
    assert accuracy([1, 0, 1, 0], [1, 1, 0, 0]) == 0.5


def test_precision():
    assert precision([1, 0, 1, 0], [1, 0, 0, 0]) == 1.0


def test_precision_with_false_positives():
    assert np.isclose(
        precision([1, 0, 1, 0], [1, 1, 1, 0]), 2 / 3
    )


def test_recall():
    assert recall([1, 1, 0, 0], [1, 1, 1, 0]) == 1.0


def test_recall_with_false_negatives():
    assert np.isclose(
        recall([1, 1, 1, 0], [1, 0, 0, 0]), 1 / 3
    )


def test_f1():
    assert f1([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0


def test_f1_with_errors():
    assert f1([1, 1, 0, 0], [1, 0, 1, 0]) == 0.5


def test_confusion_matrix():
    result = confusion_matrix([1, 0, 1, 0], [1, 0, 1, 0])
    assert result.tolist() == [[2, 0], [0, 2]]


def test_confusion_matrix_with_errors():
    result = confusion_matrix([1, 0, 1, 0], [1, 1, 0, 0])
    assert result.tolist() == [[1, 1], [1, 1]]


def test_mse():
    assert mse([1, 2, 3], [1, 2, 3]) == 0.0


def test_mse_with_errors():
    assert np.isclose(mse([1, 2, 3], [2, 2, 5]), 5 / 3)
