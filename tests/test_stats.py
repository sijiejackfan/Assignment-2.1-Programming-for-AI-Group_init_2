
import numpy as np
from numcompute.stats import (
    mean, median, std, minimum, maximum, histogram, quantile
)


def test_mean():
    assert mean([1, 2, 3, 4]) == 2.5


def test_mean_with_nan():
    data = np.array([1, np.nan, 3])
    assert mean(data) == 2.0


def test_median():
    assert median([1, 3, 5, 10]) == 4.0


def test_median_axis():
    data = np.array([[1, 3, 5], [2, 4, 10]])
    assert median(data, axis=1).tolist() == [3, 4]


def test_std():
    assert np.isclose(std([1, 2, 3]), np.sqrt(2 / 3))


def test_std_with_nan():
    data = np.array([1, np.nan, 3])
    assert std(data) == 1.0


def test_minimum():
    assert minimum([5, 2, 8, 1]) == 1


def test_minimum_axis():
    data = np.array([[1, 5], [3, 2]])
    assert minimum(data, axis=0).tolist() == [1, 2]


def test_maximum():
    assert maximum([5, 2, 8, 1]) == 8


def test_maximum_with_nan():
    data = np.array([5, np.nan, 8, 1])
    assert maximum(data) == 8


def test_histogram():
    counts, edges = histogram([0, 1, 2, 3], bins=2)
    assert counts.tolist() == [2, 2]
    assert np.allclose(edges, [0, 1.5, 3])


def test_histogram_with_nan():
    counts, edges = histogram([0, 1, np.nan, 2, 3], bins=2)
    assert counts.tolist() == [2, 2]


def test_quantile():
    assert quantile([10, 20, 30, 40], 0.5) == 25


def test_quantile_with_nan():
    data = np.array([10, 20, np.nan, 40])
    result = quantile(data, [0, 0.5, 1])
    assert result.tolist() == [10, 20, 40]
