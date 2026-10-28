
import numpy as np
from numcompute.rank import rank, percentile


def test_rank():
    assert rank([30, 10, 20]).tolist() == [3, 1, 2]


def test_rank_with_duplicates():
    assert rank([10, 20, 20, 30]).tolist() == [1, 2.5, 2.5, 4]


def test_rank_same_values():
    assert rank([5, 5, 5]).tolist() == [2, 2, 2]


def test_rank_empty():
    assert rank([]).tolist() == []


def test_percentile():
    assert percentile([10, 20, 30, 40], 50) == 25


def test_percentile_minimum():
    assert percentile([10, 20, 30, 40], 0) == 10


def test_percentile_maximum():
    assert percentile([10, 20, 30, 40], 100) == 40


def test_percentile_with_nan():
    result = percentile([10, 20, np.nan, 40], 50)
    assert result == 25
