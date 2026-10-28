"""

Tests for sorting and searching.

"""

import numpy as np
import pytest

from numcompute.sort_search import (
    stable_sort, quick_sort, merge_sort, topk, quickselect
)


def test_stable_sort():
    assert stable_sort([4, 2, 1, 3]).tolist() == [1, 2, 3, 4]


def test_quick_sort():
    assert quick_sort([4, 2, 2, 1]).tolist() == [1, 2, 2, 4]


def test_merge_sort():
    assert merge_sort([5, 1, 3, 2]).tolist() == [1, 2, 3, 5]


def test_sort_empty():
    assert quick_sort([]).tolist() == []
    assert merge_sort([]).tolist() == []


def test_sort_single_value():
    assert stable_sort([5]).tolist() == [5]
    assert quick_sort([5]).tolist() == [5]
    assert merge_sort([5]).tolist() == [5]


def test_topk_largest():
    assert topk([4, 1, 7, 3], 2).tolist() == [7, 4]


def test_topk_smallest():
    assert topk([4, 1, 7, 3], 2, largest=False).tolist() == [1, 3]


def test_topk_zero():
    assert topk([4, 1, 7], 0).tolist() == []


def test_topk_all_values():
    assert topk([3, 1, 2], 3).tolist() == [3, 2, 1]


def test_topk_invalid_k():
    with pytest.raises(ValueError):
        topk([1, 2, 3], 4)


def test_topk_invalid_type():
    with pytest.raises(TypeError):
        topk([1, 2, 3], 1.5)


def test_topk_wrong_shape():
    with pytest.raises(ValueError):
        topk([[1, 2], [3, 4]], 2)


def test_quickselect():
    assert quickselect([7, 2, 5, 1, 4], 2) == 4


def test_quickselect_first_value():
    assert quickselect([5, 2, 3], 0) == 2


def test_quickselect_duplicates():
    assert quickselect([3, 1, 3, 2], 2) == 3


def test_quickselect_invalid_k():
    with pytest.raises(ValueError):
        quickselect([1, 2, 3], 3)


def test_quickselect_invalid_type():
    with pytest.raises(TypeError):
        quickselect([1, 2, 3], 1.5)


def test_quickselect_wrong_shape():
    with pytest.raises(ValueError):
        quickselect([[1, 2], [3, 4]], 1)


def test_sort_numpy_array():
    values = np.array([3, 1, 2])
    assert quick_sort(values).tolist() == [1, 2, 3]
