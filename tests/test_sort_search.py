"""Tests for sorting and searching."""

import pytest
from numcompute.sort_search import (
    stable_sort, quick_sort, merge_sort, topk,
    top_p, quickselect, binary_search
)


def test_stable_sort():
    assert stable_sort([3, 1, 2]).tolist() == [1, 2, 3]


def test_quick_sort():
    assert quick_sort([4, 2, 2, 1]).tolist() == [1, 2, 2, 4]


def test_merge_sort():
    assert merge_sort([5, 2, 4, 1]).tolist() == [1, 2, 4, 5]


def test_topk():
    assert topk([4, 1, 7, 3], 2).tolist() == [7, 4]


def test_topk_smallest():
    assert topk([4, 1, 7, 3], 2, largest=False).tolist() == [1, 3]


def test_top_p():
    assert top_p([1, 2, 3, 4, 5], 0.4).tolist() == [5, 4]


def test_top_p_invalid():
    with pytest.raises(ValueError):
        top_p([1, 2, 3], 1.5)


def test_quickselect():
    assert quickselect([7, 2, 5, 1, 4], 2) == 4


def test_quickselect_duplicates():
    assert quickselect([3, 1, 3, 2], 2) == 3


def test_binary_search_found():
    assert binary_search([1, 3, 5, 7], 5) == (2, True)


def test_binary_search_not_found():
    assert binary_search([1, 3, 5, 7], 4) == (2, False)


def test_sort_empty():
    assert quick_sort([]).tolist() == []
    assert merge_sort([]).tolist() == []


def test_topk_zero():
    assert topk([1, 2, 3], 0).tolist() == []


def test_topk_invalid():
    with pytest.raises(ValueError):
        topk([1, 2, 3], 4)


def test_quickselect_invalid():
    with pytest.raises(ValueError):
        quickselect([1, 2, 3], 3)


def test_binary_search_wrong_shape():
    with pytest.raises(ValueError):
        binary_search([[1, 2], [3, 4]], 2)
