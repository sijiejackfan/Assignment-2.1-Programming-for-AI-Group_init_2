"""Tests for utility functions"""

import numpy as np
import pytest

from numcompute.utils import (
    euclidean_distance,
    manhattan_distance,
    cosine_similarity,
    sigmoid,
    relu,
    softmax,
    logsumexp,
    top_k,
    batch_data
)


def test_euclidean_distance():
    assert euclidean_distance([1, 2], [4, 6]) == 5


def test_manhattan_distance():
    assert manhattan_distance([1, 2], [4, 6]) == 7


def test_cosine_similarity():
    assert np.isclose(cosine_similarity([1, 0], [1, 0]), 1)


def test_cosine_similarity_zero():
    assert cosine_similarity([0, 0], [1, 2]) == 0.0


def test_sigmoid():
    assert np.allclose(sigmoid([0]), [0.5])


def test_relu():
    assert np.array_equal(relu([-2, 0, 3]), [0, 0, 3])


def test_softmax():
    result = softmax([1, 2, 3])
    assert np.isclose(np.sum(result), 1)
    assert result[2] > result[1] > result[0]


def test_logsumexp():
    assert np.isclose(logsumexp([0, 0]), np.log(2))


def test_top_k():
    assert top_k([4, 1, 7, 3], 2).tolist() == [7, 4]


def test_top_k_smallest():
    assert top_k([4, 1, 7, 3], 2, largest=False).tolist() == [1, 3]


def test_top_k_invalid():
    with pytest.raises(ValueError):
        top_k([1, 2, 3], 4)


def test_batch_data():
    result = list(batch_data([1, 2, 3, 4, 5], 2))
    assert [batch.tolist() for batch in result] == [
        [1, 2], [3, 4], [5]
    ]


def test_batch_data_invalid():
    with pytest.raises(ValueError):
        list(batch_data([1, 2, 3], 0))
