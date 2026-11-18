import numpy as np
import pytest

from numcompute_stream.tree import (
    gini, entropy, information_gain, DecisionTreeClassifier
)


def test_gini_and_entropy():
    y = np.array([0, 0, 1, 1])
    assert np.isclose(gini(y), 0.5)
    assert np.isclose(entropy(y), 1.0)


def test_information_gain():
    y = np.array([0, 0, 1, 1])
    mask = np.array([True, True, False, False])
    assert np.isclose(information_gain(y, mask, "gini"), 0.5)


def test_tree_fit_predict():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X, y)
    assert np.array_equal(tree.predict(X), y)


def test_tree_cold_start_and_retrain():
    tree = DecisionTreeClassifier(buffer_size=4)
    tree.partial_fit([[0], [1]], [0, 0])
    assert tree.root is None
    assert tree._buffer_count == 2
    tree.partial_fit([[2], [3]], [1, 1])
    assert tree.root is not None
    assert tree._buffer_count == 0
    assert tree.retrain_count == 1


def test_tree_keeps_active_model():
    tree = DecisionTreeClassifier(buffer_size=4)
    tree.partial_fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    before = tree.predict([[0], [3]])
    tree.partial_fit([[10]], [0])
    after = tree.predict([[0], [3]])
    assert np.array_equal(before, after)


def test_tree_retrains_with_new_buffer():
    tree = DecisionTreeClassifier(buffer_size=4)
    X = np.array([[0], [1], [2], [3]])
    tree.partial_fit(X, [0, 0, 1, 1])
    tree.partial_fit(X, [1, 1, 0, 0])
    assert np.array_equal(tree.predict(X), [1, 1, 0, 0])
    assert tree.retrain_count == 2


def test_tree_invalid_buffer_size():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(buffer_size=0)
    with pytest.raises(TypeError):
        DecisionTreeClassifier(buffer_size=1.5)


def test_tree_predict_proba():
    tree = DecisionTreeClassifier(max_depth=0)

    X = np.array([[0], [1], [2]])
    y = np.array([0, 1, 1])

    tree.fit(X, y)
    result = tree.predict_proba([[3]])

    assert result.shape == (1, 2)
    assert np.allclose(result[0], [1/3, 2/3])


def test_tree_predict_proba_not_fitted():
    tree = DecisionTreeClassifier()

    with pytest.raises(ValueError):
        tree.predict_proba([[1]])

