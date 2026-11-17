import numpy as np
import pytest

from numcompute_stream.preprocessing import (
    ColumnSelector, SimpleImputer, StandardScaler, MinMaxScaler, OneHotEncoder
)


def test_column_selector_partial_fit():
    selector = ColumnSelector(["Age", "City"], ["Age"])
    selector.partial_fit(np.array([["20", "Sydney"]]))
    result = selector.transform(np.array([["25", "Adelaide"]]))
    assert result.tolist() == [["25"]]


def test_mean_imputer_partial_fit():
    imputer = SimpleImputer(strategy="mean", dtype=float)
    imputer.partial_fit(np.array([["2"], [""]]))
    imputer.partial_fit(np.array([["4"], ["6"]]))
    result = imputer.transform(np.array([[""], ["8"]]))
    assert np.allclose(result, [[4], [8]])


def test_scaler_partial_fit():
    scaler = StandardScaler()
    scaler.partial_fit([[1], [3]])
    scaler.partial_fit([[5]])
    assert np.allclose(scaler.mean, [3])
    assert np.allclose(scaler.std, [np.sqrt(8 / 3)])
    assert np.allclose(scaler.transform([[3]]), [[0]])


def test_scaler_fit_resets_old_data():
    scaler = StandardScaler()
    scaler.partial_fit([[1], [3]])
    scaler.fit([[10], [20]])
    assert np.allclose(scaler.mean, [15])


def test_minmax_partial_fit():
    scaler = MinMaxScaler()
    scaler.partial_fit([[10], [20]])
    scaler.partial_fit([[0], [30]])
    assert np.allclose(scaler.transform([[0], [15], [30]]), [[0], [0.5], [1]])


def test_onehot_new_category():
    encoder = OneHotEncoder()
    encoder.partial_fit(np.array([["A"], ["B"]]))
    encoder.partial_fit(np.array([["C"]]))
    result = encoder.transform(np.array([["A"], ["C"]]))
    assert result.shape == (2, 3)
    assert np.all(result.sum(axis=1) == 1)


def test_onehot_unknown_category():
    encoder = OneHotEncoder()
    encoder.fit(np.array([["A"], ["B"]]))
    result = encoder.transform(np.array([["C"]]))
    assert result.tolist() == [[0, 0]]


def test_minmax_wrong_feature_count():
    scaler = MinMaxScaler()
    scaler.partial_fit([[1, 2]])
    with pytest.raises(ValueError):
        scaler.partial_fit([[3]])
