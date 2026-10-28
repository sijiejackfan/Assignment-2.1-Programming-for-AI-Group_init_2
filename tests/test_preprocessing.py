
import numpy as np
import pytest
'''
AI Assistance: The implementation was adapted from code suggestions provided by ChatGPT (OpenAI, 2026). AI helped was used for creating some test cases and generate the data for testing.
'''
from numcompute.preprocessing import (
    ColumnSelector,
    SimpleImputer,
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder
)


def test_column_selector():
    data = np.array([
        ["20", "5.5", "Adelaide"],
        ["22", "6.0", "Sydney"]
    ])

    selector = ColumnSelector(
        ["Age", "GPA", "City"], ["Age", "GPA"]
    )

    result = selector.fit_transform(data)

    assert result.shape == (2, 2)
    assert np.array_equal(result, data[:, :2])


def test_column_selector_invalid():
    data = np.array([["20", "Sydney"]])

    selector = ColumnSelector(
        ["Age", "City"], ["GPA"]
    )

    with pytest.raises(ValueError):
        selector.fit(data)


def test_numeric_imputer():
    data = np.array([
        ["20", ""],
        ["", "6.0"]
    ])

    imputer = SimpleImputer(fill_value=0, dtype=float)
    result = imputer.fit_transform(data)

    expected = np.array([
        [20.0, 0.0],
        [0.0, 6.0]
    ])

    assert np.array_equal(result, expected)
    assert result.dtype == float


def test_categorical_imputer():
    data = np.array([
        ["Sydney"],
        [""],
        ["Melbourne"]
    ])

    imputer = SimpleImputer(fill_value="Unknown", dtype=str)
    result = imputer.fit_transform(data)

    assert result[1, 0] == "Unknown"


def test_standard_scaler():
    data = np.array([
        [10.0, 100.0],
        [20.0, 200.0],
        [30.0, 300.0]
    ])

    scaler = StandardScaler()
    result = scaler.fit_transform(data)

    assert np.allclose(np.mean(result, axis=0), [0, 0])
    assert np.allclose(np.std(result, axis=0), [1, 1])


def test_standard_scaler_constant():
    data = np.array([
        [10.0, 5.0],
        [10.0, 6.0],
        [10.0, 7.0]
    ])

    scaler = StandardScaler()
    result = scaler.fit_transform(data)

    assert np.all(result[:, 0] == 0)


def test_minmax_scaler():
    data = np.array([
        [10.0],
        [20.0],
        [30.0]
    ])

    scaler = MinMaxScaler()
    result = scaler.fit_transform(data)

    assert np.allclose(result, [[0], [0.5], [1]])


def test_minmax_constant():
    data = np.array([
        [5.0],
        [5.0],
        [5.0]
    ])

    scaler = MinMaxScaler()
    result = scaler.fit_transform(data)

    assert np.all(result == 0)


def test_onehot_multiple_columns():
    data = np.array([
        ["Adelaide", "Computing"],
        ["Sydney", "Math"],
        ["Melbourne", "Computing"]
    ])

    encoder = OneHotEncoder()
    result = encoder.fit_transform(data)

    assert result.shape == (3, 5)
    assert np.all(np.sum(result, axis=1) == 2)
