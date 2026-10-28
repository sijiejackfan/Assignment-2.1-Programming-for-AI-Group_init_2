"""Tests for gradient and Jacobian calculations"""

from numcompute.optim import gradient, jacobian
import numpy as np


def test_gradient():
    result = gradient(lambda x: x[0]**2 + x[1]**2, [3, 4])
    assert np.allclose(result, [6, 8])


def test_gradient_linear():
    result = gradient(lambda x: 2*x[0] + 3*x[1], [1, 2])
    assert np.allclose(result, [2, 3])


def test_gradient_at_zero():
    result = gradient(lambda x: x[0]**2, [0])
    assert np.allclose(result, [0])


def test_jacobian():
    result = jacobian(
        lambda x: np.array([x[0]**2, x[1]**2]),
        [3, 4]
    )
    assert np.allclose(result, [[6, 0], [0, 8]])


def test_jacobian_linear():
    result = jacobian(
        lambda x: np.array([x[0] + x[1], x[0] - x[1]]),
        [2, 3]
    )
    assert np.allclose(result, [[1, 1], [1, -1]])


def test_jacobian_single_output():
    result = jacobian(lambda x: x[0]**2 + x[1]**2, [3, 4])
    assert np.allclose(result, [[6, 8]])


def test_gradient_empty():
    result = gradient(lambda x: np.sum(x), [])
    assert result.tolist() == []


def test_jacobian_empty_input():
    result = jacobian(lambda x: np.array([5.0, 10.0]), [])
    assert result.shape == (2, 0)
