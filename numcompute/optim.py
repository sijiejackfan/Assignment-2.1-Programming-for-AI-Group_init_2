"""
** Optimization
  * Calculate gradients using finite differences
  * Calculate the Jacobian of a function

** Sources:
  * https://numpy.org/doc/stable/reference/generated/numpy.gradient.html
  * https://cs357.cs.illinois.edu/textbook/notes/finite-difference.html
"""

import numpy as np


def gradient(func, x, h=1e-5):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)

    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()

        x1[i] += h
        x2[i] -= h

        grad[i] = (func(x1) - func(x2)) / (2 * h)

    return grad


def jacobian(func, x, h=1e-5):
    x = np.asarray(x, dtype=float)
    y = np.atleast_1d(func(x))

    jac = np.zeros((len(y), len(x)))

    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()

        x1[i] += h
        x2[i] -= h

        jac[:, i] = (
            np.atleast_1d(func(x1)) - np.atleast_1d(func(x2))
        ) / (2 * h)

    return jac
