"""

** Ranking and Percentiles
  * Rank values and handle ties
  * Calculate percentiles

** Sources:
  * https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
  * https://numpy.org/doc/stable/reference/generated/numpy.percentile.html

"""

import numpy as np


def rank(data, method="average"):
    arr = np.asarray(data)

    if arr.ndim != 1:
        raise ValueError("data must be a 1D array")

    if method not in ("average", "dense", "ordinal"):
        raise ValueError("invalid ranking method")

    order = np.argsort(arr, kind="stable")
    ranks = np.empty(len(arr), dtype=float)
    sorted_data = arr[order]

    i = 0
    dense_rank = 1

    while i < len(arr):
        j = i + 1

        while j < len(arr) and sorted_data[j] == sorted_data[i]:
            j += 1

        if method == "average":
            ranks[order[i:j]] = (i + 1 + j) / 2
        elif method == "dense":
            ranks[order[i:j]] = dense_rank
            dense_rank += 1
        else:
            ranks[order[i:j]] = np.arange(i + 1, j + 1)

        i = j

    return ranks


def percentile(data, q, interpolation="linear"):
    if interpolation not in ("linear", "lower", "higher", "midpoint"):
        raise ValueError("invalid interpolation method")

    return np.nanpercentile(data, q, method=interpolation)
