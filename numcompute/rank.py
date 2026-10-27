"""
** Ranking and Percentiles
  * Rank values and handle ties
  * Calculate percentiles

** Sources:
  * https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
  * https://numpy.org/doc/stable/reference/generated/numpy.percentile.html

"""

import numpy as np

def rank(data):
    arr = np.asarray(data)
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(len(arr), dtype=float)
    sorted_data = arr[order]

    i = 0
    while i < len(arr):
        j = i + 1

        while j < len(arr) and sorted_data[j] == sorted_data[i]:
            j += 1

        ranks[order[i:j]] = (i + 1 + j) / 2
        i = j

    return ranks


def percentile(data, q):
    return np.nanpercentile(data, q)
