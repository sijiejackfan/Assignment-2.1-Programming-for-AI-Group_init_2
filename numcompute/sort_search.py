"""
Sorting and Searching

** Implement:
  * stable_sort() - stable sorting algorithm
  * quick_sort() - Quick Sort algorithm
  * merge_sort() - Merge Sort algorithm
  * topk() - the k largest or smallest elements using "argpartition"
  * top_p() - the top p ratio of elements
  * quickselect() - the kth smallest element

** Requirements:
  * Using NumPy for array operations
  * Support numeric arrays
  * Validate inputs where necessary

** Sources:
  * https://numpy.org/doc/stable/reference/generated/numpy.sort.html
  * https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html
  * https://www.geeksforgeeks.org/python/sorting-algorithms-in-python/

"""

import numpy as np


def stable_sort(values):
    arr = np.asarray(values)
    return np.sort(arr, kind="stable")


def quick_sort(values):
    arr = np.asarray(values)

    if arr.ndim != 1:
        raise ValueError("values must be a 1D array")
    if len(arr) <= 1:
        return arr.copy()

    pivot = arr[len(arr) // 2]

    left = arr[arr < pivot]
    middle = arr[arr == pivot]
    right = arr[arr > pivot]

    return np.concatenate((
        quick_sort(left),
        middle,
        quick_sort(right)
    ))


def merge_sort(values):
    arr = np.asarray(values)

    if arr.ndim != 1:
        raise ValueError("values must be a 1D array")

    if len(arr) <= 1:
        return arr.copy()

    middle = len(arr) // 2

    left = merge_sort(arr[:middle])
    right = merge_sort(arr[middle:])

    result = []
    i = 0
    j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])

    return np.array(result)


def topk(values, k, largest=True):
    arr = np.asarray(values)

    if arr.ndim != 1:
        raise ValueError("values must be a 1D array")

    if not isinstance(k, (int, np.integer)):
        raise TypeError("k should be an integer")

    if k < 0 or k > len(arr):
        raise ValueError("k should be between 0 and array length")

    if k == 0:
        return arr[:0].copy()

    if largest:
        indices = np.argpartition(arr, len(arr) - k)[-k:]
        indices = indices[np.argsort(arr[indices])[::-1]]
    else:
        indices = np.argpartition(arr, k - 1)[:k]
        indices = indices[np.argsort(arr[indices])]

    return arr[indices]


def quickselect(values, k):
    arr = np.asarray(values)

    if arr.ndim != 1:
        raise ValueError("values must be a 1D array")

    if not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")

    if k < 0 or k >= len(arr):
        raise ValueError("k must be between 0 and len(values)-1")

    return np.partition(arr, k)[k]
