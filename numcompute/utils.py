"""
** Utility Functions
  * Calculate distances and activation functions
  * Calculate logsumexp and top-k values
  * Split data into batches

** Sources:
  * https://www.geeksforgeeks.org/machine-learning/activation-functions/ 
"""

import numpy as np

def euclidean_distance(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b))

def manhattan_distance(a, b):
    return np.sum(np.abs(np.asarray(a) - np.asarray(b)))

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return np.dot(a, b) / denominator

def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def relu(x):
    return np.maximum(0, x)

def softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def logsumexp(x, axis=None):
    x = np.asarray(x, dtype=float)
    maximum = np.max(x, axis=axis, keepdims=True)
    result = maximum + np.log(
        np.sum(np.exp(x - maximum), axis=axis, keepdims=True)
    )

    if axis is None:
        return result.item()

    return np.squeeze(result, axis=axis)

def top_k(values, k, largest=True):
    values = np.asarray(values)

    if values.ndim != 1:
        raise ValueError("values must be a 1D array")

    if not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")

    if k < 0 or k > len(values):
        raise ValueError("k must be between 0 and array length")

    if k == 0:
        return values[:0].copy()

    if largest:
        indices = np.argpartition(values, len(values) - k)[-k:]
        indices = indices[np.argsort(values[indices])[::-1]]
    else:
        indices = np.argpartition(values, k - 1)[:k]
        indices = indices[np.argsort(values[indices])]

    return values[indices]

def batch_data(data, batch_size):
    data = np.asarray(data)

    if not isinstance(batch_size, (int, np.integer)):
        raise TypeError("batch_size must be an integer")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]
