'''
### `tree.py`
- `DecisionTreeClassifier`: depth-limited, Gini or entropy-based
- `.fit(X, y)`: trains the tree from scratch on the given data
- `.partial_fit(X_chunk, y_chunk)` supports streaming via **buffered retraining**:
  - Each call appends the chunk to an internal data buffer
  - Once the buffer holds at least `buffer_size` samples, the tree is retrained on the buffered data and the buffer is cleared
  - Between retrains, `.predict()` uses the most recently trained tree
- Support config: `max_depth`, `min_samples_split`, `max_features`, `buffer_size`
  ```python
  tree = DecisionTreeClassifier(max_depth=5, buffer_size=500)
  for X_chunk, y_chunk in stream:
      tree.partial_fit(X_chunk, y_chunk)
  ```
'''

import numpy as np


def gini(y):
    """
    Calculate the Gini impurity of classification labels.

    Parameters:
    y: 1D array of classification labels.

    Returns:
    float: Gini impurity.
    """
    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError("Labels must be a 1D array")

    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return float(1.0 - np.sum(p ** 2))


def entropy(y):
    """
    Calculate the entropy of classification labels.

    Parameters:
    y: 1D array of classification labels.

    Returns:
    float: Entropy in bits.
    """
    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError("Labels must be a 1D array")

    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

