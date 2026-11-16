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
    if len(y) == 0:
        return 0.0

    p = np.bincount(y, minlength=2) / len(y)
    return 1.0 - (p ** 2).sum()



def entropy(y):
    """
    Calculate the entropy of classification labels.

    Parameters:
    y: 1D array of classification labels.

    Returns:
    float: Entropy in bits.
    """
    if len(y) == 0:
        return 0.0

    p = np.bincount(y, minlength=2) / len(y)
    p = p[p > 0]

    return -(p * np.log2(p)).sum()


def information_gain(y, mask, criterion="gini"):
    y = np.asarray(y)
    mask = np.asarray(mask)
    if y.ndim != 1 or mask.ndim != 1:
        raise ValueError("Inputs must be 1D")

    if y.shape != mask.shape:
        raise ValueError("Labels and mask must have the same shape")

    if mask.dtype != np.dtype(bool):
        raise TypeError("Mask must be boolean")

    if criterion == "gini":
        impurity = gini
    elif criterion == "entropy":
        impurity = entropy
    else:
        raise ValueError("Invalid criterion")

    if len(y) == 0 or mask.all() or not mask.any():
        return 0.0

    before = impurity(y)
    n_left = mask.sum()
    n_right = (~mask).sum()
    after = (n_left * impurity(y[mask]) + n_right * impurity(y[~mask])) / len(y)

    return float(before - after)


class Node:
    """
    A decision tree node, either an internal node
    containing a split condition or a leaf node
    containing a prediction label.
    """

    def __init__(self, feature=None, threshold=None,
                 left=None, right=None, label=None):

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label


class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_split=2,
                 max_features=None, buffer_size=500,
                 criterion="gini"):

        if criterion not in ("gini", "entropy"):
            raise ValueError("Invalid criterion")
        if isinstance(buffer_size, bool) or not isinstance(
            buffer_size, (int, np.integer)
        ):
            raise TypeError("buffer_size must be an integer")

        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")

        if max_features is not None:
            if isinstance(max_features, bool) or not isinstance(
                max_features, (int, np.integer)
            ):
                raise TypeError("max_features must be an integer or None")

            if max_features <= 0:
                raise ValueError("max_features must be positive")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.buffer_size = buffer_size
        self.criterion = criterion
        self.classes = None
        self.n_features = None  
        self.root = None
        self._X_buffer = []
        self._y_buffer = []
        self._buffer_count = 0
        self.retrain_count = 0
        self.random_state = 42
        self._rng = np.random.default_rng(self.random_state)

    
    def _best_split(self, X, y):
        """
        Find the feature and threshold with the
        highest information gain.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("Invalid input dimensions")

        if X.shape[0] != len(y) or X.size == 0:
            raise ValueError("Invalid input shape")

        if np.isinf(X).any():
            raise ValueError("Features cannot contain infinity")

        best_feature = None
        best_threshold = None
        best_gain = 0.0

        n_features = X.shape[1]
        if self.max_features is None:
            selected_features = np.arange(n_features)
        else:
            n_selected = min(self.max_features, n_features)

            selected_features = np.sort(
                self._rng.choice(
                    n_features,
                    size=n_selected,
                    replace=False
                )
            )

        for f in selected_features:
            values = X[:, f]
            unique_values = np.unique(values[~np.isnan(values)])

            if len(unique_values) < 2:
                continue

            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in thresholds:
                mask = values <= threshold
                gain = information_gain(y, mask, self.criterion)
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_feature = f
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    
    def _build(self, X, y, depth):
        """
        Recursively build a decision tree.

        Parameters:
        X: 2D NumPy array of features.
        y: 1D array of encoded integer labels.
        depth: current depth of the node.

        Returns:
        Node: root of the constructed subtree.
        """
        majority_label = np.bincount(y).argmax()
        if (depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1):
            return Node(label=majority_label)

        feature, threshold, gain = self._best_split(X, y)

        if feature is None:
            return Node(label=majority_label)
        
        mask = X[:, feature] <= threshold
        return Node(
            feature=feature,
            threshold=threshold,
            left=self._build(X[mask], y[mask], depth + 1),
            right=self._build(X[~mask], y[~mask], depth + 1)
        )


    
    def partial_fit(self, X_chunk, y_chunk):
        """
        Add a chunk to the training buffer.

        Retrain the decision tree when the buffer reaches
        buffer_size, then clear the buffer.

        Returns:
        self: updated DecisionTreeClassifier.
        """
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)

        if X_chunk.ndim != 2 or y_chunk.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D")

        if X_chunk.shape[0] != len(y_chunk):
            raise ValueError("X and y must have matching sample counts")

        if X_chunk.shape[0] == 0 or X_chunk.shape[1] == 0:
            raise ValueError("Chunk cannot be empty")

        if np.isinf(X_chunk).any():
            raise ValueError("X cannot contain infinity")

        if np.issubdtype(y_chunk.dtype, np.number):
            if not np.all(np.isfinite(y_chunk)):
                raise ValueError("Labels cannot contain NaN or infinity")

        expected_features = self.n_features

        if expected_features is None and self._X_buffer:
            expected_features = self._X_buffer[0].shape[1]

        if (expected_features is not None
            and X_chunk.shape[1] != expected_features):
            raise ValueError("Feature count does not match")

        self._X_buffer.append(X_chunk.copy())
        self._y_buffer.append(y_chunk.copy())
        self._buffer_count += len(y_chunk)
        if self._buffer_count >= self.buffer_size:
            X_train = np.vstack(self._X_buffer)
            y_train = np.concatenate(self._y_buffer)

            self.fit(X_train, y_train)
            self.retrain_count += 1

        return self

    
    def fit(self, X, y):
        """
        Train a decision tree from scratch.

        Parameters:
        X: 2D array of numerical features.
        y: 1D array of classification labels.

        Returns:
        self: fitted DecisionTreeClassifier.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D")

        if X.shape[0] != len(y):
            raise ValueError("X and y must have matching sample counts")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Training data cannot be empty")

        if np.isinf(X).any():
            raise ValueError("X cannot contain infinity")

        if np.issubdtype(y.dtype, np.number):
            if not np.all(np.isfinite(y)):
                raise ValueError("Labels cannot contain NaN or infinity")
            
        classes, encoded_y = np.unique(y, return_inverse=True)
        self._rng = np.random.default_rng(self.random_state)
        root = self._build(X, encoded_y, depth=0)
        self.classes = classes
        self.n_features = X.shape[1]
        self.root = root
        self._X_buffer.clear()
        self._y_buffer.clear()
        self._buffer_count = 0

        return self

    
    def predict(self, X):
        """
        Predict classification labels using the trained tree.

        Parameters:
        X: 2D array of numerical features.

        Returns:
        1D NumPy array of predicted labels.
        """
        if self.root is None:
            raise ValueError("Decision tree must be fitted first")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")

        if X.shape[1] != self.n_features:
            raise ValueError("Feature count does not match")

        if np.isinf(X).any():
            raise ValueError("X cannot contain infinity")

        predictions = []
        for x in X:
            node = self.root
            while node.label is None:
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right

            predictions.append(node.label)
        return self.classes[np.array(predictions, dtype=int)]








