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
Sources: 
the provided materials for tree
'''

import numpy as np


def gini(y):
    """
    Calculate the Gini impurity of classification labels.
    Parameters:
    y: 1D array of non-negative classification labels.
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
    y: 1D array of non-negative classification labels.
    Returns:
    float: Entropy in bits.
    """
    if len(y) == 0:
        return 0.0

    p = np.bincount(y, minlength=2) / len(y)
    p = p[p > 0]

    return -(p * np.log2(p)).sum()


def information_gain(y, mask, criterion="gini"):
    """
    Calculate the information gain of a proposed data split. Compares the impurity before and after splitting the labels
    into two groups. Supports Gini and Entropy as the criterion.
    Parameters:
    y: a 1D array of non-negative integer classification labels.
    mask: a 1D boolean array indicating which samples belong to the left group.
    criterion: str, "gini" or "entropy" (default "gini").
    Returns:
    float: the reduction in impurity after the split.
    Returns 0 if the split does not separate the samples.
    Raises:
    ValueError: if the inputs are not 1D, have different shapes, or the criterion is invalid.
    TypeError: if the mask is not boolean.
    """
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
    A decision tree node, either an internal node containing a split condition or a leaf node
    containing a prediction label.

    Attributes:
    feature: int or None, index of the feature used for splitting.
    threshold: float or None, threshold used for splitting.
    left: Node or None, left child node.
    right: Node or None, right child node.
    label: int or None, encoded prediction label for a leaf node.
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None, class_counts=None):

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label
        self.class_counts = class_counts


class DecisionTreeClassifier:
    """
    Train and use a Decision Tree for classification. Supports Gini and Entropy splitting, maximum depth,
    minimum samples for splitting, and optional random feature selection. Streaming training is supported through buffered
    retraining using partial_fit().

    Parameters:
    max_depth: int, maximum depth of the tree (default 5).
    min_samples_split: int, minimum samples required to split a node (default 2).
    max_features: int or None, maximum number of randomly selected features considered at each split (default None).
    buffer_size: int, minimum samples required to trigger retraining (default 500).
    criterion: str, "gini" or "entropy" (default "gini").
    Attributes:
    max_depth: maximum depth of the tree.
    min_samples_split: minimum number of samples required for splitting.
    max_features: number of features considered at each split.
    buffer_size: number of buffered samples required for retraining.
    criterion: selected impurity criterion.
    classes: a NumPy array or None, original classification labels.
    n_features: int or None, number of input features.
    root: Node or None, root of the trained decision tree.
    _X_buffer: a list of buffered feature arrays.
    _y_buffer: a list of buffered label arrays.
    _buffer_count: int, number of samples currently buffered.
    retrain_count: int, number of retraining events triggered by partial_fit().
    random_state: int, seed used for random feature selection.
    AI Assistance: The implementation was adapted from code suggestions provided by ChatGPT (OpenAI, 2026). The code was reviewed and modified
    to fit the package requirements.
    """

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
        Find the best split using the selected implementation.
        Parameters:
        X: a 2D NumPy array containing numerical features.
        y: a 1D array containing encoded classification labels.
        Returns:
        best_feature: index of the selected feature.
        best_threshold: threshold of the selected split.
        best_gain: information gain of the selected split.
        """
        return self._best_split_vectorized(X, y)


    
    def _best_split_loop(self, X, y):
        """
        Find the feature and threshold with the highest information gain. Checks possible split thresholds for each selected feature.
        If max_features is set, only a random subset of features is considered. NaN values are excluded when finding thresholds.
        Parameters:
        X: a 2D NumPy array containing numerical features.
        y: a 1D array of encoded integer classification labels.
        Returns:
        best_feature: int or None, index of the selected feature.
        best_threshold: float or None, threshold for the selected split.
        best_gain: float, highest information gain found. Returns None for the feature and threshold if no useful split is found.
        Raises:
        ValueError: if the input dimensions or shapes are invalid, the input is empty, or features contain infinity.
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


    
    def _best_split_vectorized(self, X, y):
        """
        Find the best split using NumPy vectorized calculations.
        Parameters:
        X: a 2D NumPy array containing numerical features.
        y: a 1D array of encoded integer classification labels.
        Returns:
        best_feature: index of the selected feature.
        best_threshold: threshold of the selected split.
        best_gain: highest information gain found.
        Raises:
        ValueError: if input dimensions or shapes are invalid, the input is empty, or features contain infinity.
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
        n_samples, n_features = X.shape
        if self.max_features is None:
            selected_features = np.arange(n_features)
        else:
            n_selected = min(self.max_features, n_features)
            selected_features = np.sort(self._rng.choice(n_features, size=n_selected, replace=False))
        total_counts = np.bincount(y)
        n_classes = len(total_counts)

        if self.criterion == "gini":
            before = gini(y)
        else:
            before = entropy(y)

        for f in selected_features:
            values = X[:, f]
            valid = ~np.isnan(values)
            order = np.argsort(values[valid], kind="stable")
            sorted_values = values[valid][order]
            sorted_y = y[valid][order]
            cuts = np.flatnonzero(np.diff(sorted_values) > 0)
            if cuts.size == 0:
                continue

            thresholds = (sorted_values[cuts] + sorted_values[cuts + 1]) / 2
            left_n = cuts + 1
            right_n = n_samples - left_n
            one_hot = np.eye(n_classes, dtype=int)[sorted_y]
            left_counts = np.cumsum(one_hot, axis=0)[cuts]
            right_counts = total_counts - left_counts

            left_p = left_counts / left_n[:, None]
            right_p = right_counts / right_n[:, None]

            if self.criterion == "gini":
                left_imp = 1.0 - np.sum(left_p ** 2, axis=1)
                right_imp = 1.0 - np.sum(right_p ** 2, axis=1)
            else:
                left_imp = -np.sum(
                    left_p * np.log2(np.where(left_p > 0, left_p, 1)),
                    axis=1
                )
                right_imp = -np.sum(
                    right_p * np.log2(np.where(right_p > 0, right_p, 1)),
                    axis=1
                )

            gains = before - (
                left_n * left_imp + right_n * right_imp
            ) / n_samples
            index = np.flatnonzero(
                gains >= np.max(gains) - 1e-12
            )[0]

            if gains[index] > best_gain + 1e-12:
                best_gain = float(gains[index])
                best_feature = f
                best_threshold = thresholds[index]

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
        counts = np.bincount(y)
        majority_label = counts.argmax()
        if (depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1):
            return Node(label=majority_label, class_counts=counts)

        feature, threshold, gain = self._best_split(X, y)

        if feature is None:
            return Node(label=majority_label, class_counts=counts)
        
        mask = X[:, feature] <= threshold
        return Node(
            feature=feature,
            threshold=threshold,
            left=self._build(X[mask], y[mask], depth + 1),
            right=self._build(X[~mask], y[~mask], depth + 1)
        )


    
    def partial_fit(self, X_chunk, y_chunk):
        """
        Add a new data chunk to the training buffer. Stores the input features and labels until the number
        of buffered samples reaches buffer_size. The tree is then retrained from scratch using the buffered data, and the
        buffer is cleared. Between retraining events, predictions continue using the most recently trained tree.
        Parameters:
        X_chunk: a 2D NumPy array containing numerical features.
        y_chunk: a 1D array containing classification labels.
        Returns:
        self: the updated DecisionTreeClassifier object.
        Raises:
        ValueError: if input dimensions or sample counts do not match, the chunk is empty, the feature count changes, or
        the input contains invalid numerical values.
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
        Train a decision tree from scratch. Encodes the classification labels, builds the tree recursively,
        and stores the trained model. Clears any previously buffered training data after fitting
        Parameters:
        X: 2D array of numerical features.
        y: 1D array of classification labels.
        Returns:
        self: fitted DecisionTreeClassifier.
        Raises:
        ValueError: if input dimensions or sample counts do not match, the training data is empty, X contains infinity, or
        numerical labels contain NaN or infinity.
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
        Raises:
        ValueError: if the tree has not been trained, the input is not 2D, the feature count does not match, or the input contains infinity.
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

    
    def predict_proba(self, X):
        """
        Predict class probabilities using the trained tree. Each sample follows the tree until reaching a leaf node.
        Probabilities are calculated from the class counts in that leaf.
        Parameters:
        X: a 2D NumPy array containing numerical features.
        Returns:
        A 2D NumPy array containing the probability of each class.
        The columns follow the order of self.classes.
        Raises:
        ValueError: if the tree has not been fitted or the input is invalid.
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
        probabilities = np.zeros((len(X), len(self.classes)))
        for i, x in enumerate(X):
            node = self.root
            while node.label is None:
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            counts = node.class_counts
            probabilities[i, :len(counts)] = counts / counts.sum()

        return probabilities









