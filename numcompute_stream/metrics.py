"""
# Accuracy/Precision/Recall/F1, MSE, Confusion, ROC/AUC*
### `metrics.py`
- Classification:
  - `accuracy`
  - `precision`
  - `recall`
  - `f1`
  - `confusion_matrix`
- Regression:
  - `mse(y_true, y_pred)`
- **Bonus**: `roc_curve` and `auc` for binary classification
"""
import numpy as np
from collections import deque

def accuracy(y_true, y_pred):
    """
    Calculate the accuracy of the predicted data
    Parameters:
    y_true: a numpy array containing the expected values
    y_pred: a numpy array containing the predicted values
    Returns
    an accuracy score for the predicted data
    Raises
    ValueError: if dimensions are not 1, or shapes do not match between predicted and true, or input is empty
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
      raise ValueError("Inputs must be 1D arrays")
    if y_true.shape != y_pred.shape:
      raise ValueError("Inputs must have the same shape")
    if y_true.size == 0:
        raise ValueError("Inputs cannot be empty")
    trueRes = np.equal(y_true, y_pred)
    return np.mean(trueRes)

def precision(y_true, y_pred):
    """
    Calculate the precision of the predicted data
    Parameters:
    y_true: a 1D numpy array containing the expected values
    y_pred: a 1D numpy array containing the predicted values
    Returns
    a precision score for the predicted data
    Raises
    ValueError: if dimensions are not 1, or shapes do not match between predicted and true, or if input is empty
    """
    matrix = confusion_matrix(y_true, y_pred)
    tp = matrix[1, 1]
    fp = matrix[0, 1]
    if tp + fp == 0:
        return 0.0
    prec = tp / (tp + fp)
    return prec

def recall(y_true, y_pred):
    """
    Calculate the recall of the predicted data
    Parameters:
    y_true: a 1D numpy array containing the expected values
    y_pred: a 1D numpy array containing the predicted values
    Returns
    a recall score for the predicted data
    Raises
    ValueError: if dimensions are not 1, or shapes do not match between predicted and true, or if input is empty
    """
    matrix = confusion_matrix(y_true, y_pred)
    tp = matrix[1, 1]
    fn = matrix[1, 0]
    if tp + fn == 0:
        return 0.0
    rec = tp / (tp + fn)
    return rec

def f1(y_true, y_pred):
    """
    Calculate the f1 score of the predicted data
    Parameters:
    y_true: a 1D numpy array containing the expected values
    y_pred: a 1D numpy array containing the predicted values
    Returns
    a f1 score for the predicted data
    Raises
    ValueError: if dimensions are not 1, or shapes do not match between predicted and true, or if input is empty
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    if prec + rec == 0:
        return 0.0
    rtn = 2 * ((prec * rec) / (prec + rec))
    return rtn

def confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix of the predicted data (only supports binary classification)
    Parameters:
    y_true: a 1D numpy array containing the expected values
    y_pred: a 1D numpy array containing the predicted values
    Returns
    a (2,2) shaped confusion matrix for the predicted data, with the layout ((tn, fp), (fn, tp))
    Raises
    ValueError: if dimensions are not 1, or shapes do not match between predicted and true, or if input is empty
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("Inputs must have the same shape")
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    matrix = np.array([[tn, fp], [fn, tp]])
    return matrix

def mse(y_true, y_pred):
    """
    Calculate the MSE score of the predicted data
    Parameters:
    y_true: a 1D numpy array containing the expected values
    y_pred: a 1D numpy array containing the predicted values
    Returns
    a MSE score for the predicted data
    Raises
    ValueError: if dimensions are not 1, or shapes do not match between predicted and true, or if input is empty
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("Inputs must have the same shape")
    if y_true.size == 0:
        raise ValueError("Inputs cannot be empty")
    squared_error = (y_true - y_pred) ** 2
    return float(np.mean(squared_error))


def roc_curve(y_true, y_score):
    """
    Calculate the ROC curve using binary labels and prediction scores.
    Parameters:
    y_true: a 1D array containing true labels 0 and 1.
    y_score: a 1D array containing scores for the positive class.
    Returns:
    fpr: a 1D array containing false positive rates.
    tpr: a 1D array containing true positive rates.
    thresholds: a 1D array containing the score thresholds.

    Raises:
    ValueError: if the inputs are invalid or one class is missing.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("Inputs must be 1D")

    if len(y_true) == 0 or len(y_true) != len(y_score):
        raise ValueError("Inputs must be non-empty and have the same length")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("Labels must be 0 or 1")

    if not np.all(np.isfinite(y_score)):
        raise ValueError("Scores must be finite")

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both classes are required for ROC-AUC")

    order = np.argsort(-y_score, kind="stable")
    scores_sorted = y_score[order]
    labels_sorted = y_true[order]
    ends = np.r_[np.flatnonzero(np.diff(scores_sorted)), len(y_true) - 1]
    tp = np.cumsum(labels_sorted == 1)[ends]
    fp = np.cumsum(labels_sorted == 0)[ends]

    tpr = np.r_[0.0, tp / n_pos]
    fpr = np.r_[0.0, fp / n_neg]
    thresholds = np.r_[np.inf, scores_sorted[ends]]

    return fpr, tpr, thresholds


def auc(fpr, tpr):
    """
    Calculate the area under the ROC curve using trapezoids.
    Parameters:
    fpr: a 1D array of false positive rates in increasing order.
    tpr: a 1D array of corresponding true positive rates.
    Returns:
    float: area under the ROC curve.
    Raises:
    ValueError: if the input arrays are invalid.
    """
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)

    if (fpr.ndim != 1 or tpr.ndim != 1 or
        len(fpr) < 2 or fpr.shape != tpr.shape):
        raise ValueError("Invalid FPR or TPR arrays")

    if not np.all(np.isfinite(fpr)) or not np.all(np.isfinite(tpr)):
        raise ValueError("FPR and TPR must be finite")

    if np.any(np.diff(fpr) < 0):
        raise ValueError("FPR must be non-decreasing")

    return float(np.sum((tpr[:-1] + tpr[1:]) * np.diff(fpr) / 2))


def roc_auc_score(y_true, y_score):
    """
    Calculate binary ROC-AUC using true labels and prediction scores.
    Parameters:
    y_true: a 1D array containing true labels 0 and 1.
    y_score: a 1D array containing positive-class scores.
    Returns:
    float: ROC-AUC score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)


class StreamingClassificationMetrics:
    """
    Accumulate classification metrics across data chunks.
    Currently supports binary labels (0 and 1).
    """

    def __init__(self, window_size=None):
        if window_size is not None:
            if (isinstance(window_size, bool)
                or not isinstance(window_size, (int, np.integer))
                or window_size <= 0):
                raise ValueError("window_size must be a positive integer")
        self.window_size = window_size
        self.matrix = np.zeros((2, 2), dtype=int)
        self._window = deque(maxlen=window_size)

    def update(self, y_true_chunk, y_pred_chunk):
        y_true = np.asarray(y_true_chunk)
        y_pred = np.asarray(y_pred_chunk)
        if y_true.ndim != 1 or y_pred.ndim != 1:
            raise ValueError("Inputs must be 1D arrays")

        if y_true.shape != y_pred.shape:
            raise ValueError("Inputs must have the same shape")

        if y_true.size == 0:
            raise ValueError("Inputs cannot be empty")
        
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError("y_true must contain only 0 and 1")

        if not np.all(np.isin(y_pred, [0, 1])):
            raise ValueError("y_pred must contain only 0 and 1")
        chunk_matrix = np.zeros((2, 2), dtype=int)
        np.add.at(chunk_matrix,(y_true.astype(int), y_pred.astype(int)), 1)
        self.matrix += chunk_matrix
        if self.window_size is not None:
            self._window.append(chunk_matrix)
        return self

    
    def _calculate_metrics(self, matrix):
        tn, fp = matrix[0]
        fn, tp = matrix[1]

        total = tn + fp + fn + tp

        accuracy = (tp + tn) / total if total > 0 else 0.0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": matrix.copy()
        }




    def result(self):
        return self._calculate_metrics(self.matrix)

    
    def rolling_result(self):
        if self.window_size is None:
            raise ValueError("Rolling window is not enabled")

        if not self._window:
            window_matrix = np.zeros((2, 2), dtype=int)
        else:
            window_matrix = np.sum(
                np.stack(self._window),
                axis=0
            )

        return self._calculate_metrics(window_matrix)



    def reset(self):
        self.matrix.fill(0)
        self._window.clear()
        return self