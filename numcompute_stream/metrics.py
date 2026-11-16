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
    pass

def auc(fpr, tpr):
    pass


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