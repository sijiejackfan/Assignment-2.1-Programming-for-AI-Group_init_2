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