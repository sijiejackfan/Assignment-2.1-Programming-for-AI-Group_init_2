'''
### `stream.py`
- Implements `StreamTrainer`, managing model + pipeline + logging
- Supports `.fit_chunk(X, y)` and `.score_chunk(X, y)` logic
- Logs per-chunk metrics, memory footprint, and cumulative accuracy
Sources:
https://docs.python.org/3/library/time.html
https://docs.python.org/3/library/tracemalloc.html
https://riverml.xyz/latest/api/evaluate/progressive-val-score
'''

import numpy as np
import time
import tracemalloc
from .metrics import StreamingClassificationMetrics


class StreamTrainer:

    """
    Manage chunk-by-chunk training and evaluation for a streaming pipeline. Uses the existing model to evaluate each chunk before training.
    Records per-chunk accuracy, cumulative classification metrics, training time, and memory usage.

    Parameters:
    pipeline: a Pipeline object containing the preprocessing steps and a DecisionTreeClassifier.
    Attributes:
    pipeline: the pipeline used for training and prediction.
    metrics: a StreamingClassificationMetrics object used to accumulate classification results.
    history: a list of dictionaries containing the recorded results for each chunk.
    AI Assistance: The implementation was adapted from code suggestions provided by ChatGPT (OpenAI, 2026). The code was reviewed and modified
    to fit the package requirements.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.metrics = StreamingClassificationMetrics()
        self.history = []



    def fit_chunk(self, X, y):
        """
        Train the pipeline using a new data chunk. Calls partial_fit() to update the pipeline and records
        the training time and peak memory usage. The results are added to the most recent history record if one exists.
        Parameters:
        X: a 2D NumPy array containing the input features.
        y: a 1D array containing the classification labels.
        Returns:
        self: the updated StreamTrainer object.
        """

        tracemalloc.start()
        start = time.perf_counter()

        try:
            self.pipeline.partial_fit(X, y)

            elapsed = time.perf_counter() - start
            current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        if self.history:
            self.history[-1]["training_time"] = elapsed
            self.history[-1]["peak_memory_mb"] = peak / (1024 ** 2)
        return self


    def score_chunk(self, X, y):
        """
        Evaluate a new data chunk using the current trained model. Calculates the accuracy for the current chunk and updates
        the cumulative Accuracy, Precision, Recall, and F1 scores. If the model has not been trained yet, the metrics are
        recorded as None and no predictions are made.
        Parameters:
        X: a 2D NumPy array containing the input features.
        y: a 1D array containing the true classification labels.
        Returns:
        dict: a record containing the chunk number, per-chunk accuracy, cumulative metrics, training time, and peak
        memory usage. Training time and memory are initially None until fit_chunk() updates the record.
        """
        model = self.pipeline.steps[-1][1]
        result = {
            "chunk": len(self.history) + 1,
            "accuracy": None,
            "cumulative_accuracy": None,
            "cumulative_precision": None,
            "cumulative_recall": None,
            "cumulative_f1": None,
            "training_time": None,
            "peak_memory_mb": None
        }
        if model.root is None:
            self.history.append(result)
            return result
        y_true = np.asarray(y)
        y_pred = self.pipeline.predict(X)
        result["accuracy"] = float(np.mean(y_true == y_pred))
        self.metrics.update(y_true, y_pred)
        metrics_result = self.metrics.result()
        result["cumulative_accuracy"] = metrics_result["accuracy"]
        result["cumulative_precision"] = metrics_result["precision"]
        result["cumulative_recall"] = metrics_result["recall"]
        result["cumulative_f1"] = metrics_result["f1"]
        self.history.append(result)

        return result
