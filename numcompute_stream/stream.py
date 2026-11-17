'''
### `stream.py`
- Implements `StreamTrainer`, managing model + pipeline + logging
- Supports `.fit_chunk(X, y)` and `.score_chunk(X, y)` logic
- Logs per-chunk metrics, memory footprint, and cumulative accuracy
'''

import numpy as np
import time
import tracemalloc
from .metrics import StreamingClassificationMetrics


class StreamTrainer:
    """
    Manage chunk-by-chunk training and evaluation.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.metrics = StreamingClassificationMetrics()
        self.history = []



    def fit_chunk(self, X, y):
        """
        Train on one chunk and record time and memory usage.
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
        Evaluate a chunk using the current model.
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
