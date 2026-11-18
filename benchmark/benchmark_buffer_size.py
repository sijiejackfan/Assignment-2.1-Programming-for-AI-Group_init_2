'''
AI Assistance: The implementation was adapted from code suggestions provided by ChatGPT (OpenAI, 2026). The code was reviewed and modified
to fit the package requirements.
'''
import time
import csv
import numpy as np
from pathlib import Path

from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.stream import StreamTrainer


rng = np.random.default_rng(42)

X = rng.normal(size=(300, 5))
y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

chunk_size = 30
buffer_sizes = [30, 60, 120]
repeats = 5

results = []

print("Buffer | Retrains | Retrain (ms) | Total Fit (ms) | Accuracy")

for buffer_size in buffer_sizes:
    retrain_times = []
    total_times = []
    accuracies = []

    for _ in range(repeats):
        tree = DecisionTreeClassifier(buffer_size=buffer_size)
        pipe = Pipeline([("tree", tree)])
        trainer = StreamTrainer(pipe)

        current_retrain_times = []
        common_accuracies = []

        for i in range(0, len(X), chunk_size):
            X_chunk = X[i:i + chunk_size]
            y_chunk = y[i:i + chunk_size]

            record = trainer.score_chunk(X_chunk, y_chunk)

            # Compare accuracy only after all buffer sizes
            # have had enough samples for initial training
            if i >= max(buffer_sizes) and record["accuracy"] is not None:
                common_accuracies.append(record["accuracy"])

            before = tree.retrain_count
            trainer.fit_chunk(X_chunk, y_chunk)

            if tree.retrain_count > before:
                current_retrain_times.append(
                    trainer.history[-1]["training_time"]
                )

        retrain_times.append(
            np.median(current_retrain_times) * 1000
        )

        total_times.append(
            sum(record["training_time"] for record in trainer.history) * 1000
        )

        accuracies.append(np.mean(common_accuracies))

    median_retrain = np.median(retrain_times)
    median_total = np.median(total_times)
    mean_accuracy = np.mean(accuracies)

    results.append([
        buffer_size, tree.retrain_count,
        median_retrain, median_total, mean_accuracy
    ])

    print(
        f"{buffer_size:6d} | {tree.retrain_count:8d} | "
        f"{median_retrain:12.2f} | {median_total:14.2f} | "
        f"{mean_accuracy:.3f}"
    )


output = Path(__file__).parent / "buffer_size_results.csv"

with open(output, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Buffer_Size", "Retrains", "Median_Retrain_ms",
        "Total_Fit_ms", "Common_Accuracy"
    ])
    writer.writerows(results)
