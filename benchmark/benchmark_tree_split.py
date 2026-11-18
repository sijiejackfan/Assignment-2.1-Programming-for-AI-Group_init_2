'''
AI Assistance: The implementation was adapted from code suggestions provided by ChatGPT (OpenAI, 2026). The code was reviewed and modified
to fit the package requirements.
'''
import time
import numpy as np
import csv

from numcompute_stream.tree import DecisionTreeClassifier


rng = np.random.default_rng(42)
sample_sizes = [100, 500, 1000, 2000]
repeats = 5

print("Samples | Loop (ms) | Vectorized (ms) | Speedup")
results = []
for n in sample_sizes:
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

    tree = DecisionTreeClassifier()

    loop_result = tree._best_split_loop(X, y)
    vector_result = tree._best_split_vectorized(X, y)

    assert loop_result[0] == vector_result[0]
    assert np.isclose(loop_result[1], vector_result[1])
    assert np.isclose(loop_result[2], vector_result[2])

    loop_times = []
    vector_times = []

    for _ in range(repeats):
        start = time.perf_counter()
        tree._best_split_loop(X, y)
        loop_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        tree._best_split_vectorized(X, y)
        vector_times.append(time.perf_counter() - start)

    loop_ms = np.median(loop_times) * 1000
    vector_ms = np.median(vector_times) * 1000
    speedup = loop_ms / vector_ms
    results.append([n, loop_ms, vector_ms, speedup])
    print(f"{n:7d} | {loop_ms:9.2f} | {vector_ms:15.2f} | {speedup:6.2f}x")
with open("benchmark/tree_split_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Samples", "Loop_ms", "Vectorized_ms", "Speedup"])
    writer.writerows(results)
