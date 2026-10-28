""" Benchmark operations """

import time
import csv
import platform
import statistics
import numpy as np


def mean_loop(data):
    """Calculate the mean using a loop."""
    if len(data) == 0:
        raise ValueError("Data cannot be empty")

    total = 0.0
    for value in data:
        total += value

    return total / len(data)


def square_loop(data):
    """Square each value using a loop."""
    result = []
    for value in data:
        result.append(value * value)

    return result


def distance_loop(a, b):
    """Calculate Euclidean distance using a loop."""
    if len(a) != len(b):
        raise ValueError("Arrays must have equal lengths")

    total = 0.0
    for x, y in zip(a, b):
        total += (x - y) ** 2

    return total ** 0.5


def measure_time(func, repeats=5):
    """Measure median execution time in seconds."""
    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)

    return statistics.median(times)

def run_benchmarks():
    """Compare loop implementations with NumPy."""
    rng = np.random.default_rng(42)
    results = []

    for size in [100, 1000, 10000, 100000]:
        a = rng.random(size)
        b = rng.random(size)

        tests = [
            ("Mean", lambda: mean_loop(a),
             lambda: np.mean(a)),
            ("Square", lambda: square_loop(a),
             lambda: np.square(a)),
            ("Euclidean distance", lambda: distance_loop(a, b),
             lambda: np.linalg.norm(a - b)),
        ]

        for name, loop_func, numpy_func in tests:
            # Check that both methods produce equivalent results.
            if not np.allclose(
                loop_func(), numpy_func(), rtol=1e-9, atol=1e-12
            ):
                raise ValueError(f"Results do not match: {name}")

            # Warm up both implementations before timing.
            loop_func()
            numpy_func()

            loop_time = measure_time(loop_func)
            numpy_time = measure_time(numpy_func)

            speedup = (
                loop_time / numpy_time if numpy_time > 0 else None
            )

            results.append({
                "operation": name,
                "size": size,
                "loop_seconds": loop_time,
                "numpy_seconds": numpy_time,
                "speedup": speedup,
            })

    return results

def main():
    """Run benchmarks, display results and save them."""
    print("NumCompute Benchmark")
    print("Python:", platform.python_version())
    print("NumPy:", np.__version__)
    print("System:", platform.platform())
    print("Random seed: 42")
    print("Timing: median of 5 runs\n")

    results = run_benchmarks()

    print(
        f"{'Operation':<22} {'Size':>8} "
        f"{'Loop (s)':>12} {'NumPy (s)':>12} {'Speedup':>10}"
    )
    print("-" * 68)

    for row in results:
        speedup = row["speedup"]
        speedup_text = f"{speedup:.2f}x" if speedup else "N/A"

        print(
            f"{row['operation']:<22} "
            f"{row['size']:>8} "
            f"{row['loop_seconds']:>12.6g} "
            f"{row['numpy_seconds']:>12.6g} "
            f"{speedup_text:>10}"
        )

    with open(
        "benchmark_results.csv", "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to benchmark_results.csv")


if __name__ == "__main__":
    main()
