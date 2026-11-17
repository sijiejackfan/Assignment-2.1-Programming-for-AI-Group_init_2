import numpy as np
import pytest

from numcompute_stream.stats import StreamingStats


def test_streaming_mean():
    stats = StreamingStats()
    stats.update_stats([[1, 10], [3, 20]])
    stats.update_stats([[5, 30]])
    assert np.allclose(stats.mean_result(), [3, 20])


def test_streaming_variance_and_std():
    stats = StreamingStats()
    stats.update_stats([[1], [3]])
    stats.update_stats([[5]])
    assert np.allclose(stats.variance(), [8 / 3])
    assert np.allclose(stats.std(), [np.sqrt(8 / 3)])


def test_streaming_nan():
    stats = StreamingStats()
    stats.update_stats([[1, np.nan], [3, 4]])
    assert stats.count.tolist() == [2, 1]
    assert np.allclose(stats.mean_result(), [2, 4])


def test_streaming_histogram():
    stats = StreamingStats(hist_edges=[0, 2, 4])
    stats.update_stats([[0, 1], [2, 3]])
    stats.update_stats([[4, np.nan]])
    counts, edges = stats.histogram_result()
    assert counts.tolist() == [[1, 2], [1, 1]]
    assert np.allclose(edges, [0, 2, 4])


def test_streaming_quantile():
    stats = StreamingStats()
    stats.update_stats([[1, 10], [3, 20]])
    stats.update_stats([[5, 30]])
    assert np.allclose(stats.quantile_result(0.5), [3, 20])


def test_streaming_reset():
    stats = StreamingStats(hist_edges=[0, 5])
    stats.update_stats([[2]])
    stats.reset()
    assert stats.count is None
    assert stats.hist_edges.tolist() == [0, 5]
    with pytest.raises(ValueError):
        stats.mean_result()


def test_streaming_invalid_input():
    stats = StreamingStats()
    with pytest.raises(ValueError):
        stats.update_stats([1, 2, 3])
    with pytest.raises(ValueError):
        stats.update_stats(np.empty((0, 2)))


def test_quantile_tracking_disabled():
    stats = StreamingStats(track_quantiles=False)
    stats.update_stats([[1], [2]])
    with pytest.raises(ValueError):
        stats.quantile_result(0.5)
