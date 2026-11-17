"""
Streaming stats (Welford), histogram, quantiles
### `stats.py`
- Basic descriptive statistics:
  - Mean, median, standard deviation, min, max
- Histogram
- Quantiles (with NaN handling)
- Axis-wise stats with clear dimension/shape behaviour
"""
import numpy as np

def mean(data, axis=0):
    """
    Use numpy functions to calculate the mean of the data
    Parameters:
    data: a numercial numpy array containing input values.
    axis: axis used to calculate the mean, default 0.
    Returns
    a scalar or numpy array containing the mean values along the given axis.
    """
    return np.nanmean(data, axis=axis)

def median(data, axis=0):
    """
    Use numpy functions to calculate the median of the data
    Parameters:
    data: a numercial numpy array containing input values.
    axis: axis used to calculate the median, default 0.
    Returns
    a scalar or numpy array containing the median values along the given axis.
    """
    return np.nanmedian(data, axis=axis)

def std(data, axis=0):
    """
    Use numpy functions to calculate the standard deviation of the data
    Parameters:
    data: a numercial numpy array containing input values.
    axis: axis used to calculate the standard deviation, default 0.
    Returns
    a scalar or numpy array containing the standard deviation values along the given axis.
    """
    return np.nanstd(data, axis=axis)

def minimum(data, axis=0):
    """
    Use numpy functions to calculate the minimum of the data
    Parameters:
    data: a numercial numpy array containing input values.
    axis: axis used to calculate the minimum, default 0.
    Returns
    a scalar or numpy array containing the minimum values along the given axis.
    """
    return np.nanmin(data, axis=axis)

def maximum(data, axis=0):
    """
    Use numpy functions to calculate the maximum of the data
    Parameters:
    data: a numercial numpy array containing input values.
    axis: axis used to calculate the maximum, default 0.
    Returns
    a scalar or numpy array containing the maximum values along the given axis.
    """
    return np.nanmax(data, axis=axis)

def histogram(data, bins=10):
    """
    Use numpy functions to calculate the histogram of the data.
    Parameters:
    data: a numercial numpy array containing input values.
    bins: number of bins, default 10, or an array containing bin edges.
    Returns
    hist: the values of the histogram
    bin_edges: a 1D numpy array containing the edges of each bin.
    Source: https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
    """
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    return np.histogram(data, bins=bins)

def quantile(data, q, axis=0):
    """
    Use numpy functions to calculate the histogram of the data.
    Parameters:
    data: a numercial numpy array containing input values.
    q: a float or array of floats between 0 and 1, specifying the quantiles to calculate
    axis: axis used to calculate the maximum, default 0.
    Returns
    a float or array of floats on the given quantile.
    Source: https://numpy.org/doc/stable/reference/generated/numpy.quantile.html
    """
    return np.nanquantile(data, q, axis=axis)

class StreamingStats:
    """
    Calculate and update statistics from numerical data in chunks.
    The class supports running mean, variance, standard deviation, histogram, and quantile calculations. NaN values are ignored
    when updating the statistics.
    
    Parameters:
    hist_edges: a 1D array or None, fixed bin edges for the histogram.
    track_quantiles: bool, whether to store previous chunks for exact quantile calculations (default True).
    Attributes:
    count: a 1D array, number of valid values in each column.
    mean: a 1D array, running mean of each column.
    M2: a 1D array, accumulated squared differences used for variance.
    hist_edges: a 1D array or None, histogram bin edges.
    hist_counts: a 2D array or None, accumulated histogram counts.
    Source:
    https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html
    AI Assistance: The implementation was adapted from code suggestions provided by ChatGPT (OpenAI, 2026). The code was reviewed and modified
    to fit the package requirements.
    """

    def __init__(self, hist_edges=None, track_quantiles=True):
        self.count = None
        self.mean = None
        self.M2 = None
        self.hist_edges = None
        self.hist_counts = None
        self.track_quantiles = track_quantiles
        self._quantile_chunks = []

        if hist_edges is not None:
            edges = np.asarray(hist_edges, dtype=float)

            if (edges.ndim != 1 or
                edges.size < 2 or
                not np.all(np.isfinite(edges)) or
                not np.all(np.diff(edges) > 0)):
                raise ValueError("Histogram edges must be increasing finite values")

            self.hist_edges = edges.copy()

    def update_stats(self, X_chunk):
        """
        Update the running statistics using a new chunk. Uses the previous statistics and the new chunk to update 
        the mean and variance. NaN values are ignored.
        Parameters:
        X_chunk: a 2D numerical array containing new samples.
        Returns:
        self: the updated StreamingStats object.
        Raises:
        ValueError: if the input is not 2D, is empty, or has a different number of features.
        """
        X_chunk = np.asarray(X_chunk, dtype=float)
        if X_chunk.ndim != 2:
            raise ValueError("Input must be 2D")
        if X_chunk.shape[0] == 0:
            raise ValueError("Chunk cannot be empty")
        if self.count is None:
            n_features = X_chunk.shape[1]
            self.count = np.zeros(n_features, dtype=int)
            self.mean = np.zeros(n_features)
            self.M2 = np.zeros(n_features)
        elif X_chunk.shape[1] != len(self.count):
            raise ValueError("Feature count does not match")
        valid = ~np.isnan(X_chunk)
        chunk_count = np.sum(valid, axis=0)
        chunk_sum = np.sum(np.where(valid, X_chunk, 0.0), axis=0)
        chunk_mean = np.divide(chunk_sum, chunk_count, out=np.zeros_like(chunk_sum), where=chunk_count > 0)
        centered = np.where(valid, X_chunk - chunk_mean, 0.0)
        chunk_M2 = np.sum(centered ** 2, axis=0)
        total_count = self.count + chunk_count
        delta = chunk_mean - self.mean
        correction = np.divide(delta ** 2 * self.count * chunk_count, total_count, out=np.zeros_like(self.mean), where=total_count > 0)
        self.M2 += chunk_M2 + correction
        self.mean += np.divide(delta * chunk_count, total_count, out=np.zeros_like(self.mean), where=total_count > 0)
        self.count = total_count
        if self.hist_edges is not None:
            self._update_histogram(X_chunk)
        if self.track_quantiles:
            self._quantile_chunks.append(X_chunk.copy())
        return self

    
    def _update_histogram(self, X_chunk):
        """
        Update histogram counts using a new chunk. Values outside the configured bin edges, as well as NaN 
        and infinite values, are not counted. The last bin includes the rightmost edge.
        Parameters:
        X_chunk: a 2D numerical array containing new samples.
        """
        n_features = X_chunk.shape[1]
        n_bins = len(self.hist_edges) - 1
        if self.hist_counts is None:
            self.hist_counts = np.zeros((n_features, n_bins), dtype=int)

        valid = (np.isfinite(X_chunk) & (X_chunk >= self.hist_edges[0]) & (X_chunk <= self.hist_edges[-1]))
        bin_indices = np.searchsorted(
            self.hist_edges,
            X_chunk,
            side="right"
        ) - 1
        bin_indices = np.minimum(bin_indices, n_bins - 1)
        feature_indices = np.broadcast_to(
            np.arange(n_features),
            X_chunk.shape
        )
        np.add.at(
            self.hist_counts,
            (feature_indices[valid], bin_indices[valid]),
            1
        )


    def mean_result(self):
        """
        Return the running mean of each feature.
        Returns:
        A 1D numpy array containing the mean of each feature. Features with no valid values return NaN.
        Raises:
        ValueError: if no data has been received.
        """
        if self.count is None:
            raise ValueError("No data has been received")

        return np.where(self.count > 0, self.mean, np.nan)

    
    def histogram_result(self):
        """
        Return the accumulated histogram counts and bin edges.
        Returns:
        hist_counts: a 2D numpy array containing the counts for each feature and bin.
        hist_edges: a 1D numpy array containing the histogram bin edges.
        Raises:
        ValueError: if histogram edges are not configured or no data has been received.
        """
        if self.hist_edges is None:
            raise ValueError("Histogram edges are not configured")

        if self.hist_counts is None:
            raise ValueError("No data has been received")

        return self.hist_counts.copy(), self.hist_edges.copy()

    
    def quantile_result(self, q):
        """
        Calculate exact quantiles from the stored data chunks. Previous chunks are stored when track_quantiles is True.
        NaN values are ignored during the calculation.
        Parameters:
        q: float or array of floats, quantile level(s) between 0 and 1.
        Returns:
        A numpy array containing the requested quantiles for each feature.
        Raises:
        ValueError: if no chunks have been stored or q is outside the valid range.
        """
        if not self._quantile_chunks:
            raise ValueError("No data has been received")

        q = np.asarray(q, dtype=float)

        if not np.all(np.isfinite(q)) or np.any((q < 0) | (q > 1)):
            raise ValueError("q must be between 0 and 1")

        data = np.vstack(self._quantile_chunks)

        return np.nanquantile(data, q, axis=0)




    
    def reset(self):
        """
        Reset the accumulated statistics and stored chunks. Clears the running mean, variance, histogram counts,
        and saved quantile data. The histogram edges and track_quantiles setting are kept.
        Returns:
        self: the reset StreamingStats object.
        """
        self.count = None
        self.mean = None
        self.M2 = None
        self.hist_counts = None
        self._quantile_chunks = []
        return self



    def variance(self):
        """
        Calculate the running population variance for each feature. Uses the accumulated M2 and valid sample counts.
        Features with no valid values return NaN.
        Returns:
        A 1D numpy array containing the population variance of each feature.
        Raises:
        ValueError: if no data has been received.
        """
        if self.count is None:
            raise ValueError("No data has been received")

        return np.divide(self.M2, self.count, out=np.full_like(self.M2, np.nan), where=self.count > 0)

    def std(self):
        """
        Calculate the running population standard deviation for each feature.
        Returns:
        A 1D numpy array containing the standard deviation of each feature. Features with no valid values return NaN.
        Raises:
        ValueError: if no data has been received.
        """
        return np.sqrt(self.variance())
