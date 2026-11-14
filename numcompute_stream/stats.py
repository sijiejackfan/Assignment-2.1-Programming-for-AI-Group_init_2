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