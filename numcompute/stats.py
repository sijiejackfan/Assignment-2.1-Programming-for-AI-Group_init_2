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
    return np.nanmean(data, axis=axis)

def median(data, axis=0):
    return np.nanmedian(data, axis=axis)

def std(data, axis=0):
    return np.nanstd(data, axis=axis)

def minimum(data, axis=0):
    return np.nanmin(data, axis=axis)

def maximum(data, axis=0):
    return np.nanmax(data, axis=axis)

def histogram(data, bins=10):
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    return np.histogram(data, bins=bins)

def quantile(data, q, axis=0):
    return np.nanquantile(data, q, axis=axis)