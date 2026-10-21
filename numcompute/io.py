"""
CSV reader (streaming/chunking), dtype handling
- Load CSV files using `numpy.loadtxt` or `numpy.genfromtxt`
- Support custom delimiters (e.g., comma, tab)
- Handle missing values (skip or fill with a placeholder value)
- Return data as NumPy arrays

"""
import numpy as np

"""
Read csv file into numpy array using np.genfromtxt
fileName: csv file name to read from
delim: type of the delimiter
dtype: type of the data
skipheader: number of lines to skip, default 1
missing values: filled with np.nan by default using np.genfromtxt
What's missing: streaming / chunking, 
"""
def load_csv(fileName, delim, skipheader = 1):
    data = np.genfromtxt(fileName, delimiter=delim, dtype=None, skip_header=skipheader, ndmin=2)
    return data

