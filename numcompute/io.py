"""
CSV reader (streaming/chunking), dtype handling
- Load CSV files using `numpy.loadtxt` or `numpy.genfromtxt`
- Support custom delimiters (e.g., comma, tab)
- Handle missing values (skip or fill with a placeholder value)
- Return data as NumPy arrays

"""
import numpy as np
from itertools import islice

"""
Read csv file into numpy array using np.genfromtxt
fileName: csv file name to read from
delim: type of the delimiter
dtype: type of the data
skipheader: number of lines to skip, default 1
missing values: filled with np.nan by default using np.genfromtxt
What's missing: streaming / chunking, 
Sources: https://numpy.org/devdocs/reference/generated/numpy.genfromtxt.html
"""
def load_csv(fileName, delim=","):
    data = np.genfromtxt(fileName, delimiter=delim, dtype=None, names = True, encoding=None, ndmin=2, usemask=True)
    return data

def load_csv_chunks(fileName, delim=",", chunk_size=1000):
    if not isinstance(chunk_size, int):
        raise TypeError("chunk size must be integer")
    elif chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    with open(fileName, "r", encoding="utf-8") as file:
        header = next(file).strip()
        names = header.split(delim)

        while True:
            lines = list(islice(file, chunk_size))

            if not lines:
                break

            data = np.genfromtxt(lines, delimiter=delim, dtype=None, ndmin=2, encoding=None, names = names, usemask=True)

            if data.size > 0:
                yield data

