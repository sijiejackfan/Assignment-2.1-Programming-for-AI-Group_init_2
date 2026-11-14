"""
CSV reader (streaming/chunking), dtype handling
- Load CSV files using `numpy.loadtxt` or `numpy.genfromtxt`
- Support custom delimiters (e.g., comma, tab)
- Handle missing values (skip or fill with a placeholder value)
- Return data as NumPy arrays

"""
import numpy as np
from itertools import islice
import csv

"""
Read csv file into numpy array using np.genfromtxt
Parameters:
fileName: str, csv file name to read from
delim: str, type of the delimiter (optional)
Returns
data : a 2D string array with shape (n_rows, n_columns). Missing values are kept as empty strings.
columns : a string list containing names read from csv header
missing values: kept as empty strings and can be handled later during preprocessing.
Sources: https://numpy.org/devdocs/reference/generated/numpy.genfromtxt.html
"""
def load_csv(fileName, delim=","):
    with open(fileName, "r", encoding="utf-8-sig", newline="") as file:
        columns = next(csv.reader(file, delimiter=delim))
        data = np.genfromtxt(file, delimiter=delim, dtype=str, ndmin=2, autostrip=True)
    return data, columns

"""
Read csv file into numpy array using np.genfromtxt in chunks
Parameters:
fileName: str, csv file name to read from
delim: str, type of the delimiter (optional)
chunk_size: int, how large the chunk should be
Returns
data : a 2D string array with shape (n_rows, n_columns). Missing values are kept as empty strings.
columns : a string list containing names read from csv header
missing values: kept as empty strings and can be handled later during preprocessing.
Raises
TypeError: if chunk size is not integer
ValueError: if chunk size is not positive
"""
def load_csv_chunks(fileName, delim=",", chunk_size=1000):
    if not isinstance(chunk_size, int):
        raise TypeError("chunk size must be integer")
    elif chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    with open(fileName, "r", encoding="utf-8") as file:
        columns = next(csv.reader(file, delimiter=delim))

        while True:
            lines = list(islice(file, chunk_size))

            if not lines:
                break

            data = np.genfromtxt(lines, delimiter=delim, dtype=str, ndmin=2, autostrip=True)

            if data.size > 0:
                yield data, columns

