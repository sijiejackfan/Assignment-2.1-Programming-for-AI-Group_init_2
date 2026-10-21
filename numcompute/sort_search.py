import numpy as np

#Stable Sort

def stable_sort(values, axis=-1):
    """We are sorting values while preserving the order of equal elements."""
    arr = np.asarray(values) #converting the input into a NumPy array

    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError("values must contain numeric data") #checking whether the data type is numeric

    return np.sort(arr, axis=axis, kind="stable")
