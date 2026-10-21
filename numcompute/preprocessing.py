"""
StandardScaler, MinMaxScaler, Imputer, OneHot
- Implement:
  - `StandardScaler` (z-score standardization)
  - `MinMaxScaler` (scale to a given range)
  - `OneHotEncoder` (for categorical variables)
  - *(Optional)* Simple imputer to replace NaN with a constant
- API:
  - `fit(X) -> self`
  - `transform(X) -> X_out`
  - `fit_transform(X) -> X_out`
- Fully vectorized using NumPy; avoid Python loops
- Sources:
https://www.geeksforgeeks.org/machine-learning/standardscaler-minmaxscaler-and-robustscaler-techniques-ml/
https://www.geeksforgeeks.org/machine-learning/ml-one-hot-encoding/

"""
import numpy as np


class StandardScaler:

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data, axis = 0)
        self.std = np.std(data, axis = 0)
        return self

    def transform(self, data):
        transformed = (data - self.mean) / self.std
        return transformed

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)




class MinMaxScaler:

    def __init__(self, lower = 0, upper = 1):
        if lower >= upper:
           raise ValueError("lower must be smaller than upper")
        self.min = None
        self.max = None
        self.lower = lower
        self.upper = upper

    def fit(self, data):
        self.min = np.min(data, axis = 0)
        self.max = np.max(data, axis = 0)
        return self

    def transform(self, data):
        data_range = self.max - self.min
        data_range = np.where(data_range == 0, 1, data_range)
        transformed = (data - self.min) / data_range
        transformed = transformed * (self.upper - self.lower) + self.lower
        return transformed

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)



class OneHotEncoder:

    def __init__(self):
        self.categories = None

    def fit(self, data):
        self.categories = np.unique(data[:, 0])
        return self

    def transform(self, data):
        transformed = (data[:, [0]] == self.categories).astype(int)
        return transformed

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


class SimpleImputer:
    pass