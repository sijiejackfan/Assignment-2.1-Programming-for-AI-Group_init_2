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

class ColumnSelector:

    def __init__(self, columns, selected):
        self.columns = columns
        self.selected = selected
        self.indices = None

    def fit(self, X):
        if not np.ndim(X) == 2:
            raise ValueError("dimensions must be 2")
        if not (np.shape(X)[1] == len(self.columns)):
            raise ValueError("dimensions do not match columns")
        self.indices = []
        for col in self.selected:
            if col not in self.columns:
                raise ValueError("column name not found")
            self.indices.append(self.columns.index(col))
        return self

    def transform(self, X):
        if not np.ndim(X) == 2:
            raise ValueError("dimensions must be 2")
        if not (np.shape(X)[1] == len(self.columns)):
            raise ValueError("dimensions do not match columns")
        if self.indices is None:
            raise ValueError("ColumnSelector must be fitted first")
        return X[:, self.indices]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class SimpleImputer:

    def __init__(self, fill_value=0, dtype=float):
        self.fill_value = fill_value
        self.dtype = dtype

    def fit(self, X):
        if np.ndim(X) != 2:
            raise ValueError("Input must be 2D")
        self.fitted = True
        return self

    def transform(self, X):
        if np.ndim(X) != 2:
            raise ValueError("Input must be 2D")

        data = np.asarray(X, dtype=str)
        cleaned = np.char.strip(data)
        missing = (cleaned == "") | (np.char.lower(cleaned) == "nan")

        if self.dtype == float:
            numeric = np.where(missing, "nan", data).astype(float)
            return np.where(np.isnan(numeric), self.fill_value, numeric)

        elif self.dtype == str:
            return np.where(missing, str(self.fill_value), data)

        else:
            raise ValueError("dtype must be float or str")

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class StandardScaler:

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        if np.ndim(data) != 2:
            raise ValueError("Input must be 2D")
        self.mean = np.mean(data, axis = 0)
        self.std = np.std(data, axis = 0)
        return self

    def transform(self, data):
        if np.ndim(data) != 2:
            raise ValueError("Input must be 2D")
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
        if np.ndim(data) != 2:
            raise ValueError("Input must be 2D")
        self.min = np.min(data, axis = 0)
        self.max = np.max(data, axis = 0)
        return self

    def transform(self, data):
        if np.ndim(data) != 2:
            raise ValueError("Input must be 2D")
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
        if np.ndim(data) != 2:
            raise ValueError("Input must be 2D")
        self.categories = []
        for i in range(data.shape[1]):
            self.categories.append(np.unique(data[:, i]))

        return self

    def transform(self, data):
        if self.categories is None:
            raise ValueError("OneHotEncoder must be fitted first")

        if np.ndim(data) != 2 or data.shape[1] != len(self.categories):
            raise ValueError("Input shape does not match fitted data")

        results = []

        for i, categories in enumerate(self.categories):
            encoded = (data[:, [i]] == categories).astype(int)
            results.append(encoded)

        return np.hstack(results)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

