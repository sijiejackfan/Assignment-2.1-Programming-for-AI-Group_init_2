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
    """
    Let user select specific columns from the processed 2D numpy array. Used for separating numerical and categorical columns for processing
    Parameters:
    columns : a string list, names of all columns in the input data. (this would be returned from io result)
    selected : a string list, names of the columns to select.
    Attributes
    columns : a string list, names of all columns in the input data. (this would be returned from io result)
    selected : a string list, names of the columns to select.
    indices : a int list or None, Indices of selected columns, determined during fit().
    """
    def __init__(self, columns, selected):
        self.columns = columns
        self.selected = selected
        self.indices = None

    def fit(self, data):
        """
        Find and store the indices of selected columns.
        Parameters:
        data: a 2D numpy array
        Returns
        self: ColumnSelector after fitted
        Raises
        ValueError: if dimensions are not 2 or doesn't match with column number, or column name not found
        """
        if not np.ndim(data) == 2:
            raise ValueError("dimensions must be 2")
        if not (np.shape(data)[1] == len(self.columns)):
            raise ValueError("dimensions do not match columns")
        self.indices = []
        for col in self.selected:
            if col not in self.columns:
                raise ValueError("column name not found")
            self.indices.append(self.columns.index(col))
        return self

    def transform(self, data):
        """
        Extract selected columns from input array
        Parameters:
        data: a 2D numpy array
        Returns
        a numpy array containing the selected columns
        Raises
        ValueError: if dimensions are not 2 or doesn't match with column number, or transformed without fitting
        """
        if not np.ndim(data) == 2:
            raise ValueError("dimensions must be 2")
        if not (np.shape(data)[1] == len(self.columns)):
            raise ValueError("dimensions do not match columns")
        if self.indices is None:
            raise ValueError("ColumnSelector must be fitted first")
        return data[:, self.indices]

    def fit_transform(self, data):
        """
        Combine fitting and transforming together
        Parameters:
        data: a 2D numpy array
        Returns
        a numpy array containing the selected columns
        Raises
        same as in fit and transform
        """
        self.fit(data)
        return self.transform(data)

class SimpleImputer:
    """
    Process columns and filled missing values, and transform to float array in needed.

    Parameters:
    fill_value: value used to fill missing, default 0
    dtype: type of the data (string or float). Used to determine what type the array should be
    Attributes
    fill_value: value used to fill missing, default 0
    dtype: type of the data (string or float). Used to determine what type the array should be
    """
    def __init__(self, fill_value=0, dtype=float):
        self.fill_value = fill_value
        self.dtype = dtype

    def fit(self, data):
        """
        Do not really do anything, only to fit API
        Parameters:
        data: a 2D numpy array
        Returns
        self: SimpleImputer after fitted
        Raises
        ValueError: if dimensions are not 2 
        """
        if np.ndim(data) != 2:
            raise ValueError("Input must be 2D")
        return self

    def transform(self, data):
        """
        Fill the array with fill values, and convert to the correct data type for further processing
        Parameters:
        data: a 2D numpy array
        Returns
        a transformed numpy array with filled values and correct data type
        Raises
        ValueError: if dimensions are not 2, type is not float or string
        """
        if np.ndim(data) != 2:
            raise ValueError("Input must be 2D")

        data = np.asarray(data, dtype=str)
        cleaned = np.char.strip(data)
        missing = (cleaned == "") | (np.char.lower(cleaned) == "nan")

        if self.dtype == float:
            fill = float(self.fill_value)
            numeric = np.where(missing, "nan", data).astype(float)
            return np.where(np.isnan(numeric), fill, numeric)

        elif self.dtype == str:
            return np.where(missing, str(self.fill_value), data)

        else:
            raise ValueError("dtype must be float or str")

    def fit_transform(self, data):
        """
        Combine fitting and transforming together
        Parameters:
        data: a 2D numpy array
        Returns
        a transformed numpy array with filled values and correct data type
        Raises
        same as in fit and transform
        """
        self.fit(data)
        return self.transform(data)

class StandardScaler:
    """
    Process numeric columns and standardize with z-score scaler

    Parameters:
    none
    Attributes
    mean: a numpy array, containing the mean of each column
    std: a numpy array, containing the standard deviation of each column
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        """
        Calculate and store the mean and standard deviation of each column
        Parameters:
        data: a 2D numpy array
        Returns
        self: a fitted StandScaler
        Raises
        ValueError: if dimensions are not 2 
        """
        if np.ndim(data) != 2:
            raise ValueError("Input must be 2D")
        self.mean = np.mean(data, axis = 0)
        self.std = np.std(data, axis = 0)
        return self

    def transform(self, data):
        """
        standardize the input data using the mean and standard deviation
        Parameters:
        data: a 2D numpy array
        Returns
        a transformed 2D numpy array with standardized values
        Raises
        ValueError: if dimensions are not 2 
        """
        if np.ndim(data) != 2:
            raise ValueError("Input must be 2D")
        std = np.where(self.std == 0, 1, self.std)
        transformed = (data - self.mean) / std
        return transformed

    def fit_transform(self, data):
        """
        Combine fitting and transforming together
        Parameters:
        data: a 2D numpy array
        Returns
        a transformed 2D numpy array with standardized values
        Raises
        same as in fit and transform
        """
        self.fit(data)
        return self.transform(data)




class MinMaxScaler:
    """
    Process numeric columns and standardize to a given range

    Parameters:
    lower: lower bound of the standardized range, default 0
    upper: upper bound of the standardized range, default 1
    Attributes
    min: a numpy array, containing the minimum of each column
    max: a numpy array, containing the maximum of each column
    lower: lower bound of the standardized range, default 0
    upper: upper bound of the standardized range, default 1
    """

    def __init__(self, lower = 0, upper = 1):
        if lower >= upper:
           raise ValueError("lower must be smaller than upper")
        self.min = None
        self.max = None
        self.lower = lower
        self.upper = upper

    def fit(self, data):
        """
        Calculate and store the min and max of each column
        Parameters:
        data: a 2D numpy array
        Returns
        self: a fitted MinMaxScaler
        Raises
        ValueError: if dimensions are not 2 
        """
        if np.ndim(data) != 2:
            raise ValueError("Input must be 2D")
        self.min = np.min(data, axis = 0)
        self.max = np.max(data, axis = 0)
        return self

    def transform(self, data):
        """
        standardize the input data using the min and max in the given range
        Parameters:
        data: a 2D numpy array
        Returns
        a transformed 2D numpy array with standardized values
        Raises
        ValueError: if dimensions are not 2 
        """
        if np.ndim(data) != 2:
            raise ValueError("Input must be 2D")
        data_range = self.max - self.min
        data_range = np.where(data_range == 0, 1, data_range)
        transformed = (data - self.min) / data_range
        transformed = transformed * (self.upper - self.lower) + self.lower
        return transformed

    def fit_transform(self, data):
        """
        Combine fitting and transforming together
        Parameters:
        data: a 2D numpy array
        Returns
        a transformed 2D numpy array with standardized values
        Raises
        same as in fit and transform
        """
        self.fit(data)
        return self.transform(data)



class OneHotEncoder:
    """
    Convert categorical data into numerical data using one-hot encoding.

    Parameters:
    none
    Attributes
    categories : a list of numpy arrays, containing unique categories of each column.
    """
    def __init__(self):
        self.categories = None
   

    def fit(self, data):
        """
        Find and store the unique categories of each column
        Parameters:
        data: a 2D numpy array
        Returns
        self: a fitted OneHotEncoder
        Raises
        ValueError: if dimensions are not 2 
        """
        if np.ndim(data) != 2:
            raise ValueError("Input must be 2D")
        self.categories = []
        for i in range(data.shape[1]):
            self.categories.append(np.unique(data[:, i]))
        return self

    def transform(self, data):
        """
        Convert categorical columns into one-hot encoded numerical arrays, then the encoded columns are combined into a single 2D array.
        Parameters:
        data: a 2D numpy array
        Returns
        a 2d int array with shape (n_samples, n_features), each value is either 0 or 1.
        Raises
        ValueError: if dimensions are not 2 or input shape do not match, or encoder is not fitted
        """
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
        """
        Combine fitting and transforming together
        Parameters:
        data: a 2D numpy array
        Returns
        a 2d int array with shape (n_samples, n_features), each value is either 0 or 1.
        Raises
        same as in fit and transform
        """
        self.fit(data)
        return self.transform(data)

