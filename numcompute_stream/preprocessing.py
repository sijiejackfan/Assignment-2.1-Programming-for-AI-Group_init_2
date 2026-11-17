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
from numcompute_stream.stats import StreamingStats

class ColumnSelector:
    """
    Let user select specific columns from the processed 2D numpy array. Used for separating numerical and categorical columns for processing

    Parameters:
    columns : a string list, names of all columns in the input data. (this would be returned from io result)
    selected : a string list, names of the columns to select.
    Attributes:
    columns : a string list, names of all columns in the input data. (this would be returned from io result)
    selected : a string list, names of the columns to select.
    indices : a int list or None, Indices of selected columns, determined during fit().
    AI Assistance: The implementation was adapted from code suggestions provided by ChatGPT (OpenAI, 2026). The code was reviewed and modified
    to fit the package requirements.
    """
    def __init__(self, columns, selected):
        self.columns = columns
        self.selected = selected
        self.indices = None

    
    def partial_fit(self, data):
        """
        Update column selection for a new data chunk. The column names and selection remain the same across chunks.
        Parameters:
        data: a 2D NumPy array containing the new data chunk.
        Returns:
        self: the updated ColumnSelector object.
        Raises:
        ValueError: if the input is not 2D, the number of columns does not match, or a selected column cannot be found.
        """
        return self.fit(data)


    def fit(self, data):
        """
        Find and store the indices of selected columns.
        Parameters:
        data: a 2D numpy array
        Returns:
        self: the fitted ColumnSelector object.
        Raises:
        ValueError: if the input is not 2D, the number of columns does not match, or a selected column cannot be found.
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
        Returns:
        a numpy array containing the selected columns
        Raises:
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
        Returns:
        a numpy array containing the selected columns
        Raises:
        same as in fit and transform
        """
        self.fit(data)
        return self.transform(data)

class SimpleImputer:
    """
    Process columns and filled missing values, and transform to float array in needed. 
    Supports replacing missing values with a constant value or the running mean of each numerical column. 
    Missing values include empty strings and NaN values.

    Parameters:
    fill_value: value used to fill missing, default 0
    dtype: type of the data (string or float). Used to determine what type the array should be
    strategy: str, "constant" or "mean" (default "constant")
    Attributes:
    fill_value: value used to fill missing, default 0
    dtype: type of the data (string or float). Used to determine what type the array should be
    strategy: the selected imputation strategy.
    stats: a StreamingStats object used for the mean strategy.
    n_features: int or None, number of columns determined during fitting.
    AI Assistance: The implementation was adapted from code suggestions provided by ChatGPT (OpenAI, 2026). The code was reviewed and modified
    to fit the package requirements.
    """
    def __init__(self, fill_value=0, dtype=float, strategy="constant"):
        if strategy not in ("constant", "mean"):
            raise ValueError("Invalid imputation strategy")

        if dtype not in (float, str):
            raise ValueError("dtype must be float or str")

        if strategy == "mean" and dtype != float:
            raise ValueError("Mean strategy requires float dtype")
        self.fill_value = fill_value
        self.dtype = dtype
        self.strategy = strategy
        self.stats = StreamingStats(track_quantiles=False)
        self.n_features = None

    def partial_fit(self, data):
        """
        Update the imputer using a new chunk of data.
        For the mean strategy, the running mean is updated usingthe new numerical values. 
        For the constant strategy, only the input shape is checked and stored.
        Parameters:
        data: a 2D NumPy array containing the new data chunk.
        Returns:
        self: the updated SimpleImputer object.
        Raises:
        ValueError: if the input is not 2D, is empty, has a different number of columns, or contains infinity
        when using the mean strategy.
        """
        data = np.asarray(data)

        if data.ndim != 2:
            raise ValueError("Input must be 2D")

        if 0 in data.shape:
            raise ValueError("Input cannot be empty")

        if self.n_features is not None:
            if data.shape[1] != self.n_features:
                raise ValueError("Feature count does not match")

        if self.strategy == "mean":
            cleaned = np.char.strip(data.astype(str))
            missing = ((cleaned == "") | (np.char.lower(cleaned) == "nan"))

            numeric = np.where(missing, "nan", cleaned).astype(float)

            if np.isinf(numeric).any():
                raise ValueError("Input cannot contain infinity")
            self.stats.update_stats(numeric)

        self.n_features = data.shape[1]
        return self

    def fit(self, data):
        """
        Fit the imputer using the provided data. Clears any previous statistics and then processes the
        input data. For the mean strategy, a new mean is calculated from the provided data.
        Parameters:
        data: a 2D NumPy array containing the input data.
        Returns:
        self: the fitted SimpleImputer object.
        Raises:
        ValueError: if the input is not 2D, is empty, or contains invalid numerical values when using the mean strategy.
        """
        self.stats.reset()
        self.n_features = None

        return self.partial_fit(data)

    def transform(self, data):
        """
        Replace missing values using the selected strategy. Numerical data is converted to float. Missing values
        are replaced with either the constant fill value or the running mean of each column. If a column has no
        valid values, the mean strategy uses fill_value instead.
        Parameters:
        data: a 2D NumPy array containing the input data.
        Returns:
        A 2D NumPy array with missing values filled and converted to the selected data type.
        Raises:
        ValueError: if the imputer has not been fitted, the input shape does not match, or numerical data contains
        invalid values or infinity.
        """
        if self.n_features is None:
            raise ValueError("SimpleImputer must be fitted first")

        data = np.asarray(data)

        if data.ndim != 2 or data.shape[1] != self.n_features:
            raise ValueError("Input shape does not match")

        cleaned = np.char.strip(data.astype(str))

        missing = ((cleaned == "") | (np.char.lower(cleaned) == "nan"))

        if self.dtype == float:
            numeric = np.where(missing, "nan", cleaned).astype(float)
            if np.isinf(numeric).any():
                raise ValueError("Input cannot contain infinity")

            if self.strategy == "mean":
                fill = self.stats.mean_result()
                fill = np.where(np.isnan(fill), float(self.fill_value), fill)
            else:
                fill = float(self.fill_value)
            return np.where(np.isnan(numeric), fill, numeric)
        return np.where(missing, str(self.fill_value), data.astype(str))


    def fit_transform(self, data):
        """
        Combine fitting and transforming together
        Parameters:
        data: a 2D numpy array
        Returns:
        a transformed numpy array with filled values and correct data type
        Raises:
        same as in fit and transform
        """
        self.fit(data)
        return self.transform(data)

class StandardScaler:
    """
    Process numeric columns and standardize with z-score scaler. Uses the mean and standard deviation of each column to
    transform the data. Supports incremental updates through StreamingStats, allowing statistics to be updated with
    new data chunks without resetting previous values.

    Parameters:
    none
    Attributes:
    stats: a StreamingStats object used to calculate running statistics.
    mean: a 1D NumPy array or None, running mean of each column.
    std: a 1D NumPy array or None, running standard deviation of each column.
    """

    def __init__(self):
        self.stats = StreamingStats(track_quantiles=False)
        self.mean = None
        self.std = None

    def partial_fit(self, data):
        """
        Update the mean and standard deviation using a new data chunk. Uses StreamingStats to combine the new data with the
        previously calculated statistics. NaN values are ignored when calculating the mean and standard deviation.
        Parameters:
        data: a 2D numerical NumPy array containing new samples.
        Returns:
        self: the updated StandardScaler object.
        Raises:
        ValueError: if the input is not 2D, is empty, or contains infinity.
        """
        data = np.asarray(data, dtype=float)
        if np.isinf(data).any():
            raise ValueError("Input cannot contain infinity")
        self.stats.update_stats(data)
        self.mean = self.stats.mean_result()
        self.std = self.stats.std()
        return self

    def fit(self, data):
        """
        Uses partialfit to calculate and store the mean and standard deviation of each column, reset before doing so.
        Parameters:
        data: a 2D numpy array
        Returns:
        self: a fitted StandScaler
        Raises:
        ValueError: if dimensions are not 2 
        """
        self.stats.reset()
        return self.partial_fit(data)

    def transform(self, data):
        """
        Standardize the input data using the stored statistics. Subtracts the running mean and divides by the standard
        deviation of each column. Columns with zero standard deviation use 1 to avoid division by zero.
        Parameters:
        data: a 2D numerical NumPy array to be transformed.
        Returns:
        A 2D NumPy array containing the standardized values.
        Raises:
        ValueError: if the scaler has not been fitted, the input is not 2D, the feature count does not match, or
        the input contains infinity.
        """
        if self.mean is None:
            raise ValueError("StandardScaler must be fitted first")
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError("Input must be 2D")
        if data.shape[1] != len(self.mean):
            raise ValueError("Feature count does not match")
        if np.isinf(data).any():
            raise ValueError("Input cannot contain infinity")
        std = np.where(self.std == 0, 1, self.std)
        return (data - self.mean) / std

    def fit_transform(self, data):
        """
        Combine fitting and transforming together
        Parameters:
        data: a 2D numpy array
        Returns:
        a transformed 2D numpy array with standardized values
        Raises:
        same as in fit and transform
        """
        self.fit(data)
        return self.transform(data)

    




class MinMaxScaler:
    """
    Process numeric columns and standardize to a given range. Uses the minimum and maximum values of each column to
    scale the data. Supports incremental updates, allowing the minimum and maximum to be updated with new data chunks.

    Parameters:
    lower: lower bound of the standardized range, default 0
    upper: upper bound of the standardized range, default 1
    Attributes:
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

    def partial_fit(self, data):
        """
        Update the minimum and maximum values using a new data chunk.
        Compares the new values with the previously stored minimum and maximum of each column. NaN values are
        ignored when calculating these statistics.
        Parameters:
        data: a 2D numerical NumPy array containing new samples.
        Returns:
        self: the updated MinMaxScaler object.
        Raises:
        ValueError: if the input is not 2D, is empty, has a different number of features, or contains infinity.
        """
        data = np.asarray(data, dtype=float)

        if data.ndim != 2:
            raise ValueError("Input must be 2D")

        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError("Input cannot be empty")

        if np.isinf(data).any():
            raise ValueError("Input cannot contain infinity")

        if self.min is not None and data.shape[1] != len(self.min):
            raise ValueError("Feature count does not match")
        chunk_min = np.min(np.where(np.isnan(data), np.inf, data), axis=0)
        chunk_max = np.max(np.where(np.isnan(data), -np.inf, data),axis=0)

        if self.min is None:
            self.min = chunk_min
            self.max = chunk_max
        else:
            self.min = np.minimum(self.min, chunk_min)
            self.max = np.maximum(self.max, chunk_max)
        return self

    def fit(self, data):
        """
        Uses partialfit to calculate and store the minimum and maximum of each column, reset before doing so.
        Parameters:
        data: a 2D numpy array
        Returns:
        self: a fitted MinMaxScaler
        Raises:
        ValueError: if dimensions are not 2 
        """
        self.min = None
        self.max = None
        return self.partial_fit(data)

    def transform(self, data):
        """
        Scale the input data using the stored minimum and maximum. Transforms numerical values to the specified range.
        Columns with zero range use 1 to avoid division by zero. Columns with no valid training values return NaN.
        Parameters:
        data: a 2D numerical NumPy array to be transformed.
        Returns:
        A 2D NumPy array containing the scaled values.
        Raises:
        ValueError: if the scaler has not been fitted, the input is not 2D, the feature count does not match, or
        the input contains infinity.
        """
        if self.min is None:
            raise ValueError("MinMaxScaler must be fitted first")
        data = np.asarray(data, dtype=float)

        if data.ndim != 2:
            raise ValueError("Input must be 2D")
        
        if data.shape[1] != len(self.min):
            raise ValueError("Feature count does not match")
        
        if np.isinf(data).any():
            raise ValueError("Input cannot contain infinity")
        
        valid = np.isfinite(self.min) & np.isfinite(self.max)
        data_min = np.where(valid, self.min, 0.0)
        data_max = np.where(valid, self.max, 0.0)
        data_range = data_max - data_min
        data_range = np.where(data_range == 0, 1, data_range)
        transformed = (data - data_min) / data_range
        transformed = transformed * (self.upper - self.lower) + self.lower
        return np.where(valid, transformed, np.nan)

    def fit_transform(self, data):
        """
        Combine fitting and transforming together
        Parameters:
        data: a 2D numpy array
        Returns:
        A 2D NumPy array containing values scaled to the specified range.
        Raises:
        same as in fit and transform
        """
        self.fit(data)
        return self.transform(data)



class OneHotEncoder:
    """
    Convert categorical data into numerical data using one-hot encoding. Each category is represented by a separate column containing
    0 or 1. Supports incremental updates, allowing new categories to be added when new data chunks are received.

    Parameters:
    none
    Attributes:
    categories : a list of numpy arrays, containing unique categories of each column.
    AI Assistance: The implementation was adapted from code suggestions provided by ChatGPT (OpenAI, 2026). The code was reviewed and modified
    to fit the package requirements.
    """
    def __init__(self):
        self.categories = None


    
    def partial_fit(self, data):
        """
        Update the stored categories using a new data chunk. Finds unique categories in each column and adds any new
        categories to the existing ones. Previously stored categories are kept.
        Parameters:
        data: a 2D NumPy array containing categorical data.
        Returns:
        self: the updated OneHotEncoder object.
        Raises:
        ValueError: if the input is not 2D, is empty, or has a different number of features.
        """
        data = np.asarray(data)

        if data.ndim != 2:
            raise ValueError("Input must be 2D")

        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError("Input cannot be empty")

        if self.categories is None:
            self.categories = [np.array([], dtype=data.dtype)
                for _ in range(data.shape[1])
            ]

        elif data.shape[1] != len(self.categories):
            raise ValueError("Feature count does not match")

        for i in range(data.shape[1]):
            new_categories = np.unique(data[:, i])
            unseen = new_categories[~np.isin(new_categories, self.categories[i])]
            self.categories[i] = np.concatenate([self.categories[i], unseen])

        return self

   

    def fit(self, data):
        """
        Uses partialfit to find and store the unique categories each column, reset before doing so.
        Parameters:
        data: a 2D numpy array
        Returns:
        self: a fitted OneHotEncoder
        Raises:
        ValueError: if dimensions are not 2 
        """
        self.categories = None
        return self.partial_fit(data)

    def transform(self, data):
        """
        Convert categorical data into one-hot encoded columns. Each stored category is represented by a separate binary
        column. Categories not seen during fitting are encoded as zeros for that input feature.
        Parameters:
        data: a 2D NumPy array containing categorical data.
        Returns:
        A 2D integer NumPy array containing the encoded data,
        with one output column for each stored category.
        Raises:
        ValueError: if the encoder has not been fitted, the input is not 2D, or the number of features does not match.
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
        Returns:
        A 2D integer NumPy array containing the one-hot encoded data, with one output column for each stored category.
        Raises:
        same as in fit and transform
        """
        self.fit(data)
        return self.transform(data)

