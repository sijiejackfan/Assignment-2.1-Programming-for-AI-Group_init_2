"""
Transformer/Estimator API, Compose, FeatureUnion
- Minimal `Transformer`/`Estimator` protocol:
  - Preprocessors: `fit`, `transform`
  - Models: `fit`, `predict` (no ML model implementation required)
- `Pipeline` chaining
- Example:
  ```python
  pipe = Pipeline([
    ('scale', StandardScaler()),
    ('encode', OneHotEncoder())
  ])
  X_tr = pipe.fit_transform(X)
Sources: 
https://docs.python.org/3/library/copy.html
https://scikit-learn.org/stable/common_pitfalls.html
https://scikit-learn.org/stable/computing/scaling_strategies.html
"""
import numpy as np
from copy import deepcopy

class Pipeline:
    """
    Connect multiple transformers and optionally an estimator to process data in order (predictor needs to be at the last step), with the output of each transformer passed to the next one.

    Parameters:
    steps: a list of tuples containing the name and transformer object for each step.
    Attributes:
    steps: a list of tuples containing the name and transformer object for each step.
    _pending_X: a list of arrays containing buffered input chunks.
    _pending_y: a list of arrays containing buffered target values.
    _pending_count: int, number of samples currently buffered.
    _active_transformers: a copied list of transformers or None, used to keep preprocessing consistent with the active model.
    Raises:
    ValueError: if the steps list is empty.
    AI Assistance: The implementation was adapted from code suggestions provided by ChatGPT (OpenAI, 2026). The code was reviewed and modified
    to fit the package requirements.
    """

    def __init__(self, steps):
        if not steps:
            raise ValueError("Pipeline must contain at least one step")
        self.steps = steps
        self._pending_X = []
        self._pending_y = []
        self._pending_count = 0
        self._active_transformers = None

    def fit(self, data, y=None):
        """
        Fit all transformers in the pipeline in order. Or predictor for the last step. Clears any pending streaming data before fitting.
        For a predictor pipeline, saves a copy of the fitted preprocessing steps for later predictions.
        Parameters:
        data: a 2D numpy array
        y: target values required when the last step is an estimator.
        Returns:
        self: the fitted Pipeline object.
        Raises:
        ValueError: if the predictor is not in the last step
        """
        self._pending_X.clear()
        self._pending_y.clear()
        self._pending_count = 0
        self._active_transformers = None
        processed = data
        for name, transformer in self.steps[:-1]:
            if (hasattr(transformer, "predict")):
                raise ValueError("predictor must be in the last step")
            transformer.fit(processed)
            processed = transformer.transform(processed)

        last = self.steps[-1][1]

        if hasattr(last, "predict"):
            if y is None:
                raise ValueError("Estimator requires y")
            last.fit(processed, y)
        else:
            last.fit(processed)
        if hasattr(self.steps[-1][1], "predict"):
            self._active_transformers = deepcopy(self.steps[:-1])
        return self

    

    def partial_fit(self, X, y=None):
        """
        Update the pipeline using a new data chunk. 
        For transformer-only pipelines, updates each transformer using partial_fit() in order.
        For predictor pipelines, updates preprocessing and buffers the original input chunks and labels. Once the buffer reaches
        the required size, all buffered data is transformed using the latest preprocessing state and the estimator is retrained.
        A deep copy of the preprocessing steps is then saved for consistent predictions until the next retraining.
        Parameters:
        X: a 2D NumPy array containing the new input chunk.
        y: a 1D array containing target values, required for estimators.
        Returns:
        self: the updated Pipeline object.
        Raises:
        ValueError: if labels are missing, input dimensions are invalid, chunk sizes do not match, or the feature count changes.
        TypeError: if the estimator does not support buffered training.
        """
        last_name, last_step = self.steps[-1]
        if hasattr(last_step, "transform"):
            data = X
            for name, step in self.steps:
                step.partial_fit(data)
                data = step.transform(data)
            return self

        if y is None:
            raise ValueError("Labels are required for training")
        if len(self.steps) == 1:
            last_step.partial_fit(X, y)
            return self
        if not hasattr(last_step, "buffer_size"):
            raise TypeError("Estimator must support buffered training")

        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D")

        if len(X) == 0 or len(X) != len(y):
            raise ValueError("Invalid chunk size")

        if self._pending_X:
            if X.shape[1] != self._pending_X[0].shape[1]:
                raise ValueError("Feature count mismatch")
        data = X

        for name, step in self.steps[:-1]:
            step.partial_fit(data)
            data = step.transform(data)
        self._pending_X.append(X.copy())
        self._pending_y.append(y.copy())
        self._pending_count += len(y)
        if self._pending_count >= last_step.buffer_size:
            X_train = np.vstack(self._pending_X)
            y_train = np.concatenate(self._pending_y)
            for name, step in self.steps[:-1]:
                X_train = step.transform(X_train)
            last_step.fit(X_train, y_train)
            self._active_transformers = deepcopy(self.steps[:-1])
            self._pending_X.clear()
            self._pending_y.clear()
            self._pending_count = 0
        return self



    def transform(self, data):
        """
        Transform the data through all transformers in order.
        Parameters:
        data: a 2D numpy array containing the input data.
        Returns:
        a transformed 2D numpy array containing the output of the last transformer.
        """
        last = self.steps[-1][1]
        if hasattr(last, "predict"):
            raise ValueError("Pipeline ends with an predictor")
        processed = data
        for name, transformer in self.steps:
            processed = transformer.transform(processed)
        return processed

    def fit_transform(self, data):
        """
        Combine fitting and transforming together
        Parameters:
        data: a 2D numpy array
        Returns:
        a transformed 2D numpy array containing the output of the last transformer.
        Raises:
        same as in fit and transform
        """
        last = self.steps[-1][1]

        if hasattr(last, "predict"):
            raise ValueError("Cannot fit_transform an predictor pipeline")

        self.fit(data)
        return self.transform(data)

    def predict(self, X):
        """
        Make predictions using the estimator in the last pipeline step. Transforms the input using the preprocessing snapshot
        saved when the active model was trained. If no snapshot exists, uses the current preprocessing steps.
        Parameters:
        X: a 2D NumPy array containing input data.
        Returns:
        Predicted values returned by the final estimator.
        Raises:
        TypeError: if the last step does not support predict().
        """
        last_name, model = self.steps[-1]
        if not hasattr(model, "predict"):
            raise TypeError("Final step does not support predict")
        data = X
        if self._active_transformers is not None:
            transformers = self._active_transformers
        else:
            transformers = self.steps[:-1]
        for name, step in transformers:
            data = step.transform(data)

        return model.predict(data)

    
    def predict_proba(self, X):
        """
        Predict class probabilities using the final estimator. Uses the saved preprocessing snapshot if available,
        keeping the input transformation consistent with the active trained model.
        Parameters:
        X: a 2D NumPy array containing input data.
        Returns:
        A 2D NumPy array containing predicted class probabilities.
        Raises:
        TypeError: if the final estimator does not support predict_proba().
        """
        last_name, model = self.steps[-1]
        if not hasattr(model, "predict_proba"):
            raise TypeError("Final step does not support predict_proba")
        data = X
        if self._active_transformers is not None:
            transformers = self._active_transformers
        else:
            transformers = self.steps[:-1]
        for name, step in transformers:
            data = step.transform(data)

        return model.predict_proba(data)





class FeatureUnion:
    """
    Process the same input data through multiple transformers separately, and combine their results into one numpy array.

    Parameters:
    transformers: a list of tuples containing the name and transformer object for each processing branch.
    Attributes:
    transformers: the list of transformers used for processing.
    Raises:
    ValueError: if the transformers list is empty.
    AI Assistance: The implementation was adapted from code suggestions provided by ChatGPT (OpenAI, 2026). The code was reviewed and modified
    to fit the package requirements.
    """

    def __init__(self, transformers):
        if not transformers:
            raise ValueError("FeatureUnion requires at least one transformer")
        self.transformers = transformers

    def fit(self, data):
        """
        Fit each transformer using the same input data.
        Parameters:
        data: a 2D numpy array containing the input data.
        Returns:
        self: the fitted FeatureUnion object.
        """
        for name, transformer in self.transformers:
            transformer.fit(data)
        return self

    
    def partial_fit(self, X, y=None):
        """
        Incrementally update each transformer branch using a new data chunk.
        Each branch receives the same input data and updates its own preprocessing state.
        Parameters:
        X: a 2D NumPy array containing the new data chunk.
        y: optional target values, not used by the transformer branches.
        Returns:
        self: the updated FeatureUnion object.
        Raises:
        TypeError: if any transformer does not support partial_fit().
        """
        for name, transformer in self.transformers:
            if not hasattr(transformer, "partial_fit"):
                raise TypeError(
                    f"Transformer '{name}' does not support partial_fit"
                )

            transformer.partial_fit(X)

        return self


    def transform(self, data):
        """
        Transform the input data through each transformer separately, then combine the results using np.hstack().
        Parameters:
        data: a 2D numpy array containing the input data.
        Returns: a combined 2D numpy array with the integrated data.
        Raises:
        ValueError: if any transformer output is not 2D.
        """
        results = []
        for name, transformer in self.transformers:
            result = transformer.transform(data)

            if np.ndim(result) != 2:
                raise ValueError("Transformer output must be 2D")
            results.append(result)

        return np.hstack(results)

    def fit_transform(self, data):
        """
        Combine fitting and transforming together
        Parameters:
        data: a 2D numpy array
        Returns:
        a combined 2D numpy array with the integrated data.
        Raises:
        same as in fit and transform
        """
        self.fit(data)
        return self.transform(data)