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

"""
import numpy as np

class Pipeline:
    """
    Connect multiple transformers and optionally an estimator to process data in order and process data in order (predictor needs to be at the last step), with the output of each transformer passed to the next one.

    Parameters:
    steps: a list of tuples containing the name and transformer object for each step.
    Attributes：
    steps: a list of tuples containing the name and transformer object for each step.
    Raises:
    ValueError: if the steps list is empty.
    AI Assistance: The implementation was adapted from code suggestions provided by ChatGPT (OpenAI, 2026). The code was reviewed and modified
    to fit the package requirements.
    """

    def __init__(self, steps):
        if not steps:
            raise ValueError("Pipeline must contain at least one step")
        self.steps = steps

    def fit(self, data, y=None):
        """
        Fit all transformers in the pipeline in order. Or predictor for the last step. 
        Parameters:
        data: a 2D numpy array
        Returns
        self: the fitted Pipeline object.
        """
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
        Returns
        a transformed 2D numpy array containing the output of the last transformer.
        Raises
        same as in fit and transform
        """
        last = self.steps[-1][1]

        if hasattr(last, "predict"):
            raise ValueError("Cannot fit_transform an predictor pipeline")

        self.fit(data)
        return self.transform(data)

    def predict(self, data):
        """
        Use predict if the last step in the pipeline is a predictor. Only works when predictor is the last step
        Parameters:
        data: a 2D numpy array
        Returns
        predicted values using the given predict() API from predictor
        Raises
        ValueError: if the last step is not a predictor yet predict is used
        """
        last = self.steps[-1][1]
        if not hasattr(last, "predict"):
            raise ValueError("Last step must support predict")

        processed = data

        for name, transformer in self.steps[:-1]:
            processed = transformer.transform(processed)

        return last.predict(processed)



class FeatureUnion:
    """
    Process the same input data through multiple transformers separately, and combine their results into one numpy array.

    Parameters:
    transformers: a list of tuples containing the name and transformer object for each processing branch.
    Attributes:
    transformers: the list of transformers used for processing.
    Raises:
    ValueError: if the transformers list is empty.
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