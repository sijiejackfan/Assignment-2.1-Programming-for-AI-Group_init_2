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

class Pipeline:

    def __init__(self, steps):
        if not steps:
            raise ValueError("Pipeline must contain at least one step")
        self.steps = steps

    def fit(self, X):
        data = X
        for name, transformer in self.steps[:-1]:
            transformer.fit(data)
            data = transformer.transform(data)

        last_transformer = self.steps[-1][1]
        last_transformer.fit(data)
        return self

    def transform(self, X):
        data = X
        for name, transformer in self.steps:
            data = transformer.transform(data)
        return data

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

import numpy as np

class FeatureUnion:

    def __init__(self, transformers):
        if not transformers:
            raise ValueError("FeatureUnion requires at least one transformer")
        self.transformers = transformers

    def fit(self, X):
        for name, transformer in self.transformers:
            transformer.fit(X)
        return self

    def transform(self, X):
        results = []
        for name, transformer in self.transformers:
            result = transformer.transform(X)

            if np.ndim(result) != 2:
                raise ValueError("Transformer output must be 2D")
            results.append(result)
            
        return np.hstack(results)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)