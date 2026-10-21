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