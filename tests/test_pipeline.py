import numpy as np
from numcompute.pipeline import Pipeline, FeatureUnion
from numcompute.preprocessing import (
    ColumnSelector,
    SimpleImputer,
    StandardScaler,
    OneHotEncoder
)
data = np.array([
    ["20", "5.5", "Adelaide", "Computing"],
    ["22", "6.0", "Sydney", "Math"],
    ["", "4.5", "Melbourne", "Computing"]
])

columns = ["Age", "GPA", "City", "Course"]
pipe = Pipeline([
    ("preprocess", FeatureUnion([
        ("numeric", Pipeline([
            ("select", ColumnSelector(columns, ["Age", "GPA"])),
            ("impute", SimpleImputer(0, float)),
            ("scale", StandardScaler())
        ])),

        ("categorical", Pipeline([
            ("select", ColumnSelector(columns, ["City", "Course"])),
            ("impute", SimpleImputer("Unknown", str)),
            ("encode", OneHotEncoder())
        ]))
    ]))
])
result = pipe.fit_transform(data)

print(result)
print(result.shape)
print(result.dtype)

def test_pipeline_shape():
    assert result.shape == (3, 7)


def test_pipeline_dtype():
    assert np.issubdtype(result.dtype, np.number)


def test_pipeline_scaling():
    assert np.allclose(result[:, :2].mean(axis=0), [0, 0])


def test_pipeline_encoding():
    assert np.all(result[:, 2:].sum(axis=1) == 2)

