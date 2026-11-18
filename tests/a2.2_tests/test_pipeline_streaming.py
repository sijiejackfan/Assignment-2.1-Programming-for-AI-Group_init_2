import numpy as np

from numcompute_stream.pipeline import Pipeline, FeatureUnion
from numcompute_stream.preprocessing import (
    ColumnSelector, StandardScaler, OneHotEncoder
)
from numcompute_stream.tree import DecisionTreeClassifier


def test_transformer_pipeline_partial_fit():
    pipe = Pipeline([("scale", StandardScaler())])
    pipe.partial_fit(np.array([[1.0], [3.0]]))
    pipe.partial_fit(np.array([[5.0]]))
    assert np.allclose(pipe.transform(np.array([[3.0]])), [[0]])


def test_feature_union_partial_fit():
    cols = ["Age", "City"]
    union = FeatureUnion([
        ("numeric", Pipeline([
            ("select", ColumnSelector(cols, ["Age"])),
            ("scale", StandardScaler()),
        ])),
        ("categorical", Pipeline([
            ("select", ColumnSelector(cols, ["City"])),
            ("encode", OneHotEncoder()),
        ])),
    ])
    union.partial_fit(np.array([["20", "A"], ["30", "B"]]))
    result = union.transform(np.array([["25", "A"]]))
    assert result.shape == (1, 3)


def test_pipeline_buffered_training():
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("tree", DecisionTreeClassifier(buffer_size=4)),
    ])
    pipe.partial_fit(np.array([[0], [1]]), np.array([0, 0]))
    assert pipe.steps[-1][1].root is None
    pipe.partial_fit(np.array([[10], [11]]), np.array([1, 1]))
    assert pipe.steps[-1][1].root is not None


def test_pipeline_preprocessing_snapshot():
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("tree", DecisionTreeClassifier(buffer_size=4)),
    ])
    pipe.partial_fit(np.array([[0], [1], [10], [11]]), np.array([0, 0, 1, 1]))
    before = pipe.predict(np.array([[0], [11]]))
    pipe.partial_fit(np.array([[100]]), np.array([0]))
    after = pipe.predict(np.array([[0], [11]]))
    assert np.array_equal(before, after)
    assert pipe._active_transformers is not None


def test_tree_only_pipeline_partial_fit():
    pipe = Pipeline([("tree", DecisionTreeClassifier(buffer_size=2))])
    pipe.partial_fit(np.array([[0], [1]]), np.array([0, 1]))
    assert pipe.steps[-1][1].root is not None


def test_pipeline_predict_proba():
    pipe = Pipeline([
        ("tree", DecisionTreeClassifier(max_depth=0))
    ])

    X = np.array([[0], [1], [2]])
    y = np.array([0, 1, 1])

    pipe.fit(X, y)
    result = pipe.predict_proba([[3]])

    assert result.shape == (1, 2)
    assert np.allclose(result[0], [1/3, 2/3])


def test_pipeline_predict_proba_snapshot():
    pipe = Pipeline([
        ("encode", OneHotEncoder()),
        ("tree", DecisionTreeClassifier())
    ])

    X = np.array([["Adelaide"], ["Sydney"]])
    y = np.array([0, 1])

    pipe.fit(X, y)
    pipe.steps[0][1].partial_fit(np.array([["Perth"]]))
    result = pipe.predict_proba(np.array([["Adelaide"]]))

    assert result.shape == (1, 2)
    assert np.isclose(result.sum(), 1.0)
