import numpy as np

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.stream import StreamTrainer


def test_stream_trainer_cold_start():
    trainer = StreamTrainer(Pipeline([
        ("tree", DecisionTreeClassifier(buffer_size=2)),
    ]))
    result = trainer.score_chunk(np.array([[0], [1]]), np.array([0, 1]))
    assert result["accuracy"] is None
    assert result["cumulative_accuracy"] is None
    assert len(trainer.history) == 1


def test_stream_trainer_score_and_fit():
    trainer = StreamTrainer(Pipeline([
        ("tree", DecisionTreeClassifier(buffer_size=2)),
    ]))
    trainer.score_chunk(np.array([[0], [1]]), np.array([0, 0]))
    trainer.fit_chunk(np.array([[0], [1]]), np.array([0, 0]))
    result = trainer.score_chunk(np.array([[2], [3]]), np.array([0, 0]))
    assert np.isclose(result["accuracy"], 1.0)
    assert np.isclose(result["cumulative_accuracy"], 1.0)


def test_stream_trainer_time_and_memory():
    trainer = StreamTrainer(Pipeline([
        ("tree", DecisionTreeClassifier(buffer_size=2)),
    ]))
    trainer.score_chunk(np.array([[0], [1]]), np.array([0, 1]))
    trainer.fit_chunk(np.array([[0], [1]]), np.array([0, 1]))
    result = trainer.history[-1]
    assert result["training_time"] >= 0
    assert result["peak_memory_mb"] >= 0
