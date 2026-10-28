
import numpy as np
import pytest
from pathlib import Path

from numcompute.io import load_csv, load_csv_chunks

'''
AI Assistance: The implementation was adapted from code suggestions provided by ChatGPT (OpenAI, 2026). AI helped was used for creating some test cases and generate the data for testing.
'''
file = Path(__file__).resolve().parent.parent / "assets" / "students.csv"

def test_load_csv():
    data, columns = load_csv(file)

    assert data.shape == (10, 5)
    assert columns == ["StudentID", "Age", "GPA", "City", "Course"]


def test_load_csv_missing():
    data, columns = load_csv(file)

    assert data[2, 1] == ""
    assert data[3, 2] == ""


def test_load_csv_chunks():
    chunks = list(load_csv_chunks(file, chunk_size=2))

    assert len(chunks) == 5
    assert chunks[0][0].shape == (2, 5)


def test_load_csv_chunks_last():
    chunks = list(load_csv_chunks(file, chunk_size=3))

    assert chunks[-1][0].shape == (1, 5)


def test_load_csv_delimiter(tmp_path):
    test_file = tmp_path / "sample.tsv"
    test_file.write_text("Age\tCity\n20\tAdelaide\n22\tSydney\n")

    data, columns = load_csv(test_file, delim="\t")

    assert data.shape == (2, 2)
    assert data[0, 1] == "Adelaide"


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        list(load_csv_chunks(file, chunk_size=0))