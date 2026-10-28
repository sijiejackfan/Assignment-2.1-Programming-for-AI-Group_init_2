from numcompute.io import load_csv, load_csv_chunks


def test_load_csv():
    data = load_csv("students.csv")
    assert len(data) > 0


def test_load_csv_chunks():
    data = list(load_csv_chunks("students.csv", chunk_size=2))
    assert len(data) > 0


def test_load_csv_delimiter():
    data = load_csv("students.csv", delim=",")
    assert len(data) > 0
