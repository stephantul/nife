import pytest

from pynife.utilities import batchify, iterable_iterator_dispatch


@pytest.mark.parametrize(
    "data, result",
    [
        ([1, 2, 3], [1, 2, 3]),
        (iter(range(3)), [0, 1, 2]),
    ],
)
def test_iterable_iterator_dispatch(data, result) -> None:
    """Test iterable_iterator_dispatch with an iterable."""
    i1 = iterable_iterator_dispatch(data)
    assert list(i1) == result


def test_batchify_with_list() -> None:
    """Test batchify with a list."""
    data = [1, 2, 3, 4, 5]
    batch_size = 2
    batches = list(batchify(data, batch_size))
    assert batches == [[1, 2], [3, 4], [5]]


def test_batchify_with_iterator() -> None:
    """Test batchify with an iterator."""
    data = iter([1, 2, 3, 4, 5])
    batch_size = 3
    batches = list(batchify(data, batch_size))
    assert batches == [[1, 2, 3], [4, 5]]


def test_batchify_empty() -> None:
    """Test batchify with an empty iterable."""
    data: list[int] = []
    batch_size = 2
    batches = list(batchify(data, batch_size))
    assert batches == []


def test_batchify_large_batch_size() -> None:
    """Test batchify with a batch size larger than the data."""
    data = [1, 2, 3]
    batch_size = 10
    batches = list(batchify(data, batch_size))
    assert batches == [[1, 2, 3]]
