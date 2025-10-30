import logging
from collections.abc import Iterable, Iterator
from typing import TypeVar

logger = logging.getLogger(__name__)


T = TypeVar("T")


def batchify(stream: Iterator[T] | Iterable[T], batch_size: int) -> Iterator[list[T]]:
    """Turn an iterator over something into batches."""
    # If we got an iterable, turn it into an iterator
    if isinstance(stream, Iterable):
        stream = iter(stream)

    batch: list[T] = []
    while True:
        if len(batch) < batch_size:
            try:
                batch.append(next(stream))
            except StopIteration:
                if batch:
                    yield batch
                break
        else:
            yield batch
            batch = []
