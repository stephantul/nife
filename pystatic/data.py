import random
from pathlib import Path
from typing import Sequence, TypeVar, cast

import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from huggingface_hub import HfApi

T = TypeVar("T", Dataset, IterableDataset)


def _post_process_dataset(dataset: T, to_keep: set[str]) -> T:
    """Post-process the dataset."""
    # No need to rename if the columns are not in the dataset.
    if "sentence" in to_keep:
        dataset = dataset.rename_column("text", "sentence")
    if "label" in to_keep:
        dataset = dataset.rename_column("embedding", "label")
    column_names = dataset.column_names
    assert column_names is not None
    columns_to_drop = [c for c in column_names if c not in to_keep]
    dataset = dataset.remove_columns(columns_to_drop)
    return dataset


def _get_shards_from_dataset(path_or_repo: Path) -> list[Path]:
    """Get all parquet shards in the given path."""
    if path_or_repo.is_dir():
        return list(path_or_repo.glob("train/*.parquet"))
    api = HfApi()
    local_path = api.snapshot_download(repo_id=path_or_repo.as_posix(), repo_type="dataset")
    path_or_repo = Path(local_path)
    return list(path_or_repo.glob("train/*.parquet"))


def get_datasets(
    paths: Sequence[Path],
    in_memory: bool = True,
    limit_shards: int | None = None,
    columns_to_keep: set[str] = {"sentence", "label"},
) -> tuple[IterableDataset | DatasetDict, int]:
    """
    Gets datasets from the given paths.

    The datasets can be loaded in memory or streamed from disk. If it is streamed, we load the
    datasets as collections of parquet shards. In either case, we assume that the datasets have
    a "train" split. In all cases, we assume that the datasets have "text" and "embedding"
    columns, which we rename to "sentence" and "label" respectively. We drop all other columns
    except those specified in `columns_to_keep`, which are "sentence" and "label" by default.

    Arguments:
    ---------
    paths (Sequence[Path]): Paths to the datasets.
    in_memory (bool): Whether to load the datasets in memory or stream them from disk.
    limit_shards (int | None): If streaming, limit the number of shards to load from each dataset.
    columns_to_keep (set[str]): Columns to keep in the dataset.

    """
    length = 0
    if in_memory:
        datasets = {}
        for path in paths:
            dataset = cast(Dataset, load_dataset(path=str(path), split="train"))
            length += len(dataset)
            dataset = _post_process_dataset(dataset, to_keep=columns_to_keep)
            datasets[path.stem] = dataset
        return DatasetDict(datasets), length

    all_shards: list[Path] = []
    for path in paths:
        shards_in_path = _get_shards_from_dataset(path)
        if limit_shards is not None:
            shards_in_path = shards_in_path[:limit_shards]
        all_shards.extend(shards_in_path)
    for shard in all_shards:
        length += pq.read_metadata(shard).num_rows
    random.shuffle(all_shards)
    all_shards_as_posix = [path.as_posix() for path in all_shards]
    dataset = cast(IterableDataset, load_dataset("parquet", data_files=all_shards_as_posix, streaming=True))
    dataset = _post_process_dataset(dataset, to_keep=columns_to_keep)

    return dataset, length
