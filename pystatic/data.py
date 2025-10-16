from pathlib import Path
from typing import Sequence, cast

import pyarrow.parquet as pq
from datasets import Dataset, IterableDataset, load_dataset
from huggingface_hub import HfApi


def _post_process_dataset(dataset: Dataset | IterableDataset, to_keep: set[str]) -> Dataset | IterableDataset:
    # Rename only if the source exists and the target is kept.
    assert dataset.column_names is not None
    cols = set(dataset.column_names)
    if "text" in cols and "sentence" in to_keep:
        dataset = dataset.rename_column("text", "sentence")
    if "embedding" in cols and "label" in to_keep:
        dataset = dataset.rename_column("embedding", "label")

    assert dataset.column_names is not None
    cols = set(dataset.column_names)
    dataset = dataset.remove_columns([c for c in cols if c not in to_keep])

    return dataset


def _collect_parquet_shards(path_or_repo: Path) -> list[Path]:
    """Return a sorted list of train/*.parquet for local dir or HF dataset repo."""
    if path_or_repo.is_dir():
        shards = list(path_or_repo.glob("train/*.parquet"))
    else:
        api = HfApi()
        local = Path(api.snapshot_download(repo_id=path_or_repo.as_posix(), repo_type="dataset"))
        shards = list(local.glob("train/*.parquet"))
    shards.sort(key=lambda p: p.as_posix())
    return shards


def get_datasets(
    paths: Sequence[Path],
    in_memory: bool = True,
    limit_shards: int | None = None,
    columns_to_keep: set[str] = {"sentence", "label"},
) -> tuple[Dataset | IterableDataset, int]:
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
    shards: list[Path] = []
    for path in paths:
        ps = _collect_parquet_shards(path)
        if limit_shards is not None:
            ps = ps[:limit_shards]
        shards.extend(ps)

    length = sum(pq.read_metadata(p).num_rows for p in shards)

    data_files = [p.as_posix() for p in shards]
    ds = cast(
        Dataset | IterableDataset,
        load_dataset(
            "parquet",
            data_files=data_files,
            split="train",
            streaming=not in_memory,  # the only difference
        ),
    )

    dataset = _post_process_dataset(ds, to_keep=columns_to_keep)
    return dataset, length
