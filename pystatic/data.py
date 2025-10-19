from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow.parquet as pq
import torch
from datasets import Dataset, Features, IterableDataset, Value, load_dataset
from datasets import Sequence as DatasetSequenceFeature
from huggingface_hub import HfApi
from tqdm import tqdm


def _pair_stream(
    txt_path: Path,
    emb_path: Path,
) -> Iterator[tuple[dict[str, str], np.ndarray]]:
    """Stream aligned (text, embedding_row) pairs from a txt jsonl + tensor file."""
    embs = torch.load(emb_path)
    embs = embs.float().numpy()

    with open(txt_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            yield item, embs[i]


def _iter_all_pairs(
    root: Path,
    limit: int | None,
) -> Iterator[tuple[dict[str, str], np.ndarray]]:
    """Iterate all (text, embedding) pairs across the folder in a streaming fashion."""
    mask = "pooled_{}.pt"

    total = 0
    for txt_path in tqdm(sorted(root.glob("**/*.txt"))):
        name = txt_path.stem.split("_")[1]
        emb_path = txt_path.parent / mask.format(name)
        if not emb_path.exists():
            raise ValueError(f"Embedding file {emb_path} does not exist")

        for record, emb in _pair_stream(txt_path, emb_path):
            yield record, emb
            total += 1
            if limit is not None and total >= limit:
                return


def build_parquet_shards_from_folder(
    path: str | Path,
    out_dir: str | Path,
    *,
    limit: int | None = None,
    rows_per_shard: int = 100_000,
) -> None:
    """Stream over (text, embedding) pairs and write sharded parquet files to disk."""
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_iter = _iter_all_pairs(path, limit)
    try:
        first_record, first_emb = next(pair_iter)
    except StopIteration:
        raise RuntimeError("No data found under the given path.")

    keys = list(first_record.keys())
    first_key = keys[0]
    D = int(first_emb.shape[0])
    features = Features(
        {
            **{k: Value("string") for k in keys},
            "embedding": DatasetSequenceFeature(Value("float32"), length=D),
        }
    )

    # Start first buffer with the peeked row
    buffers = {k: [first_record[k]] for k in keys}
    buf_embeddings = [first_emb.astype("float32")]

    shard_id = 0
    rows_emitted = 1

    def _flush_buffer() -> None:
        nonlocal shard_id, buffers, buf_embeddings
        if not buffers:
            return
        ds = Dataset.from_dict(
            {k: buffers[k] for k in keys} | {"embedding": np.vstack(buf_embeddings)}, features=features
        )
        # Write a single parquet file per shard for simple globbing later
        shard_path = out_dir / "train" / f"shard_{shard_id:05d}.parquet"
        ds.to_parquet(str(shard_path))
        # release memory
        del ds
        buffers.clear()
        buf_embeddings.clear()
        shard_id += 1

    for record, emb in pair_iter:
        for k, v in record.items():
            buffers.setdefault(k, []).append(str(v))
        buf_embeddings.append(emb.astype("float32"))
        rows_emitted += 1
        if len(buffers[first_key]) >= rows_per_shard:
            _flush_buffer()

    # Flush remainder
    _flush_buffer()


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
        shards = list(path_or_repo.glob("**/*.parquet"))
    else:
        api = HfApi()
        local = Path(api.snapshot_download(repo_id=path_or_repo.as_posix(), repo_type="dataset"))
        shards = list(local.glob("**/*.parquet"))
    shards.sort(key=lambda p: p.as_posix())
    return shards


def get_datasets(
    paths: Sequence[Path] | Sequence[str],
    in_memory: bool = True,
    limit_shards: int | None = None,
    columns_to_keep: set[str] = {"sentence", "label", "question"},
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
    paths = [Path(p) for p in paths]
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
            streaming=not in_memory,
        ),
    )

    if not in_memory:
        ds = cast(IterableDataset, ds)
        ds = ds.shuffle(buffer_size=50_000, seed=42)
    else:
        ds = cast(Dataset, ds)
        ds = ds.shuffle(seed=42)

    dataset = _post_process_dataset(ds, to_keep=columns_to_keep)
    return dataset, length
