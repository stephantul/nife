from __future__ import annotations

import argparse
import json
from enum import Enum
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from datasets import Dataset, Features, Sequence, Value
from tqdm import tqdm


class EmbeddingType(str, Enum):
    POOLED = "pooled"
    MEAN = "mean"


def _pair_stream(
    txt_path: Path,
    emb_path: Path,
    truncate_dim: int | None,
) -> Iterator[tuple[str, np.ndarray]]:
    """Stream aligned (text, embedding_row) pairs from a txt jsonl + tensor file."""
    embs = torch.load(emb_path)
    if truncate_dim is not None:
        embs = embs[:, :truncate_dim]
    embs = embs.float().cpu().numpy()

    with open(txt_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            yield item["text"], embs[i]


def _iter_all_pairs(
    root: Path,
    embedding_type: EmbeddingType,
    truncate_dim: int | None,
    limit: int | None,
) -> Iterator[tuple[str, np.ndarray]]:
    """Iterate all (text, embedding) pairs across the folder in a streaming fashion."""
    if embedding_type == EmbeddingType.POOLED:
        mask = "pooled_{}.pt"
    elif embedding_type == EmbeddingType.MEAN:
        mask = "mean_{}.pt"
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

    total = 0
    for txt_path in tqdm(sorted(root.glob("**/*.txt"))):
        name = txt_path.stem.split("_")[1]
        emb_path = txt_path.parent / mask.format(name)
        if not emb_path.exists():
            raise ValueError(f"Embedding file {emb_path} does not exist")

        for text, emb in _pair_stream(txt_path, emb_path, truncate_dim):
            yield text, emb
            total += 1
            if limit is not None and total >= limit:
                return


def build_parquet_shards_from_folder(
    path: str | Path,
    out_dir: str | Path,
    *,
    embedding_type: str | EmbeddingType = EmbeddingType.POOLED,
    truncate_dim: int | None = None,
    limit: int | None = None,
    rows_per_shard: int = 100_000,
) -> None:
    """Stream over (text, embedding) pairs and write sharded parquet files to disk."""
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    embedding_type = EmbeddingType(embedding_type)

    pair_iter = _iter_all_pairs(path, embedding_type, truncate_dim, limit)
    try:
        first_text, first_emb = next(pair_iter)
    except StopIteration:
        raise RuntimeError("No data found under the given path.")

    D = int(first_emb.shape[0])
    features = Features(
        {
            "sentence": Value("string"),
            "label": Sequence(Value("float32"), length=D),
        }
    )

    # Start first buffer with the peeked row
    buf_sentences = [first_text]
    buf_labels = [first_emb.astype("float32")]

    shard_id = 0
    rows_emitted = 1

    def _flush_buffer() -> None:
        nonlocal shard_id, buf_sentences, buf_labels
        if not buf_sentences:
            return
        ds = Dataset.from_dict({"sentence": buf_sentences, "label": np.vstack(buf_labels)}, features=features)
        # Write a single parquet file per shard for simple globbing later
        shard_path = out_dir / f"shard_{shard_id:05d}.parquet"
        ds.to_parquet(str(shard_path))
        # release memory
        del ds
        buf_sentences.clear()
        buf_labels.clear()
        shard_id += 1

    for text, emb in pair_iter:
        buf_sentences.append(text)
        buf_labels.append(emb.astype("float32"))
        rows_emitted += 1
        if len(buf_sentences) >= rows_per_shard:
            _flush_buffer()

    # Flush remainder
    _flush_buffer()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a folder of text files and embeddings to a dataset.")
    parser.add_argument("input", type=str, help="Input folder containing text files and embeddings.")
    parser.add_argument("output", type=str, help="Output file to save the dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples to process.")
    parser.add_argument(
        "--embedding-type",
        type=str,
        choices=[e.value for e in EmbeddingType],
        default=EmbeddingType.POOLED,
        help="Type of embeddings to use (pooled or mean).",
    )
    parser.add_argument("--truncate-dim", type=int, default=None, help="Truncate embeddings to this dimension.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_parquet_shards_from_folder(
        args.input, args.output, limit=args.limit, embedding_type=args.embedding_type, truncate_dim=args.truncate_dim
    )
