import argparse
import json
from enum import Enum
from pathlib import Path

import torch
from datasets import Dataset, Features, Sequence, Value
from tqdm import tqdm


class EmbeddingType(str, Enum):
    POOLED = "pooled"
    MEAN = "mean"


def create_dataset_from_folder(
    path: Path | str,
    limit: None | int = None,
    embedding_type: str | EmbeddingType = EmbeddingType.POOLED,
    truncate_dim: int | None = None,
) -> Dataset:
    """Create a dataset from a folder of text files."""
    embedding_type = EmbeddingType(embedding_type)

    if embedding_type == EmbeddingType.POOLED:
        mask = "pooled_{}.pt"
    elif embedding_type == EmbeddingType.MEAN:
        mask = "mean_{}.pt"
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

    texts = []
    embeddings = []
    path = Path(path)
    for file_path in tqdm(path.glob("**/*.txt")):
        name = file_path.stem.split("_")[1]
        embedding_name = mask.format(name)
        embedding_path = file_path.parent / embedding_name
        if not embedding_path.exists():
            raise ValueError(f"Embedding file {embedding_path} does not exist")
        embs = torch.load(embedding_path)
        if truncate_dim is not None:
            embs = embs[:, :truncate_dim]
        embeddings.append(embs)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                texts.append(item["text"])
        if limit is not None and len(texts) >= limit:
            break

    embeddings_tensor = torch.cat(embeddings, dim=0)
    if limit is not None:
        texts = texts[:limit]
        embeddings_tensor = embeddings_tensor[:limit]

    if len(texts) != embeddings_tensor.size(0):
        raise ValueError(
            f"Number of texts ({len(texts)}) does not match number of embeddings ({embeddings_tensor.size(0)})"
        )

    _, D = embeddings_tensor.shape
    features = Features(
        {
            "sentence": Value("string"),
            "label": Sequence(Value("float32"), length=D),  # fixed-size vector
        }
    )

    data = {
        "sentence": texts,
        "label": embeddings_tensor.numpy().astype("float32"),
    }

    ds = Dataset.from_dict(data, features=features)
    return ds


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
    ds = create_dataset_from_folder(
        args.input, limit=args.limit, embedding_type=args.embedding_type, truncate_dim=args.truncate_dim
    )
    ds.save_to_disk(args.output)
    print(f"Saved dataset with {len(ds)} samples to {args.output}")  # noqa: T201
