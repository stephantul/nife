"""Script with some hardcoded stuff for ease of use."""

import logging
from collections.abc import Iterator
from typing import cast

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from pystatic.distillation.infer import infer

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model_name = "mixedbread-ai/mxbai-embed-large-v1"
    model = SentenceTransformer(model_name)

    paths = [
        "sample/10BT/000_00000.parquet",
        "sample/10BT/001_00000.parquet",
        "sample/10BT/002_00000.parquet",
        "sample/10BT/003_00000.parquet",
        "sample/10BT/004_00000.parquet",
        "sample/10BT/005_00000.parquet",
        "sample/10BT/006_00000.parquet",
        "sample/10BT/007_00000.parquet",
        "sample/10BT/008_00000.parquet",
        "sample/10BT/009_00000.parquet",
        "sample/10BT/010_00000.parquet",
        "sample/10BT/011_00000.parquet",
        "sample/10BT/012_00000.parquet",
        "sample/10BT/013_00000.parquet",
        "sample/10BT/014_00000.parquet",
    ]
    for path in paths:
        data = cast(
            Iterator[dict[str, str]],
            load_dataset("HuggingFaceFW/fineweb", "sample-10BT", streaming=True, split="train", data_files=path),
        )
        infer(model, iter(data), batch_size=128, name=f"output/{path.replace('/', '_')}", save_every=768)
