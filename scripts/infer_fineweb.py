"""Script with some hardcoded stuff for ease of use."""

import logging
from collections.abc import Iterator
from typing import cast

from sentence_transformers import SentenceTransformer

from datasets import load_dataset
from pystatic.distillation.helpers import get_prompt_from_model, parse_inference_args
from pystatic.distillation.infer import infer

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    args = parse_inference_args()

    model_name = args.model_name
    model = SentenceTransformer(model_name)
    prompt = get_prompt_from_model(model, args.prompt_name)

    suffix = f"-{args.prompt_name}" if args.prompt_name is not None else ""

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
        infer(model, iter(data), batch_size=512, name=f"output/fineweb/{path.replace('/', '_')}", save_every=256)
