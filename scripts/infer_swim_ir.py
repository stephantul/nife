"""Script with some hardcoded stuff for ease of use."""

import logging
import shutil
from collections.abc import Iterable, Iterator
from typing import cast

from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer

from pystatic.data import build_parquet_shards_from_folder
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

    name = "nthakur/swim-ir-monolingual"
    folder_name = f"output/swim-ir_{model_name.replace('/', '__')}{suffix}"
    converted_folder_name = f"converted/swim-ir_{model_name.replace('/', '__')}{suffix}"

    dataset = cast(Dataset, load_dataset(name, "en", split="train"))

    new_records: list[dict[str, str]] = []
    for record in cast(Iterable[dict[str, str]], dataset):
        text = record["query"]
        new_records.append({"id": record["_id"], "text": text})

    dataset_iterator = cast(Iterator[dict[str, str]], iter(new_records))
    infer(
        model,
        dataset_iterator,
        batch_size=512,
        name=folder_name,
        save_every=256,
        prompt=prompt,
        limit_batches=args.limit_batches,
    )

    logger.info("Converting dataset to shards...")
    build_parquet_shards_from_folder(folder_name, converted_folder_name)
    logger.info(f"Converted dataset saved to {converted_folder_name}")
    shutil.rmtree(folder_name)
    logger.info(f"Removed temporary folder {folder_name}")
