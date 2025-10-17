"""Script with some hardcoded stuff for ease of use."""

import logging
import shutil
from typing import Iterator, cast

from datasets import load_dataset
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

    name = "google-research-datasets/paws"
    folder_name = f"output/paws_{model_name.replace('/', '__')}{suffix}"
    converted_folder_name = f"converted/paws_{model_name.replace('/', '__')}{suffix}"

    dataset = load_dataset(name, "labeled_final", split="train")

    new_records = []
    for i, record in enumerate(dataset):
        text = record["sentence1"]
        new_records.append({"id": i * 2, "text": text})
        text = record["sentence2"]
        new_records.append({"id": (i * 2) + 1, "text": text})

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
