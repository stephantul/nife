"""Script with some hardcoded stuff for ease of use."""

import logging
import shutil
from typing import Iterator, cast

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

    max_length = args.max_length

    name = "sentence-transformers/msmarco-bm25"
    dataset = cast(Dataset, load_dataset(name, "triplet", split="train", streaming=True))
    # dataset = dataset.add_column("id", list(map(str, range(len(dataset)))), new_fingerprint="id")
    dataset = dataset.remove_columns("negative")
    dataset = dataset.rename_column("positive", "text")
    dataset_iterator = cast(Iterator[dict[str, str]], iter(dataset))

    model_name_str = model_name.replace("/", "__")

    folder_name = f"output/msmarco_{max_length}_{model_name_str}_{suffix}"
    converted_folder_name = f"converted/msmarco_{max_length}_{model_name_str}_{suffix}"

    infer(
        model,
        dataset_iterator,
        batch_size=512,
        name=folder_name,
        save_every=256,
        max_length=max_length,
        limit_batches=args.limit_batches,
    )

    logger.info("Converting dataset to shards...")
    build_parquet_shards_from_folder(folder_name, converted_folder_name)
    logger.info(f"Converted dataset saved to {converted_folder_name}")
    shutil.rmtree(folder_name)
    logger.info(f"Removed temporary folder {folder_name}")
