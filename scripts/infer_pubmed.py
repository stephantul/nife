"""Script with some hardcoded stuff for ease of use."""

import logging
from typing import Iterator, cast

from datasets import Dataset, concatenate_datasets, load_dataset
from sentence_transformers import SentenceTransformer

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

    name = "qiaojin/PubMedQA"

    subsets = ["pqa_artificial", "pqa_unlabeled"]
    datasets = []
    for subset in subsets:
        dataset = cast(Dataset, load_dataset(name, subset, split="train"))
        dataset = dataset.rename_columns({"pubid": "id", "question": "text"})
        columns_to_remove = [col for col in dataset.column_names if col not in ("id", "text")]
        dataset = dataset.remove_columns(columns_to_remove)
        datasets.append(dataset)

    final_dataset = concatenate_datasets(datasets)
    dataset_iterator = cast(Iterator[dict[str, str]], iter(final_dataset))
    infer(model, dataset_iterator, batch_size=8, name=f"output/pubmed{suffix}", save_every=16384, prompt=prompt)
