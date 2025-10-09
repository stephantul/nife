"""Script with some hardcoded stuff for ease of use."""

import logging
from typing import Iterator, cast

from datasets import Dataset, concatenate_datasets, load_dataset
from sentence_transformers import SentenceTransformer

from pystatic.distillation.infer import infer

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model_name = "mixedbread-ai/mxbai-embed-large-v1"
    model = SentenceTransformer(model_name)

    prompt = model.prompts.get("query", None)

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
    infer(model, dataset_iterator, batch_size=8, name="output/pubmed", save_every=16384, prompt=prompt)
