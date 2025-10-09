"""Script with some hardcoded stuff for ease of use."""

import logging
from typing import Iterator, cast

from datasets import Dataset, concatenate_datasets, load_dataset
from sentence_transformers import SentenceTransformer

from pystatic.distillation.infer import infer

_QUERY_PATHS = [
    "lifestyle_forum-queries",
    "lifestyle_search-queries",
    "recreation_forum-queries",
    "recreation_search-queries",
    "science_forum-queries",
    "science_search-queries",
    "technology_forum-queries",
    "technology_search-queries",
    "writing_forum-queries",
    "writing_search-queries",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model_name = "mixedbread-ai/mxbai-embed-large-v1"
    model = SentenceTransformer(model_name)

    prompt = model.prompts.get("query", None)

    name = "mteb/lotte"

    big_dataset = []
    for path in _QUERY_PATHS:
        dataset = cast(Dataset, load_dataset(name, path, split="dev"))
        big_dataset.append(dataset)
    final_dataset: Dataset = concatenate_datasets(big_dataset)
    final_dataset = final_dataset.rename_column("_id", "id")
    dataset_iterator = cast(Iterator[dict[str, str]], iter(final_dataset))
    infer(model, dataset_iterator, batch_size=8, name="output/lotte", save_every=10_000, prompt=prompt)
