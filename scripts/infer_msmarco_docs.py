"""Script with some hardcoded stuff for ease of use."""

import logging
from typing import Iterator, cast

from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer

from pystatic.distillation.infer import infer

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model_name = "mixedbread-ai/mxbai-embed-large-v1"
    model = SentenceTransformer(model_name)

    name = "mteb/msmarco"

    dataset = cast(Dataset, load_dataset(name, "corpus", split="corpus"))
    dataset = dataset.rename_column("_id", "id")
    dataset_iterator = cast(Iterator[dict[str, str]], iter(dataset))
    infer(model, dataset_iterator, batch_size=8, name="output/msmarco", save_every=16384)
