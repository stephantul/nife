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

    name = "triviaqa"
    dataset = cast(Dataset, load_dataset("mandarjoshi/trivia_qa", "unfiltered.nocontext", split="train"))
    dataset = dataset.rename_columns({"question_id": "id", "question": "text"})
    columns_to_remove = [col for col in dataset.column_names if col not in ("id", "text")]
    dataset = dataset.remove_columns(columns_to_remove)
    dataset_iterator = cast(Iterator[dict[str, str]], iter(dataset))
    infer(model, dataset_iterator, batch_size=8, name="output/triviaqa", save_every=16384)
