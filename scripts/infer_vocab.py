"""Script with some hardcoded stuff for ease of use."""

import logging
from typing import cast

from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer

from pystatic.distillation.infer import infer

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model_name = "mixedbread-ai/mxbai-embed-large-v1"
    model = SentenceTransformer(model_name)

    name = "english-words-definitions"
    dataset = cast(Dataset, load_dataset("MongoDB/english-words-definitions", split="train"))
    dataset = dataset.map(lambda x: {"text": " ".join(x["definitions"])})
    dataset = dataset.add_column("id", [str(i) for i in range(len(dataset))], new_fingerprint="added_ids")
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    infer(model, iter(dataset), batch_size=2, name=f"output/{name}", save_every=256, prompt="This is a definition: ")
