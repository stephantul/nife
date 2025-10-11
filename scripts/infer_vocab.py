"""Script with some hardcoded stuff for ease of use."""

import logging
from typing import Iterator, cast

from datasets import Dataset, load_dataset
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

    name = "english-words-definitions"
    dataset = cast(Dataset, load_dataset("MongoDB/english-words-definitions", split="train"))
    dataset = dataset.map(lambda x: {"text": " ".join(x["definitions"])})
    dataset = dataset.add_column("id", [str(i) for i in range(len(dataset))], new_fingerprint="added_ids")
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    dataset_iterator = cast(Iterator[dict[str, str]], iter(dataset))
    infer(model, dataset_iterator, batch_size=8, name=f"output/{name}{suffix}", save_every=16384)
