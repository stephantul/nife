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

    max_length = args.max_length

    name = "mteb/msmarco"
    dataset = cast(Dataset, load_dataset(name, "corpus", split="corpus", streaming=True))
    dataset = dataset.rename_column("_id", "id")
    dataset_iterator = cast(Iterator[dict[str, str]], iter(dataset))
    infer(
        model,
        dataset_iterator,
        batch_size=512,
        name=f"output/msmarco_ml{max_length}{suffix}",
        save_every=256,
        max_length=max_length,
    )
