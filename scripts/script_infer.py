"""Script with some hardcoded stuff for ease of use."""

import logging

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from pystatic.distillation.infer import infer

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model_name = "mixedbread-ai/mxbai-embed-large-v1"
    model = SentenceTransformer(model_name)

    paths = [
        "sample/10BT/003_00000.parquet",
        "sample/10BT/004_00000.parquet",
        "sample/10BT/005_00000.parquet",
        "sample/10BT/006_00000.parquet",
        "sample/10BT/007_00000.parquet",
        "sample/10BT/008_00000.parquet",
    ]
    for path in paths:
        data = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", streaming=True, split="train", data_files=path)
        texts = (item["text"] for item in data)
        infer(model, texts, batch_size=512, name=f"output/{path.replace('/', '_')}", save_every=16)
