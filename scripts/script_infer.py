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

    paths = ["sample/10BT/000_00000.parquet", "sample/10BT/000_00001.parquet", "sample/10BT/000_00002.parquet"]
    for path in paths:
        data = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", streaming=True, split="train", data_files=path)
        texts = (item["text"] for item in data)
        infer(model, texts, batch_size=128, name=f"output/{model_name.replace('/', '_')}", save_every=16)
