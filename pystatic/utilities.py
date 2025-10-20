import logging
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

from sentence_transformers import SentenceTransformer

from pystatic.data import build_parquet_shards_from_folder
from pystatic.distillation.infer import infer

logger = logging.getLogger(__name__)


def run_inference(
    model: SentenceTransformer,
    output_folder: str | Path,
    records: Iterator[dict[str, str]],
    limit_batches: int | None = None,
    batch_size: int = 512,
    save_every: int = 256,
    max_length: int = 512,
) -> None:
    """Run inference and save the results to parquet shards."""
    with TemporaryDirectory() as dir_name:
        infer(
            model,
            records,
            batch_size=batch_size,
            output_dir=dir_name,
            save_every=save_every,
            limit_batches=limit_batches,
            max_length=max_length,
        )

        logger.info("Converting dataset to shards...")
        build_parquet_shards_from_folder(dir_name, output_folder)
        logger.info(f"Converted dataset saved to {output_folder}")
