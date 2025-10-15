import argparse
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar, cast

import pyarrow.parquet as pq
import torch
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from skeletoken import TokenizerModel

import wandb
from pystatic.embedding import TrainableStaticEmbedding
from pystatic.losses import CosineLoss

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)
random.seed(12)

logger = logging.getLogger(__name__)


T = TypeVar("T", Dataset, IterableDataset)


def _post_process_dataset(dataset: T) -> T:
    """Post-process the dataset."""
    dataset = dataset.rename_column("text", "sentence")
    dataset = dataset.rename_column("embedding", "label")
    column_names = dataset.column_names
    assert column_names is not None
    columns_to_drop = [c for c in column_names if c not in ("sentence", "label")]
    dataset = dataset.remove_columns(columns_to_drop)
    return dataset


def datasets(
    paths: Sequence[Path], in_memory: bool = True, limit_shards: int | None = None
) -> tuple[IterableDataset | DatasetDict, int]:
    """Load the datasets."""
    length = 0
    if in_memory:
        datasets = {}
        for path in paths:
            dataset = cast(Dataset, load_dataset(path=str(path), split="train"))
            length += len(dataset)
            dataset = _post_process_dataset(dataset)
            datasets[path.stem] = dataset
        return DatasetDict(datasets), length

    all_shards = []
    for path in paths:
        shards_in_path = [str(x) for x in Path(path).glob("**/*.parquet")]
        if limit_shards is not None:
            shards_in_path = shards_in_path[:limit_shards]
        all_shards.extend(shards_in_path)
    for shard in all_shards:
        length += pq.read_metadata(shard).num_rows
    random.shuffle(all_shards)
    dataset = cast(IterableDataset, load_dataset("parquet", data_files=all_shards, split="train", streaming=True))
    dataset = _post_process_dataset(dataset)

    return dataset, length


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distillation experiment.")
    parser.add_argument("name", help="Name of the experiment.")
    parser.add_argument(
        "--train-dataset",
        type=str,
        required=True,
        help="Path to the training datasets",
        nargs="+",
    )
    parser.add_argument("--model-dim", type=int, default=1024, help="Dimensionality of the model.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training.")
    parser.add_argument("--total-steps", type=int, help="Total number of training steps, -1 for automatic.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate for training.")
    parser.add_argument(
        "--tokenizer-path", type=str, default="mixedbread-ai/mxbai-embed-large-v1", help="Path to the tokenizer."
    )
    parser.add_argument("--limit-shards", type=int, help="Limit the number of shards.")
    parser.add_argument("--in-memory", action="store_true", help="Load the dataset in memory.")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = _parse_args()
    model_dim = parsed_args.model_dim

    tokenizer = TokenizerModel.from_pretrained(parsed_args.tokenizer_path).to_transformers()
    s = TrainableStaticEmbedding(tokenizer=tokenizer, embedding_dim=model_dim)
    model = SentenceTransformer(modules=[s])

    stsb_eval_dataset = cast(Dataset, load_dataset("sentence-transformers/stsb", split="validation"))
    stsb_test_dataset = cast(Dataset, load_dataset("sentence-transformers/stsb", split="test"))

    dev_evaluator_stsb = EmbeddingSimilarityEvaluator(
        sentences1=stsb_eval_dataset["sentence1"],
        sentences2=stsb_eval_dataset["sentence2"],
        scores=stsb_eval_dataset["score"],
        main_similarity=SimilarityFunction.COSINE,
        name="sts-dev",
    )

    # Workaround for local development
    n_workers = 7
    prefetch_factor: int | None = 2
    if torch.mps.is_available():
        n_workers = 0
        prefetch_factor = None

    name = parsed_args.name
    logger.info(f"Starting experiment: {name}")

    loss = CosineLoss(model=model)

    train_dataset, n_samples = datasets(
        [Path(path) for path in parsed_args.train_dataset], in_memory=parsed_args.in_memory
    )

    # Log every 51200 samples, this is roughly every 25 steps with batch size 2048
    logging_step = 51200 // parsed_args.batch_size
    # Evaluate and save every 4 times the logging step
    eval_step = save_step = 4 * logging_step

    # Number of steps per epoch
    n_steps = n_samples // parsed_args.batch_size
    total_steps = parsed_args.total_steps or (n_steps * parsed_args.epochs)

    wandb.init(project="distillation", name=name)
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"models/{name}",
        # Optional training parameters:
        max_steps=total_steps,
        per_device_train_batch_size=parsed_args.batch_size,
        per_device_eval_batch_size=parsed_args.batch_size,
        learning_rate=parsed_args.learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=0.0,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=eval_step,
        save_strategy="steps",
        save_steps=save_step,
        save_total_limit=2,
        logging_steps=logging_step,
        logging_first_step=True,
        run_name=name,
        use_cpu=False,
        report_to=["wandb"],
        weight_decay=0.0,
        load_best_model_at_end=False,
        greater_is_better=True,
        metric_for_best_model="sts-dev_spearman_cosine",
        dataloader_num_workers=n_workers,
        dataloader_prefetch_factor=prefetch_factor,
        dataloader_pin_memory=True,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=dev_evaluator_stsb,
    )
    trainer.train()

    dev_evaluator_stsb(model)
    model.save_pretrained(f"models/{name}/final")
