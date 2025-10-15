import argparse
import logging
import random
from pathlib import Path
from typing import cast

import torch
from datasets import Dataset, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from skeletoken import TokenizerModel

import wandb
from pystatic.data import get_datasets
from pystatic.embedding import TrainableStaticEmbedding
from pystatic.losses import CosineLoss

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)
random.seed(12)

logger = logging.getLogger(__name__)


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

    train_dataset, n_samples = get_datasets(
        [Path(path) for path in parsed_args.train_dataset],
        in_memory=parsed_args.in_memory,
        limit_shards=parsed_args.limit_shards,
        columns_to_keep={"sentence", "label"},
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
        eval_strategy="steps",
        eval_steps=eval_step,
        save_strategy="steps",
        save_steps=save_step,
        save_total_limit=2,
        logging_steps=logging_step,
        logging_first_step=True,
        run_name=name,
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
