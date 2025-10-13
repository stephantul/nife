import argparse
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq
import torch
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.losses import MatryoshkaLoss, MSELoss
from sentence_transformers.models import StaticEmbedding
from sentence_transformers.similarity_functions import SimilarityFunction
from skeletoken import TokenizerModel
from tokenizers import Tokenizer
from torch import Tensor, nn
from transformers import PreTrainedTokenizerFast

import wandb

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)
random.seed(12)

logger = logging.getLogger(__name__)


class TrainableStaticEmbedding(StaticEmbedding):
    def __init__(
        self,
        tokenizer: Tokenizer | PreTrainedTokenizerFast,
        embedding_weights: np.ndarray | torch.Tensor | None = None,
        embedding_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Static embedding layer with trainable weights."""
        super().__init__(
            tokenizer=tokenizer, embedding_dim=embedding_dim, embedding_weights=embedding_weights, **kwargs
        )
        # self.w = nn.Embedding.from_pretrained(torch.zeros(self.tokenizer.get_vocab_size(), 1), freeze=False)
        self.normalizer = nn.LayerNorm(self.embedding_dim)

    def tokenize(self, texts: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        """Tokenize the texts."""
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=False)
        encodings_ids = [torch.Tensor(encoding.ids[:512]).long() for encoding in encodings]

        input_ids = torch.nn.utils.rnn.pad_sequence(encodings_ids, batch_first=True, padding_value=0)
        return {"input_ids": input_ids}

    def forward(self, features: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forward pass."""
        x = self.embedding(features["input_ids"])
        x = self.normalizer(x)
        features["sentence_embedding"] = x
        return features

    def collapse(self) -> StaticEmbedding:
        """Collapse to a non-trainable StaticEmbedding."""
        emb_weights = self.embedding.weight.data.clone()
        # emb_weights = emb_weights * torch.sigmoid(self.w.weight)
        emb_weights = self.normalizer(emb_weights)
        return StaticEmbedding(
            tokenizer=self.tokenizer, embedding_weights=emb_weights, embedding_dim=self.embedding_dim
        )


class CosineLoss(MSELoss):
    def __init__(self, model: SentenceTransformer) -> None:
        """Mean Squared Error loss on cosine similarity of sentence embeddings."""
        super().__init__(model=model)
        self.loss_fct = nn.CosineSimilarity()  # type: ignore

    def forward(self, sentence_features: Sequence[dict[str, Tensor]], labels: Tensor) -> Tensor:  # type: ignore
        """Forward pass."""
        # Concatenate multiple inputs on the batch dimension
        if len(sentence_features) > 1:
            embeddings = torch.cat([self.model(inputs)["sentence_embedding"] for inputs in sentence_features], dim=0)
            # Repeat the labels for each input
            return 1 - self.loss_fct(embeddings, labels.repeat(len(sentence_features), 1)).mean()

        embeddings = self.model(sentence_features[0])["sentence_embedding"]
        labels = labels[:, : embeddings.shape[1]]
        return 1 - self.loss_fct(embeddings, labels[:, : embeddings.shape[1]]).mean()


def datasets(paths: Sequence[Path], in_memory: bool = True) -> tuple[IterableDataset | DatasetDict, int]:
    """Load the datasets."""
    length = 0
    if in_memory:
        datasets = {}
        for path in paths:
            dataset = load_dataset(path=str(path), split="train")
            length += len(dataset)
            dataset = dataset.rename_column("text", "sentence")
            dataset = dataset.rename_column("embedding", "label")
            column_names = dataset.column_names
            assert column_names is not None
            columns_to_drop = [c for c in column_names if c not in ("sentence", "label")]
            dataset = dataset.remove_columns(columns_to_drop)
            datasets[path.stem] = dataset
        return DatasetDict(datasets), length

    all_shards = []
    for path in paths:
        all_shards.extend([str(x) for x in Path(path).glob("**/*.parquet")])
    for shard in all_shards:
        length += pq.read_metadata(shard).num_rows
    random.shuffle(all_shards)
    dataset = load_dataset("parquet", data_files=all_shards, split="train", streaming=True)
    dataset = dataset.rename_column("text", "sentence")
    dataset = dataset.rename_column("embedding", "label")
    column_names = dataset.column_names
    assert column_names is not None
    columns_to_drop = [c for c in column_names if c not in ("sentence", "label")]
    dataset = dataset.remove_columns(columns_to_drop)

    return cast(IterableDataset, dataset), length


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
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate for training.")
    parser.add_argument(
        "--tokenizer-path", type=str, default="mixedbread-ai/mxbai-embed-large-v1", help="Path to the tokenizer."
    )
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
    n_workers = 8
    prefetch_factor: int | None = 2
    if torch.mps.is_available():
        n_workers = 0
        prefetch_factor = None

    name = parsed_args.name
    logger.info(f"Starting experiment: {name}")

    base_loss = CosineLoss(model=model)
    dims = [1024]
    while dims[-1] > 32:
        dims.append(dims[-1] // 2)
    matryoshka_dims = sorted(dims)
    matryoshka_dims = [d for d in matryoshka_dims if d <= model_dim]
    loss = MatryoshkaLoss(model, base_loss, matryoshka_dims=matryoshka_dims)

    train_dataset, n_samples = datasets(
        [Path(path) for path in parsed_args.train_dataset], in_memory=parsed_args.in_memory
    )

    # Log every 51200 samples, this is roughly every 25 steps with batch size 2048
    logging_step = 51200 // parsed_args.batch_size
    # Evaluate and save every 4 times the logging step
    eval_step = save_step = 4 * logging_step

    # Number of steps per epoch
    n_steps = n_samples // parsed_args.batch_size

    wandb.init(project="distillation", name=name)
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"models/{name}",
        # Optional training parameters:
        num_train_epochs=parsed_args.epochs,
        max_steps=n_steps * parsed_args.epochs,
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
