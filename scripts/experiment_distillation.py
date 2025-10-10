import argparse
import logging
import random
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import cast

import torch
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
from torch import Tensor, nn

import wandb
from datasets import Dataset, DatasetDict, load_dataset

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)
random.seed(12)


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
        return 1 - self.loss_fct(embeddings, labels[:, : embeddings.shape[1]]).mean()


def datasets_from_root_path(root: Path) -> DatasetDict:
    """Load all datasets from a root path."""
    paths = (path for path in root.glob("*") if path.is_dir())
    return datasets(paths)


def datasets(paths: Iterator[Path]) -> DatasetDict:
    """Load the datasets."""
    d = {}
    for path in paths:
        dataset = cast(DatasetDict, load_dataset(str(path)))
        dataset = dataset.rename_column("text", "sentence")
        dataset = dataset.rename_column("embedding", "label")
        d[str(path)] = dataset["train"]

    return DatasetDict(d)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distillation experiment.")
    parser.add_argument(
        "--train-dataset",
        type=str,
        required=True,
        help="Path to the training dataset (in HuggingFace datasets format).",
    )
    parser.add_argument("--model-dim", type=int, default=1024, help="Dimensionality of the model.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate for training.")
    parser.add_argument("--tokenizer-path", type=str, default="bert-base-uncased", help="Path to the tokenizer.")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = _parse_args()
    model_dim = parsed_args.model_dim
    tokenizer = TokenizerModel.from_pretrained(parsed_args.tokenizer_path).to_transformers()
    s = StaticEmbedding(tokenizer=tokenizer, embedding_dim=model_dim)
    # tokenizer = TokenizerModel.from_pretrained("../static-trainer/models/static-retrieval-mrl-en-1024-0.2-60k/final/tokenizer.json").to_transformers()
    # s = StaticEmbedding(tokenizer=tokenizer, embedding_dim=model_dim)
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

    base_loss = CosineLoss(model=model)
    dims = [model_dim]
    while dims[-1] > 32:
        dims.append(dims[-1] // 2)
    matryoshka_dims = sorted(dims)
    matryoshka_dims = [d for d in matryoshka_dims if d <= model_dim]
    loss = MatryoshkaLoss(model, base_loss, matryoshka_dims=matryoshka_dims)

    train_dataset = datasets_from_root_path(Path(parsed_args.train_dataset))
    # train_dataset = cast(Dataset, load_dataset("stsb_multi_mt", "en", split="train"))

    run_name = f"distillation-{model_dim}-matryoshka-cosine-mean-constant-new-data"
    wandb.init(project="distillation", name=run_name)
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=5,
        per_device_train_batch_size=2048,
        per_device_eval_batch_size=2048,
        learning_rate=0.2,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=25,
        logging_first_step=True,
        run_name=run_name,
        use_cpu=False,
        report_to=["wandb"],
        weight_decay=0.0,
        load_best_model_at_end=False,
        greater_is_better=True,
        metric_for_best_model="sts-dev_spearman_cosine",
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
    model.save_pretrained(f"models/{run_name}/final")
