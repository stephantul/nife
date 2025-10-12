import argparse
import logging
import random
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import torch
from datasets import Dataset, IterableDataset, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.losses import MSELoss
from sentence_transformers.models import StaticEmbedding
from sentence_transformers.similarity_functions import SimilarityFunction
from skeletoken import TokenizerModel
from torch import Tensor, nn

import wandb

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)
random.seed(12)

logger = logging.getLogger(__name__)


class TrainableStaticEmbedding(StaticEmbedding):
    def tokenize(self, texts: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        """Tokenize the texts."""
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=False)
        encodings_ids = [torch.Tensor(encoding.ids[:512]).long() for encoding in encodings]

        input_ids = torch.nn.utils.rnn.pad_sequence(encodings_ids, batch_first=True, padding_value=0)
        return {"input_ids": input_ids}

    def forward(self, features: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forward pass."""
        features["sentence_embedding"] = self.embedding(features["input_ids"])
        return features


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


def datasets(paths: Sequence[Path]) -> IterableDataset:
    """Load the datasets."""
    d = []
    features = None
    for path in paths:
        dataset = load_dataset(str(path), split="train", streaming=True)
        dataset = cast(Dataset, dataset)
        dataset = dataset.rename_column("text", "sentence")
        dataset = dataset.rename_column("embedding", "label")
        d.append(dataset)
        features = dataset.features

    assert features is not None

    def generator_func() -> Iterator[dict[str, Tensor]]:
        for dataset in d:
            for example in dataset:
                yield example

    return IterableDataset.from_generator(generator_func, features=features)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distillation experiment.")
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
    parser.add_argument("--tokenizer-path", type=str, default="bert-base-uncased", help="Path to the tokenizer.")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = _parse_args()
    model_dim = parsed_args.model_dim
    tokenizer = TokenizerModel.from_pretrained(parsed_args.tokenizer_path).to_transformers()
    s = TrainableStaticEmbedding(tokenizer=tokenizer, embedding_dim=model_dim)
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
    # loss = MatryoshkaLoss(model, base_loss, matryoshka_dims=matryoshka_dims)
    loss = base_loss

    train_dataset = datasets([Path(path) for path in parsed_args.train_dataset])

    logging_step = 51200 // parsed_args.batch_size
    eval_step = save_step = 4 * logging_step

    n_samples = 8840000 + 14900000 + 1000000
    n_steps = n_samples // parsed_args.batch_size

    run_name = f"distillation-{model_dim}-cosine"
    wandb.init(project="distillation", name=run_name)
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=parsed_args.epochs,
        max_steps=n_steps * parsed_args.epochs,
        per_device_train_batch_size=parsed_args.batch_size,
        per_device_eval_batch_size=parsed_args.batch_size,
        learning_rate=0.2,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=eval_step,
        save_strategy="steps",
        save_steps=save_step,
        save_total_limit=2,
        logging_steps=logging_step,
        logging_first_step=True,
        run_name=run_name,
        use_cpu=False,
        report_to=["wandb"],
        weight_decay=0.0,
        load_best_model_at_end=False,
        greater_is_better=True,
        metric_for_best_model="sts-dev_spearman_cosine",
        dataloader_num_workers=8,
        dataloader_prefetch_factor=2,
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
    model.save_pretrained(f"models/{run_name}/final")
