import argparse
import logging
import random
from typing import cast

import torch
from datasets import Dataset, IterableDataset, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SimilarityFunction,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, NanoBEIREvaluator, SentenceEvaluator
from sentence_transformers.losses import MatryoshkaLoss
from sentence_transformers.models import Module, Normalize, Router
from skeletoken import TokenizerModel
from torch import nn

import wandb
from pystatic.data import get_datasets
from pystatic.embedding import LayerNorm, TrainableStaticEmbedding, TrainableStaticEmbeddingWithW
from pystatic.losses import CosineLoss

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.ERROR)
logger = logging.getLogger(__name__)
random.seed(12)


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
    parser.add_argument("--text-field", type=str, default="sentence", help="Field in the dataset to use as text.")
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
    parser.add_argument("--initialize-from-model", type=str, help="Path to a model to initialize from.")
    parser.add_argument("--with-norm", action="store_true", help="Add layer normalization after the embeddings.")
    parser.add_argument("--with-weights", action="store_true", help="Use per-token weights in the embedding layer.")
    return parser.parse_args()


def initialize_model(
    tokenizer_path: str,
    model_to_initialize_from: str | None,
    model_dim: int,
    with_norm: bool,
    with_weights: bool,
) -> SentenceTransformer:
    """Initialize the model."""
    tokenizer = TokenizerModel.from_pretrained(tokenizer_path).to_transformers()

    modules: list[Module] = []

    cls: type[TrainableStaticEmbedding] | type[TrainableStaticEmbeddingWithW]
    if with_weights:
        cls = TrainableStaticEmbeddingWithW
    else:
        cls = TrainableStaticEmbedding

    if model_to_initialize_from:
        logger.info(f"Initializing from model {model_to_initialize_from}")
        model = SentenceTransformer(model_to_initialize_from)
        v, _ = zip(*sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]))
        vocab: list[str] = list(v)
        weights = model.encode(vocab, batch_size=2048, convert_to_numpy=False, convert_to_tensor=True)
        model_dim = weights.shape[1]
        s = cls(
            tokenizer=tokenizer,
            embedding_weights=weights.cpu().numpy(),
        )
        modules.append(s)
    else:
        s = cls(tokenizer=tokenizer, embedding_dim=model_dim)
        modules.append(s)
    if with_norm:
        modules.append(LayerNorm(dim=model_dim))
    modules.append(Normalize())

    model = SentenceTransformer(modules=modules)

    return model


def load_data(
    train_datasets: list[str], in_memory: bool, limit_shards: int | None
) -> tuple[Dataset | IterableDataset, int]:
    """Get the training data."""
    datasets, n_samples = get_datasets(
        paths=train_datasets,
        in_memory=in_memory,
        limit_shards=limit_shards,
    )
    logger.info(f"Number of training samples: {n_samples}")
    return datasets, n_samples


def run_experiment(
    model: SentenceTransformer,
    name: str,
    dataset: Dataset | IterableDataset,
    n_samples: int,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    l2_norm: float | None,
    use_matryoshka: bool,
) -> None:
    """Run the distillation experiment."""
    # Workaround for local development
    n_workers = 7
    prefetch_factor: int | None = 2
    if torch.mps.is_available():
        n_workers = 0
        prefetch_factor = None

    logger.info(f"Starting experiment: {name}")

    loss: nn.Module
    loss = CosineLoss(model=model, l2_norm=l2_norm)
    if use_matryoshka:
        emb_dim = model.get_sentence_embedding_dimension()
        assert emb_dim is not None
        dims = [emb_dim]
        while dims[-1] > 32:
            dims.append(dims[-1] // 2)
        loss = MatryoshkaLoss(model, loss, matryoshka_dims=dims)

    evaluators: list[SentenceEvaluator] = []
    stsb_eval_dataset = cast(Dataset, load_dataset("sentence-transformers/stsb", split="validation"))
    dev_evaluator_stsb = EmbeddingSimilarityEvaluator(
        sentences1=stsb_eval_dataset["sentence1"],
        sentences2=stsb_eval_dataset["sentence2"],
        scores=stsb_eval_dataset["score"],
        main_similarity=SimilarityFunction.COSINE,
        name="sts-dev",
    )
    evaluators.append(dev_evaluator_stsb)
    nanobeir_evaluator = NanoBEIREvaluator()
    evaluators.append(nanobeir_evaluator)

    # Log every 51200 samples, this is roughly every 25 steps with batch size 2048
    logging_step = 51200 // batch_size
    # Evaluate and save every 40 times the logging step
    eval_step = save_step = 40 * logging_step

    # Number of steps per epoch
    n_steps = n_samples // batch_size
    total_steps = n_steps * epochs

    wandb.init(project="distillation", name=name)
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"models/{name}",
        # Optional training parameters:
        max_steps=total_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=0.0,
        eval_strategy="steps",
        eval_steps=eval_step,
        save_strategy="steps",
        save_steps=save_step,
        save_total_limit=1,
        logging_steps=logging_step,
        logging_first_step=True,
        run_name=name,
        report_to=["wandb"],
        weight_decay=0.0,
        load_best_model_at_end=False,
        dataloader_num_workers=n_workers,
        dataloader_prefetch_factor=prefetch_factor,
        dataloader_pin_memory=True,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=loss,
        evaluator=evaluators,
    )
    trainer.train()

    model.save_pretrained(f"models/{name}/final")

    model_name = "mixedbread-ai/mxbai-embed-large-v1"
    big_model = SentenceTransformer(model_name)

    router = Router.for_query_document(query_modules=[model], document_modules=[big_model])  # type: ignore
    s = SentenceTransformer(modules=[router])

    # Evaluate the model
    results = nanobeir_evaluator(s, output_path=f"results/nanobeir/router-{name}")
    print(results)  # noqa: T201
    assert wandb.run is not None
    wandb.run.summary["nanobeir_results"] = results


if __name__ == "__main__":
    parsed_args = _parse_args()
    model_dim = parsed_args.model_dim

    model = initialize_model(
        tokenizer_path=parsed_args.tokenizer_path,
        model_to_initialize_from=parsed_args.initialize_from_model,
        model_dim=model_dim,
        with_norm=parsed_args.with_norm,
        with_weights=parsed_args.with_weights,
    )
    dataset, n_samples = load_data(
        train_datasets=parsed_args.train_dataset,
        in_memory=parsed_args.in_memory,
        limit_shards=parsed_args.limit_shards,
    )
    run_experiment(
        model,
        parsed_args.name,
        dataset,
        n_samples,
        parsed_args.batch_size,
        parsed_args.learning_rate,
        parsed_args.epochs,
        l2_norm=None,
        use_matryoshka=False,
    )
