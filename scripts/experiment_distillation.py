import argparse
import logging
import random
from typing import cast

import torch
import wandb
from datasets import Dataset, IterableDataset, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SimilarityFunction,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, NanoBEIREvaluator, SentenceEvaluator
from sentence_transformers.losses import MatryoshkaLoss
from sentence_transformers.models import Module, Normalize, Router, StaticEmbedding
from skeletoken import TokenizerModel
from torch import nn

from pynife.cards.model_card import get_model_card_template_path
from pynife.data import get_datasets, get_model_name_from_datasets
from pynife.embedding import TrainableStaticEmbedding
from pynife.initialization.model import initialize_from_model
from pynife.losses import LossFunction, select_loss

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
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate for training.")
    parser.add_argument(
        "--tokenizer-path", type=str, default="mixedbread-ai/mxbai-embed-large-v1", help="Path to the tokenizer."
    )
    parser.add_argument("--limit-shards", type=int, help="Limit the number of shards.")
    parser.add_argument("--in-memory", action="store_true", help="Load the dataset in memory.")
    parser.add_argument("--trained-model", type=str, help="Path to a trained model to continue training from.")
    parser.add_argument("--initialize-from-model", type=str, help="Path to a model to initialize from.")
    parser.add_argument("--loss-function", type=str, default="cosine", help="Loss function to use.")
    parser.add_argument("--warmup", type=float, default=0.1, help="Warmup proportion for learning rate scheduler.")
    parser.add_argument(
        "--scheduler-type",
        type=str,
        default="cosine_warmup_with_min_lr",
        help="Warmup type for learning rate scheduler.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for optimizer.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    return parser.parse_args()


def _create_model(
    tokenizer_path: str,
    model_to_initialize_from: str | None,
    model_dim: int | None,
) -> SentenceTransformer:
    """Create a model."""
    tokenizer_model = TokenizerModel.from_pretrained(tokenizer_path)
    tokenizer = tokenizer_model.to_transformers()

    modules: list[Module] = []

    if model_dim is None and model_to_initialize_from is None:
        raise ValueError("Either model_dim or model_to_initialize_from must be provided.")
    if model_dim is not None and model_to_initialize_from is not None:
        logger.warning("Both model_dim and model_to_initialize_from are provided. Ignoring model_dim.")

    if model_to_initialize_from:
        logger.info(f"Initializing from model {model_to_initialize_from}")
        model = SentenceTransformer(model_to_initialize_from)
        weights = initialize_from_model(model, tokenizer)
        s = TrainableStaticEmbedding(
            tokenizer=tokenizer,
            embedding_weights=weights.cpu().numpy(),
        )
        modules.append(s)
    else:
        s = TrainableStaticEmbedding(tokenizer=tokenizer, embedding_dim=model_dim)
        modules.append(s)
    modules.append(Normalize())

    model = SentenceTransformer(modules=modules)

    return model


def run_experiment(
    model: SentenceTransformer,
    name: str,
    dataset: Dataset | IterableDataset,
    n_samples: int,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    use_matryoshka: bool,
    loss_function_name: str | LossFunction,
    warmup_ratio: float,
    lr_scheduler_type: str,
    weight_decay: float,
    max_grad_norm: float,
    base_model_name: str | None,
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
    loss_function = select_loss(loss_function_name)
    loss = loss_function(model)
    if use_matryoshka:
        emb_dim = model.get_sentence_embedding_dimension()
        assert emb_dim is not None
        dims = [16]
        new_dim = dims[-1]
        while new_dim * 2 < emb_dim:
            new_dim *= 2
            dims.append(new_dim)
        dims.append(emb_dim)
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
    # Evaluate and save every 10 times the logging step
    eval_step = save_step = 10 * logging_step

    # Number of steps per epoch
    n_steps = n_samples // batch_size
    total_steps = n_steps * epochs

    num_cycles = 0.5
    if lr_scheduler_type == "cosine_warmup_with_min_lr":
        lr_scheduler_kwargs = {"min_lr_rate": 0.10, "num_cycles": num_cycles}
    else:
        lr_scheduler_kwargs = None

    wandb.init(project="distillation", name=name)
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"models/{name}",
        # Optional training parameters:
        max_steps=total_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        eval_strategy="steps",
        eval_steps=eval_step,
        save_strategy="steps",
        save_steps=save_step,
        save_total_limit=1,
        logging_steps=logging_step,
        logging_first_step=True,
        run_name=name,
        report_to=["wandb"],
        weight_decay=weight_decay,
        load_best_model_at_end=False,
        dataloader_num_workers=n_workers,
        dataloader_prefetch_factor=prefetch_factor,
        dataloader_pin_memory=True,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        max_grad_norm=max_grad_norm,
        metric_for_best_model="eval_NanoBEIR_mean_cosine_ndcg@10",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=loss,
        evaluator=evaluators,
    )
    trainer.train()

    static_module: TrainableStaticEmbedding = model[0]  # type: ignore
    new_static_model = StaticEmbedding(
        tokenizer=static_module.tokenizer,
        embedding_weights=static_module.embedding.weight.data.cpu().numpy(),
    )
    old_model_card_data = model.model_card_data
    new_model = SentenceTransformer(modules=[new_static_model, Normalize()])
    # This tag needs to be added, otherwise the model card is not updated.
    new_model.model_card_data_class.template_path = get_model_card_template_path()
    new_model.model_card_data = old_model_card_data
    new_model.model_card_data.tags.append("generated_from_trainer")  # type: ignore
    new_model.model_card_data.base_model = base_model_name
    new_model.model_card_data.model_name = f"NIFE model based on {base_model_name}"  # type: ignore
    new_model.save_pretrained(f"models/{name}/final")


if __name__ == "__main__":
    random.seed(12)
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    parsed_args = _parse_args()
    model_dim = parsed_args.model_dim

    loss_function = select_loss(parsed_args.loss_function)

    model_name = get_model_name_from_datasets(parsed_args.train_dataset)
    if model_name is None:
        logger.info("Could not determine model name from datasets.")
        model_name = parsed_args.initialize_from_model

    if parsed_args.trained_model:
        if parsed_args.initialize_from_model:
            logger.warning(
                "Both --trained-model and --initialize-from-model are provided. Ignoring --initialize-from-model."
            )
        if parsed_args.model_dim:
            logger.warning(
                "--model-dim is provided while continuing training from a trained model. Ignoring --model-dim."
            )
        logger.info(f"Loading trained model from {parsed_args.trained_model}")
        model = SentenceTransformer(parsed_args.trained_model)
    else:
        model = _create_model(
            tokenizer_path=parsed_args.tokenizer_path,
            model_to_initialize_from=model_name,
            model_dim=model_dim,
        )
    dataset, n_samples = get_datasets(
        paths=parsed_args.train_dataset,
        in_memory=parsed_args.in_memory,
        limit_shards=parsed_args.limit_shards,
        columns_to_keep={"sentence", "label"},
    )
    run_experiment(
        model,
        parsed_args.name,
        dataset,
        n_samples,
        parsed_args.batch_size,
        parsed_args.learning_rate,
        parsed_args.epochs,
        use_matryoshka=True,
        loss_function_name=parsed_args.loss_function,
        warmup_ratio=parsed_args.warmup,
        lr_scheduler_type=parsed_args.scheduler_type,
        weight_decay=parsed_args.weight_decay,
        max_grad_norm=parsed_args.max_grad_norm,
        base_model_name=model_name,
    )

    if model_name is not None:
        big_model = SentenceTransformer(model_name)

        router = Router.for_query_document(query_modules=[model], document_modules=[big_model])  # type: ignore
        s = SentenceTransformer(modules=[router])

        nanobeir_evaluator = NanoBEIREvaluator(batch_size=8)
        # Evaluate the model
        results = nanobeir_evaluator(s, output_path=f"results/nanobeir/router-{parsed_args.name}")
        print(results)  # noqa: T201
        assert wandb.run is not None
        wandb.run.summary["nanobeir_results"] = results
