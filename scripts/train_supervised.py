import logging
import random

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import NanoBEIREvaluator
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.models import Router, StaticEmbedding
from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers
from skeletoken import TokenizerModel

import wandb
from pystatic.supervised.data import load_retrieval_train_eval_datasets

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)
random.seed(12)


def _freeze_model_parameters(model: SentenceTransformer) -> None:
    for module in model.modules():
        for param in module.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    model_name = "mixedbread-ai/mxbai-embed-large-v1"
    large_model = SentenceTransformer(model_name)
    _freeze_model_parameters(large_model)

    model_dim = large_model.get_sentence_embedding_dimension()
    assert isinstance(model_dim, int)

    tokenizer = TokenizerModel.from_pretrained("bert-base-uncased").to_transformers()
    static = StaticEmbedding(tokenizer=tokenizer, embedding_dim=model_dim)
    small_model = SentenceTransformer(modules=[static])
    router = Router.for_query_document(query_modules=[small_model], document_modules=[large_model])  # type: ignore
    model = SentenceTransformer(modules=[router])

    # train_dataset, eval_dataset = load_retrieval_train_eval_datasets()
    train_dataset, eval_dataset = load_retrieval_train_eval_datasets()
    logger.info(train_dataset)

    base_loss = MultipleNegativesRankingLoss(model)
    dims = [model_dim]
    while dims[-1] > 32:
        dims.append(dims[-1] // 2)
    matryoshka_dims = sorted(dims)
    matryoshka_dims = [d for d in matryoshka_dims if d <= model_dim]
    loss = MatryoshkaLoss(model, base_loss, matryoshka_dims=matryoshka_dims)

    run_name = "train-router-nanobeir"
    wandb.init(project="train-router", name=run_name)
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        learning_rate=0.05,
        lr_scheduler_type="linear",
        warmup_ratio=0.0,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        logging_steps=25,
        logging_first_step=True,
        run_name=run_name,
        use_cpu=False,
        report_to=["wandb"],
        weight_decay=0.0,
        router_mapping={
            "question": "query",
            "query": "query",
            "title": "query",
            "anchor": "query",
            "answer": "document",
            "positive": "document",
            "negative": "document",
            "negative1": "document",
            "negative2": "document",
            "negative3": "document",
            "negative4": "document",
            "negative5": "document",
            "negative6": "document",
            "negative7": "document",
            "negative8": "document",
            "negative9": "document",
            "negative10": "document",
            "negative11": "document",
            "negative12": "document",
            "negative13": "document",
            "negative14": "document",
            "negative15": "document",
            "negative16": "document",
            "negative17": "document",
            "negative18": "document",
            "negative19": "document",
            "negative20": "document",
            "abstract": "document",
            "text": "document",
        },
    )

    evaluator = NanoBEIREvaluator()

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    large_model.save_pretrained(f"supervised_models/{run_name}/final")
