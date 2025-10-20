import argparse
import logging
import random
from itertools import product

from scripts.experiment_distillation import initialize_model, load_data, run_experiment

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

    initialize_from_model = ["mixedbread-ai/mxbai-embed-large-v1"]
    l2_norms = [1e-4, 1e-3]

    for model_to_initialize_from, l2_norm in product(initialize_from_model, l2_norms):
        experiment_name_parts = [
            parsed_args.name,
            f"init_{'none' if model_to_initialize_from is None else 'from_model'}",
            f"l2norm_{l2_norm}",
        ]
        parsed_args.experiment_name = "_".join(experiment_name_parts)

        model = initialize_model(
            tokenizer_path=parsed_args.tokenizer_path,
            model_to_initialize_from=model_to_initialize_from,
            model_dim=model_dim,
            with_norm=False,
            with_weights=False,
        )
        dataset, n_samples = load_data(
            train_datasets=parsed_args.train_dataset,
            in_memory=parsed_args.in_memory,
            limit_shards=parsed_args.limit_shards,
        )
        run_experiment(
            model,
            parsed_args.experiment_name,
            dataset,
            n_samples,
            parsed_args.batch_size,
            parsed_args.learning_rate,
            parsed_args.epochs,
            l2_norm=l2_norm,
        )
