import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from sentence_transformers import SentenceTransformer

from pystatic.dataset_vault import get_all_dataset_functions, short_dataset_name
from pystatic.distillation.infer import generate_and_save_embeddings


def parse_args() -> Namespace:
    """Parse command line arguments for unified inference script."""
    parser = ArgumentParser(description="Run inference on multiple datasets with different models")

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the pre-trained model to use for inference.",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="List of dataset names to process. Available datasets: " + ", ".join(get_all_dataset_functions().keys()),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for inference (default: 512)",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=256,
        help="Save embeddings every N batches (default: 256)",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max length for inference (default: 512)",
    )

    parser.add_argument(
        "--limit-batches",
        type=int,
        help="Limit the number of batches to process (for testing)",
    )

    parser.add_argument(
        "--converted-base-dir",
        type=str,
        default="converted_datasets",
        help="Base directory for converted dataset files (default: converted_datasets)",
    )

    return parser.parse_args()


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for use in file paths."""
    return model_name.replace("/", "__")


def main() -> None:
    """Main function to run inference on multiple datasets."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    args = parse_args()

    # Get all available dataset functions
    dataset_functions = get_all_dataset_functions()

    # Validate dataset names
    invalid_datasets = [name for name in args.datasets if name not in dataset_functions]
    if invalid_datasets:
        logger.error(f"Invalid dataset names: {invalid_datasets}")
        logger.info(f"Available datasets: {list(dataset_functions.keys())}")
        return

    # Load model
    logger.info(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    # Sanitize model name for file paths
    safe_model_name = sanitize_model_name(args.model_name)

    # Process each dataset
    for dataset_name in args.datasets:
        logger.info(f"Processing dataset: {dataset_name}")

        # Get dataset
        dataset_func = dataset_functions[dataset_name]
        actual_name, dataset_iterator = dataset_func()

        # actual_name now contains the full HF hub path; derive the last path segment for local saving
        short_name = short_dataset_name(actual_name)

        converted_dir = Path(args.converted_base_dir) / safe_model_name / short_name
        converted_dir.mkdir(parents=True, exist_ok=True)
        generate_and_save_embeddings(
            model=model,
            records=dataset_iterator,
            output_folder=converted_dir,
            limit_batches=args.limit_batches,
            batch_size=args.batch_size,
            save_every=args.save_every,
            max_length=args.max_length,
            model_name=args.model_name,
            dataset_name=short_name,
        )


if __name__ == "__main__":
    main()
