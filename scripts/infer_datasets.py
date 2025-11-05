import logging
from argparse import ArgumentParser, Namespace
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import cast

from datasets import Dataset, concatenate_datasets, load_dataset
from sentence_transformers import SentenceTransformer

from pynife.distillation.infer import generate_and_save_embeddings


def _simple_text_field_dataset(
    huggingface_name: str,
    text_field: str,
    config: str | None = None,
    split: str = "train",
) -> Iterator[dict[str, str]]:
    """Helper function for datasets that extract a single text field."""
    dataset = cast(Dataset, load_dataset(huggingface_name, config, split=split))

    new_records: list[dict[str, str]] = []
    for record in cast(Iterable[dict[str, str]], dataset):
        text = record[text_field]
        new_records.append({"text": text})

    return cast(Iterator[dict[str, str]], iter(new_records))


def english_words_definitions_dataset() -> tuple[str, str, Iterator[dict[str, str]]]:
    """
    Get the English Words Definitions dataset.

    Returns a tuple of (descriptive_name, huggingface_name, iterator).
    """
    hf = "MongoDB/english-words-definitions"
    name = "english-words-definitions"
    dataset = cast(Dataset, load_dataset(hf, split="train"))
    dataset = dataset.map(lambda x: {"text": " ".join(x["definitions"])})
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    dataset_iterator = cast(Iterator[dict[str, str]], iter(dataset))

    return name, hf, dataset_iterator


def fineweb_dataset() -> tuple[str, str, Iterator[dict[str, str]]]:
    """Get the FineWeb dataset (returns first path only)."""
    hf = "HuggingFaceFW/fineweb"
    name = "fineweb"
    data = cast(Iterator[dict[str, str]], load_dataset(hf, "sample-10BT", streaming=True, split="train"))
    return name, hf, iter(data)


def gooaq_dataset() -> tuple[str, str, Iterator[dict[str, str]]]:
    """Get the GoOAQ dataset."""
    hf = "sentence-transformers/gooaq"
    name = "gooaq"
    return name, hf, _simple_text_field_dataset(hf, text_field="question")


def miracl_dataset() -> tuple[str, str, Iterator[dict[str, str]]]:
    """Get the MIRACL dataset."""
    hf = "sentence-transformers/miracl"
    name = "miracl"
    return name, hf, _simple_text_field_dataset(hf, text_field="anchor", config="en-triplet")


def lotte_queries_dataset() -> tuple[str, str, Iterator[dict[str, str]]]:
    """Get the LoTTE queries dataset."""
    query_paths = [
        "lifestyle_forum-queries",
        "lifestyle_search-queries",
        "recreation_forum-queries",
        "recreation_search-queries",
        "science_forum-queries",
        "science_search-queries",
        "technology_forum-queries",
        "technology_search-queries",
        "writing_forum-queries",
        "writing_search-queries",
    ]

    big_dataset = []
    for path in query_paths:
        dataset = cast(Dataset, load_dataset("mteb/lotte", path, split="dev"))
        big_dataset.append(dataset)
    final_dataset: Dataset = concatenate_datasets(big_dataset)
    dataset_iterator = cast(Iterator[dict[str, str]], iter(final_dataset))
    return "lotte", "mteb/lotte", dataset_iterator


def snli_dataset() -> tuple[str, str, Iterator[dict[str, str]]]:
    """Get the SNLI dataset."""
    hf = "stanfordnlp/snli"
    name = "snli"
    dataset = cast(Dataset, load_dataset(hf, split="train"))

    seen = set()
    new_records: list[dict[str, str]] = []
    for record in cast(Iterable[dict[str, str]], dataset):
        text = record["premise"]
        if text not in seen:
            seen.add(text)
            new_records.append({"text": text})
        text = record["hypothesis"]
        if text not in seen:
            seen.add(text)
            new_records.append({"text": text})

    dataset_iterator = cast(Iterator[dict[str, str]], iter(new_records))
    return name, hf, dataset_iterator


def paws_dataset() -> tuple[str, str, Iterator[dict[str, str]]]:
    """Get the PAWS dataset."""
    hf = "google-research-datasets/paws"
    name = "paws"
    dataset = load_dataset(hf, "unlabeled_final", split="train")

    new_records = []
    for record in cast(Iterable[dict[str, str]], dataset):
        text = record["sentence1"]
        new_records.append({"text": text})
        text = record["sentence2"]
        new_records.append({"text": text})

    dataset_iterator = cast(Iterator[dict[str, str]], iter(new_records))
    return name, hf, dataset_iterator


def squad_dataset() -> tuple[str, str, Iterator[dict[str, str]]]:
    """Get the SQuAD dataset."""
    hf = "sentence-transformers/squad"
    name = "squad"
    return name, hf, _simple_text_field_dataset(hf, text_field="question")


def mldr_dataset() -> tuple[str, str, Iterator[dict[str, str]]]:
    """Get the MLDR dataset."""
    hf = "sentence-transformers/mldr"
    name = "mldr"
    return name, hf, _simple_text_field_dataset(hf, text_field="anchor", config="en-triplet")


def msmarco_queries_dataset() -> tuple[str, str, Iterator[dict[str, str]]]:
    """Get the MS MARCO queries dataset."""
    hf = "sentence-transformers/msmarco-corpus"
    name = "msmarco_queries"
    return name, hf, _simple_text_field_dataset(hf, text_field="text", config="query")


def msmarco_docs_dataset() -> tuple[str, str, Iterator[dict[str, str]]]:
    """Get the MS MARCO documents dataset."""
    hf = "sentence-transformers/msmarco-corpus"
    name = "msmarco_docs"
    return name, hf, _simple_text_field_dataset(hf, text_field="text", config="passage")


def pubmed_dataset() -> tuple[str, str, Iterator[dict[str, str]]]:
    """Get the PubMedQA dataset."""
    hf = "qiaojin/PubMedQA"
    name = "PubMedQA"
    subsets = ["pqa_artificial", "pqa_unlabeled"]
    datasets = []
    for subset in subsets:
        dataset = cast(Dataset, load_dataset(name, subset, split="train"))
        dataset = dataset.rename_columns({"question": "text"})
        columns_to_remove = [col for col in dataset.column_names if col not in ("text",)]
        dataset = dataset.remove_columns(columns_to_remove)
        datasets.append(dataset)

    final_dataset = concatenate_datasets(datasets)
    dataset_iterator = cast(Iterator[dict[str, str]], iter(final_dataset))
    return name, hf, dataset_iterator


def swim_ir_dataset() -> tuple[str, str, Iterator[dict[str, str]]]:
    """Get the SWIM-IR dataset."""
    hf = "sentence-transformers/swim-ir"
    name = "swim-ir"
    return name, hf, _simple_text_field_dataset(hf, text_field="query", config="en")


def triviaqa_dataset() -> tuple[str, str, Iterator[dict[str, str]]]:
    """Get the TriviaQA dataset."""
    hf = "mandarjoshi/trivia_qa"
    name = "trivia_qa"
    dataset = cast(Dataset, load_dataset(hf, "unfiltered.nocontext", split="train"))
    dataset = dataset.rename_columns({"question": "text"})
    columns_to_remove = [col for col in dataset.column_names if col not in ("text",)]
    dataset = dataset.remove_columns(columns_to_remove)
    dataset_iterator = cast(Iterator[dict[str, str]], iter(dataset))
    return name, hf, dataset_iterator


def mr_tydi_dataset() -> tuple[str, str, Iterator[dict[str, str]]]:
    """Get the Mr. TyDi dataset."""
    hf = "sentence-transformers/mr-tydi"
    name = "mr-tydi"
    return name, hf, _simple_text_field_dataset(hf, text_field="anchor", config="en-triplet")


def get_all_dataset_functions() -> dict[str, Callable[[], tuple[str, str, Iterator[dict[str, str]]]]]:
    """Get all available dataset functions."""
    return {
        "english-words-definitions": english_words_definitions_dataset,
        "fineweb": fineweb_dataset,
        "gooaq": gooaq_dataset,
        "miracl": miracl_dataset,
        "lotte": lotte_queries_dataset,
        "snli": snli_dataset,
        "paws": paws_dataset,
        "squad": squad_dataset,
        "mldr": mldr_dataset,
        "msmarco": msmarco_queries_dataset,
        "msmarco_docs": msmarco_docs_dataset,
        "PubMedQA": pubmed_dataset,
        "swim-ir-monolingual": swim_ir_dataset,
        "trivia_qa": triviaqa_dataset,
        "mr-tydi": mr_tydi_dataset,
    }


def short_dataset_name(hf_name: str) -> str:
    """
    Return the short dataset name from a HF hub identifier.

    Examples:
        - 'MongoDB/english-words-definitions' -> 'english-words-definitions'
        - 'msmarco' -> 'msmarco'

    This centralizes the logic for deriving local filenames from full HF paths.

    Args:
        hf_name: Full HF dataset identifier.

    Returns:
        Short dataset name.

    """
    if not hf_name:
        return hf_name
    return hf_name.split("/")[-1]


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
    parser.add_argument("--lowercase", action="store_true", help="Whether to lowercase the tokenizer.")
    parser.add_argument("--make-greedy", action="store_true", help="Whether to make the tokenizer greedy.")

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

        # Get dataset function and call it. Each function returns
        # (descriptive_name, huggingface_name, iterator)
        dataset_func = dataset_functions[dataset_name]
        descriptive_name, full_name, dataset_iterator = dataset_func()

        converted_dir = Path(args.converted_base_dir) / safe_model_name / descriptive_name
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
            dataset_name=full_name,
            lowercase=args.lowercase,
            make_greedy=args.make_greedy,
        )


if __name__ == "__main__":
    main()
