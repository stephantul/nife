import logging
from argparse import ArgumentParser, Namespace
from collections import defaultdict

from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from pynife.tokenizer.expand import expand_tokenizer

logger = logging.getLogger(__name__)


def _parse_args() -> Namespace:
    """Parse command line arguments for expanding tokenizer vocabulary."""
    parser = ArgumentParser(description="Expand tokenizer vocabulary.")
    parser.add_argument("--tokenizer-name", type=str, required=True, help="Name of the tokenizer model.")
    parser.add_argument("--output-folder", type=str, required=True, help="Output folder for the expanded tokenizer.")
    parser.add_argument(
        "--vocabulary-data",
        type=str,
        nargs="+",
        help="Path to one or more vocabulary datasets. This is a dataset with a 'token' column, sorted by frequency.",
    )
    parser.add_argument("--vocab-size", type=int, default=100_000, help="New vocabulary size after expansion.")
    parser.add_argument(
        "--min-subword-frequency", type=int, default=10, help="Minimum frequency for subwords to be included."
    )
    parser.add_argument("--filter-numbers", action="store_true", help="Filter out tokens that are purely numeric.")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parsed_args = _parse_args()
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(parsed_args.tokenizer_name, use_fast=True)

    counts: dict[str, int] = defaultdict(int)

    datasets = []
    for dataset in parsed_args.vocabulary_data:
        ds = load_dataset(dataset, split="train")
        datasets.append(ds)
    dataset = concatenate_datasets(datasets)

    new_tokenizer = expand_tokenizer(
        tokenizer=tokenizer,
        dataset=dataset,
        new_vocab_size=parsed_args.vocab_size,
        filter_numbers=parsed_args.filter_numbers,
        min_subword_frequency=parsed_args.min_subword_frequency,
    )

    new_tokenizer.save_pretrained(parsed_args.output_folder)
