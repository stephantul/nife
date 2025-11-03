from argparse import ArgumentParser, Namespace
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from pynife.data import get_datasets
from pynife.tokenizer.count_vocabulary import count_tokens_in_dataset


def _parse_args() -> Namespace:
    """Parse command line arguments for counting tokens."""
    parser = ArgumentParser(description="Count tokens in a text file.")
    parser.add_argument("--output", type=str, required=True, help="Output file for token counts.")
    parser.add_argument("--model-name", required=True, help="Name of the model to use.")
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Path to the datasets",
        nargs="+",
    )
    parser.add_argument("--text-column-name", type=str, default="text", help="Name of the text column.")
    parser.add_argument("--in-memory", action="store_true", help="Load the dataset in memory.")

    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = _parse_args()
    text_column_name: str = parsed_args.text_column_name
    in_memory = bool(parsed_args.in_memory)

    tokenizer = AutoTokenizer.from_pretrained(parsed_args.model_name, use_fast=True)

    data, total = get_datasets(
        [Path(p) for p in parsed_args.datasets], in_memory=in_memory, columns_to_keep={text_column_name}
    )

    records = (x[text_column_name] for x in iter(data))
    vocab_items = count_tokens_in_dataset(
        tokenizer=tokenizer,
        data=records,
        total=total,
    )
    dataset = Dataset.from_list(vocab_items)  # type: ignore  # datasets typing issue
    dataset.save_to_disk(parsed_args.output)
