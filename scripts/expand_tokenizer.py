import logging
import re
from argparse import ArgumentParser, Namespace
from collections.abc import Iterator
from typing import cast

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from skeletoken import TokenizerModel
from tqdm import tqdm

from pystatic.utilities import batchify

logger = logging.getLogger(__name__)

_NUMBERS_RE = re.compile(r"^\d+$")


def _parse_args() -> Namespace:
    """Parse command line arguments for expanding tokenizer vocabulary."""
    parser = ArgumentParser(description="Expand tokenizer vocabulary.")
    parser.add_argument("--tokenizer-name", type=str, required=True, help="Name of the tokenizer model.")
    parser.add_argument("--output", type=str, required=True, help="Output path for the expanded tokenizer.")
    parser.add_argument(
        "--vocabulary-data",
        type=str,
        nargs="+",
        help="Path to one or more vocabulary datasets. This is a dataset with a 'token' column, sorted by frequency.",
    )
    parser.add_argument("--vocab-size", type=int, default=30000, help="New vocabulary size after expansion.")
    parser.add_argument(
        "--min-subword-frequency", type=int, default=100, help="Minimum frequency for subwords to be included."
    )
    parser.add_argument("--filter-numbers", action="store_true", help="Filter out tokens that are purely numeric.")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parsed_args = _parse_args()
    tokenizer_model = TokenizerModel.from_pretrained(parsed_args.tokenizer_name)
    tokenizer = tokenizer_model.to_tokenizer()

    datasets = []
    for data in parsed_args.vocabulary_data:
        datasets.append(cast(Dataset, load_dataset(data, split="train")))
    dataset = concatenate_datasets(datasets)

    old_vocab_size = tokenizer.get_vocab_size()
    original_vocab_counts = np.zeros(old_vocab_size, dtype=np.int32)

    for batch in tqdm(batchify(dataset["token"], batch_size=10_000)):
        tokenized = tokenizer.encode_batch_fast(batch, add_special_tokens=False)
        for encoding in tokenized:
            counts = np.bincount(encoding.ids, minlength=old_vocab_size)
            original_vocab_counts += counts

    vocab_to_remove = original_vocab_counts < parsed_args.min_subword_frequency

    vocabulary = {index: token for token, index in tokenizer_model.vocabulary.items()}
    for id in np.flatnonzero(vocab_to_remove):
        token = vocabulary[id]
        tokenizer_model.remove_token_from_vocabulary(token)

    logger.info("Removed %d tokens from vocabulary.", np.sum(vocab_to_remove))
    n_tokens_to_add = parsed_args.vocab_size - tokenizer_model.vocabulary_size
    logger.info("Adding %d new tokens to vocabulary.", n_tokens_to_add)

    all_tokens = cast(Iterator[str], dataset["token"])
    tokens_added = 0
    for token in all_tokens:
        if parsed_args.filter_numbers and _NUMBERS_RE.match(token):
            continue
        if tokens_added >= n_tokens_to_add:
            break
        if token in tokenizer_model.vocabulary:
            continue
        tokenizer_model.add_token_to_vocabulary(token)
        tokens_added += 1

    tokenizer_model.to_transformers().save_pretrained(parsed_args.output)
