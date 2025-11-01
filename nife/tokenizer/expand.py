import logging
import re

import numpy as np
from skeletoken import TokenizerModel
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from nife.utilities import batchify

FreqTuple = tuple[str, int]


_NUMBERS_RE = re.compile(r"^\d+$")

logger = logging.getLogger(__name__)


def _prune_tokenizer(
    tokenizer_model: TokenizerModel, data: list[FreqTuple], min_subword_frequency: int
) -> TokenizerModel:
    """Prune a tokenizer by removing tokens that occur less than min_subword_frequency times in the provided data."""
    tokenizer_object = tokenizer_model.to_tokenizer()

    old_vocab_size = tokenizer_object.get_vocab_size()
    original_vocab_counts = np.zeros(old_vocab_size, dtype=np.int32)

    for batch in tqdm(batchify(data, batch_size=10_000)):
        token_strings, token_counts = zip(*batch)
        tokenized = tokenizer_object.encode_batch_fast(token_strings, add_special_tokens=False)
        for encoding, count in zip(tokenized, token_counts):
            counts = np.bincount(encoding.ids, minlength=old_vocab_size)
            original_vocab_counts += counts * count

    vocab_to_remove = original_vocab_counts < min_subword_frequency

    vocabulary = {index: token for token, index in tokenizer_model.vocabulary.items()}
    for id in np.flatnonzero(vocab_to_remove):
        token = vocabulary[id]
        # Prevent removing tokens if that would empty the vocabulary
        if tokenizer_model.vocabulary_size == 1:
            break
        tokenizer_model.remove_token_from_vocabulary(token)

    logger.info("Removed %d tokens from vocabulary.", np.sum(vocab_to_remove))
    return tokenizer_model


def _add_tokens_to_tokenizer(
    tokenizer_model: TokenizerModel, data: list[FreqTuple], filter_numbers: bool, new_vocab_size: int
) -> TokenizerModel:
    """Add new tokens to a tokenizer up to the specified new vocabulary size."""
    n_tokens_to_add = new_vocab_size - tokenizer_model.vocabulary_size
    if n_tokens_to_add <= 0:
        logger.info("No tokens to add to vocabulary.")
        return tokenizer_model
    logger.info("Adding %d new tokens to vocabulary.", n_tokens_to_add)

    tokens_added = 0
    # Sort by frequency and add tokens until we reach the new vocab size
    for token, _ in sorted(data, key=lambda x: x[1], reverse=True):
        if filter_numbers and _NUMBERS_RE.match(token):
            continue
        if tokens_added >= n_tokens_to_add:
            break
        if token in tokenizer_model.vocabulary:
            continue
        tokenizer_model.add_token_to_vocabulary(token)
        tokens_added += 1

    return tokenizer_model


def expand_tokenizer(
    tokenizer: PreTrainedTokenizerFast,
    data: list[FreqTuple],
    min_subword_frequency: int,
    new_vocab_size: int,
    filter_numbers: bool,
) -> PreTrainedTokenizerFast:
    """Expand a tokenizer's vocabulary using a TokenizerModel."""
    tokenizer_model = TokenizerModel.from_transformers_tokenizer(tokenizer)
    tokenizer_model = _prune_tokenizer(tokenizer_model, data, min_subword_frequency)
    new_tokenizer = _add_tokens_to_tokenizer(tokenizer_model, data, filter_numbers, new_vocab_size)
    return new_tokenizer.to_transformers()
