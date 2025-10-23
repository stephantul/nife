from typing import cast

import torch
from sentence_transformers import SentenceTransformer
from skeletoken import TokenizerModel
from tqdm import trange
from transformers import PreTrainedTokenizerFast


def initialize_from_model(model: SentenceTransformer, tokenizer: PreTrainedTokenizerFast) -> torch.Tensor:
    """Initialize embeddings from a SentenceTransformer model and a tokenizer."""
    tokenizer_model = TokenizerModel.from_transformers_tokenizer(tokenizer)

    original_tokenizer = model.tokenizer
    assert original_tokenizer is not None
    original_tokenizer = cast(PreTrainedTokenizerFast, original_tokenizer)

    original_tokenizer_model = TokenizerModel.from_transformers_tokenizer(original_tokenizer)
    pad_token = original_tokenizer_model.pad_token
    pad_token_id = original_tokenizer_model.model.vocab[pad_token] if pad_token is not None else 0
    original_vocabulary = original_tokenizer.get_vocab()

    tokens = tokenizer_model.sorted_vocabulary
    word_prefix = tokenizer_model.word_prefix or ""
    subword_prefix = tokenizer_model.subword_prefix or ""

    indices = []
    for token in tokens:
        token_without_prefix = token.removeprefix(word_prefix).removeprefix(subword_prefix)
        if token in original_vocabulary:
            token_ids = original_tokenizer.build_inputs_with_special_tokens([original_vocabulary[token]])
        elif token_without_prefix in original_vocabulary:
            token_ids = original_tokenizer.build_inputs_with_special_tokens([original_vocabulary[token_without_prefix]])
        else:
            token_ids = original_tokenizer.encode(token_without_prefix, add_special_tokens=True)
        indices.append(token_ids)

    sort_order = torch.argsort(torch.tensor([len(ids) for ids in indices]))
    indices = [indices[i] for i in sort_order]

    batch_size = 256
    out = []
    for batch_start in trange(0, len(indices), batch_size, desc="Initializing embeddings"):
        batch_indices = indices[batch_start : batch_start + batch_size]
        batch = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(token_ids) for token_ids in batch_indices],
            batch_first=True,
            padding_value=pad_token_id,
        )
        input_dict = {
            "input_ids": batch.to(model.device),
            "attention_mask": (batch != pad_token_id).long().to(model.device),
        }
        with torch.no_grad():
            embeddings = model.forward(input_dict)["sentence_embedding"]
        out.append(embeddings)

    return torch.cat(out, dim=0)[torch.argsort(sort_order)]
