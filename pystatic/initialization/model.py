import torch
from sentence_transformers import SentenceTransformer
from skeletoken import TokenizerModel
from transformers import PreTrainedTokenizerFast


def initialize_from_model(model: SentenceTransformer, tokenizer: PreTrainedTokenizerFast) -> torch.Tensor:
    """Initialize embeddings from a SentenceTransformer model and a tokenizer."""
    tokenizer_model = TokenizerModel.from_transformers_tokenizer(tokenizer)

    tokens = tokenizer_model.sorted_vocabulary
    word_prefix = tokenizer_model.word_prefix or ""
    subword_prefix = tokenizer_model.subword_prefix or ""

    vocab = []
    for token in tokens:
        token = token.removeprefix(word_prefix).removeprefix(subword_prefix)
        vocab.append(token)

    weights = model.encode(
        vocab, batch_size=512, convert_to_numpy=False, convert_to_tensor=True, show_progress_bar=True
    )
    return weights
