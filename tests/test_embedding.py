import numpy as np
import torch
from tokenizers import Tokenizer

from pynife.embedding import TrainableStaticEmbedding


def _load_test_tokenizer(path: str = "tests/data/test_tokenizer/tokenizer.json") -> Tokenizer:
    return Tokenizer.from_file(path)


def test_max_seq_length_getter_setter() -> None:
    """max_seq_length property can be read and updated."""
    tok = _load_test_tokenizer()
    model = TrainableStaticEmbedding(tokenizer=tok, embedding_dim=3)
    orig = model.max_seq_length
    model.max_seq_length = 7
    assert model.max_seq_length == 7
    assert orig != model.max_seq_length


def test_tokenize_truncates_and_pads() -> None:
    """Tokenization returns padded/truncated input ids as a torch tensor."""
    tok = _load_test_tokenizer()
    model = TrainableStaticEmbedding(tokenizer=tok, embedding_dim=3)
    # set a small max length to force truncation
    model.max_seq_length = 2

    texts = ["hello world", "hello"]
    features = model.tokenize(texts)

    assert "input_ids" in features
    ids = features["input_ids"]
    assert isinstance(ids, torch.Tensor)
    # should be (batch, seq_len) with seq_len <= model.max_seq_length
    assert ids.shape[0] == 2
    assert ids.shape[1] <= model.max_seq_length


def test_forward_computes_mean_pooling_embedding() -> None:
    """Forward computes mean-pooled embeddings matching the embedding table."""
    tok = _load_test_tokenizer()
    # build small deterministic embedding weights matching tokenizer vocab size
    vocab = tok.get_vocab()
    vocab_size = max(vocab.values()) + 1
    dim = 3
    weights = np.arange(vocab_size * dim, dtype=np.float32).reshape(vocab_size, dim)

    model = TrainableStaticEmbedding(tokenizer=tok, embedding_weights=weights)

    # single short text to avoid padding complications
    text = "hello"
    features = model.tokenize([text])
    out = model.forward(features)

    assert "sentence_embedding" in out
    emb = out["sentence_embedding"]
    assert emb.shape[0] == 1 and emb.shape[1] == dim

    # compute expected mean embedding from our deterministic weights
    enc = tok.encode(text, add_special_tokens=False)
    ids = enc.ids[: model.max_seq_length]
    expected = torch.tensor(np.mean(weights[ids], axis=0), dtype=torch.float32)

    assert torch.allclose(emb[0], expected, atol=1e-5)
