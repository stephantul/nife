from pytest import fixture
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerFast


@fixture(scope="session")
def test_model(path: str = "tests/data/test_model") -> SentenceTransformer:
    """Load the small test SentenceTransformer from the provided path."""
    model = SentenceTransformer(path, device="cpu")
    model.max_seq_length = 512
    return model


@fixture(scope="session")
def test_tokenizer(path: str = "tests/data/test_tokenizer") -> PreTrainedTokenizerFast:
    """Load the small test tokenizer from the provided path."""
    return PreTrainedTokenizerFast.from_pretrained(path)
