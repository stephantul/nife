from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerFast

from pynife.initialization.model import initialize_from_model


def test_initialize_from_model(test_model: SentenceTransformer, test_tokenizer: PreTrainedTokenizerFast) -> None:
    """Test the initialize_from_model function."""
    embeddings = initialize_from_model(test_model, test_tokenizer)

    assert embeddings.shape == (203, 3)
