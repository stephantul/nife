from transformers import PreTrainedTokenizerFast

from nife.tokenizer.count_vocabulary import count_tokens_in_dataset


def test_count_tokens_in_dataset_counts_and_df(test_tokenizer: PreTrainedTokenizerFast) -> None:
    """
    Counts frequencies and document frequencies for a tiny dataset.

    This test uses the real `test_tokenizer` fixture but mocks the
    TokenizerModel and Preprocessor used inside the module so the test
    remains fast and deterministic.
    """
    # Prepare tiny dataset: three docs, tokens overlapping
    data = [
        {"text": "apple banana apple"},
        {"text": "banana orange"},
        {"text": "apple orange"},
    ]

    vocab_map = count_tokens_in_dataset(test_tokenizer, iter(data), total=3)
    # Expected frequencies
    assert vocab_map == [
        {"token": "apple", "frequency": 3, "document_frequency": 2},
        {"token": "banana", "frequency": 2, "document_frequency": 2},
        {"token": "orange", "frequency": 2, "document_frequency": 2},
    ]
