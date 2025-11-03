from transformers import PreTrainedTokenizerFast

from pynife.tokenizer.count_vocabulary import count_tokens_in_dataset


def test_count_tokens_in_dataset_counts_and_df(test_tokenizer: PreTrainedTokenizerFast) -> None:
    """
    Counts frequencies and document frequencies for a tiny dataset.

    This test uses the real `test_tokenizer` fixture but mocks the
    TokenizerModel and Preprocessor used inside the module so the test
    remains fast and deterministic.
    """
    # Prepare tiny dataset: three docs, tokens overlapping
    data = [
        "apple banana apple",
        "banana orange",
        "apple orange",
    ]

    vocab_map = count_tokens_in_dataset(test_tokenizer, data, total=3)
    # Expected frequencies
    assert vocab_map == [
        {"token": "apple", "frequency": 3},
        {"token": "banana", "frequency": 2},
        {"token": "orange", "frequency": 2},
    ]
