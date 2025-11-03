from pynife.dataset_vault import short_dataset_name


def test_short_dataset_name_with_hf_path() -> None:
    """Return the last segment when given a HF hub path with an owner prefix."""
    assert short_dataset_name("MongoDB/english-words-definitions") == "english-words-definitions"


def test_short_dataset_name_without_slash() -> None:
    """Return the same name when no slash is present."""
    assert short_dataset_name("msmarco") == "msmarco"


def test_short_dataset_name_empty() -> None:
    """Handle empty string input by returning an empty string."""
    assert short_dataset_name("") == ""
