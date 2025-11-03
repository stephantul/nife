from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import torch
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizer

from pynife.distillation.infer import (
    _batchify,
    _generate_embeddings,
    _tokenize,
    _write_data,
    generate_and_save_embeddings,
)


def test_batchify() -> None:
    """Test the _batchify function."""
    data = [1, 2, 3, 4, 5]
    batch_size = 2
    batches = list(_batchify(data, batch_size))
    assert batches == [[1, 2], [3, 4], [5]]


def test_tokenize(test_tokenizer: PreTrainedTokenizer) -> None:
    """Test the _tokenize function using the on-disk test tokenizer."""
    # ensure there is a pad token so tokenization with padding works
    strings = ["hello world", "test string"]
    max_length = 10
    tokenized, truncated = _tokenize(strings, test_tokenizer, max_length)

    # tokenized input_ids should be a 2D tensor and truncated should be a list of strings
    assert isinstance(tokenized["input_ids"], torch.Tensor) and tokenized["input_ids"].ndim == 2  # type: ignore
    assert isinstance(truncated, list) and all(isinstance(x, str) for x in truncated)


def test_write_data() -> None:
    """Test the _write_data function."""
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir)
        pooled = [torch.tensor([[1.0, 2.0], [3.0, 4.0]])]
        records = [{"text": "example1", "truncated": "example1"}, {"text": "example2", "truncated": "example2"}]
        _write_data(path, pooled, records, 0)

        # Check if files are created
        assert (path / "pooled_0000.pt").exists()
        assert (path / "texts_0000.txt").exists()


@patch("pynife.distillation.infer._write_data")
def test_generate_embeddings(mock_write_data, test_model: SentenceTransformer) -> None:
    """Test the _generate_embeddings function."""
    records = iter([{"text": "example1"}, {"text": "example2"}])
    with TemporaryDirectory() as temp_dir:
        _generate_embeddings(test_model, records, temp_dir, batch_size=1, save_every=1)

        # Ensure _write_data is called (patched)
        mock_write_data.assert_called()


@patch("pynife.distillation.infer._generate_embeddings")
@patch("pynife.distillation.infer.build_parquet_shards_from_folder")
def test_generate_and_save_embeddings(mock_build_shards, mock_generate_embeddings) -> None:
    """Test the generate_and_save_embeddings function."""
    mock_model = MagicMock()
    records = iter([{"text": "example1"}, {"text": "example2"}])
    with TemporaryDirectory() as temp_dir:
        # generate_and_save_embeddings signature requires model_name and dataset_name now
        generate_and_save_embeddings(mock_model, "mock-model", "mock-dataset", temp_dir, records)

        # Ensure _generate_embeddings and build_parquet_shards_from_folder are called
        mock_generate_embeddings.assert_called()
        mock_build_shards.assert_called()


@patch("pynife.distillation.infer._write_data")
def test_generate_embeddings_with_max_length_and_limit(mock_write_data, test_model: SentenceTransformer) -> None:
    """Ensure _generate_embeddings handles max_length > model max and limit_batches/save_every branches."""
    records = iter([{"text": "example1"}, {"text": "example2"}])
    with TemporaryDirectory() as temp_dir:
        # save_every=1 ensures we hit the intermediate save branch, limit_batches=1 triggers stopping
        from pynife.distillation.infer import _generate_embeddings

        _generate_embeddings(test_model, records, temp_dir, batch_size=1, save_every=1, limit_batches=1, max_length=512)

        # _write_data should be called at least once because save_every == 1
        assert mock_write_data.called
        # original max length should remain unchanged
        assert test_model.max_seq_length == 512


@patch("pynife.distillation.infer.logger.warning")
def test_generate_embeddings_warns_when_max_length_too_large(mock_warning, test_model: SentenceTransformer) -> None:
    """Trigger the warning when requested max_length is larger than the model's original max length."""
    test_model.max_seq_length = 10
    with TemporaryDirectory() as temp_dir:
        _generate_embeddings(test_model, iter([]), temp_dir, batch_size=1, max_length=512)

    mock_warning.assert_called()


@patch("pynife.distillation.infer._write_data")
@patch("pynife.distillation.infer._tokenize")
def test_generate_embeddings_limit_batches_logs_and_writes(mock_tokenize, mock_write_data) -> None:
    """Ensure limit_batches triggers info log and causes a final write of accumulated records."""
    mock_model = MagicMock()
    mock_model.tokenizer = MagicMock()
    # Use a real torch.device so tensor.to(model.device) accepts it and model.device.type exists
    mock_model.device = torch.device("cpu")
    # set original max length larger than provided max to avoid warning
    mock_model.__getitem__.return_value.max_seq_length = 512

    # _tokenize should return features and truncated strings
    mock_tokenize.return_value = ({"input_ids": torch.tensor([[1]])}, ["ex"])

    # model call returns embeddings
    mock_model.return_value = {"sentence_embedding": torch.tensor([[0.1]])}

    records = iter([{"text": "example1"}])
    from pynife.distillation import infer as infer_mod

    with TemporaryDirectory() as temp_dir:
        # limit_batches=1 will cause the info log and break
        infer_mod._generate_embeddings(
            mock_model, records, temp_dir, batch_size=1, save_every=100, limit_batches=1, max_length=128
        )

    # _write_data should be called to write the accumulated records at the end
    assert mock_write_data.called
