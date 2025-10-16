import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedTokenizer

logger = logging.getLogger(__name__)


T = TypeVar("T")


def _batchify(records: Iterator[T], batch_size: int) -> Iterator[list[T]]:
    """Turn a list of texts into batches."""
    batch: list[T] = []
    pbar = tqdm(total=0, unit="batches", desc="Creating batches")
    while True:
        if len(batch) < batch_size:
            try:
                batch.append(next(records))
            except StopIteration:
                if len(batch) > 0:
                    yield batch
                break
        else:
            yield batch
            batch = []
            pbar.update(1)

    pbar.close()


def _write_data(path: Path, pooled: list[torch.Tensor], records: list[dict[str, str]], shard_index: int) -> None:
    """Write out the data to disk."""
    pooled_tensor = torch.cat(pooled, dim=0).float()
    torch.save(pooled_tensor, path / f"pooled_{shard_index:04d}.pt")
    with open(path / f"texts_{shard_index:04d}.txt", "w", encoding="utf-8") as f:
        for i, record in enumerate(records):
            line = json.dumps({"text": record["truncated"], "id": record["id"], "index": i}, ensure_ascii=False)
            f.write(line + "\n")


def _tokenize(strings: list[str], tokenizer: PreTrainedTokenizer, max_length: int) -> tuple[BatchEncoding, list[str]]:
    """Tokenize a list of strings using a HuggingFace tokenizer."""
    strings = [x.strip()[:10000] for x in strings]  # Hard limit to 10k chars
    tokenized = tokenizer(
        strings,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offset_mapping = tokenized.pop("offset_mapping")

    # The offset mapping is a 3D array (batch, max_length_in_batch, 2), with offsets
    # Where the final dimension is start, end indices. So by taking the max end index
    # we know the length to which the tokenizer tokenized the string.
    lengths = np.asarray(offset_mapping)[:, :, 1].max(axis=1)
    return tokenized, [string[:length] for string, length in zip(strings, lengths)]


def infer(
    model: SentenceTransformer,
    records: Iterator[dict[str, str]],
    name: str,
    batch_size: int = 96,
    max_length: int = 512,
    save_every: int = 8192,
    prompt: str | None = None,
    limit_batches: int | None = None,
) -> None:
    """
    Infer embeddings for a stream of texts using a SentenceTransformer model.

    This is mainly used as an inner loop for the knowledge distillation process.
    We get N texts, and infer embeddings for them using the teacher model.
    These embeddings are then used to train the student model.

    We return the pooled output for each text.

    Args:
    ----
        model: The SentenceTransformer model to use for inference.
        records: A sequence of records to infer embeddings for.
        batch_size: The batch size to use for inference. Defaults to 96.
        max_length: The maximum sequence length for tokenization. Defaults to 512.
        save_every: Save intermediate results every N batches. Defaults to 8192.
        name: The name of the directory to save the results to.
        prompt: An optional prompt to prepend to each text before encoding.
        limit_batches: An optional limit on the number of batches to process.

    """
    model.eval()

    path = Path(name)
    path.mkdir(parents=True, exist_ok=True)
    shards_saved = 0

    all_pooled, accumulated_records = [], []
    tokenizer: PreTrainedTokenizer = model.tokenizer
    original_max_length = model[0].max_seq_length
    assert original_max_length is not None
    assert isinstance(original_max_length, int)
    if max_length > original_max_length:
        logger.warning(
            f"Warning: max_length {max_length} is greater than the model's max_length {original_max_length}. Not changing it."
        )
    else:
        model[0].max_seq_length = max_length  # type: ignore[assignment]

    if prompt is not None:
        prompt = prompt.strip()
        _, text_prompt = _tokenize([prompt] if prompt is not None else [""], tokenizer, max_length=512)
        text_prompt_length = len(text_prompt[0].strip()) + 1  # +1 for the space
    else:
        text_prompt_length = 0

    seen = 0
    with torch.inference_mode():
        for batch in _batchify(records, batch_size=batch_size):
            if prompt is None:
                texts = [record["text"] for record in batch]
            else:
                texts = [f"{prompt} {record['text']}" for record in batch]
            features, truncated_strings = _tokenize(texts, tokenizer, max_length)
            features_dict = {k: v.to(model.device) for k, v in features.items()}

            # One forward through the whole SentenceTransformer
            out = model(features_dict)
            pooled = out["sentence_embedding"].cpu()

            for record, truncated in zip(batch, truncated_strings):
                record["truncated"] = truncated[text_prompt_length:]

                accumulated_records.append(record)
            all_pooled.append(pooled.cpu())
            del pooled
            torch.cuda.empty_cache()

            seen += 1
            if seen % save_every == 0:
                logger.info(f"Seen {seen * batch_size} texts, saving intermediate results to disk.")
                _write_data(path, all_pooled, accumulated_records, shards_saved)
                shards_saved += 1
                all_pooled = []
                accumulated_records = []

            if limit_batches is not None and seen >= limit_batches:
                logger.info(f"Reached limit of {limit_batches} batches, stopping inference.")
                break

    if accumulated_records:
        _write_data(path, all_pooled, accumulated_records, shards_saved)
