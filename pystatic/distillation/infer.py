import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar, cast

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


def _write_data(
    path: Path, pooled: list[torch.Tensor], means: list[torch.Tensor], records: list[dict[str, str]], shard_index: int
) -> None:
    """Write out the data to disk."""
    pooled_tensor = torch.cat(pooled, dim=0).half()
    mean_tensor = torch.stack(means, dim=0).half()
    torch.save(pooled_tensor, path / f"pooled_{shard_index:04d}.pt")
    torch.save(mean_tensor, path / f"mean_{shard_index:04d}.pt")
    with open(path / f"texts_{shard_index:04d}.txt", "w", encoding="utf-8") as f:
        for i, record in enumerate(records):
            line = json.dumps({"text": record["truncated"], "id": record["id"], "index": i}, ensure_ascii=False)
            f.write(line + "\n")


def _tokenize(strings: list[str], tokenizer: PreTrainedTokenizer, max_length: int) -> tuple[BatchEncoding, list[str]]:
    """Tokenize a list of strings using a HuggingFace tokenizer."""
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
) -> None:
    """
    Infer embeddings for a list of texts using a SentenceTransformer model.

    This is mainly used as an inner loop for the knowledge distillation process.
    We sample N texts from the corpus, and infer embeddings for them using
    the teacher model. These embeddings are then used to train the student model.

    We return both the pooled output (CLS token) and the mean-pooled output
    (mean of all token embeddings) for each text. This allows models to train on
    either objective, or switch between them as a form of regularization.

    Args:
    ----
        model: The SentenceTransformer model to use for inference.
        records: A sequence of records to infer embeddings for.
        batch_size: The batch size to use for inference. Defaults to 96.
        max_length: The maximum sequence length for tokenization. Defaults to 512.
        save_every: Save intermediate results every N batches. Defaults to 8192.
        name: The name of the directory to save the results to.
        prompt: An optional prompt to prepend to each text before encoding.

    """
    model.eval()

    path = Path(name)
    path.mkdir(parents=True, exist_ok=True)
    shards_saved = 0

    all_pooled, all_mean, accumulated_records = [], [], []
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
        tokenized_prompt, text_prompt = _tokenize([prompt] if prompt is not None else [""], tokenizer, max_length=512)
        input_ids = cast(torch.Tensor, tokenized_prompt["input_ids"]).tolist()
        prompt_length = cast(int, len(input_ids))
        text_prompt_length = len(text_prompt[0]) + 1  # 1 extra for the space
    else:
        prompt_length = 0
        text_prompt_length = 0

    seen = 0
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
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
            tokens = out["token_embeddings"].cpu()
            masks = out["attention_mask"].cpu()

            for record, token_sequence, mask, truncated in zip(batch, tokens, masks, truncated_strings):
                first_zero_index = mask.argmin() - 1

                meaned = token_sequence[1 + prompt_length : first_zero_index]
                if len(meaned) == 0:  # happens when the input is empty or only special tokens
                    dim = model.get_sentence_embedding_dimension()
                    assert dim is not None
                    meaned = torch.zeros(dim, device="cpu")
                else:
                    meaned = meaned.mean(dim=0).cpu()

                all_mean.append(meaned)
                record["truncated"] = truncated[text_prompt_length:]

                accumulated_records.append(record)
            all_pooled.append(pooled.cpu())
            del tokens
            del pooled
            torch.cuda.empty_cache()

            seen += 1
            if seen % save_every == 0:
                logger.info(f"Seen {seen * batch_size} texts, saving intermediate results to disk.")
                _write_data(path, all_pooled, all_mean, accumulated_records, shards_saved)
                shards_saved += 1
                all_pooled = []
                all_mean = []
                accumulated_records = []

    if accumulated_records:
        _write_data(path, all_pooled, all_mean, accumulated_records, shards_saved)
