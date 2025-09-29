import json
import logging
from collections.abc import Iterator
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _batchify(texts: Iterator[str], batch_size: int) -> Iterator[list[str]]:
    """Turn a list of texts into batches."""
    batch: list[str] = []
    pbar = tqdm(total=0, unit="batches", desc="Creating batches")
    while True:
        if len(batch) < batch_size:
            try:
                batch.append(next(texts))
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
    path: Path,
    all_pooled: list[torch.Tensor],
    all_mean: list[torch.Tensor],
    all_texts: list[str],
    shard: int,
    index: int,
) -> None:
    """Write out the data to disk."""
    pooled_tensor = torch.cat(all_pooled, dim=0)
    mean_tensor = torch.stack(all_mean, dim=0)
    torch.save(pooled_tensor, path / f"pooled_{shard:04d}.pt")
    torch.save(mean_tensor, path / f"mean_{shard:04d}.pt")
    with open(path / f"texts_{shard:04d}.txt", "w", encoding="utf-8") as f:
        # Save the shards saved so far, and the index of the text in the shard
        # This allows us to easily match texts to embeddings later
        for i, text in enumerate(all_texts):
            line = json.dumps({"text": text, "index": index + i}, ensure_ascii=False)
            f.write(line + "\n")


def infer(
    model: SentenceTransformer,
    texts: Iterator[str],
    name: str,
    batch_size: int = 96,
    max_length: int = 512,
    save_every: int = 8192,
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
        texts: A sequence of texts to infer embeddings for.
        batch_size: The batch size to use for inference. Defaults to 96.
        max_length: The maximum sequence length for tokenization. Defaults to 512.
        save_every: Save intermediate results every N texts. Defaults to 8192.
        name: The name of the directory to save the results to.

    """
    model.eval()

    path = Path(name)
    path.mkdir(parents=True, exist_ok=True)
    shards_saved = 0

    all_pooled, all_mean, all_texts = [], [], []

    original_max_length = model[0].max_seq_length
    assert original_max_length is not None
    assert isinstance(original_max_length, int)
    if max_length >= original_max_length:
        logger.warning(
            f"Warning: max_length {max_length} is greater than the model's max_length {original_max_length}. Not changing it."
        )
    else:
        model[0].max_seq_length = max_length  # type: ignore[assignment]

    seen = 0
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for batch in _batchify(texts, batch_size=batch_size):
            features = model.tokenize(batch)
            features = {k: v.to(model.device) for k, v in features.items()}

            # One forward through the whole SentenceTransformer
            out = model(features)
            pooled = out["sentence_embedding"].cpu()
            tokens = out["token_embeddings"].cpu()
            masks = out["attention_mask"].cpu()

            for token_sequence, mask, input_ids in zip(tokens, masks, features["input_ids"]):
                first_zero_index = mask.argmin() - 1

                meaned = token_sequence[1:first_zero_index]
                if len(meaned) == 0:  # happens when the input is empty or only special tokens
                    dim = model.get_sentence_embedding_dimension()
                    assert dim is not None
                    meaned = torch.zeros(dim, device="cpu")
                else:
                    meaned = meaned.mean(dim=0).cpu()

                all_mean.append(meaned)
                all_texts.append(model.tokenizer.decode(input_ids, skip_special_tokens=True))

            all_pooled.append(pooled.cpu())
            del tokens
            del pooled
            torch.cuda.empty_cache()

            seen += 1
            if seen % save_every == 0:
                logger.info(f"Seen {seen * batch_size} texts, saving intermediate results to disk.")
                _write_data(path, all_pooled, all_mean, all_texts, shards_saved, (seen * batch_size) - len(all_texts))
                shards_saved += 1
                all_pooled = []
                all_mean = []
                all_texts = []

    if len(all_texts) > 0:
        _write_data(path, all_pooled, all_mean, all_texts, shards_saved, (seen * batch_size) - len(all_texts))
