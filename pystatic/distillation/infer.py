import logging
from collections.abc import Iterator, Sequence

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def _batchify(texts: Sequence[str], batch_size: int) -> Iterator[list[str]]:
    """Turn a list of texts into batches."""
    for i in range(0, len(texts), batch_size):
        yield list(texts[i : i + batch_size])


def infer(
    model: SentenceTransformer,
    texts: Sequence[str],
    batch_size: int = 512,
    max_length: int = 512,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
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
        batch_size: The batch size to use for inference. Defaults to 512.
        max_length: The maximum sequence length for tokenization. Defaults to 512.

    """
    model.eval()

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

    with torch.inference_mode():
        for batch in _batchify(texts, batch_size=batch_size):
            features = model.tokenize(batch)
            features = {k: v.to(model.device) for k, v in features.items()}

            # One forward through the whole SentenceTransformer
            out = model(features)
            pooled = out["sentence_embedding"]
            tokens = out["token_embeddings"]
            masks = out["attention_mask"]

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

    model.tokenizer.model_max_length = original_max_length

    return all_texts, torch.vstack(all_pooled), torch.vstack(all_mean)
