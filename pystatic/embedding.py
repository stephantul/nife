from typing import Any, TypeVar

import numpy as np
import torch
from sentence_transformers.models import StaticEmbedding
from tokenizers import Tokenizer
from torch import nn
from transformers import PreTrainedTokenizerFast

T = TypeVar("T", bound="TrainableStaticEmbedding")


class TrainableStaticEmbedding(StaticEmbedding):
    def __init__(
        self,
        tokenizer: Tokenizer | PreTrainedTokenizerFast,
        embedding_weights: np.ndarray | torch.Tensor | None = None,
        embedding_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Static embedding layer with trainable weights."""
        super().__init__(
            tokenizer=tokenizer, embedding_dim=embedding_dim, embedding_weights=embedding_weights, **kwargs
        )
        self.normalizer = nn.LayerNorm(self.embedding_dim)
        n_tokens = self.tokenizer.get_vocab_size()
        self.w = torch.nn.Embedding(n_tokens, 1, padding_idx=0)
        self.embedding.mode = "sum"

    def tokenize(self, texts: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        """Tokenize the texts."""
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=False)
        encodings_ids = [torch.Tensor(encoding.ids[:512]).long() for encoding in encodings]

        input_ids = torch.nn.utils.rnn.pad_sequence(encodings_ids, batch_first=True, padding_value=0)
        return {"input_ids": input_ids}

    def forward(self, features: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forward pass."""
        weights = self.w(features["input_ids"])
        x = self.embedding(features["input_ids"], per_sample_weights=weights)
        x = self.normalizer(x)
        features["sentence_embedding"] = x
        return features

    @classmethod
    def load(
        cls: type[T],
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> T:
        """Load a TrainableStaticEmbedding from a pretrained model."""
        hub_kwargs = {
            "subfolder": subfolder,
            "token": token,
            "cache_folder": cache_folder,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        tokenizer_path = cls.load_file_path(model_name_or_path, filename="tokenizer.json", **hub_kwargs)  # type: ignore
        tokenizer = Tokenizer.from_file(tokenizer_path)

        weights = cls.load_torch_weights(model_name_or_path=model_name_or_path, **hub_kwargs)  # type: ignore
        try:
            weights = weights["embedding.weight"]  # type: ignore[misc]
        except KeyError:
            # For compatibility with model2vec models, which are saved with just an "embeddings" key
            weights = weights["embeddings"]  # type: ignore[misc]
        initialized = cls(tokenizer, embedding_weights=weights)
        return cls.load_torch_weights(model=initialized, model_name_or_path=model_name_or_path, **hub_kwargs)  # type: ignore
