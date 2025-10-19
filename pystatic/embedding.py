from __future__ import annotations

import os
from typing import Any, TypeVar

import numpy as np
import torch
from safetensors.torch import save_file as save_safetensors_file
from sentence_transformers.models import Module, StaticEmbedding
from tokenizers import Tokenizer
from torch import nn
from transformers import PreTrainedTokenizerFast

T = TypeVar("T", bound=Module)


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-12) -> None:
        """Layer normalization layer."""
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, features: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forward pass."""
        features["sentence_embedding"] = self.layer_norm(features["sentence_embedding"])
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
        """Load a LayerNorm from a pretrained model."""
        hub_kwargs = {
            "subfolder": subfolder,
            "token": token,
            "cache_folder": cache_folder,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        weights = cls.load_torch_weights(model_name_or_path=model_name_or_path, **hub_kwargs)  # type: ignore
        initialized = cls(dim=weights["layer_norm.weight"].shape[0])
        return cls.load_torch_weights(model=initialized, model_name_or_path=model_name_or_path, **hub_kwargs)  # type: ignore

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs: Any) -> None:
        """Save the LayerNorm to disk."""
        if safe_serialization:
            save_safetensors_file(self.state_dict(), os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))


class TrainableStaticEmbedding(StaticEmbedding):
    def tokenize(self, texts: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        """Tokenize the texts."""
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=False)
        encodings_ids = [torch.Tensor(encoding.ids[:512]).long() for encoding in encodings]

        input_ids = torch.nn.utils.rnn.pad_sequence(encodings_ids, batch_first=True, padding_value=0)
        return {"input_ids": input_ids}

    def forward(self, features: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forward pass."""
        x = self.embedding(features["input_ids"])
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


class TrainableStaticEmbeddingWithW(TrainableStaticEmbedding):
    def __init__(
        self,
        tokenizer: Tokenizer | PreTrainedTokenizerFast,
        embedding_weights: np.ndarray | torch.Tensor | None = None,
        embedding_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Static embedding layer with access to weights."""
        super().__init__(tokenizer, embedding_weights, embedding_dim, **kwargs)
        n_tokens = self.embedding.weight.shape[0]
        self.w = nn.Parameter(torch.zeros(n_tokens, dtype=torch.float32))
        self.embedding.mode = "sum"

    def forward(self, features: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forward pass."""
        input_ids = features["input_ids"]
        w = torch.sigmoid(self.w[input_ids])
        x = self.embedding(input_ids, per_sample_weights=w)
        features["sentence_embedding"] = x
        return features
