from __future__ import annotations

import os
from typing import Any, TypeVar, cast

import numpy as np
import torch
from safetensors.torch import save_file as save_safetensors_file
from sentence_transformers.models import Module, StaticEmbedding
from tokenizers import Tokenizer
from torch import nn
from transformers import PreTrainedTokenizerFast

T = TypeVar("T", bound=Module)


def _load_weights_for_model(
    model: T,
    model_name_or_path: str,
    subfolder: str = "",
    token: bool | str | None = None,
    cache_folder: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> T:
    """Load weights for a given model from a pretrained model."""
    weights = Module.load_torch_weights(
        model=model,
        model_name_or_path=model_name_or_path,
        subfolder=subfolder,
        token=token,
        cache_folder=cache_folder,
        revision=revision,
        local_files_only=local_files_only,
    )
    return cast(T, weights)


def _load_weights_as_tensor_dict(
    model_name_or_path: str,
    subfolder: str = "",
    token: bool | str | None = None,
    cache_folder: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> dict[str, torch.Tensor]:
    """Load embedding weights from a pretrained model as a tensor."""
    weights = Module.load_torch_weights(
        model_name_or_path=model_name_or_path,
        subfolder=subfolder,
        token=token,
        cache_folder=cache_folder,
        revision=revision,
        local_files_only=local_files_only,
    )
    return cast(dict[str, torch.Tensor], weights)


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
        weights = _load_weights_as_tensor_dict(
            model_name_or_path=model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        initialized = cls(dim=weights["layer_norm.weight"].shape[0])
        return _load_weights_for_model(
            initialized,
            model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs: Any) -> None:
        """Save the LayerNorm to disk."""
        if safe_serialization:
            save_safetensors_file(self.state_dict(), os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))


class TrainableStaticEmbedding(StaticEmbedding):
    def __init__(
        self,
        tokenizer: Tokenizer | PreTrainedTokenizerFast,
        embedding_weights: np.ndarray | torch.Tensor | None = None,
        embedding_dim: int | None = None,
        scale_grad_by_freq: bool = False,
        **kwargs: Any,
    ) -> None:
        """Static embedding layer."""
        super().__init__(tokenizer, embedding_weights, embedding_dim, **kwargs)
        self.embedding.scale_grad_by_freq = scale_grad_by_freq

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
    def load(  # type: ignore[misc]
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
        tokenizer_path = cls.load_file_path(
            model_name_or_path,
            filename="tokenizer.json",
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        assert tokenizer_path is not None
        tokenizer = Tokenizer.from_file(tokenizer_path)

        weights_dict = _load_weights_as_tensor_dict(
            model_name_or_path=model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        try:
            weights = weights_dict["embedding.weight"]
        except KeyError:
            # For compatibility with model2vec models, which are saved with just an "embeddings" key
            weights = weights_dict["embeddings"]
        initialized = cls(dim=weights.shape[1], tokenizer=tokenizer, embedding_weights=weights, **kwargs)
        return _load_weights_for_model(
            initialized,
            model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
