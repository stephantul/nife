from typing import Any

import numpy as np
import torch
from sentence_transformers.models import StaticEmbedding
from tokenizers import Tokenizer
from torch import nn
from transformers import PreTrainedTokenizerFast


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
        # self.w = nn.Embedding.from_pretrained(torch.zeros(self.tokenizer.get_vocab_size(), 1), freeze=False)
        self.normalizer = nn.LayerNorm(self.embedding_dim)

    def tokenize(self, texts: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        """Tokenize the texts."""
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=False)
        encodings_ids = [torch.Tensor(encoding.ids[:512]).long() for encoding in encodings]

        input_ids = torch.nn.utils.rnn.pad_sequence(encodings_ids, batch_first=True, padding_value=0)
        return {"input_ids": input_ids}

    def forward(self, features: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forward pass."""
        x = self.embedding(features["input_ids"])
        x = self.normalizer(x)
        features["sentence_embedding"] = x
        return features

    def collapse(self) -> StaticEmbedding:
        """Collapse to a non-trainable StaticEmbedding."""
        emb_weights = self.embedding.weight.data.clone()
        # emb_weights = emb_weights * torch.sigmoid(self.w.weight)
        emb_weights = self.normalizer(emb_weights)
        return StaticEmbedding(
            tokenizer=self.tokenizer, embedding_weights=emb_weights, embedding_dim=self.embedding_dim
        )
