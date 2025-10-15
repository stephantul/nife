from typing import Sequence

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MSELoss
from torch import nn


class CosineLoss(MSELoss):
    def __init__(self, model: SentenceTransformer) -> None:
        """Cosine loss."""
        super().__init__(model=model)
        self.loss_fct = nn.CosineSimilarity()  # type: ignore

    def forward(self, sentence_features: Sequence[dict[str, torch.Tensor]], labels: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass."""
        # Concatenate multiple inputs on the batch dimension
        if len(sentence_features) > 1:
            embeddings = torch.cat([self.model(inputs)["sentence_embedding"] for inputs in sentence_features], dim=0)
            # Repeat the labels for each input
            return 1 - self.loss_fct(embeddings, labels.repeat(len(sentence_features), 1)).mean()

        embeddings = self.model(sentence_features[0])["sentence_embedding"]
        labels = labels[:, : embeddings.shape[1]]
        return 1 - self.loss_fct(embeddings, labels[:, : embeddings.shape[1]]).mean()
