from typing import Sequence

import torch
import torch.nn.functional as F
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
            loss = 1 - self.loss_fct(embeddings, labels.repeat(len(sentence_features), 1)).mean()
        else:
            embeddings = self.model(sentence_features[0])["sentence_embedding"]
            labels = labels[:, : embeddings.shape[1]]
            loss = 1 - self.loss_fct(embeddings, labels[:, : embeddings.shape[1]]).mean()
        return loss


class RelationalKLLoss(MSELoss):
    def __init__(self, model: SentenceTransformer) -> None:
        """Cosine loss."""
        super().__init__(model=model)

    def forward(self, sentence_features: Sequence[dict[str, torch.Tensor]], labels: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass."""
        embeddings = self.model(sentence_features[0])["sentence_embedding"]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        # Truncate for matryoshka
        labels = torch.nn.functional.normalize(labels[:, : embeddings.shape[1]], p=2, dim=1)

        tau = 0.07
        P = F.softmax((embeddings @ embeddings.T) / tau, dim=1)  # teacher similarity distribution
        Q = F.log_softmax((labels @ labels.T) / tau, dim=1)
        loss = F.kl_div(Q, P, reduction="batchmean") * (tau**2)

        return loss


class DistillationCosineLoss(MSELoss):
    def __init__(self, model: SentenceTransformer, alpha: float = 0.5) -> None:
        """Cosine loss."""
        super().__init__(model=model)
        self.alpha = alpha
        self.loss_fct = nn.CosineSimilarity()  # type: ignore

    def forward(self, sentence_features: Sequence[dict[str, torch.Tensor]], labels: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass."""
        embeddings = self.model(sentence_features[0])["sentence_embedding"]
        labels = labels[:, : embeddings.shape[1]]

        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        labels = torch.nn.functional.normalize(labels, p=2, dim=1)

        tau = 0.07
        sim = embeddings @ labels.T

        return F.cross_entropy(sim / tau, torch.arange(embeddings.size(0), device=embeddings.device))


class SelfMSELoss(MSELoss):
    def __init__(self, model: SentenceTransformer) -> None:
        """Self MSE loss."""
        super().__init__(model=model)
        self.loss_fct = nn.MSELoss()

    def forward(self, sentence_features: Sequence[dict[str, torch.Tensor]], labels: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass."""
        embeddings = self.model(sentence_features[0])["sentence_embedding"]
        labels = labels[:, : embeddings.shape[1]]
        loss = self.loss_fct(embeddings, labels)
        return loss


def select_loss(name: str) -> type[nn.Module]:
    """Select loss function by name."""
    if name == "cosine":
        return CosineLoss
    elif name == "mse":
        return SelfMSELoss
    elif name == "relational_kl":
        return RelationalKLLoss
    elif name == "distillation_cosine":
        return DistillationCosineLoss
    else:
        raise ValueError(f"Unknown loss function: {name}")
