import pytest
import torch
from sentence_transformers import SentenceTransformer

from pynife.losses import CosineLoss, DistillationCosineLoss, select_loss


def test_cosine_loss(test_model: SentenceTransformer) -> None:
    """Test the CosineLoss class."""
    loss_fn = CosineLoss(test_model)

    sentence_features = [{"input_ids": torch.randint(0, 100, (4, 10))}]
    labels = torch.rand(4, 768)

    loss = loss_fn(sentence_features, labels)
    assert loss.item() >= 0, "Loss should be non-negative"


def test_distillation_cosine_loss(test_model: SentenceTransformer) -> None:
    """Test the DistillationCosineLoss class."""
    loss_fn = DistillationCosineLoss(test_model, tau=0.1)

    sentence_features = [{"input_ids": torch.randint(0, 100, (4, 10))}]
    labels = torch.rand(4, 3)

    loss = loss_fn(sentence_features, labels)
    assert loss.item() >= 0, "Loss should be non-negative"


def test_select_loss() -> None:
    """Test the select_loss function."""
    assert select_loss("cosine") == CosineLoss
    assert select_loss("distillation_cosine") == DistillationCosineLoss

    with pytest.raises(ValueError):
        select_loss("unknown_loss")
