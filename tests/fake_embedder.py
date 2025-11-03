from typing import Any, TypeVar, cast

import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from pynife.embedding import TrainableStaticEmbedding

T = TypeVar("T", bound=TrainableStaticEmbedding)


class TestableStaticEmbedding(TrainableStaticEmbedding):
    """A Testable version of TrainableStaticEmbedding to expose internal states for testing."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        embedding_weights: Any = None,
        embedding_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the TestableStaticEmbedding."""
        super().__init__(tokenizer, embedding_weights, embedding_dim, **kwargs)  # type: ignore
        if isinstance(tokenizer, Tokenizer):
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        else:
            self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = cast(
                PreTrainedTokenizer | PreTrainedTokenizerFast, tokenizer
            )  # type: ignore

    def tokenize(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        """Expose the tokenize method for testing."""
        tokenized = self.tokenizer(texts, **kwargs)
        input_ids = torch.cat(tokenized["input_ids"])  # type: ignore
        offsets = [0]
        for x in tokenized["input_ids"]:  # type: ignore
            offsets.append(offsets[-1] + len(x))
        return {"input_ids": input_ids, "offsets": offsets}

    @classmethod
    def load(cls: type[T], model_name_or_path: str, *args: Any, **kwargs: Any) -> T:
        """Expose the load method for testing."""
        instance = super().load(model_name_or_path, *args, **kwargs)
        instance.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        return instance
