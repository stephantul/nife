from argparse import ArgumentParser, Namespace
from pathlib import Path

from datasets import IterableDataset, concatenate_datasets
from skeletoken import TokenizerModel
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.normalizers import Sequence as NormalizerSequence
from tokenizers.pre_tokenizers import BertPreTokenizer, Metaspace
from tokenizers.pre_tokenizers import Sequence as PreTokenizerSequence
from tokenizers.trainers import BpeTrainer

from pystatic.data import get_datasets


def _parse_args() -> Namespace:
    """Parse command line arguments for training a tokenizer."""
    parser = ArgumentParser(description="Train a tokenizer.")
    parser.add_argument("--output", type=str, required=True, help="Output path for the trained tokenizer.")
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Path to the datasets",
        nargs="+",
    )
    parser.add_argument("--in-memory", action="store_true", help="Load the dataset in memory.")
    parser.add_argument("--limit-shards", type=int, help="Limit the number of shards.")

    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = _parse_args()
    data, _ = get_datasets(
        [Path(p) for p in parsed_args.datasets], in_memory=parsed_args.in_memory, limit_shards=parsed_args.limit_shards
    )

    model = BPE(
        unk_token="[UNK]",
        continuing_subword_prefix="",
        end_of_word_suffix="",
        fuse_unk=True,
        byte_fallback=False,
        dropout=0.25,
    )
    tokenizer = Tokenizer(model)
    tokenizer.normalizer = NormalizerSequence([NFD(), Lowercase(), StripAccents()])  # type: ignore
    tokenizer.pre_tokenizer = PreTokenizerSequence([Metaspace(split=True), BertPreTokenizer()])  # type: ignore

    trainer = BpeTrainer(
        vocab_size=400_000,
        special_tokens=["[UNK]", "[PAD]"],
        continuing_subword_prefix="",
        end_of_word_suffix="",
        max_token_length=16,
    )

    if isinstance(data, IterableDataset):
        sentences = (x["sentence"] for x in data)
    else:
        concatenated = concatenate_datasets([data[k] for k in data])
        sentences = concatenated["sentence"]

    tokenizer.train_from_iterator(sentences, trainer=trainer)
    tokenizer.model.dropout = 0.0

    converted = TokenizerModel.from_tokenizer(tokenizer=tokenizer)
    transformers_tokenizer = converted.to_transformers()
    transformers_tokenizer.save_pretrained(parsed_args.output)
    transformers_tokenizer = converted.make_model_greedy().to_transformers()
    transformers_tokenizer.save_pretrained(f"{parsed_args.output}_greedy")
