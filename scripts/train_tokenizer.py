from argparse import ArgumentParser, Namespace
from pathlib import Path

from datasets import IterableDataset
from skeletoken import TokenizerModel
from tokenizers import Regex, Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD, Lowercase, Replace, StripAccents
from tokenizers.normalizers import Sequence as NormalizerSequence
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.pre_tokenizers import Sequence as PreTokenizerSequence
from tokenizers.trainers import WordPieceTrainer

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
    parser.add_argument("--vocab-size", type=int, default=30_000, help="Vocabulary size for the tokenizer.")

    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = _parse_args()
    data, _ = get_datasets(
        [Path(p) for p in parsed_args.datasets], in_memory=parsed_args.in_memory, limit_shards=parsed_args.limit_shards
    )

    model = WordPiece(
        vocab={},
        unk_token="[UNK]",
        max_input_chars_per_word=100,
    )
    tokenizer = Tokenizer(model)
    tokenizer.normalizer = NormalizerSequence([NFD(), Lowercase(), StripAccents(), Replace(Regex(r"[^\w\s]+"), "")])  # type: ignore
    tokenizer.pre_tokenizer = PreTokenizerSequence([BertPreTokenizer()])  # type: ignore

    trainer = WordPieceTrainer(
        vocab_size=parsed_args.vocab_size,
        special_tokens=["[UNK]", "[PAD]"],
        continuing_subword_prefix="##",
        end_of_word_suffix="",
    )
    if isinstance(data, IterableDataset):
        sentences = (x["sentence"] for x in data)
    else:
        sentences = data["sentence"]

    tokenizer.train_from_iterator(sentences, trainer=trainer)

    converted = TokenizerModel.from_tokenizer(tokenizer=tokenizer)
    transformers_tokenizer = converted.to_transformers()
    transformers_tokenizer.save_pretrained(parsed_args.output)
    transformers_tokenizer = converted.make_model_greedy().to_transformers()
    transformers_tokenizer.save_pretrained(f"{parsed_args.output}_greedy")
