import random
from argparse import ArgumentParser, Namespace
from collections import Counter
from pathlib import Path
from typing import Sequence, cast

from datasets import Dataset, IterableDataset, concatenate_datasets, load_dataset
from skeletoken import TokenizerModel
from skeletoken.preprocessor import Preprocessor
from tqdm import tqdm


def _parse_args() -> Namespace:
    """Parse command line arguments for counting tokens."""
    parser = ArgumentParser(description="Count tokens in a text file.")
    parser.add_argument("--output", type=str, required=True, help="Output file for token counts.")
    parser.add_argument("--model-name", required=True, help="Name of the model to use.")
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Path to the datasets",
        nargs="+",
    )
    parser.add_argument("--in-memory", action="store_true", help="Load the dataset in memory.")

    return parser.parse_args()


def datasets(paths: Sequence[Path], in_memory: bool = True) -> IterableDataset:
    """Load the datasets."""
    if in_memory:
        datasets = []
        for path in paths:
            dataset = load_dataset(path=str(path), split="train")
            column_names = dataset.column_names
            assert column_names is not None
            columns_to_drop = [c for c in column_names if c not in ("text",)]
            dataset = dataset.remove_columns(columns_to_drop)
            datasets.append(dataset)
        return concatenate_datasets(datasets)

    all_shards = []
    for path in paths:
        all_shards.extend([str(x) for x in Path(path).glob("**/*.parquet")])
    random.shuffle(all_shards)
    dataset = load_dataset("parquet", data_files=all_shards, split="train", streaming=True)
    column_names = dataset.column_names
    assert column_names is not None
    columns_to_drop = [c for c in column_names if c not in ("text",)]
    dataset = dataset.remove_columns(columns_to_drop)

    return cast(IterableDataset, dataset)


if __name__ == "__main__":
    parsed_args = _parse_args()
    tokenizer_model = TokenizerModel.from_pretrained(parsed_args.model_name)
    preprocessor = Preprocessor.from_tokenizer_model(tokenizer_model)

    data = datasets([Path(p) for p in parsed_args.datasets], in_memory=parsed_args.in_memory)

    counts: Counter[str] = Counter()
    df: Counter[str] = Counter()
    for example in tqdm(data):
        text = example["text"]
        tokens = preprocessor.preprocess(text)
        counts.update(tokens)
        df.update(set(tokens))

    toks, token_counts = zip(*sorted(counts.items(), key=lambda x: x[1], reverse=True))
    token_dfs = [df[t] for t in toks]

    d = {"token": toks, "frequency": token_counts, "document_frequency": token_dfs}
    dataset = Dataset.from_dict(d)
    dataset.save_to_disk(parsed_args.output)
