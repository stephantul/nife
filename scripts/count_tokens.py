import concurrent.futures
from argparse import ArgumentParser, Namespace
from collections import Counter
from pathlib import Path

from datasets import Dataset
from skeletoken import TokenizerModel
from skeletoken.preprocessor import Preprocessor
from tqdm import tqdm

from nife.data import get_datasets


def _process_example(example) -> tuple[Counter[str], set[str]]:
    """Process a single example to count tokens."""
    text = example["text"]
    tokens = preprocessor.preprocess(text)
    return Counter(tokens), set(tokens)


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
    parser.add_argument("--text-column-name", type=str, default="text", help="Name of the text column.")
    parser.add_argument("--in-memory", action="store_true", help="Load the dataset in memory.")

    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = _parse_args()
    tokenizer_model = TokenizerModel.from_pretrained(parsed_args.model_name)
    preprocessor = Preprocessor.from_tokenizer_model(tokenizer_model)
    text_column_name: str = parsed_args.text_column_name
    in_memory = bool(parsed_args.in_memory)
    data, total = get_datasets(
        [Path(p) for p in parsed_args.datasets], in_memory=in_memory, columns_to_keep={text_column_name}
    )

    counts: Counter[str] = Counter()
    df: Counter[str] = Counter()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for item in tqdm(executor.map(_process_example, data), total=total):
            token_counter, token_set = item
            counts.update(token_counter)
            df.update(token_set)

    toks, token_counts = zip(*sorted(counts.items(), key=lambda x: x[1], reverse=True))
    token_dfs = [df[t] for t in toks]

    d = {"token": toks, "frequency": token_counts, "document_frequency": token_dfs}
    dataset = Dataset.from_dict(d)
    dataset.save_to_disk(parsed_args.output)
