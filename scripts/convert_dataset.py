import argparse

from pystatic.data import build_parquet_shards_from_folder


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a folder of text files and embeddings to a dataset.")
    parser.add_argument("input", type=str, help="Input folder containing text files and embeddings.")
    parser.add_argument("output", type=str, help="Output file to save the dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples to process.")
    parser.add_argument("--rows-per-shard", type=int, default=200_000, help="Number of rows per output shard.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_parquet_shards_from_folder(
        args.input,
        args.output,
        limit=args.limit,
        rows_per_shard=args.rows_per_shard,
    )
