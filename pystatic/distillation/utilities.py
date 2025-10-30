from argparse import ArgumentParser, Namespace


def parse_inference_args() -> Namespace:
    """Parse command line arguments for inference script."""
    parser = ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="mixedbread-ai/mxbai-embed-large-v1",
        help="Name of the pre-trained model to use for inference.",
    )
    parser.add_argument(
        "--prompt-name",
        type=str,
        default=None,
        help="Name of the prompt to use for inference. If this is None, no prompt is used.",
    )
    parser.add_argument("--max-length", type=int, default=512, help="Max length for inference.")
    parser.add_argument("--limit-batches", type=int, help="Limit the number of batches to process.")

    return parser.parse_args()
