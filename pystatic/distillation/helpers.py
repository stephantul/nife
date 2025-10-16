from argparse import ArgumentParser, Namespace

from sentence_transformers import SentenceTransformer


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

    return parser.parse_args()


def get_prompt_from_model(model: SentenceTransformer, prompt_name: str | None) -> str | None:
    """Get the prompt from the model based on the prompt name."""
    if prompt_name is None:
        return None

    prompt = model.prompts.get(prompt_name, None)
    if prompt is None:
        raise ValueError(f"Prompt '{prompt_name}' not found in model prompts.")

    return prompt
