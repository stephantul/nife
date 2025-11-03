from importlib.resources import files
from pathlib import Path


def get_model_card_template_path() -> Path:
    """Get the path to the model card template."""
    return Path(str(files("pynife.cards").joinpath("model_card.md")))
