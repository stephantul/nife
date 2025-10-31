from pathlib import Path

from huggingface_hub import HfApi, ModelCard
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Router


def _get_teacher_from_metadata(path: str | Path) -> str:
    """Gets metadata file for a given model from the Hugging Face Hub or a local path."""
    path = Path(path)
    if path.exists() and path.is_dir():
        readme_path = str(path / "README.md")
    else:
        api = HfApi()
        try:
            readme_path = api.hf_hub_download(repo_id=str(path), filename="README.md")
        except Exception as e:
            raise FileNotFoundError(f"Could not find README.md for model at {path}") from e

    model_card = ModelCard.load(readme_path)
    model_name: str | None = getattr(model_card.data, "base_model", None)
    if model_name is None:
        raise ValueError(f"Could not find 'base_model' in metadata for model at {path}")
    return model_name


def load_nife(name: str, teacher_name: str | None = None) -> SentenceTransformer:
    """
    Load a SentenceTransformer model from the Hugging Face Hub.

    Args:
        name: The name of the model to load.
        teacher_name: The name of the teacher model. If this is None, it will be inferred from the model's metadata.
            We recommend to leave this as None, because it is easy to get wrong.

    Returns:
        SentenceTransformer: The loaded model.

    Raises:
        ValueError: If the dimensionality of the teacher and student models do not match.

    """
    teacher_name = teacher_name or _get_teacher_from_metadata(name)
    big_model = SentenceTransformer(teacher_name)
    small_model = SentenceTransformer(name)

    # Ensure that both models have the same dimensionality.
    big_dim = big_model.get_sentence_embedding_dimension()
    small_dim = small_model.get_sentence_embedding_dimension()
    if big_dim != small_dim:
        raise ValueError(
            f"Dimensionality mismatch between teacher ({big_dim}) and student ({small_dim}). "
            "Please check that you have the correct teacher model."
        )

    router = Router.for_query_document(query_modules=[small_model], document_modules=[big_model])  # type: ignore
    return SentenceTransformer(modules=[router])
