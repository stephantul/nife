from pathlib import Path

import pytest
from sentence_transformers import SentenceTransformer

from nife import nife


class DummyModelCard:
    def __init__(self, data: object) -> None:
        """Dummy model card."""
        self.data = data


class DummyData:
    def __init__(self, base_model: object | None = None) -> None:
        """Dummy model data."""
        self.base_model = base_model


class DummyModel:
    def __init__(self, dim: int) -> None:
        """Dummy SentenceTransformer-like model."""
        self._dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dim


class DummyRouter:
    def __init__(self, q: SentenceTransformer, d: SentenceTransformer) -> None:
        """Dummy Router-like object."""
        self.q = q
        self.d = d


def test_get_teacher_from_metadata_local(tmp_path, monkeypatch) -> None:
    """Ensure that when a local model path contains a README, the function extracts the `base_model` value from the ModelCard."""
    # Create a fake README.md and ensure ModelCard.load reads it
    readme = tmp_path / "README.md"
    readme.write_text("base_model: test-teacher\n")

    # Monkeypatch ModelCard.load to return an object with data.base_model
    def fake_load(path: Path | str) -> DummyModelCard:
        assert str(path) == str(readme)
        return DummyModelCard(DummyData(base_model="teacher-model"))

    monkeypatch.setattr(nife.ModelCard, "load", fake_load)

    teacher = nife._get_teacher_from_metadata(tmp_path)
    assert teacher == "teacher-model"


def test_get_teacher_from_metadata_remote_hf_success(monkeypatch, tmp_path) -> None:
    """When given a remote repo id, the function should call the HF API to download the README and extract `base_model` from it."""
    # Simulate hf_hub_download returning a README path
    fake_readme = tmp_path / "remote_README.md"
    fake_readme.write_text("dummy")

    def fake_download(repo_id: str, filename: str) -> str:
        assert repo_id == "owner/repo"
        assert filename == "README.md"
        return str(fake_readme)

    monkeypatch.setattr(nife, "HfApi", lambda: type("A", (), {"hf_hub_download": staticmethod(fake_download)})())

    monkeypatch.setattr(nife.ModelCard, "load", lambda p: DummyModelCard(DummyData(base_model="teacher-remote")))

    teacher = nife._get_teacher_from_metadata("owner/repo")
    assert teacher == "teacher-remote"


def test_get_teacher_from_metadata_remote_hf_not_found(monkeypatch) -> None:
    """If the HF API fails to download a README, the function should raise FileNotFoundError."""

    def fake_download(repo_id: str, filename: str) -> str:
        raise Exception("not found")

    monkeypatch.setattr(nife, "HfApi", lambda: type("A", (), {"hf_hub_download": staticmethod(fake_download)})())

    with pytest.raises(FileNotFoundError):
        nife._get_teacher_from_metadata("owner/nonexistent")


def test_get_teacher_from_metadata_missing_base_model(monkeypatch, tmp_path) -> None:
    """If the README/ModelCard contains no `base_model` field, the function should raise ValueError."""
    # Local README but ModelCard has no base_model
    readme = tmp_path / "README.md"
    readme.write_text("nothing")

    monkeypatch.setattr(nife.ModelCard, "load", lambda p: DummyModelCard(DummyData(base_model=None)))

    with pytest.raises(ValueError):
        nife._get_teacher_from_metadata(tmp_path)


def test_load_nife_success(monkeypatch, test_model) -> None:
    """Load a student and teacher with matching embedding dimensions and verify the composed SentenceTransformer is returned."""
    # Use the provided test_model as the small student model.
    # Monkeypatch _get_teacher_from_metadata to return a fake teacher name.
    monkeypatch.setattr(nife, "_get_teacher_from_metadata", lambda name: "teacher-name")

    # Provide a fake teacher SentenceTransformer object with matching dimension
    class TeacherLike:
        def get_sentence_embedding_dimension(self) -> int:
            return test_model.get_sentence_embedding_dimension()

    teacher = TeacherLike()

    # Patch SentenceTransformer to return our teacher or the existing test_model
    def sentence_transformer_loader(name_or_modules: object | None = None, *args: object, **kwargs: object) -> object:
        if kwargs.get("modules") is not None:
            # when the real function composes, return a sentinel
            return "composed-model"
        if name_or_modules == "teacher-name":
            return teacher
        if name_or_modules == "small-model":
            return test_model
        raise ValueError("unexpected model name")

    monkeypatch.setattr(nife, "SentenceTransformer", sentence_transformer_loader)

    # Patch Router.for_query_document to return a DummyRouter
    monkeypatch.setattr(
        nife.Router,
        "for_query_document",
        staticmethod(lambda query_modules, document_modules: DummyRouter(query_modules, document_modules)),
    )

    model = nife.load_nife("small-model")
    assert model == "composed-model"


def test_load_nife_dimensionality_mismatch(monkeypatch, test_model) -> None:
    """If teacher and student embedding dimensionalities differ, load_nife should raise a ValueError."""
    # teacher has different dimension than the test_model student
    monkeypatch.setattr(nife, "_get_teacher_from_metadata", lambda name: "teacher-name")

    class BadTeacher:
        def get_sentence_embedding_dimension(self) -> int:
            return test_model.get_sentence_embedding_dimension() + 1

    monkeypatch.setattr(
        nife,
        "SentenceTransformer",
        lambda name_or_modules=None, *a, **k: BadTeacher() if name_or_modules == "teacher-name" else test_model,
    )

    with pytest.raises(ValueError):
        nife.load_nife("small-model")
