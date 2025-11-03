from unittest.mock import patch

from datasets import Dataset

from pynife.dataset_vault import (
    english_words_definitions_dataset,
    fineweb_dataset,
    get_all_dataset_functions,
    gooaq_dataset,
    lotte_queries_dataset,
    miracl_dataset,
    mldr_dataset,
    mr_tydi_dataset,
    msmarco_docs_dataset,
    msmarco_queries_dataset,
    paws_dataset,
    pubmed_dataset,
    snli_dataset,
    squad_dataset,
    swim_ir_dataset,
    triviaqa_dataset,
)


def _ds_english_words_definitions() -> Dataset:
    """Factory for english-words-definitions mock dataset."""
    return Dataset.from_dict({"definitions": [["mocked definition 1"], ["mocked definition 2"]]})


def _ds_gooaq() -> Dataset:
    """Factory for gooaq mock dataset."""
    return Dataset.from_dict({"question": ["mocked question 1", "mocked question 2"]})


def _ds_miracl() -> Dataset:
    """Factory for miracl mock dataset."""
    return Dataset.from_dict({"anchor": ["mocked anchor 1", "mocked anchor 2"]})


def _ds_snli() -> Dataset:
    """Factory for SNLI mock dataset."""
    return Dataset.from_dict(
        {
            "premise": ["mocked premise 1", "mocked premise 2"],
            "hypothesis": ["mocked hypothesis 1", "mocked hypothesis 2"],
        }
    )


def _ds_simple_text() -> Dataset:
    """Generic single-column text dataset used by several dataset helpers."""
    return Dataset.from_dict({"text": ["mocked text 1", "mocked text 2"]})


def _ds_paws() -> Dataset:
    """Factory for PAWS mock dataset with sentence1/sentence2 columns."""
    return Dataset.from_dict(
        {
            "sentence1": ["mocked sentence 1", "mocked sentence 2"],
            "sentence2": ["mocked sentence 3", "mocked sentence 4"],
        }
    )


def _ds_mldr() -> Dataset:
    """Factory for MLDR mock dataset (uses 'anchor')."""
    return Dataset.from_dict({"anchor": ["mocked mldr anchor 1", "mocked mldr anchor 2"]})


def _ds_pubmed() -> Dataset:
    """Factory for PubMedQA mock dataset (returns a 'question' column that is renamed)."""
    return Dataset.from_dict({"question": ["mocked abstract 1", "mocked abstract 2"]})


def _ds_swim_ir() -> Dataset:
    """Factory for SWIM-IR mock dataset (returns 'query')."""
    return Dataset.from_dict({"query": ["mocked swim query 1", "mocked swim query 2"]})


def _ds_triviaqa() -> Dataset:
    """Factory for TriviaQA mock dataset (returns 'question')."""
    return Dataset.from_dict({"question": ["mocked trivia q1", "mocked trivia q2"]})


def _ds_mr_tydi() -> Dataset:
    """Factory for Mr. TyDi mock dataset (uses 'anchor')."""
    return Dataset.from_dict({"anchor": ["mocked mr-tydi anchor 1", "mocked mr-tydi anchor 2"]})


def mock_load_dataset(*args: str, **kwargs: str) -> Dataset:
    """
    Mock the load_dataset function to return predefined data.

    This dispatcher delegates to small factory functions to keep complexity low.
    """
    hf_name = args[0]

    # MS MARCO needs special handling because callers pass a config (query vs passage)
    if "msmarco-corpus" in hf_name or "msmarco" in hf_name:
        config_arg = args[1] if len(args) > 1 else kwargs.get("config")
        if config_arg == "query":
            return Dataset.from_dict({"text": ["mocked query 1", "mocked query 2"]})
        return Dataset.from_dict({"text": ["mocked doc 1", "mocked doc 2"]})

    # simple substring-based dispatch to small factories
    dispatch = {
        "english-words-definitions": _ds_english_words_definitions,
        "MongoDB/english-words-definitions": _ds_english_words_definitions,
        "sentence-transformers/gooaq": _ds_gooaq,
        "gooaq": _ds_gooaq,
        "sentence-transformers/miracl": _ds_miracl,
        "miracl": _ds_miracl,
        "stanfordnlp/snli": _ds_snli,
        "snli": _ds_snli,
        "mteb/lotte": _ds_simple_text,
        "lotte": _ds_simple_text,
        "google-research-datasets/paws": _ds_paws,
        "paws": _ds_paws,
        "sentence-transformers/squad": _ds_gooaq,
        "squad": _ds_gooaq,
        "sentence-transformers/mldr": _ds_mldr,
        "mldr": _ds_mldr,
        "qiaojin/PubMedQA": _ds_pubmed,
        "PubMedQA": _ds_pubmed,
        "nthakur/swim-ir-monolingual": _ds_swim_ir,
        "swim-ir": _ds_swim_ir,
        "mandarjoshi/trivia_qa": _ds_triviaqa,
        "trivia_qa": _ds_triviaqa,
        "sentence-transformers/mr-tydi": _ds_mr_tydi,
        "mr-tydi": _ds_mr_tydi,
        "msmarco-corpus": _ds_simple_text,
    }

    for key, factory in dispatch.items():
        if key in hf_name:
            return factory()

    # fallback
    return _ds_simple_text()


@patch("pynife.dataset_vault.load_dataset", side_effect=mock_load_dataset)
def test_english_words_definitions_dataset(mock_load) -> None:
    """Test the English words definitions dataset."""
    name, hf, dataset = english_words_definitions_dataset()
    assert hf == "MongoDB/english-words-definitions"
    assert name == "english-words-definitions"
    assert list(dataset) == [
        {"definitions": ["mocked definition 1"], "text": "mocked definition 1"},
        {"definitions": ["mocked definition 2"], "text": "mocked definition 2"},
    ]


@patch("pynife.dataset_vault.load_dataset", side_effect=mock_load_dataset)
def test_fineweb_dataset(mock_load) -> None:
    """Test the FineWeb dataset."""
    name, hf, dataset = fineweb_dataset()
    assert hf == "HuggingFaceFW/fineweb"
    assert name == "fineweb"
    assert list(dataset) == [{"text": "mocked text 1"}, {"text": "mocked text 2"}]


@patch("pynife.dataset_vault.load_dataset", side_effect=mock_load_dataset)
def test_gooaq_dataset(mock_load) -> None:
    """Test the GooAQ dataset."""
    name, hf, dataset = gooaq_dataset()
    assert hf == "sentence-transformers/gooaq"
    assert name == "gooaq"
    assert list(dataset) == [{"text": "mocked question 1"}, {"text": "mocked question 2"}]


@patch("pynife.dataset_vault.load_dataset", side_effect=mock_load_dataset)
def test_miracl_dataset(mock_load) -> None:
    """Test the MIRACL dataset."""
    name, hf, dataset = miracl_dataset()
    assert hf == "sentence-transformers/miracl"
    assert name == "miracl"
    assert list(dataset) == [{"text": "mocked anchor 1"}, {"text": "mocked anchor 2"}]


@patch("pynife.dataset_vault.load_dataset", side_effect=mock_load_dataset)
def test_lotte_queries_dataset(mock_load) -> None:
    """Test the LOTTE queries dataset."""
    name, hf, dataset = lotte_queries_dataset()
    assert hf == "mteb/lotte"
    assert name == "lotte"
    assert list(dataset)[:2] == [{"text": "mocked text 1"}, {"text": "mocked text 2"}]


@patch("pynife.dataset_vault.load_dataset", side_effect=mock_load_dataset)
def test_snli_dataset(mock_load) -> None:
    """Test the SNLI dataset."""
    name, hf, dataset = snli_dataset()
    assert hf == "stanfordnlp/snli"
    assert name == "snli"
    assert list(dataset) == [
        {"text": "mocked premise 1"},
        {"text": "mocked hypothesis 1"},
        {"text": "mocked premise 2"},
        {"text": "mocked hypothesis 2"},
    ]


@patch("pynife.dataset_vault.get_all_datasets", create=True)
def test_get_all_datasets(mock_get_all_datasets) -> None:
    """Test the get_all_datasets function."""
    mock_get_all_datasets.return_value = [
        "english_words_definitions",
        "gooaq",
        "miracl",
        "snli",
        "lotte",
        "paws",
        "squad",
        "mldr",
        "msmarco_queries",
        "msmarco_docs",
        "pubmed",
        "mr_tydi",
    ]
    datasets = mock_get_all_datasets()
    assert datasets == mock_get_all_datasets.return_value


@patch("pynife.dataset_vault.load_dataset", side_effect=mock_load_dataset)
def test_paws_dataset(mock_load) -> None:
    """Test `paws_dataset` returns sentence1 and sentence2 items as text entries."""
    name, hf, dataset = paws_dataset()
    assert hf == "google-research-datasets/paws"
    assert name == "paws"
    # PAWS appends sentence1 then sentence2 for each record
    assert list(dataset) == [
        {"text": "mocked sentence 1"},
        {"text": "mocked sentence 3"},
        {"text": "mocked sentence 2"},
        {"text": "mocked sentence 4"},
    ]


@patch("pynife.dataset_vault.load_dataset", side_effect=mock_load_dataset)
def test_squad_dataset(mock_load) -> None:
    """Test `squad_dataset` exposes the `question` field as text entries."""
    name, hf, dataset = squad_dataset()
    assert hf == "sentence-transformers/squad"
    assert name == "squad"
    assert list(dataset) == [{"text": "mocked question 1"}, {"text": "mocked question 2"}]


@patch("pynife.dataset_vault.load_dataset", side_effect=mock_load_dataset)
def test_mldr_dataset(mock_load) -> None:
    """Test `mldr_dataset` returns anchor fields as text entries via the helper."""
    name, hf, dataset = mldr_dataset()
    assert hf == "sentence-transformers/mldr"
    assert name == "mldr"
    assert list(dataset) == [{"text": "mocked mldr anchor 1"}, {"text": "mocked mldr anchor 2"}]


@patch("pynife.dataset_vault.load_dataset", side_effect=mock_load_dataset)
def test_msmarco_and_docs_datasets(mock_load) -> None:
    """Test MS MARCO query and document dataset helpers return expected text entries."""
    nq_name, nq_hf, nq_dataset = msmarco_queries_dataset()
    assert nq_hf == "sentence-transformers/msmarco-corpus"
    assert nq_name == "msmarco_queries"
    assert list(nq_dataset) == [{"text": "mocked query 1"}, {"text": "mocked query 2"}]

    nd_name, nd_hf, nd_dataset = msmarco_docs_dataset()
    assert nd_hf == "sentence-transformers/msmarco-corpus"
    assert nd_name == "msmarco_docs"
    assert list(nd_dataset) == [{"text": "mocked doc 1"}, {"text": "mocked doc 2"}]


@patch("pynife.dataset_vault.load_dataset", side_effect=mock_load_dataset)
def test_pubmed_and_swim_trivia_mr_tydi(mock_load) -> None:
    """
    Test PubMedQA, SWIM-IR, TriviaQA and Mr. TyDi dataset helpers.

    This ensures column renaming and `_simple_text_field_dataset` usage behave as expected.
    """
    p_name, p_hf, p_dataset = pubmed_dataset()
    assert p_hf == "qiaojin/PubMedQA"
    assert p_name == "PubMedQA"
    # pubmed_dataset renames 'question' to 'text'
    assert list(p_dataset)[:2] == [{"text": "mocked abstract 1"}, {"text": "mocked abstract 2"}]

    s_name, s_hf, s_dataset = swim_ir_dataset()
    assert s_hf == "sentence-transformers/swim-ir"
    assert s_name == "swim-ir"
    assert list(s_dataset) == [{"text": "mocked swim query 1"}, {"text": "mocked swim query 2"}]
    t_name, t_hf, t_dataset = triviaqa_dataset()
    assert t_hf == "mandarjoshi/trivia_qa"
    assert t_name == "trivia_qa"
    assert list(t_dataset) == [{"text": "mocked trivia q1"}, {"text": "mocked trivia q2"}]

    m_name, m_hf, m_dataset = mr_tydi_dataset()
    assert m_hf == "sentence-transformers/mr-tydi"
    assert m_name == "mr-tydi"
    assert list(m_dataset) == [{"text": "mocked mr-tydi anchor 1"}, {"text": "mocked mr-tydi anchor 2"}]


def test_get_all_dataset_functions() -> None:
    """Sanity-check the dictionary returned by `get_all_dataset_functions` contains expected keys."""
    funcs = get_all_dataset_functions()
    # sanity check for presence of a few known keys
    for key in [
        "english-words-definitions",
        "gooaq",
        "snli",
        "paws",
        "msmarco",
        "PubMedQA",
    ]:
        assert key in funcs
