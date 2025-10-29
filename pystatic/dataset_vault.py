from collections.abc import Callable, Iterable, Iterator
from typing import cast

from datasets import Dataset, concatenate_datasets, load_dataset


def _simple_text_field_dataset(
    huggingface_name: str,
    text_field: str,
    config: str | None = None,
    split: str = "train",
) -> Iterator[dict[str, str]]:
    """Helper function for datasets that extract a single text field."""
    dataset = cast(Dataset, load_dataset(huggingface_name, config, split=split))

    new_records: list[dict[str, str]] = []
    for record in cast(Iterable[dict[str, str]], dataset):
        text = record[text_field]
        new_records.append({"text": text})

    return cast(Iterator[dict[str, str]], iter(new_records))


def english_words_definitions_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the English Words Definitions dataset."""
    # return the full HF hub path so callers have the original identifier
    name = "MongoDB/english-words-definitions"
    dataset = cast(Dataset, load_dataset(name, split="train"))
    dataset = dataset.map(lambda x: {"text": " ".join(x["definitions"])})
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    dataset_iterator = cast(Iterator[dict[str, str]], iter(dataset))

    return name, dataset_iterator


def fineweb_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the FineWeb dataset (returns first path only)."""
    name = "HuggingFaceFW/fineweb"
    data = cast(
        Iterator[dict[str, str]],
        load_dataset(name, "sample-10BT", streaming=True, split="train"),
    )
    return name, iter(data)


def gooaq_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the GoOAQ dataset."""
    hf = "sentence-transformers/gooaq"
    return hf, _simple_text_field_dataset(hf, text_field="question")


def miracl_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the MIRACL dataset."""
    hf = "sentence-transformers/miracl"
    return hf, _simple_text_field_dataset(hf, text_field="anchor", config="en-triplet")


def lotte_queries_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the LoTTE queries dataset."""
    query_paths = [
        "lifestyle_forum-queries",
        "lifestyle_search-queries",
        "recreation_forum-queries",
        "recreation_search-queries",
        "science_forum-queries",
        "science_search-queries",
        "technology_forum-queries",
        "technology_search-queries",
        "writing_forum-queries",
        "writing_search-queries",
    ]

    big_dataset = []
    for path in query_paths:
        dataset = cast(Dataset, load_dataset("mteb/lotte", path, split="dev"))
        big_dataset.append(dataset)
    final_dataset: Dataset = concatenate_datasets(big_dataset)
    dataset_iterator = cast(Iterator[dict[str, str]], iter(final_dataset))
    return "mteb/lotte", dataset_iterator


def snli_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the SNLI dataset."""
    name = "stanfordnlp/snli"
    dataset = cast(Dataset, load_dataset(name, split="train"))

    seen = set()
    new_records: list[dict[str, str]] = []
    for record in cast(Iterable[dict[str, str]], dataset):
        text = record["premise"]
        if text not in seen:
            seen.add(text)
            new_records.append({"text": text})
        text = record["hypothesis"]
        if text not in seen:
            seen.add(text)
            new_records.append({"text": text})

    dataset_iterator = cast(Iterator[dict[str, str]], iter(new_records))
    return name, dataset_iterator


def paws_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the PAWS dataset."""
    name = "google-research-datasets/paws"
    dataset = load_dataset(name, "unlabeled_final", split="train")

    new_records = []
    for record in cast(Iterable[dict[str, str]], dataset):
        text = record["sentence1"]
        new_records.append({"text": text})
        text = record["sentence2"]
        new_records.append({"text": text})

    dataset_iterator = cast(Iterator[dict[str, str]], iter(new_records))
    return name, dataset_iterator


def squad_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the SQuAD dataset."""
    hf = "sentence-transformers/squad"
    return hf, _simple_text_field_dataset(hf, text_field="question")


def mldr_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the MLDR dataset."""
    hf = "sentence-transformers/mldr"
    return hf, _simple_text_field_dataset(hf, text_field="anchor", config="en-triplet")


def msmarco_queries_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the MS MARCO queries dataset."""
    hf = "sentence-transformers/msmarco-corpus"
    return hf, _simple_text_field_dataset(hf, text_field="text", config="query")


def msmarco_docs_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the MS MARCO documents dataset."""
    hf = "sentence-transformers/msmarco-corpus"
    return hf, _simple_text_field_dataset(hf, text_field="text", config="passage")


def pubmed_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the PubMedQA dataset."""
    name = "qiaojin/PubMedQA"
    subsets = ["pqa_artificial", "pqa_unlabeled"]
    datasets = []
    for subset in subsets:
        dataset = cast(Dataset, load_dataset(name, subset, split="train"))
        dataset = dataset.rename_columns({"question": "text"})
        columns_to_remove = [col for col in dataset.column_names if col not in ("text",)]
        dataset = dataset.remove_columns(columns_to_remove)
        datasets.append(dataset)

    final_dataset = concatenate_datasets(datasets)
    dataset_iterator = cast(Iterator[dict[str, str]], iter(final_dataset))
    return name, dataset_iterator


def swim_ir_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the SWIM-IR dataset."""
    hf = "nthakur/swim-ir-monolingual"
    return hf, _simple_text_field_dataset(hf, text_field="query", config="en")


def triviaqa_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the TriviaQA dataset."""
    name = "mandarjoshi/trivia_qa"
    dataset = cast(Dataset, load_dataset(name, "unfiltered.nocontext", split="train"))
    dataset = dataset.rename_columns({"question": "text"})
    columns_to_remove = [col for col in dataset.column_names if col not in ("text",)]
    dataset = dataset.remove_columns(columns_to_remove)
    dataset_iterator = cast(Iterator[dict[str, str]], iter(dataset))
    return name, dataset_iterator


def mr_tydi_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the Mr. TyDi dataset."""
    hf = "sentence-transformers/mr-tydi"
    return hf, _simple_text_field_dataset(hf, text_field="anchor", config="en-triplet")


def get_all_dataset_functions() -> dict[str, Callable[[], tuple[str, Iterator[dict[str, str]]]]]:
    """Get all available dataset functions."""
    return {
        "english-words-definitions": english_words_definitions_dataset,
        "fineweb": fineweb_dataset,
        "gooaq": gooaq_dataset,
        "miracl": miracl_dataset,
        "lotte": lotte_queries_dataset,
        "snli": snli_dataset,
        "paws": paws_dataset,
        "squad": squad_dataset,
        "mldr": mldr_dataset,
        "msmarco": msmarco_queries_dataset,
        "msmarco_docs": msmarco_docs_dataset,
        "PubMedQA": pubmed_dataset,
        "swim-ir-monolingual": swim_ir_dataset,
        "trivia_qa": triviaqa_dataset,
        "mr-tydi": mr_tydi_dataset,
    }


def short_dataset_name(hf_name: str) -> str:
    """
    Return the short dataset name from a HF hub identifier.

    Examples:
        - 'MongoDB/english-words-definitions' -> 'english-words-definitions'
        - 'msmarco' -> 'msmarco'

    This centralizes the logic for deriving local filenames from full HF paths.

    Args:
        hf_name: Full HF dataset identifier.

    Returns:
        Short dataset name.

    """
    if not hf_name:
        return hf_name
    return hf_name.split("/")[-1]
