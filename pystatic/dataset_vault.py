from collections.abc import Callable, Iterable, Iterator
from typing import cast

from datasets import Dataset, concatenate_datasets, load_dataset


def _simple_text_field_dataset(
    dataset_name: str,
    huggingface_name: str,
    text_field: str,
    config: str | None = None,
    split: str = "train",
) -> tuple[str, Iterator[dict[str, str]]]:
    """Helper function for datasets that extract a single text field."""
    dataset = cast(Dataset, load_dataset(huggingface_name, config, split=split))

    new_records: list[dict[str, str]] = []
    for record in cast(Iterable[dict[str, str]], dataset):
        text = record[text_field]
        new_records.append({"text": text})

    dataset_iterator = cast(Iterator[dict[str, str]], iter(new_records))
    return dataset_name, dataset_iterator


def english_words_definitions_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the English Words Definitions dataset."""
    name = "english-words-definitions"
    dataset = cast(Dataset, load_dataset("MongoDB/english-words-definitions", split="train"))
    dataset = dataset.map(lambda x: {"text": " ".join(x["definitions"])})
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    dataset_iterator = cast(Iterator[dict[str, str]], iter(dataset))

    return name, dataset_iterator


def fineweb_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the FineWeb dataset (returns first path only)."""
    name = "fineweb"
    data = cast(
        Iterator[dict[str, str]],
        load_dataset("HuggingFaceFW/fineweb", "sample-10BT", streaming=True, split="train"),
    )
    return name, iter(data)


def gooaq_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the GoOAQ dataset."""
    return _simple_text_field_dataset(
        dataset_name="gooaq", huggingface_name="sentence-transformers/gooaq", text_field="question"
    )


def miracl_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the MIRACL dataset."""
    return _simple_text_field_dataset(
        dataset_name="miracl", huggingface_name="sentence-transformers/miracl", text_field="anchor", config="en-triplet"
    )


def lotte_queries_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the LoTTE queries dataset."""
    name = "lotte"
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
    return name, dataset_iterator


def snli_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the SNLI dataset."""
    name = "snli"
    dataset = cast(Dataset, load_dataset("stanfordnlp/snli", split="train"))

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
    name = "paws"
    dataset = load_dataset("google-research-datasets/paws", "unlabeled_final", split="train")

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
    return _simple_text_field_dataset(
        dataset_name="squad", huggingface_name="sentence-transformers/squad", text_field="question"
    )


def mldr_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the MLDR dataset."""
    return _simple_text_field_dataset(
        dataset_name="mldr", huggingface_name="sentence-transformers/mldr", text_field="anchor", config="en-triplet"
    )


def msmarco_queries_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the MS MARCO BM25 queries dataset."""
    return _simple_text_field_dataset(
        dataset_name="msmarco-bm25",
        huggingface_name="sentence-transformers/msmarco-bm25",
        text_field="query",
        config="triplet",
    )


def pubmed_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the PubMedQA dataset."""
    name = "PubMedQA"
    subsets = ["pqa_artificial", "pqa_unlabeled"]
    datasets = []
    for subset in subsets:
        dataset = cast(Dataset, load_dataset("qiaojin/PubMedQA", subset, split="train"))
        dataset = dataset.rename_columns({"question": "text"})
        columns_to_remove = [col for col in dataset.column_names if col not in ("text",)]
        dataset = dataset.remove_columns(columns_to_remove)
        datasets.append(dataset)

    final_dataset = concatenate_datasets(datasets)
    dataset_iterator = cast(Iterator[dict[str, str]], iter(final_dataset))
    return name, dataset_iterator


def swim_ir_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the SWIM-IR dataset."""
    return _simple_text_field_dataset(
        dataset_name="swim-ir-monolingual",
        huggingface_name="nthakur/swim-ir-monolingual",
        text_field="query",
        config="en",
    )


def triviaqa_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the TriviaQA dataset."""
    name = "trivia_qa"
    dataset = cast(Dataset, load_dataset("mandarjoshi/trivia_qa", "unfiltered.nocontext", split="train"))
    dataset = dataset.rename_columns({"question": "text"})
    columns_to_remove = [col for col in dataset.column_names if col not in ("text",)]
    dataset = dataset.remove_columns(columns_to_remove)
    dataset_iterator = cast(Iterator[dict[str, str]], iter(dataset))
    return name, dataset_iterator


def mr_tydi_dataset() -> tuple[str, Iterator[dict[str, str]]]:
    """Get the Mr. TyDi dataset."""
    return _simple_text_field_dataset(
        dataset_name="mr-tydi",
        huggingface_name="sentence-transformers/mr-tydi",
        text_field="anchor",
        config="en-triplet",
    )


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
        "msmarco-bm25": msmarco_queries_dataset,
        "PubMedQA": pubmed_dataset,
        "swim-ir-monolingual": swim_ir_dataset,
        "trivia_qa": triviaqa_dataset,
        "mr-tydi": mr_tydi_dataset,
    }
