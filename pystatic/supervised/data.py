import logging
from typing import cast

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


def load_retrieval_train_eval_datasets() -> tuple[DatasetDict, DatasetDict]:
    """
    Either load the train and eval datasets from disk or load them from the datasets library & save them to disk.

    Upon saving to disk, we quit() to ensure that the datasets are not loaded into memory before training.
    """
    logger.info("Loading gooaq dataset...")
    gooaq_dataset = cast(Dataset, load_dataset("sentence-transformers/gooaq", split="train"))
    gooaq_dataset_dict = gooaq_dataset.train_test_split(test_size=10_000, seed=12)
    gooaq_train_dataset: Dataset = gooaq_dataset_dict["train"]
    gooaq_eval_dataset: Dataset = gooaq_dataset_dict["test"]
    logger.info("Loaded gooaq dataset.")

    logger.info("Loading msmarco dataset...")
    msmarco_dataset = cast(
        Dataset,
        load_dataset(
            "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1", "triplet", split="train"
        ),
    )
    msmarco_dataset_dict = msmarco_dataset.train_test_split(test_size=10_000, seed=12)
    msmarco_train_dataset: Dataset = msmarco_dataset_dict["train"]
    msmarco_eval_dataset: Dataset = msmarco_dataset_dict["test"]
    logger.info("Loaded msmarco dataset.")

    logger.info("Loading squad dataset...")
    squad_dataset = cast(Dataset, load_dataset("sentence-transformers/squad", split="train"))
    squad_dataset_dict = squad_dataset.train_test_split(test_size=10_000, seed=12)
    squad_train_dataset: Dataset = squad_dataset_dict["train"]
    squad_eval_dataset: Dataset = squad_dataset_dict["test"]
    logger.info("Loaded squad dataset.")

    logger.info("Loading allnli dataset...")
    allnli_train_dataset = cast(Dataset, load_dataset("sentence-transformers/all-nli", "triplet", split="train"))
    allnli_eval_dataset = cast(Dataset, load_dataset("sentence-transformers/all-nli", "triplet", split="dev"))
    logger.info("Loaded allnli dataset.")

    logger.info("Loading trivia_qa dataset...")
    trivia_qa = cast(Dataset, load_dataset("sentence-transformers/trivia-qa", split="train"))
    trivia_qa_dataset_dict = trivia_qa.train_test_split(test_size=5_000, seed=12)
    trivia_qa_train_dataset: Dataset = trivia_qa_dataset_dict["train"]
    trivia_qa_eval_dataset: Dataset = trivia_qa_dataset_dict["test"]
    logger.info("Loaded trivia_qa dataset.")

    logger.info("Loading msmarco_10m dataset...")
    msmarco_10m_dataset = cast(Dataset, load_dataset("bclavie/msmarco-10m-triplets", split="train"))
    msmarco_10m_dataset_dict = msmarco_10m_dataset.train_test_split(test_size=10_000, seed=12)
    msmarco_10m_train_dataset: Dataset = msmarco_10m_dataset_dict["train"]
    msmarco_10m_eval_dataset: Dataset = msmarco_10m_dataset_dict["test"]
    logger.info("Loaded msmarco_10m dataset.")

    logger.info("Loading swim_ir dataset...")
    swim_ir_dataset = cast(Dataset, load_dataset("nthakur/swim-ir-monolingual", "en", split="train")).select_columns(
        ["query", "text"]
    )
    swim_ir_dataset_dict = swim_ir_dataset.train_test_split(test_size=10_000, seed=12)
    swim_ir_train_dataset: Dataset = swim_ir_dataset_dict["train"]
    swim_ir_eval_dataset: Dataset = swim_ir_dataset_dict["test"]
    logger.info("Loaded swim_ir dataset.")

    # NOTE: 20 negatives
    logger.info("Loading pubmedqa dataset...")
    pubmedqa_dataset = cast(Dataset, load_dataset("sentence-transformers/pubmedqa", "triplet-20", split="train"))
    pubmedqa_dataset_dict = pubmedqa_dataset.train_test_split(test_size=100, seed=12)
    pubmedqa_train_dataset: Dataset = pubmedqa_dataset_dict["train"]
    pubmedqa_eval_dataset: Dataset = pubmedqa_dataset_dict["test"]
    logger.info("Loaded pubmedqa dataset.")

    # NOTE: A lot of overlap with anchor/positives
    logger.info("Loading miracl dataset...")
    miracl_dataset = cast(Dataset, load_dataset("sentence-transformers/miracl", "en-triplet-all", split="train"))
    miracl_dataset_dict = miracl_dataset.train_test_split(test_size=10_000, seed=12)
    miracl_train_dataset: Dataset = miracl_dataset_dict["train"]
    miracl_eval_dataset: Dataset = miracl_dataset_dict["test"]
    logger.info("Loaded miracl dataset.")

    # NOTE: A lot of overlap with anchor/positives
    logger.info("Loading mldr dataset...")
    mldr_dataset = cast(Dataset, load_dataset("sentence-transformers/mldr", "en-triplet-all", split="train"))
    mldr_dataset_dict = mldr_dataset.train_test_split(test_size=10_000, seed=12)
    mldr_train_dataset: Dataset = mldr_dataset_dict["train"]
    mldr_eval_dataset: Dataset = mldr_dataset_dict["test"]
    logger.info("Loaded mldr dataset.")

    # NOTE: A lot of overlap with anchor/positives
    logger.info("Loading mr_tydi dataset...")
    mr_tydi_dataset = cast(Dataset, load_dataset("sentence-transformers/mr-tydi", "en-triplet-all", split="train"))
    mr_tydi_dataset_dict = mr_tydi_dataset.train_test_split(test_size=10_000, seed=12)
    mr_tydi_train_dataset: Dataset = mr_tydi_dataset_dict["train"]
    mr_tydi_eval_dataset: Dataset = mr_tydi_dataset_dict["test"]
    logger.info("Loaded mr_tydi dataset.")

    train_dataset = DatasetDict(
        {
            "gooaq": gooaq_train_dataset,
            "msmarco": msmarco_train_dataset,
            "squad": squad_train_dataset,
            "allnli": allnli_train_dataset,
            "trivia_qa": trivia_qa_train_dataset,
            "msmarco_10m": msmarco_10m_train_dataset,
            "swim_ir": swim_ir_train_dataset,
            "pubmedqa": pubmedqa_train_dataset,
            "miracl": miracl_train_dataset,
            "mldr": mldr_train_dataset,
            "mr_tydi": mr_tydi_train_dataset,
        }
    )
    eval_dataset = DatasetDict(
        {
            "gooaq": gooaq_eval_dataset,
            "msmarco": msmarco_eval_dataset,
            "squad": squad_eval_dataset,
            "allnli": allnli_eval_dataset,
            "trivia_qa": trivia_qa_eval_dataset,
            "msmarco_10m": msmarco_10m_eval_dataset,
            "swim_ir": swim_ir_eval_dataset,
            "pubmedqa": pubmedqa_eval_dataset,
            "miracl": miracl_eval_dataset,
            "mldr": mldr_eval_dataset,
            "mr_tydi": mr_tydi_eval_dataset,
        }
    )
    return train_dataset, eval_dataset


def load_similarity_train_eval_datasets() -> tuple[DatasetDict, DatasetDict]:
    """
    Either load the train and eval datasets from disk or load them from the datasets library & save them to disk.

    Upon saving to disk, we quit() to ensure that the datasets are not loaded into memory before training.
    """
    logger.info("Loading wikititles dataset...")
    wikititles_dataset = cast(
        Dataset, load_dataset("sentence-transformers/parallel-sentences-wikititles", split="train")
    )
    wikititles_dataset_dict = wikititles_dataset.train_test_split(test_size=10_000, seed=12)
    wikititles_train_dataset: Dataset = wikititles_dataset_dict["train"]
    wikititles_eval_dataset: Dataset = wikititles_dataset_dict["test"]
    logger.info("Loaded wikititles dataset.")

    logger.info("Loading tatoeba dataset...")
    tatoeba_dataset = cast(
        Dataset, load_dataset("sentence-transformers/parallel-sentences-tatoeba", "all", split="train")
    )
    tatoeba_dataset_dict = tatoeba_dataset.train_test_split(test_size=10_000, seed=12)
    tatoeba_train_dataset: Dataset = tatoeba_dataset_dict["train"]
    tatoeba_eval_dataset: Dataset = tatoeba_dataset_dict["test"]
    logger.info("Loaded tatoeba dataset.")

    logger.info("Loading talks dataset...")
    talks_dataset = cast(Dataset, load_dataset("sentence-transformers/parallel-sentences-talks", "all", split="train"))
    talks_dataset_dict = talks_dataset.train_test_split(test_size=10_000, seed=12)
    talks_train_dataset: Dataset = talks_dataset_dict["train"]
    talks_eval_dataset: Dataset = talks_dataset_dict["test"]
    logger.info("Loaded talks dataset.")

    logger.info("Loading europarl dataset...")
    europarl_dataset = cast(
        Dataset, load_dataset("sentence-transformers/parallel-sentences-europarl", "all", split="train[:5000000]")
    )
    europarl_dataset_dict = europarl_dataset.train_test_split(test_size=10_000, seed=12)
    europarl_train_dataset: Dataset = europarl_dataset_dict["train"]
    europarl_eval_dataset: Dataset = europarl_dataset_dict["test"]
    logger.info("Loaded europarl dataset.")

    logger.info("Loading global voices dataset...")
    global_voices_dataset = cast(
        Dataset, load_dataset("sentence-transformers/parallel-sentences-global-voices", "all", split="train")
    )
    global_voices_dataset_dict = global_voices_dataset.train_test_split(test_size=10_000, seed=12)
    global_voices_train_dataset: Dataset = global_voices_dataset_dict["train"]
    global_voices_eval_dataset: Dataset = global_voices_dataset_dict["test"]
    logger.info("Loaded global voices dataset.")

    """logger.info("Loading jw300 dataset...")
    jw300_dataset = cast(Dataset, load_dataset("sentence-transformers/parallel-sentences-jw300", "all", split="train"))
    jw300_dataset_dict = jw300_dataset.train_test_split(test_size=10_000, seed=12)
    jw300_train_dataset: Dataset = jw300_dataset_dict["train"]
    jw300_eval_dataset: Dataset = jw300_dataset_dict["test"]
    logger.info("Loaded jw300 dataset.")"""

    logger.info("Loading muse dataset...")
    muse_dataset = cast(Dataset, load_dataset("sentence-transformers/parallel-sentences-muse", split="train"))
    muse_dataset_dict = muse_dataset.train_test_split(test_size=10_000, seed=12)
    muse_train_dataset: Dataset = muse_dataset_dict["train"]
    muse_eval_dataset: Dataset = muse_dataset_dict["test"]
    logger.info("Loaded muse dataset.")

    logger.info("Loading wikimatrix dataset...")
    wikimatrix_dataset = cast(
        Dataset, load_dataset("sentence-transformers/parallel-sentences-wikimatrix", "all", split="train")
    )
    wikimatrix_dataset_dict = wikimatrix_dataset.train_test_split(test_size=10_000, seed=12)
    wikimatrix_train_dataset: Dataset = wikimatrix_dataset_dict["train"]
    wikimatrix_eval_dataset: Dataset = wikimatrix_dataset_dict["test"]
    logger.info("Loaded wikimatrix dataset.")

    """logger.info("Loading opensubtitles dataset...")
    opensubtitles_dataset = cast(Dataset, load_dataset("sentence-transformers/parallel-sentences-opensubtitles", "all", split="train[:5000000]"))
    opensubtitles_dataset_dict = opensubtitles_dataset.train_test_split(test_size=10_000, seed=12)
    opensubtitles_train_dataset: Dataset = opensubtitles_dataset_dict["train"]
    opensubtitles_eval_dataset: Dataset = opensubtitles_dataset_dict["test"]
    logger.info("Loaded opensubtitles dataset.")"""

    logger.info("Loading stackexchange dataset...")
    stackexchange_dataset = cast(
        Dataset, load_dataset("sentence-transformers/stackexchange-duplicates", "post-post-pair", split="train")
    )
    stackexchange_dataset_dict = stackexchange_dataset.train_test_split(test_size=10_000, seed=12)
    stackexchange_train_dataset: Dataset = stackexchange_dataset_dict["train"]
    stackexchange_eval_dataset: Dataset = stackexchange_dataset_dict["test"]
    logger.info("Loaded stackexchange dataset.")

    logger.info("Loading quora dataset...")
    quora_dataset = cast(Dataset, load_dataset("sentence-transformers/quora-duplicates", "triplet", split="train"))
    quora_dataset_dict = quora_dataset.train_test_split(test_size=10_000, seed=12)
    quora_train_dataset: Dataset = quora_dataset_dict["train"]
    quora_eval_dataset: Dataset = quora_dataset_dict["test"]
    logger.info("Loaded quora dataset.")

    """logger.info("Loading wikianswers duplicates dataset...")
    wikianswers_duplicates_dataset = cast(Dataset, load_dataset("sentence-transformers/wikianswers-duplicates", split="train[:10000000]"))
    wikianswers_duplicates_dict = wikianswers_duplicates_dataset.train_test_split(test_size=10_000, seed=12)
    wikianswers_duplicates_train_dataset: Dataset = wikianswers_duplicates_dict["train"]
    wikianswers_duplicates_eval_dataset: Dataset = wikianswers_duplicates_dict["test"]
    logger.info("Loaded wikianswers duplicates dataset.")"""

    logger.info("Loading all nli dataset...")
    all_nli_train_dataset = cast(Dataset, load_dataset("sentence-transformers/all-nli", "triplet", split="train"))
    all_nli_eval_dataset = cast(Dataset, load_dataset("sentence-transformers/all-nli", "triplet", split="dev"))
    logger.info("Loaded all nli dataset.")

    logger.info("Loading simple wiki dataset...")
    simple_wiki_dataset = cast(Dataset, load_dataset("sentence-transformers/simple-wiki", split="train"))
    simple_wiki_dataset_dict = simple_wiki_dataset.train_test_split(test_size=10_000, seed=12)
    simple_wiki_train_dataset: Dataset = simple_wiki_dataset_dict["train"]
    simple_wiki_eval_dataset: Dataset = simple_wiki_dataset_dict["test"]
    logger.info("Loaded simple wiki dataset.")

    logger.info("Loading altlex dataset...")
    altlex_dataset = cast(Dataset, load_dataset("sentence-transformers/altlex", split="train"))
    altlex_dataset_dict = altlex_dataset.train_test_split(test_size=10_000, seed=12)
    altlex_train_dataset: Dataset = altlex_dataset_dict["train"]
    altlex_eval_dataset: Dataset = altlex_dataset_dict["test"]
    logger.info("Loaded altlex dataset.")

    logger.info("Loading flickr30k captions dataset...")
    flickr30k_captions_dataset = cast(Dataset, load_dataset("sentence-transformers/flickr30k-captions", split="train"))
    flickr30k_captions_dataset_dict = flickr30k_captions_dataset.train_test_split(test_size=10_000, seed=12)
    flickr30k_captions_train_dataset: Dataset = flickr30k_captions_dataset_dict["train"]
    flickr30k_captions_eval_dataset: Dataset = flickr30k_captions_dataset_dict["test"]
    logger.info("Loaded flickr30k captions dataset.")

    logger.info("Loading coco captions dataset...")
    coco_captions_dataset = cast(Dataset, load_dataset("sentence-transformers/coco-captions", split="train"))
    coco_captions_dataset_dict = coco_captions_dataset.train_test_split(test_size=10_000, seed=12)
    coco_captions_train_dataset: Dataset = coco_captions_dataset_dict["train"]
    coco_captions_eval_dataset: Dataset = coco_captions_dataset_dict["test"]
    logger.info("Loaded coco captions dataset.")

    logger.info("Loading nli for simcse dataset...")
    nli_for_simcse_dataset = cast(
        Dataset, load_dataset("sentence-transformers/nli-for-simcse", "triplet", split="train")
    )
    nli_for_simcse_dataset_dict = nli_for_simcse_dataset.train_test_split(test_size=10_000, seed=12)
    nli_for_simcse_train_dataset: Dataset = nli_for_simcse_dataset_dict["train"]
    nli_for_simcse_eval_dataset: Dataset = nli_for_simcse_dataset_dict["test"]
    logger.info("Loaded nli for simcse dataset.")

    logger.info("Loading negation dataset...")
    negation_dataset = cast(Dataset, load_dataset("jinaai/negation-dataset", split="train"))
    negation_dataset_dict = negation_dataset.train_test_split(test_size=100, seed=12)
    negation_train_dataset: Dataset = negation_dataset_dict["train"]
    negation_eval_dataset: Dataset = negation_dataset_dict["test"]
    logger.info("Loaded negation dataset.")

    train_dataset = DatasetDict(
        {
            "wikititles": wikititles_train_dataset,
            "tatoeba": tatoeba_train_dataset,
            "talks": talks_train_dataset,
            "europarl": europarl_train_dataset,
            "global_voices": global_voices_train_dataset,
            # "jw300": jw300_train_dataset,
            "muse": muse_train_dataset,
            "wikimatrix": wikimatrix_train_dataset,
            # "opensubtitles": opensubtitles_train_dataset,
            "stackexchange": stackexchange_train_dataset,
            "quora": quora_train_dataset,
            # "wikianswers_duplicates": wikianswers_duplicates_train_dataset,
            "all_nli": all_nli_train_dataset,
            "simple_wiki": simple_wiki_train_dataset,
            "altlex": altlex_train_dataset,
            "flickr30k_captions": flickr30k_captions_train_dataset,
            "coco_captions": coco_captions_train_dataset,
            "nli_for_simcse": nli_for_simcse_train_dataset,
            "negation": negation_train_dataset,
        }
    )
    eval_dataset = DatasetDict(
        {
            "wikititles": wikititles_eval_dataset,
            "tatoeba": tatoeba_eval_dataset,
            "talks": talks_eval_dataset,
            "europarl": europarl_eval_dataset,
            "global_voices": global_voices_eval_dataset,
            # "jw300": jw300_eval_dataset,
            "muse": muse_eval_dataset,
            "wikimatrix": wikimatrix_eval_dataset,
            # "opensubtitles": opensubtitles_eval_dataset,
            "stackexchange": stackexchange_eval_dataset,
            "quora": quora_eval_dataset,
            # "wikianswers_duplicates": wikianswers_duplicates_eval_dataset,
            "all_nli": all_nli_eval_dataset,
            "simple_wiki": simple_wiki_eval_dataset,
            "altlex": altlex_eval_dataset,
            "flickr30k_captions": flickr30k_captions_eval_dataset,
            "coco_captions": coco_captions_eval_dataset,
            "nli_for_simcse": nli_for_simcse_eval_dataset,
            "negation": negation_eval_dataset,
        }
    )

    return train_dataset, eval_dataset
