from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from skeletoken import TokenizerModel
from skeletoken.preprocessor import Preprocessor
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from nife.tokenizer.datamodels import VocabItem


def _process_example(example: dict[str, str], preprocessor: Preprocessor) -> tuple[Counter[str], set[str]]:
    """Process a single example to count tokens."""
    text = example["text"]
    tokens = preprocessor.preprocess(text)
    return Counter(tokens), set(tokens)


def count_tokens_in_dataset(
    tokenizer: PreTrainedTokenizerFast, data: Iterator[dict[str, str]], total: int
) -> list[VocabItem]:
    """Count token frequencies and document frequencies in the given dataset."""
    counts: Counter[str] = Counter()
    df: Counter[str] = Counter()

    tokenizer_model = TokenizerModel.from_transformers_tokenizer(tokenizer)
    preprocessor = Preprocessor.from_tokenizer_model(tokenizer_model)

    partialed_process_example = partial(_process_example, preprocessor=preprocessor)

    with ThreadPoolExecutor() as executor:
        for token_counter, token_set in tqdm(executor.map(partialed_process_example, data), total=total):
            counts.update(token_counter)
            df.update(token_set)

    vocabulary_data: list[VocabItem] = []
    for token, frequency in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        document_frequency = df[token]
        vocabulary_data.append(VocabItem(token=token, frequency=frequency, document_frequency=document_frequency))

    return vocabulary_data
