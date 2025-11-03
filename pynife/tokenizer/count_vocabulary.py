from collections import Counter
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from skeletoken import TokenizerModel
from skeletoken.preprocessor import Preprocessor
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from pynife.tokenizer.datamodels import VocabItem
from pynife.utilities import iterable_iterator_dispatch


def _process_example(example: str, preprocessor: Preprocessor) -> Counter[str]:
    """Process a single example to count tokens."""
    tokens = preprocessor.preprocess(example)
    return Counter(tokens)


def count_tokens_in_dataset(
    tokenizer: PreTrainedTokenizerFast, data: Iterator[str] | Iterable[str], total: int | None = None
) -> list[VocabItem]:
    """Count token frequencies and document frequencies in the given dataset."""
    data = iterable_iterator_dispatch(data)
    counts: Counter[str] = Counter()

    tokenizer_model = TokenizerModel.from_transformers_tokenizer(tokenizer)
    preprocessor = Preprocessor.from_tokenizer_model(tokenizer_model)

    partialed_process_example = partial(_process_example, preprocessor=preprocessor)

    with ThreadPoolExecutor() as executor:
        for token_counter in tqdm(executor.map(partialed_process_example, data), total=total):
            counts.update(token_counter)

    vocabulary_data: list[VocabItem] = []
    for token, frequency in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        vocabulary_data.append(VocabItem(token=token, frequency=frequency))

    return vocabulary_data
