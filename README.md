
<h2 align="center">
  <img width="35%" alt="A man shooting a thing to the ground." src="https://github.com/stephantul/pynife/blob/main/assets/william-blake.jpg"><br/>
</h2>
<h1 align="center"> pyNIFE </h1>

<div align="center">
  <h2>
    <a href="https://pypi.org/project/pynife/"><img src="https://img.shields.io/pypi/v/pynife?color=f29bdb" alt="Package version">
</a>
    <a href="https://codecov.io/gh/stephantul/pynife" >
      <img src="https://codecov.io/gh/stephantul/pynife/graph/badge.svg?token=DD8BK7OZHG"/>
    </a>
    <a href="https://github.com/stephantul/pynife/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/license-MIT-green" alt="License - MIT">
    </a>
</div>

NIFE compresses large embedding models into static, drop-in replacements with up to 200x faster query embedding [see benchmarks]().

## Features

- 200x faster CPU query embedding
- Fully aligned with their teacher models
- Re-use your existing vector index

## Introduction

Nearly Inference Free Embedding (NIFE) models are [static embedding](https://huggingface.co/blog/static-embeddings) models that are fully aligned with a much larger model. Because static models are so small and fast, NIFE allows you to:

1. Speed up query time immensely: 200x embed time speed-up on CPU.
2. Get away with using a much smaller memory/compute footprint. Create embeddings in your DB service.
3. Reuse your big model index: Switch dynamically between your big model and the NIFE model.

Some possible use-cases for NIFE include search engines with slow and fast paths, RAGs in agent loops, and on-the-fly document comparisons.

## Quickstart

This snippet loads [`stephantulkens/NIFE-mxbai-embed-large-v1`](https://huggingface.co/stephantulkens/NIFE-mxbai-embed-large-v1), which is aligned with [`mixedbread-ai/mxbai-embed-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1). Use it in any spot where you use `mixedbread-ai/mxbai-embed-large-v1`.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("stephantulkens/NIFE-mxbai-embed-large-v1", device="cpu")
# Loads in 41ms.
query_vec = model.encode(["What is the capital of France?"])
# Embedding a query takes 90.4 microseconds.

big_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device="cpu")
# Four cities near France
index_doc = big_model.encode(["Paris is the largest city in France", "Lyon is pretty big", "Antwerp is really great, and in Belgium", "Berlin is pretty gloomy in winter", "France is a country in Europe"])

similarity = model.similarity(query_vec, index_doc)
print(similarity)
# It correctly retrieved the document containing the statement about paris.
# tensor([[0.7065, 0.5012, 0.3596, 0.2765, 0.6648]])

big_model_query_vec = big_model.encode(["What is the capital of France?"])
# Embedding a query takes 68.1 ms (~750 times slower)
similarity = model.similarity(big_model_query_vec, index_doc)
# Compare to the above. Very similar.
# tensor([[0.7460, 0.5301, 0.3816, 0.3423, 0.6692]])

similarity_queries = model.similarity(big_model_query_vec, query_vec)
# The two vectors are very similar.
# tensor([[0.9377]])

```

This snippet is an example of how you could use it. But in reality you should just use it wherever you encode a query using your teacher model. There's no need to keep the teacher in memory. This makes NIFE extremely flexible, because you can decouple the inference model from the indexing model. Because the models load extremely quickly, they can be used in edge environments and one-off things like lambda functions.

## Installation

On [PyPi](https://pypi.org/project/pynife/):

```
pip install pynife
```

## Usage

A NIFE model is just a [sentence transformer](https://github.com/huggingface/sentence-transformers) router model, so you don't need to install `pynife` to use NIFE models. Nevertheless, NIFE contains some helper functions for loading a model trained with NIFE.

Note that with all NIFE models the teacher model is unchanged; so if you have a large set of documents indexed with the teacher model, you can use the NIFE model as a drop-in replacement.

### Standalone

Use just like any other sentence transformer:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("stephantulkens/NIFE-mxbai-embed-large-v1", device="cpu")
X = model.encode(["What is the capital of France?"])
```

### As a router

You can also use the small model and big model together as a single [router](https://sbert.net/docs/package_reference/sentence_transformer/models.html#sentence_transformers.models.Router) using a helper function from `pynife`. This is useful for benchmarking; in production you should probably use the query model by itself.

```python
from pynife import load_as_router

model = load_as_router("stephantulkens/NIFE-mxbai-embed-large-v1")
# Use the fast model
query = model.encode_query("What is the capital of France?")
# Use the slow model
docs = model.encode_document("What is the capital of France?")

print(model.similarity(query, docs))
# Same result as above in the quickstart.
# tensor([[0.9377]])

```

## Rationale

For retrieval using dense models, the normal mode of operation is to embed your documents, and put them in some index. Then, using that same model, also embed your queries. In general, larger embedding models are better than smaller models, so you're often better off by making your embedder as large as possible. This however, makes inference more difficult; you need to host a larger model, and embedding queries might take longer.

For sparse models, like [SPLADE](https://arxiv.org/pdf/2107.05720), there is an interesting alternative, which they call doc-SPLADE, and which sentence transformers calls _inference free_. In doc-SPLADE, you only embed using the full model for documents in your index. When querying, you just index the sparse index using the tokenizer.

NIFE is the answer to the question: what would inference free dense retrieval be? It is called _Nearly_ Inference Free, because you still need to have some mapping from tokens to embeddings.

See this table:

|                | Sparse     | Dense                |
|----------------|------------|----------------------|
| Full           | SPLADE     | Sentence transformer |
| Inference free | doc-SPLADE | NIFE                 |

As in doc-SPLADE, you lose performance. No real way about it, but as with other fast models, the gap is smaller than you might think.

## How does it work?

We use knowledge distillation from an initialized static model to the teacher we want to emulate. Some special things:

1) The static model is initialized directly from the teacher by inferring all tokens in the tokenizer through the whole model. This is similar to how this was done in [model2vec](https://github.com/MinishLab/model2vec), except we skip the PCA and weighting steps.
2) The knowledge distillation is done in cosine space. We don't guarantee any alignment in euclidean space. Using, e.g., MSE or KLDiv between the student and teacher did not work as well.
3) We train a custom tokenizer on our pre-training corpus, which is MsMARCO. This custom tokenizer is based on `bert-base-uncased`, but with a lot of added vocabulary. The models used in NIFE all have around 100k vocabulary size.
4) We perform two stages of training; following [LEAF](https://arxiv.org/pdf/2509.12539), we also train on _queries_. This raises performance considerably, but training on interleaved queries and documents does not work very well. So we first train on a corpus of documents (MsMarco), and then finetune using a lower learning rate on a large selection of queries from a variety of sources.
5) Unlike LEAF, we leave out all instructions from the knowledge distillation process. Static models can't deal with instructions, because there is no interaction between the instruction and other tokens in the document. Instructions can therefore at best be a constant _offset_ of your embedding space. This can be really useful, but not for this specific task.

## Caveats/weaknesses

NIFE can't do the following things:

1) Ignore words based on context: the query "What is the capital of France?" the word "France" will cause documents containing the term "France" to be retrieved. There is no way for the model to attenuate this vector and morph it into the answer ("Paris").
2) Deal with negation: for the same reason as above; there is no interaction between tokens, so the similarity between "Cars that aren't red" and "Cars that are red" will be really high.

# Creating a NIFE model

To create a NIFE model, you can run the scripts in `scripts`, or directly use the code from the repository. First, you should create a corpus of embeddings for your embedder. You can also use pre-computed collections of embeddings I created:

* [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/collections/stephantulkens/mxbai-large-v1-embedpress)
* [Alibaba-NLP/gte-modernbert-base](https://huggingface.co/collections/stephantulkens/gte-modernbert-embedpress)

Broadly construed, training a NIFE model has 5 separate steps.

### 1. Create a set of embeddings using the teacher

Let's assume we want to create embeddings on [trivia QA](https://huggingface.co/mandarjoshi/trivia_qa), using `mxbai-embed-large-v1` as a teacher.

```python
from datasets import load_dataset
from pynife.distillation.infer import generate_and_save_embeddings
from sentence_transformers import SentenceTransformer

model_name = "mixedbread-ai/mxbai-embed-large-v1"
model = SentenceTransformer(model_name)

dataset_name = "mandarjoshi/trivia_qa"
dataset = load_dataset(dataset_name, "rc", split="train")
dataset_iterator = (x['question'] for x in dataset)

output_directory = "my-trivia-qa"

generate_and_save_embeddings(
    model=model,
    records=dataset_iterator,
    output_folder=output_directory,
    limit_batches=None,
    batch_size=8,
    save_every=512,
    max_length=512,
    model_name=model_name,
    dataset_name=dataset_name,
    lowercase=False,
    make_greedy=False,
    )

```

This piece of code loads the model, the dataset and then starts inference. Inference takes a while, and will stream snippets to disk as .txt files and torch tensor files. After the whole dataset has been inferenced, the .txt and tensor files are converted into parquet files, and the .txt and torch tensor files are deleted.

Your dataset will be ready and saved as parquet files in `output_directory`. If you want to upload these, please use the `HfAPI`, not `dataset.push_to_hub`, because we rely on some metadata embedded in the README to infer the base model later on. Note that the dataset iterator can be anything, and does not need to be a Hugging Face dataset. For example, it could also work with a stream from your database.

For a simple inference script with a lot of pre-made datasets, see [the infer_datasets script](./scripts/infer_datasets.py).

### 2. (optional) Expanding a tokenizer

NIFE models work really well if you create a custom tokenizer for your domain. Empirically, it also works really well if you just expand the tokenizer of your teacher model with additional words. We call this _tokenizer expansion_. We have a pre-defined corpus to work on:

```python
from transformers import AutoTokenizer

from datasets import load_dataset
from pynife.tokenizer.expand_tokenizer import expand_tokenizer


dataset = load_dataset("stephantulkens/msmarco-vocab", split="train")
print(dataset.tolist()[:5])
# [{'token': '.', 'frequency': 36174594, 'document_frequency': 8701009},
# {'token': 'the', 'frequency': 28806701, 'document_frequency': 7712172},
# {'token': ',', 'frequency': 25825435, 'document_frequency': 7411743},
# {'token': 'of', 'frequency': 15196930, 'document_frequency': 6562023},
# {'token': 'a', 'frequency': 13702107, 'document_frequency': 6064770},

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Function expects an iterator over dictionaries with "token" and "frequency" as keys.
new_tokenizer = expand_tokenizer(tokenizer, data, new_vocabulary_size=30000)
new_tokenizer.save_pretrained("my_tokenizer")

```

This will do a couple of things:
1) It will remove all tokens from the original tokenizer that aren't present in your data.
2) It will then add the most frequent tokens until the size of the tokenizer == `new_vocabulary_size`.

This works a lot better than training a tokenizer from scratch on equivalent data. For a runnable version, see [the expand_tokenizer script](./scripts/expand_tokenizer.py).

To get frequency counts, you can use `count_tokens_in_dataset`, as follows:

```python
from datasets import load_dataset, Dataset

from pynife.tokenizer.count_vocabulary import count_tokens_in_dataset

dataset = load_dataset("sentence-transformers/msmarco", "corpus", split="train", streaming=True)
dataset_iterator = (item["passage"] for item in dataset)
counts = count_tokens_in_dataset(dataset_iterator)

# Save the counts as a dataset if you want.
dataset = Dataset.from_list(counts, split="train")
dataset.push_to_hub("my_hub")

```

This dataset can be used directly to expand your tokenizer, above. For a runnable version, see [the create_vocabulary script](./scripts/create_vocabulary.py)

### 3. Train

Given a dataset and optionally a tokenizer, there's two steps to complete for a successful training.

#### 3a Initialize a static model using your teacher

Using *your teacher model*, initialize a static model. For example, when using [`mixedbread-ai/mxbai-embed-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1):

```python
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from pynife.initialization import initialize_from_model

teacher = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
# The tokenizer you trained in step 2. or an off-the-shelf tokenizer.
tokenizer = AutoTokenizer.from_pretrained("my_tokenizer")
model = initialize_from_model(teacher, tokenizer)

```

#### 3b Actually train

Now you can train, just like a regular sentence transformer. In my experiments, I found that using the cosine distance as a loss function was superior to using MSE, so I recommend using that, find it in `pynife.losses`. In addition, I also recommend using [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147). There's a bunch of helper functions in `pynife` to make training easier. In general, I recommend using hyperparameters like the following:

* `batch_size`: 128
* `learning rate`: 0.01
* `scheduler`: "cosine_warmup_with_min_lr"
* `warmup_ratio`: 0.1
* `weight_decay`: 0.01
* `epochs`: 5

It can be tempting to move to very high batch sizes, but this has a very large detrimental effect on performance, even with higher learning rates. As a consequence, GPU usage during training is actually pretty low, because there's very little actual computation happening. For a complete runnable training loop, including model initialization, see [the training script](./scripts/experiment_distillation.py).

```python
from pynife.losses import CosineLoss
from pynife.data import get_datasets

# Fill with datasets you trained yourself.
datasets_you_made = [""]
train_dataset = get_datasets(datasets_you_made)

# Model is initialized in step 3a.
loss = CosineLoss(model=model)

# Train as usual.

```

This will train a model and report the result to wandb. The `experiment_distillation` script is otherwise completely the same as a regular sentence transformers training loop, so there's very little actual code involved.

## License

MIT

## Author

Stéphan Tulkens
