# (k)NIFE

This is the repository for training Nearly Inference Free Embedding (NIFE) models. NIFE models are [static embedding](https://huggingface.co/blog/static-embeddings) models that are fully aligned with a much larger model. NIFE allows you to:

1. Use a much smaller model during querying. 400x embed time speed-up on CPU.
2. Use a much smaller memory/compute footprint. Create embeddings in your DB service.
3. Use the same index as your big model. Switch dynamically between your big model and the NIFE model.

# Use cases

Here's some things you can do with NIFE.

## Search engine

You have a search engine and want to offer both a fast path and slow path to your customers. The fast path computes the query using the NIFE model, the slow path computes it using the slow model. Your customers pay less for the NIFE model. You can use the same document index for both models, so there's no overhead to deploying NIFE.

## RAG

Your agent sometimes needs to retrieve large sets of documents and you want to get this context to the agent as fast as possible. You are bound to using relatively low amounts of compute. A NIFE model can run on a toaster, so you run a small server next to your agent that lets it compute embeddings really quickly.

## On the fly document comparisons

You have a very fast pipeline of incoming documents for which you need to compute similarities, e.g., to detect near duplicates. You can compute your corpus using the big transformer, but when new documents come in, you use the NIFE model to create vectors.

## Free additional features

NIFE models compute different things than their teacher models, so they can be used as an additional source of information during ranking.

# Usage

A NIFE model is just a [sentence transformer](https://github.com/huggingface/sentence-transformers) router model, so you don't need to install anything except that. Nevertheless, NIFE contains some helper functions for loading a model trained with NIFE. As an example, we'll load our base model [], which was trained using [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) as a teacher.

Note that in all cases the teacher model is unchanged; so if you have a large set of documents indexe with the teacher model, you can use the NIFE model as a drop-in replacement.

## Usage with NIFE

```python
from nife import load_nife
from sentence_transformers import SentenceTransformer, Router

model = load_nife("")

documents = ["", ""]
index = model.encode_document(documents, normalize_embeddings=True)
query = [""]
query_vector = model.encode_query(query, normalize_embeddings=True)

similarity = query_vector @ index.T

```

## Usage with sentence-transformers

This is just the snipper from the `nife` library. Use this is if you don't feel like downloading `nife`.

```python
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

model = load_nife("")

```

# Rationale

For retrieval using dense models, the normal mode of operation is to embed your documents, and put them in some index. Then, using that same model, also embed your queries. In general, larger embedding models are better than smaller models, so you're often better off by making your embedder as large as possible. This however, makes inference more difficult; you need to host a larger model, and embedding queries might take longer.

For sparse models, like [SPLADE](https://arxiv.org/pdf/2107.05720), there is an interesting alternative, which they call doc-SPLADE, and which sentence transformers calls inference free_. In doc-SPLADE, you only embed using the full model for documents in your index. When querying, you just index the sparse index using the tokenizer.

NIFE is the answer to the question: what would inference-free dense retrieval be? It is called _Nearly_ Inference Free, because you still need to have some mapping from tokens to embeddings.

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

Let's assume you have some dataset you want to create embeddings for. This works as follows:

```python
from datasets import load_dataset
from nife.distillation.infer import generate_and_save_embeddings
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

After a while, your dataset will be ready and saved as parquet files in `output_directory`. If you want to upload these, please use the `HfAPI`, not `dataset.push_to_hub`.

You can then train on this using the script in the scripts folder.

```bash
python3 scripts/experiment_distillation.py my-new-model --train-dataset output_directory
```

This will train a model and report the result to wandb. The `experiment_distillation` script is otherwise completely the same as a regular sentence transformers training loop, so there's very little actual code involved.

# License

MIT

# Author

Stéphan Tulkens
