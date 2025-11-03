from jinja2 import Template

DATASET_CARD_TEMPLATE = """---
dataset_info:
  features:
  - name: text
    dtype: string
  - name: embedding
    list: float32
    length: {{ length }}
  splits:
  - name: train
{% if size_kind == "examples" %}
    num_examples: {{ size }}
{% elif size_kind == "bytes" %}
    num_bytes: {{ size }}
{% endif %}
  dataset_size: {{ dataset_size }}
  download_size: {{ download_size }}
configs:
- config_name: default
  data_files:
  - split: train
    path: train/*
metadata:
  model_name: {{ model_name }}
  dataset_name: {{ dataset_name }}
---

# Embedpress: {{ model_name }} on the {{ dataset_name }} dataset

This is the [{{ dataset_name }}](https://huggingface.co/datasets/{{ dataset_name }}) dataset,
embedded with [{{ model_name }}](https://huggingface.co/{{ model_name }}).

For each example, we embed the text directly (no additional instruction prompt).
Embeddings have dimensionality **{{ length }}**.

These embeddings are intended for tasks like large-scale distillation, retrieval, and similarity search.
Because the raw text may exceed the model's limit, we recommend truncating to the model's maximum token length at build time.

## Schema

- `text` *(string)* — the query text used for embedding
- `embedding` *(float32[{{ length }}])* — the vector representation from `{{ model_name }}`

## Split

- `train` — {% if size_kind == "examples" %}**{{ size }} examples**{% elif size_kind == "bytes" %}**{{ size }} bytes**{% else %}size not specified{% endif %}

## Notes

- Produced with `{{ model_name }}` from Hugging Face Hub.
- If you need a smaller embedding size (e.g., matryoshka/truncated vectors), you can safely slice the embeddings without re-embedding.
"""


def generate_dataset_card(
    model_name: str,
    dataset_name: str,
    length: int,
    size: int,
    size_kind: str = "examples",
    dataset_size: int | None = None,
    download_size: int | None = None,
) -> str:
    """
    Generate a dataset card in Markdown from a Jinja2 template.

    Args:
        model_name: Full model identifier (e.g. 'mixedbread-ai/mxbai-embed-large-v1')
        dataset_name: Hugging Face dataset identifier (e.g. 'qiaojin/PubMedQA')
        length: Dimensionality of the embedding vectors
        size: Either number of examples or bytes, depending on `size_kind`
        size_kind: Either 'examples' or 'bytes' (default: 'examples')
        dataset_size: Total dataset size in bytes
        download_size: Compressed download size in bytes

    Returns:
        Rendered dataset card as a string

    """
    template = Template(DATASET_CARD_TEMPLATE)
    rendered = template.render(
        model_name=model_name,
        dataset_name=dataset_name,
        length=length,
        size=size,
        size_kind=size_kind,
        dataset_size=dataset_size or size,
        download_size=download_size or size,
    )
    return rendered
