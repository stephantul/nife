# (k)NIFE

This is the repository for training Nearly Inference Free Embedding (NIFE) models. NIFE models are [static embedding](https://huggingface.co/blog/static-embeddings) models that are fully aligned with a much larger model. NIFE allows you to:

1. Use a much smaller model during querying. 400x embed time speed-up on CPU.
2. Use a much smaller memory/compute footprint. Create embeddings in your DB service.
3. Use the same index as your big model. Switch dynamically between your big model and the NIFE model.

# Usage

A NIFE model is just a [sentence transformer](https://github.com/huggingface/sentence-transformers) model, so you don't need to install anything except that. Nevertheless, NIFE contains some helper functions for loading a model trained with NIFE. As an example, we'll load our base model [], which was trained using [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) as a teacher.

```python
from nife import load_nife
from sentence_transformers import SentenceTransformer, Router

model = load_nife("")


```

# Rationale
