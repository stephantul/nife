from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jinja2 import Template

from pynife.cards.model_card import get_model_card_template_path


@dataclass
class LossConfig:
    fullname: str
    config_code: str | None = None


def _build_model_card_context(
    *,
    # top matter
    card_data: str,
    model_name: str | None = None,
    base_model: str | None = None,
    output_dimensionality: int,
    license: str | None = None,
    model_string: str,
    model_id: str,
    # training logs
    eval_lines: str | None = None,
    hide_eval_lines: bool = False,
    explain_bold_in_eval: bool = False,
    # citations (loss_name -> bibtex string)
    citations: dict[str, str] = field(default_factory=dict),
) -> dict[str, Any]:
    """
    Build a context dict exposing all variables used by the Jinja template.

    Return value is safe to splat into Jinja: Template(...).render(**context)
    """
    context: dict[str, Any] = {
        "card_data": card_data,
        "model_name": model_name,
        "base_model": base_model,
        "output_dimensionality": output_dimensionality,
        "license": license,
        "model_string": model_string,
        "model_id": model_id,
        "eval_lines": eval_lines or None,
        "hide_eval_lines": hide_eval_lines,
        "explain_bold_in_eval": explain_bold_in_eval,
        "citations": citations or {},
    }
    return context


def _loss(fullname: str, config_code: str | None = None) -> LossConfig:
    """Loss helper."""
    return LossConfig(fullname=fullname, config_code=config_code)


if __name__ == "__main__":
    tmpl = Template(get_model_card_template_path().read_text())

    model = """SentenceTransformer(
  (0): StaticEmbedding(
    (embedding): EmbeddingBag(107962, 1024, mode='mean')
  )
  (1): Normalize()
)
"""

    base_model = "mixedbread-ai/mxbai-embed-large-v1"

    ctx = _build_model_card_context(
        card_data=f"language: en\nlicense: mit\nbase_model: {base_model}",
        model_name="NIFE mxbai embed large v1",
        base_model=base_model,
        output_dimensionality=1024,
        model_string=model,
        model_id="stephantulkens/NIFE-mxbai-embed-large-v1",
    )

    rendered = tmpl.render(**ctx)
