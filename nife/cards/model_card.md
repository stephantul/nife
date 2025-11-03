---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# {{ model_name if model_name else "NIFE model" }}

This is a [NIFE](https://github.com/stephantul/nife) model{% if base_model %} finetuned from [{{ base_model }}](https://huggingface.co/{{ base_model }}){% else %} trained{% endif %}{% if train_datasets | selectattr("name") | list %} on {% if train_datasets | selectattr("name") | map(attribute="name") | join(", ") | length > 200 %}{{ train_datasets | length }}{% else %}the {% for dataset in (train_datasets | selectattr("name")) %}{% if dataset.id %}[{{ dataset.name if dataset.name else dataset.id }}](https://huggingface.co/datasets/{{ dataset.id }}){% else %}{{ dataset.name }}{% endif %}{% if not loop.last %}{% if loop.index == (train_datasets | selectattr("name") | list | length - 1) %} and {% else %}, {% endif %}{% endif %}{% endfor %}{% endif %} dataset{{"s" if train_datasets | selectattr("name") | list | length > 1 else ""}}{% endif %}. It is fully aligned with its base model [{{ base_model }}](https://huggingface.co/{{ base_model }}), and can be used to perform Inference Free querying using an index made by this model. It can also be used in standalone mode.

## Model Details

### Model Description
- **Model Type:** Static Model
{% if base_model -%}
    - **Base model:** [{{ base_model }}](https://huggingface.co/{{ base_model }})
{%- else -%}
    <!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
{%- endif %}
- **Output Dimensionality:** {{ output_dimensionality }} dimensions

{% if license -%}
    - **License:** {{ license }}
{%- else -%}
    <!-- - **License:** Unknown -->
{%- endif %}

### Full Model Architecture

```
{{ model_string }}
```

## Usage

### Direct Usage (NIFE)

First install the NIFE library

```bash
pip install -U NIFE
```

Then you can run the model as follows:

```python
from nife import load_as_router

model = load_as_router("{{ model_id }}")

query = "What is the capital of France?"
query_embeddings = model.encode_query(query)

# Five locales near France
index_doc = model.encode_document(["Paris is the largest city in France", "Lyon is pretty big", "Antwerp is really great, and in Belgium", "Berlin is pretty gloomy in winter", "France is a country in Europe"])

similarity = model.similarity(query_vec, index_doc)
print(similarity)
# It correctly retrieved the document containing the statement about paris.
# tensor([[0.7065, 0.5012, 0.3596, 0.2765, 0.6648]])

```

{% if eval_metrics %}
## Evaluation

### Metrics
{% for metrics in eval_metrics %}
#### {{ metrics.description }}
{% if metrics.dataset_name %}
* Dataset{% if metrics.dataset_name is not string and metrics.dataset_name | length > 1 %}s{% endif %}: {% if metrics.dataset_name is string -%}
        `{{ metrics.dataset_name }}`
    {%- else -%}
        {%- for name in metrics.dataset_name -%}
            `{{ name }}`
            {%- if not loop.last -%}
                {%- if loop.index == metrics.dataset_name | length - 1 %} and {% else -%}, {% endif -%}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endif %}
* Evaluated with {% if metrics.class_name.startswith("sentence_transformers.") %}[<code>{{ metrics.class_name.split(".")[-1] }}</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.{{ metrics.class_name.split(".")[-1] }}){% else %}<code>{{ metrics.class_name }}</code>{% endif %}{% if metrics.config_code %} with these parameters:
{{ metrics.config_code }}{% endif %}

{{ metrics.table }}
{%- endfor %}{% endif %}

## Training Details
{% for dataset_type, dataset_list in [("training", train_datasets), ("evaluation", eval_datasets)] %}{% if dataset_list %}
### {{ dataset_type.title() }} Dataset{{"s" if dataset_list | length > 1 else ""}}
{% for dataset in dataset_list %}{% if dataset_list | length > 3 %}<details><summary>{{ dataset['name'] or 'Unnamed Dataset' }}</summary>
{% endif %}
#### {{ dataset['name'] or 'Unnamed Dataset' }}
{% if dataset['name'] %}
* Dataset: {% if 'id' in dataset %}[{{ dataset['name'] }}](https://huggingface.co/datasets/{{ dataset['id'] }}){% else %}{{ dataset['name'] }}{% endif %}
{%- if 'revision' in dataset and 'id' in dataset %} at [{{ dataset['revision'][:7] }}](https://huggingface.co/datasets/{{ dataset['id'] }}/tree/{{ dataset['revision'] }}){% endif %}{% endif %}
{% if dataset['size'] %}* Size: {{ "{:,}".format(dataset['size']) }} {{ dataset_type }} samples
{% endif %}* Columns: {% if dataset['columns'] | length == 1 %}{{ dataset['columns'][0] }}{% elif dataset['columns'] | length == 2 %}{{ dataset['columns'][0] }} and {{ dataset['columns'][1] }}{% else %}{{ dataset['columns'][:-1] | join(', ') }}, and {{ dataset['columns'][-1] }}{% endif %}
{% if dataset['stats_table'] %}* Approximate statistics based on the first {{ [dataset['size'], 1000] | min }} samples:
{{ dataset['stats_table'] }}{% endif %}{% if dataset['examples_table'] %}* Samples:
{{ dataset['examples_table'] }}{% endif %}* Loss: {% if dataset["loss"]["fullname"].startswith("sentence_transformers.") %}[<code>{{ dataset["loss"]["fullname"].split(".")[-1] }}</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#{{ dataset["loss"]["fullname"].split(".")[-1].lower() }}){% else %}<code>{{ dataset["loss"]["fullname"] }}</code>{% endif %}{% if "config_code" in dataset["loss"] %} with these parameters:
{{ dataset["loss"]["config_code"] }}{% endif %}
{% if dataset_list | length > 3 %}</details>
{% endif %}{% endfor %}{% endif %}{% endfor -%}

{% if all_hyperparameters %}
### Training Hyperparameters
{% if non_default_hyperparameters -%}
#### Non-Default Hyperparameters

{% for name, value in non_default_hyperparameters.items() %}- `{{ name }}`: {{ value }}
{% endfor %}{%- endif %}
#### All Hyperparameters
<details><summary>Click to expand</summary>

{% for name, value in all_hyperparameters.items() %}- `{{ name }}`: {{ value }}
{% endfor %}
</details>
{% endif %}

{%- if eval_lines %}
### Training Logs
{% if hide_eval_lines %}<details><summary>Click to expand</summary>

{% endif -%}
{{ eval_lines }}{% if explain_bold_in_eval %}
* The bold row denotes the saved checkpoint.{% endif %}
{%- if hide_eval_lines %}
</details>{% endif %}
{% endif %}

## Citation

### BibTeX

```bibtex

```
