install:
	uv sync --all-extras
	uv run pre-commit install

test:
	uv run pytest --cov=nife --cov-report=term-missing
