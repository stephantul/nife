from typing import TypedDict


class VocabItem(TypedDict):
    token: str
    frequency: int
    document_frequency: int
