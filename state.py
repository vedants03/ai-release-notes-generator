from typing import Literal, TypedDict, List
from models.schema import ChangeItem


class GraphState(TypedDict):
    raw_input: str

    change_items: List[ChangeItem]

    version_bump: Literal["major", "minor", "patch", "none"]
    customer_items: List[ChangeItem]
    internal_items: List[ChangeItem]

    customer_notes: str
    internal_notes: str

    hallucination_report: dict
    retry_count: int
