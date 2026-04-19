from pydantic import BaseModel
from typing import List, Literal


class ChangeItem(BaseModel):
    id: int
    title: str
    source_refs: List[str]
    raw_evidence: str
    category: Literal["bugfix", "enhancement", "breaking_change", "other"] | None = None
