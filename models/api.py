from pydantic import BaseModel, Field
from typing import List, Literal
from uuid import UUID

MAX_INPUT_CHARS = 50_000


class GenerateRequest(BaseModel):
    raw_input: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)


class ResumeItem(BaseModel):
    id: int
    category: Literal["bugfix", "enhancement", "breaking_change", "other"]


class ResumeRequest(BaseModel):
    thread_id: UUID
    items: List[ResumeItem]
