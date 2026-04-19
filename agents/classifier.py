from typing import List, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from state import GraphState
from utils.llm import get_llm


class _Classification(BaseModel):
    id: int
    category: Literal["bugfix", "enhancement", "breaking_change", "other"]


class _Classifications(BaseModel):
    items: List[_Classification]


SYSTEM_PROMPT = """You are a release-notes classifier. For each change item, assign exactly one category:

- bugfix: fixes a defect in existing behavior.
- enhancement: new feature or non-breaking improvement to existing behavior.
- breaking_change: requires consumers to update code, config, or migrations (renames, removed APIs, behavior changes that break backward compatibility).
- other: internal work not directly user-facing (refactor, chore, dependency bumps, docs, tests, CI).

Rules:
- Decide from the item's title and evidence.
- The `id` field is an INTEGER provided in the input (e.g. id=1, id=2). Echo it back as-is. Do NOT use ticket numbers, PR numbers, or commit SHAs as the id.
- When an item plausibly fits more than one category, prefer (in this order): breaking_change, bugfix, enhancement, other.
- Return a classification for EVERY input item — no omissions."""


def _format_items(items) -> str:
    lines = []
    for item in items:
        refs = ", ".join(item.source_refs) if item.source_refs else "no refs"
        lines.append(f"id={item.id} | refs: {refs} | title: {item.title}")
        lines.append(f"  evidence: {item.raw_evidence}")
    return "\n".join(lines)


async def classifier_node(state: GraphState) -> dict:
    items = state["change_items"]
    if not items:
        return {"change_items": []}

    llm = get_llm()
    structured = llm.with_structured_output(_Classifications, method="function_calling")

    result: _Classifications = await structured.ainvoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=_format_items(items)),
        ]
    )

    cat_map = {c.id: c.category for c in result.items}
    for item in items:
        if item.id in cat_map:
            item.category = cat_map[item.id]

    return {"change_items": items}
