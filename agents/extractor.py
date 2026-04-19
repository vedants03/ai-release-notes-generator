from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from models.schema import ChangeItem
from state import GraphState
from utils.llm import get_llm


class _ExtractedChange(BaseModel):
    title: str
    source_refs: List[str]
    raw_evidence: str


class _ExtractedChanges(BaseModel):
    items: List[_ExtractedChange]


SYSTEM_PROMPT = """You are a release-notes extractor. Given a blob of mixed engineering artifacts (JIRA tickets, commits, pull requests, free-form notes), extract a deduplicated list of distinct changes.

Rules:
- Each item represents ONE logical change. If a JIRA ticket, a PR, and commits all describe the same change, merge them into a single item.
- Include ALL changes that shipped: features, bug fixes, breaking changes, AND internal work (refactors, dependency bumps, docs updates, tests, CI). Chores and internal-only work are valid change items — do not filter them out.
- title: a short, clear description of the change (8-12 words).
- source_refs: stable identifiers found in the input (e.g., "JIRA-421", "PR #892", "abc123f"). Include every identifier that refers to this same change.
- raw_evidence: a short verbatim snippet from the input that justifies this item (1-2 sentences max).
- Do NOT classify or judge the change; just extract.
- If the input contains no discrete changes, return an empty list."""


async def extractor_node(state: GraphState) -> dict:
    llm = get_llm()
    structured = llm.with_structured_output(_ExtractedChanges, method="function_calling")

    result: _ExtractedChanges = await structured.ainvoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=state["raw_input"]),
        ]
    )

    change_items = [
        ChangeItem(
            id=i + 1,
            title=c.title,
            source_refs=c.source_refs,
            raw_evidence=c.raw_evidence,
            category=None,
        )
        for i, c in enumerate(result.items)
    ]
    return {"change_items": change_items}
