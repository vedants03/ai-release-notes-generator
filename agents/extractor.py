import re
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from langgraph.graph import END
from langgraph.types import Command
from models.schema import ChangeItem
from state import GraphState
from utils.llm import get_llm, ainvoke_with_retry


class _ExtractedChange(BaseModel):
    title: str
    source_refs: List[str]
    raw_evidence: str


class _ExtractedChanges(BaseModel):
    items: List[_ExtractedChange]


SYSTEM_PROMPT = """You are a release-notes extractor. Given a blob of mixed engineering artifacts (JIRA tickets, commits, pull requests, free-form notes), extract a deduplicated list of distinct changes.

CRITICAL — DO NOT FABRICATE:
- Do NOT invent JIRA numbers, PR numbers, commit SHAs, or any other identifier. Every value in `source_refs` MUST appear verbatim in the input text.
- Do NOT invent change titles or evidence. Every title and evidence snippet must come directly from text that is actually present in the input.
- Do NOT infer changes from product names, technology names, or marketing copy. If the input only mentions "LangGraph" or "FastAPI" without describing a concrete change, that is NOT a change — return an empty list.
- If the input is a description, README blurb, question, or anything other than a list of engineering artifacts, return an empty list.
- Better to return an empty list than to invent a change.

Rules:
- Each item represents ONE logical change. If a JIRA ticket, a PR, and commits all describe the same change, merge them into a single item.
- Include ALL changes that shipped: features, bug fixes, breaking changes, AND internal work (refactors, dependency bumps, docs updates, tests, CI). Chores and internal-only work are valid change items — do not filter them out.
- title: a short, clear description of the change (8-12 words), paraphrased from the actual input text.
- source_refs: stable identifiers found verbatim in the input (e.g., "JIRA-421", "PR #892", "abc123f"). If no identifier is present for an item, use an empty list — do NOT invent one.
- raw_evidence: a short verbatim snippet from the input that justifies this item (1-2 sentences max).
- Do NOT classify or judge the change; just extract."""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _verify_items(
    items: List[_ExtractedChange], raw_input: str
) -> List[_ExtractedChange]:
    """Drop any item whose evidence or source_refs don't actually appear in raw_input."""
    norm_input = _normalize(raw_input)
    verified = []
    for item in items:
        evidence = _normalize(item.raw_evidence)
        if evidence and evidence not in norm_input:
            print(f"[extractor] dropped item (evidence not in input): {item.title!r}")
            continue
        if any(_normalize(ref) not in norm_input for ref in item.source_refs):
            print(
                f"[extractor] dropped item (ref not in input): {item.title!r} refs={item.source_refs}"
            )
            continue
        verified.append(item)
    return verified


async def extractor_node(state: GraphState) -> dict:
    raw_input = state["raw_input"]
    llm = get_llm()
    structured = llm.with_structured_output(
        _ExtractedChanges, method="function_calling"
    )

    result: _ExtractedChanges = await ainvoke_with_retry(
        structured,
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=raw_input),
        ],
    )

    verified = _verify_items(result.items, raw_input)

    change_items = [
        ChangeItem(
            id=i + 1,
            title=c.title,
            source_refs=c.source_refs,
            raw_evidence=c.raw_evidence,
            category=None,
        )
        for i, c in enumerate(verified)
    ]

    if not change_items:
        return Command(
            goto=END,
            update={
                "change_items": [],
                "customer_items": [],
                "internal_items": [],
                "customer_notes": "",
                "internal_notes": "",
                "version_bump": "none",
                "hallucination_report": {"customer": [], "internal": []},
            },
        )

    return {"change_items": change_items}
