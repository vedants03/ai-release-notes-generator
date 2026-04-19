from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from state import GraphState
from utils.llm import get_llm


class _Issue(BaseModel):
    quote: str
    problem: str


class _Report(BaseModel):
    customer_issues: List[_Issue]
    internal_issues: List[_Issue]


SYSTEM_PROMPT = """You are a fidelity checker for release notes. Compare generated notes against the source change items and flag claims that are NOT supported by the evidence.

What COUNTS as a hallucination:
- Invented facts: root causes, migration steps, dates, counts, or features that are not in any item's title or evidence.
- Wrong attribution: saying something was "fixed" when the evidence only says "added", or vice versa.
- Specific technical details (e.g., "null pointer dereference in UserService") that are not in the evidence.

What does NOT count as a hallucination (do NOT flag):
- Paraphrasing or reframing technical details as user-impact (especially in customer notes).
- Omitting items (omission is not hallucination).
- Stylistic tightening, word choice, or tone adjustments.
- Adding generic statements like "ensuring a more stable experience" as long as they are consistent with a bug fix.

For each audience (customer / internal), return a list of issues. Each issue has:
- quote: a verbatim fragment from the generated notes that is not grounded
- problem: one sentence explaining why it is not supported by the evidence

Return empty lists if the notes are faithful."""


def _format_source(items) -> str:
    lines = []
    for i in items:
        refs = ", ".join(i.source_refs) if i.source_refs else "no refs"
        lines.append(f"Item {i.id} [{i.category}] ({refs}): {i.title}")
        lines.append(f"  Evidence: {i.raw_evidence}")
    return "\n".join(lines)


async def hallucinator_node(state: GraphState) -> dict:
    items = state["change_items"]
    customer_notes = state.get("customer_notes") or ""
    internal_notes = state.get("internal_notes") or ""

    llm = get_llm()
    structured = llm.with_structured_output(_Report, method="function_calling")

    user_msg = (
        f"SOURCE CHANGE ITEMS:\n{_format_source(items)}\n\n"
        f"CUSTOMER NOTES:\n{customer_notes or '(empty)'}\n\n"
        f"INTERNAL NOTES:\n{internal_notes or '(empty)'}"
    )

    result: _Report = await structured.ainvoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]
    )

    report = {
        "customer": [i.model_dump() for i in result.customer_issues],
        "internal": [i.model_dump() for i in result.internal_issues],
    }

    has_issues = bool(report["customer"]) or bool(report["internal"])
    retry_count = state.get("retry_count", 0)
    new_retry_count = retry_count + 1 if has_issues else retry_count

    return {"hallucination_report": report, "retry_count": new_retry_count}
