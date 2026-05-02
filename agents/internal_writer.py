from langchain_core.messages import HumanMessage, SystemMessage

from state import GraphState
from utils.llm import get_llm, ainvoke_with_retry


SYSTEM_PROMPT = """You are a release-notes writer for an engineering audience.

Audience: engineers on the team and technical stakeholders. They want precise, technical descriptions of what shipped.

Style:
- Concise, factual, technical. No marketing language.
- Include source references (ticket IDs, PR numbers, commit SHAs) in parentheses at the end of each bullet where available.
- Mention function names, modules, or internal symbols when they appear in the evidence — engineers need this detail.
- Each bullet: 1-2 sentences.

CRITICAL:
- Exactly ONE bullet per input item. Do not split an item into a "what it is" bullet and a "what it means" bullet. Combine everything about one item into a single bullet.
- Include EVERY input item. Do not drop items. If an item belongs to the "Other" category, render it under the Other section.

Structure (omit any section with no items):

## Internal release notes ({version_bump})

### Breaking changes
- Describe the change precisely, including migration notes when evident.

### Enhancements
- New capabilities or improvements.

### Bug fixes
- What was broken, root cause if stated in the evidence, what was fixed.

### Other
- Refactors, chores, dependency bumps, docs, tests, CI, etc.

If breaking changes exist, place that section FIRST.

Rules:
- Only write about the changes provided. Do not invent facts, root causes, or migration steps that are not in the evidence.
- Omit empty sections entirely."""


def _format_items(items) -> str:
    by_cat = {
        "breaking_change": [],
        "enhancement": [],
        "bugfix": [],
        "other": [],
    }
    for i in items:
        if i.category in by_cat:
            by_cat[i.category].append(i)

    order = [
        ("Breaking changes", "breaking_change"),
        ("Enhancements", "enhancement"),
        ("Bug fixes", "bugfix"),
        ("Other", "other"),
    ]
    lines = []
    for heading, key in order:
        if not by_cat[key]:
            continue
        lines.append(f"{heading}:")
        for i in by_cat[key]:
            refs = ", ".join(i.source_refs) if i.source_refs else "no refs"
            lines.append(f"- {i.title} [refs: {refs}]")
            lines.append(f"  Evidence: {i.raw_evidence}")
        lines.append("")
    return "\n".join(lines).strip()


async def internal_writer_node(state: GraphState) -> dict:
    items = state.get("internal_items") or []
    if not items:
        return {"internal_notes": ""}

    version_bump = state.get("version_bump", "patch")
    llm = get_llm()

    user_msg = f"Version bump: {version_bump}\n\n{_format_items(items)}"

    issues = (state.get("hallucination_report") or {}).get("internal") or []
    if issues:
        feedback = "\n".join(f'- "{i["quote"]}" — {i["problem"]}' for i in issues)
        user_msg = (
            "Your previous draft had the following fidelity issues. "
            "Rewrite the internal notes and avoid repeating them:\n"
            f"{feedback}\n\n"
            f"{user_msg}"
        )

    result = await ainvoke_with_retry(
        llm,
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ],
    )
    return {"internal_notes": result.content}
