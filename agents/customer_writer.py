from langchain_core.messages import HumanMessage, SystemMessage

from state import GraphState
from utils.llm import get_llm


SYSTEM_PROMPT = """You are a release-notes writer for end users.

Audience: everyday product users — NOT developers. They do not have "code", "APIs", "integrations", or "imports". They just use the product.

CRITICAL — NEVER include in your output:
- Ticket IDs (e.g., JIRA-123, #892), commit SHAs (e.g., abc123f), PR numbers
- File names, module names, class names, or function/method names — even inside backticks
- HTTP status codes, URLs, endpoint paths, error types, or internal service names
- Any internal identifier, symbol, or variable name
- Developer instructions such as "update your code/usage", "adjust your calls", "modify your imports", "refactor your integration"

Always write in plain, everyday language — like you are telling a friend what is new in an app they use. Describe what the USER SEES or CAN DO, not how it was built.

CRITICAL:
- Exactly ONE bullet per input item. Do not split one change into multiple bullets.
- Include every input item. Do not drop items.

Good example (bug fix):
  Source evidence: "JIRA-418: API /v1/users returns 500 on empty query - fixed null deref in UserService.search()"
  Output: "Searching for users with an empty search box no longer causes an error."

Bad example — DO NOT do this:
  Output: "Fixed a null dereference in UserService.search() causing a 500 error on /v1/users with empty queries."

Good example (feature):
  Source evidence: "JIRA-421: Add dark mode toggle to settings page"
  Output: "You can now switch to dark mode from the settings page."

Style:
- Plain, everyday language. Short sentences. Friendly tone.
- Each bullet: 1-2 sentences max.

Structure (omit any section with no items):

## What's new ({version_bump})

### New features
- User-facing improvements, described in everyday terms.

### Bug fixes
- What the user will notice is now working better — in plain language.

Rules:
- Only write about the changes provided. Do not invent or speculate.
- Omit empty sections entirely."""


def _format_items(items) -> str:
    by_cat = {"enhancement": [], "bugfix": []}
    for i in items:
        if i.category in by_cat:
            by_cat[i.category].append(i)

    order = [
        ("New features", "enhancement"),
        ("Bug fixes", "bugfix"),
    ]
    lines = []
    for heading, key in order:
        if not by_cat[key]:
            continue
        lines.append(f"{heading}:")
        for i in by_cat[key]:
            lines.append(f"- {i.title}")
            lines.append(f"  Evidence: {i.raw_evidence}")
        lines.append("")
    return "\n".join(lines).strip()


async def customer_writer_node(state: GraphState) -> dict:
    items = state.get("customer_items") or []
    if not items:
        return {"customer_notes": ""}

    version_bump = state.get("version_bump", "patch")
    llm = get_llm()

    user_msg = f"Version bump: {version_bump}\n\n{_format_items(items)}"

    issues = (state.get("hallucination_report") or {}).get("customer") or []
    if issues:
        feedback = "\n".join(f'- "{i["quote"]}" — {i["problem"]}' for i in issues)
        user_msg = (
            "Your previous draft had the following fidelity issues. "
            "Rewrite the customer notes and avoid repeating them:\n"
            f"{feedback}\n\n"
            f"{user_msg}"
        )

    result = await llm.ainvoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]
    )
    return {"customer_notes": result.content}
