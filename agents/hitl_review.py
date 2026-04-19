from langgraph.types import interrupt
from state import GraphState


async def hitl_review_node(state: GraphState) -> dict:
    edits = interrupt(
        {"change_items": [item.model_dump() for item in state["change_items"]]}
    )
    items = state["change_items"]
    valid_ids = {item.id for item in items}

    unknown = [e["id"] for e in edits if e["id"] not in valid_ids]
    if unknown:
        raise ValueError(
            f"Resume payload references unknown change_item ids: {unknown}. "
            f"Valid ids: {sorted(valid_ids)}"
        )

    edit_map = {e["id"]: e["category"] for e in edits}
    for item in items:
        if item.id in edit_map:
            item.category = edit_map[item.id]
    return {"change_items": items}
