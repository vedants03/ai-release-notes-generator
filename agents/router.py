from state import GraphState

CUSTOMER_VISIBLE = {"bugfix", "enhancement"}


def router_node(state: GraphState) -> dict:
    items = state["change_items"]
    categories = {item.category for item in items}

    if "breaking_change" in categories:
        bump = "major"
    elif "enhancement" in categories:
        bump = "minor"
    elif "bugfix" in categories:
        bump = "patch"
    else:
        bump = "none"

    customer_items = [i for i in items if i.category in CUSTOMER_VISIBLE]
    internal_items = list(items)

    return {
        "version_bump": bump,
        "customer_items": customer_items,
        "internal_items": internal_items,
    }
