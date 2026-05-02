from langgraph.graph import END, START, StateGraph

from agents.classifier import classifier_node
from agents.customer_writer import customer_writer_node
from agents.extractor import extractor_node
from agents.hallucinator import hallucinator_node
from agents.hitl_review import hitl_review_node
from agents.internal_writer import internal_writer_node
from agents.router import router_node
from state import GraphState

MAX_RETRIES = 1


def route_after_router(state: GraphState):
    targets = []
    if state.get("customer_items"):
        targets.append("customer_writer")
    if state.get("internal_items"):
        targets.append("internal_writer")
    return targets or END


def route_after_hallucinator(state: GraphState):
    report = state.get("hallucination_report") or {}
    retry_count = state.get("retry_count", 0)

    has_customer_issues = bool(report.get("customer"))
    has_internal_issues = bool(report.get("internal"))

    if not (has_customer_issues or has_internal_issues):
        return END
    if retry_count > MAX_RETRIES:
        return END

    targets = []
    if has_customer_issues:
        targets.append("customer_writer")
    if has_internal_issues:
        targets.append("internal_writer")
    return targets


def build_graph(checkpointer):
    g = StateGraph(GraphState)

    g.add_node("extractor", extractor_node)
    g.add_node("classifier", classifier_node)
    g.add_node("hitl_review", hitl_review_node)
    g.add_node("router", router_node)
    g.add_node("customer_writer", customer_writer_node)
    g.add_node("internal_writer", internal_writer_node)
    g.add_node("hallucinator", hallucinator_node)

    g.add_edge(START, "extractor")

    # Conditional edge so extractor's Command(goto=END) is respected
    # when no change items are found (e.g. random text input).
    g.add_conditional_edges(
        "extractor",
        lambda state: END if not state.get("change_items") else "classifier",
        ["classifier", END],
    )
    g.add_edge("classifier", "hitl_review")
    g.add_edge("hitl_review", "router")

    g.add_conditional_edges(
        "router",
        route_after_router,
        ["customer_writer", "internal_writer", END],
    )

    g.add_edge("customer_writer", "hallucinator")
    g.add_edge("internal_writer", "hallucinator")

    g.add_conditional_edges(
        "hallucinator",
        route_after_hallucinator,
        ["customer_writer", "internal_writer", END],
    )

    return g.compile(checkpointer=checkpointer)
