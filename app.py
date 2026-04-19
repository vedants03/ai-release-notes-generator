import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from langgraph.graph.state import CompiledStateGraph

from agents.graph import build_graph
from models.api import GenerateRequest, ResumeRequest
from utils.tracing import get_langfuse_handler

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DB = str(BASE_DIR / "checkpoints.db")
STATIC_DIR = str(BASE_DIR / "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.langfuse_handler = get_langfuse_handler()
    print(
        f"[startup] Langfuse handler: "
        f"{'enabled' if app.state.langfuse_handler else 'DISABLED (no keys)'}"
    )
    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        app.state.graph = build_graph(checkpointer)
        yield
    if app.state.langfuse_handler:
        try:
            from langfuse import get_client
            get_client().flush()
        except Exception as e:
            print(f"[shutdown] Langfuse flush failed: {e}")


def _build_config(app: FastAPI, thread_id: str) -> dict:
    config: dict = {"configurable": {"thread_id": thread_id}}
    handler = getattr(app.state, "langfuse_handler", None)
    if handler:
        config["callbacks"] = [handler]
        config["metadata"] = {"langfuse_session_id": thread_id}
    return config


app = FastAPI(title="AI Release Notes Generator", lifespan=lifespan)


async def stream_graph(graph: CompiledStateGraph, input_or_command: dict, config: dict):
    thread_id = config["configurable"]["thread_id"]
    try:
        async for event in graph.astream(input_or_command, config, stream_mode="updates"):
            for node, update in event.items():
                if node == "__interrupt__":
                    payload = update[0].value if isinstance(update, tuple) else update
                    yield (
                        json.dumps(
                            {
                                "type": "interrupt",
                                "thread_id": thread_id,
                                "payload": payload,
                            }
                        )
                        + "\n"
                    )
                else:
                    yield json.dumps({"type": "node_completed", "node": node}) + "\n"

        snapshot = await graph.aget_state(config)
        if not snapshot.next:
            yield (
                json.dumps(
                    {
                        "type": "result",
                        "customer_notes": snapshot.values.get("customer_notes"),
                        "internal_notes": snapshot.values.get("internal_notes"),
                        "version_bump": snapshot.values.get("version_bump"),
                        "hallucination_report": snapshot.values.get("hallucination_report"),
                    }
                )
                + "\n"
            )
    except Exception as e:
        yield json.dumps(
            {
                "type": "error",
                "thread_id": thread_id,
                "error_type": type(e).__name__,
                "message": str(e),
            }
        ) + "\n"


@app.post("/api/generate")
async def generate(req: GenerateRequest, request: Request):
    graph = request.app.state.graph
    thread_id = str(uuid.uuid4())
    config = _build_config(request.app, thread_id)
    return StreamingResponse(
        stream_graph(graph, {"raw_input": req.raw_input}, config),
        media_type="application/x-ndjson",
    )


@app.post("/api/resume")
async def resume(req: ResumeRequest, request: Request):
    graph = request.app.state.graph
    config = _build_config(request.app, str(req.thread_id))
    edits = [item.model_dump() for item in req.items]
    return StreamingResponse(
        stream_graph(graph, Command(resume=edits), config),
        media_type="application/x-ndjson",
    )


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
