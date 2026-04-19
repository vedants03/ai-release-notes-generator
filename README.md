# AI Release Notes Generator

A web app that takes pasted engineering artifacts (Jira tickets, PRs, commits, free-form notes) and produces two audience-specific release notes:

- **Customer notes** — plain, non-technical language for end users.
- **Internal notes** — technical tone with source references for engineers.

Built with LangGraph (orchestration), FastAPI (backend), Langfuse (tracing), and Groq (Llama 3.3 via OpenAI-compatible API). Minimal HTML/JS frontend. Deployed on Azure Web App for Containers.

---

## Input assumptions

- Single text blob pasted into the UI or sent as `raw_input` to `/api/generate`.
- Max **50,000 characters** (enforced at the API layer).
- Can freely mix formats: JIRA tickets, PR descriptions, commit lines, informal notes.
- Multiple artifacts referring to the same change (ticket + PR + commits) get deduplicated into a single change item.
- No external fetches — the app works only from text you paste. It does not call GitHub, JIRA, or any other system.

---

## Example

**Input:**
```
JIRA-421: Add dark mode toggle to settings page
JIRA-418: API /v1/users returns 500 on empty query - fixed null deref in UserService.search()
PR #892: BREAKING - Rename getUser to fetchUser across SDK, consumers must update imports
abc123f chore: bump eslint to 9.0
```

**Customer notes (end users):**
```markdown
## What's new (major)

### New features
- You can now switch to dark mode from the settings page.

### Bug fixes
- Searching for users with an empty search box no longer causes an error.
```

**Internal notes (engineers):**
```markdown
## Internal release notes (major)

### Breaking changes
- The `getUser` function has been renamed to `fetchUser` across the SDK (PR #892).

### Enhancements
- A dark mode toggle has been added to the settings page (JIRA-421).

### Bug fixes
- A null dereference in `UserService.search()` that returned 500 on empty queries has been fixed (JIRA-418).

### Other
- Bumped `eslint` to 9.0 (abc123f).
```

Breaking changes appear only in internal notes (by design — end users see product impact, not API mechanics). The version bump (`major` here, driven by the presence of a breaking change) is reported on both.

---

## Setup (local)

```
python -m venv .venv
.venv\Scripts\activate     # or source .venv/bin/activate on Unix
pip install -r requirements.txt
copy .env.example .env     # or cp on Unix
```

Fill in `.env`:
```
LLM_API_KEY=<your Groq key>
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile

# optional — tracing
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
```

Run:
```
uvicorn app:app --reload
```

Open http://localhost:8000.

---

## Deployment (Azure Web App for Containers)

```
# Build
docker build -t release-notes .

# Push to Azure Container Registry (ACR must already exist)
az acr login --name <acr-name>
docker tag release-notes <acr-name>.azurecr.io/release-notes:v1
docker push <acr-name>.azurecr.io/release-notes:v1
```

Then in the Azure Portal:
1. Create a **Web App for Containers** pointing at the ACR image.
2. Configure **Application settings** with the same environment variables as `.env`.
3. Enable system-assigned managed identity and grant it `AcrPull` on the registry (so Azure can pull without credentials).

---

## Design

Pipeline of 7 nodes, mostly stateless LLM calls plus one deterministic router and one human-in-the-loop pause.

```
                       raw_input
                           │
                           ▼
                   ┌───────────────┐
                   │   extractor   │   LLM  · parse + dedupe into ChangeItems
                   └───────┬───────┘
                           ▼
                   ┌───────────────┐
                   │   classifier  │   LLM  · assign category per item
                   └───────┬───────┘
                           ▼
                   ┌───────────────┐
                   │  hitl_review  │   ⏸   · user overrides categories
                   └───────┬───────┘
                           ▼
                   ┌───────────────┐
                   │     router    │   Python · version bump + audience split
                   └───────┬───────┘
                           │
                ┌──────────┴──────────┐          parallel fan-out
                ▼                     ▼
        ┌───────────────┐     ┌───────────────┐
        │   customer    │     │   internal    │   both LLM
        │    writer     │     │    writer     │
        └───────┬───────┘     └───────┬───────┘
                └──────────┬──────────┘
                           ▼
                   ┌───────────────┐
                   │  hallucinator │   LLM  · verify prose vs. source evidence
                   └───────┬───────┘
                           │
            ┌──────────────┼──────────────┐
         clean     flagged + retry ≤ 1    flagged + retry > 1
            │              │                     │
            ▼              ▼                     ▼
           END   re-run writer(s) with         END
                   feedback, then back
                   to hallucinator
```

**State is typed** (`GraphState` TypedDict) and persists via a SQLite checkpointer, which is what makes the HITL interrupt / resume work across two HTTP requests.

**Hallucination retry** is capped at one round: writers get their flagged quotes injected as feedback on the second pass. If issues persist they ship to the UI as warnings rather than silently hiding.

**Audience routing** is deterministic, derived from category:

| category | customer notes | internal notes |
|---|---|---|
| bugfix | ✓ | ✓ |
| enhancement | ✓ | ✓ |
| breaking_change | ✗ | ✓ |
| other | ✗ | ✓ |

---

## AI tools & references

Built with [Claude Code](https://claude.com/claude-code) as pair-programmer — design discussions, iterative prompt refinement.

Documentation referenced:

- **LangGraph**
  - [Human-in-the-loop](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/)
  - [Streaming](https://langchain-ai.github.io/langgraph/how-tos/streaming/) (for NDJSON from FastAPI endpoints)
  - [Checkpointers & persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/) (SQLite saver, thread_id)
  - [State & reducers](https://langchain-ai.github.io/langgraph/concepts/low_level/)
- **LangChain**
  - [RunnableConfig](https://python.langchain.com/docs/concepts/runnables/) (`configurable`, `callbacks`, `metadata`)
  - [Structured output](https://python.langchain.com/docs/how_to/structured_output/) (used for extractor, classifier, hallucinator)
- **Langfuse**
  - [LangChain integration](https://langfuse.com/docs/integrations/langchain/tracing) (`CallbackHandler`, session grouping via metadata)
- **FastAPI**
  - [Lifespan events](https://fastapi.tiangolo.com/advanced/events/) (for graph + checkpointer setup)
  - [StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse) (for NDJSON output)
- **Azure**
  - [Web App for Containers](https://learn.microsoft.com/en-us/azure/app-service/quickstart-custom-container)
  - [Pull from ACR with managed identity](https://learn.microsoft.com/en-us/azure/app-service/configure-custom-container)
- **Groq**
  - [Supported models for structured outputs](https://console.groq.com/docs/structured-outputs#supported-models)
