import os
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import agents
import tools

# --- Session Store ---

# session_id -> list of {"role": "user"|"assistant", "content": ...}. In-memory, single process.
sessions: dict[str, list] = {}
# session_id -> the Trace of the answer being worked on right now, so the page can poll /progress
live: dict[str, agents.Trace] = {}

# --- FastAPI App ---

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[dict]  # every tool call by every agent; each has an "agent" field
    slam_ids: list[str] = []  # the Slams the answer is about, for the draw chart


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Get or create the session
    session_id = request.session_id or str(uuid.uuid4())
    history = sessions.setdefault(session_id, [])

    # Append user's message to the context
    history += [{"role": "user", "content": request.message}]

    live[session_id] = trace = agents.Trace()
    try:
        response, tool_calls, slam_ids = agents.answer(history, trace)
    except Exception as e:
        # Auth, billing, a model that is not running: show it in the chat, not as a 500.
        response, tool_calls, slam_ids = f"Model call failed: {type(e).__name__}: {str(e)[:300]}", trace.calls, []
    finally:
        live.pop(session_id, None)

    history += [{"role": "assistant", "content": response}]
    return ChatResponse(response=response, session_id=session_id, tool_calls=tool_calls, slam_ids=slam_ids)


@app.get("/progress/{session_id}")
def progress(session_id: str):
    """Tool calls finished so far in the answer being worked on (the page polls this)."""
    trace = live.get(session_id)
    return {"running": trace is not None,
            "calls": [{k: c[k] for k in ("agent", "name", "args", "start_s", "end_s")} for c in trace.calls] if trace else []}


@app.post("/clear")
def clear(session_id: str | None = None):
    sessions.pop(session_id, None)
    return {"status": "ok"}


# --- Data endpoints for the front end's draw chart (no model involved) ---


@app.get("/api/slams")
def slams():
    return tools.list_slams()


@app.get("/api/draw/{slam_id}")
def draw(slam_id: str):
    rows = tools._query(
        """SELECT slot, quarter, player, seed, entry, rank, elo_rank_in_field, luck_wins, exp_wins_real,
                  exp_wins_random, p_W_real, p_W_random, actual_result
           FROM draw WHERE slam_id = ? ORDER BY slot""", [slam_id])
    if not rows:
        raise HTTPException(404, f"Unknown slam_id {slam_id}")
    return {"slam": tools._slam(slam_id), "players": rows}


if __name__ == "__main__":
    # Cloud Run sets PORT and needs 0.0.0.0; locally this is http://localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
