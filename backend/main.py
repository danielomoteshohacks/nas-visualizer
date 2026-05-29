# main.py
#
# FastAPI application — the HTTP + WebSocket server that connects
# the ML backend to the React frontend.
#
# Endpoints:
#
#   POST /api/search/start
#       Accept search config, create a run, start it in background thread.
#       Returns run_id immediately — frontend uses this to subscribe.
#
#   GET  /api/search/{run_id}
#       Return current status and results of a run.
#
#   GET  /api/search/{run_id}/export
#       Generate and return PyTorch code for the best architecture.
#
#   GET  /api/runs
#       List all runs.
#
#   WS   /ws/{run_id}
#       WebSocket endpoint — streams live progress events to the frontend.
#       The frontend connects here immediately after POST /api/search/start.

import asyncio
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.search.runner import create_run, start_run, get_run, list_runs
from backend.export.codegen import generate_code
from backend.bench.database import init_db, get_stats


# -----------------------------------------------------------------------
# Startup
# -----------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the database when the server starts."""
    init_db()
    yield


app = FastAPI(
    title       = 'NAS Visualizer API',
    description = 'Neural Architecture Search with real-time visualization',
    version     = '1.0.0',
    lifespan    = lifespan,
)

# Allow the React dev server (port 5173) to call this API.
# CORS = Cross-Origin Resource Sharing — browsers block requests between
# different origins (ports count as different origins) unless the server
# explicitly allows it.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ['http://localhost:5173', 'http://localhost:3000'],
    allow_credentials = True,
    allow_methods     = ['*'],
    allow_headers     = ['*'],
)


# -----------------------------------------------------------------------
# Request / Response models
# Pydantic validates incoming JSON automatically and gives clear errors
# when required fields are missing or have wrong types.
# -----------------------------------------------------------------------

class SearchConfig(BaseModel):
    strategy:        str = Field('darts', pattern='^(darts|evolutionary|random)$')
    proxy_budget:    int = Field(500,  ge=50,  le=2000)
    training_budget: int = Field(20,   ge=5,   le=100)
    seed:            int = Field(42,   ge=0)


# -----------------------------------------------------------------------
# WebSocket connection manager
#
# Keeps track of all open WebSocket connections per run_id.
# When a search emits an event, we broadcast it to every connected client.
# This means multiple browser tabs watching the same run all get updates.
# -----------------------------------------------------------------------

class ConnectionManager:

    def __init__(self):
        # run_id → list of active WebSocket connections
        self.connections: dict[str, list[WebSocket]] = {}

    async def connect(self, run_id: str, ws: WebSocket):
        await ws.accept()
        self.connections.setdefault(run_id, []).append(ws)

    def disconnect(self, run_id: str, ws: WebSocket):
        if run_id in self.connections:
            self.connections[run_id] = [
                c for c in self.connections[run_id] if c is not ws
            ]

    async def broadcast(self, run_id: str, data: dict):
        """Send an event to all clients watching this run."""
        dead = []
        for ws in self.connections.get(run_id, []):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(run_id, ws)


manager = ConnectionManager()


# -----------------------------------------------------------------------
# REST endpoints
# -----------------------------------------------------------------------

@app.get('/api/health')
async def health():
    """Simple health check — confirms the server is running."""
    return {
        'status': 'ok',
        'time':   time.time(),
        'db':     get_stats(),
    }


@app.post('/api/search/start')
async def start_search(config: SearchConfig):
    """
    Create and start a new NAS search run.

    Returns the run_id immediately. The frontend should then open
    a WebSocket connection to /ws/{run_id} to receive live updates.

    The actual search runs in a background thread — this endpoint
    returns in milliseconds regardless of search duration.
    """
    run = create_run(config.model_dump())

    # Queue to pass events from the background thread to async WebSocket
    # The search thread puts events here; the WebSocket coroutine reads them.
    event_queue: asyncio.Queue = asyncio.Queue()

    # Get a reference to the running event loop so the background thread
    # can safely schedule coroutines on it.
    loop = asyncio.get_event_loop()

    def on_event(event: dict):
        """Called by the search thread for every progress event."""
        asyncio.run_coroutine_threadsafe(
            event_queue.put(event), loop
        )

    # Store the queue on the run object so the WebSocket handler can find it
    run._event_queue = event_queue

    start_run(run, callback=on_event)

    return {
        'run_id':  run.run_id,
        'status':  run.status,
        'message': 'Search started. Connect to /ws/{run_id} for live updates.',
    }


@app.get('/api/search/{run_id}')
async def get_search(run_id: str):
    """Return current status and results of a search run."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f'Run {run_id} not found')
    return run.to_dict()


@app.get('/api/search/{run_id}/export')
async def export_architecture(run_id: str):
    """
    Generate and return PyTorch code for the best architecture found.

    Returns a plain text response containing a complete .py file
    that the user can save and run to train the architecture themselves.
    """
    from fastapi.responses import PlainTextResponse

    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f'Run {run_id} not found')
    if run.status != 'complete':
        raise HTTPException(status_code=400, detail='Search not complete yet')

    results   = run.results
    arch_idx  = results.get('best_arch_index')

    if arch_idx is None:
        raise HTTPException(status_code=400, detail='No best architecture found')

    code = generate_code(
        arch_index   = arch_idx,
        val_accuracy = results.get('best_val_accuracy', 0.0),
        extrapolated = results.get('best_extrapolated_acc', 0.0),
        params       = 0,
        flops        = 0.0,
    )

    return PlainTextResponse(
        content  = code,
        media_type = 'text/plain',
        headers  = {
            'Content-Disposition': f'attachment; filename="nas_arch_{arch_idx}.py"'
        },
    )


@app.get('/api/runs')
async def get_all_runs():
    """List all search runs with their current status."""
    return {'runs': list_runs()}


# -----------------------------------------------------------------------
# WebSocket endpoint
# -----------------------------------------------------------------------

@app.websocket('/ws/{run_id}')
async def websocket_endpoint(ws: WebSocket, run_id: str):
    """
    Real-time event stream for a search run.

    The frontend connects here immediately after starting a search.
    This coroutine:
      1. Sends all events that already happened (replay buffer)
         so a late-connecting client catches up instantly
      2. Then streams new events as they arrive from the search thread

    Why asyncio.Queue?
    The search runs in a regular Python thread (not async).
    WebSockets require async code. The Queue is the bridge:
    the thread puts events in, the async coroutine takes them out
    and sends them over the WebSocket without blocking.
    """
    run = get_run(run_id)
    if not run:
        await ws.close(code=4004)
        return

    await manager.connect(run_id, ws)

    try:
        # Replay all events that happened before the client connected
        # (important if the browser was slow to open the WebSocket)
        for past_event in run.events:
            await ws.send_json(past_event)

        # Stream new events until the search completes or client disconnects
        queue = getattr(run, '_event_queue', None)

        while run.status not in ('complete', 'error'):
            if queue:
                try:
                    # Wait up to 1 second for a new event
                    # The timeout prevents blocking forever if events are slow
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    await ws.send_json(event)
                except asyncio.TimeoutError:
                    # No event yet — send a heartbeat so the connection stays alive
                    await ws.send_json({'phase': 'heartbeat', 'run_id': run_id})
            else:
                await asyncio.sleep(0.5)

        # Send any remaining events that arrived after the loop condition
        # was evaluated (race condition safety)
        if queue:
            while not queue.empty():
                event = queue.get_nowait()
                await ws.send_json(event)

        # Send the final status so the frontend knows we're done
        await ws.send_json({
            'phase':   'stream_end',
            'run_id':  run_id,
            'status':  run.status,
            'results': run.results,
        })

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(run_id, ws)