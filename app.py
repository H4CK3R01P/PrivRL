"""
PrivRL FastAPI Server — OpenEnv-compatible HTTP/WebSocket API.

Exposes the PrivRL environment via REST endpoints:
    POST /reset   — Start a new episode
    POST /step    — Take an action
    GET  /state   — Get current state
    GET  /health  — Health check
    GET  /tasks   — List available tasks
"""

import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional

from env.environment import PrivRLEnv
from env.models import PrivRLAction


# =============================================================================
# Request/Response Models
# =============================================================================

class ResetRequest(BaseModel):
    task_id: str = "easy"
    episode_id: Optional[str] = None


class StepRequest(BaseModel):
    classification: str


# =============================================================================
# App Lifecycle
# =============================================================================

# Global environment instances (per session)
environments: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown."""
    yield
    environments.clear()


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="PrivRL — Privacy Risk Classification Environment",
    description="OpenEnv RL environment where agents learn to classify website privacy risk.",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# REST Endpoints
# =============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "environment": "PrivRL", "version": "1.0.0"}


@app.get("/tasks")
async def list_tasks():
    """List all available tasks."""
    return {"tasks": PrivRLEnv.list_tasks()}


@app.post("/reset")
async def reset(request: ResetRequest):
    """Reset environment for a new episode."""
    env = PrivRLEnv()
    episode_id = request.episode_id or str(uuid.uuid4())
    try:
        obs = env.reset(task_id=request.task_id, episode_id=episode_id)
    except ValueError as e:
        return {"error": str(e)}
    environments[episode_id] = env
    return {
        "episode_id": episode_id,
        "observation": obs.model_dump(),
    }


@app.post("/step")
async def step(request: StepRequest, episode_id: str = ""):
    """Execute a step in the environment."""
    if episode_id not in environments:
        return {"error": f"No active episode with id '{episode_id}'. Call /reset first."}

    env = environments[episode_id]
    action = PrivRLAction(classification=request.classification)
    obs, reward, done, info = env.step(action)

    response = {
        "episode_id": episode_id,
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }

    # Include final score if episode is done
    if done:
        response["final_score"] = info.get("final_score", env.get_normalized_score())
        del environments[episode_id]

    return response


@app.get("/state")
async def get_state(episode_id: str = ""):
    """Get the current state of an episode."""
    if episode_id not in environments:
        return {"error": f"No active episode with id '{episode_id}'."}
    env = environments[episode_id]
    return {"state": env.state.model_dump()}


# =============================================================================
# WebSocket Endpoint (OpenEnv standard)
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for persistent session communication."""
    await websocket.accept()
    env = PrivRLEnv()
    episode_id = None

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            method = message.get("method", "")

            if method == "reset":
                task_id = message.get("task_id", "easy")
                episode_id = message.get("episode_id", str(uuid.uuid4()))
                obs = env.reset(task_id=task_id, episode_id=episode_id)
                await websocket.send_json({
                    "method": "reset",
                    "episode_id": episode_id,
                    "observation": obs.model_dump(),
                })

            elif method == "step":
                classification = message.get("classification", "")
                action = PrivRLAction(classification=classification)
                obs, reward, done, info = env.step(action)
                response = {
                    "method": "step",
                    "episode_id": episode_id,
                    "observation": obs.model_dump(),
                    "reward": reward,
                    "done": done,
                    "info": info,
                }
                if done:
                    response["final_score"] = info.get("final_score", env.get_normalized_score())
                await websocket.send_json(response)

            elif method == "state":
                await websocket.send_json({
                    "method": "state",
                    "state": env.state.model_dump(),
                })

            else:
                await websocket.send_json({
                    "error": f"Unknown method '{method}'. Use 'reset', 'step', or 'state'."
                })

    except WebSocketDisconnect:
        pass


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
