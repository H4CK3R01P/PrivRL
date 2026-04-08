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
    seed: int = 42


env_instance = None


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

env = PrivRLEnv()
@app.post("/reset")
def reset(request: Optional[ResetRequest] = None):
    global env_instance

    task_id = request.task_id if request else "easy"
    seed = request.seed if request else 42

    env_instance = PrivRLEnv()
    obs = env_instance.reset(task_id=task_id, seed=seed)

    return obs


@app.post("/step")
def step(action: PrivRLAction):
    global env_instance

    if env_instance is None:
        return {"error": "No active episode. Call /reset first."}

    obs, reward, done, info = env_instance.step(action)

    response = {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info,
    }

    # Include score when episode ends (Phase 2 requirement)
    if done:
        response["score"] = env_instance.get_normalized_score()

    return response


@app.get("/state")
async def get_state(episode_id: str = ""):
    """Get the current state of an episode."""
    if episode_id not in environments:
        return {"error": f"No active episode with id '{episode_id}'."}
    env = environments[episode_id]
    return {"state": env.state.model_dump()}


@app.post("/grade")
def grade():
    """Grade the current episode. Returns score in (0, 1)."""
    global env_instance
    if env_instance is None:
        return {"error": "No active episode. Call /reset first."}
    return env_instance.grade()


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

import uvicorn

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
