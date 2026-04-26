"""
FastAPI server — Adaptive Enterprise Autopilot OpenEnv.

Endpoints:
  GET  /health                     Health check
  POST /reset?task=easy            Start new episode
  POST /step                       Submit tool call action
  GET  /state                      Episode metadata
  GET  /workflow                   Full active workflow definition
  GET  /docs                       Swagger UI
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from envs.autopilot_env.environment import AutopilotEnvironment
from envs.autopilot_env.judge_model import LearnedJudge
from envs.autopilot_env.models import AutopilotAction

DEFAULT_TASK = os.getenv("AUTOPILOT_TASK", "easy")
JUDGE_ENABLED = os.getenv("JUDGE_ENABLED", "0") == "1"
JUDGE_MODEL_PATH = os.getenv("JUDGE_MODEL_PATH", "")
JUDGE_ALPHA = float(os.getenv("JUDGE_ALPHA", "0.05"))

app = FastAPI(
    title="Adaptive Enterprise Autopilot — OpenEnv",
    description=(
        "OpenEnv-compatible environment where an AI agent orchestrates multi-step "
        "enterprise workflows using real tool APIs (Jira, Slack, Email, HR, Calendar). "
        "Difficulty auto-escalates via a self-improvement generator after each episode."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env: Optional[AutopilotEnvironment] = None
_current_task: str = DEFAULT_TASK
_judge: Optional[LearnedJudge] = None
if JUDGE_ENABLED:
    _judge = LearnedJudge(model_path=JUDGE_MODEL_PATH, enabled=True).load()


# ── Request / Response Schemas ────────────────────────────────────────────────

class ActionRequest(BaseModel):
    tool: str = Field(..., description=(
        "Enterprise tool to call. One of: jira_create_ticket, jira_update_ticket, "
        "jira_assign_ticket, slack_send_message, slack_create_channel, email_send, "
        "hr_create_user, hr_update_user, calendar_create_event, done"
    ))
    params: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    reasoning: str = Field("", description="Agent's reasoning for this action (improves reward)")


class ObservationOut(BaseModel):
    workflow_id: str
    workflow_name: str
    workflow_description: str
    tasks: List[Dict[str, Any]]
    completed_task_ids: List[str]
    available_task_ids: List[str]
    pending_task_ids: List[str]
    tool_results: List[Dict[str, Any]]
    available_tools: List[str]
    step_feedback: str
    reward: float
    done: bool
    difficulty_level: int
    metadata: Dict[str, Any]


class StepOut(BaseModel):
    observation: ObservationOut
    reward: float
    done: bool
    info: Dict[str, Any]


class StateOut(BaseModel):
    episode_id: str
    task_name: str
    workflow_id: str
    workflow_name: str
    step_count: int
    total_reward: float
    tasks_completed: int
    tasks_total: int
    dependency_violations: int
    rule_violations: int
    difficulty_level: int
    metadata: Dict[str, Any]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _obs_to_dict(obs) -> Dict[str, Any]:
    return {
        "workflow_id": obs.workflow_id,
        "workflow_name": obs.workflow_name,
        "workflow_description": obs.workflow_description,
        "tasks": obs.tasks,
        "completed_task_ids": obs.completed_task_ids,
        "available_task_ids": obs.available_task_ids,
        "pending_task_ids": obs.pending_task_ids,
        "tool_results": obs.tool_results,
        "available_tools": obs.available_tools,
        "step_feedback": obs.step_feedback,
        "reward": obs.reward,
        "done": obs.done,
        "difficulty_level": obs.difficulty_level,
        "metadata": obs.metadata,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Adaptive Enterprise Autopilot — OpenEnv",
        "version": "1.0.0",
        "endpoints": {
            "GET  /health": "Health check",
            "POST /reset?task=easy|medium|hard": "Start new episode",
            "POST /step": "Submit a tool call action",
            "GET  /state": "Episode metadata and stats",
            "GET  /workflow": "Full active workflow definition",
            "GET  /docs": "Interactive API docs",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "task": _current_task,
        "version": "1.0.0",
    }


@app.post("/reset", response_model=ObservationOut)
def reset(task: Optional[str] = Query(default=None)):
    """Reset the environment. Pass ?task=easy|medium|hard to switch difficulty."""
    global _env, _current_task
    chosen = (task or DEFAULT_TASK).strip().lower()
    if chosen not in ("easy", "medium", "hard"):
        raise HTTPException(400, f"Invalid task {chosen!r}. Use easy, medium, or hard.")
    _current_task = chosen
    _env = AutopilotEnvironment(
        task=chosen,
        learned_judge=_judge if JUDGE_ENABLED else None,
        judge_alpha=JUDGE_ALPHA,
        judge_enabled=JUDGE_ENABLED,
        judge_buffer=None,
    )
    obs = _env.reset()
    return _obs_to_dict(obs)


@app.post("/step", response_model=StepOut)
def step(req: ActionRequest):
    """Submit a tool call. Call /reset first."""
    global _env
    if _env is None or not _env._episode_started:
        raise HTTPException(400, "No active episode — call /reset first.")
    action = AutopilotAction(
        tool=req.tool,
        params=req.params,
        reasoning=req.reasoning,
    )
    obs, reward, done, info = _env.step(action)
    return {
        "observation": _obs_to_dict(obs),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state", response_model=StateOut)
def state():
    """Current episode state."""
    global _env
    if _env is None:
        raise HTTPException(400, "No active episode — call /reset first.")
    s = _env.state
    return {
        "episode_id": s.episode_id,
        "task_name": s.task_name,
        "workflow_id": s.workflow_id,
        "workflow_name": s.workflow_name,
        "step_count": s.step_count,
        "total_reward": round(s.total_reward, 4),
        "tasks_completed": s.tasks_completed,
        "tasks_total": s.tasks_total,
        "dependency_violations": s.dependency_violations,
        "rule_violations": s.rule_violations,
        "difficulty_level": s.difficulty_level,
        "metadata": s.metadata,
    }


@app.get("/workflow")
def workflow():
    """Return the full active workflow definition (useful for debugging)."""
    global _env
    if _env is None or _env._workflow is None:
        raise HTTPException(400, "No active episode — call /reset first.")
    return _env._workflow


@app.get("/diagnostics")
def diagnostics():
    """
    Live reward decomposition for the most recent step.

    Frontend (demo.html) polls this once per /step response so judges can see
    every term in the v2 reward stack contribute in real time.
    """
    env = _env
    last = getattr(env, "_last_reward_breakdown", {}) or {}
    combiner = getattr(env, "_reward_combiner", None)
    intrinsic = getattr(env, "_intrinsic", None)
    return {
        "version": "v2.0",
        "last_step": {
            "extrinsic_step": last.get("extrinsic_step", 0.0),
            "extrinsic_total": last.get("extrinsic_total", 0.0),
            "pbrs_shaping": last.get("pbrs_shaping", 0.0),
            "intrinsic_count": last.get("intrinsic_count", 0.0),
            "intrinsic_rnd": last.get("intrinsic_rnd", 0.0),
            "weighted_judge": last.get("weighted_judge", 0.0),
            "difference_reward": last.get("difference_reward", 0.0),
            "difference_reward_raw": last.get("difference_reward_raw", 0.0),
            "difference_baseline_tool": last.get("difference_baseline_tool", ""),
            "difference_baseline_step_reward": last.get("difference_baseline_step_reward", 0.0),
            "ird_posterior_correction": last.get("ird_posterior_correction", 0.0),
            "ird_proxy_reward": last.get("ird_proxy_reward", 0.0),
            "ird_posterior_expected_reward": last.get("ird_posterior_expected_reward", 0.0),
            "ird_top_hypothesis": last.get("ird_top_hypothesis", ""),
            "ird_posterior": last.get("ird_posterior", {}),
            "phi_before": last.get("phi_before", 0.0),
            "phi_after": last.get("phi_after", 0.0),
            "intrinsic_decay_factor": last.get("intrinsic_decay_factor", 1.0),
            "total": last.get(
                "total",
                last.get("extrinsic_total", 0.0)
                + last.get("pbrs_shaping", 0.0)
                + last.get("intrinsic_count", 0.0),
            ),
        },
        "config": {
            "mode": combiner.mode if combiner else "full",
            "w_extrinsic": combiner.w_extrinsic if combiner else 1.0,
            "w_pbrs": combiner.w_pbrs if combiner else 1.0,
            "w_intrinsic": combiner.w_intrinsic if combiner else 1.0,
            "w_intrinsic_rnd": combiner.w_intrinsic_rnd if combiner else 1.0,
            "w_judge": combiner.w_judge if combiner else 1.0,
            "w_difference": combiner.w_difference if combiner else 0.0,
            "w_ird": combiner.w_ird if combiner else 1.0,
            "episode_idx": intrinsic.episode_idx if intrinsic else 0,
        },
    }


@app.post("/diagnostics/mode")
def diagnostics_set_mode(payload: dict):
    """Live ablation toggle: payload = {'mode': 'full'|'proxy_only'|'no_pbrs'|'no_intrinsic'}."""
    env = _env
    combiner = getattr(env, "_reward_combiner", None)
    if combiner is None:
        return {"ok": False, "error": "RewardCombiner not present"}
    new_mode = (payload or {}).get("mode", "full")
    if new_mode not in {"full", "proxy_only", "no_pbrs", "no_intrinsic"}:
        return {"ok": False, "error": f"unknown mode: {new_mode}"}
    combiner.mode = new_mode
    return {"ok": True, "mode": new_mode}


def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
