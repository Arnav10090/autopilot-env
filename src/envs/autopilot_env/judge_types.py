from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class JudgeInput:
    workflow_id: str
    workflow_name: str
    completed_task_ids: List[str]
    available_task_ids: List[str]
    pending_task_ids: List[str]
    tool_summary: Dict[str, int]
    recent_tool_results: List[Dict[str, Any]]
    action_tool: str
    action_params: Dict[str, Any]
    action_reasoning: str


@dataclass
class JudgePrediction:
    score: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class JudgeExample:
    judge_input: JudgeInput
    deterministic_step_reward: float
    deterministic_breakdown: Dict[str, Any]
    episode_success: float
    completion_rate: float
