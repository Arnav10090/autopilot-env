from __future__ import annotations

from typing import Any, Dict, List

from .judge_types import JudgeInput
from .models import AutopilotAction


def build_judge_input(
    workflow: Dict[str, Any],
    completed_ids: List[str],
    available_ids: List[str],
    pending_ids: List[str],
    tool_summary: Dict[str, int],
    tool_history: List[Dict[str, Any]],
    action: AutopilotAction,
) -> JudgeInput:
    return JudgeInput(
        workflow_id=workflow.get("workflow_id", ""),
        workflow_name=workflow.get("name", ""),
        completed_task_ids=list(completed_ids),
        available_task_ids=list(available_ids),
        pending_task_ids=list(pending_ids),
        tool_summary=dict(tool_summary),
        recent_tool_results=list(tool_history[-3:]),
        action_tool=action.tool,
        action_params=dict(action.params),
        action_reasoning=action.reasoning or "",
    )
