"""
Deterministic difference rewards for step-level credit assignment.

The first shipped baseline is intentionally simple and cheap:

- compare the actual action's step reward against a counterfactual "done" action
- evaluate both in the same pre-action state
- return the delta as the raw difference reward

This keeps the computation deterministic and side-effect free while giving a
useful signal for "was this action better than simply stopping now?".
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .grader import grade_step
from .models import AutopilotAction


def default_baseline_action(
    workflow: Dict[str, Any],
    completed_ids: List[str],
) -> AutopilotAction:
    """
    Return the default counterfactual action for the current state.

    The baseline is always `"done"` in the first implementation. It is cheap,
    deterministic, and gives a stable "do nothing further" comparison.
    """
    return AutopilotAction(tool="done", params={}, reasoning="")


def compute_difference_reward(
    action: AutopilotAction,
    workflow: Dict[str, Any],
    completed_ids: List[str],
    tool_registry_summary: Dict[str, int],
    *,
    actual_step_reward: Optional[float] = None,
    baseline_action: Optional[AutopilotAction] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compare the actual action against a deterministic counterfactual baseline.

    Returns
    -------
    difference_reward : float
        `actual_step_reward - baseline_step_reward`
    metadata : dict
        Includes baseline action details for diagnostics.
    """
    if actual_step_reward is None:
        actual_step_reward, _ = grade_step(
            action,
            workflow,
            completed_ids,
            tool_registry_summary,
        )

    baseline = baseline_action or default_baseline_action(workflow, completed_ids)
    baseline_step_reward, baseline_breakdown = grade_step(
        baseline,
        workflow,
        completed_ids,
        tool_registry_summary,
    )
    diff = round(float(actual_step_reward) - float(baseline_step_reward), 4)
    metadata = {
        "baseline_tool": baseline.tool,
        "baseline_step_reward": round(float(baseline_step_reward), 4),
        "baseline_breakdown": baseline_breakdown,
    }
    return diff, metadata
