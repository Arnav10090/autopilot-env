"""
Deterministic Reward Function — Adaptive Enterprise Autopilot.

Grading is split into two levels:

  Step-level  (called after every tool call)
  ─────────────────────────────────────────
  +0.20   correct tool for a ready task
  +0.15   all required params present and non-empty
  +0.10   dependencies satisfied at time of call
  +0.05   reasoning string mentions the task name or tool name
  −0.20   dependency violated (called tool when deps not satisfied)
  −0.25   business rule violated
  −0.10   unknown or "done" called when tasks remain
  −0.05   tool call fails due to missing required params

  Episode-level  (added when done=True or max_steps reached)
  ──────────────────────────────────────────────────────────
  +1.00   all tasks completed
  +0.60   ≥80% tasks completed (partial)
  +0.30   ≥50% tasks completed (partial)
  +0.20   efficiency bonus: completed with ≤ (n_tasks × 1.5) steps

All values are clamped to [0.0, 1.0] per step. Episode score is in [0.0, 2.0]
(exceeds 1.0 on perfect runs — intentional, signals excellence).
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from .models import AutopilotAction


# ── Step-level reward ─────────────────────────────────────────────────────────

def grade_step(
    action: AutopilotAction,
    workflow: Dict[str, Any],
    completed_ids: List[str],
    tool_registry_summary: Dict[str, int],
) -> Tuple[float, Dict[str, Any]]:
    """
    Grade a single tool call action.

    Returns
    -------
    reward    : float in [-0.55, 0.50]
    breakdown : dict with per-component scores and which task was matched
    """

    tasks = workflow["tasks"]
    completed_set = set(completed_ids)

    # ── Find the best-matching task for this tool call ────────────────
    matched_task, dep_ok = _match_task(action.tool, tasks, completed_set)

    breakdown: Dict[str, Any] = {
        "matched_task": matched_task["task_id"] if matched_task else None,
        "task_name": matched_task["name"] if matched_task else None,
        "tool_score": 0.0,
        "param_score": 0.0,
        "dep_score": 0.0,
        "reasoning_score": 0.0,
        "dep_violation": 0.0,
        "rule_violation": 0.0,
        "invalid_tool": 0.0,
        "failed_call": 0.0,
    }

    # ── No matching task found ────────────────────────────────────────
    if matched_task is None:
        if action.tool == "done" and len(completed_set) < len(tasks):
            breakdown["invalid_tool"] = -0.10
            return max(-0.55, -0.10), breakdown
        if action.tool == "done" and len(completed_set) == len(tasks):
            breakdown["tool_score"] = 0.20
            return 0.20, breakdown
        breakdown["invalid_tool"] = -0.10
        return -0.10, breakdown

    # ── Correct tool ──────────────────────────────────────────────────
    breakdown["tool_score"] = 0.20

    # ── Dependencies ──────────────────────────────────────────────────
    if not dep_ok:
        breakdown["dep_violation"] = -0.20
        breakdown["dep_score"] = -0.20
    else:
        breakdown["dep_score"] = 0.10

    # ── Business rule ─────────────────────────────────────────────────
    rule_ok = _check_business_rule(matched_task, tool_registry_summary, completed_set)
    if not rule_ok:
        breakdown["rule_violation"] = -0.25

    # ── Required params present ───────────────────────────────────────
    required = matched_task.get("required_params", [])
    if required:
        present = [r for r in required if action.params.get(r)]
        if len(present) == len(required):
            breakdown["param_score"] = 0.15
        elif len(present) > 0:
            breakdown["param_score"] = 0.15 * (len(present) / len(required))
        else:
            breakdown["failed_call"] = -0.05

    # ── Reasoning quality ─────────────────────────────────────────────
    if action.reasoning:
        task_name_lower = matched_task["name"].lower()
        reasoning_lower = action.reasoning.lower()
        if task_name_lower in reasoning_lower or action.tool in reasoning_lower:
            breakdown["reasoning_score"] = 0.05

    # ── Sum ───────────────────────────────────────────────────────────
    reward = (
        breakdown["tool_score"]
        + breakdown["param_score"]
        + breakdown["dep_score"]
        + breakdown["reasoning_score"]
        + breakdown["dep_violation"]
        + breakdown["rule_violation"]
        + breakdown["failed_call"]
    )
    reward = max(-0.55, min(0.50, reward))
    return round(reward, 4), breakdown


def _match_task(
    tool: str,
    tasks: List[Dict],
    completed_set: set,
) -> Tuple[Optional[Dict], bool]:
    """
    Find the best task this tool call maps to.

    Priority:
    1. Uncompleted task requiring this tool with deps satisfied  → (task, dep_ok=True)
    2. Uncompleted task requiring this tool with deps NOT satisfied → (task, dep_ok=False)
    3. No match → (None, False)
    """
    for t in tasks:
        if t["task_id"] in completed_set:
            continue
        if t["required_tool"] == tool:
            deps_satisfied = all(d in completed_set for d in t["dependencies"])
            if deps_satisfied:
                return t, True

    # Second pass: wrong order
    for t in tasks:
        if t["task_id"] in completed_set:
            continue
        if t["required_tool"] == tool:
            return t, False

    return None, False


def _check_business_rule(
    task: Dict,
    tool_summary: Dict[str, int],
    completed_set: set,
) -> bool:
    """
    Run the rule_check hook for a task.

    hr_first     : At least one HR user must be in the registry.
    ticket_exists: At least one Jira ticket must exist.
    None         : Always passes.
    """
    rule = task.get("rule_check")
    if rule is None:
        return True
    if rule == "hr_first":
        return tool_summary.get("hr_users", 0) > 0
    if rule == "ticket_exists":
        return tool_summary.get("jira_tickets", 0) > 0
    return True


# ── Episode-level reward ──────────────────────────────────────────────────────

def grade_episode(
    workflow: Dict[str, Any],
    completed_ids: List[str],
    step_count: int,
    dependency_violations: int,
    rule_violations: int,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute the end-of-episode reward bonus.

    Returns
    -------
    bonus     : float
    breakdown : dict
    """
    n_total = len(workflow["tasks"])
    n_done = len(completed_ids)
    completion_rate = n_done / n_total if n_total > 0 else 0.0
    max_steps = workflow.get("max_steps", n_total * 3)

    bonus = 0.0
    b: Dict[str, Any] = {
        "completion_rate": round(completion_rate, 3),
        "completion_bonus": 0.0,
        "efficiency_bonus": 0.0,
        "violation_penalty": 0.0,
    }

    if completion_rate == 1.0:
        b["completion_bonus"] = 1.00
    elif completion_rate >= 0.8:
        b["completion_bonus"] = 0.60
    elif completion_rate >= 0.5:
        b["completion_bonus"] = 0.30

    # Efficiency: bonus if completed in ≤ 1.5× number of tasks
    efficiency_threshold = n_total * 1.5
    if completion_rate == 1.0 and step_count <= efficiency_threshold:
        b["efficiency_bonus"] = 0.20

    # Violation penalty
    total_violations = dependency_violations + rule_violations
    b["violation_penalty"] = round(-0.10 * total_violations, 2)

    bonus = b["completion_bonus"] + b["efficiency_bonus"] + b["violation_penalty"]
    b["total_bonus"] = round(bonus, 4)
    return round(bonus, 4), b


# ── Blocker handling ──────────────────────────────────────────────────────────

def is_blocker_task(task: Dict, first_attempt_ids: set) -> bool:
    """
    Return True if this task is a blocker AND this is the agent's first attempt.
    On the second attempt, the blocker resolves (simulating retry / auth refresh).
    """
    return task.get("is_blocker", False) and task["task_id"] not in first_attempt_ids


# ── Convenience: mark task complete ──────────────────────────────────────────

def resolve_task(
    action: AutopilotAction,
    workflow: Dict[str, Any],
    completed_ids: List[str],
    attempted_blocker_ids: set,
) -> Optional[str]:
    """
    If the action matches a ready task that isn't a blocker (or is on retry),
    return the task_id to mark as complete. Otherwise return None.
    """
    completed_set = set(completed_ids)
    tasks = workflow["tasks"]
    for t in tasks:
        if t["task_id"] in completed_set:
            continue
        if t["required_tool"] != action.tool:
            continue
        deps_ok = all(d in completed_set for d in t["dependencies"])
        if not deps_ok:
            continue
        if is_blocker_task(t, attempted_blocker_ids):
            attempted_blocker_ids.add(t["task_id"])
            return None  # tool "fails" — agent must retry
        return t["task_id"]
    return None
