"""
Workflow Generator — Theme 4 (Self-Improvement) loop.

After each episode, generate_harder_workflow() produces a new workflow that is
slightly harder than the one just completed. This creates an auto-escalating
curriculum without any manual authoring.

Mutation strategies (applied cumulatively based on difficulty delta):
  delta=1   Add 2 notification tasks with cross-dependencies.
  delta=2   +1: Add a verification/audit task.
  delta=3   +2: Introduce a new blocker task.
  delta=4   +3: Add a parallel track (second team), merge at a final task.
  delta=5+  +4: Add strict business-rule constraints with new rule_check hooks.

The difficulty_level integer is used to choose which mutations to apply,
capped at 10 (after which the hardest seed workflow is recycled with minor naming changes).
"""

from __future__ import annotations
import copy
import uuid
from typing import Any, Dict, List, Optional


# ── Public API ────────────────────────────────────────────────────────────────

def generate_harder_workflow(base: Dict[str, Any], delta: int = 1) -> Dict[str, Any]:
    """
    Mutate *base* into a harder variant.

    Parameters
    ----------
    base  : workflow dict (same schema as workflows.py)
    delta : how many difficulty levels to increase (usually 1)

    Returns
    -------
    New workflow dict — does NOT modify base.
    """
    wf = copy.deepcopy(base)
    wf["workflow_id"] = f"W_GEN_{uuid.uuid4().hex[:6].upper()}"
    wf["difficulty_level"] = min(10, base.get("difficulty_level", 1) + delta)
    wf["generated"] = True

    target_level = wf["difficulty_level"]
    if target_level >= 2:
        wf = _add_notification_tasks(wf)
    if target_level >= 3:
        wf = _add_verification_task(wf)
    if target_level >= 5:
        wf = _add_blocker(wf)
    if target_level >= 7:
        wf = _add_parallel_track(wf)
    if target_level >= 9:
        wf = _add_strict_rule(wf)
    if target_level >= 8:
        wf = _add_chaos_mode(wf)

    # Recalculate max_steps
    n = len(wf["tasks"])
    wf["max_steps"] = max(base.get("max_steps", n * 3), n * 3)
    wf["name"] = f"{base['name']} (difficulty {wf['difficulty_level']})"
    wf["description"] = (
        base["description"]
        + f"\n\n[Generated variant — difficulty {wf['difficulty_level']}/10. "
        "Additional tasks and constraints have been added.]"
    )
    return wf

def generate_easier_workflow(workflow: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    De-escalate: remove the most recently added peripheral tasks.
    Returns None if already at minimum difficulty (can't simplify further).
    """
    import copy
    current_level = workflow.get("difficulty_level", 1)
    if current_level <= 1:
        return None  # already at floor

    wf = copy.deepcopy(workflow)
    wf["workflow_id"] = f"W_GEN_{uuid.uuid4().hex[:6].upper()}"
    wf["difficulty_level"] = max(1, current_level - 1)
    wf["generated"] = True

    tasks = wf["tasks"]
    all_ids = {t["task_id"] for t in tasks}
    depended_on = set()
    for t in tasks:
        depended_on.update(t.get("dependencies", []))
    leaves = [t for t in tasks if t["task_id"] not in depended_on]

    # Remove up to 2 leaf tasks (the ones most recently generated)
    # Only remove tasks that were clearly added by the generator
    # (they have generated names or are beyond the seed task count)
    removable = [
        t for t in leaves
        if any(kw in t["name"].lower() for kw in [
            "notification", "compliance", "consolidat", "audit",
            "verification", "secondary", "confirm"
        ])
    ][:2]

    remove_ids = {t["task_id"] for t in removable}
    wf["tasks"] = [t for t in tasks if t["task_id"] not in remove_ids]

    # Clean up any dependencies pointing to removed tasks
    for t in wf["tasks"]:
        t["dependencies"] = [d for d in t.get("dependencies", []) if d not in remove_ids]

    n = len(wf["tasks"])
    wf["max_steps"] = max(n * 3, 10)
    wf["name"] = f"{workflow['name'].split(' (difficulty')[0]} (difficulty {wf['difficulty_level']})"
    wf["description"] = (
        workflow["description"].split("\n\n[Generated")[0]
        + f"\n\n[De-escalated variant — difficulty {wf['difficulty_level']}/10.]"
    )
    return wf


def difficulty_score(workflow: Dict[str, Any]) -> float:
    """
    Compute a normalised difficulty score in [0.0, 1.0] for a workflow.
    Used to reward the generator for producing genuinely hard variants.
    """
    n_tasks = len(workflow["tasks"])
    n_deps = sum(len(t.get("dependencies", [])) for t in workflow["tasks"])
    n_rules = sum(1 for t in workflow["tasks"] if t.get("rule_check"))
    n_blockers = sum(1 for t in workflow["tasks"] if t.get("is_blocker"))
    n_strict = sum(1 for t in workflow["tasks"] if t.get("business_rule"))

    raw = (
        n_tasks * 0.4
        + n_deps * 0.25
        + n_rules * 0.5
        + n_blockers * 1.0
        + n_strict * 0.3
    )
    # Normalise: a perfect hard workflow (13 tasks, 18 deps, 4 rules, 2 blockers) ≈ 15
    return min(1.0, round(raw / 15.0, 3))


# ── Mutation helpers ──────────────────────────────────────────────────────────

def _next_id(tasks: List[Dict]) -> str:
    existing = [t["task_id"] for t in tasks]
    i = len(tasks) + 1
    while f"T{i}" in existing:
        i += 1
    return f"T{i}"


def _last_task_id(tasks: List[Dict]) -> str:
    """Return the task_id of the last leaf (no other task depends on it)."""
    all_ids = {t["task_id"] for t in tasks}
    deps = set()
    for t in tasks:
        deps.update(t.get("dependencies", []))
    leaves = all_ids - deps
    # Return the leaf with the highest numeric index
    def sort_key(tid):
        digits = "".join(c for c in tid if c.isdigit())
        return int(digits) if digits else 0
    return sorted(leaves, key=sort_key)[-1] if leaves else tasks[-1]["task_id"]


def _add_notification_tasks(wf: Dict) -> Dict:
    """
    Add a manager notification (Slack) and a confirmation email after the last leaf task.
    Dependency chain: existing leaf → slack → email.
    """
    leaf = _last_task_id(wf["tasks"])
    t_id_a = _next_id(wf["tasks"])
    wf["tasks"].append({
        "task_id": t_id_a,
        "name": "Notify manager on Slack",
        "description": "Send a completion summary to the manager's Slack channel.",
        "required_tool": "slack_send_message",
        "required_params": ["channel", "message"],
        "dependencies": [leaf],
        "business_rule": None,
        "rule_check": None,
        "is_blocker": False,
        "points": 1.0,
    })
    t_id_b = _next_id(wf["tasks"])
    wf["tasks"].append({
        "task_id": t_id_b,
        "name": "Send confirmation email",
        "description": "Email the requestor confirming the workflow is complete.",
        "required_tool": "email_send",
        "required_params": ["to", "subject", "body"],
        "dependencies": [t_id_a],
        "business_rule": None,
        "rule_check": None,
        "is_blocker": False,
        "points": 1.0,
    })
    return wf


def _add_verification_task(wf: Dict) -> Dict:
    """
    Add an audit/verification Jira ticket that depends on ALL current leaves.
    Forces the agent to complete everything before the audit.
    """
    all_ids = {t["task_id"] for t in wf["tasks"]}
    depended_on = set()
    for t in wf["tasks"]:
        depended_on.update(t.get("dependencies", []))
    current_leaves = list(all_ids - depended_on)

    t_id = _next_id(wf["tasks"])
    wf["tasks"].append({
        "task_id": t_id,
        "name": "Create audit/verification ticket",
        "description": "Open a Jira ticket confirming all workflow steps have been reviewed.",
        "required_tool": "jira_create_ticket",
        "required_params": ["summary", "issue_type"],
        "dependencies": current_leaves,
        "business_rule": "All prior steps must be complete before audit.",
        "rule_check": "ticket_exists",
        "is_blocker": False,
        "points": 1.5,
    })
    return wf


def _add_blocker(wf: Dict) -> Dict:
    """
    Convert the highest-points non-blocker task into a blocker.
    First call will fail; agent must retry the same tool.
    """
    candidates = [
        t for t in wf["tasks"]
        if not t.get("is_blocker") and t.get("points", 1.0) >= 1.0
    ]
    if not candidates:
        return wf
    # Pick the task with highest points that isn't the first one
    target = sorted(candidates, key=lambda x: -x.get("points", 1.0))[0]
    for t in wf["tasks"]:
        if t["task_id"] == target["task_id"]:
            t["is_blocker"] = True
            break
    return wf


def _add_parallel_track(wf: Dict) -> Dict:
    """
    Add a second parallel track (3 tasks) that must all complete before
    a new merge/consolidation task. Models a second team working in parallel.
    """
    # Parallel track starts independently
    t_a = _next_id(wf["tasks"])
    wf["tasks"].append({
        "task_id": t_a,
        "name": "Create secondary team HR record",
        "description": "Register the secondary team lead in the HR system.",
        "required_tool": "hr_create_user",
        "required_params": ["name", "role", "department"],
        "dependencies": [],
        "business_rule": None,
        "rule_check": None,
        "is_blocker": False,
        "points": 1.0,
    })
    t_b = _next_id(wf["tasks"])
    wf["tasks"].append({
        "task_id": t_b,
        "name": "Brief secondary team on Slack",
        "description": "Message the secondary team channel with task overview.",
        "required_tool": "slack_send_message",
        "required_params": ["channel", "message"],
        "dependencies": [t_a],
        "business_rule": "Secondary team must be registered in HR before briefing.",
        "rule_check": "hr_first",
        "is_blocker": False,
        "points": 1.0,
    })

    # Merge task: depends on current leaf + parallel track
    primary_leaf = _last_task_id([t for t in wf["tasks"] if t["task_id"] not in {t_a, t_b}])
    t_merge = _next_id(wf["tasks"])
    wf["tasks"].append({
        "task_id": t_merge,
        "name": "Consolidate and schedule all-hands",
        "description": "Schedule a calendar meeting merging both team tracks for final review.",
        "required_tool": "calendar_create_event",
        "required_params": ["title", "attendees", "date"],
        "dependencies": [primary_leaf, t_b],
        "business_rule": "Both primary and secondary tracks must complete before consolidation.",
        "rule_check": None,
        "is_blocker": False,
        "points": 2.0,
    })
    return wf


def _add_strict_rule(wf: Dict) -> Dict:
    """
    Upgrade the existing rule_check on a task from None to 'hr_first',
    and add a new high-stakes email task with a rule dependency.
    """
    # Promote one non-rule task to have a rule
    for t in wf["tasks"]:
        if t.get("rule_check") is None and t["required_tool"] != "hr_create_user":
            t["rule_check"] = "hr_first"
            t["business_rule"] = "Must verify HR records exist before proceeding."
            break

    # Add a final compliance email
    leaf = _last_task_id(wf["tasks"])
    t_id = _next_id(wf["tasks"])
    wf["tasks"].append({
        "task_id": t_id,
        "name": "Send compliance confirmation email",
        "description": "Email the compliance team confirming all rules were followed.",
        "required_tool": "email_send",
        "required_params": ["to", "subject", "body"],
        "dependencies": [leaf],
        "business_rule": "Compliance email must be the FINAL action after everything else.",
        "rule_check": None,
        "is_blocker": False,
        "points": 2.0,
    })
    return wf

def _add_chaos_mode(wf: Dict) -> Dict:
    """
    Difficulty 8+: two APIs simultaneously degraded.
    Converts the two highest-points non-blocker tasks into blockers.
    Adds a description note that both must be retried.
    """
    candidates = [
        t for t in wf["tasks"]
        if not t.get("is_blocker") and t.get("points", 1.0) >= 1.0
    ]
    if len(candidates) < 2:
        return wf

    targets = sorted(candidates, key=lambda x: -x.get("points", 1.0))[:2]
    target_ids = {t["task_id"] for t in targets}

    for t in wf["tasks"]:
        if t["task_id"] in target_ids:
            t["is_blocker"] = True
            t["business_rule"] = (
                (t.get("business_rule") or "") +
                " [CHAOS MODE: API degraded — first call will fail. Retry required.]"
            ).strip()

    wf["description"] += (
        "\n\n⚠ CHAOS MODE ACTIVE: Two enterprise APIs are simultaneously degraded. "
        "The agent must detect failures via tool_results and retry both."
    )
    return wf
