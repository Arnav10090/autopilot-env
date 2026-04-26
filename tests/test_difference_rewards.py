from __future__ import annotations

from src.envs.autopilot_env.difference_rewards import compute_difference_reward
from src.envs.autopilot_env.models import AutopilotAction


def _workflow():
    return {
        "workflow_id": "wf_test",
        "tasks": [
            {
                "task_id": "T1",
                "name": "Create Jira Ticket",
                "required_tool": "jira_create_ticket",
                "required_params": ["title"],
                "dependencies": [],
            },
            {
                "task_id": "T2",
                "name": "Assign Ticket",
                "required_tool": "jira_assign_ticket",
                "required_params": ["ticket_id", "assignee"],
                "dependencies": ["T1"],
            },
        ],
    }


def test_difference_reward_positive_when_action_beats_baseline():
    workflow = _workflow()
    action = AutopilotAction(
        tool="jira_create_ticket",
        params={"title": "Investigate billing issue"},
        reasoning="Create Jira Ticket to start the workflow",
    )

    diff, meta = compute_difference_reward(
        action,
        workflow,
        completed_ids=[],
        tool_registry_summary={},
    )

    assert diff > 0.0
    assert meta["baseline_tool"] == "done"
    assert meta["baseline_step_reward"] == -0.1


def test_difference_reward_non_positive_for_poor_action():
    workflow = _workflow()
    action = AutopilotAction(
        tool="totally_unknown_tool",
        params={},
        reasoning="",
    )

    diff, _ = compute_difference_reward(
        action,
        workflow,
        completed_ids=[],
        tool_registry_summary={},
    )

    assert diff <= 0.0


def test_difference_reward_is_deterministic():
    workflow = _workflow()
    action = AutopilotAction(
        tool="jira_create_ticket",
        params={"title": "Investigate billing issue"},
        reasoning="Create Jira Ticket to start the workflow",
    )

    diff1, meta1 = compute_difference_reward(
        action,
        workflow,
        completed_ids=[],
        tool_registry_summary={},
    )
    diff2, meta2 = compute_difference_reward(
        action,
        workflow,
        completed_ids=[],
        tool_registry_summary={},
    )

    assert diff1 == diff2
    assert meta1["baseline_step_reward"] == meta2["baseline_step_reward"]
    assert meta1["baseline_tool"] == meta2["baseline_tool"]
