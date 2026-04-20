"""
Adaptive Enterprise Autopilot — Type-safe models.

Action:      A single tool call the agent makes.
Observation: Current workflow state the agent sees.
State:       Full episode metadata for the grader.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AutopilotAction:
    """
    One tool invocation by the agent.

    Fields
    ------
    tool : str
        Name of the enterprise tool to call.
        One of VALID_TOOLS or "done" to signal workflow completion.
    params : Dict[str, Any]
        Tool-specific parameters (see tools.py for each tool's schema).
    reasoning : str
        Agent's chain-of-thought explaining why it's calling this tool now.
        Used for partial reward on reasoning quality.
    """

    tool: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""

    VALID_TOOLS: List[str] = field(default_factory=lambda: [
        "jira_create_ticket",
        "jira_update_ticket",
        "jira_assign_ticket",
        "slack_send_message",
        "slack_create_channel",
        "email_send",
        "hr_create_user",
        "hr_update_user",
        "calendar_create_event",
        "done",
    ])


@dataclass
class AutopilotObservation:
    """
    What the agent sees at each step.

    Fields
    ------
    workflow_id           : Unique workflow identifier.
    workflow_name         : Human-readable workflow name.
    workflow_description  : Full description of what must be accomplished.
    tasks                 : All tasks in the workflow (name, description, dependencies).
    completed_task_ids    : IDs of tasks already finished this episode.
    available_task_ids    : IDs whose dependencies are satisfied — agent CAN work on these.
    pending_task_ids      : IDs still blocked by unsatisfied dependencies.
    tool_results          : Results of every tool call so far (tool, params, result, success).
    available_tools       : Tools the agent may call.
    step_feedback         : Natural-language feedback from the previous action.
    reward                : Reward received for the last action.
    done                  : Whether the episode is over.
    difficulty_level      : Current workflow difficulty (1–10).
    metadata              : Extra context (e.g., business rules summary).
    """

    workflow_id: str = ""
    workflow_name: str = ""
    workflow_description: str = ""
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    completed_task_ids: List[str] = field(default_factory=list)
    available_task_ids: List[str] = field(default_factory=list)
    pending_task_ids: List[str] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    available_tools: List[str] = field(default_factory=lambda: [
        "jira_create_ticket", "jira_update_ticket", "jira_assign_ticket",
        "slack_send_message", "slack_create_channel",
        "email_send", "hr_create_user", "hr_update_user",
        "calendar_create_event", "done",
    ])
    step_feedback: str = ""
    reward: float = 0.0
    done: bool = False
    difficulty_level: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AutopilotState:
    """
    Full episode state — used internally by the environment.

    Fields
    ------
    episode_id              : UUID for this episode.
    task_name               : Difficulty level ("easy" | "medium" | "hard").
    workflow_id             : Active workflow ID.
    workflow_name           : Active workflow name.
    step_count              : Steps taken so far.
    total_reward            : Cumulative reward this episode.
    tasks_completed         : Count of finished tasks.
    tasks_total             : Total tasks in the workflow.
    dependency_violations   : Times agent called a tool with unmet dependencies.
    rule_violations         : Times agent violated a business rule.
    difficulty_level        : Current workflow difficulty integer.
    generated_next_workflow : Dict of the auto-generated next workflow (T4 loop).
    metadata                : Anything else.
    """

    episode_id: str = ""
    task_name: str = "easy"
    workflow_id: str = ""
    workflow_name: str = ""
    step_count: int = 0
    total_reward: float = 0.0
    tasks_completed: int = 0
    tasks_total: int = 0
    dependency_violations: int = 0
    rule_violations: int = 0
    difficulty_level: int = 1
    generated_next_workflow: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
