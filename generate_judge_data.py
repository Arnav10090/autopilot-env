# generate_judge_data.py
import json
import sys
sys.path.insert(0, "src")
from envs.autopilot_env.workflows import WORKFLOWS

TOOLS = [
    "hr_create_user", "jira_create_ticket", "jira_assign_ticket",
    "slack_send_message", "slack_create_channel",
    "email_send", "hr_update_user", "calendar_create_event", "done"
]

examples = []

for wf in WORKFLOWS:
    tasks = wf["tasks"]
    completed = []
    tool_summary = {
        "jira_tickets": 0, "hr_users": 0, "slack_channels": 0,
        "slack_messages": 0, "emails_sent": 0,
        "calendar_events": 0, "total_calls": 0
    }

    for i, task in enumerate(tasks):
        available = [
            t["task_id"] for t in tasks
            if t["task_id"] not in completed
            and all(d in completed for d in t.get("dependencies", []))
        ]
        pending = [
            t["task_id"] for t in tasks
            if t["task_id"] not in completed
            and t["task_id"] not in available
        ]

        # --- Good action: correct tool for available task ---
        examples.append({
            "judge_input": {
                "workflow_id": wf["workflow_id"],
                "workflow_name": wf["name"],
                "completed_task_ids": list(completed),
                "available_task_ids": list(available),
                "pending_task_ids": list(pending),
                "tool_summary": dict(tool_summary),
                "recent_tool_results": [],
                "action_tool": task["required_tool"],
                "action_params": {p: "value" for p in task.get("required_params", [])},
                "action_reasoning": f"Completing {task['name']} as it is now available."
            },
            "deterministic_step_reward": 0.50,
            "deterministic_breakdown": {"tool_score": 0.2, "dep_score": 0.1, "param_score": 0.15, "reasoning_score": 0.05},
            "episode_success": 1.0 if i == len(tasks) - 1 else 0.0,
            "completion_rate": (i + 1) / len(tasks)
        })

        # --- Bad action: wrong tool (dependency violation) ---
        wrong_tool = "jira_create_ticket" if task["required_tool"] != "jira_create_ticket" else "email_send"
        examples.append({
            "judge_input": {
                "workflow_id": wf["workflow_id"],
                "workflow_name": wf["name"],
                "completed_task_ids": list(completed),
                "available_task_ids": list(available),
                "pending_task_ids": list(pending),
                "tool_summary": dict(tool_summary),
                "recent_tool_results": [],
                "action_tool": wrong_tool,
                "action_params": {},
                "action_reasoning": ""
            },
            "deterministic_step_reward": -0.30,
            "deterministic_breakdown": {"tool_score": 0.0, "dep_violation": -0.20, "invalid_tool": -0.10},
            "episode_success": 0.0,
            "completion_rate": i / max(len(tasks), 1)
        })

        # --- Bad action: early done ---
        if available:
            examples.append({
                "judge_input": {
                    "workflow_id": wf["workflow_id"],
                    "workflow_name": wf["name"],
                    "completed_task_ids": list(completed),
                    "available_task_ids": list(available),
                    "pending_task_ids": list(pending),
                    "tool_summary": dict(tool_summary),
                    "recent_tool_results": [],
                    "action_tool": "done",
                    "action_params": {},
                    "action_reasoning": ""
                },
                "deterministic_step_reward": -0.80,
                "deterministic_breakdown": {"early_done_penalty": -0.80},
                "episode_success": 0.0,
                "completion_rate": i / max(len(tasks), 1)
            })

        # Advance state
        completed.append(task["task_id"])
        tool = task["required_tool"]
        tool_summary["total_calls"] += 1
        if "jira" in tool:     tool_summary["jira_tickets"] += 1
        if "hr" in tool:       tool_summary["hr_users"] += 1
        if "slack_send" in tool:   tool_summary["slack_messages"] += 1
        if "slack_create" in tool: tool_summary["slack_channels"] += 1
        if "email" in tool:    tool_summary["emails_sent"] += 1
        if "calendar" in tool: tool_summary["calendar_events"] += 1

print(f"Generated {len(examples)} examples")
with open("judge_examples.jsonl", "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")
print("Saved judge_examples.jsonl")