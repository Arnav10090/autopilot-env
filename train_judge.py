from __future__ import annotations

import json
import os
import pickle
from typing import Dict, List

from sklearn.ensemble import RandomForestRegressor


def featurize(example: Dict) -> List[float]:
    judge_input = example["judge_input"]
    tool_summary = judge_input.get("tool_summary", {})
    return [
        float(len(judge_input.get("completed_task_ids", []))),
        float(len(judge_input.get("available_task_ids", []))),
        float(len(judge_input.get("pending_task_ids", []))),
        float(len(judge_input.get("recent_tool_results", []))),
        1.0 if (judge_input.get("action_reasoning", "") or "").strip() else 0.0,
        float(len(judge_input.get("action_params", {}))),
        float(tool_summary.get("total_calls", 0)),
        float(tool_summary.get("jira_tickets", 0)),
        float(tool_summary.get("hr_users", 0)),
        float(tool_summary.get("emails_sent", 0)),
        float(tool_summary.get("calendar_events", 0)),
    ]


def target(example: Dict) -> float:
    step_reward = float(example.get("deterministic_step_reward", 0.0))
    episode_success = float(example.get("episode_success", 0.0))
    completion_rate = float(example.get("completion_rate", 0.0))
    return step_reward + (0.2 * episode_success) + (0.1 * completion_rate)


def main():
    data_path = os.getenv("JUDGE_LOG_PATH", "judge_examples.jsonl")
    out_path = os.getenv("JUDGE_MODEL_PATH", "judge_model.pkl")

    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    x_train = [featurize(row) for row in rows]
    y_train = [target(row) for row in rows]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=6,
    )
    model.fit(x_train, y_train)

    with open(out_path, "wb") as f:
        pickle.dump(model, f)

    print(f"[judge] Trained on {len(rows)} examples -> {out_path}")


if __name__ == "__main__":
    main()
