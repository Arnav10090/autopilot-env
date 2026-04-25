from __future__ import annotations

import os
import pickle
from typing import Dict

from .judge_types import JudgeInput, JudgePrediction


def _featurize(judge_input: JudgeInput) -> Dict[str, float]:
    return {
        "calendar_events": float(judge_input.tool_summary.get("calendar_events", 0)),
        "emails_sent": float(judge_input.tool_summary.get("emails_sent", 0)),
        "has_reasoning": 1.0 if (judge_input.action_reasoning or "").strip() else 0.0,
        "hr_users": float(judge_input.tool_summary.get("hr_users", 0)),
        "jira_tickets": float(judge_input.tool_summary.get("jira_tickets", 0)),
        "n_available": float(len(judge_input.available_task_ids)),
        "n_completed": float(len(judge_input.completed_task_ids)),
        "n_params": float(len(judge_input.action_params)),
        "n_pending": float(len(judge_input.pending_task_ids)),
        "n_recent_results": float(len(judge_input.recent_tool_results)),
        "total_calls": float(judge_input.tool_summary.get("total_calls", 0)),
    }


class LearnedJudge:
    def __init__(self, model_path: str = "", enabled: bool = False):
        self.model_path = model_path
        self._enabled = enabled
        self.model = None

    def load(self) -> "LearnedJudge":
        if self.model_path and os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
        return self

    def enabled(self) -> bool:
        return self._enabled and self.model is not None

    def score(self, judge_input: JudgeInput) -> JudgePrediction:
        if not self.enabled():
            return JudgePrediction(score=0.0, components={}, confidence=0.0)

        feats = _featurize(judge_input)
        values = [feats[key] for key in sorted(feats.keys())]
        raw = float(self.model.predict([values])[0])
        clipped = max(-1.0, min(1.0, raw))
        return JudgePrediction(
            score=clipped,
            components={"soft_quality": clipped},
            confidence=0.5,
        )
