"""
Lightweight IRD-style posterior correction for proxy reward misspecification.

This is intentionally not a research-grade IRD implementation. Instead, it
maintains a tiny posterior over a few interpretable reward hypotheses and uses
that posterior to produce a bounded correction on top of the deterministic
proxy reward.

The goal is practical: surface a non-zero "proxy vs likely true reward" signal
without adding heavy inference machinery or expensive rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from typing import Dict, Iterable, List, Mapping, Tuple


MAX_ABS_CORRECTION = 0.3


@dataclass(frozen=True)
class RewardHypothesis:
    """One simple candidate for what the true reward might prioritize."""

    name: str
    prior_logit: float
    reward_weights: Mapping[str, float] = field(default_factory=dict)
    context_weights: Mapping[str, float] = field(default_factory=dict)


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _softmax(logits: Iterable[float]) -> List[float]:
    values = list(logits)
    if not values:
        return []
    pivot = max(values)
    exps = [exp(v - pivot) for v in values]
    total = sum(exps) or 1.0
    return [v / total for v in exps]


class IRDPosterior:
    """
    Small posterior over reward hypotheses.

    Each hypothesis scores the same context slightly differently. The posterior
    tilts toward completion-oriented or safety-oriented interpretations when the
    current step provides evidence that the hand-written proxy may be missing
    part of the real objective.
    """

    def __init__(self, max_abs_correction: float = MAX_ABS_CORRECTION):
        self.max_abs_correction = float(max_abs_correction)
        self._hypotheses: Tuple[RewardHypothesis, ...] = (
            RewardHypothesis(
                name="proxy_faithful",
                prior_logit=0.0,
                reward_weights={
                    "proxy_reward": 1.00,
                    "progress_delta": 0.03,
                    "availability_gain": 0.02,
                    "remaining_fraction": -0.02,
                    "violation_flag": -0.04,
                    "premature_done": -0.06,
                    "stalled_step": -0.03,
                    "episode_success": 0.04,
                    "failed_end": -0.05,
                },
                context_weights={
                    "progress_signal": 0.08,
                    "violation_signal": -0.04,
                    "near_completion": 0.04,
                    "premature_done_signal": -0.08,
                },
            ),
            RewardHypothesis(
                name="completion_first",
                prior_logit=0.0,
                reward_weights={
                    "proxy_reward": 0.88,
                    "progress_delta": 0.55,
                    "availability_gain": 0.12,
                    "remaining_fraction": -0.02,
                    "violation_flag": -0.08,
                    "premature_done": -0.16,
                    "stalled_step": -0.06,
                    "episode_success": 0.32,
                    "failed_end": -0.06,
                },
                context_weights={
                    "progress_signal": 0.32,
                    "near_completion": 0.28,
                    "violation_signal": -0.10,
                    "premature_done_signal": -0.20,
                    "failed_end_signal": -0.08,
                },
            ),
            RewardHypothesis(
                name="safety_first",
                prior_logit=-0.05,
                reward_weights={
                    "proxy_reward": 0.82,
                    "progress_delta": 0.10,
                    "availability_gain": 0.04,
                    "remaining_fraction": -0.03,
                    "violation_flag": -0.24,
                    "premature_done": -0.24,
                    "stalled_step": -0.10,
                    "episode_success": 0.08,
                    "failed_end": -0.12,
                },
                context_weights={
                    "violation_signal": 0.36,
                    "premature_done_signal": 0.30,
                    "failed_end_signal": 0.26,
                    "stalled_signal": 0.12,
                    "progress_signal": -0.04,
                },
            ),
        )

    def correction(
        self,
        *,
        proxy_reward: float,
        step_breakdown: Mapping[str, float],
        action_tool: str,
        total_tasks: int,
        completed_before: int,
        completed_after: int,
        available_before: int,
        available_after: int,
        episode_done: bool,
        episode_success: bool,
    ) -> Tuple[float, Dict[str, object]]:
        total = max(1, int(total_tasks))
        completion_rate = completed_after / total
        progress_delta = max(0, completed_after - completed_before) / total
        availability_gain = max(0, available_after - available_before) / total

        violation_flag = 1.0 if (
            step_breakdown.get("dep_violation", 0.0) < 0.0
            or step_breakdown.get("rule_violation", 0.0) < 0.0
        ) else 0.0
        premature_done = 1.0 if action_tool == "done" and completed_after < total else 0.0
        stalled_step = 1.0 if (
            action_tool != "done"
            and completed_after == completed_before
            and not violation_flag
        ) else 0.0
        failed_end = 1.0 if episode_done and not episode_success else 0.0

        reward_features = {
            "proxy_reward": float(proxy_reward),
            "progress_delta": progress_delta,
            "availability_gain": availability_gain,
            "remaining_fraction": 1.0 - completion_rate,
            "violation_flag": violation_flag,
            "premature_done": premature_done,
            "stalled_step": stalled_step,
            "episode_success": 1.0 if episode_success else 0.0,
            "failed_end": failed_end,
        }
        context_features = {
            "progress_signal": 1.0 if progress_delta > 0.0 else 0.0,
            "violation_signal": violation_flag,
            "near_completion": 1.0 if completion_rate >= 0.8 else 0.0,
            "premature_done_signal": premature_done,
            "failed_end_signal": failed_end,
            "stalled_signal": stalled_step,
        }

        logits = [self._logit(hyp, context_features) for hyp in self._hypotheses]
        probs = _softmax(logits)
        hypothesis_rewards = [
            self._reward_under_hypothesis(hyp, reward_features)
            for hyp in self._hypotheses
        ]
        posterior_expected = sum(p * r for p, r in zip(probs, hypothesis_rewards))
        correction = _clip(
            posterior_expected - float(proxy_reward),
            -self.max_abs_correction,
            self.max_abs_correction,
        )

        top_idx = max(range(len(probs)), key=probs.__getitem__) if probs else 0
        metadata = {
            "proxy_reward": round(float(proxy_reward), 4),
            "posterior_expected_reward": round(float(posterior_expected), 4),
            "top_hypothesis": self._hypotheses[top_idx].name,
            "posterior": {
                hyp.name: round(prob, 4)
                for hyp, prob in zip(self._hypotheses, probs)
            },
            "features": {
                key: round(float(val), 4)
                for key, val in reward_features.items()
            },
        }
        return round(correction, 4), metadata

    @staticmethod
    def _logit(hypothesis: RewardHypothesis, features: Mapping[str, float]) -> float:
        score = float(hypothesis.prior_logit)
        for key, weight in hypothesis.context_weights.items():
            score += float(weight) * float(features.get(key, 0.0))
        return score

    @staticmethod
    def _reward_under_hypothesis(
        hypothesis: RewardHypothesis,
        features: Mapping[str, float],
    ) -> float:
        score = 0.0
        for key, weight in hypothesis.reward_weights.items():
            score += float(weight) * float(features.get(key, 0.0))
        return score
