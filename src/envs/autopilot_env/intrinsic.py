"""
Intrinsic motivation with shared linear decay.

Two intrinsic signals are combined:

1. Count bonus:
   β / sqrt(N(s, a) + 1)
2. Lightweight RND bonus:
   clipped predictor error against a frozen random target

Both use the same decay schedule:

decay_factor = max(0, 1 − episode_idx / DECAY_EPISODES)

The decay is the anti-reward-hacking guarantee — by ~episode 200 both bonuses
collapse to zero, so the deterministic grader is the only long-run signal.
"""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
import random
from typing import Dict, Hashable, List, Sequence, Tuple


COUNT_BETA: float = 0.05
RND_BETA: float = 0.05
DECAY_EPISODES: int = 200
INTRINSIC_WEIGHT: float = 1.0
RND_FEATURE_DIM: int = 16
RND_OUTPUT_DIM: int = 8
RND_LEARNING_RATE: float = 0.15
RND_ERROR_CLIP: float = 1.0
RND_TARGET_SEED: int = 1337


@dataclass(frozen=True)
class IntrinsicComponents:
    count_bonus: float = 0.0
    rnd_bonus: float = 0.0
    decay_factor: float = 1.0
    rnd_error: float = 0.0


class IntrinsicCounter:
    """Holds count and lightweight RND state. One instance per environment."""

    def __init__(self):
        self._counts: Dict[Tuple[Hashable, ...], int] = {}
        self._episode_idx: int = 0
        rng = random.Random(RND_TARGET_SEED)
        self._target_weights: List[List[float]] = [
            [rng.uniform(-0.75, 0.75) for _ in range(RND_FEATURE_DIM)]
            for _ in range(RND_OUTPUT_DIM)
        ]
        self._predictor_weights: List[List[float]] = [
            [0.0 for _ in range(RND_FEATURE_DIM)]
            for _ in range(RND_OUTPUT_DIM)
        ]

    def reset_episode(self) -> None:
        self._episode_idx += 1

    @property
    def episode_idx(self) -> int:
        return self._episode_idx

    def decay_factor(self) -> float:
        """Linear decay from 1.0 → 0.0 over DECAY_EPISODES episodes."""
        return max(0.0, 1.0 - self._episode_idx / DECAY_EPISODES)

    def _key(self, workflow_id: str, completed_ids: List[str], tool: str) -> Tuple:
        return (workflow_id, frozenset(completed_ids or []), tool or "")

    def count_bonus(
        self,
        workflow_id: str,
        completed_ids: List[str],
        tool: str,
        weight: float = INTRINSIC_WEIGHT,
    ) -> float:
        """Compute count bonus for the current state-action and increment visits."""
        if not tool:
            return 0.0
        key = self._key(workflow_id, completed_ids, tool)
        n = self._counts.get(key, 0)
        raw = COUNT_BETA / math.sqrt(n + 1)
        bonus = weight * raw * self.decay_factor()
        self._counts[key] = n + 1
        return round(bonus, 4)

    def bonus(
        self,
        workflow_id: str,
        completed_ids: List[str],
        tool: str,
        weight: float = INTRINSIC_WEIGHT,
    ) -> float:
        """Backward-compatible alias for the count bonus."""
        return self.count_bonus(
            workflow_id=workflow_id,
            completed_ids=completed_ids,
            tool=tool,
            weight=weight,
        )

    def peek(self, workflow_id: str, completed_ids: List[str], tool: str) -> int:
        """Return current count without incrementing."""
        return self._counts.get(self._key(workflow_id, completed_ids, tool), 0)

    def components(
        self,
        workflow_id: str,
        completed_ids: List[str],
        available_ids: List[str],
        tool: str,
        weight: float = INTRINSIC_WEIGHT,
    ) -> IntrinsicComponents:
        decay = self.decay_factor()
        if not tool:
            return IntrinsicComponents(decay_factor=round(decay, 4))

        count_bonus = self.count_bonus(
            workflow_id=workflow_id,
            completed_ids=completed_ids,
            tool=tool,
            weight=weight,
        )
        rnd_error = self._rnd_error(
            workflow_id=workflow_id,
            completed_ids=completed_ids,
            available_ids=available_ids,
            tool=tool,
        )
        rnd_bonus = round(weight * RND_BETA * min(rnd_error, RND_ERROR_CLIP) * decay, 4)
        return IntrinsicComponents(
            count_bonus=count_bonus,
            rnd_bonus=rnd_bonus,
            decay_factor=round(decay, 4),
            rnd_error=round(rnd_error, 4),
        )

    def _rnd_error(
        self,
        workflow_id: str,
        completed_ids: List[str],
        available_ids: List[str],
        tool: str,
    ) -> float:
        features = self._state_features(
            workflow_id=workflow_id,
            completed_ids=completed_ids,
            available_ids=available_ids,
            tool=tool,
        )
        target = self._forward(self._target_weights, features)
        pred_before = self._forward(self._predictor_weights, features)
        error = self._mean_squared_error(target, pred_before)
        self._train_predictor(features, target, pred_before)
        return error

    def _state_features(
        self,
        workflow_id: str,
        completed_ids: Sequence[str],
        available_ids: Sequence[str],
        tool: str,
    ) -> List[float]:
        tokens = [
            f"wf:{workflow_id or ''}",
            f"tool:{tool or ''}",
            f"done:{','.join(sorted(completed_ids or []))}",
            f"avail:{','.join(sorted(available_ids or []))}",
        ]
        vec = [0.0 for _ in range(RND_FEATURE_DIM)]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for idx in range(RND_FEATURE_DIM):
                vec[idx] += (digest[idx] / 255.0) * 2.0 - 1.0
        scale = 1.0 / max(1, len(tokens))
        return [value * scale for value in vec]

    @staticmethod
    def _forward(weights: Sequence[Sequence[float]], features: Sequence[float]) -> List[float]:
        return [
            sum(weight * feature for weight, feature in zip(row, features))
            for row in weights
        ]

    @staticmethod
    def _mean_squared_error(target: Sequence[float], pred: Sequence[float]) -> float:
        if not target:
            return 0.0
        return sum((t - p) ** 2 for t, p in zip(target, pred)) / len(target)

    def _train_predictor(
        self,
        features: Sequence[float],
        target: Sequence[float],
        pred_before: Sequence[float],
    ) -> None:
        scale = 2.0 / max(1, len(target))
        for out_idx, row in enumerate(self._predictor_weights):
            grad_common = scale * (pred_before[out_idx] - target[out_idx])
            for feat_idx, feature in enumerate(features):
                row[feat_idx] -= RND_LEARNING_RATE * grad_common * feature
