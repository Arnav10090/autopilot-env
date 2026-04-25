"""
Count-based intrinsic motivation with linear decay.

Bonus = β / sqrt(N(s, a) + 1)  ·  decay_factor(episode_idx)

State key:  (workflow_id, frozenset(completed_task_ids), action.tool)
Decay:      decay_factor = max(0, 1 − episode_idx / DECAY_EPISODES)

The decay is the anti-reward-hacking guarantee — by ~episode 200 the bonus is
exactly zero, so any policy converged on intrinsic-only behaviour pays nothing
and the deterministic grader is the sole signal at convergence.
"""

from __future__ import annotations
import math
from typing import Any, Dict, Hashable, List, Tuple


BETA: float           = 0.05    # max per-step bonus magnitude
DECAY_EPISODES: int   = 200     # bonus → 0 after this many episodes
INTRINSIC_WEIGHT: float = 1.0   # global on/off knob


class IntrinsicCounter:
    """Holds the visitation counts. One instance per AutopilotEnvironment."""

    def __init__(self):
        self._counts: Dict[Tuple[Hashable, ...], int] = {}
        self._episode_idx: int = 0

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

    def bonus(
        self,
        workflow_id: str,
        completed_ids: List[str],
        tool: str,
        weight: float = INTRINSIC_WEIGHT,
    ) -> float:
        """Compute bonus for the current state-action and increment the counter."""
        if not tool:
            return 0.0
        key = self._key(workflow_id, completed_ids, tool)
        n = self._counts.get(key, 0)
        raw = BETA / math.sqrt(n + 1)
        bonus = weight * raw * self.decay_factor()
        self._counts[key] = n + 1
        return bonus

    def peek(self, workflow_id: str, completed_ids: List[str], tool: str) -> int:
        """Return current count without incrementing."""
        return self._counts.get(self._key(workflow_id, completed_ids, tool), 0)
