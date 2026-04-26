"""
Potential-Based Reward Shaping (PBRS) for the Adaptive Enterprise Autopilot.

Reference: Ng, Harada, Russell (1999) — "Policy invariance under reward
transformations: Theory and application to reward shaping."

The shaping term  F(s, a, s') = γ·Φ(s') − Φ(s)  is added on top of the
deterministic step reward. The Ng et al. theorem guarantees that the
optimal policy π* is unchanged for any bounded potential Φ.

Potential Φ(s) for this environment:
    Φ(s) = w_done * (completed / total)
         + w_avail * (available / total)

Both terms are in [0, 1]. The combination is bounded in [0, w_done + w_avail].
"""

from __future__ import annotations
from typing import Any, Dict, List


# ── Hyperparameters ──────────────────────────────────────────────────────────
GAMMA: float       = 0.99   # discount factor (must match RL training)
W_DONE: float      = 0.5    # weight on completed-fraction
W_AVAIL: float     = 0.2    # weight on currently-available fraction
PBRS_WEIGHT: float = 1.0    # global on/off knob; set 0.0 to disable shaping


def potential(workflow: Dict[str, Any], completed_ids: List[str]) -> float:
    """
    Compute Φ(s) for the current environment state.

    Bounded in [0, W_DONE + W_AVAIL]. Pure function — no side effects.
    """
    tasks = workflow.get("tasks", [])
    n = len(tasks)
    if n == 0:
        return 0.0

    completed = set(completed_ids)
    n_done = sum(1 for t in tasks if t["task_id"] in completed)

    n_avail = 0
    for t in tasks:
        if t["task_id"] in completed:
            continue
        deps = t.get("dependencies", [])
        if all(d in completed for d in deps):
            n_avail += 1

    return W_DONE * (n_done / n) + W_AVAIL * (n_avail / n)


def shaping_term(phi_before: float, phi_after: float, gamma: float = GAMMA) -> float:
    """F = γ·Φ(s′) − Φ(s)."""
    return gamma * phi_after - phi_before


def shaped_reward(
    base_reward: float,
    phi_before: float,
    phi_after: float,
    weight: float = PBRS_WEIGHT,
    gamma: float = GAMMA,
) -> float:
    """Convenience: returns base + weight * F."""
    return base_reward + weight * shaping_term(phi_before, phi_after, gamma)
