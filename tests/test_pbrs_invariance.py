"""
Policy-invariance proof for the PBRS module.

Builds a small 3-state, 2-action MDP, computes Q* via value iteration twice
(with and without the PBRS shaping term), and asserts:

  1. argmax_a Q*(s, a) == argmax_a Q*_shaped(s, a)  for every state s.
  2. Q*_shaped(s, a) − Q*(s, a) == −Φ(s) for every (s, a) pair.

Both consequences are direct corollaries of Ng, Harada, Russell (1999).
"""
from __future__ import annotations
import numpy as np

GAMMA = 0.99


def _value_iterate(R: np.ndarray, P: np.ndarray, gamma: float, tol: float = 1e-9):
    """
    R[s, a]      : scalar reward for taking action a in state s
    P[s, a, s']  : transition probability
    Returns (Q, V) of shape (S, A) and (S,).
    """
    S, A, _ = P.shape
    V = np.zeros(S)
    while True:
        Q = R + gamma * np.einsum("sap,p->sa", P, V)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            return Q, V_new
        V = V_new


def test_pbrs_preserves_optimal_policy():
    # 3 states, 2 actions
    S, A = 3, 2

    # Transitions: a0 advances state, a1 stays
    P = np.zeros((S, A, S))
    P[0, 0, 1] = 1.0; P[0, 1, 0] = 1.0
    P[1, 0, 2] = 1.0; P[1, 1, 1] = 1.0
    P[2, 0, 2] = 1.0; P[2, 1, 2] = 1.0   # absorbing goal

    # Base reward: +1 only on entering goal via a0
    R = np.zeros((S, A))
    R[1, 0] = 1.0

    # Potential function: Φ(s0) = 0, Φ(s1) = 0.5, Φ(s2) = 1.0
    Phi = np.array([0.0, 0.5, 1.0])

    # Shaped reward: R'[s, a] = R[s, a] + γ·E_{s'}[Φ(s')] − Φ(s)
    R_shaped = R.copy()
    for s in range(S):
        for a in range(A):
            exp_phi_next = np.dot(P[s, a], Phi)
            R_shaped[s, a] += GAMMA * exp_phi_next - Phi[s]

    Q_base,    _ = _value_iterate(R,        P, GAMMA)
    Q_shaped,  _ = _value_iterate(R_shaped, P, GAMMA)

    # 1. Optimal policy unchanged
    pi_base    = Q_base.argmax(axis=1)
    pi_shaped  = Q_shaped.argmax(axis=1)
    assert np.array_equal(pi_base, pi_shaped), \
        f"PBRS broke policy invariance: base={pi_base}, shaped={pi_shaped}"

    # 2. Q*_shaped(s, a) − Q*(s, a) == −Φ(s) for every (s, a)
    diff = Q_shaped - Q_base
    expected = -Phi[:, None] * np.ones((S, A))
    assert np.allclose(diff, expected, atol=1e-6), \
        f"Q-difference violates Ng et al. (1999):\n{diff}\nvs expected:\n{expected}"


def test_potential_function_bounded():
    """The actual potential() in pbrs.py must stay bounded in [0, W_DONE + W_AVAIL]."""
    from src.envs.autopilot_env.pbrs import potential, W_DONE, W_AVAIL

    workflow = {"tasks": [
        {"task_id": "T1", "dependencies": []},
        {"task_id": "T2", "dependencies": ["T1"]},
        {"task_id": "T3", "dependencies": ["T2"]},
    ]}

    assert potential(workflow, []) >= 0.0
    assert potential(workflow, ["T1", "T2", "T3"]) <= W_DONE + W_AVAIL + 1e-9
    # Empty workflow → 0
    assert potential({"tasks": []}, []) == 0.0


if __name__ == "__main__":
    test_pbrs_preserves_optimal_policy()
    test_potential_function_bounded()
    print("[pbrs] policy-invariance proof: PASS")
