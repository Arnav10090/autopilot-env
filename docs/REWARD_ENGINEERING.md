# Reward Engineering — Adaptive Enterprise Autopilot v2

This document explains the v2 reward stack in 4 sections. Each section names one technique, points at one file, and states one falsifiable claim.

## 1. Deterministic step grader (existing)

`src/envs/autopilot_env/grader.py` — multi-component grader: tool correctness, parameter completeness, dependency satisfaction, business rules, reasoning quality. Step reward in [-0.5, +0.5], episode bonus in [0, +1.2].

This is the **proxy reward**. We treat it as imperfect (see §2 and §3) and shape on top of it rather than tuning its constants.

## 2. Potential-Based Reward Shaping (PBRS)

`src/envs/autopilot_env/pbrs.py`. The shaping term is

> F(s, a, s′) = γ · Φ(s′) − Φ(s)

where Φ(s) = w_done · (completed/total) + w_avail · (available/total).

**Claim:** for any choice of weights, the optimal policy of the shaped MDP is identical to the optimal policy of the unshaped MDP. The proof is the Ng–Harada–Russell (1999) theorem; we ship it as a unit test (`tests/test_pbrs_invariance.py`) that builds a 3-state MDP, runs value iteration twice, and asserts both `argmax_a Q*(s, a) == argmax_a Q*_shaped(s, a)` and `Q*_shaped − Q* == −Φ(s)` numerically.

## 3. Count-based intrinsic motivation with linear decay

`src/envs/autopilot_env/intrinsic.py`. Per (workflow_id, frozenset(completed), tool) bonus of β / √(N + 1) multiplied by `max(0, 1 − episode/200)`.

**Claim:** the decay schedule rules out the standard reward-hacking failure mode where the agent learns to maximise novelty at the expense of the true objective. By episode 200 the bonus is exactly zero, so any policy converged on intrinsic-only behaviour pays nothing and the deterministic grader is the sole signal at convergence.

## 4. `RewardCombiner` and the ablation harness

`src/envs/autopilot_env/reward_combiner.py` exposes 4 modes (`full / proxy_only / no_pbrs / no_intrinsic`) toggleable via `POST /diagnostics/mode` without restarting the environment.

`eval_ablations.py` runs the same scripted oracle policy across all 4 modes and produces `ablation_curve.png` and `ablation_table.md`. **Claim:** every term contributes positively to mean total reward — anything that didn't was removed before submission.
