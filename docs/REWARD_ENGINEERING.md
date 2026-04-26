# Reward Engineering — Adaptive Enterprise Autopilot v3

This document explains the current reward stack in 8 sections. Each section names one technique, points at one file, and states one falsifiable claim. Components are grouped by type: **deterministic**, **shaping**, **intrinsic**, and **correction**.

---

## Deterministic Proxy Reward

### 1. Step grader

`src/envs/autopilot_env/grader.py` — multi-component grader: tool correctness, parameter completeness, dependency satisfaction, business rules, reasoning quality. Step reward in [−0.55, +0.50], episode bonus in [0, +1.20].

This is the **proxy reward**. We treat it as imperfect (see §3 and §6) and shape on top of it rather than tuning its constants.

**Guarantee:** deterministic — same state + same action = same reward, always.

---

## Shaping Terms (policy-invariant)

### 2. Potential-Based Reward Shaping (PBRS)

`src/envs/autopilot_env/pbrs.py`. The shaping term is

> F(s, a, s′) = γ · Φ(s′) − Φ(s)

where Φ(s) = w_done · (completed/total) + w_avail · (available/total).

**Claim (mathematical guarantee):** for any choice of weights, the optimal policy of the shaped MDP is identical to the optimal policy of the unshaped MDP. The proof is the Ng–Harada–Russell (1999) theorem; we ship it as a unit test (`tests/test_pbrs_invariance.py`) that builds a 3-state MDP, runs value iteration twice, and asserts both `argmax_a Q*(s, a) == argmax_a Q*_shaped(s, a)` and `Q*_shaped − Q* == −Φ(s)` numerically.

---

## Intrinsic Motivation Terms

### 3. Count-based exploration bonus with linear decay

`src/envs/autopilot_env/intrinsic.py` (class `IntrinsicCounter`). Per (workflow_id, frozenset(completed), tool) bonus of β / √(N + 1) multiplied by `max(0, 1 − episode/200)`.

**Claim (mathematical guarantee):** the decay schedule rules out the standard reward-hacking failure mode where the agent learns to maximise novelty at the expense of the true objective. By episode 200 the bonus is exactly zero, so any policy converged on intrinsic-only behaviour pays nothing and the deterministic grader is the sole signal at convergence.

### 4. Lightweight RND (Random Network Distillation)

`src/envs/autopilot_env/intrinsic.py` (class `LightweightRND`). A fixed random target network and a trainable predictor network compute a prediction error on state features (workflow ID, completed set, available set, tool). Higher error = more novel state = higher bonus.

The RND bonus is scaled by a configurable cap and uses the **same linear decay factor** as the count bonus — both collapse to zero by episode 200.

**Claim (heuristic):** RND captures structural novelty that the count table misses when the state space is large. The prediction error drops as the predictor learns, providing a natural curriculum signal that is complementary to, not redundant with, the count bonus.

---

## Correction & Credit-Assignment Terms

### 5. Difference rewards (counterfactual credit assignment)

`src/envs/autopilot_env/difference_rewards.py`. Compares the actual action's step reward against a deterministic counterfactual baseline (calling `done` from the current state):

> difference_reward = actual_step_reward − baseline_step_reward

The computation is deterministic and side-effect-free — it evaluates both actions through the grader without modifying environment state.

**Claim (heuristic):** a positive difference reward means the agent's action was strictly better than stopping. A negative value means stopping would have been less damaging. This provides step-level credit assignment without expensive full-rollout counterfactuals.

**Note:** this term is wired through the `RewardCombiner` but ships with `w_difference=0.0` by default. It can be activated for experimental training runs.

### 6. IRD posterior correction (proxy misspecification detection)

`src/envs/autopilot_env/ird.py`. The deterministic grader is the project's hand-written proxy reward. The runtime adds a small bounded correction derived from a posterior over three interpretable reward hypotheses:

- `proxy_faithful` — trusts the current grader almost as-is
- `completion_first` — puts more weight on irreversible task progress and successful endings
- `safety_first` — puts more weight on avoiding violations, premature `done`, and failed endings

For each step, context features are scored under those hypotheses, normalized into a posterior via softmax, and the posterior-expected reward is computed:

> ird_posterior_correction = clip(posterior_expected_reward − proxy_reward, −0.3, +0.3)

**Claim (heuristic, honestly stated):** this term makes proxy misspecification visible in `/diagnostics` without pretending to solve full inverse reward design. It is a lightweight heuristic IRD framing, not a research-grade Bayesian IRD implementation.

---

## Learned Signal

### 7. Learned judge (RandomForest soft quality signal)

`src/envs/autopilot_env/judge_model.py`. An optional `RandomForest` regressor trained on (state, action, grader_score) triples provides a soft quality estimate. The contribution is scaled by a small mixing coefficient (α = 0.05 default) to prevent the learned signal from overriding the deterministic grader.

Enabled via `JUDGE_ENABLED=1` environment variable. When disabled, this term contributes zero.

**Claim (heuristic):** the judge adds a smooth interpolation signal in states where the deterministic grader produces identical scores for qualitatively different actions. It does not replace the grader — it softens the reward landscape.

---

## Dispatch & Ablation

### 8. `RewardCombiner` and the ablation harness

`src/envs/autopilot_env/reward_combiner.py` dispatches all reward terms with per-component weights and exposes 4 modes toggleable via `POST /diagnostics/mode` without restarting the environment:

| Mode | Active terms |
|------|-------------|
| `full` | All terms at configured weights |
| `proxy_only` | Extrinsic grader only (all shaping/intrinsic/correction zeroed) |
| `no_pbrs` | Everything except PBRS |
| `no_intrinsic` | Everything except count + RND intrinsic bonuses |

`eval_ablations.py` runs the same scripted oracle policy across all 4 modes and produces `ablation_curve.png` and `ablation_table.md`. **Claim:** every term contributes positively to mean total reward — anything that didn't was removed before submission.

The live demo exposes a **mode toggle button** in the control bar so judges can switch modes mid-episode and see the reward decomposition change in real time.

---

## Sample Complexity Metrics

`train.py` tracks two convergence milestones:

- `episodes_to_threshold_0_5` — first episode where eval reward ≥ 0.5
- `episodes_to_threshold_1_0` — first episode where eval reward ≥ 1.0

These are persisted to `training_metrics.json` and served via `GET /metrics`. The demo frontend displays them in a **sample complexity tile** and populates the `sampleComplexity` story card with real values when available.

**Purpose:** judges can evaluate not just final performance but how quickly the reward stack reaches useful behavior — a direct measure of sample efficiency.

---

## Component → Guarantee Summary

| # | Component | File | Type |
|---|-----------|------|------|
| 1 | Step grader | `grader.py` | Deterministic ✓ |
| 2 | PBRS | `pbrs.py` | Mathematical guarantee ✓ |
| 3 | Count bonus | `intrinsic.py` | Mathematical guarantee (decay) ✓ |
| 4 | RND bonus | `intrinsic.py` | Heuristic |
| 5 | Difference reward | `difference_rewards.py` | Heuristic (deterministic computation) |
| 6 | IRD correction | `ird.py` | Heuristic (bounded, honestly stated) |
| 7 | Learned judge | `judge_model.py` | Heuristic (optional, α-scaled) |
| 8 | RewardCombiner | `reward_combiner.py` | Dispatch layer |
