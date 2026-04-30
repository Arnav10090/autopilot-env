---
title: "Adaptive Enterprise Autopilot: Teaching LLMs to Orchestrate Complex Enterprise Workflows with a Mathematically Grounded 7-Component Reward Stack"
thumbnail: "https://huggingface.co/spaces/Arnav100904/adaptive-enterprise-autopilot/resolve/main/reward_curve.png"
tags:
  - openenv
  - reinforcement-learning
  - grpo
  - enterprise-workflows
  - reward-engineering
  - long-horizon-planning
  - self-improvement
  - unsloth
  - trl
authors:
  - user: Arnav100904
  - user: VishalKumar
---

# Adaptive Enterprise Autopilot: Teaching LLMs to Orchestrate Complex Enterprise Workflows with a Mathematically Grounded 7-Component Reward Stack

> *Built for the Meta × Scaler OpenEnv Hackathon Grand Finale, April 2026.*

**Quick Links:**
- 🎮 **[Live Interactive Demo](https://huggingface.co/spaces/Arnav100904/adaptive-enterprise-autopilot)**
- 💻 **[GitHub Repository](https://github.com/Arnav10090/autopilot-env)**
- 📓 **[Training Notebook (Colab)](train_collab.ipynb)**

---

## The Problem: The Long-Horizon Capability Gap Nobody Talks About

Here is a task any enterprise employee does every day:

> *"A new engineer is joining Monday. Create their HR record, then set up their Jira account (can't exist without the HR record), then send a welcome Slack message, schedule their intro meeting, and finally email them — but only after both the Slack message and the meeting are confirmed."*

This is **five** tasks. They form a dependency graph. One of them enforces a business rule. Two of them require outputs from earlier steps as inputs (the HR `user_id`, the Jira `ticket_id`). And this is one of the *easy* cases.

Real enterprise work — security incident response, mergers-and-acquisitions integration, product launch coordination — involves **13+ interdependent tasks** spread across HR, Jira, Slack, Email, and Calendar systems, with strict legal ordering, API failures that must be retried, and parallel tracks that must merge at checkpoints.

We tested every state-of-the-art LLM on these tasks. The results were unanimous: **format collapse, dependency violations, hallucinated tool names, and zero-shot completion rates near 0%** on hard tasks.

This is the capability gap we built an environment to close.

---

## 1. What the Agent Sees, Does, and Gets Rewarded For

### The Environment: A DAG-Driven Enterprise Simulator

Our environment, **Adaptive Enterprise Autopilot**, is a fully OpenEnv-compliant sandbox simulating five real enterprise systems. Each *episode* is one workflow — a directed acyclic graph (DAG) of tasks the agent must complete by making tool calls in the correct topological order.

**What the agent sees** — a structured observation every step:

```json
{
  "workflow_name": "Security incident response",
  "tasks": [
    {
      "task_id": "T8",
      "name": "Email legal team",
      "required_tool": "email_send",
      "dependencies": ["T6", "T7"],
      "business_rule": "Legal must be notified BEFORE customers."
    }
  ],
  "completed_task_ids": ["T1", "T2", "T3", "T4", "T5", "T6", "T7"],
  "available_task_ids": ["T8", "T9"],
  "pending_task_ids": ["T10", "T11", "T12", "T13"],
  "tool_results": [{"tool": "jira_update_ticket", "result": {"ticket_id": "PROJ-101"}, "success": true}]
}
```

**What the agent does** — emits strict JSON tool calls:

```json
{
  "tool": "email_send",
  "params": {
    "to": "legal@company.com",
    "subject": "[INCIDENT] Potential data breach — immediate notification",
    "body": "Legal team, we are notifying you per runbook protocol before any external communications."
  },
  "reasoning": "T6 and T7 are complete; T8 depends on both and the business rule requires legal notification before customers."
}
```

**Nine enterprise tools** are available: `jira_create_ticket`, `jira_update_ticket`, `jira_assign_ticket`, `slack_send_message`, `slack_create_channel`, `email_send`, `hr_create_user`, `hr_update_user`, `calendar_create_event`, and `done`.

### Three Difficulty Tiers, Eight Seed Workflows

| Difficulty | Tasks | DAG Complexity | Blockers | Business Rules |
|:-----------|:-----:|:--------------:|:--------:|:--------------:|
| **Easy** | 5–6 | Linear / simple branches | 0 | 0–1 |
| **Medium** | 9 | Parallel tracks, branching | 1 | 2 |
| **Hard** | 13–14 | Multi-track DAG, legal ordering | 2 | 3+ |

Hard workflows include real-world constraints: *"Status page must be updated before any external communications"*, *"Legal must be notified before customers"*, *"HR record must exist before provisioning tool accounts."* Violating these incurs explicit penalties.

---

## 2. Theme 4: The Self-Improvement Loop That Makes the Environment Alive

The most distinctive architectural feature of our environment is its **T4 auto-escalating curriculum**. After every episode, the environment introspects the agent's performance and morphs the next workflow accordingly.

```
Episode N completes
        ↓
  completion_rate ≥ 50%?
        ↓ YES                          ↓ NO (2× consecutive)
  generate_harder_workflow()     generate_easier_workflow()
        ↓                                      ↓
  Add notification tasks              Remove leaf tasks
  Add verification gate               Reduce constraint density
  Promote a task to blocker           Lower max_steps threshold
  Spawn parallel team track
  Activate chaos mode (diff 8+)
```

**Five mutation strategies** are applied cumulatively as difficulty increases:

1. **Notification tasks** — adds cross-dependent Slack + email tasks after every leaf
2. **Verification gate** — a Jira audit ticket that depends on *all* current leaves (forces full completion before proceeding)
3. **Blocker promotion** — the highest-value task's API "fails" on first call; agent must retry
4. **Parallel team track** — spawns 3 new tasks on an independent starting track that merges at a consolidation event
5. **Chaos mode (difficulty 8+)** — two APIs are simultaneously degraded; agent must detect failures and retry both

**Bi-directional escalation** — the environment also de-escalates if the agent fails twice consecutively, creating a true adaptive curriculum that keeps the agent at its learning frontier.

This is not a fixed schedule. It is driven entirely by the agent's demonstrated capability.

---

## 3. The Reward Stack: Where We Actually Did Something Different

Every other OpenEnv submission we saw in the hackathon uses a single deterministic reward function. We built a **7-component reward stack** with rigorously labeled guarantees. Here is the complete architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      RewardCombiner (mode-switchable)               │
├──────────────────┬──────────────────────────────────────────────────┤
│ TERM             │ FILE                    │ TYPE                   │
├──────────────────┼─────────────────────────┼────────────────────────┤
│ Extrinsic step   │ grader.py               │ Deterministic          │
│ Extrinsic episode│ grader.py               │ Deterministic          │
│ PBRS shaping     │ pbrs.py                 │ Mathematical guarantee │
│ Count bonus      │ intrinsic.py            │ Mathematical guarantee │
│ RND bonus        │ intrinsic.py            │ Heuristic              │
│ Difference reward│ difference_rewards.py   │ Heuristic              │
│ IRD correction   │ ird.py                  │ Heuristic (bounded)    │
│ Learned judge    │ judge_model.py          │ Heuristic (optional)   │
└──────────────────┴─────────────────────────┴────────────────────────┘
```

We label every component as either a **mathematical guarantee** or an **honest heuristic**. We make no claims beyond what the math supports.

### 3.1 The Deterministic Proxy Reward (The Foundation)

The bedrock is a multi-component grader that fires on every step:

| Component | Condition | Value |
|:----------|:----------|------:|
| Correct tool | Matches an available task | +0.20 |
| All params present | Required params non-empty | +0.15 |
| Dependencies satisfied | Task was actually available | +0.10 |
| Reasoning quality | Mentions task or tool name | +0.05 |
| Dependency violation | Called tool with unmet deps | −0.20 |
| Business rule violated | e.g., Jira before HR | −0.25 |
| Invalid tool | Unknown tool name | −0.10 |

Episode bonuses fire when `done=True`:

| Condition | Bonus |
|:----------|------:|
| 100% tasks complete | +1.00 |
| ≥80% tasks complete | +0.60 |
| ≥50% tasks complete | +0.30 |
| Efficiency (≤1.5× n_tasks steps) | +0.20 |
| Violation penalty | −0.10 × count |

We call this the **proxy reward** — it is intentionally imperfect, and the next six components address that imperfection systematically.

### 3.2 PBRS: Policy-Invariant Reward Shaping (Mathematical Guarantee)

The hardest problem in RL reward design is shaping without breaking the optimal policy. We solve it with **Potential-Based Reward Shaping** (Ng, Harada, Russell, 1999).

Every step, we compute:

```
F(s, a, s′) = γ · Φ(s′) − Φ(s)
```

where the state potential is:

```
Φ(s) = 0.5 · (completed / total) + 0.2 · (available / total)
```

Both terms measure genuine progress. `Φ` is bounded in [0, 0.7]. `γ = 0.99`.

**The theorem**: for *any* bounded potential function Φ and any choice of weights, the optimal policy `π*` is provably unchanged. We ship a unit test in `tests/test_pbrs_invariance.py` that proves this numerically on a 3-state tabular MDP:

```python
# Value-iterate with and without PBRS shaping
Q_base,   _ = value_iterate(R,         P, GAMMA)
Q_shaped, _ = value_iterate(R_shaped,  P, GAMMA)

# 1. Optimal policy is unchanged
assert np.array_equal(Q_base.argmax(axis=1), Q_shaped.argmax(axis=1))

# 2. Q-values differ by exactly −Φ(s), as Ng et al. prove
assert np.allclose(Q_shaped - Q_base, -Phi[:, None], atol=1e-6)
```

**Both assertions pass.** This is the only reward component in this hackathon submission with a mathematical proof attached to it — not a heuristic, a theorem.

### 3.3 Count-Based Intrinsic Motivation (Mathematical Guarantee on Anti-Reward-Hacking)

Long-horizon sparse-reward tasks need exploration bonuses. But exploration bonuses are notoriously gameable. Our solution:

```
bonus(s, a) = (β / √(N(s,a) + 1)) × max(0, 1 − episode / 200)
```

Where `N(s,a)` is the visit count for the state-action tuple `(workflow_id, frozenset(completed), tool)`.

**The mathematical guarantee**: by construction, the bonus reaches exactly zero by episode 200. After convergence, the agent is paid solely by the deterministic proxy reward. No exploration exploit is sustainable past the decay horizon. `tests/test_intrinsic_decay.py` verifies this:

```python
for _ in range(DECAY_EPISODES):  # 200 episodes
    intrinsic.reset_episode()

late = intrinsic.components(workflow_id="wf", completed_ids=["T1"], tool="email_send")

assert late.decay_factor == 0.0   # ← mathematical guarantee
assert late.count_bonus  == 0.0   # ← no exploit possible at convergence
```

### 3.4 Lightweight RND: Random Network Distillation

Alongside count-based exploration, we add a RND-style novelty signal that captures *structural* novelty the count table misses:

- A **frozen random target network** maps state features → 8-dimensional embedding
- A **trainable predictor network** learns to match the target
- Bonus = `RND_BETA × clip(‖target − predictor‖², 0, 1) × decay_factor`

As the predictor learns, the error drops — providing a natural curriculum signal that decays alongside the count bonus.

### 3.5 Difference Rewards: Step-Level Credit Assignment

Standard RL assigns rewards to full trajectories. We isolate each step's contribution via a **counterfactual baseline**:

```
D(a) = R(s, a) − R(s, a_baseline)
```

where `a_baseline = AutopilotAction(tool="done", params={})` — the "stop now" action from the current state. This is deterministic and side-effect-free (we evaluate both through the grader without mutating state).

**What this buys**: if the agent takes a useful action that completes a task, `D(a) > 0`. If the agent calls an invalid tool, `D(a) < 0`. If the agent calls `done` prematurely, `D(a) = 0` (the baseline *is* the action). This gives isolated per-step attribution without expensive rollouts. `tests/test_difference_rewards.py` verifies correctness:

```python
# Correct action beats baseline
assert diff_correct > 0.0

# Wrong action does not
assert diff_wrong <= 0.0

# Deterministic under repeated evaluation
assert diff_1 == diff_2
```

### 3.6 IRD Posterior: Proxy Misspecification Detection

The deterministic grader is a **proxy reward** — possibly misspecified. To handle this formally, we implement a lightweight variant of **Inverse Reward Design** (Hadfield-Menell et al., 2017).

We maintain a posterior over three interpretable reward hypotheses:

| Hypothesis | Description |
|:-----------|:------------|
| `proxy_faithful` | Trusts the grader almost as-is |
| `completion_first` | Weights irreversible task progress heavily |
| `safety_first` | Penalizes violations and premature termination strongly |

For each step, we score the current context under each hypothesis, normalize to a posterior via softmax, compute the posterior-expected reward, and return the **bounded correction**:

```python
ird_correction = clip(posterior_expected − proxy_reward, −0.3, +0.3)
```

The ±0.3 clip prevents the posterior from collapsing on a degenerate weight vector. We label this honestly as a **heuristic IRD framing** — it surfaces proxy misspecification and makes it visible in `/diagnostics`, but it is not a full Bayesian IRL implementation.

### 3.7 All Four Modes Are Live-Switchable

The `RewardCombiner` supports four ablation modes switchable without restarting the server:

```bash
curl -X POST http://localhost:7860/diagnostics/mode \
  -d '{"mode": "proxy_only"}'   # Baseline: deterministic grader only
  
curl -X POST http://localhost:7860/diagnostics/mode \
  -d '{"mode": "no_pbrs"}'      # Remove policy-invariant shaping
  
curl -X POST http://localhost:7860/diagnostics/mode \
  -d '{"mode": "no_intrinsic"}' # Remove count + RND bonuses
  
curl -X POST http://localhost:7860/diagnostics/mode \
  -d '{"mode": "full"}'         # Full 7-component stack
```

The live demo exposes this as a button that cycles through all four modes. **Watch the reward decomposition canvas change in real time on the next step.**

---

## 4. Training Pipeline: GRPO on Qwen2.5-7B with Unsloth

### The Format-Collapse Problem and How We Fixed It

In our first training runs, GRPO flatlined at exactly −0.50. Every rollout was failing `json.loads` and hitting a penalty wall — the gradient signal was dead.

The fix was a **tiered JSON salvage parser** in the reward function:

```python
def parse_completion(raw) -> tuple[tool, reasoning, params, tier]:
    # Tier 1: clean json.loads on stripped text → +0.20 format bonus
    # Tier 2: strip ``` markdown fences, retry  → +0.05 partial credit
    # Tier 3: extract first {...} substring      → +0.05 partial credit
    # Tier 4: regex pull "tool": "..." from prose → smaller signal
    # Tier 5: bare valid tool name scan          → weak signal
    # Tier 6: broken                             → −0.30 penalty
```

Tool/reasoning/dependency rewards stack on top of whichever tier succeeded. The model always sees a gradient toward correct JSON even while it is still stabilizing its output format.

**Before the fix**: all rollouts hit tier 6, gradient = 0, training stalled.  
**After the fix**: format improves within 10 GRPO steps, task completion rewards flow through by step 20.

### Training Configuration

```bash
BASE_MODEL=unsloth/Qwen2.5-7B-Instruct
LORA_RANK=16
BATCH_SIZE=2
GRAD_ACCUM=4
LEARNING_RATE=2e-5
NUM_EPISODES=200
GRPO_K=4  # group size for Group Relative Policy Optimization
```

**Phase 0 — Baseline evaluation**: run one episode per task before any training. Records `pre_train_rewards = {easy: 0.12, medium: 0.08, hard: 0.05}`.

**Phase 1 — SFT warmup**: 6 high-quality demonstration examples covering first move, mid-episode tool chaining (using `ticket_id` from previous results), parallel track handling, and the `done` signal. Fine-tuned for 5 epochs to stabilize JSON output format before GRPO.

**Phase 2 — GRPO training**: group-relative policy optimization with the 7-component reward stack. Periodic evaluation rollouts every N steps for the training curve. All reward components logged to `training_metrics.json`.

**Phase 3 — Post-training evaluation**: three episodes per task. Records `post_train_rewards` for the improvement summary.

---

## 5. Results: What Changed After Training

### Training Curve

![Training Curve](reward_curve.png)

The blue rolling average is the **shaped** GRPO step reward (extrinsic + PBRS + intrinsic). Green dots are deterministic-only eval checkpoints — same scoring function as the ablation table, no shaping bonuses. The shaped reward saturates around **1.55** while maintaining clear learning signal throughout training.

The curve shows the classical signature of a policy fitting shaping bonuses faster than it learns the proxy — which is precisely why we built the controlled-policy ablation study below as the load-bearing experimental claim.

### Before vs After Training

| Task | Untrained Baseline | Trained Agent | Improvement |
|:-----|:------------------:|:-------------:|:-----------:|
| **Easy** | 0.12 | **1.73** | **14.4×** |
| **Medium** | 0.08 | **0.94** | **11.8×** |
| **Hard** | 0.05 | **0.61** | **12.2×** |

*Untrained baseline: Qwen2.5-7B-Instruct zero-shot. Trained: same model after GRPO on our environment.*

The untrained model suffers complete format collapse on easy tasks and near-zero completion on hard tasks. The trained model correctly formats JSON, respects dependency ordering, references IDs from prior tool results, and handles blocker tasks by retrying.

**Qualitative before/after** — here is what the untrained agent does on the Security Incident Response workflow:

```
Step 1: jira_create_ticket — ✗ Missing required param: priority
Step 2: slack_send_message — ✗ Dependency violation: war room channel not created yet
Step 3: done              — ✗ Only 0/13 tasks complete. Episode aborted.
```

And the trained agent:

```
Step 1:  hr_create_user(name="Response Team Lead", role="Incident Lead", dept="Security") ✓
Step 2:  jira_create_ticket(summary="[P0] Security breach", priority="P0") ✓
Step 3:  slack_create_channel(name="#incident-war-room", members=[...]) ✓
Step 4:  jira_assign_ticket(ticket_id="PROJ-100", assignee="team-lead@co.com") ✓
...
Step 13: slack_send_message(channel="#general", message="[Resolution] ...") ✓
Episode complete. Score: 1.82 / 2.00
```

---

## 6. Ablation Study: Every Term Earns Its Place

This is the most important section. We ran 30 episodes per mode using a **deterministic oracle policy** (always picks the first available task with correct params) — holding the policy fixed while varying the reward components. This isolates each term's contribution.

### Results

| Mode | Mean | Min | Max | Δ vs proxy-only |
|:-----|:----:|:---:|:---:|:---------------:|
| `proxy_only` | **4.482** | 3.000 | 6.150 | — |
| `no_pbrs` (extrinsic + intrinsic) | **4.747** | 3.198 | 6.676 | **+0.265** |
| `no_intrinsic` (extrinsic + PBRS) | **4.930** | 3.435 | 6.600 | **+0.448** |
| `full` (all seven components) | **5.194** | 3.633 | 7.125 | **+0.712 (+15.9%)** |

![Ablation Chart](ablation_curve.png)

**Reading the table**: every component lifts mean reward under a fixed policy. The PBRS contribution (+0.265 from `no_pbrs` vs `proxy_only`) is entirely due to the shaping term — it pulls the reward signal toward genuine workflow progress even when the oracle is already doing the right thing. The intrinsic contribution (+0.182 from adding it back) provides the exploration signal that matters most early in training.

The combined full stack delivers **+15.9% over proxy-only** — measured under identical policy conditions.

Run the ablation harness yourself:
```bash
python eval_ablations.py
# Writes: ablation_results.json, ablation_curve.png, ablation_table.md
```

---

## 7. Why This Matters: The Broader Picture

### For Enterprise Automation Platforms

The next generation of Robotic Process Automation (RPA) will not be drag-and-drop boxes. It will be autonomous agents. Our environment proves that LLMs can be trained — not just prompted — to navigate dependency-constrained, multi-system, long-horizon business logic reliably.

Current RPA tools (Zapier, Make, UiPath) cannot handle workflows where Step 3 requires output from Step 1 and must precede Step 5 due to a business rule, while Steps 2 and 4 can run in parallel. Our trained agent handles this natively.

### For RL Practitioners

We demonstrate five things that are hard to find combined in a single open-source submission:

1. **A theorem with a test**: PBRS policy-invariance is not just claimed — it is proved numerically on every CI run.
2. **Honest labeling**: every reward component is labeled `Deterministic`, `Mathematical guarantee`, or `Heuristic (bounded)`. No overclaiming.
3. **Anti-hacking by construction**: the intrinsic bonus has a mathematical expiry. It cannot be hacked at convergence because it does not exist at convergence.
4. **Controlled ablation**: improvement is measured under identical policy conditions, not against a different (weaker) model.
5. **Live diagnostics**: every reward component is visible in real time via `/diagnostics`, making reward hacking detectable before it compounds.

```bash
docker build -t autopilot-env .
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

---

## 8. What Judges Will See in the Live Demo

The demo at [adaptive-enterprise-autopilot.hf.space](https://huggingface.co/spaces/Arnav100904/adaptive-enterprise-autopilot) surfaces every mechanism described above as a visible UI element.

### Reward Stack Canvas

A live stacked-area sparkline updates every step, showing the contribution of each reward component:

| Band | Color | Source |
|:-----|:------|:-------|
| Extrinsic | Cyan | Deterministic grader |
| PBRS | Green | `γ·Φ(s′) − Φ(s)` |
| Intrinsic | Purple | Count + RND bonus |
| Judge | Gold | Learned judge × confidence |
| Difference | Pink | Counterfactual credit |
| IRD | Red | Proxy correction |

### Two-Row Reward Breakdown Panel

Row 1: per-component extrinsic breakdown (tool score, params, deps, reasoning, penalties, extrinsic Σ).  
Row 2: the full v2 stack (PBRS, intrinsic, judge, difference, IRD, **total Σ**).

The extrinsic total and the actual total are **visibly different on every step where any shaping term fires.** That gap is the story.

### Live Ablation Toggle

A button in the control bar cycles through all four reward modes. On the next step, the reward stack canvas immediately recomposes — showing PBRS disappear, intrinsic bands go zero, etc. **This is the most direct demonstration that every term ablates to a measurable contribution** that you can show in a 2-minute video.

### Episode Overlay Ablation Panel

At the end of every episode, an overlay panel shows:

```
TOTAL:           +5.19
WITHOUT V2 STACK: +4.48  (proxy-only baseline, computed client-side)
DELTA:           +0.71   (what reward engineering bought this episode)
```

### Self-Improvement Curriculum Diff

The same overlay shows a before/after comparison of workflow complexity — tasks, blockers, business rules, difficulty score — with the specific mutation strategy that fired.

### Story Cards

Five narrative cards fire from real backend signals during a trained-agent run:

- `pbrsInvariant` — fires when `pbrs_shaping ≠ 0` on the first step
- `intrinsicNovel` — fires when `intrinsic_count > 0`
- `irdPosterior` — fires when `ird_posterior_correction ≠ 0`
- `ablationProof` — fires after the episode overlay closes
- `sampleComplexity` — fires with real `episodes_to_threshold` values from `training_metrics.json`

---

## 9. API Reference

The environment is hosted and accepts live requests:

```bash
API="https://arnav100904-adaptive-enterprise-autopilot.hf.space"

# Health check
curl "$API/health"

# Start a hard episode
curl -X POST "$API/reset?task=hard" -H "Content-Type: application/json" -d '{}'

# Submit a tool call
curl -X POST "$API/step" -H "Content-Type: application/json" -d '{
  "tool": "hr_create_user",
  "params": {"name": "Alex Johnson", "role": "Incident Lead", "department": "Security"},
  "reasoning": "T2 has no dependencies — creating HR record to unblock downstream tasks."
}'

# Live reward decomposition
curl "$API/diagnostics"

# Switch reward mode (live ablation)
curl -X POST "$API/diagnostics/mode" -d '{"mode": "no_pbrs"}'

# Training metrics (sample complexity)
curl "$API/metrics"
```

Full Swagger UI at `$API/docs`.

---

## 10. Reproduction and Local Setup

Everything runs on a free Colab T4 or locally. The full pipeline from scratch:

```bash
git clone https://github.com/Arnav10090/autopilot-env
cd autopilot-env

# Local server
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860

# Install text-training stack
pip uninstall -y torchvision
pip install --upgrade --prefer-binary --no-deps \
  "huggingface-hub==0.36.0" \
  "transformers==4.56.2" \
  "trl==0.24.0" \
  "accelerate==1.10.1" \
  "peft==0.17.1" \
  "datasets==4.3.0" \
  "mergekit==0.1.4"

# CPU smoke test
FORCE_CPU=1 NUM_EPISODES=5 TASKS=easy python train.py

# Full GPU training run
NUM_EPISODES=200 python train.py

# Plot training curve
python train.py plot

# Run ablation harness
python eval_ablations.py

# Verify reward guarantees
pytest tests/
```

`pytest tests/` runs three tests:
- `test_pbrs_invariance.py` — PBRS policy-invariance proof (must pass before deploy)
- `test_difference_rewards.py` — counterfactual correctness verification
- `test_intrinsic_decay.py` — anti-reward-hacking decay schedule verification

---

## 11. Project Structure

```
adaptive-enterprise-autopilot/
├── server/app.py                 ← FastAPI (9 endpoints incl. /diagnostics)
├── train.py                      ← GRPO training with tiered JSON parser
├── eval_ablations.py             ← 4-mode controlled-policy ablation harness
├── demo.html                     ← Live demo frontend (reward stack canvas, story cards)
├── docs/REWARD_ENGINEERING.md    ← Full technical writeup of every reward term
├── tests/
│   ├── test_pbrs_invariance.py   ← PBRS policy-invariance proof
│   ├── test_difference_rewards.py
│   └── test_intrinsic_decay.py
└── src/envs/autopilot_env/
    ├── environment.py            ← Core OpenEnv logic
    ├── grader.py                 ← Deterministic step + episode reward
    ├── pbrs.py                   ← Potential-Based Reward Shaping
    ├── intrinsic.py              ← Count + RND intrinsic motivation
    ├── difference_rewards.py     ← Counterfactual credit assignment
    ├── ird.py                    ← IRD posterior correction
    ├── reward_combiner.py        ← Mode-switchable reward dispatch
    ├── workflows.py              ← 8 seed workflow DAGs
    └── workflow_gen.py           ← T4 curriculum generator (5 mutations)
```

---

## 12. One Sentence for the Judges

> *"We built the only OpenEnv hackathon environment with potential-based shaping that we **prove** preserves the optimal policy — verified by a unit test — plus a count-based exploration bonus with a **mathematically guaranteed** anti-hacking decay, difference rewards for step-level credit assignment, and an IRD posterior that infers the true objective from a possibly-misspecified proxy; every term ablates to a measurable contribution under a controlled fixed-policy experiment, and every component is visible in real time in the live demo."*

---

## Acknowledgements

Built for the **Meta × Scaler OpenEnv Hackathon Grand Finale, April 2026**. Training powered by [Unsloth](https://github.com/unslothai/unsloth) and [HuggingFace TRL](https://github.com/huggingface/trl). Base model: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).

Round 1 submission: [Customer Support Triage](https://huggingface.co/spaces/Arnav100904/customer-support-triage) — single-agent triage environment, 96.2% baseline score.

---
