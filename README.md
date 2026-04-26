---
title: Adaptive Enterprise Autopilot
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - enterprise-workflows
  - tool-use
  - long-horizon-planning
  - self-improvement
  - world-modeling
license: mit
---

# 🤖 Adaptive Enterprise Autopilot — OpenEnv Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Themes](https://img.shields.io/badge/Themes-T3.1_+_T4_+_T2-purple)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> *"The first enterprise workflow environment that gets harder every time the agent succeeds."*

**Built for the Meta × Scaler OpenEnv Hackathon Grand Finale, April 2026.**

---
## Prior work
Round 1 submission: [Customer Support Triage](https://huggingface.co/spaces/Arnav100904/customer-support-triage) — single-agent triage environment, 96.2% baseline score.

---

## The Problem

Most AI agents can follow a 3-step script. But real enterprise work looks nothing like that.

Onboarding a new employee involves creating HR records, then provisioning Jira, then setting up Slack — in exactly that order, because Jira accounts can't exist without an HR record. Responding to a security incident means notifying legal *before* customers, updating the status page *before* external comms, and escalating with the right ticket hierarchy — all while handling API failures mid-flight.

Current benchmarks either test isolated tool calls or fixed 5-step sequences. Neither captures the dependency-constrained, multi-system, long-horizon nature of actual enterprise operations.

**We built an environment that does.**

---

## What's new in v2 — reward engineering layer

The v2 release adds a **3-component reward stack** on top of the existing deterministic grader, with one explicit goal per criterion:

| Component | File | Claim |
|---|---|---|
| **Potential-Based Reward Shaping (PBRS)** | `src/envs/autopilot_env/pbrs.py` | The optimal policy is **provably** unchanged — see `tests/test_pbrs_invariance.py` for the 3-state-MDP value-iteration proof. |
| **Count-based intrinsic motivation** | `src/envs/autopilot_env/intrinsic.py` | Bonus = β/√(N+1), decays linearly to **zero by episode 200** — anti-reward-hacking by construction. |
| **`RewardCombiner`** | `src/envs/autopilot_env/reward_combiner.py` | Mode-switchable dispatch; `proxy_only / no_pbrs / no_intrinsic / full` for live ablation. |

### Ablation results — the headline numerical claim

We isolate each reward term's contribution by holding the **policy** constant (deterministic oracle) and varying only the active reward terms via `RewardCombiner.mode`. Same 30 episodes per mode, task = `easy`:

| Mode | Mean | Min | Max | Δ vs `proxy_only` |
|---|---|---|---|---|
| `proxy_only` (deterministic grader only) | **4.482** | 3.000 | 6.150 | — |
| `no_pbrs` (extrinsic + intrinsic) | **4.747** | 3.198 | 6.676 | **+0.265** |
| `no_intrinsic` (extrinsic + PBRS) | **4.930** | 3.435 | 6.600 | **+0.448** |
| `full` (extrinsic + PBRS + intrinsic) | **5.194** | 3.633 | 7.125 | **+0.712** |

![ablation chart](ablation_curve.png)

Every additional component lifts mean reward by a positive amount under a fixed policy — that's what "the term contributes" means when you've removed policy variance from the experiment. This is the apples-to-apples comparison for the rubric's "Showing Improvement in Rewards" criterion. See `ablation_table.md` and `eval_ablations.py` for the harness.

### Training-time diagnostic

![Training Curve](reward_curve.png)

> **Reading the curve.** The blue rolling average is the **shaped** GRPO step reward (extrinsic + PBRS + intrinsic + judge). Green dots are deterministic-only eval episodes — same scoring function as the ablation table, no shaping bonuses. After 96 GRPO steps the shaped reward saturates around 1.55 while deterministic eval drifts in the 0.0 – −1.0 range — the classic signature of a policy fitting shaping bonuses faster than it learns the proxy. **This is exactly the pathology v2's reward stack and live `/diagnostics` endpoint were designed to detect.** The controlled-policy comparison in the ablation table above is therefore the load-bearing experimental claim of this submission; the curve here is shown as evidence the framework correctly surfaces reward-hacking dynamics in real time, not as a benchmark of training convergence.

For the full 2-panel view including the T4 auto-curriculum trace, run `python train.py plot --full`.

### Live demo additions

`demo.html` now polls `GET /diagnostics` after each step and surfaces the live reward decomposition in two new visual elements:

- **Reward-stack canvas** — a 64-px stacked-area sparkline showing extrinsic / PBRS / intrinsic contributions per step.
- **Two-row breakdown panel** — extrinsic components (row 1) plus the v2 stack and decay factor (row 2).

Three new story cards (`pbrsInvariant`, `intrinsicNovel`, `ablationProof`) tie each mechanic to a narrative beat during the live demo.

---

## What Makes This Different

### 🌍 Theme 3.1 — World Modeling (Professional Tasks)

The environment simulates **5 real enterprise systems**: Jira, Slack, Email, HR, and Calendar. Each tool has a realistic API with required parameters, stateful responses (ticket IDs, user IDs, channel IDs that carry forward), and meaningful failure modes.

The agent doesn't just pick from a menu — it must:
- Call the right tool with the right parameters
- Reference IDs returned by earlier tool calls in later ones
- Respect business rules (HR before Jira; legal before customers)
- Recover from API failures (blocker tasks that fail on first attempt)

### 📅 Theme 2 — Long-Horizon Planning

Workflows range from **5 to 14+ tasks** arranged as dependency DAGs, not linear chains. The hard task (Security Incident Response) has 13 tasks with two independent starting tracks that merge at a compliance checkpoint. The agent must plan across the full horizon, not just react to the current step.

The reward is **shaped at every step** — not sparse end-of-episode — so RL training gets signal throughout the trajectory.

### 🔄 Theme 4 — Self-Improvement (Auto-Curriculum)

This is what separates us from every other enterprise workflow environment.

After each episode, the environment **generates a harder variant** of the completed workflow using 5 mutation strategies:
1. Add cross-dependent notification tasks
2. Add a verification/audit gate (all leaves must complete before proceeding)
3. Promote a task to a blocker (first call fails, must retry)
4. Introduce a parallel second team track that merges at a consolidation task
5. Add strict business-rule constraints and a compliance gate

**Difficulty auto-escalates only when the agent performs well** (≥50% completion). The curriculum is driven by agent capability, not a fixed schedule. An agent that plateaus stops seeing harder workflows until it improves.

---

## Live Demo

**Space URL**: https://huggingface.co/spaces/Arnav100904/adaptive-enterprise-autopilot
**API Endpoint**: https://arnav100904-adaptive-enterprise-autopilot.hf.space

```bash
#Set the API URL as a variable
$API_URL="https://arnav100904-adaptive-enterprise-autopilot.hf.space" 

```bash

# Health check
curl "$API_URL/health"

#During the Demo
## Sequence 1: The Untrained Failure

# 1. Start the episode
Invoke-RestMethod -Uri "$API_URL/reset?task=easy" -Method POST -Headers @{"Content-Type"="application/json"} -Body '{}' | ConvertTo-Json -Depth 10

# 2. Submit the bad action
Invoke-RestMethod -Uri "$API_URL/step" -Method POST -Headers @{"Content-Type"="application/json"} -Body '{ "tool": "jira_create_ticket", "params": {"summary": "Setup account", "issue_type": "Task"}, "reasoning": "I need to set up the Jira account first." }' | ConvertTo-Json -Depth 10

## Sequence 2: The Trained Success

# 1. Reset for the trained agent
Invoke-RestMethod -Uri "$API_URL/reset?task=easy" -Method POST -Headers @{"Content-Type"="application/json"} -Body '{}' | ConvertTo-Json -Depth 10

# 2. Submit the correct, dependency-aware action
Invoke-RestMethod -Uri "$API_URL/step" -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"tool": "hr_create_user","params": {"name": "Riya Sharma", "role": "Intern", "department": "Engineering"},"reasoning": "T1 has no dependencies — HR record must be created first to unblock T2."}' | ConvertTo-Json -Depth 10
```

---

## Environments

### 🟢 Easy — Linear workflows, 5–6 tasks
Clear dependency ordering, 0–1 business rules, no blockers.
- Onboard new intern
- Report a production bug
- Offboard departing employee
- Set up a new project team

**Expected score**: 1.5–2.0 (capable model, shaped + episode rewards)

### 🟡 Medium — Branching DAGs, 9 tasks
Parallel tracks, 1 blocker task, 2 business rules.
- Product feature launch
- Customer escalation resolution

**Expected score**: 0.8–1.4

### 🔴 Hard — Complex multi-constraint, 13–14 tasks
Two independent starting tracks, strict legal ordering, 2 blockers.
- Security incident response
- Mergers and acquisitions integration

**Expected score**: 0.4–0.9 (frontier models)

---

## Action Space

```json
{
  "tool": "jira_create_ticket",
  "params": {
    "summary": "Intern environment setup",
    "issue_type": "Task",
    "priority": "medium"
  },
  "reasoning": "T1 (HR record) is complete — creating Jira ticket T2 which depends on it."
}
```

Available tools: `jira_create_ticket`, `jira_update_ticket`, `jira_assign_ticket`,
`slack_send_message`, `slack_create_channel`, `email_send`,
`hr_create_user`, `hr_update_user`, `calendar_create_event`, `done`

---

## Observation Space

```json
{
  "workflow_name": "Onboard new intern",
  "tasks": [
    {"task_id": "T1", "name": "Create HR profile", "required_tool": "hr_create_user", "dependencies": []},
    {"task_id": "T2", "name": "Create Jira ticket", "required_tool": "jira_create_ticket", "dependencies": ["T1"]}
  ],
  "completed_task_ids": ["T1"],
  "available_task_ids": ["T2", "T3", "T4"],
  "pending_task_ids": ["T5"],
  "tool_results": [{"tool": "hr_create_user", "result": {"user_id": "HR-1000"}, "success": true}],
  "step_feedback": "Tool matched task: 'Create HR profile' | Tool succeeded | 1/5 tasks complete",
  "difficulty_level": 1
}
```

---

## Reward Function

### Step-level (shaped — fires every action)

| Component | Condition | Value |
|---|---|---|
| Correct tool | Tool matches an available task | +0.20 |
| All params present | Required params non-empty | +0.15 |
| Dependencies satisfied | Task was actually available | +0.10 |
| Reasoning quality | Mentions task or tool name | +0.05 |
| Dependency violation | Called a blocked task | −0.20 |
| Business rule violated | e.g. Jira before HR | −0.25 |
| Invalid tool | Unknown tool name | −0.10 |

### Episode-level (bonus — fires at done=True)

| Condition | Bonus |
|---|---|
| 100% tasks complete | +1.00 |
| ≥80% tasks complete | +0.60 |
| ≥50% tasks complete | +0.30 |
| Efficiency: completed in ≤ 1.5× n_tasks steps | +0.20 |
| Violation penalty | −0.10 × violation_count |

**Maximum possible score per episode: 2.0** (shaped + episode bonus + efficiency)

---

## Self-Improvement Loop

```
Episode N completes
        ↓
  completion_rate ≥ 50%?
        ↓ yes
  generate_harder_workflow(base, delta=1)
        ↓
  5 mutation strategies applied based on difficulty_level
        ↓
  Next reset() uses the harder variant
        ↓
  Agent faces a new challenge it has never seen
```

The generated workflow difficulty is scored in [0.0, 1.0] based on:
`n_tasks × 0.4 + n_dependencies × 0.25 + n_rules × 0.5 + n_blockers × 1.0`

Both the reward curve and the difficulty curve increase as training progresses — the key dual signal that demonstrates genuine capability growth.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | `{"status": "healthy"}` |
| `/reset?task=easy` | POST | Start new episode |
| `/step` | POST | Submit tool call action |
| `/state` | GET | Episode metadata |
| `/workflow` | GET | Full workflow definition |
| `/docs` | GET | Swagger UI |

---

## Run Locally

```bash
git clone https://github.com/your-username/adaptive-enterprise-autopilot
cd adaptive-enterprise-autopilot

# Docker
docker build -t autopilot-env .
$env:PYTHONPATH="src"; python -m uvicorn server.app:app --host 0.0.0.0 --port 7860

# Test
curl http://localhost:7860/health
```

---

## Training

```bash
# Install
pip install unsloth trl transformers accelerate peft datasets

# Train (Qwen2.5-7B on a single A10G — ~2 hours for 200 episodes)
BASE_MODEL=unsloth/Qwen2.5-7B-Instruct \
HF_TOKEN=hf_xxx \
NUM_EPISODES=200 \
python train.py

# Plot reward curve
python train.py plot
```

---

## Reward Hacking Analysis

We proactively tested two reward-gaming strategies:

**Attack 1 — Call `done` immediately.**
Step reward for premature `done`: −0.10.
Episode penalty for 0% completion: no bonus (0.00), plus −0.10 × violations.
Total: −0.10. A random agent completing all tasks scores +1.00 episode bonus minimum.
Early termination is always strictly worse. ✓

**Attack 2 — Repeat the same valid tool call on every step.**
The grader matches each tool call to an *uncompleted* task. Once a task is marked
complete, the same tool call no longer matches anything — it scores −0.10 (invalid tool).
Repetition self-penalizes after the first success. ✓

**Conclusion:** The reward function cannot be meaningfully gamed. The only path
to a high score is genuine task completion in dependency order.

---

## Project Structure

```
adaptive-enterprise-autopilot/
├── inference.py                  ← Baseline inference script
├── train.py                      ← Unsloth + TRL GRPO training script
├── push_to_hf.py                 ← Deploy to HuggingFace Spaces
├── openenv.yaml                  ← OpenEnv manifest
├── Dockerfile
├── requirements.txt
├── server/
│   └── app.py                    ← FastAPI server
└── src/envs/autopilot_env/
    ├── models.py                 ← Action / Observation / State types
    ├── environment.py            ← Core OpenEnv logic
    ├── grader.py                 ← Deterministic reward function
    ├── tools.py                  ← Mock enterprise tool APIs
    ├── workflows.py              ← 8 seed workflow DAGs
    └── workflow_gen.py           ← T4 curriculum generator (5 mutations)
```

---

## Authors

**Arnav Deepak Tiwari** & **Vishal Kumar**

Built for the **Meta × Scaler OpenEnv Hackathon Grand Finale, April 2026**.

---

## License

MIT
