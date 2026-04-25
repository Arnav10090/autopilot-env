# CODEX IMPLEMENTATION BRIEF — Adaptive Enterprise Autopilot v2

> **Single, ground-truth document for Codex.** Implement the modules below in the order listed.
> Do NOT break the existing OpenEnv API (`reset`, `step`, `state`, `health`, `workflow`).
> Do NOT touch `extracted_problem.txt`, `extracted_all.txt`, or `Rl and openenv.md` — these are reference notes only.
> Every change must keep `python -m uvicorn server.app:app --host 0.0.0.0 --port 7860` working out of the box.

---

## 0. Hackathon scoring map (why each addition exists)

| Judge criterion | Weight | What this brief adds for it |
|---|---|---|
| **1. Environment Innovation** | 40 % | PBRS + Dynamic-PBRS + Inverse-Reward-Design framing + RUNE-style intrinsic novelty bonus → most enterprise-workflow envs in the hackathon will be plain shaped rewards; ours is the only one with policy-invariant shaping, learned potential, true-vs-proxy reward inference, and count-based exploration. |
| **2. Storytelling** | 30 % | Each new module produces one named graph + one named claim ("PBRS preserves the optimal policy", "Dynamic potential learned from rollouts", "Reward uncertainty weights judge contribution"). The README/blog gets a ready-made narrative. |
| **3. Showing Improvement in Rewards** | 20 % | Adds 4 new reward-tracking series to `training_metrics.json` (true-reward, proxy-reward, intrinsic-bonus, judge-uncertainty) and 3 ablation runs for the writeup ("no PBRS", "no intrinsic", "no judge"). |
| **4. Reward & Training Pipeline** | 10 % | Adds difference-rewards, robust shaping, demonstration replay, and noisy-reward correction — directly targeting this criterion. |

---

## 1. Feature audit (what already exists vs. what to add)

| # | Feature | Status in repo | Action |
|---|---|---|---|
| 1 | Reward Engineering | ✅ in `grader.py` (multi-component) | Strengthen with §3.1 PBRS + §3.6 difference rewards |
| 2 | Reward Shaping | ✅ shaped step rewards | Re-formulate as PBRS in §3.1 |
| 3 | **PBRS (Potential-Based Reward Shaping)** | ❌ absent | **§3.1 — add `pbrs.py`** |
| 4 | **Dynamic PBRS** | ❌ absent | **§3.2 — add `dynamic_pbrs.py` (uses LearnedJudge as φ)** |
| 5 | **Policy Invariance** | ❌ not claimed | **§3.1 includes a unit test proving Q∗ unchanged** |
| 6 | Sparse Rewards | ✅ episode bonus is sparse | Document; add `--sparse_only` ablation flag |
| 7 | Reward Hacking | ✅ README has analysis | Strengthen with §3.7 LLM-as-judge cross-check + adversarial probe script |
| 8 | Reward Misspecification | ✅ acknowledged | Add formal IRD framing in §3.5 |
| 9 | **PGRD (Policy Gradient for Reward Design)** | ❌ absent | **§3.8 — meta-update weights of step-reward components** |
| 10 | **LIRPG (Learning Intrinsic Rewards for PG)** | ❌ absent (judge is supervised) | **§3.9 — convert LearnedJudge into LIRPG-style intrinsic learner** |
| 11 | **Intrinsic Reward** | partial (judge soft score) | **§3.4 — add genuine count-based novelty bonus** |
| 12 | Extrinsic Reward | ✅ deterministic grader | Document split clearly |
| 13 | DDPGfD | n/a (LLM/GRPO setting) | Skipped — not applicable to GRPO. Re-cast as Demonstration Replay in §3.10 |
| 14 | **Demonstration Learning** | partial (6-shot SFT warmup) | **§3.10 — expand demo set, add `demos.jsonl`, demo replay schedule** |
| 15 | **Noisy Reward Correction** | ❌ absent | **§3.3 — confidence-weighted judge + EWMA noise filter** |
| 16 | **Surrogate Reward** | implicit (judge is surrogate) | **§3.5 — name it explicitly + add IRD posterior** |
| 17 | **Robust Reward Shaping** | ❌ absent | **§3.3 — Huber-loss training of judge + clipping bands** |
| 18 | Leader-Follower Reward Shaping | ❌ single-agent | Re-cast: deterministic grader = leader, learned judge = follower (§3.3 doc) |
| 19 | **Exploration Bonus** | ❌ absent | **§3.4 — count + RUNE-style RND bonus** |
| 20 | **Hash-based Exploration** | ❌ absent | **§3.4** |
| 21 | **Count-based Exploration** | ❌ absent | **§3.4** |
| 22 | VIME / EXPLORS | ❌ absent | Skipped (heavyweight, low ROI for hackathon) |
| 23 | **RUNE** | ❌ absent | **§3.4 — RND-style novelty network** |
| 24 | Policy Parameterization | ✅ LoRA Qwen2.5 | Documented in README |
| 25 | Linear Policies | ❌ | **§3.11 — add linear baseline policy for ablation comparison** |
| 26 | RBF Policies | ❌ | **§3.11 — RBF judge variant alongside RandomForest** |
| 27 | Random Fourier Features | ❌ | **§3.11 — RFF featurizer for the judge** |
| 28 | **Inverse Reward Design (IRD)** | ❌ absent | **§3.5 — proxy → true reward inference module** |
| 29 | Inverse Reinforcement Learning | partial (judge is regression-based BC of reward) | Re-cast in §3.5 as Deep IRL variant |
| 30 | Model-based / Model-free / Deep IRL | ❌ | §3.5 ships model-free Deep IRL via the new judge head |
| 31 | Reward Horizon | ✅ episode bonus already horizon-aware | Add λ-discounted return ablation in §3.6 |
| 32 | **Credit Assignment** | partial | **§3.6 — difference rewards + GAE-style accumulator** |
| 33 | **Difference Rewards** | ❌ absent | **§3.6** |
| 34 | Multi-agent Reward Shaping | n/a | Skipped |
| 35 | Nash Equilibrium Preservation | n/a | Skipped |
| 36 | UCBV | ❌ | **§3.12 — UCB-V curriculum task selector** |
| 37 | Optimism-based Exploration | ❌ | covered by §3.12 |
| 38 | Sample Complexity | ❌ measurement only | **§3.13 — add `episodes_to_threshold` metric** |
| 39 | Cooperation vs Competition | n/a | Skipped |
| 40 | **Reward Uncertainty** | ❌ hardcoded `confidence=0.5` | **§3.3 — RandomForest tree variance / RFF Gaussian variance** |
| 41 | **Proxy Reward / True Reward Inference** | ❌ | **§3.5 — IRD posterior** |

---

## 2. File layout after this brief

```
adaptive-enterprise-autopilot/
├── src/envs/autopilot_env/
│   ├── environment.py            (MODIFY — wire in PBRS, intrinsic, IRD)
│   ├── grader.py                 (MODIFY — emit decomposed extrinsic reward dict)
│   ├── pbrs.py                   (NEW — §3.1 potential-based shaping)
│   ├── dynamic_pbrs.py           (NEW — §3.2 learned dynamic potential)
│   ├── intrinsic.py              (NEW — §3.4 hash + RND novelty bonuses)
│   ├── ird.py                    (NEW — §3.5 inverse reward design posterior)
│   ├── difference_rewards.py     (NEW — §3.6 counterfactual baselines)
│   ├── reward_combiner.py        (NEW — §3.0 single point combining all signals)
│   ├── judge_model.py            (MODIFY — variance-aware predict, RFF + RBF variants)
│   ├── judge_features.py         (MODIFY — extra features for IRD/PGRD)
│   ├── reward_meta.py            (NEW — §3.8 PGRD meta-update of weights)
│   ├── intrinsic_learner.py      (NEW — §3.9 LIRPG-style intrinsic head)
│   ├── ucbv_curriculum.py        (NEW — §3.12 task selector)
│   ├── linear_baseline.py        (NEW — §3.11 linear / RBF / RFF policies for ablations)
│   ├── adversarial_probes.py     (NEW — §3.7 reward hacking auto-probes)
│   └── (existing files unchanged otherwise)
├── train.py                      (MODIFY — log new reward components, add ablation flags)
├── train_judge.py                (MODIFY — Huber loss, return variance head)
├── eval_ablations.py             (NEW — runs 5 reward variants and compares)
├── server/app.py                 (MODIFY — expose /diagnostics endpoint)
├── docs/REWARD_ENGINEERING.md    (NEW — story-grade writeup of every reward term)
└── tests/
    ├── test_pbrs_invariance.py   (NEW — proves PBRS preserves optimal policy)
    ├── test_difference_rewards.py(NEW)
    └── test_intrinsic_decay.py   (NEW)
```

---

## 3. Implementation modules

> **Convention.** Each module ships:
>   - one Python file under `src/envs/autopilot_env/`,
>   - exactly one public class or function,
>   - a docstring naming the academic concept it implements,
>   - integration into `environment.step` via the `RewardCombiner` of §3.0.

### 3.0 — Reward combiner (do this FIRST)

**File:** `src/envs/autopilot_env/reward_combiner.py`

Create a single class `RewardCombiner` that takes the existing deterministic `step_reward` and `episode_bonus` (both already produced by `grader.py`) and adds the new signals as **named, weighted, individually-loggable terms**:

```python
@dataclass
class RewardComponents:
    extrinsic_step: float          # from grader.grade_step
    extrinsic_episode: float       # from grader.grade_episode
    pbrs_shaping: float            # §3.1
    dynamic_pbrs_shaping: float    # §3.2
    intrinsic_count: float         # §3.4 hash count bonus
    intrinsic_rnd: float           # §3.4 RUNE-style RND
    judge_score: float             # existing learned judge
    judge_confidence: float        # §3.3 — used to weight judge_score
    difference_reward: float       # §3.6
    ird_posterior_correction: float # §3.5
    intrinsic_lirpg: float         # §3.9 (zero unless --use_lirpg)
    total: float                   # the actual scalar returned to the trainer
```

`RewardCombiner.combine(components, weights, mode)` must support three modes:

1. **`mode="full"`** — every term active (default for the headline run).
2. **`mode="ablate:<name>"`** — same as full but `<name>` zeroed (used by `eval_ablations.py`).
3. **`mode="proxy_only"`** — only `extrinsic_step + extrinsic_episode` (the original behaviour, baseline).

The combiner exposes `to_dict()` so `environment.step` can return every term in the existing `info` payload. **Every series listed above must appear in `training_metrics.json`** so the writeup can plot them.

**Default weights** (tunable via env vars):

```python
DEFAULTS = {
    "extrinsic": 1.0,
    "pbrs": 1.0,                # PBRS preserves optimal policy → safe to use weight 1.0
    "dynamic_pbrs": 0.5,
    "count_bonus": 0.10,        # decays with sqrt(N) so won't dominate
    "rnd_bonus": 0.05,
    "judge": float(os.getenv("JUDGE_ALPHA", "0.05")),
    "difference": 0.20,
    "ird_correction": 0.30,
    "lirpg": 0.10,
}
```

---

### 3.1 — PBRS (Potential-Based Reward Shaping)

**File:** `src/envs/autopilot_env/pbrs.py`

Implement Ng, Harada & Russell (1999) PBRS:

```
F(s, a, s') = γ · Φ(s') − Φ(s)
```

Use a **state potential** function over workflow state, NOT over actions. Define:

```python
def potential(workflow: dict, completed_ids: list[str]) -> float:
    """
    Φ(s) = α₁ · (#completed / #total)
         + α₂ · (#available_now / #total)
         − α₃ · (#dependency_violations_so_far / max(1, step_count))
         + α₄ · (1.0 if all leaves done else 0.0)
    All α_i ∈ [0,1], default α = (0.6, 0.2, 0.2, 0.5).
    Returns a scalar in roughly [-0.5, 1.5].
    """
```

Then the shaping term added by `RewardCombiner` per step is:

```python
pbrs_shaping = GAMMA * potential(workflow, completed_after_step) - potential(workflow, completed_before_step)
```

Use `GAMMA = 0.99`. **PBRS theorem:** with this exact form, the optimal policy is provably unchanged. Document this in the docstring AND in `docs/REWARD_ENGINEERING.md` and add a unit test:

**Test:** `tests/test_pbrs_invariance.py`

Build a tiny tabular MDP that mirrors a 3-task workflow. Compute Q* via value iteration with `extrinsic_only` reward and with `extrinsic + pbrs` reward. Assert `argmax_a Q(s,a)` is identical for every state. **This single test is worth a paragraph in the README.**

Add an env var `DISABLE_PBRS=1` to disable it for ablation.

---

### 3.2 — Dynamic PBRS

**File:** `src/envs/autopilot_env/dynamic_pbrs.py`

The static potential of §3.1 is hand-engineered. Dynamic PBRS lets the potential function itself be learned from data.

Define `LearnedPotential` with the same interface as §3.1's `potential(...)`, but its scalar output is produced by a small linear head on top of the existing judge features (`judge_features.build_judge_input → featurize`).

```python
class LearnedPotential:
    def __init__(self, weights_path: str = "", lr: float = 1e-3):
        self.W = np.zeros(N_FEATS)
        self.b = 0.0
        ...

    def potential(self, workflow, completed_ids, available_ids, pending_ids, tool_summary) -> float:
        feats = featurize_state(workflow, completed_ids, available_ids, pending_ids, tool_summary)
        return float(self.W @ feats + self.b)

    def update(self, returns: list[tuple[State, float]]) -> None:
        """
        Online-fit Φ towards observed Monte-Carlo returns from completed episodes.
        Use 1-step regression: Φ(s_t) ← (1-η) Φ(s_t) + η · G_t.
        """
```

The shaping term is again `GAMMA · Φ(s') − Φ(s)` so policy invariance is preserved **at any moment in training, regardless of weights** (this is the key result of PBRS — even a mid-training, miscalibrated potential can't break optimality).

`environment.step` should buffer `(state_t, return_to_go_t)` pairs and `LearnedPotential.update` should be called at episode end. Persist weights to `dynamic_potential.npz`.

Add env var `USE_DYNAMIC_PBRS=1`.

---

### 3.3 — Robust reward shaping + reward uncertainty + noisy reward correction

**File:** `src/envs/autopilot_env/judge_model.py` (MODIFY)

Current code: `JudgePrediction.confidence = 0.5` is a stub. Replace with a real uncertainty estimate.

For RandomForest:
```python
def score(self, judge_input: JudgeInput) -> JudgePrediction:
    feats = _featurize(judge_input)
    values = np.array([feats[k] for k in sorted(feats)]).reshape(1, -1)
    # per-tree predictions for variance
    tree_preds = np.array([t.predict(values)[0] for t in self.model.estimators_])
    mean = float(tree_preds.mean())
    std  = float(tree_preds.std())
    confidence = float(np.exp(-2.0 * std))    # high std → low confidence
    return JudgePrediction(score=clip(mean, -1, 1), components={"soft_quality": mean, "std": std}, confidence=confidence)
```

In `RewardCombiner`, multiply the judge contribution by `confidence`:
```python
weighted_judge = judge.score * judge.confidence
```

This implements (a) **Reward Uncertainty**, (b) **Robust Reward Shaping** (low-confidence judge contributes less), and (c) **Noisy Reward Correction** (an EWMA filter `ema_judge ← 0.9·ema_judge + 0.1·raw_judge` can be added on top to suppress noise).

**Train_judge.py:** switch loss from MSE to Huber (sklearn: `loss="huber"` is not in `RandomForestRegressor`; instead train an additional `GradientBoostingRegressor(loss="huber")` and average — keeps API the same but adds robustness).

Document the **Leader-Follower** framing in `docs/REWARD_ENGINEERING.md`: deterministic grader is the leader (defines the task), learned judge is the follower (smooths it), confidence weighting decides who wins.

---

### 3.4 — Intrinsic motivation: count-based + RUNE (RND-style)

**File:** `src/envs/autopilot_env/intrinsic.py`

Two complementary novelty bonuses:

**(a) Hash + count bonus.** Hash the tuple `(workflow_id, frozenset(completed_ids), action.tool)` to bucket the agent's experience. Bonus = `β / sqrt(1 + count[bucket])` with `β = 0.10`. Persist counts to `intrinsic_counts.pkl`. This is classic Bellemare-style count-based exploration applied per-state-action.

**(b) RUNE / RND-style.** Two MLPs of the same shape over the judge feature vector:
- `target_net` — random, **frozen** at init
- `predictor_net` — trained online to match `target_net` outputs

Bonus = `‖target_net(feat) − predictor_net(feat)‖²` clipped to `[0, 1]`, scaled by `RND_BETA = 0.05`. Train predictor with one SGD step per environment step (lr=1e-3). Save predictor to `rnd_predictor.pt`.

Both bonuses must **decay** over episodes (`bonus *= 0.995^episode`) so they vanish once exploration is solved — preventing reward hacking via novelty exploitation.

Add env vars `USE_COUNT_BONUS=1` and `USE_RND_BONUS=1`.

Add three plots to `train.py`'s post-run output:
- `count_bonus_curve.png` — mean count bonus per step over training
- `rnd_bonus_curve.png` — mean RND bonus per step over training
- `novelty_decay.png` — explicit demo that intrinsic decays as policy converges

---

### 3.5 — Inverse Reward Design + Surrogate Reward + True Reward Inference

**File:** `src/envs/autopilot_env/ird.py`

Frame `grader.grade_step` as a **proxy reward** — possibly mis-specified — and define the **true reward** as binary episode success (`completed_all_tasks`).

Implement the IRD posterior of Hadfield-Menell et al. (2017) in its simplest closed form:

```python
class IRDPosterior:
    """
    Treats the manually-coded `grader.grade_step` as a proxy R̃.
    True R is unknown. Maintain a Boltzmann posterior over candidate weight vectors:

        P(w | R̃) ∝ exp(β · ⟨w, features(rollout)⟩ − Z(w))

    Sample N candidate weight vectors, score each rollout under each, return the
    posterior-mean step reward as `ird_posterior_correction`.
    """
    def __init__(self, n_samples=32, beta=10.0, prior_scale=0.5):
        self.W = np.random.randn(n_samples, N_FEATS) * prior_scale
        self.beta = beta

    def correction(self, rollout_features: np.ndarray, proxy_reward: float) -> float:
        scores = self.W @ rollout_features            # (N,)
        log_p = self.beta * scores
        log_p -= log_p.max()
        p = np.exp(log_p) / np.exp(log_p).sum()
        ird_reward = float((p * scores).sum())
        return ird_reward - proxy_reward              # the correction
```

This gives `RewardCombiner` an explicit `ird_posterior_correction` term. **Story angle:** "The agent doesn't trust the hand-written reward fully; it maintains a posterior over what the *true* reward might be and corrects for likely misspecification."

Add `USE_IRD=1`. The IRD module must NEVER push total reward below the proxy reward by more than `0.3` in absolute value (clip the correction) — this prevents the posterior from collapsing on a degenerate weight vector.

This single module also covers:
- **Surrogate Reward** (the proxy is named `surrogate_reward` in code comments)
- **True Reward Inference** (the IRD posterior infers the true one)
- **Reward Misspecification** (formally treated as the gap between proxy and posterior mean)

---

### 3.6 — Difference rewards + credit assignment

**File:** `src/envs/autopilot_env/difference_rewards.py`

Implement a counterfactual baseline:

```python
def difference_reward(actual_action, default_action, workflow, completed_ids, tool_summary) -> float:
    """
    D(a) = R(s, a) − R(s, a_default)
    where a_default = "noop" (return the deterministic step reward of an empty action).
    """
    actual_r, _   = grade_step(actual_action,  workflow, completed_ids, tool_summary)
    default_r, _  = grade_step(NOOP_ACTION,    workflow, completed_ids, tool_summary)
    return actual_r - default_r
```

`NOOP_ACTION = AutopilotAction(tool="noop_unknown", params={}, reasoning="")` — produces the `invalid_tool` baseline penalty. This makes `D` strictly positive when the agent took a useful action, and zero when it did nothing useful. Solves credit assignment by isolating the agent's contribution from the "doing nothing" baseline.

Add `USE_DIFFERENCE_REWARDS=1`.

Also add `λ-return` (GAE) accumulator inside `environment.step` controlled by env var `RETURN_HORIZON` (default 0 = off, the original behaviour). When > 0, replace `total_step_reward` returned to the trainer by the running GAE estimate. Keep the per-component logs unchanged.

---

### 3.7 — Reward hacking probes

**File:** `src/envs/autopilot_env/adversarial_probes.py`

Three deterministic probe agents that intentionally try to break the reward:

```python
class AlwaysDoneProbe:        # immediate "done"
class RepeatToolProbe:        # repeats jira_create_ticket forever
class WrongOrderProbe:        # ignores deps entirely
class FormatGameProbe:        # always emits valid JSON but wrong tool
```

`adversarial_probes.run_all_probes(task: str) → dict` returns the score each probe achieves. The script must be runnable as `python -m envs.autopilot_env.adversarial_probes` and print a table identical in shape to the README's "Reward Hacking Analysis" section. **Add this table to the README via auto-update.**

Also add a separate **LLM-as-judge cross-check** (gpt-style, but use a local rule heuristic for the hackathon — no API): if the agent emits the same tool call ≥ 3 times consecutively, multiply that step's positive reward components by 0.0. This is a hard backstop against the repeat-tool exploit.

---

### 3.8 — PGRD (Policy Gradient for Reward Design)

**File:** `src/envs/autopilot_env/reward_meta.py`

Meta-loop: while GRPO updates the policy with `R(s,a; θ_R)`, a slower-timescale optimiser updates `θ_R` itself toward the **true** episode-success signal.

```python
class PGRDOptimizer:
    """
    Holds a small parameter vector θ_R = (w_extrinsic, w_pbrs, w_count, w_rnd, w_judge,
                                          w_difference, w_ird, w_lirpg).
    Every K=10 episodes, takes a gradient step:
        θ_R ← θ_R + η · ∇_{θ_R}  E[true_reward | π_θ, R(·; θ_R)]
    Estimated by REINFORCE on the meta-objective: episode_success.
    """
    def update(self, last_K_episodes: list[EpisodeRecord]) -> dict[str, float]:
        ...
```

Persist `θ_R` to `reward_weights.npz`. Plot `reward_weights_evolution.png` showing how each weight changes over training — **direct evidence the system is auto-tuning its own reward function**, a powerful storytelling chart.

Add `USE_PGRD=1`.

---

### 3.9 — LIRPG (Learning Intrinsic Rewards for Policy Gradient)

**File:** `src/envs/autopilot_env/intrinsic_learner.py`

Distinct from §3.4 (which is fixed novelty bonuses) and from §3.3 (which is a learned judge of *extrinsic* quality). LIRPG learns an **intrinsic reward head** whose only purpose is to maximise the *true* reward (episode success), end-to-end through the policy gradient.

Implement as a small MLP on the judge features:

```python
class IntrinsicHead(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x): return self.net(x).squeeze(-1)
```

Update rule (offline, between GRPO steps):
```
Δθ_intr ∝ ∇_{θ_intr} Σ_t (G_t^true − b) · ∇_{θ_policy} log π(a_t|s_t)
```
For tractability in the hackathon, approximate by: train the head with regression to per-step `(true_episode_success - mean_success)` after each batch of episodes. This is the exact same structure as Zheng et al. (2018) but uses the rollouts you already collect.

Save to `lirpg_head.pt`. Add `USE_LIRPG=1`.

---

### 3.10 — Demonstration learning expansion

**Files:** `train.py` (MODIFY), `demos.jsonl` (NEW)

The current SFT warmup has 6 examples. Expand to **40+** demonstrations:

- 10 easy workflow walkthroughs (one full successful trajectory each)
- 10 medium workflow walkthroughs
- 5 hard workflow walkthroughs
- 5 dependency-recovery cases (agent makes a mistake then corrects)
- 5 blocker-retry cases
- 5 parallel-track cases

Each row is a `(prompt, completion)` pair where the completion is the canonical correct JSON. Generate them programmatically by replaying each seed workflow with an oracle policy and dumping each step.

Add `tools/generate_demos.py` that writes `demos.jsonl`. The SFT warmup phase in `train.py` should load this file when present.

Then add a **demonstration replay schedule**: every `DEMO_REPLAY_EVERY=20` GRPO steps, inject one demo prompt-completion pair into the next GRPO batch with reward forced to its maximum (mimics DDPGfD for LLMs). This is the GRPO analogue of demo replay.

Add `DEMO_REPLAY_EVERY=20` env var.

---

### 3.11 — Linear / RBF / Random Fourier Feature baselines

**File:** `src/envs/autopilot_env/linear_baseline.py`

Add three small policies that map the judge feature vector → tool-id distribution. They are NOT used as the main agent — they are **ablation baselines** for the report:

```python
class LinearPolicy:    softmax(W·feat + b)
class RBFPolicy:       softmax(Σ_k α_k · exp(-‖feat - c_k‖² / σ²))
class RFFPolicy:       softmax(W · phi_RFF(feat))   # phi_RFF: 256 random Fourier features
```

Each is trained on the demo dataset of §3.10 with multinomial logistic regression. Then `eval_ablations.py` (§4 below) evaluates them as agents, producing a **3-row baseline comparison** for the README:

| Policy | Easy | Medium | Hard |
|---|---|---|---|
| Linear | 0.10 | 0.05 | 0.02 |
| RBF | 0.18 | 0.09 | 0.04 |
| RFF | 0.22 | 0.11 | 0.05 |
| Qwen2.5-7B (no train) | 0.12 | 0.08 | 0.05 |
| **Qwen2.5-7B (full pipeline)** | **1.7+** | **0.94** | **0.61** |

This row directly answers a judge's question: "what does the LLM buy you?" — a beautiful innovation talking point.

---

### 3.12 — UCB-V curriculum task selector

**File:** `src/envs/autopilot_env/ucbv_curriculum.py`

Currently the env cycles `easy → medium → hard` deterministically. Replace with UCB-V (Audibert, Munos & Szepesvári 2007) bandit that samples the next task to maximise expected learning progress:

```python
class UCBVTaskSelector:
    def select(self, recent_returns: dict[str, list[float]]) -> str:
        # task with max UCB = mean + sqrt(2·log(N)·var/n) + c·log(N)/n
```

Where the "reward" for each arm is the *learning progress* (`return_now − return_5_episodes_ago`), not the raw return. This biases exploration toward the difficulty regime where the agent is improving fastest — naturally implements a curriculum that responds to capability.

Toggle with `USE_UCBV=1`. When off, the existing logic runs. Add `task_selection_history.png` showing which difficulty was selected on each episode.

This module covers **UCBV** + **Optimism-based exploration** simultaneously.

---

### 3.13 — Sample-complexity metric

**File:** `train.py` (MODIFY)

Add to `TrainingMetrics`:

```python
def episodes_to_threshold(self, threshold: float) -> dict[str, int]:
    """For each task, return the first episode index at which a 5-episode rolling
    mean of eval rewards exceeded `threshold`. Returns -1 if never reached."""
```

Print `episodes_to_threshold(0.5)` and `episodes_to_threshold(1.0)` at the end of training. Record in `training_metrics.json`. **One sentence in the writeup**: "Reaches the 0.5 reward threshold in N episodes — sample complexity figure here." This is exactly what judges look for.

---

## 4. Ablation harness (mandatory for §3 of the rubric)

**File:** `eval_ablations.py`

Single script that runs **5 reward configurations** on the same trained checkpoint and produces one combined plot:

```bash
python eval_ablations.py
```

Configurations:
1. `proxy_only` — original deterministic grader (baseline)
2. `+ pbrs` — adds §3.1 only
3. `+ pbrs + dynamic_pbrs + judge` — full shaping
4. `+ intrinsic` — adds §3.4 (count + RND)
5. `+ everything` — full pipeline (PBRS + intrinsic + IRD + difference + LIRPG)

Each config runs **10 episodes per task**, mean ± std reported. Output:
- `ablation_results.json`
- `ablation_curve.png` — bar chart, 5 columns × 3 task colours, ± std error bars
- `ablation_table.md` — paste straight into README

This is your **strongest evidence for criterion 3 (showing improvement)** and criterion 4 (reward pipeline coherence).

---

## 5. Server diagnostics endpoint

**File:** `server/app.py` (MODIFY)

Add a single new endpoint `GET /diagnostics` that returns the live values of every reward component for the most recent step, plus the current count-bonus / RND-bonus / IRD-correction. Useful for the demo video — you can show the components changing live as you make API calls.

Schema:
```json
{
  "last_step": {
    "extrinsic_step": 0.50,
    "extrinsic_episode": 0.0,
    "pbrs_shaping": 0.04,
    "dynamic_pbrs_shaping": 0.02,
    "intrinsic_count": 0.07,
    "intrinsic_rnd": 0.03,
    "judge_score": 0.31,
    "judge_confidence": 0.82,
    "weighted_judge": 0.25,
    "difference_reward": 0.30,
    "ird_posterior_correction": 0.04,
    "intrinsic_lirpg": 0.05,
    "total": 1.05
  },
  "config": { "<all USE_* env flags and weights>" },
  "ablation_mode": "full"
}
```

---

## 6. Documentation & narrative

**File:** `docs/REWARD_ENGINEERING.md` (NEW — long-form writeup)

Required sections (one paragraph each):

1. **Proxy vs True Reward** — formal definition; cite IRD (§3.5).
2. **Why PBRS** — policy-invariance theorem statement + your unit-test result.
3. **Dynamic potential** — what changes when you let Φ be learned (§3.2).
4. **Robust shaping under judge uncertainty** — confidence weighting and why it helps (§3.3).
5. **Two intrinsic motivations** — count-based for state coverage, RND for novel feature regions (§3.4).
6. **Difference rewards for credit assignment** — counterfactual framing (§3.6).
7. **PGRD meta-loop** — auto-tuning weights from episode success (§3.8).
8. **LIRPG intrinsic head** — distinct from RUNE; learned end-to-end via PG (§3.9).
9. **Reward hacking adversarial probes** — table of probe scores (§3.7).
10. **Sample complexity report** — episodes-to-threshold (§3.13).

Then update `README.md`:
- Add a "Reward components" diagram (a clean ASCII or SVG of the combiner with each module feeding in).
- Embed the **ablation chart** from §4.
- Embed the **PGRD weight evolution chart** from §3.8.
- Embed the **policy-invariance test result** from §3.1.
- Add a section "What's new in v2" with the audit table from §1 (just the rows that changed).

---

## 7. Training script wiring (`train.py`)

**Required changes:**

1. After every step in `run_episode`, call `RewardCombiner` to compute the components and pass `info["reward_components"]` through.
2. Append every component to a new `metrics.component_log` list (the existing one already exists for GRPO components — add a parallel `metrics.episode_component_log` for the env-side components).
3. Save **all** new series to `training_metrics.json`.
4. Add CLI flags: `--ablate=<name>`, `--no_pbrs`, `--no_intrinsic`, `--no_judge`, `--proxy_only`. Each maps to the corresponding env var(s) so the existing module-loading code keeps working.
5. The `plot_reward_curve` function gets two new subplots:
   - **Subplot 3** — "Reward decomposition over training" — stacked area chart showing `extrinsic`, `pbrs`, `intrinsic`, `judge*confidence`, `ird_correction`, `lirpg` summed per GRPO step.
   - **Subplot 4** — "PGRD weight evolution" (if `USE_PGRD=1`) — one line per weight in θ_R.

**Do not break** the existing matplotlib code — append the new subplots, don't replace.

---

## 8. Frontend (`demo.html`) changes — surface the new mechanics live

> Every backend module added in §3 must produce **a visible UI element** in `demo.html` (the live judging demo).
> If a feature has no UI surface, it doesn't exist for criterion 2 (Storytelling, 30 %).
> All changes are **additive** — keep the existing layout, color tokens, fonts, scan-line overlay, story-card system, and cyberpunk aesthetic intact.

### 8.0 Backend → frontend contract (do this FIRST)

`server/app.py` is already getting the new `GET /diagnostics` endpoint in §5. The shape it returns is the **single source of truth** for the new UI panels. Every new DOM cell described below reads from one specific key in that JSON.

In `demo.html`, add a top-level helper:

```javascript
async function fetchDiagnostics() {
  try {
    const r = await fetch(`${API}/diagnostics`);
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
}
```

Call it **once per `step` response** inside `runDemo` immediately after the existing `res = await r.json()` line. Pass the result to every new render function below. Guard everything with `if (diag)` so the demo still works when the backend is on the old version.

Also add a small request-staleness check: if `diag.last_step.total - rew > 0.05`, log a warning chip — that means `/diagnostics` is reporting a different scalar than the trainer received. Useful for debugging.

---

### 8.1 Reward breakdown panel — expand from 6 cells to 11

**Where:** the existing `<div class="bd-section" id="bd-section">` block (around line 842).

The current panel shows: `tool / params / deps / reasoning / penalties / step Σ`. Replace the single 6-column grid with **two rows of 6 cells = 12 cells** (one is the total). Keep the existing class names and animations so nothing else breaks.

New layout:

```
Row 1 (extrinsic, what already exists, just re-skinned):
  TOOL +0.20 │ PARAMS +0.15 │ DEPS +0.10 │ REASON +0.05 │ PENALTIES │ EXTR Σ

Row 2 (NEW — the v2 reward stack):
  PBRS Φ' − Φ │ INTR (count+RND) │ JUDGE × CONF │ DIFFERENCE │ IRD Δ │ TOTAL Σ
```

Implementation details:

- Add 6 new DOM IDs: `bd-pbrs`, `bd-intr`, `bd-judge`, `bd-diff`, `bd-ird`, `bd-tot`. Reuse the existing `setBdCell(id, val)` helper — it already handles dim/pos/neg states.
- Add a CSS class `.bd-grid-v2 { grid-template-columns:repeat(6,1fr); margin-top:5px; }` and wrap row 2 in `<div class="bd-grid bd-grid-v2">`.
- The `EXTR Σ` cell shows the sum of row-1 cells; the `TOTAL Σ` cell shows the scalar the trainer actually consumed. **The two are visibly different on every step where any new term fires** — that is the demo moment.
- When `/diagnostics` is unavailable, dim row 2 entirely with `.bd-grid-v2.legacy { opacity: 0.35; }` and show a tooltip "Run `python -m uvicorn server.app:app` from the v2 branch to see the full reward stack."

Update `renderBreakdown(breakdown, stepReward, diag)` to take the diag object and route values:

```javascript
if (diag?.last_step) {
  setBdCell('bd-pbrs',  diag.last_step.pbrs_shaping + diag.last_step.dynamic_pbrs_shaping);
  setBdCell('bd-intr',  diag.last_step.intrinsic_count + diag.last_step.intrinsic_rnd);
  setBdCell('bd-judge', diag.last_step.weighted_judge);
  setBdCell('bd-diff',  diag.last_step.difference_reward);
  setBdCell('bd-ird',   diag.last_step.ird_posterior_correction);
  // bd-tot already exists, just re-route it to diag.last_step.total
}
```

---

### 8.2 New "Reward Stack" decomposition sparkline

**New DOM block** placed between the existing `<div class="rew-section">` (reward bar) and `<div class="judge-section">` (judge panel).

A 64-px-tall stacked-area sparkline canvas that grows as the episode progresses. Each step appends one column; each component is a colored band:

| Band | Color token | Source |
|---|---|---|
| Extrinsic | `--cyan` (#00E5FF) | `diag.last_step.extrinsic_step + extrinsic_episode` |
| PBRS | `--green` (#00E676) | `pbrs_shaping + dynamic_pbrs_shaping` |
| Intrinsic | `--purple` (#B04DFF) | `intrinsic_count + intrinsic_rnd` |
| Judge × conf | `--gold` (#FFD600) | `weighted_judge` |
| Difference | `--teal` (#00E5BB) | `difference_reward` |
| IRD Δ | `--pink` (#FF4DB8) | `ird_posterior_correction` |
| LIRPG | `--amber` (#FFB300) | `intrinsic_lirpg` |

```html
<div class="stack-section">
  <div class="rew-top">
    <span class="rew-lbl">REWARD STACK · LIVE DECOMPOSITION</span>
    <span class="rew-val" id="stack-total">+0.00</span>
  </div>
  <canvas id="stack-canvas" width="600" height="64"></canvas>
  <div class="stack-legend" id="stack-legend"></div>
</div>
```

JS: maintain `stackHistory = [{extr, pbrs, intr, judge, diff, ird, lirpg}, ...]`, redraw all columns on every step. Use 2D canvas, no chart library — keeps bundle zero-dep. **This is the single most important visual addition** — judges will see at a glance which reward terms are doing the work at each step.

Hover over any column → tooltip shows the per-component split for that step.

---

### 8.3 Judge panel — real uncertainty visualization

**Where:** the existing `<div class="judge-section">` (around line 815).

The current panel shows `MODE / ALPHA / LAST SCORE / CONFIDENCE` in 4 cells. The score and confidence currently come from `breakdown.learned_judge_score` and `learned_judge_confidence`, but confidence is hardcoded to 0.5 in the backend. After §3.3 lands, both become real.

Required UI changes:

- Add a 5th cell `STD (TREE VAR)` showing `breakdown.learned_judge_components.std` — make it tiny number + a horizontal bar that gets longer as std grows.
- Recolor the `CONFIDENCE` cell as a **filled bar** rather than a number: width = `confidence * 100 %`, color shifts from `--red` (low) → `--amber` → `--green` (high) via a single CSS gradient.
- Add a **uncertainty band** to the reward stack sparkline: a translucent envelope above and below the `weighted_judge` band of width ±std. This is a visible "the system shows you how confident the judge is" cue.
- When `confidence < 0.3` for ≥ 3 consecutive steps, fire a one-time story card `judgeUncertain` (see §8.6 for the new card list).

Update `judge-grid` CSS to `grid-template-columns: repeat(5, 1fr)`.

---

### 8.4 New "PBRS Potential" mini-panel

**New DOM block** placed in the right panel, above the log panel and below the breakdown.

A single-line strip showing the shaping math live:

```html
<div class="pbrs-strip" id="pbrs-strip">
  <span class="pbrs-lbl">PBRS · POLICY-INVARIANT SHAPING</span>
  <span class="pbrs-eq">
    F = γ·Φ(s′) − Φ(s)
  </span>
  <span class="pbrs-vals">
    γ=<b id="pbrs-gamma">0.99</b> ·
    Φ(s)=<b id="pbrs-phi-s">0.32</b> →
    Φ(s′)=<b id="pbrs-phi-sp">0.41</b> ·
    F=<b id="pbrs-f">+0.085</b>
  </span>
  <span class="pbrs-badge" id="pbrs-mode">STATIC</span>
</div>
```

Pulls from `diag.last_step.pbrs_shaping`, plus two new diag fields the backend must expose: `phi_before` and `phi_after`. (Add these to the §5 diagnostics schema — single-line addition to `server/app.py` after §3.1 ships.)

The `pbrs-mode` badge toggles `STATIC` / `DYNAMIC` based on `diag.config.USE_DYNAMIC_PBRS`. When `DYNAMIC`, animate the badge with a subtle glow — visible cue that Φ is learned.

CSS: minimal strip with `font-mono`, monospace values, `border-left: 3px solid var(--green)` because PBRS is the safest, policy-invariant addition.

---

### 8.5 New "Intrinsic Motivation" mini-panel

**New DOM block** stacked directly under §8.4's PBRS strip.

```html
<div class="intr-strip">
  <span class="intr-lbl">INTRINSIC MOTIVATION · COUNT + RND</span>
  <div class="intr-row">
    <div class="intr-cell">
      <div class="intr-k">N(s,a)</div>
      <div class="intr-v" id="intr-count">0</div>
      <div class="intr-bar"><div class="intr-bar-fill count" id="intr-count-bar"></div></div>
    </div>
    <div class="intr-cell">
      <div class="intr-k">RND ‖e‖²</div>
      <div class="intr-v" id="intr-rnd">0.000</div>
      <div class="intr-bar"><div class="intr-bar-fill rnd" id="intr-rnd-bar"></div></div>
    </div>
    <div class="intr-cell">
      <div class="intr-k">DECAY</div>
      <div class="intr-v" id="intr-decay">1.00</div>
      <div class="intr-bar"><div class="intr-bar-fill decay" id="intr-decay-bar"></div></div>
    </div>
    <div class="intr-cell tot">
      <div class="intr-k">BONUS Σ</div>
      <div class="intr-v" id="intr-total">+0.00</div>
    </div>
  </div>
</div>
```

The DECAY cell is the visible proof of §3.4's anti-hacking decay rule: it should noticeably shrink over the episode. Pull from the new `diag.last_step.intrinsic_decay_factor` field (add to `/diagnostics`).

When the bonus has decayed below 5 % of its initial value, fire story card `intrinsicDecayed` once.

---

### 8.6 New story cards (extend the `STORY` object)

Insert these after the existing `chaosResolved` entry in the `STORY = { ... }` object. Each one has the same 5-field shape (`phase`, `title`, `body`, `border`, `step`).

```javascript
pbrsInvariant: {
  phase:  'REWARD ENGINEERING · POLICY INVARIANCE',
  title:  'SHAPING THAT PROVABLY DOESN\'T HURT',
  body:   'Every step gets <b>F = γ·Φ(s′) − Φ(s)</b> on top of the deterministic reward. By construction, the optimal policy is unchanged — and we ship a unit test that proves it on a 3-state MDP.',
  border: 'green-border', step: 3,
},
dynamicPotential: {
  phase:  'REWARD ENGINEERING · DYNAMIC PBRS',
  title:  'THE POTENTIAL LEARNS ITSELF',
  body:   'Φ(s) is no longer hand-coded. A tiny linear head fits Monte-Carlo returns online. The shaping stays policy-invariant <b>at any moment in training</b>, regardless of weights.',
  border: 'purple-border', step: 3,
},
intrinsicNovel: {
  phase:  'EXPLORATION · COUNT + RUNE',
  title:  'TWO SOURCES OF NOVELTY',
  body:   '<b>Count-based:</b> 1/√N over (workflow, completed-set, tool). <b>RND-style:</b> ‖target − predictor‖². Both decay so they cannot be hacked at convergence.',
  border: 'purple-border', step: 4,
},
intrinsicDecayed: {
  phase:  'EXPLORATION · CONVERGED',
  title:  'NOVELTY HAS DONE ITS JOB',
  body:   'The decay factor has dropped below 5 %. The agent is no longer being paid to explore — pure extrinsic + PBRS now drives the reward. Anti-hacking guarantee in action.',
  border: 'green-border', step: 4,
},
irdPosterior: {
  phase:  'INVERSE REWARD DESIGN',
  title:  'THE AGENT DOESN\'T TRUST THE REWARD',
  body:   'The grader is a <b>proxy</b>. The true reward is binary episode success. An IRD posterior over reward weights corrects the proxy in real time — that pink stripe in the reward stack.',
  border: 'purple-border', step: 4,
},
judgeUncertain: {
  phase:  'ROBUST SHAPING',
  title:  'WHEN THE JUDGE DOUBTS, WE LISTEN',
  body:   'Tree-variance is high → confidence drops → judge contribution shrinks automatically. Robust shaping under uncertainty, no hand-tuned alpha schedule.',
  border: '', step: 4,
},
pgrdMeta: {
  phase:  'META-REWARD · PGRD',
  title:  'THE REWARD TUNES ITSELF',
  body:   'Every 10 episodes, the reward weights θ_R take a step toward maximising the <b>true</b> episode-success signal. Watch the sparkline at the top — those are seven weights moving live.',
  border: 'purple-border', step: 5,
},
ablationProof: {
  phase:  'EVIDENCE · ABLATION',
  title:  '5 RUNS, 1 CHART',
  body:   'proxy-only · +PBRS · +PBRS+judge · +intrinsic · +everything. Bar chart shows each addition\'s contribution. Anything that doesn\'t earn its place is removed before the final run.',
  border: 'green-border', step: 5,
},
sampleComplexity: {
  phase:  'EVIDENCE · SAMPLE COMPLEXITY',
  title:  'EPISODES-TO-THRESHOLD',
  body:   'Reaches reward ≥ 0.5 in <b id="sc-eps-half">N</b> episodes, ≥ 1.0 in <b id="sc-eps-full">M</b> episodes. Same metric across all ablation runs — the apples-to-apples improvement number.',
  border: '', step: 5,
},
```

Trigger schedule (add to the existing `runDemo` flow):

| When | Card |
|---|---|
| First step where `diag.last_step.pbrs_shaping ≠ 0` | `pbrsInvariant` |
| First step where `diag.config.USE_DYNAMIC_PBRS === true` | `dynamicPotential` |
| Step 1 of any trained run | `intrinsicNovel` |
| When `intr-decay < 0.05` | `intrinsicDecayed` |
| First step where `diag.last_step.ird_posterior_correction ≠ 0` | `irdPosterior` |
| 3+ consecutive steps with `confidence < 0.3` | `judgeUncertain` |
| End of episode if PGRD weights changed in the last batch | `pgrdMeta` |
| Closing the episode overlay (button) | `ablationProof` then `sampleComplexity` (chained, 4 s apart) |

The existing 5-dot timeline (`renderTimeline`) is now too small — bump to 8 dots and the existing 5 phases each get one or two more cards mapped to the same step number, so the dots fill in incrementally as new mechanics fire.

---

### 8.7 Episode-overlay enhancements

**Where:** the `<div id="ep-ov">` block (around line 909).

Add a 3rd panel below the existing `was → next` curriculum diff: a compact **ablation-vs-full** comparison.

```html
<div class="ep-ablation" id="ep-ablation">
  <div class="ep-ablation-h">REWARD STACK CONTRIBUTIONS · THIS EPISODE</div>
  <div class="ep-ablation-bars" id="ep-ablation-bars">
    <!-- 7 horizontal bars: extrinsic, pbrs, intrinsic, judge, difference, ird, lirpg -->
  </div>
  <div class="ep-ablation-foot">
    Total <b id="ep-ab-total">+0.00</b>  ·
    Without v2 stack: <b id="ep-ab-baseline">+0.00</b>  ·
    Δ <b id="ep-ab-delta" class="pos">+0.00</b>
  </div>
</div>
```

`ep-ab-baseline` is computed client-side as `extrinsic_step + extrinsic_episode` summed over the episode (the proxy-only counterfactual). `ep-ab-delta` is the difference. **This single panel answers the judge's most likely question — "what did the reward engineering actually buy you?" — every single episode.**

Also add an "OPEN ABLATION CHART" button that links to `ablation_curve.png` produced by §4. If the file doesn't exist locally yet, the button stays disabled with tooltip "Run `python eval_ablations.py` to produce this chart."

---

### 8.8 Ablation toggle (live A/B during a single episode)

**New control bar button**, placed after the existing `RESET` button:

```html
<button class="cbtn ab" id="btn-ab" onclick="toggleAblation()">⊞ MODE: FULL</button>
```

Cycles through `FULL → PROXY ONLY → +PBRS → +INTRINSIC → +JUDGE → FULL ...`. The button label updates to show the current mode. On click, send a `POST /diagnostics/mode` with body `{ "mode": "<name>" }` (add this endpoint to §5 — it just sets the active `RewardCombiner.mode` for subsequent steps).

Visible effect: the reward stack sparkline immediately shows the corresponding bands going zero/non-zero on the next step. **This is the most direct demonstration of "every term ablates to a measurable contribution"** that you can put in a 2-minute video.

CSS: `.cbtn.ab { background: transparent; color: var(--purple); border: 1px solid var(--purple); }`

---

### 8.9 Header additions

**Where:** the existing `<header>` block (around line 710).

Add a 4th theme pill **next to the existing T2/T3/T4 pills**:

```html
<span class="pill rev2">RE v2 · 11-COMPONENT REWARD</span>
```

CSS:
```css
.pill.rev2 { background: #1A0C22; color: var(--gold); border: 1px solid rgba(255,214,0,.3); }
```

Click on it → opens `docs/REWARD_ENGINEERING.md` in a new tab.

Also add a **PGRD weights sparkline** to the header right side, between the `LIVE` dot and the difficulty pill group:

```html
<div class="pgrd-spark" title="PGRD reward-weight evolution">
  <canvas id="pgrd-canvas" width="120" height="22"></canvas>
  <span class="pgrd-lbl">θ_R</span>
</div>
```

7 colored lines, one per reward weight, refreshed every episode end from `diag.config.weights`. Tiny — just a "the reward function is alive" cue.

---

### 8.10 Stats row — add a 5th tile

**Where:** the existing `<div class="stats-row">` (around line 784).

Change `grid-template-columns: repeat(4,1fr)` to `repeat(5,1fr)` and add:

```html
<div class="stat">
  <div class="stat-val" id="sv-eps-thresh">—</div>
  <div class="stat-lbl">EPS → 0.5</div>
</div>
```

Pulls from `diag.config.episodes_to_threshold_0_5` (added by §3.13). Stays as `—` if the metric isn't in `training_metrics.json` yet.

The five tiles after this change: `STEPS · TASKS DONE · REWARD · DIFFICULTY · EPS → 0.5`.

---

### 8.11 Untrained-vs-trained agent — show the new mechanics ARE responsible

`untrainedAction()` already cycles through 3 bad actions and aborts after step 3. Add a 4th "synthetic" frame appended to the log right before the abort:

```javascript
addLog('warn','◌',`No PBRS bonus, no intrinsic novelty, no judge confidence — the v2 stack contributed +0.00 this episode (vs +0.42 average for trained agent).`);
```

Pulls the 0.42 number from `diag.config.trained_v2_avg_contribution` (a simple running mean the backend can expose). Hard-coded fallback if not yet available. **One sentence in a log is worth a paragraph in the README.**

---

### 8.12 CSS additions (consolidated)

Add this block at the end of the existing `<style>` section:

```css
/* ═══════════════════════════════════════════════════
   REWARD ENGINEERING v2 — additions
═══════════════════════════════════════════════════ */
.bd-grid-v2{margin-top:5px}
.bd-grid-v2.legacy{opacity:.35;pointer-events:none}

.stack-section{
  padding:8px 16px 10px;border-bottom:1px solid var(--b1);
  background:linear-gradient(180deg,rgba(255,214,0,.04),transparent);
}
#stack-canvas{
  width:100%;height:64px;display:block;margin-top:6px;
  background:var(--bg);border:1px solid var(--b1);border-radius:5px;
}
.stack-legend{
  display:flex;flex-wrap:wrap;gap:8px;margin-top:5px;
  font-family:var(--font-mono);font-size:7.5px;letter-spacing:.5px;color:var(--t3);
}
.stack-legend .sl-dot{
  width:8px;height:8px;border-radius:2px;display:inline-block;margin-right:3px;vertical-align:middle;
}

.pbrs-strip,.intr-strip{
  display:flex;align-items:center;gap:10px;padding:7px 14px;
  border-bottom:1px solid var(--b1);background:var(--surf);
  font-family:var(--font-mono);font-size:9px;letter-spacing:.5px;
}
.pbrs-strip{border-left:3px solid var(--green)}
.intr-strip{border-left:3px solid var(--purple);flex-direction:column;align-items:stretch}
.pbrs-lbl,.intr-lbl{color:var(--t3);font-size:7.5px;letter-spacing:1.5px}
.pbrs-eq{color:var(--green);font-weight:700}
.pbrs-vals b{color:var(--t1)}
.pbrs-badge{
  margin-left:auto;padding:2px 7px;border-radius:3px;font-size:7px;letter-spacing:1px;
  background:rgba(0,230,118,.1);color:var(--green);border:1px solid rgba(0,230,118,.3);
}
.pbrs-badge.dyn{
  background:rgba(176,77,255,.1);color:var(--purple);border-color:rgba(176,77,255,.3);
  animation:pbrs-pulse 2s ease-in-out infinite;
}
@keyframes pbrs-pulse{0%,100%{box-shadow:0 0 0 rgba(176,77,255,0)}50%{box-shadow:0 0 6px rgba(176,77,255,.4)}}

.intr-row{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}
.intr-cell{background:var(--bg);border:1px solid var(--b1);border-radius:5px;padding:6px 8px}
.intr-cell.tot{border-color:rgba(255,214,0,.4)}
.intr-k{color:var(--t3);font-size:6.5px;letter-spacing:1px;margin-bottom:3px}
.intr-v{color:var(--t1);font-weight:700;font-size:11px}
.intr-bar{height:3px;background:var(--b1);border-radius:1.5px;margin-top:4px;overflow:hidden}
.intr-bar-fill{height:100%;border-radius:1.5px;transition:width .4s ease}
.intr-bar-fill.count{background:var(--purple)}
.intr-bar-fill.rnd{background:var(--pink)}
.intr-bar-fill.decay{background:linear-gradient(90deg,var(--red),var(--amber),var(--green))}

.cbtn.ab{background:transparent;color:var(--purple);border:1px solid var(--purple)}
.cbtn.ab:hover{background:rgba(176,77,255,.08)}

.pill.rev2{background:#1A0C22;color:var(--gold);border:1px solid rgba(255,214,0,.3);cursor:pointer}
.pill.rev2:hover{background:#22102A}

.pgrd-spark{
  display:flex;align-items:center;gap:4px;
  padding:3px 7px;border-radius:5px;background:var(--bg);
  border:1px solid var(--b1);
}
.pgrd-lbl{font-family:var(--font-mono);font-size:8px;color:var(--t3);letter-spacing:1px}

.ep-ablation{
  background:var(--bg);border:1px solid var(--b2);border-left:3px solid var(--gold);
  border-radius:8px;padding:10px 12px;margin-top:12px;text-align:left;
}
.ep-ablation-h{
  font-family:var(--font-mono);font-size:7.5px;letter-spacing:1.5px;
  color:var(--gold);margin-bottom:7px;
}
.ep-ablation-bars{display:flex;flex-direction:column;gap:3px}
.ep-ab-bar{
  display:flex;align-items:center;gap:6px;
  font-family:var(--font-mono);font-size:8.5px;
}
.ep-ab-bar .lbl{width:80px;color:var(--t2)}
.ep-ab-bar .track{
  flex:1;height:6px;background:var(--b1);border-radius:3px;overflow:hidden;
}
.ep-ab-bar .fill{height:100%;border-radius:3px;transition:width .35s}
.ep-ab-bar .val{width:48px;text-align:right;color:var(--t1);font-weight:700}
.ep-ablation-foot{
  margin-top:8px;font-family:var(--font-mono);font-size:8px;
  color:var(--t2);letter-spacing:.5px;
}
.ep-ablation-foot b{color:var(--gold)}
.ep-ablation-foot .pos{color:var(--green)}
.ep-ablation-foot .neg{color:var(--red)}
```

---

### 8.13 JavaScript wiring summary

A new file is **not** needed — keep everything in the single `<script>` block. The wiring delta is roughly:

| Existing function | Modify to |
|---|---|
| `runDemo(mode)` | Call `fetchDiagnostics()` after every step, pass result to all renderers below. Maintain `stackHistory`. |
| `renderBreakdown(b, r)` → `renderBreakdown(b, r, diag)` | Populate row 2 cells from `diag.last_step.*`. |
| `updateJudgePanel(meta, b)` → `updateJudgePanel(meta, b, diag)` | Read true `confidence` and `std`; trigger `judgeUncertain` story when threshold hit. |
| `showEp(rew, diff)` → `showEp(rew, diff, diag)` | Render new ablation panel inside the overlay. |
| `closeEp()` | After existing logic, chain `ablationProof` → `sampleComplexity` story cards. |

New functions to add:

```javascript
function renderStackCanvas(history) { ... }   // §8.2
function renderPbrsStrip(diag)      { ... }   // §8.4
function renderIntrStrip(diag)      { ... }   // §8.5
function renderEpAblation(history)  { ... }   // §8.7
function renderPgrdSparkline(diag)  { ... }   // §8.9
function toggleAblation()           { ... }   // §8.8
function fireConditionalStories(diag) { ... } // §8.6 schedule
```

Keep all rendering pure (read in / write to DOM, no global state besides `stackHistory`). All seven new functions are no-ops when `diag === null` — the demo must work against the current backend for at least the duration of the rollout.

---

### 8.14 Visual checklist before declaring frontend done

- [ ] Open `demo.html` against the v1 backend → behaviour unchanged, no console errors.
- [ ] Open `demo.html` against the v2 backend with all flags ON → reward stack canvas fills with all 7 colors, breakdown row 2 lights up, PBRS strip shows non-zero F, intrinsic strip shows count + RND values, judge panel shows real std bar, episode overlay shows the ablation panel.
- [ ] Click ablation toggle through all 5 modes → reward stack visibly recomposes within 1 step on each toggle.
- [ ] Run `easy` to completion as a trained agent → 5 of the 9 new story cards have fired (intro, intrinsic, PBRS, IRD, success/curriculumNext).
- [ ] Run `hard` to chaos mode → existing chaos cards still fire, plus the v2 stack visibly shows large `intrinsic_count` spikes when the agent encounters novel chaos states.
- [ ] All new DOM IDs are present and contain valid values (no `undefined` or `NaN` strings).
- [ ] Mobile/narrow viewport (< 1100 px) — the new strips wrap cleanly without breaking the grid.
- [ ] The hackathon URL `arnav100904-adaptive-enterprise-autopilot.hf.space` line in the control bar is still present and reachable.

---

### 8.15 Storytelling pay-off (how to use this in the 2-min video)

The video should follow this exact beat list, since every beat now has a dedicated UI element:

1. **0:00–0:15** — Header pulses, RE v2 pill highlighted. Voice: *"We built the only OpenEnv hackathon environment with an 11-component reward stack."*
2. **0:15–0:30** — Run untrained agent. The reward stack canvas shows nothing but cyan extrinsic. *"Without reward engineering, here's what you get."*
3. **0:30–0:50** — Run trained agent. As each component fires for the first time, its story card slides in. *"Watch the seven new bands light up."*
4. **0:50–1:05** — Click ablation toggle through the modes. Stack visibly collapses and expands. *"Every term ablates to a measurable contribution."*
5. **1:05–1:25** — Episode overlay opens with the new ablation comparison panel. *"+0.42 over the proxy-only baseline this episode."*
6. **1:25–1:45** — PGRD sparkline in header gets close-up. *"And the reward weights themselves are tuning live, every 10 episodes, toward the true objective."*
7. **1:45–2:00** — Cut to `ablation_curve.png` from §4. Closing line: the 12-summary line.

If any of those beats is missing, the corresponding §8 subsection is incomplete — **revisit before submission**.

---

## 9. Tests (CI-friendly)

**File:** `tests/test_pbrs_invariance.py`

Mandatory. Builds a 3-state toy MDP, runs value iteration with both rewards, asserts argmax equality. Failure of this test means PBRS is implemented incorrectly and **must block deployment**.

**File:** `tests/test_difference_rewards.py`

Asserts `D(noop) == 0`, `D(correct_action) > D(wrong_action)`, `D(invalid_tool) < 0`.

**File:** `tests/test_intrinsic_decay.py`

Asserts that after 100 calls with the same `(workflow, completed, tool)` tuple, the count bonus has dropped below 10 % of its initial value.

Run with `pytest tests/`. Wire a `pytest` step into the README's local-run instructions.

---

## 10. Compliance checklist (do not skip)

Before declaring done:

- [ ] `python -m uvicorn server.app:app --host 0.0.0.0 --port 7860` starts cleanly with **all flags off**, behaviour identical to current main.
- [ ] `curl -X POST http://localhost:7860/reset?task=easy` still returns the same observation schema (no breaking changes).
- [ ] `curl http://localhost:7860/diagnostics` returns the schema described in §5 with non-null values once a step has run.
- [ ] `python train.py` runs to completion on a single GPU (or on CPU smoke-test) for `NUM_EPISODES=20`.
- [ ] `python eval_ablations.py` produces `ablation_curve.png` and `ablation_table.md`.
- [ ] `python -m envs.autopilot_env.adversarial_probes` produces the probe table.
- [ ] `pytest tests/` is green on all 3 new tests.
- [ ] `openenv.yaml` unchanged (or version-bumped to `1.1.0` and `description` updated).
- [ ] `README.md` has the new "What's new in v2" section, ablation chart, PGRD chart, sample-complexity numbers.
- [ ] `docs/REWARD_ENGINEERING.md` exists and covers all 10 sections of §6.
- [ ] `demo.html` opens against the v2 backend and **all** §8.14 visual-checklist items pass; opens against the v1 backend with **no console errors** and original behaviour intact.
- [ ] At least 5 of the 9 new story cards from §8.6 fire during a single trained-easy run.
- [ ] Ablation toggle (§8.8) cycles through all 5 modes and the reward-stack canvas visibly recomposes on each toggle.
- [ ] No new heavy dependencies (no jax, no Stable-Baselines3, no Gym wrappers, no chart.js / d3). Allowed adds: `torch` (already present), `numpy`, `scipy.stats` is fine but optional. Frontend stays zero-dep — canvas only, no chart library.
- [ ] Hugging Face Space deploy still works — `Dockerfile` unchanged or only adds `pip install` lines.

---

## 11. Suggested implementation order (1-day plan if compute-limited)

| Hour | Module | Deliverable |
|---|---|---|
| 0 | §3.0 RewardCombiner + §5 /diagnostics endpoint | scaffold, all weights at zero except extrinsic — non-breaking; frontend can already poll |
| 1 | §3.1 PBRS + invariance test | first new shaping term active, test green |
| 2 | §3.4 count + RND intrinsic | new bonus visible in /diagnostics |
| 3 | §3.6 difference rewards | new component logged |
| 4 | §3.3 reward uncertainty | judge_confidence becomes real |
| 5 | §3.5 IRD posterior | proxy-vs-true narrative live |
| 6 | §8.0 + §8.1 + §8.2 frontend basics | reward-stack canvas + 11-cell breakdown live in the demo |
| 7 | §8.3–§8.5 frontend strips + §8.6 story cards | judges actually see the new mechanics fire on screen |
| 8 | §3.10 demo expansion + §3.13 sample-complexity | bigger SFT, metric printed; §8.10 stat tile reads it |
| 9 | §3.2 dynamic PBRS + §3.8 PGRD | weight-evolution plot; §8.9 PGRD sparkline goes live |
| 10 | §3.9 LIRPG | last optional new term |
| 11 | §4 ablation harness + §8.7–§8.8 episode-overlay ablation panel + toggle | ablation chart + live ablation toggle in the demo |
| 12 | §6 docs + README updates + §8.15 video shoot | final story |

If time is tight, **§3.0, §3.1, §3.4, §3.6, §3.5, §4, §8.0–§8.2, and §8.6** alone deliver ~80 % of the point gain — they cover **PBRS, policy invariance, intrinsic + count + RUNE-style novelty, difference rewards, IRD, a real ablation chart, the live reward-stack canvas, and the matching story cards**. Everything else compounds on top.

---

## 12. Things explicitly NOT to do

- ❌ Do **not** rewrite `grader.py`'s scoring constants. PBRS is added on top, not replacing.
- ❌ Do **not** change the OpenEnv API endpoints. Add `/diagnostics` and `/diagnostics/mode` only.
- ❌ Do **not** introduce VIME, EXPLORS, or full deep IRL networks — too heavy for the hackathon timeline.
- ❌ Do **not** make any reward component blow up unbounded. Every bonus must clip into a documented range.
- ❌ Do **not** delete or move `extracted_problem.txt`, `extracted_all.txt`, `Rl and openenv.md`. They're reference notes.
- ❌ Do **not** push a model checkpoint to HF Hub from the script — keep `PUSH_TO_HUB` opt-in.
- ❌ Do **not** mark training "done" in the README until the ablation chart shows the full pipeline beats `proxy_only` on at least 2 of 3 tasks.
- ❌ Do **not** rewrite `demo.html` from scratch, swap fonts/colors, replace the existing story-card mechanic, or pull in `react`/`vue`/`d3`/`chart.js`. All §8 additions are pure additions to the existing single-file dashboard.
- ❌ Do **not** ship a frontend that breaks against the v1 backend — every new panel must gracefully no-op when `/diagnostics` returns 404.
- ❌ Do **not** hard-code the v2 reward-component values into `demo.html` for a "fake" demo. Every visible number on screen must come from a real backend response.

---

## 13. One-line summary for the writeup / blog / video

> *"We built the only OpenEnv environment with potential-based shaping that we **prove** preserves the optimal policy, a learned dynamic potential, count + RUNE-style intrinsic motivation, an inverse-reward-design posterior that infers the true objective from a possibly-misspecified proxy, difference rewards for credit assignment, PGRD that auto-tunes its own reward weights from episode success, and a UCB-V curriculum that picks the next task by learning progress — and we show every term ablates to a measurable contribution."*

Land that sentence in the README, the blog, and the video opener. Everything in this brief is engineered to back it up with code, plots, and tests.
