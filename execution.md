# Execution Plan For Remaining Features

This document covers the exact ordered plan to implement the remaining high-value features that are still missing from the current `v2` codebase.

It is intentionally limited to the gaps identified in the audit:

- backend reward terms still missing
- diagnostics payload expansion still missing
- frontend visibility for those new terms still missing
- story cards and episode overlay enhancements still missing
- tests and validation still missing

The current codebase already has:

- PBRS
- count-based intrinsic bonus
- reward combiner with 4 modes
- ablation harness
- sample-complexity metrics
- diagnostics endpoints
- reward stack canvas
- reduced breakdown panel
- story cards for `pbrsInvariant`, `intrinsicNovel`, `ablationProof`

The plan below focuses only on what remains.

---

## Phase 0 - Freeze The Baseline

Goal: lock down the current `v2` behavior before adding more reward terms.

Files:

- `src/envs/autopilot_env/environment.py`
- `src/envs/autopilot_env/reward_combiner.py`
- `server/app.py`
- `demo.html`
- `train.py`
- `eval_ablations.py`

Tasks:

1. Record the current diagnostics schema from `GET /diagnostics`.
2. Record the current reward breakdown keys produced inside `environment.step()`.
3. Record the current frontend assumptions in `demo.html`:
   - breakdown row 2 expects `pbrs_shaping`, `intrinsic_count`, `intrinsic_decay_factor`
   - canvas expects `extr`, `pbrs`, `intr`
4. Preserve the current 4 ablation modes as backward-compatible behavior.
5. Preserve current `Update.txt` claims as the minimum acceptable output.

Exit criteria:

- Existing demo still works unchanged.
- Existing ablation script still runs unchanged.
- Existing PBRS test still passes.

---

## Phase 1 - Expand Reward Data Model First

Goal: create a single source of truth for reward decomposition before adding new reward terms.

Reason for order:

- Every later feature depends on a stable reward component schema.
- Frontend, diagnostics, training logs, and tests all become easier once the structure is explicit.

Primary files:

- `src/envs/autopilot_env/reward_combiner.py`
- `src/envs/autopilot_env/environment.py`
- `server/app.py`

Tasks:

1. Refactor `RewardCombiner` to operate on an explicit component structure rather than only three scalar inputs.
2. Keep support for current fields:
   - `extrinsic`
   - `pbrs_shaping`
   - `intrinsic_count`
3. Add new placeholder fields with zero defaults:
   - `intrinsic_rnd`
   - `weighted_judge`
   - `difference_reward`
   - `ird_posterior_correction`
4. Make `combine()` return:
   - all weighted components
   - `total`
   - `mode`
5. Preserve current modes:
   - `full`
   - `proxy_only`
   - `no_pbrs`
   - `no_intrinsic`
6. Add a clear extension point for future modes if needed.
7. Update `environment.step()` so `_last_reward_breakdown` always includes every component key, even if zero.
8. Update `/diagnostics` to return the full normalized shape.

Recommended output schema for `diag.last_step`:

```json
{
  "extrinsic_step": 0.0,
  "extrinsic_total": 0.0,
  "pbrs_shaping": 0.0,
  "intrinsic_count": 0.0,
  "intrinsic_rnd": 0.0,
  "weighted_judge": 0.0,
  "difference_reward": 0.0,
  "ird_posterior_correction": 0.0,
  "phi_before": 0.0,
  "phi_after": 0.0,
  "intrinsic_decay_factor": 1.0,
  "total": 0.0
}
```

Exit criteria:

- Existing frontend continues to work with old fields.
- New diagnostics payload contains the new zeroed fields.
- No behavior regression in current reward totals.

---

## Phase 2 - Implement Difference Rewards

Goal: add the simplest missing backend reward term first.

Reason for order:

- Lower complexity than IRD and RND.
- Gives immediate visible reward decomposition gains.
- Useful for both backend and frontend validation.

Primary files:

- `src/envs/autopilot_env/difference_rewards.py` (new)
- `src/envs/autopilot_env/environment.py`
- `src/envs/autopilot_env/reward_combiner.py`
- `tests/test_difference_rewards.py` (new)

Tasks:

1. Create `difference_rewards.py`.
2. Implement a function that compares:
   - actual action reward
   - default or counterfactual baseline action reward
3. Keep the first implementation deterministic and cheap.
4. Use the current grader as the baseline evaluator instead of simulating full environment rollouts.
5. Suggested first baseline:
   - `done` when tasks remain, or
   - a null/default no-progress action
6. Compute:
   - `difference_reward = actual_extrinsic_step - baseline_extrinsic_step`
7. Wire this into `environment.step()`.
8. Pass it into `RewardCombiner`.
9. Expose it in `/diagnostics`.
10. Add unit tests for:
   - positive credit when actual action beats baseline
   - zero or negative credit for poor actions
   - determinism under repeated evaluation

Exit criteria:

- `difference_reward` appears in diagnostics and is non-zero on meaningful steps.
- Current total reward remains stable when its weight is set to `0.0`.
- New tests pass.

---

## Phase 3 - Implement IRD Posterior Correction

Goal: add the missing proxy-vs-true-reward correction term.

Reason for order:

- This is one of the biggest missing differentiators from the screenshot plan.
- It depends on the expanded reward schema from Phase 1.

Primary files:

- `src/envs/autopilot_env/ird.py` (new)
- `src/envs/autopilot_env/environment.py`
- `src/envs/autopilot_env/reward_combiner.py`
- `docs/REWARD_ENGINEERING.md`

Tasks:

1. Create `ird.py`.
2. Start with a lightweight posterior correction instead of a heavy Bayesian system.
3. Use existing deterministic signals plus episode success framing to estimate proxy misspecification.
4. First practical version:
   - define a small candidate set of reward weight hypotheses
   - score each hypothesis using current step context
   - normalize to a posterior
   - compute posterior expected reward
   - return `posterior_expected_reward - proxy_reward`
5. Keep the result bounded and numerically stable.
6. Wire `ird_posterior_correction` into `environment.step()`.
7. Route the term through `RewardCombiner`.
8. Expose it in `/diagnostics`.
9. Update docs to explain the simplified IRD framing honestly.

Important constraint:

- Do not overclaim a research-grade IRD implementation if the shipped code is heuristic or lightweight.

Exit criteria:

- `ird_posterior_correction` appears in diagnostics.
- The term is visibly non-zero in at least some steps.
- Documentation matches the actual implementation.

---

## Phase 4 - Add RND As The Second Intrinsic Component

Goal: upgrade intrinsic motivation from count-only to count-plus-RND.

Reason for order:

- This completes the missing "count + RND" screenshot requirement.
- It is more complex than difference rewards, so it should come after the reward schema is stable.

Primary files:

- `src/envs/autopilot_env/intrinsic.py`
- optionally `src/envs/autopilot_env/rnd.py` (new) if separation is cleaner
- `src/envs/autopilot_env/environment.py`
- `tests/test_intrinsic_decay.py` (new or expanded)

Tasks:

1. Decide implementation shape:
   - extend `intrinsic.py`, or
   - split RND into a dedicated module
2. Add a target network and predictor network, or a lightweight equivalent if dependencies must stay minimal.
3. Build state features from:
   - workflow id
   - completed task ids
   - available task ids
   - selected tool
4. Compute:
   - prediction error
   - clipped/scaled `intrinsic_rnd`
5. Apply the same decay factor policy used for count bonus unless you intentionally separate them.
6. Store RND contribution independently from count bonus.
7. Update diagnostics payload to include:
   - `intrinsic_rnd`
   - `intrinsic_count`
   - `intrinsic_decay_factor`
8. Add tests for:
   - decay toward zero
   - predictor error dropping on repeated similar states
   - no crash on empty or small states

Exit criteria:

- `intrinsic_rnd` is non-zero early in runs.
- `intrinsic_rnd` trends down with repeated exposure.
- Count and RND are separately visible in diagnostics.

---

## Phase 5 - Expand Training And Logging

Goal: make the new reward terms measurable outside the live demo.

Reason for order:

- Backend terms should be fully implemented before logging and charting them.

Primary files:

- `train.py`
- `training_metrics.json` generation path
- optionally plotting helpers inside `train.py`

Tasks:

1. Extend component logging in training to include:
   - `pbrs_shaping`
   - `intrinsic_count`
   - `intrinsic_rnd`
   - `weighted_judge`
   - `difference_reward`
   - `ird_posterior_correction`
2. Preserve `episodes_to_threshold_0_5` and `episodes_to_threshold_1_0`.
3. Save aggregate summaries for each reward component.
4. Ensure training metrics remain JSON-serializable and compact.
5. If charts are already produced in `train.py`, add one decomposition plot only if it is cheap and readable.

Exit criteria:

- `training_metrics.json` includes the new component series or summaries.
- Existing threshold metrics are unchanged.

---

## Phase 6 - Upgrade The Diagnostics And Frontend Breakdown

Goal: make the new reward stack visible in the demo.

Reason for order:

- Only start frontend work once backend diagnostics are stable.

Primary files:

- `server/app.py`
- `demo.html`

Tasks:

1. Expand row 2 of the breakdown panel.
2. Replace the reduced v2 row with the planned extended row:
   - `PBRS`
   - `INTR`
   - `JUDGE`
   - `DIFF`
   - `IRD`
   - `TOTAL`
3. Keep legacy fallback dimming when `/diagnostics` is unavailable.
4. Update `renderBreakdown()` so it reads from the expanded diagnostics payload.
5. Keep row 1 extrinsic breakdown unchanged.
6. Make `TOTAL` reflect the scalar actually consumed by the trainer.

Recommended frontend mapping:

- `bd-pbrs` <- `pbrs_shaping`
- `bd-intr` <- `intrinsic_count + intrinsic_rnd`
- `bd-judge` <- `weighted_judge`
- `bd-diff` <- `difference_reward`
- `bd-ird` <- `ird_posterior_correction`
- `bd-tot` <- `total`

Exit criteria:

- Breakdown panel visually distinguishes extrinsic total from actual total.
- No UI error if some advanced fields are still zero.

---

## Phase 7 - Upgrade The Reward Stack Canvas

Goal: make the sparkline reflect the full reward architecture instead of only 3 bands.

Primary files:

- `demo.html`

Tasks:

1. Expand `stackHistory` from:
   - `extr`
   - `pbrs`
   - `intr`
2. To:
   - `extr`
   - `pbrs`
   - `intr`
   - `judge`
   - `diff`
   - `ird`
3. Keep the existing canvas implementation and extend it rather than replacing it.
4. Add a visible legend for the new colors.
5. Ensure negative values still render cleanly below the midline.
6. Keep total label at the top of the section.

Suggested band grouping:

- `intr` = `intrinsic_count + intrinsic_rnd`
- `judge` = `weighted_judge`
- `diff` = `difference_reward`
- `ird` = `ird_posterior_correction`

Exit criteria:

- The sparkline visibly changes when each term fires.
- Full stack remains readable on small episodes.

---

## Phase 8 - Add Missing Story Cards

Goal: complete the narrative layer required by the screenshot spec.

Primary files:

- `demo.html`

Tasks:

1. Keep existing cards:
   - `pbrsInvariant`
   - `intrinsicNovel`
   - `ablationProof`
2. Add missing cards:
   - `irdPosterior`
   - `sampleComplexity`
3. Optional high-value extra card:
   - `intrinsicDecayed`
4. Add trigger rules:
   - fire `irdPosterior` on first non-zero `ird_posterior_correction`
   - fire `sampleComplexity` after episode overlay close or when metrics load
   - fire `intrinsicDecayed` when decay factor drops below `0.05`
5. Do not add too many cards at once if it overwhelms the demo flow.

Exit criteria:

- At least the screenshot-required 5-card set exists.
- Cards trigger from real signals, not only hardcoded timers.

---

## Phase 9 - Add Episode Overlay Ablation Panel

Goal: answer the judge question "what did reward engineering buy you?" per episode.

Primary files:

- `demo.html`

Tasks:

1. Extend the existing episode overlay with a third section.
2. Show:
   - total episode reward
   - proxy-only baseline estimate
   - delta contributed by the extra reward stack
3. Compute baseline client-side from accumulated extrinsic-only values if backend does not already provide an episode summary.
4. Add a button or link to open `ablation_curve.png`.
5. Keep the current curriculum panel intact.

Exit criteria:

- Overlay shows `Total`, `Without v2 stack`, and `Delta`.
- Overlay works even if some advanced terms are zero.

---

## Phase 10 - Add Frontend Reward Mode Toggle

Goal: expose live reward-mode switching in the demo UI.

Primary files:

- `demo.html`
- `server/app.py` only if additional modes are introduced

Tasks:

1. Add a control button near the existing run/reset controls.
2. Wire the button to `POST /diagnostics/mode`.
3. Keep current modes at minimum:
   - `full`
   - `proxy_only`
   - `no_pbrs`
   - `no_intrinsic`
4. If new components get their own modes, add them only if the naming stays clear.
5. Reflect the active mode in the button label.

Exit criteria:

- Switching mode immediately changes visible reward bands on subsequent steps.
- Existing behavior is preserved when the button is never used.

---

## Phase 11 - Surface Sample Complexity In The Frontend

Goal: connect training metrics to demo storytelling.

Primary files:

- `demo.html`
- `training_metrics.json`
- optionally `server/app.py` if metrics must be served

Tasks:

1. Add a small UI tile or stat showing:
   - `episodes_to_threshold_0_5`
2. Add the `sampleComplexity` story card body using real values if available.
3. Keep a graceful fallback if metrics file is absent.

Exit criteria:

- Sample-complexity number is visible in the demo or overlay.
- UI does not break if metrics are missing.

---

## Phase 12 - Documentation And Claim Alignment

Goal: make sure docs match the shipped code exactly.

Primary files:

- `README.md`
- `docs/REWARD_ENGINEERING.md`
- `Update.txt` only if you intentionally revise the event summary

Tasks:

1. Update the "what's new" section to reflect implemented advanced terms.
2. Add clear language distinguishing:
   - deterministic proxy reward
   - shaping terms
   - intrinsic terms
   - correction terms
3. Document which components are mathematically guaranteed vs heuristic.
4. Add screenshot-ready explanation for:
   - difference rewards
   - IRD correction
   - count + RND intrinsic
   - sample complexity

Exit criteria:

- No doc claims a feature that does not exist in code.
- Judges can map each visible UI element to one concrete backend module.

---

## Phase 13 - Validation And Final Checks

Goal: verify the whole stack before treating it as complete.

Required tests:

1. Existing:
   - `tests/test_pbrs_invariance.py`
2. New:
   - `tests/test_difference_rewards.py`
   - `tests/test_intrinsic_decay.py`
   - IRD smoke test or boundedness test

Manual validation:

1. Run backend locally.
2. Run one trained demo episode.
3. Confirm `/diagnostics` shows all fields.
4. Confirm breakdown row 2 updates live.
5. Confirm sparkline shows new colored bands.
6. Confirm `irdPosterior` story card fires when expected.
7. Confirm episode overlay shows ablation delta.
8. Confirm reward mode toggle changes the next step visibly.
9. Confirm sample-complexity metric renders if `training_metrics.json` exists.

Artifact validation:

1. Re-run `eval_ablations.py`.
2. Confirm `ablation_results.json` updates.
3. Confirm `ablation_curve.png` still generates.
4. Confirm `ablation_table.md` still matches current modes.

Exit criteria:

- Tests pass.
- Demo works.
- Docs match code.
- Feature list from the screenshot is either implemented or explicitly marked out of scope.

---

## Recommended Build Order Summary

Follow this exact order:

1. Phase 0 - baseline freeze
2. Phase 1 - reward data model expansion
3. Phase 2 - difference rewards
4. Phase 3 - IRD posterior correction
5. Phase 4 - RND intrinsic
6. Phase 5 - training and metrics logging
7. Phase 6 - frontend breakdown expansion
8. Phase 7 - reward stack canvas expansion
9. Phase 8 - story cards
10. Phase 9 - episode overlay ablation panel
11. Phase 10 - live reward mode toggle
12. Phase 11 - sample-complexity UI
13. Phase 12 - docs alignment
14. Phase 13 - final validation

---

## Highest Value Milestone Cuts

If time is limited, implement in this shortened order:

1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 6
5. Phase 7
6. Phase 8
7. Phase 9
8. Phase 13

That delivers the biggest visible jump against the audit:

- difference rewards
- IRD correction
- full reward decomposition
- richer sparkline
- missing story cards
- per-episode ablation answer

---

## Definition Of Done

This plan is complete only when all of the following are true:

- backend computes all intended reward terms
- `/diagnostics` exposes all intended reward terms
- frontend renders all intended reward terms
- story cards cover the missing narrative beats
- episode overlay shows reward-engineering delta
- sample-complexity is surfaced
- tests cover the new reward modules
- docs do not overstate what exists
