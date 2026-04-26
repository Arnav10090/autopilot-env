"""
Adaptive Enterprise Autopilot — Core Environment.

OpenEnv-compliant. Call reset() → step() × N.

Key design:
  • Each episode = one workflow. Agent makes tool calls until "done" or max_steps.
  • Reward is shaped per step + episode bonus.
  • After every completed episode, workflow_gen creates a harder variant
    stored in state.generated_next_workflow (T4 self-improvement loop).
  • The generated workflow replaces the base one when the same task is reset again.
"""

from __future__ import annotations
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .grader import grade_step, grade_episode, resolve_task
from .difference_rewards import compute_difference_reward
from .ird import IRDPosterior
from .pbrs import potential as pbrs_potential, shaping_term, GAMMA as PBRS_GAMMA, PBRS_WEIGHT
from .intrinsic import IntrinsicCounter, INTRINSIC_WEIGHT
from .reward_combiner import RewardCombiner, RewardComponents
from .judge_features import build_judge_input
from .judge_types import JudgeExample
from .models import AutopilotAction, AutopilotObservation, AutopilotState
from .tools import MockToolRegistry
from .workflow_gen import generate_harder_workflow, generate_easier_workflow, difficulty_score
from .workflows import TASK_WORKFLOWS


class AutopilotEnvironment:
    """
    Core environment. The FastAPI server in server/app.py wraps this class.
    """

    VALID_TASKS = ("easy", "medium", "hard")

    def __init__(
        self,
        task: str = "easy",
        learned_judge=None,
        judge_alpha: float = 0.05,
        judge_enabled: bool = False,
        judge_buffer=None,
    ):
        if task not in self.VALID_TASKS:
            raise ValueError(f"task must be one of {self.VALID_TASKS}, got {task!r}")

        self.task = task
        self._base_workflows: List[Dict[str, Any]] = list(TASK_WORKFLOWS[task])
        self._generated_workflows: Dict[str, Dict] = {}  # workflow_id → generated harder variant

        self._tools = MockToolRegistry()
        self._state = AutopilotState()
        self._workflow: Optional[Dict] = None
        self._completed_ids: List[str] = []
        self._attempted_blocker_ids: set = set()
        self._tool_history: List[Dict] = []
        self._episode_started: bool = False
        self._workflow_index: int = 0
        self._learned_judge = learned_judge
        self._judge_alpha = float(judge_alpha)
        self._judge_enabled = bool(judge_enabled)
        self._judge_buffer = judge_buffer
        self._intrinsic = IntrinsicCounter()
        self._ird = IRDPosterior()
        self._reward_combiner = RewardCombiner()
        self._last_reward_breakdown: Dict[str, Any] = {}
        self._current_episode_judge_examples: List[JudgeExample] = []

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(self) -> AutopilotObservation:
        """Start a new episode. Advances to the next workflow (or a generated harder one)."""
        # Pick workflow: prefer a generated variant if available
        base = self._base_workflows[self._workflow_index % len(self._base_workflows)]
        generated = self._generated_workflows.get(base["workflow_id"])
        self._workflow = generated if generated else base
        self._workflow_index += 1

        self._tools.reset()
        self._completed_ids = []
        self._attempted_blocker_ids = set()
        self._tool_history = []
        self._current_episode_judge_examples = []
        self._episode_started = True
        self._intrinsic.reset_episode()
        self._last_reward_breakdown = {}

        self._state = AutopilotState(
            episode_id=str(uuid.uuid4()),
            task_name=self.task,
            workflow_id=self._workflow["workflow_id"],
            workflow_name=self._workflow["name"],
            step_count=0,
            total_reward=0.0,
            tasks_completed=0,
            tasks_total=len(self._workflow["tasks"]),
            dependency_violations=0,
            rule_violations=0,
            difficulty_level=self._workflow.get("difficulty_level", 1),
        )
        return self._make_observation(
            feedback="Episode started. Analyse the workflow and begin tool calls.",
            reward=0.0,
            done=False,
        )

    def step(
        self, action: AutopilotAction
    ) -> Tuple[AutopilotObservation, float, bool, Dict[str, Any]]:
        """
        Process one tool call from the agent.

        Returns
        -------
        observation : AutopilotObservation
        reward      : float
        done        : bool
        info        : dict  (step breakdown + tool result)
        """
        if not self._episode_started or self._workflow is None:
            raise RuntimeError("Call reset() before step().")

        self._state.step_count += 1
        # PBRS: snapshot Φ(s) BEFORE the action mutates self._completed_ids
        _phi_before = pbrs_potential(self._workflow, self._completed_ids)
        max_steps = self._workflow.get("max_steps", len(self._workflow["tasks"]) * 3)
        tool_result: Dict[str, Any] = {}
        episode_bonus = 0.0

        # ── Grade this action ─────────────────────────────────────────
        step_reward, breakdown = grade_step(
            action,
            self._workflow,
            self._completed_ids,
            self._tools.summary(),
        )
        difference_raw, difference_meta = compute_difference_reward(
            action,
            self._workflow,
            self._completed_ids,
            self._tools.summary(),
            actual_step_reward=step_reward,
        )
        breakdown["_phi_before"] = _phi_before
        breakdown["difference_reward_raw"] = difference_raw
        breakdown["difference_baseline_tool"] = difference_meta["baseline_tool"]
        breakdown["difference_baseline_step_reward"] = difference_meta["baseline_step_reward"]
        judge_input = None
        judge_prediction = None
        judge_score = 0.0

        available_ids = self._available_task_ids()
        pending_ids = self._pending_task_ids()
        completed_before = len(self._completed_ids)
        available_before = len(available_ids)

        if self._judge_enabled and self._learned_judge is not None:
            judge_input = build_judge_input(
                workflow=self._workflow,
                completed_ids=self._completed_ids,
                available_ids=available_ids,
                pending_ids=pending_ids,
                tool_summary=self._tools.summary(),
                tool_history=self._tool_history,
                action=action,
            )
            judge_prediction = self._learned_judge.score(judge_input)
            judge_score = float(judge_prediction.score)

        # Track violations for episode penalty
        if breakdown.get("dep_violation", 0) < 0:
            self._state.dependency_violations += 1
        if breakdown.get("rule_violation", 0) < 0:
            self._state.rule_violations += 1

        breakdown["learned_judge_score"] = round(judge_score, 4)
        if judge_prediction is not None:
            breakdown["learned_judge_components"] = dict(judge_prediction.components)
            breakdown["learned_judge_confidence"] = round(judge_prediction.confidence, 4)

        # ── Execute tool call (if valid) ──────────────────────────────
        if action.tool and action.tool != "done":
            raw = self._tools.call(action.tool, action.params)
            tool_result = raw

            # Mark task complete if this call resolves it
            resolved = resolve_task(
                action, self._workflow, self._completed_ids, self._attempted_blocker_ids
            )
            if resolved:
                self._completed_ids.append(resolved)
                self._state.tasks_completed += 1

            # Record for observation
            self._tool_history.append({
                "tool": action.tool,
                "params": action.params,
                "result": raw,
                "resolved_task": resolved,
            })

        # ── Check episode termination ─────────────────────────────────
        all_done = len(self._completed_ids) == len(self._workflow["tasks"])
        timed_out = self._state.step_count >= max_steps
        agent_done = action.tool == "done"
        done = all_done or timed_out or agent_done

        if done:
            episode_bonus, ep_breakdown = grade_episode(
                self._workflow,
                self._completed_ids,
                self._state.step_count,
                self._state.dependency_violations,
                self._state.rule_violations,
            )
            breakdown["episode_bonus"] = ep_breakdown
            completion_rate = (
                len(self._completed_ids) / len(self._workflow["tasks"])
                if self._workflow["tasks"] else 0.0
            )
            episode_success = 1.0 if all_done else 0.0
            for example in self._current_episode_judge_examples:
                example.episode_success = episode_success
                example.completion_rate = completion_rate
                if self._judge_buffer is not None:
                    self._judge_buffer.add(example)
            self._current_episode_judge_examples = []
            # T4: generate harder workflow for next time
            self._run_self_improvement()

        # ── PBRS shaping + count-based intrinsic ──────────────────────────────
        phi_before = float(breakdown.get("_phi_before", 0.0))
        phi_after = pbrs_potential(self._workflow, self._completed_ids)
        pbrs_term = shaping_term(phi_before, phi_after, gamma=PBRS_GAMMA)
        completed_after = len(self._completed_ids)
        available_after = len(self._available_task_ids())
        proxy_reward = step_reward + episode_bonus
        ird_term, ird_meta = self._ird.correction(
            proxy_reward=proxy_reward,
            step_breakdown=breakdown,
            action_tool=action.tool or "",
            total_tasks=len(self._workflow["tasks"]),
            completed_before=completed_before,
            completed_after=completed_after,
            available_before=available_before,
            available_after=available_after,
            episode_done=done,
            episode_success=all_done,
        )

        intrinsic_term = self._intrinsic.bonus(
            workflow_id=self._workflow.get("workflow_id", ""),
            completed_ids=list(self._completed_ids),
            tool=action.tool or "",
        )

        extrinsic_total = step_reward + episode_bonus + (self._judge_alpha * judge_score)
        components = RewardComponents(
            extrinsic=extrinsic_total,
            pbrs_shaping=pbrs_term,
            intrinsic_count=intrinsic_term,
            difference_reward=difference_raw,
            ird_posterior_correction=ird_term,
        )
        combined = self._reward_combiner.combine(
            components=components,
        )

        breakdown["pbrs_shaping"] = round(combined["pbrs_shaping"], 4)
        breakdown["intrinsic_count"] = round(combined["intrinsic_count"], 4)
        breakdown["intrinsic_rnd"] = round(combined["intrinsic_rnd"], 4)
        breakdown["weighted_judge"] = round(combined["weighted_judge"], 4)
        breakdown["difference_reward"] = round(combined["difference_reward"], 4)
        breakdown["ird_posterior_correction"] = round(combined["ird_posterior_correction"], 4)
        breakdown["ird_proxy_reward"] = ird_meta["proxy_reward"]
        breakdown["ird_posterior_expected_reward"] = ird_meta["posterior_expected_reward"]
        breakdown["ird_top_hypothesis"] = ird_meta["top_hypothesis"]
        breakdown["ird_posterior"] = dict(ird_meta["posterior"])
        breakdown["extrinsic_step"] = round(step_reward, 4)
        breakdown["extrinsic_total"] = round(extrinsic_total, 4)
        breakdown["phi_before"] = round(phi_before, 4)
        breakdown["phi_after"] = round(phi_after, 4)
        breakdown["intrinsic_decay_factor"] = round(self._intrinsic.decay_factor(), 4)
        breakdown["intrinsic_episode_idx"] = self._intrinsic.episode_idx
        breakdown["reward_combiner_mode"] = self._reward_combiner.mode

        total_step_reward = round(combined["total"], 4)
        breakdown["total"] = total_step_reward
        self._state.total_reward += total_step_reward
        self._last_reward_breakdown = dict(breakdown)

        if judge_input is not None:
            self._current_episode_judge_examples.append(JudgeExample(
                judge_input=judge_input,
                deterministic_step_reward=float(step_reward),
                deterministic_breakdown=dict(breakdown),
                episode_success=0.0,
                completion_rate=0.0,
            ))

        feedback = self._build_feedback(action, breakdown, tool_result, done)
        obs = self._make_observation(
            feedback=feedback,
            reward=total_step_reward,
            done=done,
        )
        info = {
            "breakdown": breakdown,
            "tool_result": tool_result,
            "workflow_id": self._workflow["workflow_id"],
            "completed_task_ids": list(self._completed_ids),
        }
        return obs, total_step_reward, done, info

    @property
    def state(self) -> AutopilotState:
        return self._state

    # ── T4: Self-improvement ──────────────────────────────────────────────────

    def _run_self_improvement(self):
        if self._workflow is None:
            return
        completion_rate = (
            len(self._completed_ids) / len(self._workflow["tasks"])
            if self._workflow["tasks"] else 0.0
        )

        # Track consecutive poor episodes per workflow
        base_id = self._workflow.get("workflow_id", "")
        for base in self._base_workflows:
            if base["workflow_id"] == base_id or self._workflow.get("generated"):
                base_id = base["workflow_id"]
                break

        if not hasattr(self, "_poor_streak"):
            self._poor_streak = {}

        if completion_rate >= 0.5:
            # Escalate
            self._poor_streak[base_id] = 0
            harder = generate_harder_workflow(self._workflow, delta=1)
            self._generated_workflows[base_id] = harder
            self._state.generated_next_workflow = harder
            self._state.metadata["next_difficulty"] = harder.get("difficulty_level", 1)
            self._state.metadata["curriculum_direction"] = "UP"

        elif completion_rate < 0.3:
            # De-escalate after 2 consecutive poor episodes
            streak = self._poor_streak.get(base_id, 0) + 1
            self._poor_streak[base_id] = streak
            self._state.metadata["poor_streak"] = streak

            if streak >= 2:
                current = self._generated_workflows.get(base_id, self._workflow)
                easier = generate_easier_workflow(current)
                if easier is not None:
                    self._generated_workflows[base_id] = easier
                    self._state.generated_next_workflow = easier
                    self._state.metadata["next_difficulty"] = easier.get("difficulty_level", 1)
                    self._state.metadata["curriculum_direction"] = "DOWN"
                    self._poor_streak[base_id] = 0
        else:
            self._poor_streak[base_id] = 0
            self._state.metadata["curriculum_direction"] = "HOLD"

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _available_task_ids(self) -> List[str]:
        completed_set = set(self._completed_ids)
        return [
            t["task_id"]
            for t in self._workflow["tasks"]
            if t["task_id"] not in completed_set
            and all(d in completed_set for d in t.get("dependencies", []))
        ]

    def _pending_task_ids(self) -> List[str]:
        completed_set = set(self._completed_ids)
        available_set = set(self._available_task_ids())
        return [
            t["task_id"]
            for t in self._workflow["tasks"]
            if t["task_id"] not in completed_set
            and t["task_id"] not in available_set
        ]

    def _make_observation(
        self,
        feedback: str,
        reward: float,
        done: bool,
    ) -> AutopilotObservation:
        tasks_for_agent = [
            {
                "task_id": t["task_id"],
                "name": t["name"],
                "description": t["description"],
                "required_tool": t["required_tool"],
                "dependencies": t["dependencies"],
                "business_rule": t.get("business_rule"),
            }
            for t in self._workflow["tasks"]
        ]
        return AutopilotObservation(
            workflow_id=self._workflow["workflow_id"],
            workflow_name=self._workflow["name"],
            workflow_description=self._workflow["description"],
            tasks=tasks_for_agent,
            completed_task_ids=list(self._completed_ids),
            available_task_ids=self._available_task_ids(),
            pending_task_ids=self._pending_task_ids(),
            tool_results=list(self._tool_history[-5:]),  # last 5 only
            step_feedback=feedback,
            reward=reward,
            done=done,
            difficulty_level=self._state.difficulty_level,
            metadata={
                "step_count": self._state.step_count,
                "tasks_completed": self._state.tasks_completed,
                "tasks_total": self._state.tasks_total,
                "tool_summary": self._tools.summary(),
                "judge_enabled": self._judge_enabled,
                "judge_alpha": self._judge_alpha,
                "judge_model_loaded": bool(
                    self._learned_judge is not None
                    and hasattr(self._learned_judge, "enabled")
                    and self._learned_judge.enabled()
                ),
            },
        )

    def _build_feedback(
        self,
        action: AutopilotAction,
        breakdown: Dict,
        tool_result: Dict,
        done: bool,
    ) -> str:
        parts = []
        task_name = breakdown.get("task_name", "unknown")
        if breakdown.get("tool_score", 0) > 0:
            parts.append(f"Tool matched task: {task_name!r}")
        else:
            parts.append(f"No matching task for tool {action.tool!r}")
        if breakdown.get("dep_violation", 0) < 0:
            parts.append("Dependency violation: prerequisites not yet complete")
        if breakdown.get("rule_violation", 0) < 0:
            parts.append("Business rule violated")
        if tool_result:
            if tool_result.get("success"):
                parts.append(f"Tool succeeded: {list(tool_result.get('result', {}).keys())}")
            else:
                parts.append(f"Tool failed: {tool_result.get('error', '')}")
        parts.append(f"{self._state.tasks_completed}/{self._state.tasks_total} tasks complete")
        if done:
            rate = self._state.tasks_completed / max(1, self._state.tasks_total)
            parts.append(f"Episode done. Completion: {rate:.0%}")
            if self._state.generated_next_workflow:
                parts.append(
                    f"Next workflow generated (difficulty {self._state.generated_next_workflow.get('difficulty_level')})"
                )
        return " | ".join(parts)
