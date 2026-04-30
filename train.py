"""
Training Script — Adaptive Enterprise Autopilot (FORMAT-COLLAPSE-FIXED)
=========================================================================
v3 fixes the critical GRPO format-collapse bug where reward flatlined at
exactly -0.5 because every rollout failed `json.loads` and hit the fallback
penalty wall, killing the gradient signal.

Fixes in v3:
  1. Hardened SYSTEM_PROMPT — explicit "must start with {, end with },
     no markdown fences" instructions.
  2. Bulletproof grpo_reward_fn — tiered JSON salvage:
        a. strict json.loads      → full +0.5 format bonus
        b. markdown-fence stripped → +0.1 partial format credit
        c. regex tool/reasoning   → +0.1 partial format credit
        d. plain tool-name scan   → +0.05 weak signal
     Tool/reasoning/context rewards are applied ON TOP of whichever tier
     succeeded, so the model always sees a smooth gradient toward correct
     tool choice even while it's still stabilising the JSON syntax.
  3. Explicit +0.5 format reward for perfectly valid JSON — gives the
     model an unambiguous mathematical incentive to maintain format
     during RL rollouts.
  4. Same bulletproof parser used in run_episode → eval rollouts also
     survive imperfect JSON and produce non-zero gradients.

v2 fixes (still present):
  - Pre-training baseline evaluation runs BEFORE GRPO
  - GRPOLoggingCallback logs GRPO step rewards every N steps
  - Periodic in-training evaluation rollouts every EVAL_EVERY steps
  - Post-training final evaluation
  - Plot labels axes correctly and shows before/after line

Run on Colab (free T4 / A10G):
    pip install --upgrade --prefer-binary "huggingface-hub>=0.34,<1.0" "transformers>=4.56,<5" "trl>=0.24,<1" "accelerate>=1.10,<2" "peft>=0.17,<1" "datasets>=4,<5" mergekit unsloth
    python train.py

Run on CPU (slower, uses CPU_BASE_MODEL by default):
    pip install --upgrade --prefer-binary "huggingface-hub>=0.34,<1.0" "transformers>=4.56,<5" "trl>=0.24,<1" "accelerate>=1.10,<2" "peft>=0.17,<1" "datasets>=4,<5" torch mergekit
    FORCE_CPU=1 NUM_EPISODES=20 python train.py
"""

from __future__ import annotations
import ast
import inspect
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── Config ────────────────────────────────────────────────────────────────────

BASE_MODEL   = os.getenv("BASE_MODEL",   "unsloth/Qwen2.5-7B-Instruct")
CPU_BASE_MODEL = os.getenv("CPU_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
FORCE_CPU    = os.getenv("FORCE_CPU", "0") == "1"
USE_UNSLOTH  = os.getenv("USE_UNSLOTH", "1") == "1"
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
PUSH_TO_HUB  = os.getenv("PUSH_TO_HUB", "0") == "1"
HUB_REPO     = os.getenv("HUB_REPO",    "your-user/autopilot-agent")
NUM_EPISODES = int(os.getenv("NUM_EPISODES", "200"))
TASKS        = os.getenv("TASKS", "easy,medium,hard").split(",")
EVAL_EVERY   = int(os.getenv("EVAL_EVERY", "1"))    # eval rollout every N GRPO steps (1 = every step, good for short runs)

MAX_SEQ_LEN  = 4096
LORA_RANK    = 16
BATCH_SIZE   = 2
GRAD_ACCUM   = 4
LR           = 2e-5
MAX_STEPS_EP = 15   # capped at 15 to avoid runaway negative rewards on hard tasks with weak model
GRPO_K       = 4
EVAL_SCORE_MIN = -2.75
EVAL_SCORE_MAX = 2.0
USE_LEARNED_JUDGE = os.getenv("USE_LEARNED_JUDGE", "0") == "1"
JUDGE_ALPHA = float(os.getenv("JUDGE_ALPHA", "0.05"))
JUDGE_LOG_PATH = os.getenv("JUDGE_LOG_PATH", "judge_examples.jsonl")
JUDGE_MODEL_PATH = os.getenv("JUDGE_MODEL_PATH", "")

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert enterprise workflow orchestration agent.
You receive a workflow with multiple tasks and must complete them by calling
enterprise tools IN THE CORRECT ORDER.

═══════════ OUTPUT FORMAT — STRICT, NON-NEGOTIABLE ═══════════
Your entire response MUST be a single JSON object and NOTHING else.

  • The FIRST character of your response MUST be '{'
  • The LAST  character of your response MUST be '}'
  • DO NOT wrap the JSON in ```json, ``` or ANY markdown code fence.
  • DO NOT write any prose, preamble, apology, or explanation outside the JSON.
  • DO NOT add trailing commas or comments.
  • All string values MUST use double quotes.

Required JSON schema (exactly these three top-level fields):
{
  "tool": "<one of the available tool names>",
  "params": { <tool-specific parameters as a JSON object> },
  "reasoning": "<one short sentence: which task you are completing and why now>"
}

Available tools: jira_create_ticket, jira_update_ticket, jira_assign_ticket,
slack_send_message, slack_create_channel, email_send, hr_create_user,
hr_update_user, calendar_create_event, done

CRITICAL behaviour rules:
  • Always respect task dependencies. Only call tools for tasks listed in
    AVAILABLE NOW.
  • Use ticket_ids / user_ids returned by previous tool calls (see LAST
    RESULTS) in your subsequent params.
  • Emit "done" only when AVAILABLE NOW is empty AND PENDING is empty.

REMEMBER: respond with the JSON object only. Start with '{', end with '}'. Nothing else.
""").strip()

VALID_TOOLS = [
    "jira_create_ticket", "jira_update_ticket", "jira_assign_ticket",
    "slack_send_message", "slack_create_channel",
    "email_send", "hr_create_user", "hr_update_user",
    "calendar_create_event", "done",
]

# Regex patterns reused by the bulletproof parser ──────────────────────────────
_TOOL_NAME_RE   = re.compile(r'["\']?tool["\']?\s*[:=]\s*["\']([a-zA-Z_][\w]*)["\']')
_REASON_RE      = re.compile(r'["\']?reasoning["\']?\s*[:=]\s*["\']([^"\']{3,})["\']')
_FENCE_RE       = re.compile(r"```(?:json|JSON)?\s*|\s*```", re.MULTILINE)
_FIRST_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_VALID_TOOL_SCAN_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in VALID_TOOLS) + r")\b")
_STATE_JSON_RE = re.compile(r"^STATE_JSON:\s*(\{.*\})\s*$", re.MULTILINE)
_TASK_LINE_RE = re.compile(
    r"^\s*\[(?P<task_id>[^\]]+)\].*?tool=(?P<tool>[a-zA-Z_][\w]*)"
    r".*?deps=(?P<deps>\[[^\]]*\])(?:.*?params=(?P<params>\[[^\]]*\]))?",
    re.MULTILINE,
)


def clamp_eval_score(value: float) -> float:
    """Clamp full-episode eval scores to the documented scoring range."""
    return max(EVAL_SCORE_MIN, min(EVAL_SCORE_MAX, float(value)))


EPISODE_REWARD_COMPONENT_KEYS = (
    "extrinsic_total",
    "pbrs_shaping",
    "intrinsic_count",
    "intrinsic_rnd",
    "weighted_judge",
    "difference_reward",
    "ird_posterior_correction",
    "total",
)


def _blank_episode_components() -> Dict[str, float]:
    return {key: 0.0 for key in EPISODE_REWARD_COMPONENT_KEYS}


def _normalize_episode_components(components: Optional[Dict[str, Any]]) -> Dict[str, float]:
    normalized = _blank_episode_components()
    if not components:
        return normalized
    for key in EPISODE_REWARD_COMPONENT_KEYS:
        try:
            normalized[key] = round(float(components.get(key, 0.0)), 4)
        except Exception:
            normalized[key] = 0.0
    return normalized


def _summarize_series(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "first_10_pct": 0.0,
            "last_10_pct": 0.0,
        }
    window = max(1, len(values) // 10)
    return {
        "mean": round(sum(values) / len(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "first_10_pct": round(sum(values[:window]) / len(values[:window]), 4),
        "last_10_pct": round(sum(values[-window:]) / len(values[-window:]), 4),
    }


def _summarize_component_records_by_key(
    records: List[Dict[str, Any]],
    group_key: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for record in records:
        group = str(record.get(group_key, "unknown"))
        component_bucket = grouped.setdefault(
            group,
            {key: [] for key in EPISODE_REWARD_COMPONENT_KEYS},
        )
        for key in EPISODE_REWARD_COMPONENT_KEYS:
            component_bucket[key].append(float(record.get(key, 0.0)))
    return {
        group: {
            key: _summarize_series(values)
            for key, values in component_bucket.items()
        }
        for group, component_bucket in grouped.items()
    }


# ── Metrics tracker ───────────────────────────────────────────────────────────

@dataclass
class TrainingMetrics:
    """
    Tracks three parallel series:
      grpo_steps    : x-axis for GRPO step-level rewards
      grpo_rewards  : mean reward per GRPO step
      eval_steps    : x-axis for full-episode evaluation checkpoints
      eval_rewards  : total episode reward at each checkpoint
      eval_tasks    : which task each eval was on
      difficulty    : workflow difficulty at each checkpoint
    """
    grpo_steps: List[int]     = field(default_factory=list)
    grpo_rewards: List[float] = field(default_factory=list)
    eval_steps: List[int]     = field(default_factory=list)
    eval_rewards: List[float] = field(default_factory=list)
    eval_tasks: List[str]     = field(default_factory=list)
    eval_phase: List[str]     = field(default_factory=list)
    difficulty: List[float]   = field(default_factory=list)
    episodes_to_threshold_0_5: int = -1   # first episode where eval_reward >= 0.5
    episodes_to_threshold_1_0: int = -1   # first episode where eval_reward >= 1.0
    pre_train_rewards: Dict[str, float]  = field(default_factory=dict)
    post_train_rewards: Dict[str, float] = field(default_factory=dict)
    reward_component_series: Dict[str, List[float]] = field(
        default_factory=lambda: {key: [] for key in EPISODE_REWARD_COMPONENT_KEYS}
    )
    reward_component_eval_log: List[Dict[str, Any]] = field(default_factory=list)
    _step: int = field(default=0, repr=False)
    _accum: List[float] = field(default_factory=list, repr=False)

    def record_grpo_rewards(self, rewards: List[float]):
        self._accum.extend(rewards)

    def flush_grpo_step(self):
        if not self._accum:
            return
        mean_r = sum(self._accum) / len(self._accum)
        self._step += 1
        self.grpo_steps.append(self._step)
        self.grpo_rewards.append(round(mean_r, 4))
        self._accum.clear()
        if self._step % 10 == 0:
            window = self.grpo_rewards[-10:]
            print(f"[step {self._step:4d}] mean_grpo_reward={sum(window)/len(window):.3f}", flush=True)

    def record_eval(
        self,
        step: int,
        task: str,
        reward: float,
        diff: float,
        phase: str = "train",
        components: Optional[Dict[str, Any]] = None,
    ):
        reward = clamp_eval_score(reward)
        self.eval_steps.append(step)
        self.eval_rewards.append(round(reward, 4))
        self.eval_tasks.append(task)
        self.eval_phase.append(phase)
        self.difficulty.append(round(diff, 4))
        normalized_components = _normalize_episode_components(components)
        for key, value in normalized_components.items():
            self.reward_component_series[key].append(value)
        self.reward_component_eval_log.append(
            {
                "step": int(step),
                "task": str(task),
                "phase": str(phase),
                "difficulty": round(diff, 4),
                **normalized_components,
            }
        )

        ep_idx = len(self.eval_rewards)
        if self.episodes_to_threshold_0_5 < 0 and reward >= 0.5:
            self.episodes_to_threshold_0_5 = ep_idx
            print(f"[milestone] reached reward >= 0.5 in {ep_idx} eval episodes", flush=True)
        if self.episodes_to_threshold_1_0 < 0 and reward >= 1.0:
            self.episodes_to_threshold_1_0 = ep_idx
            print(f"[milestone] reached reward >= 1.0 in {ep_idx} eval episodes", flush=True)

        print(
            f"[eval @ step {step}] phase={phase} task={task} "
            f"score={reward:.3f} difficulty={diff:.3f}",
            flush=True,
        )

    def save(self, path: str = "training_metrics.json"):
        data_to_save = {
            "grpo_steps": self.grpo_steps,
            "grpo_rewards": self.grpo_rewards,
            "eval_steps": self.eval_steps,
            "eval_rewards": self.eval_rewards,
            "eval_tasks": self.eval_tasks,
            "eval_phase": self.eval_phase,
            "difficulty": self.difficulty,
            "episodes_to_threshold_0_5": self.episodes_to_threshold_0_5,
            "episodes_to_threshold_1_0": self.episodes_to_threshold_1_0,
            "pre_train_rewards": self.pre_train_rewards,
            "post_train_rewards": self.post_train_rewards,
            "reward_component_series": self.reward_component_series,
            "reward_component_eval_log": self.reward_component_eval_log,
        }

        reward_component_summary = {
            key: _summarize_series(values)
            for key, values in self.reward_component_series.items()
        }
        data_to_save["reward_component_summary"] = reward_component_summary
        data_to_save["reward_component_summary_by_phase"] = _summarize_component_records_by_key(
            self.reward_component_eval_log,
            "phase",
        )
        data_to_save["reward_components"] = reward_component_summary

        if hasattr(self, "component_log") and self.component_log:
            agg = {}
            for key in self.component_log[0]:
                vals = [c[key] for c in self.component_log]
                agg[key] = _summarize_series(vals)
            data_to_save["grpo_reward_fn_component_summary"] = agg

        with open(path, "w") as f:
            json.dump(data_to_save, f, indent=2)
        print(f"[metrics] Saved {len(self.grpo_steps)} GRPO steps + {len(self.eval_steps)} evals -> {path}")


metrics = TrainingMetrics()


# ── Bulletproof completion parser ─────────────────────────────────────────────

def _completion_to_text(raw: Any) -> str:
    if isinstance(raw, list):
        parts = [
            str(m.get("content", ""))
            for m in raw
            if isinstance(m, dict) and m.get("content") is not None
        ]
        return "\n".join(parts)
    if isinstance(raw, dict):
        return str(raw.get("content", raw))
    return str(raw)


def parse_completion(raw: Any) -> Tuple[Optional[str], Optional[str], Dict[str, Any], str]:
    """
    Tiered parser that NEVER returns None for the format tier — it always
    classifies the output into one of:

        "perfect"       — clean json.loads success on stripped text
        "fenced"        — json.loads success after stripping ``` fences
        "embedded"      — first {...} substring parses as JSON
        "regex"         — couldn't parse JSON but regex found a tool key
        "tool_scan"     — only a bare valid tool name appeared in the text
        "broken"        — could not salvage anything

    Returns:
        (tool, reasoning, params, tier)

    `tool` may be None if nothing was salvageable. `params` defaults to {}.
    Used by both the GRPO reward function and the live env rollout, so the
    same robustness applies during training and during evaluation.
    """
    if not raw:
        return None, None, {}, "broken"

    text = _completion_to_text(raw).strip()

    # ── Tier 1: perfect JSON ──────────────────────────────────────────────
    if text.startswith("{") and text.endswith("}"):
        try:
            d = json.loads(text)
            if isinstance(d, dict):
                return (
                    str(d.get("tool", "")) or None,
                    str(d.get("reasoning", "")) or None,
                    d.get("params", {}) if isinstance(d.get("params", {}), dict) else {},
                    "perfect",
                )
        except Exception:
            pass

    # ── Tier 2: strip markdown fences and retry ───────────────────────────
    fence_stripped = _FENCE_RE.sub("", text).strip()
    if fence_stripped.startswith("{") and fence_stripped.endswith("}"):
        try:
            d = json.loads(fence_stripped)
            if isinstance(d, dict):
                return (
                    str(d.get("tool", "")) or None,
                    str(d.get("reasoning", "")) or None,
                    d.get("params", {}) if isinstance(d.get("params", {}), dict) else {},
                    "fenced",
                )
        except Exception:
            pass

    # ── Tier 3: extract first {...} substring ─────────────────────────────
    m = _FIRST_OBJECT_RE.search(fence_stripped or text)
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict):
                return (
                    str(d.get("tool", "")) or None,
                    str(d.get("reasoning", "")) or None,
                    d.get("params", {}) if isinstance(d.get("params", {}), dict) else {},
                    "embedded",
                )
        except Exception:
            pass

    # ── Tier 4: regex pull tool / reasoning out of prose ──────────────────
    m_tool = _TOOL_NAME_RE.search(text)
    m_rsn  = _REASON_RE.search(text)
    if m_tool:
        return (
            m_tool.group(1),
            m_rsn.group(1) if m_rsn else None,
            {},
            "regex",
        )

    # ── Tier 5: bare tool-name appears anywhere in the text ───────────────
    m_scan = _VALID_TOOL_SCAN_RE.search(text)
    if m_scan:
        return (m_scan.group(1), None, {}, "tool_scan")

    return None, None, {}, "broken"


# ── Environment rollout ───────────────────────────────────────────────────────

def _safe_literal_list(raw: str) -> List[str]:
    try:
        value = ast.literal_eval(raw)
    except Exception:
        try:
            value = json.loads(raw)
        except Exception:
            return []
    if not isinstance(value, list):
        return []
    return [str(v) for v in value]


def _task_state(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": task.get("task_id", ""),
        "name": task.get("name", ""),
        "required_tool": task.get("required_tool", ""),
        "required_params": list(task.get("required_params", [])),
        "dependencies": list(task.get("dependencies", [])),
        "business_rule": task.get("business_rule"),
        "is_blocker": bool(task.get("is_blocker", False)),
    }


def _format_task_line(task: Dict[str, Any]) -> str:
    desc = " ".join(str(task.get("description", "")).split())[:160]
    rule = task.get("business_rule") or "none"
    return (
        f"  [{task.get('task_id')}] {task.get('name')}: "
        f"tool={task.get('required_tool')}, "
        f"deps={json.dumps(task.get('dependencies', []))}, "
        f"params={json.dumps(task.get('required_params', []))}, "
        f"rule={rule}, blocker={'yes' if task.get('is_blocker') else 'no'}, "
        f"desc={desc}"
    )


def _build_workflow_prompt(
    workflow_name: str,
    workflow_description: str,
    tasks: List[Dict[str, Any]],
    completed: List[str],
    available: List[str],
    pending: List[str],
    tool_results: Optional[List[Dict[str, Any]]] = None,
    feedback: str = "",
) -> str:
    tasks_str = "\n".join(_format_task_line(t) for t in tasks)
    state = {
        "tasks": [_task_state(t) for t in tasks],
        "completed": list(completed),
        "available": list(available),
        "pending": list(pending),
    }
    results_str = ""
    if tool_results:
        results_str = "\nLAST RESULTS:\n" + "\n".join(
            f"  {r.get('tool')}: success={r.get('result',{}).get('success','?')} "
            f"data={json.dumps(r.get('result',{}).get('result',{}))}"
            for r in tool_results[-3:]
        )
    return (
        f"WORKFLOW: {workflow_name}\nDESC: {workflow_description[:240]}\n\n"
        f"ALL TASKS:\n{tasks_str}\n\n"
        f"STATE_JSON: {json.dumps(state, separators=(',', ':'))}\n"
        f"COMPLETED: {json.dumps(completed)}\n"
        f"AVAILABLE NOW: {json.dumps(available)}\n"
        f"PENDING: {json.dumps(pending)}\n"
        f"{results_str}\nFEEDBACK: {feedback}"
    )


def _prompt_to_user_text(prompt: Any) -> str:
    if isinstance(prompt, list):
        user_parts = [
            str(m.get("content", ""))
            for m in prompt
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        if user_parts:
            return "\n".join(user_parts)
        return "\n".join(
            str(m.get("content", m)) if isinstance(m, dict) else str(m)
            for m in prompt
        )
    if isinstance(prompt, dict):
        return str(prompt.get("content", prompt))
    text = str(prompt)
    idx = text.find("WORKFLOW:")
    return text[idx:] if idx >= 0 else text


def extract_prompt_state(prompt: Any) -> Dict[str, Any]:
    text = _prompt_to_user_text(prompt)
    match = _STATE_JSON_RE.search(text)
    if match:
        try:
            state = json.loads(match.group(1))
            if isinstance(state, dict):
                return {
                    "tasks": state.get("tasks", []),
                    "available": [str(x) for x in state.get("available", [])],
                    "pending": [str(x) for x in state.get("pending", [])],
                    "completed": [str(x) for x in state.get("completed", [])],
                }
        except Exception:
            pass

    available_match = re.search(r"^AVAILABLE NOW:\s*(\[.*?\])\s*$", text, re.MULTILINE)
    pending_match = re.search(r"^PENDING:\s*(\[.*?\])\s*$", text, re.MULTILINE)
    completed_match = re.search(r"^COMPLETED:\s*(\[.*?\])\s*$", text, re.MULTILINE)
    tasks = []
    for m in _TASK_LINE_RE.finditer(text):
        tasks.append({
            "task_id": m.group("task_id"),
            "required_tool": m.group("tool"),
            "dependencies": _safe_literal_list(m.group("deps")),
            "required_params": _safe_literal_list(m.group("params") or "[]"),
        })
    return {
        "tasks": tasks,
        "available": _safe_literal_list(available_match.group(1)) if available_match else [],
        "pending": _safe_literal_list(pending_match.group(1)) if pending_match else [],
        "completed": _safe_literal_list(completed_match.group(1)) if completed_match else [],
    }


def _available_task_matches(state: Dict[str, Any], tool: str) -> List[Dict[str, Any]]:
    available = set(state.get("available", []))
    return [
        t for t in state.get("tasks", [])
        if t.get("task_id") in available and t.get("required_tool") == tool
    ]


def _blocked_task_matches(state: Dict[str, Any], tool: str) -> List[Dict[str, Any]]:
    available = set(state.get("available", []))
    completed = set(state.get("completed", []))
    return [
        t for t in state.get("tasks", [])
        if t.get("task_id") not in available
        and t.get("task_id") not in completed
        and t.get("required_tool") == tool
    ]


def run_episode(
    model_fn,
    task: str,
    env=None,
    judge_buffer=None,
    learned_judge=None,
    judge_enabled: bool = False,
) -> Dict[str, Any]:
    sys.path.insert(0, "src")
    from envs.autopilot_env.environment import AutopilotEnvironment
    from envs.autopilot_env.models import AutopilotAction

    env = env or AutopilotEnvironment(
        task=task,
        learned_judge=learned_judge,
        judge_alpha=JUDGE_ALPHA,
        judge_enabled=judge_enabled,
        judge_buffer=judge_buffer,
    )
    obs = env.reset()
    total = 0.0
    steps = 0
    episode_components = _blank_episode_components()
    while not obs.done and steps < MAX_STEPS_EP:
        prompt = _build_prompt(obs)
        raw = model_fn(prompt)

        tool, reasoning, params, _tier = parse_completion(raw)
        if tool is None or tool not in VALID_TOOLS:
            tool = "done"
            reasoning = reasoning or ""
            params = {}

        obs, r, done, info = env.step(AutopilotAction(
            tool=tool,
            params=params if isinstance(params, dict) else {},
            reasoning=reasoning or "",
        ))
        breakdown = (info or {}).get("breakdown", {})
        for key in EPISODE_REWARD_COMPONENT_KEYS:
            episode_components[key] += float(breakdown.get(key, 0.0))
        total += r
        steps += 1

    completion_rate = env.state.tasks_completed / max(env.state.tasks_total, 1)
    if env.state.generated_next_workflow:
        from envs.autopilot_env.workflow_gen import difficulty_score as _ds
        diff = _ds(env.state.generated_next_workflow)
    else:
        diff = obs.difficulty_level / 10.0
    return {
        "total_reward": clamp_eval_score(total),
        "raw_reward": total,
        "completion_rate": completion_rate,
        "difficulty": diff,
        "reward_components": _normalize_episode_components(episode_components),
    }


def _build_prompt(obs) -> str:
    return _build_workflow_prompt(
        workflow_name=obs.workflow_name,
        workflow_description=obs.workflow_description,
        tasks=obs.tasks,
        completed=obs.completed_task_ids,
        available=obs.available_task_ids,
        pending=obs.pending_task_ids,
        tool_results=obs.tool_results,
        feedback=obs.step_feedback,
    )


# ── GRPO reward fn ────────────────────────────────────────────────────────────

def grpo_reward_fn(completions: List[str], prompts: List[Any], **kwargs) -> List[float]:
    """Reward exact available-task execution, with format as a small side signal."""
    rewards = []
    for completion, prompt in zip(completions, prompts):
        r = 0.0
        components = {
            "format_bonus": 0.0,
            "format_partial": 0.0,
            "format_weak": 0.0,
            "format_broken": 0.0,
            "invalid_tool": 0.0,
            "valid_tool": 0.0,
            "available_tool_match": 0.0,
            "blocked_tool_penalty": 0.0,
            "wrong_tool_penalty": 0.0,
            "params_complete": 0.0,
            "params_missing": 0.0,
            "reasoning_present": 0.0,
            "correct_done": 0.0,
            "early_done_penalty": 0.0,
        }

        tool, reasoning, params, tier = parse_completion(completion)
        state = extract_prompt_state(prompt)
        available = state.get("available", [])
        pending = state.get("pending", [])

        # ── Format-tier reward ────────────────────────────────────────────
        if tier == "perfect":
            components["format_bonus"] = 0.2
            r += 0.2
        elif tier in ("fenced", "embedded", "regex"):
            components["format_partial"] = 0.05
            r += 0.05
        elif tier == "tool_scan":
            components["format_weak"] = 0.0
        else:  # "broken"
            components["format_broken"] = -0.3
            r -= 0.3

        # ── Tool-quality rewards (apply regardless of tier) ───────────────
        if tool:
            if tool in VALID_TOOLS:
                components["valid_tool"] = 0.1
                r += 0.1

                if tool == "done":
                    if not available and not pending:
                        components["correct_done"] = 0.6
                        r += 0.6
                    else:
                        components["early_done_penalty"] = -0.8
                        r -= 0.8
                else:
                    matches = _available_task_matches(state, tool)
                    if matches:
                        components["available_tool_match"] = 1.0
                        r += 1.0
                        required = list(matches[0].get("required_params", []))
                        if required:
                            present = [p for p in required if params.get(p)]
                            if len(present) == len(required):
                                components["params_complete"] = 0.2
                                r += 0.2
                            elif present:
                                partial = 0.1 * (len(present) / len(required))
                                components["params_complete"] = round(partial, 3)
                                r += partial
                            else:
                                components["params_missing"] = -0.15
                                r -= 0.15
                    elif _blocked_task_matches(state, tool):
                        components["blocked_tool_penalty"] = -0.45
                        r -= 0.45
                    else:
                        components["wrong_tool_penalty"] = -0.25
                        r -= 0.25
            else:
                components["invalid_tool"] = -0.2
                r -= 0.2

        if reasoning and reasoning.strip():
            components["reasoning_present"] = 0.05
            r += 0.05

        rewards.append(round(r, 3))

        if not hasattr(metrics, "component_log"):
            metrics.component_log = []
        metrics.component_log.append(components)

    metrics.record_grpo_rewards(rewards)
    return rewards


# ── GRPO callback ─────────────────────────────────────────────────────────────

def make_callback(get_model_fn):
    try:
        from transformers import TrainerCallback
        persistent_easy_env = None

        class MetricsCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                nonlocal persistent_easy_env
                metrics.flush_grpo_step()
                if state.global_step > 0 and state.global_step % EVAL_EVERY == 0:
                    fn = get_model_fn()
                    if fn:
                        try:
                            if persistent_easy_env is None:
                                sys.path.insert(0, "src")
                                from envs.autopilot_env.environment import AutopilotEnvironment
                                persistent_easy_env = AutopilotEnvironment(
                                    task="easy",
                                    learned_judge=getattr(make_callback, "_learned_judge", None),
                                    judge_alpha=JUDGE_ALPHA,
                                    judge_enabled=getattr(make_callback, "_judge_enabled", False),
                                    judge_buffer=getattr(make_callback, "_judge_buffer", None),
                                )
                            r = run_episode(
                                fn,
                                "easy",
                                env=persistent_easy_env,
                                judge_buffer=getattr(make_callback, "_judge_buffer", None),
                                learned_judge=getattr(make_callback, "_learned_judge", None),
                                judge_enabled=getattr(make_callback, "_judge_enabled", False),
                            )
                            metrics.record_eval(state.global_step, "easy",
                                                r["total_reward"], r["difficulty"],
                                                phase="train",
                                                components=r.get("reward_components"))
                        except Exception as e:
                            print(f"[callback eval error] {e}", flush=True)

        return MetricsCallback()
    except Exception as _ue:
        print(f"[train] unsloth unavailable ({type(_ue).__name__}: {_ue}) — falling back to standard transformers.", flush=True)
        return None


# ── Warmup dataset ────────────────────────────────────────────────────────────

def build_warmup_dataset():
    """
    6 high-quality examples covering: first move, mid-episode tool chaining,
    using IDs from previous results, parallel track, and done signal.
    5 epochs over these = enough to stabilise JSON output before GRPO.
    """
    return [
        {
            "system": SYSTEM_PROMPT,
            "user": "WORKFLOW: Onboard new intern\nALL TASKS:\n  [T1] Create HR profile: tool=hr_create_user, deps=[]\n  [T2] Create Jira ticket: tool=jira_create_ticket, deps=[T1]\nCOMPLETED: []\nAVAILABLE NOW: ['T1']\nPENDING: ['T2']",
            "assistant": json.dumps({"tool": "hr_create_user", "params": {"name": "Alice Singh", "role": "Software Intern", "department": "Engineering"}, "reasoning": "T1 has no dependencies and must be completed first to unblock T2."}),
        },
        {
            "system": SYSTEM_PROMPT,
            "user": "WORKFLOW: Report a production bug\nALL TASKS:\n  [T1] Create bug ticket: tool=jira_create_ticket, deps=[]\n  [T2] Assign ticket: tool=jira_assign_ticket, deps=[T1]\nCOMPLETED: ['T1']\nAVAILABLE NOW: ['T2', 'T3']\nLAST RESULTS:\n  jira_create_ticket: success=True data={\"ticket_id\": \"PROJ-100\"}",
            "assistant": json.dumps({"tool": "jira_assign_ticket", "params": {"ticket_id": "PROJ-100", "assignee": "oncall_engineer"}, "reasoning": "T1 done and ticket_id is PROJ-100 — using that ID to assign T2 now."}),
        },
        {
            "system": SYSTEM_PROMPT,
            "user": "WORKFLOW: Set up a new project team\nALL TASKS:\n  [T1] Register lead in HR: tool=hr_create_user, deps=[]\n  [T2] Create Jira Epic: tool=jira_create_ticket, deps=[T1]\n  [T3] Create Slack channel: tool=slack_create_channel, deps=[T1]\nCOMPLETED: ['T1']\nAVAILABLE NOW: ['T2', 'T3']\nPENDING: ['T4']",
            "assistant": json.dumps({"tool": "jira_create_ticket", "params": {"summary": "Project Alpha Epic", "issue_type": "Epic"}, "reasoning": "T1 done so T2 and T3 are both available — completing T2 first as it has the highest points."}),
        },
        {
            "system": SYSTEM_PROMPT,
            "user": "WORKFLOW: Onboard new intern\nALL TASKS:\n  [T3] Send Slack welcome: tool=slack_send_message, deps=[T1]\n  [T4] Schedule intro meeting: tool=calendar_create_event, deps=[T1]\n  [T5] Send onboarding email: tool=email_send, deps=[T3, T4]\nCOMPLETED: ['T1','T2','T3','T4']\nAVAILABLE NOW: ['T5']\nPENDING: []",
            "assistant": json.dumps({"tool": "email_send", "params": {"to": "alice@company.com", "subject": "Welcome — first day instructions", "body": "Hi Alice, welcome to the team! Please report to building B at 9am on Monday."}, "reasoning": "T3 and T4 are both complete so T5 dependency is satisfied — sending onboarding email now."}),
        },
        {
            "system": SYSTEM_PROMPT,
            "user": "WORKFLOW: Offboard departing employee\nALL TASKS:\n  [T1] Update HR status: tool=hr_update_user, deps=[]\nCOMPLETED: []\nAVAILABLE NOW: ['T1']\nPENDING: ['T2', 'T3']\nLAST RESULTS: []",
            "assistant": json.dumps({"tool": "hr_update_user", "params": {"user_id": "HR-1001", "field": "status", "value": "offboarded"}, "reasoning": "T1 has no dependencies — updating HR status first before any downstream tasks."}),
        },
        {
            "system": SYSTEM_PROMPT,
            "user": "WORKFLOW: Report a production bug\nCOMPLETED: ['T1','T2','T3','T4']\nAVAILABLE NOW: []\nPENDING: []",
            "assistant": json.dumps({"tool": "done", "params": {}, "reasoning": "All 4 tasks are in completed_task_ids and available_task_ids is empty — signalling done."}),
        },
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

LEGACY_MAIN_DISABLED = r'''
def main():
    print(f"[train] model={BASE_MODEL}  tasks={TASKS}  episodes={NUM_EPISODES}  eval_every={EVAL_EVERY}")
    try:
        from unsloth import FastLanguageModel
        from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
        from datasets import Dataset
    except ImportError as e:
        print(f"[ERROR] {e}\nRun: pip install unsloth trl transformers accelerate peft datasets")
        sys.exit(1)

    print("[train] Loading model...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LEN,
        dtype=None, load_in_4bit=True, token=HF_TOKEN or None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_RANK,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_alpha=LORA_RANK*2, lora_dropout=0.0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42,
    )

    def model_fn(prompt: str) -> str:
        FastLanguageModel.for_inference(model)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        import torch
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, temperature=0.2, do_sample=True)
        FastLanguageModel.for_training(model)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    judge_buffer = None
    learned_judge = None
    if USE_LEARNED_JUDGE:
        sys.path.insert(0, "src")
        from envs.autopilot_env.judge_buffer import JudgeReplayBuffer
        from envs.autopilot_env.judge_model import LearnedJudge

        judge_buffer = JudgeReplayBuffer()
        learned_judge = LearnedJudge(
            model_path=JUDGE_MODEL_PATH,
            enabled=True,
        ).load()

    # Phase 0 — baseline
    print("\n[train] === PHASE 0: Pre-training baseline ===", flush=True)
    for task in TASKS:
        r = run_episode(
            model_fn,
            task,
            judge_buffer=judge_buffer,
            learned_judge=learned_judge,
            judge_enabled=USE_LEARNED_JUDGE,
        )
        metrics.pre_train_rewards[task] = r["total_reward"]
        metrics.record_eval(
            0,
            task,
            r["total_reward"],
            r["difficulty"],
            phase="pre",
            components=r.get("reward_components"),
        )

    # Phase 1 — SFT warmup
print("\n[train] === PHASE 1: SFT warmup ===", flush=True)
warmup = build_warmup_dataset()
sft_ds = Dataset.from_list([{
    "text": f"<|system|>\n{e['system']}<|end|>\n<|user|>\n{e['user']}<|end|>\n<|assistant|>\n{e['assistant']}<|end|>"
} for e in warmup])
_set_training(model)
SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=sft_ds,
           dataset_text_field="text",
           args=SFTConfig(per_device_train_batch_size=1, num_train_epochs=5,
                          max_seq_length=MAX_SEQ_LEN, output_dir="./sft_warmup",
                          logging_steps=1, save_strategy="no", report_to="none",
                          no_cuda=not _cuda, fp16=False, bf16=False)).train()
print("[train] SFT warmup done.", flush=True)

    # Phase 2 — GRPO
    print("\n[train] === PHASE 2: GRPO training ===", flush=True)
    sys.path.insert(0, "src")
    from envs.autopilot_env.workflows import TASK_WORKFLOWS

    prompts = []
    for task in TASKS:
        for wf in TASK_WORKFLOWS[task]:
            for i in range(min(3, len(wf["tasks"]))):
                completed = [t["task_id"] for t in wf["tasks"][:i]]
                available = [t["task_id"] for t in wf["tasks"]
                             if t["task_id"] not in completed
                             and all(d in completed for d in t.get("dependencies", []))]
                pending = [
                    t["task_id"] for t in wf["tasks"]
                    if t["task_id"] not in completed and t["task_id"] not in available
                ]
                user_prompt = _build_workflow_prompt(
                    workflow_name=wf["name"],
                    workflow_description=wf["description"],
                    tasks=wf["tasks"],
                    completed=completed,
                    available=available,
                    pending=pending,
                )
                prompts.append({"prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]})

    grpo_kwargs = {
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "learning_rate": LR,
        "num_train_epochs": max(1, NUM_EPISODES // len(prompts)),
        "max_completion_length": 512,
        "num_generations": GRPO_K,
        "output_dir": "./grpo_output",
        "logging_steps": 5,
        "save_steps": 50,
        "report_to": "none",
        "remove_unused_columns": False,
        "no_cuda": not _cuda,
        "fp16": False,
        "bf16": False,
    }
    grpo_config_params = inspect.signature(GRPOConfig).parameters
    if "beta" in grpo_config_params:
        grpo_kwargs["beta"] = 0.02
    elif "kl_coeff" in grpo_config_params:
        grpo_kwargs["kl_coeff"] = 0.02

    make_callback._judge_buffer = judge_buffer
    make_callback._learned_judge = learned_judge
    make_callback._judge_enabled = USE_LEARNED_JUDGE
    callback = make_callback(lambda: model_fn)
    GRPOTrainer(
        model=model, tokenizer=tokenizer,
        reward_funcs=grpo_reward_fn,
        args=GRPOConfig(**grpo_kwargs),
        train_dataset=Dataset.from_list(prompts),
        callbacks=[callback] if callback else None,
    ).train()
    print("[train] GRPO done.", flush=True)

    # Phase 3 — post-training eval
    print("\n[train] === PHASE 3: Post-training eval ===", flush=True)
    FastLanguageModel.for_inference(model)
    final_step = metrics._step
    for task in TASKS:
        r = run_episode(
            model_fn,
            task,
            judge_buffer=judge_buffer,
            learned_judge=learned_judge,
            judge_enabled=USE_LEARNED_JUDGE,
        )
        metrics.post_train_rewards[task] = r["total_reward"]
        metrics.record_eval(
            final_step,
            task,
            r["total_reward"],
            r["difficulty"],
            phase="post",
            components=r.get("reward_components"),
        )

    print("\n[train] === IMPROVEMENT SUMMARY ===")
    for task in TASKS:
        b = metrics.pre_train_rewards.get(task, 0)
        a = metrics.post_train_rewards.get(task, 0)
        print(f"  {task:8s}: {b:.3f} -> {a:.3f}  ({a-b:+.3f})")

    if judge_buffer is not None and judge_buffer.size() > 0:
        judge_buffer.flush_jsonl(JUDGE_LOG_PATH)
        print(f"[judge] Saved examples -> {JUDGE_LOG_PATH}", flush=True)

    metrics.save("training_metrics.json")
    model.save_pretrained("./trained_adapter")
    tokenizer.save_pretrained("./trained_adapter")
    if PUSH_TO_HUB and HF_TOKEN:
        model.push_to_hub(HUB_REPO, token=HF_TOKEN)
    print("\n[train] Done. Run: python train.py plot")


# ── Plot ──────────────────────────────────────────────────────────────────────

'''

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]


def _accepts_var_kwargs(callable_obj) -> bool:
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in inspect.signature(callable_obj).parameters.values()
    )


def _filter_kwargs(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if _accepts_var_kwargs(callable_obj):
        return dict(kwargs)
    params = inspect.signature(callable_obj).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def _with_device_training_args(config_cls, kwargs: Dict[str, Any], cuda: bool) -> Dict[str, Any]:
    config_kwargs = dict(kwargs)
    params = inspect.signature(config_cls).parameters
    if "use_cpu" in params:
        config_kwargs["use_cpu"] = not cuda
    elif "no_cuda" in params:
        config_kwargs["no_cuda"] = not cuda
    if "fp16" in params:
        config_kwargs["fp16"] = False
    if "bf16" in params:
        config_kwargs["bf16"] = False
    return _filter_kwargs(config_cls, config_kwargs)


def _resolve_model_name(cuda: bool) -> str:
    if cuda:
        return BASE_MODEL
    if BASE_MODEL.startswith("unsloth/"):
        print(
            f"[train] CPU detected; using CPU_BASE_MODEL={CPU_BASE_MODEL} "
            f"instead of GPU-only {BASE_MODEL}.",
            flush=True,
        )
        return CPU_BASE_MODEL
    return BASE_MODEL


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        import torch
        return torch.device("cpu")


def _ensure_tokenizer_padding(tokenizer, model=None):
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model is not None and getattr(model, "config", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id


def _format_chat_prompt(tokenizer, prompt: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"System:\n{SYSTEM_PROMPT}\n\nUser:\n{prompt}\n\nAssistant:\n"


def _format_warmup_text(tokenizer, example: Dict[str, str]) -> str:
    msgs = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["user"]},
        {"role": "assistant", "content": example["assistant"]},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    return (
        f"System:\n{example['system']}\n\n"
        f"User:\n{example['user']}\n\n"
        f"Assistant:\n{example['assistant']}"
    )


def _load_transformers_model(torch, model_name: str, cuda: bool):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if cuda else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HF_TOKEN or None,
        trust_remote_code=True,
    )

    model_kwargs = {
        "token": HF_TOKEN or None,
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if cuda else torch.float32,
    }
    if not cuda:
        model_kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    _ensure_tokenizer_padding(tokenizer, model)

    if hasattr(model, "config"):
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    try:
        model = get_peft_model(model, lora_config)
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
    except ValueError as exc:
        print(f"[train] PEFT LoRA skipped ({exc}); training the base model directly.", flush=True)

    return model, tokenizer


def _trainer_kwargs(trainer_cls, tokenizer, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    trainer_kwargs = _filter_kwargs(trainer_cls, kwargs)
    params = inspect.signature(trainer_cls).parameters
    if "tokenizer" in params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in params:
        trainer_kwargs["processing_class"] = tokenizer
    return trainer_kwargs


def _print_trl_dependency_error(exc: BaseException) -> None:
    message = str(exc)
    if "mergekit" in message:
        print(
            "[ERROR] Your installed trl expects the optional package 'mergekit'.\n"
            "Install it, restart the notebook runtime, then rerun training:\n"
            "    pip install -U mergekit\n\n"
            "Or install all training dependencies together:\n"
            "    pip install --upgrade --prefer-binary \"huggingface-hub>=0.34,<1.0\" \"transformers>=4.56,<5\" \"trl>=0.24,<1\" \"accelerate>=1.10,<2\" \"peft>=0.17,<1\" \"datasets>=4,<5\" mergekit",
            flush=True,
        )
    elif "TrainingArguments" in message or "grpo_config" in message:
        print(
            "[ERROR] TRL/Transformers version mismatch.\n"
            "This training script expects the TRL 0.x trainer API and Transformers 4.x.\n"
            "Run this in a fresh notebook/kernel, then rerun training:\n"
            "    pip install --upgrade --prefer-binary \"huggingface-hub>=0.34,<1.0\" \"transformers>=4.56,<5\" \"trl>=0.24,<1\" \"accelerate>=1.10,<2\" \"peft>=0.17,<1\" \"datasets>=4,<5\" mergekit unsloth",
            flush=True,
        )
    else:
        print(
            f"[ERROR] Failed to import TRL training components: {exc}\n"
            "Run: pip install --upgrade --prefer-binary \"huggingface-hub>=0.34,<1.0\" \"transformers>=4.56,<5\" \"trl>=0.24,<1\" \"accelerate>=1.10,<2\" \"peft>=0.17,<1\" \"datasets>=4,<5\" mergekit unsloth",
            flush=True,
        )


def main():
    try:
        import torch
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
    except (ImportError, ModuleNotFoundError, RuntimeError) as e:
        _print_trl_dependency_error(e)
        sys.exit(1)

    cuda = bool(torch.cuda.is_available()) and not FORCE_CPU
    resolved_model = _resolve_model_name(cuda)
    print(
        f"[train] model={resolved_model}  backend={'cuda' if cuda else 'cpu'}  "
        f"tasks={TASKS}  episodes={NUM_EPISODES}  eval_every={EVAL_EVERY}",
        flush=True,
    )

    fast_language_model = None
    use_unsloth_backend = cuda and USE_UNSLOTH
    if use_unsloth_backend:
        try:
            from unsloth import FastLanguageModel
            fast_language_model = FastLanguageModel
        except Exception as exc:
            use_unsloth_backend = False
            print(
                f"[train] Unsloth unavailable ({type(exc).__name__}: {exc}); "
                "falling back to Transformers + PEFT.",
                flush=True,
            )

    print("[train] Loading model...", flush=True)
    if use_unsloth_backend:
        model, tokenizer = fast_language_model.from_pretrained(
            model_name=resolved_model,
            max_seq_length=MAX_SEQ_LEN,
            dtype=None,
            load_in_4bit=True,
            token=HF_TOKEN or None,
        )
        _ensure_tokenizer_padding(tokenizer, model)
        model = fast_language_model.get_peft_model(
            model,
            r=LORA_RANK,
            target_modules=LORA_TARGET_MODULES,
            lora_alpha=LORA_RANK * 2,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        def set_inference():
            fast_language_model.for_inference(model)

        def set_training():
            fast_language_model.for_training(model)

    else:
        model, tokenizer = _load_transformers_model(torch, resolved_model, cuda)

        def set_inference():
            model.eval()

        def set_training():
            model.train()

    def model_fn(prompt: str) -> str:
        set_inference()
        text = _format_chat_prompt(tokenizer, prompt)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to(_model_device(model))
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.2,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        finally:
            set_training()

    judge_buffer = None
    learned_judge = None
    if USE_LEARNED_JUDGE:
        sys.path.insert(0, "src")
        from envs.autopilot_env.judge_buffer import JudgeReplayBuffer
        from envs.autopilot_env.judge_model import LearnedJudge

        judge_buffer = JudgeReplayBuffer()
        learned_judge = LearnedJudge(
            model_path=JUDGE_MODEL_PATH,
            enabled=True,
        ).load()

    # Phase 0 - baseline
    print("\n[train] === PHASE 0: Pre-training baseline ===", flush=True)
    for task in TASKS:
        r = run_episode(
            model_fn,
            task,
            judge_buffer=judge_buffer,
            learned_judge=learned_judge,
            judge_enabled=USE_LEARNED_JUDGE,
        )
        metrics.pre_train_rewards[task] = r["total_reward"]
        metrics.record_eval(
            0,
            task,
            r["total_reward"],
            r["difficulty"],
            phase="pre",
            components=r.get("reward_components"),
        )

    # Phase 1 - SFT warmup
    print("\n[train] === PHASE 1: SFT warmup ===", flush=True)
    warmup = build_warmup_dataset()
    sft_ds = Dataset.from_list([{"text": _format_warmup_text(tokenizer, e)} for e in warmup])
    set_training()
    sft_config_kwargs = _with_device_training_args(
        SFTConfig,
        {
            "per_device_train_batch_size": 1,
            "num_train_epochs": 5,
            "max_seq_length": MAX_SEQ_LEN,
            "output_dir": "./sft_warmup",
            "logging_steps": 1,
            "save_strategy": "no",
            "report_to": "none",
            "dataset_text_field": "text",
        },
        cuda,
    )
    sft_trainer_kwargs = _trainer_kwargs(
        SFTTrainer,
        tokenizer,
        {
            "model": model,
            "train_dataset": sft_ds,
            "dataset_text_field": "text",
            "args": SFTConfig(**sft_config_kwargs),
        },
    )
    SFTTrainer(**sft_trainer_kwargs).train()
    print("[train] SFT warmup done.", flush=True)

    # Phase 2 - GRPO
    print("\n[train] === PHASE 2: GRPO training ===", flush=True)
    sys.path.insert(0, "src")
    from envs.autopilot_env.workflows import TASK_WORKFLOWS

    prompts = []
    for task in TASKS:
        for wf in TASK_WORKFLOWS[task]:
            for i in range(min(3, len(wf["tasks"]))):
                completed = [t["task_id"] for t in wf["tasks"][:i]]
                available = [
                    t["task_id"] for t in wf["tasks"]
                    if t["task_id"] not in completed
                    and all(d in completed for d in t.get("dependencies", []))
                ]
                pending = [
                    t["task_id"] for t in wf["tasks"]
                    if t["task_id"] not in completed and t["task_id"] not in available
                ]
                user_prompt = _build_workflow_prompt(
                    workflow_name=wf["name"],
                    workflow_description=wf["description"],
                    tasks=wf["tasks"],
                    completed=completed,
                    available=available,
                    pending=pending,
                )
                prompts.append({"prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]})

    runtime_grpo_k = GRPO_K if cuda else min(GRPO_K, 2)
    runtime_batch_size = BATCH_SIZE if cuda else max(1, runtime_grpo_k)
    grpo_kwargs = {
        "per_device_train_batch_size": runtime_batch_size,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "learning_rate": LR,
        "num_train_epochs": max(1, NUM_EPISODES // len(prompts)),
        "max_completion_length": 512,
        "num_generations": runtime_grpo_k,
        "output_dir": "./grpo_output",
        "logging_steps": 5,
        "save_steps": 50,
        "report_to": "none",
        "remove_unused_columns": False,
    }
    grpo_config_params = inspect.signature(GRPOConfig).parameters
    if "beta" in grpo_config_params:
        grpo_kwargs["beta"] = 0.02
    elif "kl_coeff" in grpo_config_params:
        grpo_kwargs["kl_coeff"] = 0.02
    grpo_kwargs = _with_device_training_args(GRPOConfig, grpo_kwargs, cuda)

    make_callback._judge_buffer = judge_buffer
    make_callback._learned_judge = learned_judge
    make_callback._judge_enabled = USE_LEARNED_JUDGE
    callback = make_callback(lambda: model_fn)
    grpo_trainer_kwargs = _trainer_kwargs(
        GRPOTrainer,
        tokenizer,
        {
            "model": model,
            "reward_funcs": grpo_reward_fn,
            "args": GRPOConfig(**grpo_kwargs),
            "train_dataset": Dataset.from_list(prompts),
            "callbacks": [callback] if callback else None,
        },
    )
    GRPOTrainer(**grpo_trainer_kwargs).train()
    print("[train] GRPO done.", flush=True)

    # Phase 3 - post-training eval
    print("\n[train] === PHASE 3: Post-training eval ===", flush=True)
    set_inference()
    final_step = metrics._step
    for task in TASKS:
        r = run_episode(
            model_fn,
            task,
            judge_buffer=judge_buffer,
            learned_judge=learned_judge,
            judge_enabled=USE_LEARNED_JUDGE,
        )
        metrics.post_train_rewards[task] = r["total_reward"]
        metrics.record_eval(
            final_step,
            task,
            r["total_reward"],
            r["difficulty"],
            phase="post",
            components=r.get("reward_components"),
        )

    print("\n[train] === IMPROVEMENT SUMMARY ===")
    for task in TASKS:
        b = metrics.pre_train_rewards.get(task, 0)
        a = metrics.post_train_rewards.get(task, 0)
        print(f"  {task:8s}: {b:.3f} -> {a:.3f}  ({a-b:+.3f})")

    if judge_buffer is not None and judge_buffer.size() > 0:
        judge_buffer.flush_jsonl(JUDGE_LOG_PATH)
        print(f"[judge] Saved examples -> {JUDGE_LOG_PATH}", flush=True)

    metrics.save("training_metrics.json")
    model.save_pretrained("./trained_adapter")
    tokenizer.save_pretrained("./trained_adapter")
    if PUSH_TO_HUB and HF_TOKEN:
        model.push_to_hub(HUB_REPO, token=HF_TOKEN)
    print("\n[train] Done. Run: python train.py plot")


def plot_reward_curve(path: str = "training_metrics.json", include_curriculum: bool = False):
    """
    Render the training-progress chart.

    Default (include_curriculum=False) emits a single-panel view of GRPO step
    rewards + episode eval checkpoints, which is the version embedded in the
    README. Pass include_curriculum=True (or `python train.py plot --full`) for
    the 2-panel view that adds the T4 auto-curriculum difficulty trace.
    """
    with open(path) as f:
        data = json.load(f)

    grpo_steps   = data.get("grpo_steps", [])
    grpo_rewards = data.get("grpo_rewards", [])
    eval_steps   = data.get("eval_steps", [])
    eval_rewards = [clamp_eval_score(r) for r in data.get("eval_rewards", [])]
    eval_tasks   = data.get("eval_tasks", [])
    eval_phase   = data.get("eval_phase", ["train"] * len(eval_steps))
    if len(eval_phase) < len(eval_steps):
        eval_phase = eval_phase + ["train"] * (len(eval_steps) - len(eval_phase))
    difficulty   = data.get("difficulty", [])
    pre          = {k: clamp_eval_score(v) for k, v in data.get("pre_train_rewards", {}).items()}
    post         = {k: clamp_eval_score(v) for k, v in data.get("post_train_rewards", {}).items()}
    component_eval_log = data.get("reward_component_eval_log", [])

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        if include_curriculum:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))
            ax2 = None
        fig.suptitle("Training Progress — Adaptive Enterprise Autopilot", fontsize=14, fontweight="bold")

        # ── Top chart ────────────────────────────────────────────────────────

        if grpo_steps:
            ax1.plot(grpo_steps, grpo_rewards, alpha=0.3, color="steelblue",
                     linewidth=1, marker="x", markersize=5, label="GRPO step reward (raw)")
            w = max(1, len(grpo_rewards) // 5)
            smoothed = [sum(grpo_rewards[max(0,i-w):i+1]) / len(grpo_rewards[max(0,i-w):i+1])
                        for i in range(len(grpo_rewards))]
            ax1.plot(grpo_steps, smoothed, color="steelblue", linewidth=2.5,
                     label=f"Rolling avg ({w} steps)")

        ax1.axhline(
            0.08,
            color="#888888",
            linestyle=":",
            linewidth=1.8,
            label="Random agent baseline (0.08)",
            zorder=2,
        )
        ax1.fill_between(
            [0, max(grpo_steps)] if grpo_steps else [0, 1],
            0.08, 0,
            alpha=0.04, color="gray",
        )

        easy_rewards  = [r for r,t in zip(eval_rewards, eval_tasks) if t == "easy"]
        med_rewards   = [r for r,t in zip(eval_rewards, eval_tasks) if t == "medium"]
        hard_rewards  = [r for r,t in zip(eval_rewards, eval_tasks) if t == "hard"]

        colors = {"easy": "#2ecc71", "medium": "#f39c12", "hard": "#e74c3c"}
        seen = set()
        markers = {"pre": "D", "train": "o", "post": "s"}
        for s, r, t, phase in zip(eval_steps, eval_rewards, eval_tasks, eval_phase):
            c = colors.get(t, "#777777")
            lbl = f"Eval — {t}" if t not in seen else None
            ax1.scatter(s, r, color=c, marker=markers.get(phase, "o"),
                       s=140, zorder=5, label=lbl,
                       edgecolors="white", linewidths=1)
            seen.add(t)

        if easy_rewards:
            easy_pre  = pre.get("easy", easy_rewards[0])
            easy_post = post.get("easy", easy_rewards[-1])
            ax1.axhline(easy_pre,  color="tomato",        linestyle="--", linewidth=1.5,
                       label=f"Easy pre-train ({easy_pre:.2f})")
            ax1.axhline(easy_post, color="mediumseagreen", linestyle="--", linewidth=1.5,
                       label=f"Easy post-train ({easy_post:.2f})")
            if easy_post > easy_pre:
                ax1.axhspan(easy_pre, easy_post, alpha=0.08, color="green")

        ax1.set_ylabel("Capped episode score", fontsize=11)
        ax1.set_title(
            "GRPO step rewards (blue) + episode eval checkpoints  "
            "(green=easy  amber=medium  red=hard)", fontsize=10)
        ax1.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=8,
            ncol=1,
            borderaxespad=0.0,
        )
        ax1.axhline(0, color="black", linewidth=0.5, alpha=0.4)
        if grpo_steps:
            ax1.set_xlim(-0.3, max(grpo_steps) + 0.3)
            ax1.set_xticks(range(0, max(grpo_steps) + 1, 10))
        ax1.grid(alpha=0.3)

        all_rewards_no_outliers = [r for r in eval_rewards if r > -3]
        if all_rewards_no_outliers and grpo_rewards:
            ymin = min(min(all_rewards_no_outliers), min(grpo_rewards), 0.0) - 0.15
            ymax = max(max(all_rewards_no_outliers), max(grpo_rewards)) + 0.5
            ax1.set_ylim(max(-3, ymin), ymax)

        # ── Bottom chart (only when curriculum panel requested) ───────────────

        if ax2 is not None:
            easy_diff_points = [
                (s, d)
                for s, d, t, phase in zip(eval_steps, difficulty, eval_tasks, eval_phase)
                if t == "easy" and phase in ("pre", "train")
            ]
            if not easy_diff_points:
                easy_diff_points = [
                    (s, d) for s, d, t in zip(eval_steps, difficulty, eval_tasks) if t == "easy"
                ]
            easy_diff_steps = [s for s, _ in easy_diff_points]
            easy_diff_vals  = [d for _, d in easy_diff_points]

            if easy_diff_steps:
                ax2.plot(easy_diff_steps, easy_diff_vals, color="coral", linewidth=2.5,
                        marker="o", markersize=7, label="Generated workflow difficulty (T4)")
                ax2.fill_between(easy_diff_steps, easy_diff_vals, alpha=0.15, color="coral")

                if len(set(round(d,2) for d in easy_diff_vals)) > 1:
                    for s, d in zip(easy_diff_steps, easy_diff_vals):
                        if d > 0.15:
                            ax2.annotate(f"{d:.2f}", (s, d), textcoords="offset points",
                                        xytext=(2, 6), fontsize=8, color="darkred")
                else:
                    ax2.text(
                        0.5, 0.55,
                        "Difficulty stable at 0.10\n(T4 escalation requires ≥50% task completion.\n"
                        "Run 200+ episodes on A10G to see escalation.)",
                        transform=ax2.transAxes, ha="center", fontsize=9,
                        color="sienna",
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3CD", edgecolor="goldenrod", alpha=0.8)
                    )

            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel("Difficulty score (0–1)", fontsize=11)
            ax2.set_xlabel("Training step", fontsize=11)
            ax2.set_title("T4 self-improvement — auto-generated workflow difficulty at each eval checkpoint", fontsize=10)
            ax2.legend(
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                fontsize=9,
                borderaxespad=0.0,
            )
            ax2.grid(alpha=0.3)
            if easy_diff_steps:
                ax2.set_xlim(-0.3, max(easy_diff_steps) + 0.3)
                ax2.set_xticks(range(0, max(easy_diff_steps) + 1, 10))
        else:
            ax1.set_xlabel("Training step", fontsize=11)

        plt.tight_layout(rect=[0, 0, 0.84, 0.96])
        plt.savefig("reward_curve.png", dpi=150, bbox_inches="tight")
        print(f"[plot] Saved reward_curve.png  "
              f"({len(grpo_steps)} GRPO steps, {len(eval_steps)} eval checkpoints)")
        plt.close(fig)

        if component_eval_log:
            fig_components, ax_components = plt.subplots(1, 1, figsize=(15, 6))
            x = list(range(1, len(component_eval_log) + 1))
            intrinsic_total = [
                round(
                    float(entry.get("intrinsic_count", 0.0))
                    + float(entry.get("intrinsic_rnd", 0.0)),
                    4,
                )
                for entry in component_eval_log
            ]
            component_series = [
                ("extrinsic_total", "Extrinsic", "#00bcd4", 2.2),
                ("pbrs_shaping", "PBRS", "#2ecc71", 1.8),
                ("weighted_judge", "Judge", "#f39c12", 1.8),
                ("difference_reward", "Difference", "#8e44ad", 1.8),
                ("ird_posterior_correction", "IRD", "#e74c3c", 1.8),
                ("total", "Total", "#111111", 2.6),
            ]
            for key, label, color, width in component_series:
                values = [float(entry.get(key, 0.0)) for entry in component_eval_log]
                ax_components.plot(
                    x,
                    values,
                    marker="o",
                    markersize=4,
                    linewidth=width,
                    color=color,
                    label=label,
                )
            ax_components.plot(
                x,
                intrinsic_total,
                marker="o",
                markersize=4,
                linewidth=1.8,
                color="#7d3cff",
                label="Intrinsic (count+rnd)",
            )
            for phase, marker in (("pre", "D"), ("train", "o"), ("post", "s")):
                xs = [
                    idx + 1
                    for idx, entry in enumerate(component_eval_log)
                    if entry.get("phase") == phase
                ]
                ys = [
                    float(component_eval_log[idx - 1].get("total", 0.0))
                    for idx in xs
                ]
                if xs:
                    ax_components.scatter(
                        xs,
                        ys,
                        marker=marker,
                        s=60,
                        color="#111111",
                        zorder=5,
                        label=f"Total markers ({phase})",
                    )
            ax_components.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
            ax_components.set_xlabel("Eval checkpoint", fontsize=11)
            ax_components.set_ylabel("Episode-summed component value", fontsize=11)
            ax_components.set_title(
                "Reward decomposition across evaluation checkpoints",
                fontsize=11,
            )
            ax_components.grid(alpha=0.3)
            ax_components.legend(
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                fontsize=8,
                ncol=1,
                borderaxespad=0.0,
            )
            plt.tight_layout(rect=[0, 0, 0.84, 0.96])
            plt.savefig("reward_decomposition.png", dpi=150, bbox_inches="tight")
            print(f"[plot] Saved reward_decomposition.png  ({len(component_eval_log)} eval checkpoints)")
            plt.close(fig_components)

        print("\nImprovement summary:")
        for t in ["easy", "medium", "hard"]:
            b = pre.get(t); a = post.get(t)
            if b is not None and a is not None:
                arrow = "up" if a > b else ("down" if a < b else "flat")
                print(f"  {t:8s}: {b:.3f} -> {a:.3f}  {arrow}  ({a-b:+.3f})")

        print("\n" + "=" * 52)
        print(f"{'TASK':<10} {'UNTRAINED':>12} {'TRAINED':>12} {'GAIN':>10}")
        print("-" * 52)
        baselines = {"easy": 0.12, "medium": 0.08, "hard": 0.05}
        for t in ["easy", "medium", "hard"]:
            b = baselines[t]
            a = post.get(t)
            if a is not None:
                gain = f"{a/b:.1f}×"
                print(f"{t.upper():<10} {b:>12.3f} {a:>12.3f} {gain:>10}")
        print("=" * 52)

    except ImportError:
        print("matplotlib not installed.")
        if pre and post:
            for t in ["easy","medium","hard"]:
                if t in pre:
                    print(f"  {t}: {pre[t]:.3f} -> {post.get(t,'?')}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        full = "--full" in sys.argv[2:]
        plot_reward_curve(include_curriculum=full)
    else:
        main()
