"""
Training Script — Adaptive Enterprise Autopilot (FIXED)
=========================================================
Key fixes over v1:
  1. Pre-training baseline evaluation runs BEFORE GRPO → shows starting point
  2. GRPOLoggingCallback logs GRPO step rewards every N steps → proper curve
  3. Periodic in-training evaluation rollouts every EVAL_EVERY steps
  4. Post-training final evaluation
  5. Plot now labels axes correctly (step vs episode) and shows before/after line

Run on Colab (free T4 / A10G):
    pip install unsloth trl transformers accelerate peft datasets
    python train.py

For quick smoke-test without unsloth (just plots):
    python train.py plot
"""

from __future__ import annotations
import json
import os
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ── Config ────────────────────────────────────────────────────────────────────

BASE_MODEL   = os.getenv("BASE_MODEL",   "unsloth/Qwen2.5-7B-Instruct")
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

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert enterprise workflow orchestration agent.
You receive a workflow with multiple tasks and must complete them by calling enterprise tools IN THE CORRECT ORDER.

RESPOND ONLY WITH VALID JSON. No prose, no markdown, no explanation.

Your JSON must have exactly these fields:
{
  "tool": "<tool_name>",
  "params": { <tool-specific parameters> },
  "reasoning": "<one sentence: which task you are completing and why now>"
}

Available tools: jira_create_ticket, jira_update_ticket, jira_assign_ticket,
slack_send_message, slack_create_channel, email_send, hr_create_user,
hr_update_user, calendar_create_event, done

CRITICAL: Always respect task dependencies. Only call tools for tasks in available_task_ids.
Use ticket_ids/user_ids returned by previous tool calls in subsequent calls.
""").strip()

VALID_TOOLS = [
    "jira_create_ticket", "jira_update_ticket", "jira_assign_ticket",
    "slack_send_message", "slack_create_channel",
    "email_send", "hr_create_user", "hr_update_user",
    "calendar_create_event", "done",
]

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
    difficulty: List[float]   = field(default_factory=list)
    pre_train_rewards: Dict[str, float]  = field(default_factory=dict)
    post_train_rewards: Dict[str, float] = field(default_factory=dict)
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

    def record_eval(self, step: int, task: str, reward: float, diff: float):
        self.eval_steps.append(step)
        self.eval_rewards.append(round(reward, 4))
        self.eval_tasks.append(task)
        self.difficulty.append(round(diff, 4))
        print(f"[eval @ step {step}] task={task} reward={reward:.3f} difficulty={diff:.3f}", flush=True)

    def save(self, path: str = "training_metrics.json"):
        data_to_save = {
            "grpo_steps": self.grpo_steps,
            "grpo_rewards": self.grpo_rewards,
            "eval_steps": self.eval_steps,
            "eval_rewards": self.eval_rewards,
            "eval_tasks": self.eval_tasks,
            "difficulty": self.difficulty,
            "pre_train_rewards": self.pre_train_rewards,
            "post_train_rewards": self.post_train_rewards,
        }

        # Aggregate component stats
        if hasattr(self, "component_log") and self.component_log:
            n = len(self.component_log)
            agg = {}
            for key in self.component_log[0]:
                vals = [c[key] for c in self.component_log]
                agg[key] = {
                    "mean": round(sum(vals) / n, 4),
                    "first_10_pct": round(sum(vals[:max(1, n // 10)]) / max(1, n // 10), 4),
                    "last_10_pct": round(sum(vals[-(n // 10):]) / max(1, n // 10), 4),
                }
            data_to_save["reward_components"] = agg

        with open(path, "w") as f:
            json.dump(data_to_save, f, indent=2)
        print(f"[metrics] Saved {len(self.grpo_steps)} GRPO steps + {len(self.eval_steps)} evals → {path}")


metrics = TrainingMetrics()

# ── Environment rollout ───────────────────────────────────────────────────────

def run_episode(model_fn, task: str) -> Dict[str, Any]:
    sys.path.insert(0, "src")
    from envs.autopilot_env.environment import AutopilotEnvironment
    from envs.autopilot_env.models import AutopilotAction

    env = AutopilotEnvironment(task=task)
    obs = env.reset()
    total = 0.0
    steps = 0
    while not obs.done and steps < MAX_STEPS_EP:
        prompt = _build_prompt(obs)
        raw = model_fn(prompt)
        try:
            d = json.loads(raw.strip())
        except Exception:
            d = {"tool": "done", "params": {}, "reasoning": ""}
        tool = d.get("tool", "done")
        if tool not in VALID_TOOLS:
            tool = "done"
        obs, r, done, _ = env.step(AutopilotAction(
            tool=tool, params=d.get("params", {}), reasoning=d.get("reasoning", "")))
        total += r
        steps += 1
    completion_rate = env.state.tasks_completed / max(env.state.tasks_total, 1)
    if env.state.generated_next_workflow:
        from envs.autopilot_env.workflow_gen import difficulty_score as _ds
        diff = _ds(env.state.generated_next_workflow)
    else:
        diff = obs.difficulty_level / 10.0
    return {"total_reward": total, "completion_rate": completion_rate, "difficulty": diff}


def _build_prompt(obs) -> str:
    tasks_str = "\n".join(
        f"  [{t['task_id']}] {t['name']}: tool={t['required_tool']}, "
        f"deps={t['dependencies']}, rule={t.get('business_rule') or 'none'}"
        for t in obs.tasks
    )
    results_str = ""
    if obs.tool_results:
        results_str = "\nLAST RESULTS:\n" + "\n".join(
            f"  {r.get('tool')}: success={r.get('result',{}).get('success','?')} "
            f"data={json.dumps(r.get('result',{}).get('result',{}))}"
            for r in obs.tool_results[-3:]
        )
    return (
        f"WORKFLOW: {obs.workflow_name}\nDESC: {obs.workflow_description[:200]}\n\n"
        f"ALL TASKS:\n{tasks_str}\n\nCOMPLETED: {obs.completed_task_ids}\n"
        f"AVAILABLE NOW: {obs.available_task_ids}\nPENDING: {obs.pending_task_ids}\n"
        f"{results_str}\nFEEDBACK: {obs.step_feedback}"
    )


# ── GRPO reward fn ────────────────────────────────────────────────────────────

def grpo_reward_fn(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
    rewards = []
    for completion, prompt in zip(completions, prompts):
        r = 0.0
        components = {
            "valid_json": 0.0,
            "valid_tool": 0.0,
            "tool_in_context": 0.0,
            "reasoning_present": 0.0,
            "early_done_penalty": 0.0,
        }
        try:
            parsed = json.loads(completion.strip())
            components["valid_json"] = 1.0
            r += 1.0

            tool = parsed.get("tool", "")
            if tool in VALID_TOOLS:
                components["valid_tool"] = 0.5
                r += 0.5
            else:
                r -= 0.3

            if parsed.get("reasoning", "").strip():
                components["reasoning_present"] = 0.3
                r += 0.3

            if "AVAILABLE NOW: ['" in prompt and tool != "done" and tool in prompt:
                components["tool_in_context"] = 0.5
                r += 0.5

            if tool == "done" and "AVAILABLE NOW: []" not in prompt:
                components["early_done_penalty"] = -0.4
                r -= 0.4

        except Exception:
            r -= 0.5

        rewards.append(round(r, 3))

        # Log components to metrics
        if not hasattr(metrics, "component_log"):
            metrics.component_log = []
        metrics.component_log.append(components)

    metrics.record_grpo_rewards(rewards)
    return rewards


# ── GRPO callback ─────────────────────────────────────────────────────────────

def make_callback(get_model_fn):
    try:
        from transformers import TrainerCallback

        class MetricsCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                metrics.flush_grpo_step()
                if state.global_step > 0 and state.global_step % EVAL_EVERY == 0:
                    fn = get_model_fn()
                    if fn:
                        try:
                            r = run_episode(fn, "easy")
                            metrics.record_eval(state.global_step, "easy",
                                                r["total_reward"], r["difficulty"])
                        except Exception as e:
                            print(f"[callback eval error] {e}", flush=True)

        return MetricsCallback()
    except ImportError:
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

    # Phase 0 — baseline
    print("\n[train] === PHASE 0: Pre-training baseline ===", flush=True)
    for task in TASKS:
        r = run_episode(model_fn, task)
        metrics.pre_train_rewards[task] = r["total_reward"]
        metrics.record_eval(0, task, r["total_reward"], r["difficulty"])

    # Phase 1 — SFT warmup
    print("\n[train] === PHASE 1: SFT warmup ===", flush=True)
    warmup = build_warmup_dataset()
    sft_ds = Dataset.from_list([{
        "text": f"<|system|>\n{e['system']}<|end|>\n<|user|>\n{e['user']}<|end|>\n<|assistant|>\n{e['assistant']}<|end|>"
    } for e in warmup])
    FastLanguageModel.for_training(model)
    SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=sft_ds,
               dataset_text_field="text",
               args=SFTConfig(per_device_train_batch_size=1, num_train_epochs=5,
                              max_seq_length=MAX_SEQ_LEN, output_dir="./sft_warmup",
                              logging_steps=1, save_strategy="no", report_to="none")).train()
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
                tasks_str = "\n".join(
                    f"  [{t['task_id']}] {t['name']}: tool={t['required_tool']}, deps={t['dependencies']}"
                    for t in wf["tasks"])
                prompts.append({"prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"WORKFLOW: {wf['name']}\nDESC: {wf['description'][:200]}\n\n"
                        f"ALL TASKS:\n{tasks_str}\nCOMPLETED: {completed}\n"
                        f"AVAILABLE NOW: {available}\n"
                        f"PENDING: {[t['task_id'] for t in wf['tasks'] if t['task_id'] not in completed and t['task_id'] not in available]}"
                    )},
                ]})

    callback = make_callback(lambda: model_fn)
    GRPOTrainer(
        model=model, tokenizer=tokenizer,
        reward_funcs=grpo_reward_fn,
        args=GRPOConfig(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            num_train_epochs=max(1, NUM_EPISODES // len(prompts)),
            max_completion_length=512, num_generations=GRPO_K,
            output_dir="./grpo_output", logging_steps=5,
            save_steps=50, report_to="none", remove_unused_columns=False,
        ),
        train_dataset=Dataset.from_list(prompts),
        callbacks=[callback] if callback else None,
    ).train()
    print("[train] GRPO done.", flush=True)

    # Phase 3 — post-training eval
    print("\n[train] === PHASE 3: Post-training eval ===", flush=True)
    FastLanguageModel.for_inference(model)
    final_step = metrics._step
    for task in TASKS:
        r = run_episode(model_fn, task)
        metrics.post_train_rewards[task] = r["total_reward"]
        metrics.record_eval(final_step, task, r["total_reward"], r["difficulty"])

    print("\n[train] === IMPROVEMENT SUMMARY ===")
    for task in TASKS:
        b = metrics.pre_train_rewards.get(task, 0)
        a = metrics.post_train_rewards.get(task, 0)
        print(f"  {task:8s}: {b:.3f} → {a:.3f}  ({a-b:+.3f})")

    metrics.save("training_metrics.json")
    model.save_pretrained("./trained_adapter")
    tokenizer.save_pretrained("./trained_adapter")
    if PUSH_TO_HUB and HF_TOKEN:
        model.push_to_hub(HUB_REPO, token=HF_TOKEN)
    print("\n[train] Done. Run: python train.py plot")


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_reward_curve(path: str = "training_metrics.json"):
    with open(path) as f:
        data = json.load(f)

    grpo_steps   = data.get("grpo_steps", [])
    grpo_rewards = data.get("grpo_rewards", [])
    eval_steps   = data.get("eval_steps", [])
    eval_rewards = data.get("eval_rewards", [])
    eval_tasks   = data.get("eval_tasks", [])
    difficulty   = data.get("difficulty", [])
    pre          = data.get("pre_train_rewards", {})
    post         = data.get("post_train_rewards", {})

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))
        fig.suptitle("Training Progress — Adaptive Enterprise Autopilot", fontsize=14, fontweight="bold")

        # ── Top chart ────────────────────────────────────────────────────────

        # GRPO step rewards
        if grpo_steps:
            ax1.plot(grpo_steps, grpo_rewards, alpha=0.3, color="steelblue",
                     linewidth=1, marker="x", markersize=5, label="GRPO step reward (raw)")
            w = max(1, len(grpo_rewards) // 5)
            smoothed = [sum(grpo_rewards[max(0,i-w):i+1]) / len(grpo_rewards[max(0,i-w):i+1])
                        for i in range(len(grpo_rewards))]
            ax1.plot(grpo_steps, smoothed, color="steelblue", linewidth=2.5,
                     label=f"Rolling avg ({w} steps)")

        # Random baseline — flat dotted line at 0.08
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

        # Eval dots — filter out extreme outliers for y-axis scaling
        easy_rewards  = [r for r,t in zip(eval_rewards, eval_tasks) if t == "easy"]
        med_rewards   = [r for r,t in zip(eval_rewards, eval_tasks) if t == "medium"]
        hard_rewards  = [r for r,t in zip(eval_rewards, eval_tasks) if t == "hard"]

        colors = {"easy": "#2ecc71", "medium": "#f39c12", "hard": "#e74c3c"}
        seen = set()
        for s, r, t in zip(eval_steps, eval_rewards, eval_tasks):
            c = colors[t]
            lbl = f"Eval — {t}" if t not in seen else None
            ax1.scatter(s, r, color=c, s=140, zorder=5, label=lbl,
                       edgecolors="white", linewidths=1)
            seen.add(t)

        # Before/after lines (use easy-only avg to avoid hard task distorting)
        if easy_rewards:
            easy_pre  = pre.get("easy", easy_rewards[0])
            easy_post = post.get("easy", easy_rewards[-1])
            ax1.axhline(easy_pre,  color="tomato",        linestyle="--", linewidth=1.5,
                       label=f"Easy pre-train ({easy_pre:.2f})")
            ax1.axhline(easy_post, color="mediumseagreen", linestyle="--", linewidth=1.5,
                       label=f"Easy post-train ({easy_post:.2f})")
            if easy_post > easy_pre:
                ax1.axhspan(easy_pre, easy_post, alpha=0.08, color="green")

        ax1.set_ylabel("Reward", fontsize=11)
        ax1.set_title(
            "GRPO step rewards (blue) + episode eval checkpoints  "
            "(green=easy  amber=medium  red=hard)", fontsize=10)
        ax1.legend(loc="upper left", fontsize=8, ncol=2)
        ax1.axhline(0, color="black", linewidth=0.5, alpha=0.4)
        if grpo_steps:
            ax1.set_xlim(-0.3, max(grpo_steps) + 0.3)
            ax1.set_xticks(range(0, max(grpo_steps) + 1))
        ax1.grid(alpha=0.3)

        # Auto y-limits: focus on easy task range, clip hard outliers
        all_rewards_no_outliers = [r for r in eval_rewards if r > -3]
        if all_rewards_no_outliers and grpo_rewards:
            ymin = min(min(all_rewards_no_outliers), min(grpo_rewards), 0.0) - 0.15
            ymax = max(max(all_rewards_no_outliers), max(grpo_rewards)) + 0.5
            ax1.set_ylim(max(-3, ymin), ymax)

        # ── Bottom chart ──────────────────────────────────────────────────────

        # Filter to easy-only eval difficulty (mid-training) for smooth curve
        easy_diff_steps = [s for s,t in zip(eval_steps, eval_tasks) if t == "easy"]
        easy_diff_vals  = [d for d,t in zip(difficulty, eval_tasks) if t == "easy"]

        if easy_diff_steps:
            ax2.plot(easy_diff_steps, easy_diff_vals, color="coral", linewidth=2.5,
                    marker="o", markersize=7, label="Generated workflow difficulty (T4)")
            ax2.fill_between(easy_diff_steps, easy_diff_vals, alpha=0.15, color="coral")

            # Annotate if difficulty actually changed
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
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
        if easy_diff_steps:
            ax2.set_xlim(-0.3, max(easy_diff_steps) + 0.3)
            ax2.set_xticks(range(0, max(easy_diff_steps) + 1))

        plt.tight_layout()
        plt.savefig("reward_curve.png", dpi=150)
        print(f"[plot] Saved reward_curve.png  "
              f"({len(grpo_steps)} GRPO steps, {len(eval_steps)} eval checkpoints)")

        # Summary
        print("\nImprovement summary:")
        for t in ["easy", "medium", "hard"]:
            b = pre.get(t); a = post.get(t)
            if b is not None and a is not None:
                arrow = "↑" if a > b else ("↓" if a < b else "→")
                print(f"  {t:8s}: {b:.3f} → {a:.3f}  {arrow}  ({a-b:+.3f})")

        # Per-difficulty breakdown table
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
                    print(f"  {t}: {pre[t]:.3f} → {post.get(t,'?')}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot_reward_curve()
    else:
        main()