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
EVAL_EVERY   = int(os.getenv("EVAL_EVERY", "10"))   # eval rollout every N GRPO steps

MAX_SEQ_LEN  = 4096
LORA_RANK    = 16
BATCH_SIZE   = 2
GRAD_ACCUM   = 4
LR           = 2e-5
MAX_STEPS_EP = 30
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
        with open(path, "w") as f:
            json.dump({
                "grpo_steps": self.grpo_steps,
                "grpo_rewards": self.grpo_rewards,
                "eval_steps": self.eval_steps,
                "eval_rewards": self.eval_rewards,
                "eval_tasks": self.eval_tasks,
                "difficulty": self.difficulty,
                "pre_train_rewards": self.pre_train_rewards,
                "post_train_rewards": self.post_train_rewards,
            }, f, indent=2)
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
    return {
        "total_reward": total,
        "completion_rate": env.state.tasks_completed / max(env.state.tasks_total, 1),
        "difficulty": obs.difficulty_level / 10.0,
    }


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
        try:
            parsed = json.loads(completion.strip())
            r += 1.0
            tool = parsed.get("tool", "")
            if tool in VALID_TOOLS:
                r += 0.5
            else:
                r -= 0.3
            if parsed.get("reasoning", "").strip():
                r += 0.3
            if "AVAILABLE NOW: ['" in prompt and tool != "done" and tool in prompt:
                r += 0.5
            if tool == "done" and "AVAILABLE NOW: []" not in prompt:
                r -= 0.4
        except Exception:
            r -= 0.5
        rewards.append(round(r, 3))
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
    return [
        {
            "system": SYSTEM_PROMPT,
            "user": "WORKFLOW: Onboard new intern\nALL TASKS:\n  [T1] Create HR profile: tool=hr_create_user, deps=[]\nCOMPLETED: []\nAVAILABLE NOW: ['T1']\nPENDING: ['T2']",
            "assistant": json.dumps({"tool": "hr_create_user", "params": {"name": "Intern", "role": "Intern", "department": "Engineering"}, "reasoning": "T1 has no deps — first move."}),
        },
        {
            "system": SYSTEM_PROMPT,
            "user": "WORKFLOW: Report bug\nCOMPLETED: ['T1']\nAVAILABLE NOW: ['T2']\nLAST RESULTS:\n  jira_create_ticket: success=True data={\"ticket_id\": \"PROJ-100\"}",
            "assistant": json.dumps({"tool": "jira_assign_ticket", "params": {"ticket_id": "PROJ-100", "assignee": "oncall"}, "reasoning": "T1 done, assigning ticket PROJ-100."}),
        },
        {
            "system": SYSTEM_PROMPT,
            "user": "WORKFLOW: Onboard\nCOMPLETED: ['T1','T2','T3','T4','T5']\nAVAILABLE NOW: []\nPENDING: []",
            "assistant": json.dumps({"tool": "done", "params": {}, "reasoning": "All tasks complete."}),
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
               args=SFTConfig(per_device_train_batch_size=1, num_train_epochs=3,
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

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle("Training Progress — Adaptive Enterprise Autopilot", fontsize=14, fontweight="bold")

        # Top: GRPO rewards
        if grpo_steps:
            ax1.plot(grpo_steps, grpo_rewards, alpha=0.25, color="steelblue", linewidth=1, label="GRPO step reward")
            w = max(1, len(grpo_rewards) // 10)
            smoothed = [sum(grpo_rewards[max(0,i-w):i+1])/len(grpo_rewards[max(0,i-w):i+1]) for i in range(len(grpo_rewards))]
            ax1.plot(grpo_steps, smoothed, color="steelblue", linewidth=2, label=f"Rolling avg ({w} steps)")

        # Eval dots coloured by task
        colors = {"easy": "#2ecc71", "medium": "#f39c12", "hard": "#e74c3c"}
        seen = set()
        for s, r, t in zip(eval_steps, eval_rewards, eval_tasks):
            c = colors.get(t, "grey")
            lbl = f"Eval — {t}" if t not in seen else None
            ax1.scatter(s, r, color=c, s=100, zorder=5, label=lbl, edgecolors="white", linewidths=0.5)
            seen.add(t)

        # Before / after lines
        if pre and post:
            pre_avg  = sum(pre.values()) / len(pre)
            post_avg = sum(post.values()) / len(post)
            ax1.axhline(pre_avg,  color="tomato",        linestyle="--", linewidth=1.5, label=f"Pre-train avg ({pre_avg:.2f})")
            ax1.axhline(post_avg, color="mediumseagreen", linestyle="--", linewidth=1.5, label=f"Post-train avg ({post_avg:.2f})")

        ax1.set_ylabel("Reward", fontsize=11)
        ax1.set_title("GRPO step rewards + evaluation checkpoints (coloured by task difficulty)", fontsize=10)
        ax1.legend(loc="upper left", fontsize=8, ncol=2)
        ax1.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax1.grid(alpha=0.3)

        # Bottom: difficulty
        if eval_steps and difficulty:
            ax2.plot(eval_steps, difficulty, color="coral", linewidth=2, marker="o", markersize=6, label="Curriculum difficulty (T4)")
            ax2.fill_between(eval_steps, difficulty, alpha=0.15, color="coral")
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel("Difficulty (0–1)", fontsize=11)
            ax2.set_xlabel("Training step", fontsize=11)
            ax2.set_title("Auto-generated workflow difficulty — T4 self-improvement loop", fontsize=10)
            ax2.legend(fontsize=9)
            ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("reward_curve.png", dpi=150)
        print(f"[plot] Saved reward_curve.png  ({len(grpo_steps)} GRPO steps, {len(eval_steps)} eval points)")

        if pre and post:
            print("\nImprovement summary:")
            for t in TASKS:
                b = pre.get(t, 0); a = post.get(t, 0)
                print(f"  {t:8s}: {b:.3f} → {a:.3f}  ({a-b:+.3f})")

    except ImportError:
        if pre and post:
            for t in pre:
                print(f"  {t}: {pre[t]:.3f} → {post.get(t,'?')}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot_reward_curve()
    else:
        main()