"""
Training Script — Adaptive Enterprise Autopilot
================================================
Uses GRPO (Group Relative Policy Optimisation) via TRL + Unsloth.
Trains a small LLM to complete enterprise workflows by calling tools
in the correct dependency order.

Run on Colab (free T4) or HuggingFace Spaces with A10G:
    pip install unsloth trl transformers accelerate peft datasets
    python train.py

Environment variables:
    BASE_MODEL        HuggingFace model ID to fine-tune (default: unsloth/Qwen2.5-7B-Instruct)
    HF_TOKEN          Your HuggingFace token (needed to push the trained model)
    PUSH_TO_HUB       Set to "1" to push the trained adapter to HF Hub
    HUB_REPO          HF repo to push to (default: your-user/autopilot-agent)
    NUM_EPISODES      Episodes per training run (default: 200)
    TASKS             Comma-separated list of tasks (default: easy,medium,hard)
    ENV_URL           If set, uses live HTTP environment; otherwise uses in-process env
"""

from __future__ import annotations
import json
import os
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ── Config ────────────────────────────────────────────────────────────────────

BASE_MODEL   = os.getenv("BASE_MODEL",   "unsloth/Qwen2.5-7B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
PUSH_TO_HUB  = os.getenv("PUSH_TO_HUB", "0") == "1"
HUB_REPO     = os.getenv("HUB_REPO",    "your-user/autopilot-agent")
NUM_EPISODES = int(os.getenv("NUM_EPISODES", "200"))
TASKS        = os.getenv("TASKS", "easy,medium,hard").split(",")
ENV_URL      = os.getenv("ENV_URL", "")

MAX_SEQ_LEN  = 4096
LORA_RANK    = 16
BATCH_SIZE   = 2
GRAD_ACCUM   = 4
LR           = 2e-5
MAX_STEPS_EP = 30
GRPO_K       = 4          # Number of sampled completions per prompt

# ── System prompt (matches inference.py) ─────────────────────────────────────

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

# ── Reward curve tracker ──────────────────────────────────────────────────────

@dataclass
class TrainingMetrics:
    episode: int = 0
    rewards: List[float] = None
    difficulty_scores: List[float] = None
    task_completion_rates: List[float] = None

    def __post_init__(self):
        self.rewards = []
        self.difficulty_scores = []
        self.task_completion_rates = []

    def log(self, reward: float, difficulty: float, completion: float):
        self.rewards.append(reward)
        self.difficulty_scores.append(difficulty)
        self.task_completion_rates.append(completion)
        self.episode += 1
        if self.episode % 20 == 0:
            window = self.rewards[-20:]
            print(
                f"[ep {self.episode:4d}] "
                f"avg_reward={sum(window)/len(window):.3f}  "
                f"avg_completion={sum(self.task_completion_rates[-20:])/20:.1%}  "
                f"avg_difficulty={sum(self.difficulty_scores[-20:])/20:.3f}",
                flush=True,
            )

    def save(self, path: str = "training_metrics.json"):
        with open(path, "w") as f:
            json.dump({
                "rewards": self.rewards,
                "difficulty_scores": self.difficulty_scores,
                "task_completion_rates": self.task_completion_rates,
            }, f)
        print(f"[metrics] Saved to {path}", flush=True)


metrics = TrainingMetrics()

# ── In-process environment rollout ───────────────────────────────────────────

def run_episode_inprocess(model_fn, task: str) -> Dict[str, Any]:
    """
    Run one episode using the in-process environment.
    model_fn(prompt: str) -> str  (generates JSON action)
    """
    sys.path.insert(0, "src")
    from envs.autopilot_env.environment import AutopilotEnvironment
    from envs.autopilot_env.models import AutopilotAction

    env = AutopilotEnvironment(task=task)
    obs = env.reset()

    total_reward = 0.0
    steps = 0

    while not obs.done and steps < MAX_STEPS_EP:
        prompt = _build_prompt(obs)
        raw_response = model_fn(prompt)

        try:
            decision = json.loads(raw_response.strip())
        except Exception:
            decision = {"tool": "done", "params": {}, "reasoning": "Parse error fallback"}

        tool = decision.get("tool", "done")
        if tool not in VALID_TOOLS:
            tool = "done"

        action = AutopilotAction(
            tool=tool,
            params=decision.get("params", {}),
            reasoning=decision.get("reasoning", ""),
        )
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

    completion_rate = env.state.tasks_completed / max(env.state.tasks_total, 1)
    difficulty = obs.difficulty_level / 10.0

    metrics.log(total_reward, difficulty, completion_rate)

    return {
        "total_reward": total_reward,
        "completion_rate": completion_rate,
        "steps": steps,
        "difficulty": difficulty,
        "generated_next": env.state.generated_next_workflow is not None,
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
            f"  {r.get('tool')}: success={r.get('result', {}).get('success', '?')} "
            f"data={json.dumps(r.get('result', {}).get('result', {}))}"
            for r in obs.tool_results[-3:]
        )

    return (
        f"WORKFLOW: {obs.workflow_name}\n"
        f"DESC: {obs.workflow_description[:200]}\n\n"
        f"ALL TASKS:\n{tasks_str}\n\n"
        f"COMPLETED: {obs.completed_task_ids}\n"
        f"AVAILABLE NOW: {obs.available_task_ids}\n"
        f"PENDING (blocked): {obs.pending_task_ids}\n"
        f"{results_str}\n\n"
        f"FEEDBACK: {obs.step_feedback}"
    )


# ── GRPO reward function ──────────────────────────────────────────────────────

def grpo_reward_fn(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
    """
    GRPO reward function — called by TRL's GRPOTrainer.
    Takes K completions for the same prompt, returns a reward for each.

    Reward signal:
      +1.0  valid JSON with all required fields
      +0.5  correct tool name
      +0.5  correct tool for the most-available task
      +0.3  reasoning is non-empty
      −0.5  invalid JSON
      −0.3  unknown tool name
    """
    rewards = []
    for completion, prompt in zip(completions, prompts):
        r = 0.0
        try:
            parsed = json.loads(completion.strip())
            r += 1.0  # valid JSON
            tool = parsed.get("tool", "")
            if tool in VALID_TOOLS:
                r += 0.5
            else:
                r -= 0.3
            if parsed.get("reasoning", "").strip():
                r += 0.3
            # Bonus: does the chosen tool match an available task?
            if "AVAILABLE NOW: ['" in prompt:
                avail_section = prompt.split("AVAILABLE NOW: ")[1].split("\n")[0]
                if tool != "done" and tool in prompt:
                    r += 0.5
            # Penalise calling "done" if available tasks still exist
            if tool == "done" and "AVAILABLE NOW: []" not in prompt:
                r -= 0.4
        except Exception:
            r -= 0.5  # invalid JSON
        rewards.append(round(r, 3))
    return rewards


# ── Dataset builder (for supervised warm-up) ─────────────────────────────────

def build_warmup_dataset() -> List[Dict]:
    """
    Build a small supervised dataset of perfect trajectory examples.
    Used for a short SFT warm-up before GRPO to stabilise JSON output.
    """
    examples = [
        {
            "system": SYSTEM_PROMPT,
            "user": (
                "WORKFLOW: Onboard new intern\n"
                "ALL TASKS:\n"
                "  [T1] Create HR profile: tool=hr_create_user, deps=[], rule=none\n"
                "  [T2] Create Jira ticket for setup: tool=jira_create_ticket, deps=[T1], rule=HR profile must exist before creating tool accounts.\n"
                "COMPLETED: []\nAVAILABLE NOW: ['T1']\nPENDING: ['T2']"
            ),
            "assistant": json.dumps({
                "tool": "hr_create_user",
                "params": {"name": "New Intern", "role": "Software Engineering Intern", "department": "Engineering"},
                "reasoning": "T1 has no dependencies and must be completed first to unblock T2."
            }),
        },
        {
            "system": SYSTEM_PROMPT,
            "user": (
                "WORKFLOW: Onboard new intern\n"
                "ALL TASKS:\n"
                "  [T1] Create HR profile: tool=hr_create_user, deps=[], rule=none\n"
                "  [T2] Create Jira ticket for setup: tool=jira_create_ticket, deps=[T1], rule=HR profile must exist.\n"
                "COMPLETED: ['T1']\nAVAILABLE NOW: ['T2', 'T3', 'T4']\nPENDING: ['T5']\n"
                "LAST RESULTS:\n  hr_create_user: success=True data={\"user_id\": \"HR-1000\"}"
            ),
            "assistant": json.dumps({
                "tool": "jira_create_ticket",
                "params": {"summary": "Engineering intern environment setup", "issue_type": "Task", "priority": "medium"},
                "reasoning": "T1 is done so T2's dependency is satisfied — creating Jira ticket now."
            }),
        },
        {
            "system": SYSTEM_PROMPT,
            "user": (
                "WORKFLOW: Report a production bug\n"
                "ALL TASKS:\n"
                "  [T1] Create bug ticket: tool=jira_create_ticket, deps=[], rule=none\n"
                "  [T2] Assign ticket to on-call engineer: tool=jira_assign_ticket, deps=[T1], rule=Ticket must exist.\n"
                "  [T3] Notify team on Slack: tool=slack_send_message, deps=[T1], rule=none\n"
                "COMPLETED: ['T1', 'T2', 'T3']\nAVAILABLE NOW: ['T4']\nPENDING: []"
            ),
            "assistant": json.dumps({
                "tool": "email_send",
                "params": {"to": "customer@enterprise.com", "subject": "Issue acknowledgement", "body": "We are aware of the issue and our team is investigating urgently."},
                "reasoning": "T1, T2, T3 are complete so T4 (email customer) is now available."
            }),
        },
        {
            "system": SYSTEM_PROMPT,
            "user": (
                "WORKFLOW: Product feature launch\n"
                "ALL TASKS: [...13 tasks...]\n"
                "COMPLETED: ['T1','T2','T3','T4','T5','T6','T7','T8','T9']\n"
                "AVAILABLE NOW: []\nPENDING: []"
            ),
            "assistant": json.dumps({
                "tool": "done",
                "params": {},
                "reasoning": "All 9 tasks are complete — signalling done."
            }),
        },
    ]
    return examples


# ── Main training loop ────────────────────────────────────────────────────────

def main():
    print(f"[train] Base model: {BASE_MODEL}", flush=True)
    print(f"[train] Tasks: {TASKS}", flush=True)
    print(f"[train] Episodes: {NUM_EPISODES}", flush=True)

    try:
        from unsloth import FastLanguageModel
        from trl import GRPOConfig, GRPOTrainer
        from datasets import Dataset
        import torch
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Run: pip install unsloth trl transformers accelerate peft datasets")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    print("[train] Loading model with Unsloth...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
        token=HF_TOKEN or None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    print(f"[train] Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)

    # ── SFT warm-up ───────────────────────────────────────────────────────────
    print("[train] Running short SFT warm-up...", flush=True)
    warmup_data = build_warmup_dataset()

    def format_example(ex):
        return {
            "text": (
                f"<|system|>\n{ex['system']}<|end|>\n"
                f"<|user|>\n{ex['user']}<|end|>\n"
                f"<|assistant|>\n{ex['assistant']}<|end|>"
            )
        }

    warmup_dataset = Dataset.from_list([format_example(e) for e in warmup_data])

    from trl import SFTConfig, SFTTrainer
    sft_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=warmup_dataset,
        dataset_text_field="text",
        args=SFTConfig(
            per_device_train_batch_size=1,
            num_train_epochs=3,
            max_seq_length=MAX_SEQ_LEN,
            output_dir="./sft_warmup",
            logging_steps=1,
            save_strategy="no",
            report_to="none",
        ),
    )
    sft_trainer.train()
    print("[train] SFT warm-up complete.", flush=True)

    # ── GRPO training ─────────────────────────────────────────────────────────
    print("[train] Starting GRPO training...", flush=True)

    # Build GRPO prompt dataset — one entry per workflow per task combination
    grpo_prompts = []
    sys.path.insert(0, "src")
    from envs.autopilot_env.workflows import TASK_WORKFLOWS

    for task in TASKS:
        for wf in TASK_WORKFLOWS[task]:
            # Simulate a mid-episode observation for training diversity
            for i in range(min(3, len(wf["tasks"]))):
                completed = [t["task_id"] for t in wf["tasks"][:i]]
                available = [
                    t["task_id"] for t in wf["tasks"]
                    if t["task_id"] not in completed
                    and all(d in completed for d in t.get("dependencies", []))
                ]
                tasks_str = "\n".join(
                    f"  [{t['task_id']}] {t['name']}: tool={t['required_tool']}, "
                    f"deps={t['dependencies']}"
                    for t in wf["tasks"]
                )
                prompt = (
                    f"WORKFLOW: {wf['name']}\n"
                    f"DESC: {wf['description'][:200]}\n\n"
                    f"ALL TASKS:\n{tasks_str}\n\n"
                    f"COMPLETED: {completed}\n"
                    f"AVAILABLE NOW: {available}\n"
                    f"PENDING: {[t['task_id'] for t in wf['tasks'] if t['task_id'] not in completed and t['task_id'] not in available]}"
                )
                grpo_prompts.append({
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                })

    grpo_dataset = Dataset.from_list(grpo_prompts)

    grpo_config = GRPOConfig(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=max(1, NUM_EPISODES // len(grpo_prompts)),
        max_completion_length=512,
        num_generations=GRPO_K,
        output_dir="./grpo_output",
        logging_steps=5,
        save_steps=50,
        report_to="none",
        remove_unused_columns=False,
    )

    grpo_trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=grpo_reward_fn,
        args=grpo_config,
        train_dataset=grpo_dataset,
    )

    grpo_trainer.train()
    print("[train] GRPO training complete.", flush=True)

    # ── Evaluation rollouts ───────────────────────────────────────────────────
    print("[train] Running post-training evaluation rollouts...", flush=True)
    FastLanguageModel.for_inference(model)

    def model_fn(prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with __import__("torch").no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=True,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    eval_results = {}
    for task in TASKS:
        result = run_episode_inprocess(model_fn, task)
        eval_results[task] = result
        print(
            f"[eval] task={task} score={result['total_reward']:.3f} "
            f"completion={result['completion_rate']:.1%}",
            flush=True,
        )

    overall = sum(r["total_reward"] for r in eval_results.values()) / len(eval_results)
    print(f"\n[eval] overall_score={overall:.3f}", flush=True)

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics.save("training_metrics.json")
    with open("eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    # ── Save and optionally push model ────────────────────────────────────────
    model.save_pretrained("./trained_adapter")
    tokenizer.save_pretrained("./trained_adapter")
    print("[train] Adapter saved to ./trained_adapter", flush=True)

    if PUSH_TO_HUB and HF_TOKEN:
        model.push_to_hub(HUB_REPO, token=HF_TOKEN)
        tokenizer.push_to_hub(HUB_REPO, token=HF_TOKEN)
        print(f"[train] Pushed to hub: {HUB_REPO}", flush=True)

    print("\n[train] Done.", flush=True)


# ── Plot reward curve (run after training) ────────────────────────────────────

def plot_reward_curve(metrics_path: str = "training_metrics.json"):
    """
    Load saved metrics and print a text-based reward curve for terminals,
    or save a PNG if matplotlib is available.
    """
    with open(metrics_path) as f:
        data = json.load(f)

    rewards = data["rewards"]
    difficulty = data["difficulty_scores"]
    n = len(rewards)

    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
        window = 10
        smoothed = [
            sum(rewards[max(0, i - window):i + 1]) / len(rewards[max(0, i - window):i + 1])
            for i in range(n)
        ]
        ax1.plot(rewards, alpha=0.3, color="steelblue", label="raw reward")
        ax1.plot(smoothed, color="steelblue", linewidth=2, label=f"rolling avg ({window})")
        ax1.set_ylabel("Episode reward")
        ax1.set_title("Reward curve — Adaptive Enterprise Autopilot")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(difficulty, color="coral", linewidth=1.5)
        ax2.set_ylabel("Workflow difficulty")
        ax2.set_xlabel("Episode")
        ax2.set_title("Auto-generated workflow difficulty (T4 self-improvement)")
        ax2.set_ylim(0, 1)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("reward_curve.png", dpi=150)
        print("[plot] Saved to reward_curve.png")
    except ImportError:
        # ASCII fallback
        print(f"\nReward curve ({n} episodes):")
        step = max(1, n // 40)
        for i in range(0, n, step):
            r = rewards[i]
            bar = "█" * int(max(0, r) * 10)
            print(f"  ep {i:4d}: {r:+.3f} {bar}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot_reward_curve()
    else:
        main()
