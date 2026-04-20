"""
Baseline Inference Script — Adaptive Enterprise Autopilot
==========================================================
MANDATORY VARIABLES:
    HF_TOKEN          Your Hugging Face / API key
    API_BASE_URL      LLM endpoint (default: HF router)
    MODEL_NAME        Model identifier
    ENVIRONMENT_URL   Running environment URL

STDOUT FORMAT:
    [START] task=<task> env=adaptive_enterprise_autopilot model=<model>
    [STEP]  step=<n> tool=<tool> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>

Run:
    HF_TOKEN=hf_xxx python inference.py
"""

from __future__ import annotations
import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
ENVIRONMENT_URL: str = os.getenv("ENVIRONMENT_URL", "http://localhost:7860")

TASKS = ["easy", "medium", "hard"]
MAX_STEPS_PER_EPISODE = 30
TEMPERATURE = 0.2
MAX_TOKENS = 1024
SUCCESS_SCORE_THRESHOLD = 0.5
BENCHMARK = "adaptive_enterprise_autopilot"

VALID_TOOLS = [
    "jira_create_ticket", "jira_update_ticket", "jira_assign_ticket",
    "slack_send_message", "slack_create_channel",
    "email_send", "hr_create_user", "hr_update_user",
    "calendar_create_event", "done",
]

# ── Logging ───────────────────────────────────────────────────────────────────

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)

def log_step(step: int, tool: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} tool={tool} "
        f"reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── System prompt ─────────────────────────────────────────────────────────────

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

=== AVAILABLE TOOLS ===
jira_create_ticket    params: summary (required), issue_type (required), priority, description, project, labels
jira_update_ticket    params: ticket_id (required), field (required), value (required)
jira_assign_ticket    params: ticket_id (required), assignee (required)
slack_send_message    params: channel (required), message (required), mention_user
slack_create_channel  params: name (required), members, purpose
email_send            params: to (required), subject (required), body (required)
hr_create_user        params: name (required), role (required), department (required), start_date
hr_update_user        params: user_id (required), field (required), value (required)
calendar_create_event params: title (required), attendees (required), date, duration_minutes, description
done                  params: {} — call this ONLY when ALL tasks are complete

=== CRITICAL RULES ===
1. ALWAYS respect task dependencies. A task with dependencies listed must NOT be called before those dependencies are complete.
2. If a business rule is listed for a task, follow it strictly (e.g. HR must exist before Jira accounts).
3. Check available_task_ids — only work on tasks listed there. Tasks in pending_task_ids are BLOCKED.
4. If a tool call fails (success=false in tool_results), retry it with corrected or same params.
5. Use ticket_ids / user_ids / channel_ids returned in tool_results for subsequent calls.
6. Call "done" ONLY when completed_task_ids contains ALL task IDs.
7. Include a clear one-sentence reasoning explaining which task you are completing.

=== DEPENDENCY STRATEGY ===
- Look at available_task_ids first — these are the tasks you CAN do right now.
- Multiple tasks may be available simultaneously (parallel tracks) — pick the highest-points one.
- Never call a tool for a task still in pending_task_ids.

=== PARAM TIPS ===
- For jira_create_ticket: issue_type is usually "Bug", "Story", or "Epic"
- For slack_send_message: channel can be "#general", "#incidents", "#launch" etc.
- For email_send: "to" is an email address e.g. "team@company.com"
- For hr_create_user: department e.g. "Engineering", "Legal", "HR"
- Use descriptive summaries and messages — quality matters for scoring
""").strip()


# ── LLM call ──────────────────────────────────────────────────────────────────

def call_llm(client: OpenAI, obs: dict, step_feedback: str) -> dict:
    """Call the LLM and parse its JSON tool decision."""
    tasks_summary = "\n".join(
        f"  [{t['task_id']}] {t['name']}: {t['description']} "
        f"(tool={t['required_tool']}, deps={t['dependencies']}, "
        f"rule={t.get('business_rule') or 'none'})"
        for t in obs.get("tasks", [])
    )
    completed = obs.get("completed_task_ids", [])
    available = obs.get("available_task_ids", [])
    pending = obs.get("pending_task_ids", [])
    tool_hist = obs.get("tool_results", [])[-3:]  # last 3 results

    tool_results_str = ""
    if tool_hist:
        tool_results_str = "\nLAST TOOL RESULTS:\n" + "\n".join(
            f"  {r.get('tool')}: success={r.get('result', {}).get('success', r.get('success'))} "
            f"result={json.dumps(r.get('result', {}).get('result', r.get('result', {})))}"
            for r in tool_hist
        )

    user_msg = (
        f"WORKFLOW: {obs.get('workflow_name')}\n"
        f"DESCRIPTION: {obs.get('workflow_description', '')[:300]}\n\n"
        f"ALL TASKS:\n{tasks_summary}\n\n"
        f"COMPLETED: {completed}\n"
        f"AVAILABLE NOW (work on these): {available}\n"
        f"PENDING (blocked by deps): {pending}\n"
        f"{tool_results_str}\n\n"
        f"LAST FEEDBACK: {step_feedback}"
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        # Fallback: pick first available task's tool
        available_ids = obs.get("available_task_ids", [])
        if available_ids:
            task_map = {t["task_id"]: t for t in obs.get("tasks", [])}
            t = task_map.get(available_ids[0], {})
            return {
                "tool": t.get("required_tool", "done"),
                "params": {"summary": "Fallback action", "issue_type": "Task"},
                "reasoning": f"Fallback: completing {t.get('name', 'unknown')}",
            }
        return {"tool": "done", "params": {}, "reasoning": "Fallback: no tasks available"}


# ── Run one episode ───────────────────────────────────────────────────────────

def run_task(client: OpenAI, task: str) -> dict:
    import urllib.request

    base_url = ENVIRONMENT_URL

    def http_post(path: str, payload: dict, query: str = "") -> dict:
        url = f"{base_url}{path}{query}"
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    log_start(task=task, model=MODEL_NAME)
    rewards: List[float] = []
    steps_taken = 0

    try:
        obs_data = http_post("/reset", {}, f"?task={task}")
    except Exception as e:
        print(f"[ERROR] Reset failed: {e}", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return {"steps": 0, "rewards": [], "score": 0.0, "success": False}

    done = obs_data.get("done", False)
    step_feedback = obs_data.get("step_feedback", "")
    step = 0

    while not done and step < MAX_STEPS_PER_EPISODE:
        step += 1
        steps_taken = step

        decision = call_llm(client, obs_data, step_feedback)
        tool = decision.get("tool", "done")
        if tool not in VALID_TOOLS:
            tool = "done"

        action_payload = {
            "tool": tool,
            "params": decision.get("params", {}),
            "reasoning": decision.get("reasoning", ""),
        }

        error_msg = None
        try:
            result = http_post("/step", action_payload)
            reward = float(result.get("reward", 0.0))
            done = result.get("done", False)
            obs_data = result.get("observation", obs_data)
            step_feedback = obs_data.get("step_feedback", "")
        except Exception as e:
            reward = 0.0
            done = True
            error_msg = str(e)

        rewards.append(reward)
        log_step(step=step, tool=tool, reward=reward, done=done, error=error_msg)

    score = sum(rewards) / max(len(rewards), 1)
    score = min(max(score, 0.0), 2.0)   # episode bonus can push above 1.0
    success = score >= SUCCESS_SCORE_THRESHOLD

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {"steps": steps_taken, "rewards": rewards, "score": score, "success": success}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not API_KEY:
        print("[ERROR] HF_TOKEN or API_KEY not set.", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    all_scores = []

    for task in TASKS:
        result = run_task(client, task)
        all_scores.append(result["score"])

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\n[SUMMARY] overall_score={overall:.3f} tasks={','.join(TASKS)}", flush=True)


if __name__ == "__main__":
    main()
