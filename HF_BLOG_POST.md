---
title: "Adaptive Enterprise Autopilot: Curing Agent Hallucinations in Long-Horizon Workflows"
thumbnail: "https://huggingface.co/spaces/Arnav100904/adaptive-enterprise-autopilot/resolve/main/ablation_curve.png"
tags:
  - openenv
  - reinforcement-learning
  - agents
  - grpo
  - unsloth
---

# 🤖 Adaptive Enterprise Autopilot: Curing Agent Hallucinations in Long-Horizon Workflows

*Built for the Meta × Scaler OpenEnv Hackathon Grand Finale*

**Quick Links:**
*   🎯 **[Play with the Live Demo (HF Space)](https://huggingface.co/spaces/Arnav100904/adaptive-enterprise-autopilot)**
*   💻 **[GitHub Repository](https://github.com/Arnav10090/autopilot-env)**

---

## 1. The Problem: The Long-Horizon Capability Gap

We are targeting the domain of **complex enterprise workflow automation**. 

Real enterprise work isn't about answering a single question or making a single API call. It involves **10+ interdependent steps** spanning multiple platforms—creating a Jira ticket, waiting for a Slack approval, provisioning an HR account, and scheduling a Calendar onboarding.

**The Gap:** Current LLMs (even state-of-the-art models) fail spectacularly at these long-horizon tasks. They hallucinate tool calls, ignore dependency chains, and get stuck in infinite loops when API calls fail. To solve this, we need environments that can teach agents *how to think ahead*, not just react.

---

## 2. The Environment: What the Agent Sees, Does, and Gets Rewarded For

We built an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant sandbox that simulates real enterprise tasks. But this isn't a static benchmark; it's an **auto-escalating curriculum** driven by the agent's own capability.

*   **What the Agent Sees:** The agent receives a state observation containing a list of `AVAILABLE NOW` tasks, `PENDING` dependencies, and the `LAST RESULTS` from previous API calls.
*   **What the Agent Does:** It must emit strict JSON to call enterprise tools (`jira_create_ticket`, `slack_send_message`, `hr_create_user`, etc.) in the exact correct order dictated by the workflow DAG (Directed Acyclic Graph).
*   **The T4 Curriculum (Self-Improvement):** If the agent succeeds (≥50% completion), the environment mutates the workflow to make the next episode harder—adding cross-dependencies, verification gates, parallel tracks, and simulated API failures.

### What it Gets Rewarded For: The 7-Term Reward Stack
Training agents on long-horizon tasks suffers from extreme sparse rewards. We engineered a dense, mathematically sound **7-term reward stack** to provide feedback at every single step:

| Reward Component | Type | Purpose |
| :--- | :--- | :--- |
| **1. Extrinsic** | Ground Truth | Base reward for completing tasks and obeying rules. |
| **2. PBRS** | Math Guarantee | Potential-Based Reward Shaping. Guarantees policy invariance while rewarding progress toward leaves. |
| **3. Intrinsic Count** | Exploration | Decaying bonus for trying new tool combinations. |
| **4. Intrinsic RND** | Deep Exploration | Random Network Distillation. Rewards the agent for encountering novel environment states. |
| **5. Difference Rewards** | Credit Assignment | Counterfactual evaluation. Did this specific action improve the outcome over a baseline policy? |
| **6. IRD Posterior** | Safety / Alignment | Inverse Reward Design correction. Penalizes the agent if it exploits the proxy reward in a way a human wouldn't intend. |
| **7. Learned Judge** | Heuristic | A Random Forest model trained on human traces to evaluate the "vibe" and safety of a step. |

---

## 3. The Results: What Changed After Training?

We trained `Qwen2.5-7B-Instruct` using **GRPO (Group Relative Policy Optimization)** via TRL and Unsloth.

**Before Training:** The base model suffered from complete format collapse. It would output markdown fences, conversational filler, and hallucinated tool names, resulting in a 0% workflow completion rate and flatlined rewards.
**After Training:** The agent learned to perfectly format strict JSON tool calls, respect complex dependency chains, and consistently solve complex multi-branch workflows.

![Training Curve](https://huggingface.co/spaces/Arnav100904/adaptive-enterprise-autopilot/resolve/main/reward_curve.png)
*(The training curve above shows the steady improvement as the agent masters format compliance and workflow execution across hundreds of GRPO steps.)*

To prove our reward stack isn't just noise, we ran extensive ablations. As the chart below shows, removing PBRS or Intrinsic Motivation drastically slows down learning in long-horizon tasks.

![Ablation Results](https://huggingface.co/spaces/Arnav100904/adaptive-enterprise-autopilot/resolve/main/ablation_curve.png)

*(Note on GRPO Engineering: We fixed early format-collapse by implementing a tiered JSON salvage parser in our reward function, allowing the model to receive partial credit and a smooth gradient toward correct syntax).*

---

## 4. Why Does It Matter?

Who cares about this, and why?

1.  **Enterprise Automation Platforms (Zapier, Make, UiPath):** The next generation of RPA (Robotic Process Automation) won't be drag-and-drop boxes; it will be autonomous agents. Our environment proves that agents can be trained to navigate complex, non-linear business logic reliably.
2.  **AI Startups building "Agentic Workforces":** Anyone deploying AI teammates (like a virtual HR assistant or automated DevOps responder) needs agents that don't just act, but *plan*. Our 7-term reward stack provides a blueprint for how to train these agents without sparse-reward stagnation.
3.  **The Open Source RL Community:** We demonstrate that integrating advanced concepts—like Potential-Based Reward Shaping and auto-escalating curriculums—into a standard Hugging Face/TRL training loop is not only possible but highly effective.

---

## 🎮 Try it Yourself!

We built a gorgeous, interactive frontend where you can watch the agent execute workflows in real-time. You can even **toggle the reward ablations live** in the UI to see how disabling PBRS or Intrinsic Motivation changes the live mathematical decomposition of the reward signal.

**👉 [Launch the Adaptive Enterprise Autopilot Space](https://huggingface.co/spaces/Arnav100904/adaptive-enterprise-autopilot)**

*We are incredibly excited to share this with the OpenEnv community. Let us know what you think in the comments!*
