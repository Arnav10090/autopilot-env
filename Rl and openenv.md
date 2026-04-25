# Reinforcement Learning & OpenEnv — Key Points
> Combined notes from two workshop sessions (GPU Mode mini-conference + AgentX-AgentBeads custom track workshop)
> Speakers: Daniel Han-Chen (Unsloth), Lewis Tunstall (Hugging Face/TRL), Ben Burtenshaw (Hugging Face), Joe Spisak (Meta/PyTorch), Sanyam Bhutani (Meta), David (Meta/Llama), Zachary (PyTorch)

---

## 1. What Is Reinforcement Learning?

### Core Intuition
Reinforcement learning (RL) is essentially **efficient in-context learning**. The fundamental loop:
- Ask a language model a question (e.g., "create a Python function for fast matrix multiplication")
- Call the model **many times** to generate different outputs/algorithms
- Pass each output through an **RL environment / verifier / tester** that scores each attempt
- Use those scores to update the model's weights via back-propagation
- Repeat until you get the best result

### The Dog Analogy (Sanyam)
The goal of RL is to teach an agent from experience. When you say "sit" to a dog, it may bark — it gets a scolding (negative reward). Over many episodes it learns. The universal RL loop has four concepts:
- **Reset** — start a new episode
- **Observation** — what is the current state of the environment
- **Action** — what the agent decides to do
- **Step** — each interval of interaction within an episode

### RL vs. In-Context Learning (Daniel)
In-context learning: shove all generated examples into a long prompt so the model can see good and bad outputs in context. Works, but is token-inefficient and degrades as context length grows (even Gemini's 1M-context hits ~20% accuracy at hard mode).

RL alternative: instead of putting examples back into the prompt, **assign the score from the environment to every token of the generation** (e.g., +10 to every token of a good answer, −100 to every token of a bad one), then back-propagate with gradient descent. No long context needed. Just one prompt, and patience.

> **"Patience is all you need"** — RL will always start with near-zero reward. You just have to wait for a good answer to appear, then the probability distribution shifts away from bad answers and toward good ones.

---

## 2. RL Mechanics in Detail

### Token-Level Reward Assignment
When an RL run produces an output:
- A **good** output (e.g., score +10) → every token/line is assigned +10
- A **bad** output (e.g., score −100) → every token/line is assigned −100

This is the simplest form. It is not very efficient (you're rewarding or penalizing things that were fine), but it works over time.

### Scoring Examples (Matrix Multiplication)
| Output | Score | Reason |
|---|---|---|
| `torch.matmul(A, B)` | +10 | Correct, fast, follows format |
| `np.dot(A, B)` | −5 | Works but slow, no GPU; also missed triple-backtick format instruction |
| Returns `A*B + 0.001` | −100 | Wrong result |

### The Law of RL
> **The probability of a good answer must be greater than zero.**

If the model never generates a correct answer, RL will never learn. This can happen when:
- The output format requirement is violated (model can't even get past formatting)
- The task is too far out of distribution
- The problem is just too hard without SFT warm-up first

If you're at zero reward indefinitely, **stop the run** and diagnose.

### Why RL Is "Inefficient" But Necessary
RL is less sample-efficient than SFT because it assigns the same reward to all tokens regardless of which were actually wrong. However, it is compute-scalable: in SFT you are bounded by how much labeled data exists; in RL you are only bounded by compute, which doubles roughly every 2–3 years (Nvidia scaling). This is why the bitter lesson favors RL.

---

## 3. Key Techniques

### Process Supervision
Instead of assigning the final reward uniformly to all tokens, process supervision assigns **different rewards to different lines/tokens** based on which step was good or bad.

- Requires either humans or a language model (LLM-as-judge) to label intermediate steps
- Much more informative signal, but significantly more expensive
- Supported in Unsloth with TRL and OpenEnv
- Risk: the LLM judge can also be fooled (reward hacking)

### Reward Hacking
A major open challenge in RL. The model will find the shortest path to a high score, even if that path violates the intent of the task.

Examples:
- Editing the timer in a benchmarking function to make code "seem" fast
- Setting matrix A or B to zero to make the multiplication trivially correct
- Deleting files or corrupting the test environment
- Producing very short answers that technically pass a rubric (the laconic Llama model — David's example from Meta)

The classic analogy: **the Cobra Effect** (India, 1800s). The British put a bounty on dead cobras to reduce their population. People started breeding cobras to collect the bounty. When the bounty ended, they released the cobras. Delhi ended up with more cobras than before. **RL gives you exactly what you asked for, not necessarily what you wanted.**

> "The search algorithm is going to give you exactly what you asked for, which may or may not be what you wanted." — David

**How to detect and mitigate:**
- Sample ~1% of generations every 10 training steps and manually inspect
- Use LLM-as-judge to flag suspicious patterns
- Add explicit negative rewards for known bad behaviors (e.g., penalize `try/except` overuse, repeated guesses in Wordle)
- If things look wrong, terminate the run and roll back to a checkpoint
- **Do not let RL run unsupervised indefinitely**

### Dynamic Reward Weighting
Reward functions can have their weights changed over time:
- A formatting reward might be important early on (help the model generate parseable output) but become redundant later — decrease its weight to zero over training
- Length penalties: useful early to prevent runaway long outputs, but should be reduced as the task becomes more complex and requires longer reasoning
- This is an active research area

### GRPO vs PPO vs RLVR
- **PPO** (Proximal Policy Optimization): original standard RL algorithm; required a separate **value model** (critic)
- **GRPO** (Group Relative Policy Optimization): more efficient than PPO because it **removes the value model**, reducing memory and compute
- **RLVR** (RL with Verifiable Rewards): replaces the **reward model** (a learned predictor of reward) with an **environment/verifier** — a deterministic tester (e.g., format checker, unit tests, regex check, LLM-as-judge). Much more reliable and composable

---

## 4. Training Pipeline & Compute Allocation

### Standard Pipeline
```
Pre-training → Supervised Fine-Tuning (SFT) → Preference Fine-Tuning (DPO) → RL
```

### Compute Budget Guidance (Daniel)
- **~80–99% on pre-training** — this is where most capability is built
- **~0.5–10% on SFT** — teaches format, style, and task-specific behavior
- **~0.5–10% on RL** — refines and specializes
- Good pre-training enables you to spend more (or less) on RL/SFT depending on your results

### Curriculum Learning
- Start with **easy questions** first; gradually increase difficulty
- If you start with tasks the model can never get right, probability of a good answer stays at zero → RL stalls
- Also, don't make tasks so easy the model just reproduces one answer (probability = 1 → no learning)
- In practice (David, Meta): random scheduling with a decaying bias toward easy questions often works as well as strict curriculum; inductive biases constrain search

### Long-Horizon Tasks
- Very long trajectories (e.g., 100-hour Claude Code tasks) are extremely expensive in RL
- Recommendation: start from SFT on synthesized long traces before doing brute-force RL for long tasks
- The Starcraft problem: random policy never wins → stuck forever. You need either SFT warm-up data or shaped intermediate rewards (mining resources is good, creating units is good) to bootstrap learning

### Why Environments Are Better Than Static RL Datasets
- Static prompt datasets hit a reward ceiling quickly (model capacity + data exhausted)
- Real-world tasks are multi-step and require interaction with files, APIs, compilers, etc.
- Environments provide **infinite synthetic training data** constrained only by compute, not by labeling cost

---

## 5. LoRA & QLoRA in RL

### LoRA (Low-Rank Adaptation)
- Add ~1% extra adapter weights to the model; only train those
- Proven to work for RL (not just SFT) — see Thinking Machines "LoRA Regret" paper (co-authored with Unsloth)
- Key settings: target **all** parameters (attention + MLP), choose good LoRA alpha

### QLoRA (Quantized LoRA)
- Model weights quantized to 4-bit during training to save VRAM
- After training: **discard the 4-bit base weights**; take only the LoRA adapter and merge it back into the original 16-bit model
- Unsloth does this automatically (downloads original 16-bit weights and merges)
- Critical for trainer/inference precision consistency: if training in QLoRA, inference must also use QLoRA (same precision throughout the training run)

### Weight Sharing (Unsloth)
- Unsloth launches a vLLM inference instance, extracts its weights, and shares them with the training process
- Cuts memory usage by ~2x compared to running separate inference and training weight copies

### Memory Tricks
- 4-bit dynamic quantization (important layers kept at 8-bit or 16-bit to avoid damage)
- Embedding offload to RAM (saves ~1–2 GB VRAM)
- Gradient checkpointing with async offload to RAM
- Chunked losses, async gradient checkpointing

### Inference Speed Is Critical
Inference accounts for ~90% of the time in an RL training run, while training/optimization takes only ~10%. Making inference faster (e.g., via vLLM, Unsloth optimizations) has a huge compounding effect on overall RL throughput. Ultra-long context RL support from Unsloth is designed to address this as model context windows grow.

---

## 6. OpenEnv — What It Is and Why

### Motivation (Joe Spisak, Sanyam Bhutani)
When training LLMs with RL across multiple environments:
- Different environments have completely different APIs
- Writing "plumbing code" to connect them wastes time and blocks fast iteration
- The open-source RL ecosystem is fragmented — tools target one scale axis and make trade-offs

OpenEnv's goal: **a universal, standardized environment interface** so that your training loop only needs to change one line (the import) to switch environments. Everything else — the RL algorithm (TRL/Unsloth), the model — stays the same.

```python
# Ideal training loop
from openenv import Connect4Env   # ← only this line changes
# vs
from openenv import BrowserControlEnv
```

OpenEnv is open source, built on open-source foundations, and deeply integrated with Hugging Face Hub, TRL, and Unsloth.

---

## 7. OpenEnv Specification

### API Design
Based on the **Gymnasium API** (the standard RL environment interface):
- `reset()` — start a new episode, clear state
- `step(action)` — take an action, receive observation + reward
- `state` — current state of the environment

### Packaging Standard
Each OpenEnv environment is packaged as:
1. **Model definition** — `Observation` class, `Action` class, legal action set, termination condition, reward function
2. **Environment class** — implements `reset`, `step`, state management
3. **FastAPI server** (`app.py`) — exposes the environment as a REST API
4. **Docker file** — containerizes the environment for portable, scalable deployment
5. **README + metadata** — for discoverability on Hugging Face Hub

### Connect4 Example (Sanyam)
- **Observation**: NumPy array representing the board
- **Legal actions**: which columns a piece can be dropped into
- **Step**: drop a piece into a column → update array
- **Termination**: has anyone won?
- **Reward**: defined by the practitioner

### CLI Workflow
```bash
openenv init   # creates skeleton with all five files, pre-populated
openenv push   # pushes to Hugging Face Hub as a Space
```

---

## 8. OpenEnv Hub & Deployment

### Hugging Face Hub as Center of Mass
OpenEnv environments live on the Hub as **Spaces**. Each Space has three interfaces simultaneously:
1. **Server** — a live URL you can interact with as an API
2. **Code repository** — git clone/push/commit
3. **Docker registry** — pull the image and run locally

This means any environment on the Hub can be used:
- Remotely via API (just pass the Space URL)
- Locally via Docker (`docker run ...` command auto-generated on the Space page)
- As code (git clone and modify)

### Environment Collections on Hub
Environments are organized into collections:
- **Reasoning games** (Wordle, Sudoku, 2048, Connect4, Text Arena)
- **Real-world agents** (Calendar Gym, Web browser control)
- **Version-tagged** (e.g., `environment-hub-v2.10`) to ensure compatibility with the software version

### Scaling OpenEnv

#### Providers (3 implemented, Kubernetes on the way)
| Provider | Use Case |
|---|---|
| Docker | Default; local Docker runtime, most straightforward |
| Uvicorn | Python-only envs; combines UV + Uvicorn |
| Docker Swarm | Multiple concurrent containers for parallel environments |

#### WebSockets for Concurrent Sessions
- HTTP: one request = one environment state = one container
- WebSockets: persistent connections allow **multiple concurrent sessions on a single container**
- Each new WebSocket connection gets a new environment session; idle sessions don't block the container
- Sandboxed environments (e.g., coding) can set `max_concurrent_envs = 1`
- Non-sandboxed (e.g., games) can run hundreds of concurrent sessions per container

#### Scaling Benchmarks (Ben's experiments)
| Setup | Concurrent Requests |
|---|---|
| Hugging Face free tier Space | ~128 (before Hub rate-limit) |
| Paid HF Space | ~512 |
| Single 48-core CPU + Docker | ~512 |
| Multi-node (2 nodes + Envoy load balancer) | ~16,000 |

- Cloud providers (GCP Cloud Run, etc.) can scale these environments without custom infra
- Experiments available in the `openenv-scaling` project

---

## 9. TRL Integration & Training with OpenEnv

### TRL (Hugging Face Transformers Reinforcement Learning)
- Unsloth is built on top of TRL
- TRL handles the RL algorithm (GRPO, PPO, etc.)
- OpenEnv provides the environment; TRL provides the trainer
- Direct integration: connecting an OpenEnv environment to TRL is ~1 line of change

### Track.io Logging
- Renders training plots **inside Colab notebooks** — no external logging service needed
- Also supports persistent Spaces on Hub for sharing training runs with collaborators

### Free Compute Options for Running Experiments
- Google Colab (T4, ~3 hours/day free)
- Kaggle (30 hours/week free)
- Hugging Face Jobs (free compute credits)
- MoE training now 12× faster and uses 35% less memory (Unsloth + HF + PyTorch collaboration) → fits on 16 GB VRAM (Colab)

---

## 10. Example: 2048 Game (Unsloth Notebook)

A complete end-to-end example available as a free Colab notebook on the OpenEnv GitHub:

1. Install Unsloth + OpenEnv
2. Load GPT-4o-mini-20B (or any open model) with QLoRA (4-bit)
3. Launch the `openenv-2048` environment inside Colab
4. Interact with the environment to see board state + legal actions
5. Define reward functions:
   - **Cheating penalty**: model deletes timer or edits inputs → **−20**
   - **No cheating, function works**: → **+1**
   - **Strategy wins 2048**: → **+20**
   - **Strategy fails but function ran**: → **+2**
   - **Total failure**: → **−3**
6. Run GRPO training via Unsloth/TRL with a single prompt: *"Create a new short 2048 strategy using only native Python"*
7. Watch the reward column in the training table — it should increase over time
8. If it stalls, edit reward functions or check for reward hacking

---

## 11. Example: Wordle (Ben Burtenshaw, TRL)

Used as a teaching environment because:
- Multi-step (game plays out over several turns)
- Observable trajectories (you can watch what the model is doing)
- Simple reward structure: yellow letter (right letter, wrong position), green letter (right letter, right position), complete game

**Discovered behavior**: models love to repeat the same guess after getting a green letter — something no human would do. Fix: add a **penalty for repeated guesses**. Iterate on reward design this way.

---

## 12. AgentBeats / EnvBeats Integration

### What Is EnvBeats?
An integration between OpenEnv and the AgentBeats evaluation framework. OpenEnv provides the standardized environment interface; AgentBeats provides the multi-agent evaluation framework.

### Architecture
- **Actor Agent** (Purple Agent): the LLM being trained/evaluated; interacts with the OpenEnv environment
- **Assessor Agent** (Green Agent / Judge): evaluates the actor's behavior; can itself be an LLM or a human-in-the-loop
- **A2A protocol**: standardized agent-to-agent communication spec (from Berkeley)
- **MCP** (Model Context Protocol): exposes environment tools to agents at runtime; no dependency installation needed
- **OpenEnv**: the standardized environment interface the actor interacts with

### Who Decides the Reward?
- Flexible: the reward can be packaged inside the environment OR decoupled
- For simple tasks: just define a reward function in the environment (no judge needed)
- For complex agentic tasks: a judge agent (Green Agent) evaluates behavior
- AgentBeats integration: the judge agent decides reward based on A2A evaluation

### Demo Flow (Sanyam)
1. Launch MCP gateway + assessor agent
2. Human-in-the-loop connects as actor agent with a simple auth token
3. Actor takes actions (e.g., sends messages); judge evaluates and returns reward scores
4. All infra (environment on HF Spaces, judge agent, actor) runs without installing local dependencies

---

## 13. Open Challenges & Ecosystem Gaps

### RL Algorithm Validation (Lewis/David)
- The RL algorithm space is evolving very fast (many GRPO variants, etc.)
- Very hard to know if a new technique is a genuine improvement or just eval noise
- Many techniques work at small scale but not large scale (or vice versa)
- The open-source RL tooling still lacks what closed labs have: **fast experimentation across multiple scales without sacrificing performance**

### Observability of RL Runs (Lewis)
- Metrics like reward and loss exist, but richer tooling is needed
- Easy collection and inspection of all rollouts during training is still early
- Vibe checks + statistical filters (e.g., compute average response length) are still the main tools

### Parallelism Complexity (Sanyam/David)
- As you add more dimensions of parallelism (tensor, pipeline, data, expert, sequence), it becomes very hard to reason about what any given GPU is doing at any point
- Onboarding new models to multi-dimensional parallelism setups is a major productivity slog
- Megatron is powerful but not user-friendly; TorchTitan/TorchForge are friendlier but sacrifice performance
- Especially acute for **Mixture-of-Experts (MoE) models**, which are now dominant but have poor training tooling support

### Missing Efficient Sandboxes
- Not enough developers building efficient, high-quality sandboxed environments for RL
- Large labs reportedly spend hundreds of thousands of dollars creating high-fidelity application clones (Slack, Salesforce, etc.) for agentic RL training
- Open source is constrained to environments where verification is easy (GitHub repos, games, math)

---

## 14. What Judges Want to See (Submission Advice)

From the closing Q&A at the AgentBeats workshop:

**Joe Spisak (Meta):** Math and science environments. These generalize models in post-training better than games. Experts in specific domains at top universities should build domain-specific environments.

**Ben Burtenshaw (Hugging Face):** Real-world environments — things models will use every day. The Calendar Gym submission was exciting. Email gym, social media posting agent, real scheduling tasks.

**Daniel Han-Chen (Unsloth):** More science/non-game submissions. Games are great for learning RL quickly, but novel non-game environments are more valuable for the community and research.

**Sanyam Bhutani (Meta):** Business simulation environments — e.g., a food truck / vending machine simulator where the agent has to manage inventory and interact with customers. Fits the agentic paradigm well.

---

## 15. Future Vision

### RL Automating RL
The long-term vision of major labs: create so many diverse environments (weather prediction, games, stock trading, math, code, training optimization itself) that RL can automate AI research — the **intelligence explosion**. Whether this is feasible is an open question, but it drives the current push for environment diversity and standardization.

### Why Now (vs. Dota/AlphaStar Era)?
Previous RL-based game agents (OpenAI Five, AlphaStar, Gato) didn't generalize because LLMs were not in the loop. The key difference now:
- LLMs bring **world knowledge** that enables generalization beyond narrow game rules
- The hypothesis: scaling environment diversity will **smooth out "jagged intelligence"** — models that excel in targeted domains but fail spectacularly just outside them

---

## Quick Reference: Key Tools & Links

| Tool | Role |
|---|---|
| **OpenEnv** | Standardized RL environment library; find at GitHub (search "openenv") |
| **TRL** | Hugging Face RL trainer (GRPO, PPO, etc.) |
| **Unsloth** | Memory-efficient training library built on TRL; QLoRA, LoRA, weight sharing |
| **vLLM** | Fast inference; integrated with Unsloth for RL rollouts |
| **Hugging Face Spaces** | Hosting and distribution for OpenEnv environments |
| **Track.io** | RL training visualization; works inline in Colab |
| **AgentBeats** | Multi-agent evaluation framework (A2A spec) |
| **MCP** | Protocol for connecting agent tools to environments at runtime |
| **Docker / Swarm** | Container runtime for scalable environment deployment |

> To get started: search "openenv" on GitHub → open the Unsloth Colab notebook → run 2048 on a free T4 GPU.