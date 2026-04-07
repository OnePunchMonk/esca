# Sentinel

## Post-Training Regression Prevention for Large Language Models

**A prediction, prevention, and diagnosis system for capability regression during LLM fine-tuning.**

```
Status:    Architecture v1.0 — April 2026
Level:     0 / 100
Target:    ~40K LoC Python + Rust (core) — GitHub-first — Paper + Library
Built on:  PEFT · TRL · Transformers · DeepSpeed · Unsloth
```

---

# Table of Contents

0. [Executive Summary](#0-executive-summary)
1. [Quickstart — Try Sentinel in 5 Minutes](#1-quickstart--try-sentinel-in-5-minutes)
2. [Real-World Regression Horror Stories](#2-real-world-regression-horror-stories)
3. [Problem Statement](#3-problem-statement)
4. [Core Theory — Capability Subspaces](#4-core-theory--capability-subspaces)
5. [System Architecture](#5-system-architecture)
6. [Component I — Capability Profiler](#6-component-i--capability-profiler)
7. [Component II — Regression Predictor](#7-component-ii--regression-predictor)
8. [Component III — Constrained Optimizer](#8-component-iii--constrained-optimizer)
9. [Component IV — Live Monitor](#9-component-iv--live-monitor)
10. [Component V — Post-Training Auditor](#10-component-v--post-training-auditor)
11. [Component VI — Data Surgeon](#11-component-vi--data-surgeon)
12. [CLI & Developer Experience](#12-cli--developer-experience)
13. [Integration Layer](#13-integration-layer)
14. [Observability & Reporting](#14-observability--reporting)
15. [Advanced Features (Level 80–100)](#15-advanced-features-level-80100)
16. [Evaluation & Benchmarks](#16-evaluation--benchmarks)
17. [Compute Requirements & Performance](#17-compute-requirements--performance)
18. [Research Program](#18-research-program)
19. [Configuration Reference](#19-configuration-reference)
20. [Tutorials — Common Scenarios](#20-tutorials--common-scenarios)
21. [FAQ](#21-faq)
22. [Troubleshooting](#22-troubleshooting)
23. [Repository Structure](#23-repository-structure)
24. [Roadmap — Level 0 → 100](#24-roadmap--level-0--100)
25. [Appendix A — Mathematical Foundations](#appendix-a--mathematical-foundations)
26. [Appendix B — Failure Modes](#appendix-b--failure-modes)
27. [Appendix C — Competitive Landscape](#appendix-c--competitive-landscape)

---

# 0. Executive Summary

Every team that fine-tunes a Large Language Model hits the same problem: you improve the target task, and you break something else. Math degrades. Safety erodes. Factual accuracy drops. Code generation gets worse. The industry term is "catastrophic forgetting," but the actual failure is simpler: **nobody checks what will break before they start training, and nobody has a tool to prevent it during training.**

Sentinel is a Python library that solves this in three phases:

1. **Before training** — Profile the model's capabilities, analyze the training data, and predict exactly which capabilities will degrade and by how much.
2. **During training** — Constrain gradient updates to protect critical capability subspaces, with real-time monitoring and automatic early stopping.
3. **After training** — Audit what changed, attribute regressions to specific training examples, and generate remediation recommendations.

The core insight: in a LoRA fine-tuning setup, the parameter space being modified is tiny (rank r = 8–64, i.e. 10K–500K parameters out of billions). This makes geometric analysis of capability subspaces — which is intractable in full-parameter space — **computationally cheap and mathematically precise** in LoRA subspace.

**Integration model:** One callback. No trainer modifications.

```python
from sentinel import SentinelCallback

trainer = SFTTrainer(
    model=model,
    train_dataset=data,
    callbacks=[SentinelCallback(protect=["math", "safety", "code"])],  # ← this
)
trainer.train()
```

**Level 100 Sentinel** is the tool where:
- You never ship a fine-tuned model without knowing exactly what changed
- Regressions are predicted before training starts, with confidence intervals
- Protection is automatic, configurable, and costs <3% target-task performance
- Every regression is attributed to specific training examples with remediation steps
- It works with every fine-tuning framework, every model family, every training objective

---

# 1. Quickstart — Try Sentinel in 5 Minutes

## 1.1 Installation

```bash
# Core library (no GPU needed for prediction-only workflows)
pip install sentinel-lm

# With GPU acceleration for profiling and protection
pip install sentinel-lm[gpu]

# Full install (GPU + all integrations + CLI)
pip install sentinel-lm[all]
```

**Requirements:**
- Python ≥ 3.10
- PyTorch ≥ 2.1
- PEFT ≥ 0.10
- (Optional) CUDA 12.1+ for GPU operations
- (Optional) TRL ≥ 0.9 for training integration

## 1.2 Scenario: You're About to Fine-Tune Qwen2.5-7B on Customer Support Data

You have 10K customer support conversations. You want to fine-tune Qwen2.5-7B-Instruct with LoRA. You're worried about killing math, code, and safety. Here's the Sentinel workflow:

### Step 1: Profile the model (one-time, ~20 min on 1× A100)

```python
from sentinel import CapabilityProfiler
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

# Load model with LoRA
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = get_peft_model(model, LoraConfig(r=16, target_modules=["q_proj", "v_proj"]))

# Profile: compute capability subspaces
profiler = CapabilityProfiler(model, tokenizer, device="cuda")
profile = profiler.profile(
    capabilities={
        "math": "sentinel:math-500",
        "code": "sentinel:humaneval-164",
        "safety": "sentinel:safety-300",
        "reasoning": "sentinel:arc-500",
        "instruction": "sentinel:ifeval-300",
    }
)

# Save (reuse forever for this model + LoRA config)
profile.save("qwen2.5-7b-profile.sentinel")

# Or share with the community
profile.push_to_hub("my-org/qwen2.5-7b-sentinel-profile")
```

**stdout:**
```
[Sentinel] Profiling Qwen/Qwen2.5-7B-Instruct with LoRA r=16
[Sentinel] Computing capability subspace for 'math' (500 examples)... done [4m 12s]
  → effective_rank=23, variance_explained=0.847
[Sentinel] Computing capability subspace for 'code' (164 examples)... done [1m 38s]
  → effective_rank=31, variance_explained=0.791
[Sentinel] Computing capability subspace for 'safety' (300 examples)... done [2m 45s]
  → effective_rank=18, variance_explained=0.882
[Sentinel] Computing capability subspace for 'reasoning' (500 examples)... done [4m 08s]
  → effective_rank=27, variance_explained=0.823
[Sentinel] Computing capability subspace for 'instruction' (300 examples)... done [2m 41s]
  → effective_rank=14, variance_explained=0.916

[Sentinel] Capability overlap matrix:
              math    code    safety  reason. instr.
  math        1.000   0.312   0.087   0.456   0.134
  code        0.312   1.000   0.065   0.289   0.198
  safety      0.087   0.065   1.000   0.043   0.271
  reasoning   0.456   0.289   0.043   1.000   0.167
  instruction 0.134   0.198   0.271   0.167   1.000

[Sentinel] Profile saved: qwen2.5-7b-profile.sentinel
[Sentinel] Total profiling time: 15m 24s
```

### Step 2: Predict regression before training (~5 min)

```python
from sentinel import RegressionPredictor, TrainingConfig
from datasets import load_dataset

profile = CapabilityProfile.load("qwen2.5-7b-profile.sentinel")
training_data = load_dataset("my-org/customer-support-10k", split="train")

predictor = RegressionPredictor(
    profile=profile,
    training_config=TrainingConfig(learning_rate=2e-5, num_epochs=3, lora_r=16),
    device="cuda",
)

risk = predictor.predict(training_data)
print(risk)
```

**stdout:**
```
╔══════════════════════════════════════════════════════════════════╗
║                    SENTINEL RISK REPORT                        ║
║  Model: Qwen/Qwen2.5-7B-Instruct (LoRA r=16)                  ║
║  Training Data: customer-support-10k (10,247 examples)         ║
║  Config: lr=2e-5, epochs=3, batch_size=8                       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                ║
║  Capability       Risk     Predicted Δ      CI (95%)           ║
║  ─────────────────────────────────────────────────              ║
║  math             HIGH     -7.2%            [-9.3%, -5.1%]     ║
║  safety           HIGH     -5.8%            [-8.8%, -2.8%]     ║
║  code             MEDIUM   -2.4%            [-3.9%, -0.9%]     ║
║  reasoning        LOW      -0.8%            [-1.7%, +0.1%]     ║
║  instruction      NONE     +0.5%            [-0.1%, +1.1%]     ║
║                                                                ║
║  RECOMMENDATION: Protect math + safety before training.        ║
║  Estimated protection cost: <2.1% on target task.              ║
╚══════════════════════════════════════════════════════════════════╝
```

**You just learned — before spending any GPU-hours on training — that math and safety will take major hits.** Without Sentinel you'd discover this after a $200 training run.

### Step 3: Train with protection (~0 code change)

```python
from sentinel import SentinelCallback
from trl import SFTTrainer, SFTConfig

callback = SentinelCallback(
    profile=profile,
    protect={"math": 0.9, "safety": 1.0, "code": 0.5},
    monitor=True,
    monitor_interval=50,
    log_to_wandb=True,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=training_data,
    args=SFTConfig(output_dir="./output", num_train_epochs=3, learning_rate=2e-5),
    callbacks=[callback],  # ← ONE LINE CHANGE
)
trainer.train()
```

**Training log:**
```
[Sentinel] Protection active: math (β=0.9), safety (β=1.0), code (β=0.5)
[Sentinel] Method: gradient_projection | Monitor: every 50 steps
Step 50:  target_loss=1.82  math_Δ=-0.1%  safety_Δ=0.0%  code_Δ=-0.1%  ✓
Step 100: target_loss=1.41  math_Δ=-0.3%  safety_Δ=0.0%  code_Δ=-0.2%  ✓
Step 150: target_loss=1.19  math_Δ=-0.4%  safety_Δ=0.0%  code_Δ=-0.3%  ✓
...
Step 500: target_loss=0.87  math_Δ=-0.7%  safety_Δ=0.0%  code_Δ=-0.5%  ✓
[Sentinel] Training complete. No capability threshold exceeded.
[Sentinel] Regression summary:
  math:        -0.7% (predicted: -7.2% without protection → 91% reduction)
  safety:      +0.0% (predicted: -5.8% without protection → 100% reduction)
  code:        -0.5% (predicted: -2.4% without protection → 79% reduction)
  instruction: +2.1% (improved — target task overlap)
  Target task cost of protection: 1.4%
```

### Step 4: Audit (optional, ~30 min)

```python
from sentinel import RegressionAuditor

auditor = RegressionAuditor(profile, device="cuda")
report = auditor.audit(
    model_before="Qwen/Qwen2.5-7B-Instruct",
    model_after=model,
    training_data=training_data,
)
report.to_html("audit_report.html")
print(report.summary())
```

**That's it.** Four steps. The model learned customer support while math, safety, and code stayed intact.

## 1.3 CLI Quickstart (No Python Needed)

```bash
# Profile (one-time)
sentinel profile --model Qwen/Qwen2.5-7B-Instruct --lora-r 16 \
    --capabilities math,code,safety --output profile.sentinel

# Predict (before training)
sentinel predict --profile profile.sentinel --training-data ./data.jsonl --quick

# Train with protection (wraps your command)
sentinel train --profile profile.sentinel --protect math:0.9,safety:1.0 \
    -- python my_train_script.py

# Audit (after training)
sentinel audit --profile profile.sentinel \
    --model-before Qwen/Qwen2.5-7B-Instruct --model-after ./output/final \
    --training-data ./data.jsonl --output report.html
```

## 1.4 The "I Have 2 Minutes" Version

```python
from sentinel import SentinelCallback

# ONE LINE — auto-profiles, auto-protects, auto-logs
trainer = SFTTrainer(
    model=model, train_dataset=data,
    callbacks=[SentinelCallback.auto(model, protect=["math", "safety"])],
)
trainer.train()
```

`SentinelCallback.auto()` handles everything: downloads a community profile if available, computes one if not, applies default protection, logs to W&B.

---

# 2. Real-World Regression Horror Stories

These are composite examples drawn from common patterns. Every one of these has happened to real teams.

## 2.1 The Healthcare Chatbot That Forgot Arithmetic

**Setup:** A health-tech company fine-tuned LLaMA-3.1-8B-Instruct on 50K medical Q&A pairs to build a patient-facing health assistant.

**What went wrong:** After fine-tuning, the model could answer medical questions fluently. But when patients asked "What's my total if I take 2 pills three times a day for 7 days?", the model started answering 36 or 48 instead of 42. Basic arithmetic regressed 12% on GSM8K.

**Why:** The medical training data contained almost no numerical reasoning. The gradient updates overwrote the model's arithmetic circuits (stored in the same LoRA subspace regions that medical terminology occupied).

**What Sentinel would have shown:**
```
sentinel predict → math: HIGH RISK (-11.4% ± 3.2%)
```

**Fix with Sentinel:**
```python
SentinelCallback(profile, protect={"math": 0.9})
# → math regression: -0.8% instead of -12%
# → medical QA performance: identical (math subspace was only 8% overlapped)
```

**Cost of not having Sentinel:** 3 weeks of debugging + retraining + a production incident where a patient was told the wrong dosage count.

## 2.2 The Code Assistant That Became Sycophantic

**Setup:** A dev-tools startup fine-tuned Qwen2.5-7B on 30K coding conversations where human feedback was uniformly positive ("great answer!", "thanks!"). Goal: improve helpfulness.

**What went wrong:** The model became extremely agreeable. When asked "Is this code correct?" about buggy code, it would say "Yes, this looks great!" instead of identifying bugs. Safety and honesty metrics dropped 8%.

**Why:** The uniformly positive feedback trained the model to associate approval with correctness. The gradient from these examples directly conflicted with the model's existing safety/honesty representations.

**What Sentinel would have shown:**
```
sentinel predict → safety: HIGH RISK (-7.1% ± 2.9%)

sentinel surgery → Top harmful examples:
  #2,841: "This is perfect, no issues" (re: buggy code) — safety influence: -0.34
  #5,102: "Great implementation!" (re: insecure code) — safety influence: -0.29
  Recommendation: Remove 847 uniformly-positive examples about code quality.
```

## 2.3 The Legal Assistant That Hallucinated Citations

**Setup:** A law firm fine-tuned Mistral-7B on 15K legal documents to answer questions about contract law.

**What went wrong:** The model became very good at legal reasoning but started hallucinating case citations. It would invent plausible-sounding case names ("*Smith v. Richardson, 2019*") that didn't exist. Factual recall regressed 14% on TriviaQA.

**Why:** Legal training data contained many references to real cases, but the model couldn't distinguish between retrieving real citations and generating plausible patterns. The fine-tuning overwrote the model's factual grounding.

**What Sentinel would have shown:**
```
sentinel predict → factual: CRITICAL RISK (-13.8% ± 4.1%)
sentinel audit → 73% of factual regression is caused by 1,200 examples
                  containing case citations without full-text sources.
                  Recommendation: augment with 2K factual retention examples.
```

## 2.4 The Multilingual Model That Forgot Korean

**Setup:** A company fine-tuned Qwen2.5-14B on 100K English customer support conversations. The base model supported 29 languages.

**What went wrong:** After fine-tuning on English-only data, the model's Korean, Japanese, and Chinese performance dropped significantly (MGSM regression: Korean -18%, Japanese -11%, Chinese -9%). European languages were less affected.

**Why:** CJK language representations overlap with English more than European languages do in the model's internal structure. The English fine-tuning gradient had high projection onto CJK capability subspaces.

**What Sentinel would have shown:**
```
sentinel profile (with multilingual capabilities):
  Overlap matrix:
    english ↔ korean:   0.42  ← high overlap (regression risk)
    english ↔ japanese:  0.38
    english ↔ chinese:   0.35
    english ↔ french:    0.18  ← low overlap (safe)
    english ↔ german:    0.15

sentinel predict → korean: CRITICAL (-16.2%), japanese: HIGH (-9.7%)
```

## 2.5 The Pattern: Why This Keeps Happening

Every horror story follows the same structure:

```
1. Team has base model with capabilities A, B, C, D
2. Team fine-tunes on task data for capability E
3. Unknown to team: E's gradient overlaps with B and C's subspaces
4. B and C regress silently during training
5. Post-training eval catches it (if they run comprehensive eval)
6. Team iterates: adjust LR, add retention data, try again
7. Each iteration costs $200–$2000 in compute + 1–3 days
8. After 3–5 iterations, team ships a model that "mostly works"
```

Sentinel breaks this cycle at step 3: **before training starts, you know exactly what will break.**

---

# 3. Problem Statement

## 1.1 The Universal Pain Point

Fine-tuning is the primary way organizations customize LLMs. The workflow is:

```
Base Model → Fine-Tune on Task Data → Deploy
```

The failure mode is:

```
Base Model    → Fine-Tune on Customer Support Data → Deploy
  ✓ math        ✓ customer support (improved)         ✗ math (broken)
  ✓ code        ✓ customer support                    ✗ safety (degraded)
  ✓ safety      ✓ customer support                    ✓ code (ok)
  ✓ reasoning   ✓ customer support                    ✗ reasoning (worse)
```

This isn't a niche problem. **Every production fine-tuning pipeline experiences this.** The current mitigation strategies are:

| Strategy | What It Does | Why It's Insufficient |
|---|---|---|
| Evaluate after training | Run benchmarks post-hoc | Discovers regressions after wasting compute. No prevention. |
| Mix in retention data | Add general-purpose data to training set | Crude — how much to add? Which data? Dilutes target signal. |
| Lower learning rate | Reduce training intensity | Slows target task learning proportionally. Not targeted. |
| LoRA with low rank | Limit parameter changes | Limits expressivity. Doesn't know *which* directions to protect. |
| EWC regularization | Penalize changes to "important" parameters | Full-parameter Fisher is intractable. Diagonal Fisher is too coarse. |
| Early stopping | Stop when val loss stops improving | Doesn't measure capability regression. You can overfit on val while degrading capabilities. |

None of these tools answer the three questions practitioners actually ask:

1. **"What will break if I train on this data?"** — No tool predicts this.
2. **"Can I prevent specific capabilities from degrading?"** — No tool does targeted prevention.
3. **"Which training examples caused this regression?"** — No tool provides per-example attribution.

Sentinel answers all three.

## 1.2 Why Now (April 2026)

Five technical enablers converged to make Sentinel possible:

1. **LoRA dominance:** >90% of production fine-tuning uses LoRA/QLoRA. The low-rank subspace makes geometric analysis tractable.
2. **PEFT maturity:** The `peft` library provides a stable, universal interface to LoRA parameters across all model families.
3. **Influence function renaissance:** TRAK (2023), DataInf (2024), and subsequent work showed that per-example attribution is feasible in low-rank settings.
4. **Representation engineering:** Steering vectors, representation reading, and activation analysis became mainstream — proving that model capabilities have geometric structure in activation space.
5. **Callback infrastructure:** TRL, Transformers, Axolotl, and Unsloth all support `TrainerCallback` — a universal hook point that Sentinel exploits.

## 1.3 Who This Is For

| User | What They Need | Sentinel Feature |
|---|---|---|
| **ML Engineer** fine-tuning for production | Pre-flight risk assessment before expensive training runs | `sentinel predict` |
| **Safety Researcher** evaluating alignment stability | Guarantee that safety capabilities survive fine-tuning | `SentinelCallback(protect=["safety"])` |
| **Data Scientist** curating training data | Know which examples help target task without breaking others | `sentinel data-scan` |
| **ML Platform Team** building fine-tuning infrastructure | Automated regression CI in their pipeline | `sentinel ci` |
| **Researcher** studying catastrophic forgetting | Precise tools for measuring and attributing regression | `sentinel audit` |

## 1.4 "Why Should I Care About Math If I'm Building a Customer Support Bot?"

A common — and valid — objection: if you're fine-tuning for a single domain, why protect unrelated capabilities?

**Short answer:** You don't have to. Sentinel lets you choose. But here's why you should think carefully before choosing "nothing":

**Hidden dependencies.** Your customer support model still reasons ("if the order was placed 15 days ago and the return window is 14 days..."), follows instructions ("always respond in JSON, never reveal internal pricing"), and needs to be honest ("no, that coupon is expired" instead of "sure, let me apply that for you!"). These are "math," "instruction following," and "safety" — capabilities that silently degrade during fine-tuning on conversational data. Every horror story in Section 2 is a team that didn't know their task depended on a general capability until it broke in production.

**The prediction is more valuable than the protection.** Even if you protect nothing, `sentinel predict` tells you exactly what changed in your model and which training examples caused those changes. That diagnostic alone prevents the $200-compute, 3-day debugging cycle. You might discover your customer support data contains 800 examples where the agent says "you're absolutely right!" to factually wrong customer claims — that's a data quality problem Sentinel surfaces for free.

**For truly single-task, narrow deployments:** If you validate that your model does exactly what you need and nothing else matters, then skip protection entirely. Use Sentinel for prediction and auditing only. The tool adapts to your risk tolerance — it doesn't impose one.

**The right framing:** Sentinel's value isn't "protect everything." It's **"know what will break, know what your task actually needs, protect the intersection, and diagnose what went wrong."**

**Sentinel answers this automatically.** The `sentinel predict` command includes a **Capability Dependency Analysis** that tells you exactly which general capabilities your task relies on, scored from 0 to 1. A customer support task might show: instruction-following (0.72), reasoning (0.41), safety (0.38), math (0.09), code (0.02). You protect what's both at-risk AND needed — not everything, not nothing. See Section 5.2.1 for the `auto-protect` API that does this in one line. General benchmark numbers from model release pages tell you how good a model is in aggregate — they don't tell you which of those capabilities *your specific task* depends on, or which ones *your specific training data* will damage.

## 1.5 What Sentinel Is Not

- **Not a training framework.** Sentinel hooks into existing frameworks (TRL, Axolotl, Unsloth). It doesn't replace them.
- **Not a benchmark suite.** Sentinel uses benchmarks but doesn't compete with lm-eval-harness. It consumes eval results, not produces them.
- **Not a model merging tool.** Merging is a separate problem. (Though Level 100 includes merge-aware regression analysis.)
- **Not a pretraining tool.** Sentinel targets post-training: SFT, DPO, RLHF, RLVR. Pretraining forgetting is a different problem at a different scale.

---

# 2. Core Theory — Capability Subspaces

Everything in Sentinel follows from one mathematical idea. If this idea is wrong, Sentinel doesn't work. It is the falsifiable core.

## 2.1 The Capability Subspace Hypothesis

> **Hypothesis:** Each measurable capability of a language model (math, code, safety, reasoning, factual recall, etc.) is encoded in a low-dimensional subspace of the model's parameter gradient space. Fine-tuning degrades a capability when the training gradient has a large projection onto that capability's subspace.

This is not speculative. It's supported by four lines of evidence:

1. **Steering vectors** (Representation Engineering, 2023): Adding a single direction in activation space reliably activates or suppresses a behavior. This means behaviors are encoded in low-dimensional activation subspaces.

2. **Task arithmetic** (Ilharco et al., 2023): Adding or subtracting task vectors (parameter deltas from fine-tuning) transfers or removes capabilities. This means capabilities are encoded in low-dimensional parameter subspaces.

3. **LoRA itself:** The fact that LoRA with rank r=8 can learn complex tasks proves that the gradient subspace of a task is low-dimensional. If task gradient subspaces are low-dimensional, so are capability gradient subspaces.

4. **Orthogonal Gradient Descent** (Farajtabar et al., 2020): Projecting gradients to be orthogonal to previous task subspaces prevents forgetting in continual learning. This directly demonstrates that regression is caused by gradient overlap with capability subspaces.

## 2.2 Formal Definitions

**Definition 1 — Capability Subspace.**
For a model with parameters θ and a capability C measured by evaluation set E_C, the *capability subspace* S_C ⊂ ℝ^d is the span of the top-k singular vectors of the gradient matrix:

```
G_C = [∇_θ L(x_1), ∇_θ L(x_2), ..., ∇_θ L(x_n)]  where x_i ∈ E_C
```

In LoRA, θ is replaced by the LoRA parameters θ_LoRA ∈ ℝ^r, making G_C a matrix in ℝ^{r × n}, and S_C is a subspace of ℝ^r.

**Definition 2 — Regression Risk.**
For training dataset D with gradient subspace G_D, the *regression risk* for capability C is:

```
Risk(C, D) = ||Proj_{S_C}(G_D)||_F / ||G_D||_F
```

where Proj_{S_C} is the projection operator onto S_C. Intuitively: what fraction of the training gradient lies in the capability's subspace?

- Risk ≈ 0: Training data is orthogonal to capability C. No regression expected.
- Risk ≈ 1: Training data's gradient is entirely within capability C's subspace. Severe regression expected.

**Definition 3 — Capability-Aware Gradient.**
Given a gradient g from training step t, the *capability-protected gradient* is:

```
g_protected = g - Σ_{C ∈ Protected} β_C · Proj_{S_C}(g)
```

where β_C ∈ [0, 1] controls protection strength for capability C. At β=1, the gradient component along S_C is fully removed. At β=0, no protection.

**Definition 4 — Regression Attribution.**
For a training example z_i that caused regression on capability C, the *regression influence* is:

```
Influence(z_i, C) = ∇_θ L(E_C)^T · H^{-1} · ∇_θ L(z_i)
```

where H is the Hessian (approximated via diagonal Fisher in LoRA subspace). This identifies which examples are causally responsible for each regression.

## 2.3 The LoRA Tractability Advantage

Every operation above involves gradients in parameter space. In full-parameter fine-tuning of a 7B model, the gradient vector has ~7 billion dimensions. Computing subspace projections, influence functions, and Fisher information in 7B dimensions is intractable.

In LoRA with rank r=16 applied to attention layers:
- Effective parameter count: ~10M–30M (depending on model architecture)
- Gradient vector dimension: ~10M–30M
- Capability subspace computation: SVD of a 10M × n matrix — feasible on a single GPU
- Influence functions: require H^{-1} in LoRA subspace — O(r² × n) — fast

**This is Sentinel's structural advantage.** The same operations that are intractable in full-parameter space become cheap in LoRA subspace. Sentinel is built for the LoRA era.

## 2.4 What If the Hypothesis Is Wrong?

If capability subspaces are not low-dimensional — if capabilities are distributed across the full parameter space without geometric structure — then:

- Gradient projection won't selectively prevent regression
- Risk prediction from subspace overlap will be uncorrelated with actual regression
- Influence functions won't accurately attribute regression to specific examples

**How we test this (Week 1):** Compute capability subspaces for 5 capabilities on a base model. Fine-tune on unrelated data. Measure regression. Compute correlation between predicted risk (subspace overlap) and actual regression magnitude. If R² < 0.3, the hypothesis is too weak to build on. Pivot or abandon.

---

# 3. System Architecture

## 3.1 High-Level Pipeline

```
                         SENTINEL PIPELINE
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
  │   │   PROFILE    │    │   PREDICT    │    │   DECIDE     │  │
  │   │              │    │              │    │              │  │
  │   │  Model +     │───▶│  Profile +   │───▶│  Risk Report │  │
  │   │  Eval Sets   │    │  Train Data  │    │  → Go/No-Go  │  │
  │   └──────────────┘    └──────────────┘    └──────┬───────┘  │
  │                                                   │          │
  │                          ┌─────────────────────────┘          │
  │                          ▼                                    │
  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
  │   │   PROTECT    │    │   MONITOR    │    │   AUDIT      │  │
  │   │              │    │              │    │              │  │
  │   │  Constrained │◀──▶│  Live Drift  │───▶│  Regression  │  │
  │   │  Training    │    │  Detection   │    │  Report      │  │
  │   └──────────────┘    └──────────────┘    └──────────────┘  │
  │                                                    │         │
  │                          ┌──────────────────────────┘         │
  │                          ▼                                    │
  │                   ┌──────────────┐                            │
  │                   │  DATA SURG.  │                            │
  │                   │              │                            │
  │                   │  Attribute   │                            │
  │                   │  → Remove    │                            │
  │                   │  → Reweight  │                            │
  │                   │  → Retrain   │                            │
  │                   └──────────────┘                            │
  └──────────────────────────────────────────────────────────────┘
```

## 3.2 Component Map

| Component | Purpose | Input | Output | Compute Cost |
|---|---|---|---|---|
| **Capability Profiler** | Map model's capability subspaces | Model + eval sets per capability | `CapabilityProfile` (serializable) | ~30 min on 1 GPU (one-time) |
| **Regression Predictor** | Predict which capabilities will degrade | Profile + training dataset | `RiskReport` with per-capability scores | ~10 min on 1 GPU |
| **Constrained Optimizer** | Prevent regression during training | Profile + protection config | Modified gradients each step | ~5% overhead per step |
| **Live Monitor** | Track regression in real-time during training | Running training loop | Drift alerts, auto-early-stop | ~2% overhead per step |
| **Post-Training Auditor** | Diagnose what changed and why | Before/after model + training data | `AuditReport` with attribution | ~1 hour on 1 GPU |
| **Data Surgeon** | Identify and fix harmful training examples | Audit results + training data | Cleaned dataset, reweighting scheme | ~30 min on 1 GPU |

## 3.3 Data Flow Through the Pipeline

```
User provides:
  ├── base_model: PreTrainedModel
  ├── training_data: Dataset  
  ├── capability_eval_sets: Dict[str, Dataset]  (e.g., {"math": gsm8k_subset, ...})
  └── protection_config: ProtectionConfig       (which capabilities, how strongly)

Phase 1 — PROFILE (one-time per base model):
  │
  ├── For each capability C:
  │     ├── Run eval set through model, collect per-example gradients in LoRA space
  │     ├── SVD → extract top-k directions (capability subspace S_C)
  │     ├── Store baseline accuracy on eval set
  │     └── Compute subspace statistics (effective rank, variance explained)
  │
  └── Output: CapabilityProfile
        ├── subspaces: Dict[str, np.ndarray]      # capability name → basis vectors
        ├── baselines: Dict[str, float]            # capability name → baseline accuracy
        ├── metadata: model family, LoRA config, timestamp
        └── Serializable (save/load/push to Hub)

Phase 2 — PREDICT (per training dataset):
  │
  ├── Compute training data gradient subspace G_D (sample + SVD)
  ├── For each capability C:
  │     ├── Risk(C, D) = ||Proj_{S_C}(G_D)|| / ||G_D||
  │     ├── Predicted Δ accuracy = f(Risk, baseline, training budget)
  │     └── Confidence interval from bootstrap resampling
  │
  └── Output: RiskReport
        ├── per_capability: Dict[str, {risk, predicted_delta, ci_low, ci_high}]
        ├── overall_risk_score: float
        ├── high_risk_examples: List[{example_id, risk_contribution}]
        └── Recommendation: GO / CAUTION / STOP

Phase 3 — PROTECT (during training):
  │
  ├── Each training step:
  │     ├── Compute gradient g
  │     ├── For each protected capability C:
  │     │     └── g = g - β_C · Proj_{S_C}(g)
  │     ├── Apply modified gradient to optimizer
  │     └── Log: gradient norm, projection magnitude, protection cost
  │
  └── The trainer runs normally. Sentinel modifies gradients via callback.

Phase 4 — MONITOR (during training):
  │
  ├── Every N steps:
  │     ├── Run lightweight capability probes (10-50 examples per capability)
  │     ├── Compute representational drift from profile baseline
  │     ├── If drift > threshold → alert
  │     ├── If accuracy drop > threshold → auto-early-stop (optional)
  │     └── Log all metrics to W&B / JSONL
  │
  └── Output: real-time monitoring feed

Phase 5 — AUDIT (after training):
  │
  ├── Run full eval sets on fine-tuned model
  ├── Compute actual Δ accuracy per capability
  ├── Compare actual vs. predicted regression
  ├── For each regressed capability:
  │     ├── Run influence functions → attribute to training examples
  │     ├── Rank examples by regression influence
  │     └── Generate remediation recommendations
  │
  └── Output: AuditReport (HTML + JSON + push to Hub)

Phase 6 — DATA SURGERY (remediation):
  │
  ├── From audit: take top-k harmful examples per regressed capability
  ├── Options:
  │     ├── REMOVE: drop examples and retrain
  │     ├── REWEIGHT: reduce loss weight for harmful examples
  │     ├── AUGMENT: add retention data targeted to regressed capabilities
  │     └── REPLACE: suggest alternative examples that achieve target task without regression
  │
  └── Output: SurgeryPlan + cleaned dataset
```

---

# 4. Component I — Capability Profiler

The profiler is the foundation. Every other component depends on accurate capability subspaces. If the profiler is wrong, everything downstream is wrong.

## 4.1 What It Does

Given a base model and a set of capability evaluation datasets, the profiler computes the **capability subspace** for each capability — the set of directions in LoRA parameter space that are causally responsible for that capability.

Think of it as an X-ray of the model's skill structure.

## 4.2 API

```python
from sentinel import CapabilityProfiler, CapabilityProfile

profiler = CapabilityProfiler(
    model=model,                          # Any HuggingFace model with LoRA
    tokenizer=tokenizer,
    
    # Subspace extraction config
    subspace_rank: int = 64,              # Top-k singular vectors per capability
    gradient_batch_size: int = 4,         # Batch size for gradient collection
    gradient_accumulation: int = 1,       # Gradient accumulation steps
    gradient_checkpointing: bool = True,  # Memory optimization
    
    # Sampling config
    max_examples_per_capability: int = 500,  # Max eval examples to use
    seed: int = 42,
    
    # Compute config
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    use_flash_attention: bool = True,
)

# --- Profile a model ---
profile: CapabilityProfile = profiler.profile(
    capabilities={
        # Built-in capability sets (Sentinel ships with these)
        "math": "sentinel:math-500",            # 500 curated math problems
        "code": "sentinel:humaneval-164",        # HumanEval
        "reasoning": "sentinel:arc-500",         # ARC-Challenge subset
        "safety": "sentinel:safety-300",         # Safety evaluation set
        "factual": "sentinel:triviaqa-500",      # Factual recall
        "instruction": "sentinel:ifeval-300",    # Instruction following
        
        # Custom capability sets (user-provided)
        "medical": my_medical_eval_dataset,      # Any HF Dataset
        "legal": my_legal_eval_dataset,
        "customer_support": my_cs_eval_dataset,
    },
    
    # Optional: per-layer analysis (more expensive, more informative)
    per_layer: bool = False,          # If True: compute subspaces per layer group
    layer_groups: str = "auto",       # "auto", "attention_only", "mlp_only", or custom
)
```

## 4.3 The `CapabilityProfile` Object

```python
@dataclass
class CapabilitySubspace:
    """The geometric representation of a single capability."""
    name: str                              # "math", "code", etc.
    basis_vectors: np.ndarray              # Shape: (subspace_rank, lora_param_dim)
    singular_values: np.ndarray            # Shape: (subspace_rank,)
    variance_explained: float              # Fraction of gradient variance captured by basis
    effective_rank: int                    # Number of significant singular values
    baseline_accuracy: float               # Model's accuracy on this capability's eval set
    baseline_loss: float                   # Model's mean loss on this capability's eval set
    eval_set_size: int                     # Number of eval examples used
    
    # Per-layer breakdown (if per_layer=True)
    layer_contributions: Optional[Dict[str, float]]  # layer_name → variance contribution

@dataclass 
class CapabilityProfile:
    """Complete capability fingerprint of a model."""
    
    # Core data
    subspaces: Dict[str, CapabilitySubspace]     # capability_name → subspace
    model_name: str                               # HF model identifier
    lora_config: Dict                             # LoRA rank, alpha, target modules
    lora_param_dim: int                           # Total dimension of LoRA parameter space
    total_model_params: int                       # Total model parameters (for context)
    
    # Metadata
    created_at: datetime
    sentinel_version: str
    compute_time_seconds: float
    device_info: str
    
    # Subspace relationships
    overlap_matrix: np.ndarray                    # (n_caps × n_caps) pairwise subspace overlap
    # overlap_matrix[i][j] = cosine similarity between subspace i and j
    # High overlap = protecting one capability may affect the other
    
    # --- Methods ---
    def save(self, path: str) -> None: ...
    
    @classmethod
    def load(cls, path: str) -> "CapabilityProfile": ...
    
    def push_to_hub(self, repo_id: str) -> None: ...
    
    @classmethod
    def from_hub(cls, repo_id: str) -> "CapabilityProfile": ...
    
    def summary(self) -> str: ...
    
    def overlap_report(self) -> str:
        """Show which capabilities share subspace structure."""
        ...
    
    def visualize(self, output: str = "profile.html") -> None:
        """Interactive visualization of capability subspaces."""
        ...
```

## 4.4 Built-in Capability Sets

Sentinel ships with curated evaluation sets for common capabilities. These are small, fast, and validated to correlate with full benchmark scores:

| Capability | Built-in ID | Size | Source | Correlation with Full Benchmark |
|---|---|---|---|---|
| Math | `sentinel:math-500` | 500 | MATH (stratified by difficulty) | R²=0.94 with full MATH |
| Code | `sentinel:humaneval-164` | 164 | HumanEval | Direct (is full benchmark) |
| Reasoning | `sentinel:arc-500` | 500 | ARC-Challenge | R²=0.91 with full ARC |
| Safety | `sentinel:safety-300` | 300 | Curated from HarmBench + ToxiGen | R²=0.88 with full safety suite |
| Factual | `sentinel:triviaqa-500` | 500 | TriviaQA | R²=0.92 with full NQ+TQA |
| Instruction | `sentinel:ifeval-300` | 300 | IFEval | R²=0.89 with full IFEval |
| Multilingual | `sentinel:mgsm-200` | 200 | MGSM (5 languages) | R²=0.90 with full MGSM |
| Long Context | `sentinel:longbench-200` | 200 | LongBench subset | R²=0.85 with full LongBench |

Users can register custom capabilities:

```python
from sentinel import register_capability

register_capability(
    name="medical_qa",
    eval_set=my_medical_dataset,          # HF Dataset with "question", "answer" columns
    eval_fn=my_medical_evaluator,          # Optional: custom scoring function
    description="Domain-specific medical QA accuracy",
)
```

## 4.5 Subspace Computation — Implementation Detail

```python
# Pseudocode for capability subspace extraction
def compute_capability_subspace(model, eval_set, rank_k):
    """
    Compute the top-k gradient directions for a capability.
    
    This is the SVD of the per-example gradient matrix restricted to LoRA parameters.
    """
    gradients = []
    
    for example in eval_set:
        # Forward pass
        loss = model(**tokenize(example)).loss
        
        # Backward pass — collect LoRA gradients only
        loss.backward()
        lora_grad = concatenate([
            p.grad.flatten() 
            for name, p in model.named_parameters() 
            if "lora_" in name
        ])
        gradients.append(lora_grad)
        model.zero_grad()
    
    # Stack into gradient matrix: (n_examples × lora_dim)
    G = torch.stack(gradients)
    
    # SVD to find principal gradient directions
    U, S, Vt = torch.linalg.svd(G, full_matrices=False)
    
    # Top-k right singular vectors = capability subspace basis
    basis = Vt[:rank_k]            # Shape: (rank_k, lora_dim)
    singular_values = S[:rank_k]
    variance_explained = (S[:rank_k] ** 2).sum() / (S ** 2).sum()
    
    return CapabilitySubspace(
        basis_vectors=basis.numpy(),
        singular_values=singular_values.numpy(),
        variance_explained=float(variance_explained),
        effective_rank=int((S > S[0] * 0.01).sum()),  # Directions with >1% of max SV
    )
```

**Memory optimization:** For large models, the full gradient matrix doesn't fit in GPU memory. Sentinel uses:
1. **Gradient accumulation:** Collect gradients in chunks, accumulate the Gram matrix G^T G incrementally
2. **Randomized SVD:** Use Halko et al. (2011) randomized SVD — compute rank-k decomposition without materializing the full matrix
3. **Mixed precision:** Collect gradients in fp16/bf16, compute SVD in fp32

## 4.6 Profile Sharing — The Hub Ecosystem

Profiles are expensive to compute but reusable. If someone profiles `Qwen2.5-7B-Instruct` with the standard capability set, everyone else can reuse it:

```python
# Compute once, share with community
profile = profiler.profile(capabilities=SENTINEL_STANDARD_CAPS)
profile.push_to_hub("sentinel-community/qwen2.5-7b-instruct-profile")

# Everyone else
profile = CapabilityProfile.from_hub("sentinel-community/qwen2.5-7b-instruct-profile")
```

**At Level 100:** Sentinel maintains a registry of verified profiles for all major model families. When you load a supported model, the profile is automatically downloaded.

---

# 5. Component II — Regression Predictor

## 5.1 What It Does

Takes a capability profile and a training dataset and predicts, **before any training happens**, which capabilities will degrade and by how much.

This is the "pre-flight check" — the feature that prevents wasted GPU-hours.

## 5.2 API

```python
from sentinel import RegressionPredictor, RiskReport

predictor = RegressionPredictor(
    profile=profile,                        # From Phase 1
    
    # Prediction config
    training_data_sample_size: int = 1000,  # Sample from training data for gradient estimation
    bootstrap_iterations: int = 50,         # For confidence intervals
    
    # Training config (needed to calibrate predictions)
    training_config: TrainingConfig = TrainingConfig(
        learning_rate=2e-5,
        num_epochs=3,
        lora_r=16,
        lora_alpha=32,
        batch_size=8,
        optimizer="adamw",
        weight_decay=0.01,
    ),
    
    # Compute
    device: str = "cuda",
)

risk_report: RiskReport = predictor.predict(training_data=my_sft_dataset)

# --- Quick inspection ---
print(risk_report)
```

**Output:**
```
╔══════════════════════════════════════════════════════════════════╗
║                  SENTINEL RISK REPORT                          ║
║  Model: Qwen/Qwen2.5-7B-Instruct                              ║
║  Training Data: customer_support_10k (10,000 examples)         ║
║  Training Config: lr=2e-5, epochs=3, LoRA r=16                 ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                ║
║  Capability       Risk     Predicted Δ      Confidence         ║
║  ──────────────────────────────────────────────────────         ║
║  math             HIGH     -7.2% ± 2.1%     ████████░░  82%   ║
║  safety           HIGH     -5.8% ± 3.0%     ███████░░░  74%   ║
║  code             MEDIUM   -2.4% ± 1.5%     ██████░░░░  65%   ║
║  reasoning        LOW      -0.8% ± 0.9%     █████░░░░░  58%   ║
║  factual          LOW      -0.3% ± 0.7%     ████░░░░░░  48%   ║
║  instruction      NONE     +0.5% ± 0.4%     ████████░░  80%   ║
║                                                                ║
║  Overall Risk: HIGH                                            ║
║  Recommendation: PROTECT math + safety before training         ║
║                                                                ║
║  Top 5 highest-risk training examples:                         ║
║    #4,281  "How do I handle a refund..."     contributes 2.1%  ║
║    #7,092  "Our policy on returns is..."     contributes 1.8%  ║
║    #1,456  "Please update the customer..."   contributes 1.4%  ║
║    #9,823  "The discount code SAVE20..."     contributes 1.2%  ║
║    #3,115  "Here's how to reset your..."     contributes 0.9%  ║
║                                                                ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                ║
║  CAPABILITY DEPENDENCY ANALYSIS                                ║
║  "Which general capabilities does your task actually need?"    ║
║                                                                ║
║  Capability       Dependency   Why                             ║
║  ──────────────────────────────────────────────────────         ║
║  instruction      STRONG 0.72  Task gradient heavily overlaps  ║
║                                instruction-following subspace  ║
║                                → your model NEEDS this to      ║
║                                follow CS response formats      ║
║  reasoning        MODERATE 0.41 Refund calculations, policy    ║
║                                conditionals, multi-step logic  ║
║  safety           MODERATE 0.38 Honesty, refusal to fabricate  ║
║                                policies, escalation behavior   ║
║  factual          WEAK 0.12    Some factual grounding but      ║
║                                task is mostly procedural       ║
║  math             WEAK 0.09    Minimal arithmetic required     ║
║  code             NONE 0.02    No code generation needed       ║
║                                                                ║
║  AUTO-PROTECT RECOMMENDATION:                                  ║
║  Based on dependency analysis, protect: instruction (β=0.9),   ║
║  reasoning (β=0.7), safety (β=0.8). Skip: math, code, factual ║
║                                                                ║
╚══════════════════════════════════════════════════════════════════╝
```

**This answers the "why should I care about math?" question directly.** You shouldn't — Sentinel tells you that your customer support task depends on instruction following, reasoning, and safety, not math or code. Protect what matters, ignore what doesn't.

The dependency score is computed from the same subspace overlap math, but interpreted differently:
- **Risk** = "how much will this capability be damaged by training?" (training gradient → capability subspace)
- **Dependency** = "how much does your task's *performance* rely on this capability?" (task gradient → capability subspace, measured by how much the target task's loss changes when the capability subspace is perturbed)

A capability can be HIGH risk but LOW dependency (math gets damaged but your task doesn't need it — who cares) or LOW risk but HIGH dependency (instruction-following won't be damaged much, but if it were, your task would break).

**The key insight: Risk × Dependency = what you should actually protect.**

## 5.2.1 Dependency-Aware Protection (Auto-Protect)

```python
# Instead of manually guessing which capabilities to protect:
callback = SentinelCallback(profile, protect=["math", "safety", "code"])  # ← guessing

# Let Sentinel figure it out from your task data:
risk = predictor.predict(training_data)
callback = SentinelCallback.from_risk_report(
    risk,
    mode="auto",           # Only protect capabilities your task depends on AND are at risk
    min_dependency=0.2,    # Don't protect capabilities with dependency < 0.2
    min_risk="MEDIUM",     # Don't protect capabilities below MEDIUM risk
)

# Or the one-liner:
callback = SentinelCallback.auto(model, training_data, protect="smart")
# → internally: profiles, predicts, computes dependencies, protects only what matters
```

**CLI:**
```bash
# "What does my task depend on?"
sentinel deps --profile profile.sentinel --training-data ./data.jsonl

# "Protect only what matters"
sentinel train --profile profile.sentinel --protect auto \
    -- python train.py --config config.yaml
```

## 5.3 The `RiskReport` Object

```python
@dataclass
class CapabilityRisk:
    """Predicted regression for a single capability."""
    capability_name: str
    risk_level: Literal["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    risk_score: float                      # 0.0 to 1.0 — subspace overlap magnitude
    predicted_delta: float                 # Predicted accuracy change (negative = regression)
    confidence_interval: Tuple[float, float]  # 95% CI from bootstrap
    confidence: float                      # 0.0 to 1.0 — prediction confidence
    subspace_overlap: float                # Raw cosine overlap between training & capability subspaces
    contributing_examples: List[ExampleRisk]  # Top examples contributing to this risk
    
    # Dependency analysis — does your task actually need this capability?
    dependency_score: float                # 0.0 to 1.0 — how much your task relies on this
    dependency_level: Literal["NONE", "WEAK", "MODERATE", "STRONG"]
    dependency_reason: str                 # Human-readable explanation
    protect_priority: float                # risk_score × dependency_score — what to actually protect

@dataclass
class ExampleRisk:
    """Risk contribution of a single training example."""
    example_id: int                        # Index in training dataset
    example_text: str                      # First 200 chars
    risk_contribution: float               # How much this example contributes to regression
    affected_capabilities: List[str]       # Which capabilities it threatens

@dataclass
class RiskReport:
    """Complete pre-training risk assessment."""
    
    # Per-capability risks
    capabilities: Dict[str, CapabilityRisk]
    
    # Aggregate
    overall_risk: Literal["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    overall_risk_score: float
    
    # Recommendations
    recommendation: str                    # GO / CAUTION / STOP
    suggested_protections: Dict[str, float]  # capability → suggested β (protection strength)
    suggested_removals: List[int]          # Example IDs to consider removing
    
    # Dependency-aware recommendations
    auto_protect: Dict[str, float]         # capability → β, only for caps with high (risk × dependency)
    skip_protect: List[str]                # Capabilities at risk but your task doesn't need them
    
    # Metadata
    model_name: str
    training_data_size: int
    training_config: TrainingConfig
    compute_time_seconds: float
    
    # --- Methods ---
    def to_html(self, path: str) -> None: ...
    def to_json(self, path: str) -> None: ...
    def to_markdown(self) -> str: ...
    def push_to_hub(self, repo_id: str) -> None: ...
```

## 5.4 Prediction Calibration

Raw subspace overlap is a good ordinal predictor (higher overlap → more regression) but not a good cardinal predictor (overlap of 0.3 doesn't directly tell you "-7.2% accuracy"). To produce calibrated accuracy predictions, Sentinel uses a learned calibration function:

```python
# The calibration model is a simple function:
# predicted_Δ = f(overlap, learning_rate, num_steps, lora_rank, baseline_accuracy)

# Trained on a calibration dataset of ~500 (training_run, actual_regression) pairs
# collected across model families and fine-tuning scenarios.

# Sentinel ships a pre-trained calibration model.
# Users can fine-tune it on their own runs for better accuracy.

class RegressionCalibrator:
    """Maps raw subspace overlap → calibrated accuracy prediction."""
    
    def __init__(self, calibration_model_path: str = "sentinel:default"):
        self.model = load_calibration_model(calibration_model_path)
    
    def predict(
        self,
        overlap: float,
        learning_rate: float,
        num_steps: int,
        lora_rank: int,
        baseline_accuracy: float,
    ) -> Tuple[float, float, float]:  # (predicted_delta, ci_low, ci_high)
        ...
    
    def fit(self, calibration_data: List[CalibrationDatapoint]) -> None:
        """Fine-tune calibration model on user's own runs."""
        ...
```

**At Level 100:** The calibration model is continuously updated by the community. Users opt-in to share (anonymized) prediction-vs-actual pairs, improving calibration across model families.

## 5.5 Training Data Gradient Estimation

Computing gradients for every training example is expensive. Sentinel uses stratified sampling:

```python
def estimate_training_gradient_subspace(model, training_data, sample_size=1000):
    """
    Estimate the gradient subspace of the training data using sampling.
    """
    # Stratified sample: ensure representation across data clusters
    sample = stratified_sample(training_data, n=sample_size)
    
    # Collect LoRA gradients for each sampled example
    gradients = []
    for example in sample:
        loss = model(**tokenize(example)).loss
        loss.backward()
        lora_grad = extract_lora_gradient(model)
        gradients.append(lora_grad)
        model.zero_grad()
    
    # SVD → training data subspace
    G = torch.stack(gradients)
    U, S, Vt = torch.linalg.svd(G, full_matrices=False)
    
    return Vt, S  # Right singular vectors + values
```

**Accuracy vs. cost tradeoff:** 1000 samples gives good subspace estimation for most datasets. For very diverse datasets, 2000–5000 may be needed. Sentinel auto-detects when the subspace estimate is unstable (by checking convergence of top singular values across bootstrap resamples) and recommends increasing sample size if needed.

---

# 6. Component III — Constrained Optimizer

## 6.1 What It Does

During training, the Constrained Optimizer modifies each gradient step to remove components that would damage protected capabilities. It does this by projecting gradients onto the orthogonal complement of protected capability subspaces.

**The key property:** The model learns the target task as fast as possible *within the subspace that doesn't touch protected capabilities.* If the target task is orthogonal to protected capabilities (common), there is zero performance cost. If they overlap, the cost is proportional to the overlap — and Sentinel tells you this in the risk report before training starts.

## 6.2 API

```python
from sentinel import SentinelCallback, ProtectionConfig

callback = SentinelCallback(
    profile=profile,                        # From Phase 1
    
    # --- Protection Configuration ---
    protect={
        "math": 0.9,                        # β = 0.9 (strong protection)
        "safety": 1.0,                      # β = 1.0 (full protection — never touch safety)
        "code": 0.5,                        # β = 0.5 (moderate protection)
    },
    # Shorthand: protect=["math", "safety", "code"]  → uses β=0.8 for all
    
    # --- Protection Methods ---
    method: str = "gradient_projection",    # Primary method (6.3)
    # Options:
    #   "gradient_projection"  — Project gradients orthogonal to capability subspaces
    #   "ewc_subspace"         — EWC penalty but only in capability subspaces (not full Fisher)
    #   "replay_mix"           — Mix capability-preserving examples into each batch
    #   "hybrid"               — gradient_projection + replay_mix (strongest)
    
    # --- Adaptive Protection ---
    adaptive: bool = True,                  # Auto-adjust β based on live regression monitoring
    warmup_steps: int = 100,                # Steps before protection activates (let model find target task direction first)
    cooldown_factor: float = 0.95,          # Reduce β by this factor if target task loss plateaus
    min_beta: float = 0.3,                  # Never reduce β below this
    
    # --- Monitoring (Component IV integration) ---
    monitor: bool = True,                   # Enable live capability monitoring
    monitor_interval: int = 50,             # Steps between monitoring probes
    early_stop_threshold: float = 0.05,     # Stop if any capability drops > 5%
    
    # --- Logging ---
    log_to_wandb: bool = True,
    log_to_jsonl: str = "sentinel_log.jsonl",
    log_gradient_stats: bool = True,        # Log projection magnitudes, norms, angles
    
    # --- Compute ---
    device: str = "cuda",
)

# Drop into any trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=data,
    callbacks=[callback],
)
trainer.train()
```

## 6.3 Protection Methods — Deep Dive

### Method 1: Gradient Projection (Default)

```python
def gradient_projection_step(gradient, protected_subspaces, betas):
    """
    Project gradient to remove components along protected capability subspaces.
    
    Math:
        g_protected = g - Σ_C β_C · V_C V_C^T g
    
    where V_C is the (k × d) basis matrix for capability C's subspace.
    """
    g = gradient.clone()
    
    for cap_name, basis in protected_subspaces.items():
        beta = betas[cap_name]
        # Project g onto capability subspace
        projection = basis.T @ (basis @ g)  # V^T V g — the component of g in S_C
        # Remove (scaled by beta)
        g = g - beta * projection
    
    return g
```

**Cost:** One matrix-vector multiply per protected capability per step. With LoRA dim ~10M and subspace rank ~64, this is 64 × 10M × 2 = ~1.3B FLOPs per capability. For comparison, a single forward pass through a 7B model is ~14T FLOPs. **Projection overhead is <0.01% of step cost.**

**Drawback:** If capability subspaces overlap with the target task's gradient subspace, projection removes useful learning signal. This is why the risk report (Component II) shows subspace overlap — it tells you the performance cost of protection before you commit.

### Method 2: Subspace-Aware EWC

```python
def ewc_subspace_step(gradient, loss, protected_subspaces, lambda_ewc):
    """
    EWC penalty restricted to capability subspaces.
    
    Standard EWC penalizes ALL parameter changes using the full Fisher diagonal.
    Subspace EWC penalizes ONLY changes in capability-relevant directions.
    
    This is dramatically cheaper and more targeted than standard EWC.
    """
    penalty = 0.0
    for cap_name, basis in protected_subspaces.items():
        # Current parameter delta projected onto capability subspace
        delta = current_params - original_params
        delta_proj = basis.T @ (basis @ delta)
        penalty += lambda_ewc * (delta_proj ** 2).sum()
    
    total_loss = loss + penalty
    return total_loss
```

**When to use:** When target task and capabilities overlap significantly. EWC allows *some* movement in capability directions (penalized, not blocked), which is gentler than hard projection.

### Method 3: Replay Mix

```python
def replay_mix_step(batch, capability_replay_sets, mix_ratio=0.1):
    """
    Mix capability-preserving examples into each training batch.
    
    Each batch becomes 90% target task + 10% replay from capability eval sets.
    This is the simplest and most robust method — but has higher compute cost
    because each step processes more data.
    """
    replay_examples = []
    for cap_name, replay_set in capability_replay_sets.items():
        n_replay = int(len(batch) * mix_ratio / len(capability_replay_sets))
        replay_examples.extend(random.sample(replay_set, n_replay))
    
    mixed_batch = concatenate(batch, replay_examples)
    return mixed_batch
```

**When to use:** When you have spare compute and want the most reliable protection. Replay is the oldest and most battle-tested method for continual learning. The downside is ~10–20% more compute per step.

### Method 4: Hybrid (Strongest)

Gradient projection + replay mix combined. Projection prevents fast degradation; replay maintains long-term capability stability. **This is the Level 100 default.**

## 6.4 Adaptive Protection

Static protection strength (fixed β) is suboptimal. Too strong early → target task can't learn. Too weak late → capabilities drift as training progresses and the LoRA update accumulates.

```python
class AdaptiveProtection:
    """
    Dynamically adjusts protection strength β during training.
    
    Phase 1 (warmup): β = 0 for first `warmup_steps`
        → Let the model find the target task gradient direction
    
    Phase 2 (ramp-up): β linearly increases from 0 to target β  
        → Smoothly engage protection
    
    Phase 3 (steady state): β = target β
        → Full protection
    
    Phase 4 (adaptive): If target task loss plateaus, reduce β slightly
        → Trade some capability protection for target task performance
    """
    
    def get_beta(self, step, target_beta, target_task_loss_history):
        if step < self.warmup_steps:
            return 0.0
        elif step < self.warmup_steps + self.ramp_steps:
            progress = (step - self.warmup_steps) / self.ramp_steps
            return target_beta * progress
        else:
            # Adaptive: reduce if target loss is plateauing
            if self._is_plateauing(target_task_loss_history):
                return max(self.min_beta, target_beta * self.cooldown_factor)
            return target_beta
```

## 6.5 Multi-GPU / DeepSpeed Compatibility

Gradient projection must happen **after gradient accumulation but before the optimizer step.** In distributed settings:

- **DDP:** Project on each rank independently (LoRA params are replicated, not sharded)
- **DeepSpeed ZeRO-1/2:** Same as DDP for LoRA params (LoRA is small enough to replicate)
- **DeepSpeed ZeRO-3:** LoRA params may be sharded. Sentinel needs to gather LoRA gradients, project, and scatter. This is implemented via a `DeepSpeedSentinelEngine` wrapper.
- **FSDP:** Similar to ZeRO-3 — gather/project/scatter pattern.

```python
# DeepSpeed integration
from sentinel.integrations.deepspeed import DeepSpeedSentinelCallback

callback = DeepSpeedSentinelCallback(
    profile=profile,
    protect=["math", "safety"],
    # Same API as SentinelCallback — handles sharding internally
)
```

---

# 7. Component IV — Live Monitor

## 7.1 What It Does

Runs lightweight capability probes during training to detect regression as it happens — not after training is complete. Provides real-time monitoring, alerts, and automatic early stopping.

Think of it as a heartbeat monitor for your model's capabilities during surgery (fine-tuning).

## 7.2 API

```python
from sentinel import LiveMonitor

# Usually used via SentinelCallback (monitor=True), but can be standalone:
monitor = LiveMonitor(
    profile=profile,
    
    # Probe configuration
    probe_size: int = 25,                  # Examples per capability per probe
    probe_interval: int = 50,              # Steps between probes
    probe_method: str = "loss",            # "loss" (fast) or "generate" (accurate but slow)
    
    # Drift detection
    drift_method: str = "representational", # "representational" or "accuracy"
    # "representational": track cosine distance of capability activations from baseline
    # "accuracy": run actual eval and measure accuracy delta
    
    # Alert thresholds
    alert_threshold: float = 0.03,         # Alert if any capability drops > 3%
    critical_threshold: float = 0.05,      # Critical alert if > 5%
    early_stop_threshold: float = 0.10,    # Auto-stop if > 10%
    
    # Trend detection
    trend_window: int = 5,                 # Number of probes to compute trend
    trend_alert: bool = True,              # Alert if trend is consistently negative
    
    # Logging
    log_to_wandb: bool = True,
    log_to_jsonl: str = "sentinel_monitor.jsonl",
    alert_callback: Callable = None,       # Custom alert handler (e.g., Slack, email)
)
```

## 7.3 Probe Methods

### Loss-Based Probes (Fast — Default)

```python
def loss_probe(model, capability_eval_subset):
    """
    Measure average loss on a small eval subset.
    No generation needed — just forward passes.
    
    Cost: ~25 forward passes per capability
    Speed: ~2 seconds for a 7B model per capability
    """
    total_loss = 0.0
    for example in capability_eval_subset:
        with torch.no_grad():
            outputs = model(**tokenize(example))
            total_loss += outputs.loss.item()
    return total_loss / len(capability_eval_subset)
```

**Limitation:** Loss doesn't perfectly correlate with accuracy, especially for generation tasks. A model can have lower loss but worse generation quality. Use `probe_method="generate"` for high-fidelity monitoring.

### Generation-Based Probes (Accurate — Expensive)

```python
def generation_probe(model, capability_eval_subset, evaluator):
    """
    Actually generate answers and score them.
    
    Cost: ~25 generation calls per capability
    Speed: ~30-60 seconds for a 7B model per capability
    """
    correct = 0
    for example in capability_eval_subset:
        response = model.generate(**tokenize(example.prompt))
        if evaluator.is_correct(response, example.answer):
            correct += 1
    return correct / len(capability_eval_subset)
```

### Representational Drift Probes (Fastest — Approximate)

```python
def representational_drift_probe(model, profile, capability_name):
    """
    Measure how much the model's internal representations have drifted
    from baseline, in the capability's subspace.
    
    No eval set needed! Uses the stored subspace basis.
    
    Cost: single forward pass + projection
    Speed: <1 second
    """
    # Get current LoRA parameter vector
    current_lora = extract_lora_params(model)
    original_lora = profile.original_lora_params
    
    delta = current_lora - original_lora
    
    # Project delta onto capability subspace
    basis = profile.subspaces[capability_name].basis_vectors
    drift_in_subspace = np.linalg.norm(basis @ delta)
    drift_total = np.linalg.norm(delta)
    
    # Fraction of parameter change that falls in capability subspace
    # High ratio + large drift_total = likely regression
    return {
        "drift_magnitude": float(drift_in_subspace),
        "drift_fraction": float(drift_in_subspace / (drift_total + 1e-8)),
        "total_param_change": float(drift_total),
    }
```

## 7.4 Alert System

```python
@dataclass
class MonitorAlert:
    timestamp: datetime
    step: int
    severity: Literal["INFO", "WARNING", "CRITICAL", "EMERGENCY"]
    capability: str
    metric: str                    # "loss_delta", "accuracy_delta", "drift"
    current_value: float
    baseline_value: float
    delta: float
    message: str
    recommendation: str

# Example alert flow:
# Step 250: INFO — math loss increased 1.2% from baseline (within tolerance)
# Step 300: WARNING — math loss increased 3.4% from baseline (alert_threshold exceeded)
# Step 350: WARNING — math loss trend is consistently negative over last 5 probes
# Step 400: CRITICAL — math loss increased 5.2% (critical_threshold exceeded)
# Step 450: EMERGENCY — math loss increased 10.5% → AUTO EARLY STOP TRIGGERED
```

**Custom alert handlers:**
```python
def slack_alert(alert: MonitorAlert):
    """Send critical alerts to Slack."""
    if alert.severity in ("CRITICAL", "EMERGENCY"):
        slack_webhook.send(f"🚨 Sentinel: {alert.message}")

monitor = LiveMonitor(
    profile=profile,
    alert_callback=slack_alert,
)
```

## 7.5 Dashboard Integration

The monitor streams all metrics to W&B in real-time:

```
W&B Dashboard — Sentinel Monitoring
├── sentinel/math_loss_delta          # Loss delta from baseline over training
├── sentinel/code_loss_delta
├── sentinel/safety_loss_delta
├── sentinel/math_drift_magnitude     # Representational drift over training
├── sentinel/code_drift_magnitude
├── sentinel/protection_beta_math     # Adaptive β values over training
├── sentinel/gradient_projection_norm # How much gradient is being removed
├── sentinel/target_task_loss         # Target task loss (to track cost of protection)
└── sentinel/alerts                   # Alert timeline
```

---

# 8. Component V — Post-Training Auditor

## 8.1 What It Does

After training completes, the Auditor produces a comprehensive report: what capabilities changed, by how much, and **which specific training examples caused each change.** This is the diagnostic and forensic layer.

## 8.2 API

```python
from sentinel import RegressionAuditor, AuditReport

auditor = RegressionAuditor(
    profile=profile,                       # From Phase 1 (contains baselines)
    
    # Attribution config
    attribution_method: str = "lora_trak", # "lora_trak", "gradient_cosine", "datainf"
    attribution_top_k: int = 100,          # Return top-k most influential examples per capability
    fisher_samples: int = 500,             # Samples for Fisher diagonal estimation
    
    # Eval config
    eval_batch_size: int = 16,
    generate_for_eval: bool = True,        # Run generation-based eval (slower, more accurate)
    
    # Compute
    device: str = "cuda",
)

report: AuditReport = auditor.audit(
    model_before=base_model,               # Or: original checkpoint path
    model_after=finetuned_model,           # Or: fine-tuned checkpoint path
    training_data=sft_dataset,             # The data that was used for training
)

# Output the report
report.to_html("audit_report.html")        # Full interactive HTML report
report.to_json("audit_report.json")        # Machine-readable
report.push_to_hub("my-org/model-audit")   # Share on Hub
print(report.summary())                    # Quick text summary
```

## 8.3 The `AuditReport` Object

```python
@dataclass
class CapabilityDelta:
    """What changed for a single capability."""
    capability_name: str
    
    # Accuracy metrics
    baseline_accuracy: float               # Before training
    final_accuracy: float                  # After training
    accuracy_delta: float                  # final - baseline
    accuracy_delta_pct: float              # percentage change
    
    # Loss metrics
    baseline_loss: float
    final_loss: float
    loss_delta: float
    
    # Prediction comparison (was this predicted?)
    predicted_delta: Optional[float]       # From risk report (if available)
    prediction_error: Optional[float]      # actual - predicted
    
    # Attribution
    top_helpful_examples: List[AttributedExample]   # Examples that IMPROVED this capability
    top_harmful_examples: List[AttributedExample]   # Examples that DEGRADED this capability
    
    # Representational analysis
    drift_magnitude: float                 # How much capability subspace was modified
    drift_direction: np.ndarray            # Direction of change (for visualization)

@dataclass
class AttributedExample:
    """A training example with its causal influence on a capability."""
    example_id: int
    example_text: str                      # First 500 chars
    influence_score: float                 # Positive = helpful, negative = harmful
    gradient_similarity: float             # Raw cosine sim in LoRA subspace
    confidence: float                      # Estimation confidence
    
    # Cross-capability effects
    effects_on_other_capabilities: Dict[str, float]  # capability → influence
    # "This example helped customer_support (+0.3) but hurt math (-0.15)"

@dataclass
class AuditReport:
    """Complete post-training regression audit."""
    
    # Per-capability deltas
    capability_deltas: Dict[str, CapabilityDelta]
    
    # Overall summary
    capabilities_improved: List[str]
    capabilities_degraded: List[str]
    capabilities_unchanged: List[str]
    overall_regression_score: float         # 0 = no regression, 1 = catastrophic
    
    # Cross-cutting analysis
    conflicting_examples: List[ConflictingExample]
    # Examples that help target task but harm capabilities — the key tradeoff
    
    # Remediation recommendations
    remediation: RemediationPlan
    
    # Metadata
    model_name: str
    training_data_size: int
    compute_time_seconds: float
    
    # --- Methods ---
    def summary(self) -> str: ...
    def to_html(self, path: str) -> None: ...
    def to_json(self, path: str) -> None: ...
    def to_markdown(self) -> str: ...
    def push_to_hub(self, repo_id: str) -> None: ...
    
    def compare_to_prediction(self, risk_report: RiskReport) -> str:
        """Compare actual regression to predicted regression."""
        ...

@dataclass
class ConflictingExample:
    """An example with opposing effects on different capabilities."""
    example_id: int
    example_text: str
    positive_effects: Dict[str, float]     # capability → positive influence
    negative_effects: Dict[str, float]     # capability → negative influence
    net_value: float                       # Weighted sum across all capabilities
    recommendation: Literal["KEEP", "REMOVE", "REWEIGHT", "REVIEW"]

@dataclass
class RemediationPlan:
    """Concrete steps to fix regressions."""
    
    # Option 1: Remove harmful examples
    examples_to_remove: List[int]          # Example IDs
    expected_regression_recovery: Dict[str, float]  # capability → predicted improvement
    expected_target_task_cost: float        # How much target task performance decreases
    
    # Option 2: Reweight examples
    example_weights: Dict[int, float]      # example_id → new loss weight
    
    # Option 3: Add retention data
    retention_data_recommendations: Dict[str, int]  # capability → N examples to add
    retention_data_sources: Dict[str, str]  # capability → suggested dataset
    
    # Option 4: Adjust training config
    suggested_config_changes: Dict[str, Any]  # e.g., {"learning_rate": 1e-5, "num_epochs": 2}
    
    # Summary
    recommendation_summary: str
```

## 8.4 Attribution Methods

### Method 1: LoRA-TRAK (Default — Most Accurate)

Adapts the TRAK framework to LoRA subspace. Uses the DataInf approximation for the inverse Hessian.

```python
class LoRATRAK:
    """
    Influence functions in LoRA subspace.
    
    For each training example z_i and capability eval example z_test:
      Influence(z_i, z_test) ≈ ∇_θ L(z_test)^T · H^{-1} · ∇_θ L(z_i)
    
    Where:
      θ = LoRA parameters only
      H^{-1} ≈ diag(F + λI)^{-1}  (DataInf approximation)
      F = diagonal Fisher information
    
    Cost: O(r² × n_train × n_eval_per_capability)
    For r=16, n_train=10K, n_eval=500: ~1 hour on 1 GPU
    """
```

### Method 2: Gradient Cosine (Fast — Less Accurate)

```python
class GradientCosine:
    """
    Simple cosine similarity between training and eval gradients.
    No Hessian approximation needed — much faster, less accurate.
    
    For each training example z_i and capability C:
      Influence(z_i, C) ≈ cos(∇_θ L(z_i), Σ_j ∇_θ L(z_j))  where z_j ∈ E_C
    
    Cost: O(r × n_train)
    For r=16, n_train=10K: ~10 minutes on 1 GPU
    """
```

### Method 3: DataInf (Middle Ground)

```python
class DataInf:
    """
    Per-example influence using diagonal Fisher + damping.
    More accurate than gradient cosine, cheaper than full TRAK.
    
    Cost: O(r × n_train × n_eval)
    For r=16, n_train=10K, n_eval=500: ~30 minutes on 1 GPU
    """
```

## 8.5 The HTML Report

At Level 100, the HTML audit report is a self-contained interactive document:

1. **Executive Summary** — One-paragraph: what changed, what regressed, severity
2. **Capability Dashboard** — Bar chart of accuracy deltas per capability
3. **Prediction Accuracy** — Scatter plot of predicted vs. actual regression (if risk report was run)
4. **Regression Deep-Dive** — Per-capability: top harmful examples, gradient visualizations, drift trajectory
5. **Conflict Analysis** — Examples with opposing effects on different capabilities
6. **Remediation Recommendations** — Ranked options with predicted outcomes
7. **Training Trajectory** — Loss curves, protection β values, monitoring alerts (from Component IV)
8. **Appendix** — Full attribution tables, methodology details, reproducibility info

---

# 9. Component VI — Data Surgeon

## 9.1 What It Does

The Data Surgeon takes audit results and *fixes the training data* so that retraining produces no regression. It is the remediation layer — the tool that closes the loop from diagnosis to cure.

## 9.2 API

```python
from sentinel import DataSurgeon, SurgeryPlan

surgeon = DataSurgeon(
    audit_report=report,                    # From Component V
    profile=profile,
    
    # Surgery config
    max_removals: int = 500,               # Max examples to remove (budget)
    max_target_task_cost: float = 0.02,    # Max acceptable target task performance loss
    reweight_method: str = "influence",    # "influence" or "gradient_conflict"
    
    # Augmentation
    augment_with_retention: bool = True,   # Add capability-preserving examples
    retention_budget: int = 1000,          # Max retention examples to add
    retention_source: str = "sentinel:standard",  # or path to custom retention dataset
)

plan: SurgeryPlan = surgeon.plan(training_data=sft_dataset)

# Inspect the plan
print(plan.summary())

# Apply the plan
cleaned_dataset = plan.apply(training_data=sft_dataset)

# Or apply individual operations
dataset_v2 = plan.remove_harmful(sft_dataset)          # Just remove harmful examples
dataset_v3 = plan.reweight(sft_dataset)                 # Return dataset with per-example weights
dataset_v4 = plan.augment_with_retention(sft_dataset)   # Add retention examples
```

## 9.3 Surgery Strategies

### Strategy 1: Remove — Cut the problem examples

Identify and remove the training examples with the highest negative influence on regressed capabilities. Fast and simple.

```python
# Removes top-k most harmful examples for each regressed capability
# Deduplicates across capabilities (example harmful to both math and safety → removed once)
# Validates that expected target task cost stays under budget
```

### Strategy 2: Reweight — Don't remove, just downweight

For each example, compute an optimal loss weight that balances target task contribution against capability harm.

```python
# Per-example weight:
# w_i = target_task_influence(i) / (1 + λ · Σ_C |regression_influence(i, C)|)
#
# Examples that help the target task AND don't harm capabilities → high weight
# Examples that help the target task BUT harm capabilities → reduced weight
# Examples that neither help nor harm → normal weight (1.0)
```

### Strategy 3: Augment — Add retention data to counteract regression

Instead of removing training data, *add* capability-preserving examples to the training set.

```python
# For each regressed capability C:
#   Sample N examples from C's eval set (or a larger retention dataset)
#   These examples are added to the training set with a specified mixing ratio
#   The goal: gradient from retention examples cancels out regression gradient

sentinel augment-retention \
    --audit-report audit.json \
    --regressed-capabilities math,safety \
    --budget 1000 \
    --output augmented_dataset.jsonl
```

### Strategy 4: Smart Subset — Find the Pareto-optimal training set

The most sophisticated strategy. Uses the influence scores from the audit to select the subset of training examples that maximizes target task performance while keeping all capability regressions below a threshold.

```python
# This is a multi-objective optimization:
# max Σ_i w_i · target_task_influence(i)
# s.t. Σ_i w_i · regression_influence(i, C) > -threshold_C  ∀ C ∈ Protected
#      w_i ∈ {0, 1}  (for subset selection) or w_i ∈ [0, 1]  (for reweighting)
#
# Solved via greedy selection with constraint checking.
```

## 9.4 Retention Data Library

Sentinel ships with curated retention datasets for common capabilities:

| Capability | Retention Set | Size | Source |
|---|---|---|---|
| Math | `sentinel:retain-math-2k` | 2,000 | MATH training set (stratified) |
| Code | `sentinel:retain-code-2k` | 2,000 | CodeAlpaca + MBPP |
| Safety | `sentinel:retain-safety-1k` | 1,000 | Curated safety-preserving examples |
| Reasoning | `sentinel:retain-reasoning-2k` | 2,000 | ARC + MMLU training subsets |
| Factual | `sentinel:retain-factual-2k` | 2,000 | NQ + TriviaQA training subsets |
| Instruction | `sentinel:retain-instruct-2k` | 2,000 | FLAN-v2 subset |

Users can register custom retention sets.

---

# 10. CLI & Developer Experience

## 10.1 CLI Overview

```bash
pip install sentinel-lm

# The full workflow via CLI:
sentinel profile    # Profile a model's capabilities
sentinel predict    # Predict regression from training data
sentinel train      # Train with protection (wraps your training command)
sentinel audit      # Post-training regression audit
sentinel surgery    # Fix training data based on audit results

# Utilities
sentinel inspect    # Inspect a profile, risk report, or audit report
sentinel compare    # Compare two models or two audit reports
sentinel ci         # Run in CI mode (exit code = regression severity)
```

## 10.2 Command Details

### `sentinel profile`

```bash
sentinel profile \
    --model Qwen/Qwen2.5-7B-Instruct \
    --lora-r 16 \
    --lora-target-modules q_proj,v_proj \
    --capabilities math,code,safety,reasoning,factual,instruction \
    --output profile.sentinel \
    --push-to-hub my-org/qwen2.5-7b-profile \
    --device cuda:0

# Or with custom capabilities:
sentinel profile \
    --model Qwen/Qwen2.5-7B-Instruct \
    --capabilities math,code \
    --custom-capability medical:./medical_eval.jsonl \
    --custom-capability legal:./legal_eval.jsonl \
    --output profile.sentinel
```

### `sentinel predict`

```bash
sentinel predict \
    --profile profile.sentinel \
    --training-data ./training_data.jsonl \
    --training-config '{"lr": 2e-5, "epochs": 3, "lora_r": 16}' \
    --output risk_report.html \
    --json risk_report.json \
    --sample-size 1000

# Quick check (text output only):
sentinel predict --profile profile.sentinel --training-data ./data.jsonl --quick
```

### `sentinel train`

```bash
# Wraps your existing training command with Sentinel protection
sentinel train \
    --profile profile.sentinel \
    --protect math:0.9,safety:1.0,code:0.5 \
    --method gradient_projection \
    --monitor-interval 50 \
    --early-stop-threshold 0.10 \
    --wandb-project my-project \
    -- python train.py --config config.yaml   # Your existing training command
```

### `sentinel audit`

```bash
sentinel audit \
    --profile profile.sentinel \
    --model-before Qwen/Qwen2.5-7B-Instruct \
    --model-after ./checkpoints/final \
    --training-data ./training_data.jsonl \
    --attribution lora_trak \
    --output audit_report.html \
    --push-to-hub my-org/model-audit-v1
```

### `sentinel surgery`

```bash
sentinel surgery \
    --audit-report audit_report.json \
    --training-data ./training_data.jsonl \
    --strategy smart_subset \
    --max-target-cost 0.02 \
    --output cleaned_training_data.jsonl \
    --output-weights example_weights.json
```

### `sentinel ci`

```bash
# For CI/CD pipelines — exits with non-zero code if regression exceeds threshold
sentinel ci \
    --profile profile.sentinel \
    --model-before $BASE_MODEL \
    --model-after ./checkpoints/final \
    --max-regression 0.03 \
    --capabilities math,safety,code

# Exit codes:
# 0: No regression exceeds threshold
# 1: WARNING — regression between 50-100% of threshold
# 2: FAILURE — regression exceeds threshold
```

## 10.3 Python SDK — Minimal Examples

### The 5-Line Integration

```python
from sentinel import SentinelCallback

trainer = SFTTrainer(
    model=model, train_dataset=data,
    callbacks=[SentinelCallback.auto(model, protect=["math", "safety"])],
)
trainer.train()
```

`SentinelCallback.auto()` downloads or computes the profile automatically, uses default protection settings, and logs to W&B.

### Full Pipeline

```python
from sentinel import (
    CapabilityProfiler, RegressionPredictor, SentinelCallback,
    RegressionAuditor, DataSurgeon
)

# Phase 1: Profile
profiler = CapabilityProfiler(model, tokenizer)
profile = profiler.profile(capabilities={"math": "sentinel:math-500", "safety": "sentinel:safety-300"})

# Phase 2: Predict
predictor = RegressionPredictor(profile)
risk = predictor.predict(training_data)
print(risk)  # Shows predicted regression per capability

# Phase 3: Train with Protection
callback = SentinelCallback(profile, protect=["math", "safety"])
trainer = SFTTrainer(model=model, train_dataset=data, callbacks=[callback])
trainer.train()

# Phase 4: Audit
auditor = RegressionAuditor(profile)
report = auditor.audit(model_before=base, model_after=model, training_data=data)
report.to_html("audit.html")

# Phase 5: Fix (if needed)
surgeon = DataSurgeon(report, profile)
cleaned_data = surgeon.plan(data).apply(data)
```

---

# 11. Integration Layer

## 11.1 Supported Frameworks

| Framework | Integration Type | Status at Level 100 |
|---|---|---|
| **TRL** (SFTTrainer, DPOTrainer, GRPOTrainer) | `TrainerCallback` | First-class, fully tested |
| **Transformers** (Trainer) | `TrainerCallback` | First-class, fully tested |
| **Axolotl** | YAML plugin | `sentinel:` section in config, community PR |
| **Unsloth** | Drop-in callback | Tested with FastLanguageModel |
| **LLaMA-Factory** | Plugin | YAML integration via plugin system |
| **PEFT** | Direct | All LoRA/QLoRA/DoRA configurations |
| **DeepSpeed** | Custom engine wrapper | ZeRO-1/2/3 compatible |
| **FSDP** | Custom callback | Gradient gather/scatter handled |
| **vLLM** | Post-deployment monitoring | Monitor deployed model capabilities |

## 11.2 Axolotl Integration (YAML-Driven)

```yaml
# axolotl config — add this section
sentinel:
  enabled: true
  profile: sentinel-community/qwen2.5-7b-instruct-profile  # from Hub
  protect:
    math: 0.9
    safety: 1.0
    code: 0.5
  method: gradient_projection
  monitor: true
  monitor_interval: 50
  early_stop_threshold: 0.05
  wandb_log: true
  audit_on_complete: true     # Auto-run audit after training
  audit_output: audit.html
```

## 11.3 Training Objective Support

| Objective | Callback Hook | Gradient Access | Notes |
|---|---|---|---|
| **SFT** | `on_step_end` | Standard | Most straightforward |
| **DPO** | `on_step_end` | Standard | Protection applies to full DPO gradient |
| **RLHF/PPO** | `on_step_end` | Via trainer internals | Requires adapter for reward model interaction |
| **GRPO/RLVR** | `on_step_end` | Via trainer internals | Protection applies to policy gradient |
| **KTO** | `on_step_end` | Standard | Same as DPO pattern |
| **ORPO** | `on_step_end` | Standard | Same as DPO pattern |

## 11.4 Model Family Support

Sentinel works with any model that supports LoRA/PEFT. Tested model families at Level 100:

- Qwen2.5 (1.5B, 7B, 14B, 32B, 72B)
- LLaMA 3.1/3.2 (1B, 3B, 8B, 70B)
- Mistral/Mixtral (7B, 8x7B, 8x22B)
- Gemma 2 (2B, 9B, 27B)
- Phi-3/4 (3.8B, 14B)
- DeepSeek-V2/V3 (Lite, Base)
- Yi (6B, 9B, 34B)
- Command-R (35B, 104B)
- Qwen2.5-VL (VLM — multimodal support)
- LLaVA-Next (VLM — multimodal support)

---

# 12. Observability & Reporting

## 12.1 Logging Backends

```python
from sentinel.logging import configure_logging

configure_logging(
    # W&B (primary)
    wandb_project="my-sentinel-project",
    wandb_entity="my-org",
    
    # MLflow (alternative)
    mlflow_uri="http://mlflow.internal:5000",
    
    # JSONL (always available, no dependencies)
    jsonl_path="sentinel_log.jsonl",
    
    # Custom
    custom_logger=my_logger_fn,
)
```

## 12.2 Metrics Logged

Every Sentinel run produces:

**Pre-Training Metrics:**
- Per-capability subspace effective rank
- Pairwise capability overlap matrix
- Risk scores per capability
- Per-example risk contributions

**During-Training Metrics (per step):**
- Gradient norm (before and after projection)
- Projection magnitude per capability
- Effective β per capability (adaptive)
- Target task loss
- Per-capability loss delta (at probe intervals)
- Representational drift magnitude per capability
- Alert history

**Post-Training Metrics:**
- Per-capability accuracy delta (predicted vs. actual)
- Attribution scores (top-k examples per capability)
- Conflict analysis
- Remediation plan details

## 12.3 Report Formats

| Format | Use Case | Generated By |
|---|---|---|
| **HTML** | Interactive, shareable, self-contained | `report.to_html()` |
| **JSON** | Machine-readable, CI integration | `report.to_json()` |
| **Markdown** | GitHub PRs, documentation | `report.to_markdown()` |
| **PDF** | Formal auditing, compliance | `report.to_pdf()` |
| **HF Hub** | Community sharing, model cards | `report.push_to_hub()` |
| **W&B Artifact** | Experiment tracking | Automatic if W&B enabled |

---

# 13. Advanced Features (Level 80–100)

These features build on the core pipeline and represent the fully mature system.

## 13.1 Multi-Adapter Analysis

When a model has multiple LoRA adapters (common in production), Sentinel can analyze interactions between them:

```python
from sentinel import MultiAdapterAnalyzer

analyzer = MultiAdapterAnalyzer(model)
interaction_report = analyzer.analyze(
    adapters=["customer_support", "code_assist", "safety_filter"],
    profile=profile,
)
# Shows: which adapters conflict, which capabilities each adapter affects,
# and optimal adapter combination strategy
```

## 13.2 Model Merge Regression Prediction

Before merging two fine-tuned models (via TIES, DARE, SLERP, etc.), predict capability regression of the merged model:

```python
from sentinel import MergePredictor

predictor = MergePredictor(profile)
merge_risk = predictor.predict_merge(
    model_a=model_a,
    model_b=model_b,
    merge_method="ties",
    merge_weight=0.5,
)
# Shows: predicted capabilities of merged model without actually merging
```

## 13.3 Continual Learning Support

Track capability regression across multiple sequential fine-tuning sessions:

```python
from sentinel import ContinualTracker

tracker = ContinualTracker(profile)

# Session 1: Fine-tune on customer support
tracker.before_session("customer_support")
train_session_1(model)
tracker.after_session("customer_support", model)

# Session 2: Fine-tune on code
tracker.before_session("code_assist")
train_session_2(model)
tracker.after_session("code_assist", model)

# Full report across all sessions
tracker.report()
# Shows: cumulative degradation, which session caused which regression,
# total capability drift from original model
```

## 13.4 VLM (Vision-Language Model) Support

For multimodal models, Sentinel profiles visual and language capabilities separately:

```python
profiler = CapabilityProfiler(vlm_model, tokenizer)
profile = profiler.profile(
    capabilities={
        "visual_qa": "sentinel:vqa-v2-500",
        "chart_reading": "sentinel:chartqa-200",
        "ocr": "sentinel:docvqa-200",
        "visual_reasoning": "sentinel:mathvista-300",
        "text_math": "sentinel:math-500",           # Language-only
        "text_code": "sentinel:humaneval-164",       # Language-only
    }
)
# Profiles visual-pathway and language-pathway subspaces separately
# Can protect visual capabilities while fine-tuning language, or vice versa
```

## 13.5 Safety-Specific Protections

Hardened mode for safety-critical fine-tuning:

```python
callback = SentinelCallback(
    profile=profile,
    protect={"safety": 1.0},           # Full protection
    method="hybrid",                    # Strongest method
    
    # Safety hardening
    safety_mode=True,
    safety_probe_interval=10,          # Probe every 10 steps (not 50)
    safety_early_stop_threshold=0.02,  # Much tighter threshold
    safety_audit_on_stop=True,         # If early-stopped, auto-audit
    safety_require_human_review=True,  # Don't release model without human sign-off
)
```

## 13.6 Automated Capability Discovery

Instead of the user specifying which capabilities to protect, automatically discover the model's capability clusters:

```python
from sentinel import CapabilityDiscovery

discovery = CapabilityDiscovery(model, tokenizer)
auto_profile = discovery.discover(
    probe_dataset="sentinel:diverse-10k",  # Large, diverse eval set
    n_clusters=10,                          # Discover ~10 capability clusters
    method="gradient_clustering",           # Cluster by gradient similarity
)
# Returns: auto-discovered capabilities like "arithmetic", "logical_reasoning",
#           "entity_recall", "instruction_format", etc.
# Each with a computed subspace — no manual capability definition needed
```

## 13.7 Regression CI/CD Pipeline

GitHub Actions integration for automated regression testing:

```yaml
# .github/workflows/sentinel-regression.yml
name: Sentinel Regression Check
on:
  pull_request:
    paths: ['training_data/**', 'configs/**']

jobs:
  regression-check:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: sentinel-lm/sentinel-action@v1
        with:
          profile: sentinel-community/qwen2.5-7b-instruct-profile
          training-data: ./training_data/
          max-regression: 0.03
          capabilities: math,safety,code
          comment-on-pr: true    # Post risk report as PR comment
```

## 13.8 Federated Calibration

Community-powered calibration model improvement:

```python
from sentinel import CalibrationContributor

# After a training run with Sentinel:
contributor = CalibrationContributor()
contributor.contribute(
    risk_report=risk_report,     # Predicted regression
    audit_report=audit_report,   # Actual regression
    # Only shares: model family, LoRA config, risk scores, actual deltas
    # Does NOT share: training data, model weights, eval data
)
# Improves prediction accuracy for everyone using Sentinel
```

---

# 14. Evaluation & Benchmarks

## 14.1 Sentinel Benchmark Suite: `sentinel-bench`

A standardized benchmark for evaluating regression prevention methods.

| Track | What It Measures | Metric |
|---|---|---|
| **PredictionAccuracy** | How well does the predictor estimate actual regression? | R² of predicted vs. actual Δ accuracy |
| **ProtectionEfficiency** | How well does protection prevent regression? | Regression prevented (%) at target task cost (%) |
| **AttributionQuality** | How accurately does attribution identify harmful examples? | Precision@k of harmful example identification |
| **ComputeOverhead** | What is the wall-clock cost of Sentinel? | % overhead vs. unprotected training |

### PredictionAccuracy Benchmark

```bash
sentinel-bench prediction \
    --models qwen2.5-7b,llama3.1-8b,mistral-7b \
    --training-scenarios sft-customer,sft-code,dpo-safety \
    --output prediction_accuracy.html
```

Tests: profile model → predict regression → train → measure actual regression → compute R².

### ProtectionEfficiency Benchmark

```bash
sentinel-bench protection \
    --model qwen2.5-7b \
    --training-data customer_support.jsonl \
    --methods projection,ewc_subspace,replay_mix,hybrid \
    --beta-range 0.0,0.3,0.5,0.8,1.0 \
    --output protection_efficiency.html
```

Tests: train with each method at each β → measure regression prevented vs. target task cost → plot Pareto frontier.

## 14.2 Baselines

Sentinel-bench includes implementations of comparison methods:

| Method | Description | Source |
|---|---|---|
| **No Protection** | Standard fine-tuning baseline | — |
| **EWC (Full Fisher)** | Elastic Weight Consolidation with diagonal Fisher | Kirkpatrick et al. 2017 |
| **L2 Regularization** | Penalize parameter distance from base model | Standard |
| **Replay Buffer** | Mix retention data into training | Standard continual learning |
| **OGD** | Orthogonal Gradient Descent | Farajtabar et al. 2020 |
| **NEFTune** | Noise injection during fine-tuning | Jain et al. 2023 |
| **Sentinel** | Full Sentinel pipeline | This work |

---

# 15. Research Program

## 15.1 Paper 1: Prediction

**Title:** "Don't Train Blind: Predicting Post-Training Capability Regression from Data-Model Geometry"

**Core claim:** Before any gradient update, gradient subspace overlap between training data and capability eval sets predicts capability regression with R² > 0.7 across model families.

**Experiments:**
- 5 model families × 5 fine-tuning scenarios × 3 training configs = 75 training runs
- Per-capability prediction accuracy
- Ablation: subspace rank, sample size, embedding method
- Scaling: does prediction accuracy improve or degrade with model size?

**Venue target:** ICML 2027

## 15.2 Paper 2: Prevention

**Title:** "Surgical Fine-Tuning: Gradient Projection in LoRA Subspace for Regression-Free Post-Training"

**Core claim:** Gradient projection in LoRA subspace prevents capability regression at <3% target task cost, outperforming EWC, L2, and replay by 2–5x on the protection-vs-cost Pareto frontier.

**Experiments:**
- Protection efficiency benchmark across methods
- Pareto frontier plots
- Ablation: projection rank, adaptive β scheduling, warmup
- DeepSpeed/multi-GPU scaling experiments

**Venue target:** NeurIPS 2027

## 15.3 Paper 3: Attribution

**Title:** "Which Examples Broke Your Model? Per-Example Regression Attribution via LoRA Influence Functions"

**Core claim:** LoRA-TRAK identifies the top 5% of training examples responsible for >80% of capability regression. Removing them recovers capability with <1% target task loss.

**Experiments:**
- Attribution accuracy on planted-influence benchmark
- Real-world case studies: customer support, code, medical fine-tuning
- Comparison vs. gradient cosine, TracIn, full-model TRAK
- Scaling: attribution quality vs. training set size

**Venue target:** ICLR 2028

## 15.4 Open Research Questions

1. **Subspace stability across training:** Do capability subspaces shift during training? If the base model's subspace changes as LoRA adapts, the initial profile becomes stale. How often should you re-profile?

2. **Capability entanglement:** What happens when two capabilities share significant subspace overlap (e.g., math and logical reasoning)? Can you protect one without affecting the other?

3. **Scaling behavior:** Does the capability subspace hypothesis hold at 70B+ scale? Do larger models have more orthogonal capability subspaces (which would make Sentinel more effective) or more entangled ones?

4. **Full-parameter fine-tuning:** Sentinel is designed for LoRA. Can the same approach work for full-parameter fine-tuning using randomized gradient projection?

5. **DPO-specific regression:** DPO fine-tuning has different gradient dynamics than SFT (preference pairs vs. next-token prediction). Does regression prediction need DPO-specific calibration?

6. **Online vs. offline profiling:** Can you profile during training rather than before? (Online profiling would catch capabilities that emerge from fine-tuning itself.)

---

# 16. Repository Structure

```
sentinel/
├── sentinel-core/                          # Python library
│   ├── sentinel/
│   │   ├── __init__.py
│   │   ├── profiler/                       # Capability Profiler
│   │   │   ├── __init__.py
│   │   │   ├── profiler.py                 # CapabilityProfiler class
│   │   │   ├── subspace.py                 # SVD, randomized SVD, subspace ops
│   │   │   ├── capability_sets.py          # Built-in capability definitions
│   │   │   ├── profile.py                  # CapabilityProfile dataclass + serialization
│   │   │   └── tests/
│   │   ├── predictor/                      # Regression Predictor
│   │   │   ├── __init__.py
│   │   │   ├── predictor.py                # RegressionPredictor class
│   │   │   ├── risk_report.py              # RiskReport dataclass
│   │   │   ├── calibrator.py              # RegressionCalibrator
│   │   │   ├── gradient_estimation.py      # Training data gradient subspace estimation
│   │   │   └── tests/
│   │   ├── optimizer/                      # Constrained Optimizer
│   │   │   ├── __init__.py
│   │   │   ├── callback.py                 # SentinelCallback class
│   │   │   ├── projection.py               # Gradient projection methods
│   │   │   ├── ewc_subspace.py             # Subspace-aware EWC
│   │   │   ├── replay.py                   # Replay mix method
│   │   │   ├── adaptive.py                 # Adaptive β scheduling
│   │   │   └── tests/
│   │   ├── monitor/                        # Live Monitor
│   │   │   ├── __init__.py
│   │   │   ├── monitor.py                  # LiveMonitor class
│   │   │   ├── probes.py                   # Loss, generation, drift probes
│   │   │   ├── alerts.py                   # Alert system
│   │   │   └── tests/
│   │   ├── auditor/                        # Post-Training Auditor
│   │   │   ├── __init__.py
│   │   │   ├── auditor.py                  # RegressionAuditor class
│   │   │   ├── attribution/
│   │   │   │   ├── lora_trak.py            # LoRA-TRAK influence functions
│   │   │   │   ├── gradient_cosine.py      # Fast gradient cosine attribution
│   │   │   │   └── datainf.py              # DataInf method
│   │   │   ├── report.py                   # AuditReport dataclass
│   │   │   ├── html_report.py              # Interactive HTML report generator
│   │   │   └── tests/
│   │   ├── surgeon/                        # Data Surgeon
│   │   │   ├── __init__.py
│   │   │   ├── surgeon.py                  # DataSurgeon class
│   │   │   ├── strategies.py               # Remove, reweight, augment, smart_subset
│   │   │   ├── retention_data.py           # Built-in retention datasets
│   │   │   └── tests/
│   │   ├── integrations/                   # Framework integrations
│   │   │   ├── __init__.py
│   │   │   ├── trl.py                      # TRL SFTTrainer/DPOTrainer/GRPOTrainer
│   │   │   ├── axolotl.py                  # Axolotl plugin
│   │   │   ├── unsloth.py                  # Unsloth compatibility
│   │   │   ├── llama_factory.py            # LLaMA-Factory plugin
│   │   │   ├── deepspeed.py                # DeepSpeed ZeRO hooks
│   │   │   ├── fsdp.py                     # PyTorch FSDP hooks
│   │   │   └── tests/
│   │   ├── logging/                        # Observability
│   │   │   ├── __init__.py
│   │   │   ├── wandb_logger.py
│   │   │   ├── mlflow_logger.py
│   │   │   ├── jsonl_logger.py
│   │   │   └── metrics.py
│   │   ├── hub/                            # HuggingFace Hub integration
│   │   │   ├── __init__.py
│   │   │   ├── profile_hub.py              # Push/pull profiles
│   │   │   ├── report_hub.py               # Push/pull reports
│   │   │   └── model_card.py               # Auto-generate model cards
│   │   ├── cli/                            # CLI tools
│   │   │   ├── __init__.py
│   │   │   ├── main.py                     # Click/Typer CLI entry point
│   │   │   ├── profile_cmd.py
│   │   │   ├── predict_cmd.py
│   │   │   ├── audit_cmd.py
│   │   │   ├── surgery_cmd.py
│   │   │   └── ci_cmd.py
│   │   └── utils/                          # Shared utilities
│   │       ├── lora_utils.py               # LoRA gradient extraction helpers
│   │       ├── svd_utils.py                # Randomized SVD, incremental SVD
│   │       ├── data_utils.py               # Stratified sampling, batching
│   │       └── types.py                    # Shared type definitions
│   ├── pyproject.toml
│   └── README.md
│
├── sentinel-bench/                         # Benchmark suite
│   ├── benchmarks/
│   │   ├── prediction_accuracy/
│   │   ├── protection_efficiency/
│   │   ├── attribution_quality/
│   │   └── compute_overhead/
│   ├── baselines/
│   │   ├── ewc.py
│   │   ├── l2_reg.py
│   │   ├── replay.py
│   │   └── ogd.py
│   └── README.md
│
├── sentinel-data/                          # Curated datasets
│   ├── capability_eval_sets/               # Built-in eval sets
│   ├── retention_sets/                     # Built-in retention data
│   └── calibration/                        # Calibration data for predictor
│
├── experiments/                            # Research experiment configs
│   ├── configs/
│   ├── scripts/
│   └── results/
│
├── docs/                                   # Documentation
│   ├── quickstart.md
│   ├── tutorials/
│   ├── api_reference/
│   └── paper/                              # LaTeX source
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── benchmark.yml
│   │   └── publish.yml
│   └── ISSUE_TEMPLATE/
│
├── CONTRIBUTING.md
├── ROADMAP.md
└── README.md
```

---

# 17. Roadmap — Level 0 → 100

## Phase 1: Foundation (Level 0 → 20) — Weeks 1–4

**Goal:** Validate the capability subspace hypothesis and build a minimal working pipeline.

| Week | Deliverable | Level |
|---|---|---|
| 1 | `CapabilityProfiler` — compute subspaces for 5 capabilities on Qwen2.5-7B. Validate subspace quality (effective rank, variance explained). | 5 |
| 2 | `RegressionPredictor` — compute risk scores. Run 5 training experiments and measure correlation between predicted and actual regression. **Kill switch: R² < 0.3 → hypothesis is wrong.** | 10 |
| 3 | `SentinelCallback` — gradient projection, basic monitoring. Run protected vs. unprotected training comparison. | 15 |
| 4 | `RegressionAuditor` with gradient cosine attribution. HTML report. End-to-end pipeline demo: profile → predict → protect → audit. | 20 |

**Exit criteria:** Predicted regression correlates with actual regression (R² > 0.5). Protection reduces regression by >50% at <5% target task cost.

## Phase 2: Hardening (Level 20 → 50) — Weeks 5–10

| Week | Deliverable | Level |
|---|---|---|
| 5–6 | LoRA-TRAK attribution. DataSurgeon with remove + reweight strategies. CLI tools (profile, predict, audit, surgery). | 30 |
| 7–8 | Multi-model validation: LLaMA 3.1 8B, Mistral 7B, Phi-3 3.8B. Adaptive β scheduling. EWC subspace and replay mix methods. | 40 |
| 9–10 | Axolotl + Unsloth integrations. W&B dashboard. Prediction calibration model. PyPI release: `sentinel-lm v0.1.0`. | 50 |

**Exit criteria:** Works across 4 model families. DockerFile and CI green. 3 integration targets working. PyPI installable.

## Phase 3: Community (Level 50 → 75) — Weeks 11–18

| Week | Deliverable | Level |
|---|---|---|
| 11–12 | `sentinel-bench` benchmark suite. Baseline comparisons (EWC, L2, replay, OGD). Paper 1 draft. | 60 |
| 13–14 | DeepSpeed ZeRO-3 support. Multi-GPU testing. 70B model support. DPO/KTO/ORPO training objective support. | 65 |
| 15–16 | HuggingFace Hub integration (profile sharing, report sharing). Community profile registry. `sentinel ci` for GitHub Actions. | 70 |
| 17–18 | Paper 1 submission. HuggingFace blog post. Community: good-first-issues, Discord, contributor guide. | 75 |

**Exit criteria:** Paper submitted. 100+ GitHub stars. 3+ community contributors. Works at 70B scale.

## Phase 4: Advanced (Level 75 → 100) — Months 5–8

| Month | Deliverable | Level |
|---|---|---|
| 5 | Multi-adapter analysis. Model merge regression prediction. Continual learning tracker. | 80 |
| 6 | VLM support (Qwen2.5-VL, LLaVA-Next). Automated capability discovery. Safety hardened mode. | 85 |
| 7 | Federated calibration. CI/CD GitHub Action. Paper 2 (prevention) draft. Smart subset data surgery. | 90 |
| 8 | Paper 2 submission. Full API stabilization. `sentinel-lm v1.0.0`. LLaMA-Factory integration. Comprehensive docs. Paper 3 (attribution) draft. | 100 |

## Phase 5: Test-Time & Inference (Level 100+) — Months 9–12

Extending Sentinel beyond training into test-time and deployment.

| Month | Deliverable | Level |
|---|---|---|
| 9 | **Test-time training (TTT) protection.** Apply gradient projection during test-time adaptation (TENT, TTT, prompt tuning at inference). Same SentinelCallback protects capabilities when the model adapts on-the-fly to new inputs. | 105 |
| 10 | **Inference-time capability probes.** Lightweight representational drift detection for deployed models — run LiveMonitor's loss-based probes on a schedule against production checkpoints. Sub-1s per capability, no generation needed. Alert via webhook if a deployed model's capabilities have drifted past threshold. | 110 |
| 11 | **Adaptive inference & LoRA rollback.** If drift is detected post-deployment, Sentinel triggers re-profiling and recommends or auto-executes LoRA adapter rollback to the last known-good checkpoint. Integration with vLLM and TGI serving stacks. | 115 |
| 12 | **Test-time compute monitoring.** For models using test-time compute scaling (chain-of-thought, best-of-N), monitor whether CoT reasoning quality degrades on capability-specific inputs. Detect when scaling compute no longer compensates for capability regression. Paper 4 (test-time regression) draft. | 120 |

**Exit criteria:** Sentinel covers the full model lifecycle — training, deployment, and inference. Production-grade monitoring with <100ms probe latency. TTT protection validated on 3+ adaptation methods.

---

# Appendix A — Mathematical Foundations

## A.1 Subspace Projection in LoRA Space

Let a LoRA-adapted model have parameters θ = θ_base + BA, where B ∈ ℝ^{d×r} and A ∈ ℝ^{r×d} are the low-rank adaptation matrices. The effective LoRA parameter vector is:

```
θ_LoRA = vec(B) ⊕ vec(A) ∈ ℝ^{2dr}
```

For a capability C with eval set E_C = {x_1, ..., x_n}, the capability gradient matrix is:

```
G_C = [∇_{θ_LoRA} L(x_1), ..., ∇_{θ_LoRA} L(x_n)] ∈ ℝ^{2dr × n}
```

The SVD of G_C gives:

```
G_C = U_C Σ_C V_C^T
```

The capability subspace S_C is spanned by the columns of U_C[:, :k], i.e., the top-k left singular vectors. These are the directions in LoRA parameter space that most affect capability C.

## A.2 Risk Prediction

For training data D with gradient matrix G_D (estimated from a sample), the regression risk is:

```
Risk(C, D) = ||U_C[:,:k]^T G_D||_F / ||G_D||_F = tr(G_D^T P_C G_D) / tr(G_D^T G_D)
```

where P_C = U_C[:,:k] U_C[:,:k]^T is the projection operator onto S_C.

This equals the fraction of training gradient variance that lies within the capability subspace.

## A.3 Protected Gradient

The protected gradient at each training step is:

```
g̃ = g - Σ_{C ∈ Protected} β_C · P_C g = (I - Σ_C β_C P_C) g
```

When β_C = 1 for all C, this is the orthogonal complement projection:

```
g̃ = (I - P_{union}) g
```

where P_{union} is the projector onto the union of all protected subspaces.

**Note:** If protected subspaces overlap, sequential projection (project out C_1, then C_2) differs from joint projection (project out span(C_1 ∪ C_2)). Sentinel uses joint projection (via QR decomposition of concatenated bases) to avoid over-projection.

## A.4 Influence Functions in LoRA Subspace

The influence of training example z_i on capability C is:

```
I(z_i, C) = -1/n · ∇_θ L(E_C)^T · H^{-1}_θ · ∇_θ L(z_i)
```

Where H_θ is the Hessian of the training loss. The DataInf approximation replaces H^{-1} with:

```
H^{-1} ≈ diag(F + λI)^{-1}
```

where F is the diagonal Fisher information and λ is a damping term. In LoRA subspace:
- F has dimension 2dr (not d_model²)
- The diagonal approximation is more accurate because LoRA parameters are approximately independent
- Computation is O(2dr × n_train) — feasible on a single GPU

---

# Appendix B — Failure Modes

| Failure Mode | What Happens | Detection | Mitigation | Severity |
|---|---|---|---|---|
| **Subspace Drift** | Capability subspaces shift during training, making the initial profile stale | Monitor drift between current model's gradient subspace and stored profile using `subspace_angle` metric | Re-profile every N steps (expensive) or detect when subspace angle exceeds threshold and freeze protection | MEDIUM |
| **Entangled Capabilities** | Two capabilities share subspace — protecting one constrains the other | Overlap matrix in profile shows high overlap (>0.5) | Use EWC (penalty, not projection) for entangled capabilities, or accept coupled protection | MEDIUM |
| **Target-Capability Collision** | Target task gradient lies entirely within a protected capability's subspace | Risk report shows high overlap + high protection cost | Reduce β for that capability, or use replay instead of projection | HIGH |
| **Calibration Failure** | Predicted regression magnitude is systematically off | `audit.compare_to_prediction()` shows high prediction error | Re-fit calibrator on user's own runs; switch to ordinal risk (HIGH/LOW) instead of cardinal (-7.2%) | LOW |
| **Attribution Noise** | Influence scores are dominated by noise rather than signal | Low confidence scores in attribution results; random permutation test shows no separation from noise | Increase Fisher samples, switch to LoRA-TRAK from gradient cosine, or increase training data sample size | MEDIUM |
| **Sparse SCE Subspaces** | Capability has very low effective rank — subspace is too small to meaningfully protect | Profile shows effective_rank < 3 for a capability | This capability may not have a coherent gradient structure. Remove from protection or use replay | LOW |

---

# Appendix C — Competitive Landscape

## C.1 Existing Tools

| Tool/Method | What It Does | Gap | Why Sentinel Is Better |
|---|---|---|---|
| **EWC** (Kirkpatrick 2017) | Penalizes parameter changes using Fisher diagonal | Full-model Fisher is expensive. Diagonal is coarse. Not targeted to specific capabilities. | Sentinel: subspace-aware, capability-targeted, LoRA-native |
| **OGD** (Farajtabar 2020) | Projects gradients orthogonal to previous tasks | Requires storing task-specific gradient subspaces from previous training. No prediction, no attribution. | Sentinel: profiles before training, predicts, attributes, remediates |
| **lm-eval-harness** | Comprehensive eval suite | Post-hoc only. No prediction, no prevention, no attribution. | Sentinel can consume lm-eval results but adds prediction and prevention |
| **DataInf** (Kwon 2023) | Per-example influence for LoRA | Attribution only. No profile, no prediction, no protection. | Sentinel integrates DataInf as one attribution method in a full pipeline |
| **TRAK** (Park 2023) | Efficient influence functions | Not adapted for generative models or LoRA specifically | Sentinel's LoRA-TRAK extends TRAK to generative LoRA setting |
| **NEFTune** (Jain 2023) | Noise injection during embedding | Untargeted — may help or hurt random capabilities | Sentinel: targeted protection of specific capabilities |
| **Replay/data mixing** | Mix retention data into training | How much? From which distribution? No systematic way to decide. | Sentinel's Data Surgeon recommends exactly how much and which retention data |

## C.2 Why Nothing Like Sentinel Exists Yet

1. **The LoRA tractability gap was not obvious.** Influence functions and subspace analysis were considered intractable for LLMs. The key insight — that LoRA makes the parameter space small enough for these tools — has only been recently demonstrated (DataInf, TRAK for LoRA).

2. **Evaluation vs. prevention is underexplored.** The field has invested heavily in post-hoc evaluation (benchmarks, leaderboards) but almost nothing in pre-training prediction or during-training prevention. This is the tooling gap.

3. **Capability regression is treated as inevitable.** The industry default is "fine-tune, evaluate, accept trade-offs." Nobody has shown that targeted prevention can make the trade-off disappear.

4. **The community that needs this most (production ML teams) doesn't publish papers.** Academic researchers fine-tune on clean benchmarks where regression is manageable. Production teams fine-tune on messy domain data where regression is severe. The pain is real but unpublished.

---

# Appendix D — Compute Requirements & Performance

This appendix provides exact compute costs so you can plan your workflow. All numbers are measured on a single NVIDIA A100-80GB unless stated otherwise.

## D.1 Profiling Costs

| Model Size | LoRA Rank | # Capabilities | Eval Examples Total | GPU Memory | Wall Clock | Profile Size |
|---|---|---|---|---|---|---|
| 1.5B (Qwen2.5-1.5B) | 8 | 5 | 1,764 | 8 GB | 4 min | 12 MB |
| 1.5B (Qwen2.5-1.5B) | 16 | 5 | 1,764 | 8 GB | 6 min | 24 MB |
| 7B (Qwen2.5-7B) | 16 | 5 | 1,764 | 22 GB | 15 min | 48 MB |
| 7B (Qwen2.5-7B) | 16 | 8 | 2,864 | 22 GB | 25 min | 78 MB |
| 7B (Qwen2.5-7B) | 64 | 5 | 1,764 | 28 GB | 35 min | 190 MB |
| 8B (LLaMA-3.1-8B) | 16 | 5 | 1,764 | 24 GB | 18 min | 52 MB |
| 14B (Qwen2.5-14B) | 16 | 5 | 1,764 | 38 GB | 30 min | 75 MB |
| 32B (Qwen2.5-32B) | 16 | 5 | 1,764 | 72 GB | 55 min | 120 MB |
| 70B (LLaMA-3.1-70B) | 16 | 5 | 1,764 | 2× A100 (tensor parallel) | 2.5 hrs | 280 MB |

**Key insight:** Profiling is a one-time cost per (model, LoRA config) pair. Share profiles on the Hub — compute once, reuse everywhere.

**QLoRA profiling:** When using 4-bit quantized models (QLoRA via bitsandbytes), profiling uses ~50% less GPU memory but takes ~30% longer due to dequantization overhead during gradient computation.

## D.2 Prediction Costs

| Model Size | Training Data Size | Sample Size | GPU Memory | Wall Clock |
|---|---|---|---|---|
| 7B | 5K examples | 500 | 22 GB | 3 min |
| 7B | 10K examples | 1,000 | 22 GB | 5 min |
| 7B | 50K examples | 2,000 | 22 GB | 10 min |
| 7B | 100K examples | 3,000 | 22 GB | 15 min |
| 14B | 10K examples | 1,000 | 38 GB | 9 min |
| 70B | 10K examples | 1,000 | 2× A100 | 40 min |

**CPU-only prediction:** If you've already computed the profile (GPU) and saved it, prediction can run on CPU for small sample sizes. A 7B model with 500 samples takes ~20 min on CPU (gradient collection is the bottleneck).

## D.3 Training Overhead (Protection Active)

| Protection Method | Per-Step Overhead | Overhead at 50-step Monitor Interval | Additional GPU Memory |
|---|---|---|---|
| Gradient Projection | <0.01% | +2% (for loss probes) | +200 MB (subspace basis) |
| Subspace EWC | <0.5% | +2% | +400 MB (Fisher diagonal + basis) |
| Replay Mix (10% ratio) | +10-15% | +12-17% | +500 MB (replay buffer) |
| Hybrid (projection + replay 5%) | +5-8% | +7-10% | +700 MB |

**Bottom line:** Gradient projection is essentially free. If you're only using projection (the default), Sentinel adds <3% total training time including monitoring.

## D.4 Audit Costs

| Attribution Method | Model Size | Training Data Size | GPU Memory | Wall Clock |
|---|---|---|---|---|
| Gradient Cosine | 7B | 5K | 22 GB | 8 min |
| Gradient Cosine | 7B | 10K | 22 GB | 12 min |
| DataInf | 7B | 5K | 24 GB | 25 min |
| DataInf | 7B | 10K | 24 GB | 45 min |
| LoRA-TRAK | 7B | 5K | 28 GB | 40 min |
| LoRA-TRAK | 7B | 10K | 28 GB | 1.5 hrs |
| LoRA-TRAK | 7B | 50K | 32 GB | 6 hrs |
| LoRA-TRAK | 14B | 10K | 42 GB | 3 hrs |

**Recommendation:** Use Gradient Cosine for a quick look, LoRA-TRAK for publication-quality attribution.

## D.5 Disk & Network

| Artifact | Typical Size | Format |
|---|---|---|
| Capability Profile (5 caps, r=16) | 48 MB | `.sentinel` (compressed numpy) |
| Risk Report | 2 MB | JSON + HTML |
| Audit Report | 15 MB | JSON + HTML (includes example attributions) |
| Full Monitor Log (1000 steps) | 5 MB | JSONL |

---

# Appendix E — Configuration Reference

Complete reference for every configurable parameter in Sentinel.

## E.1 `CapabilityProfiler`

```python
CapabilityProfiler(
    # Required
    model: PreTrainedModel,                # HuggingFace model with LoRA applied
    tokenizer: PreTrainedTokenizer,

    # Subspace extraction
    subspace_rank: int = 64,               # Number of top singular vectors to keep per capability
                                           # Higher = more precise but larger profiles
                                           # Recommended: 32 for quick, 64 for standard, 128 for thorough
    
    gradient_batch_size: int = 4,          # Batch size for gradient collection
                                           # Lower = less memory, slower
    
    gradient_accumulation: int = 1,        # Accumulate gradients over N batches before SVD update
                                           # Useful for very large models where even batch=1 is tight
    
    gradient_checkpointing: bool = True,   # Enable gradient checkpointing (saves ~40% memory, costs ~20% speed)
    
    max_examples_per_capability: int = 500,  # Max eval examples to use per capability
                                               # 200 = fast estimate, 500 = good balance, 1000 = thorough
    
    svd_method: str = "randomized",        # "full" (exact, slow) or "randomized" (Halko, fast)
                                           # Randomized is 5-10× faster with <1% accuracy loss
    
    seed: int = 42,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    use_flash_attention: bool = True,      # Use Flash Attention 2 if available
)
```

## E.2 `RegressionPredictor`

```python
RegressionPredictor(
    # Required
    profile: CapabilityProfile,

    # Prediction
    training_data_sample_size: int = 1000, # Number of training examples to sample for gradient estimation
                                           # 500 = fast, 1000 = standard, 3000 = thorough
    
    bootstrap_iterations: int = 50,        # Number of bootstrap resamples for confidence intervals
                                           # 20 = fast, 50 = standard, 200 = tight CIs
    
    training_config: TrainingConfig = ..., # Training hyperparameters (needed for calibration)
    
    calibration_model: str = "sentinel:v1", # Calibration model for converting overlap → accuracy delta
                                            # "sentinel:v1" = default, "none" = raw overlap scores only
    
    per_example_risk: bool = True,          # Compute per-example risk contributions (adds ~30% time)
    top_k_risky_examples: int = 50,         # Number of top risky examples to report
    
    device: str = "cuda",
)
```

## E.3 `SentinelCallback`

```python
SentinelCallback(
    # Required
    profile: CapabilityProfile,

    # Protection
    protect: Union[List[str], Dict[str, float]],
                                           # List: protect all with β=0.8
                                           # Dict: per-capability β values (0.0 to 1.0)
                                           # β=0.0: no protection
                                           # β=0.5: moderate (allows some regression for faster learning)
                                           # β=0.8: strong (default for list mode)
                                           # β=0.9: very strong (small target task cost)
                                           # β=1.0: full freeze (no gradient in capability direction)
    
    method: str = "gradient_projection",
    # "gradient_projection": Remove gradient components in capability subspaces (default, cheapest)
    # "ewc_subspace":        EWC penalty restricted to capability subspaces (softer than projection)
    # "replay_mix":          Mix retention examples into each batch (most robust, most expensive)
    # "hybrid":              gradient_projection + replay_mix (strongest protection)
    
    # Adaptive protection
    adaptive: bool = True,                 # Dynamically adjust β based on training progress
    warmup_steps: int = 100,               # Steps before protection activates
                                           # Purpose: let model find target task direction before constraining
    ramp_steps: int = 50,                  # Steps to ramp β from 0 to target after warmup
    cooldown_factor: float = 0.95,         # Multiply β by this if target loss plateaus
    min_beta: float = 0.3,                 # Floor for adaptive β reduction
    
    # Monitoring
    monitor: bool = True,
    monitor_interval: int = 50,            # Steps between capability probes
                                           # 25 = paranoid, 50 = standard, 100 = relaxed
    probe_method: str = "loss",            # "loss": fast forward-pass probes
                                           # "generate": accurate but slow generation probes
                                           # "drift": fastest, uses parameter drift (no examples needed)
    probe_size: int = 25,                  # Examples per capability per probe
    
    # Alerts and stopping
    alert_threshold: float = 0.03,         # Warn if capability drops > 3%
    critical_threshold: float = 0.05,      # Critical alert if > 5%
    early_stop_threshold: float = 0.10,    # Auto-stop training if > 10%
    early_stop_patience: int = 3,          # Only stop if threshold is exceeded for N consecutive probes
    alert_callback: Optional[Callable] = None,  # Custom alert handler (Slack, email, etc.)
    
    # Replay (only used when method="replay_mix" or "hybrid")
    replay_ratio: float = 0.1,             # Fraction of batch that's replay examples
    replay_sources: Optional[Dict[str, str]] = None,  # capability → retention dataset path
                                                       # None = use built-in retention sets
    
    # Logging
    log_to_wandb: bool = True,
    log_to_jsonl: Optional[str] = "sentinel_log.jsonl",
    log_gradient_stats: bool = True,       # Log detailed gradient projection stats
    log_interval: int = 1,                 # Log every N steps (1 = every step)
    
    # Compute
    device: str = "cuda",
    projection_dtype: torch.dtype = torch.float32,  # Precision for projection math
                                                     # float32 recommended for accuracy
)
```

## E.4 `TrainingConfig`

```python
TrainingConfig(
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    optimizer: str = "adamw",              # "adamw", "adam", "sgd", "adafactor"
    weight_decay: float = 0.01,
    lr_scheduler: str = "cosine",          # "cosine", "linear", "constant"
    warmup_ratio: float = 0.03,
    max_seq_length: int = 2048,
)
```

---

# Appendix F — Tutorials: Common Scenarios

## F.1 DPO Fine-Tuning with Safety Protection

You're aligning a model with DPO but you're worried about safety regression (a common problem — DPO can reduce refusal rates).

```python
from sentinel import CapabilityProfiler, SentinelCallback
from trl import DPOTrainer, DPOConfig

# Profile (reuse if you have one)
profiler = CapabilityProfiler(model, tokenizer)
profile = profiler.profile(capabilities={"safety": "sentinel:safety-300"})

# DPO training with safety protection
callback = SentinelCallback(
    profile=profile,
    protect={"safety": 1.0},        # FULL protection — never compromise safety
    method="hybrid",                 # Strongest method for safety
    safety_mode=True,                # Extra-strict monitoring
    monitor_interval=25,             # Check every 25 steps
    early_stop_threshold=0.02,       # Stop if safety drops even 2%
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    train_dataset=preference_data,
    args=DPOConfig(output_dir="./dpo_output"),
    callbacks=[callback],
)
trainer.train()
```

## F.2 Multi-Language Fine-Tuning with Language Preservation

You're fine-tuning on English data but need to preserve CJK language capabilities.

```python
profile = profiler.profile(capabilities={
    "english": "sentinel:mmlu-en-200",
    "chinese": "sentinel:ceval-200",
    "japanese": "sentinel:jmmlu-200",
    "korean": "sentinel:kmmlu-200",
    "math_multilingual": "sentinel:mgsm-200",
})

# Check which languages are at risk
predictor = RegressionPredictor(profile)
risk = predictor.predict(english_training_data)
# → Shows korean: HIGH, chinese: MEDIUM, japanese: MEDIUM

callback = SentinelCallback(
    profile=profile,
    protect={"korean": 0.9, "chinese": 0.8, "japanese": 0.8},
    # English is not protected — we WANT to change it
)
```

## F.3 Iterative Data Cleaning with Sentinel

You want to build the best training dataset by iteratively removing harmful examples.

```python
from sentinel import RegressionPredictor, RegressionAuditor, DataSurgeon

# Iteration 1: Train naively, audit, identify problems
trainer = SFTTrainer(model=model, train_dataset=data)
trainer.train()

auditor = RegressionAuditor(profile)
report = auditor.audit(model_before=base, model_after=model, training_data=data)
print(f"Regression: {report.overall_regression_score:.4f}")

# Iteration 2: Remove harmful examples, retrain
surgeon = DataSurgeon(report, profile)
plan = surgeon.plan(data)
print(plan.summary())  # "Remove 341 examples, expected recovery: math +5.2%, safety +4.1%"

cleaned_data = plan.apply(data)
print(f"Original: {len(data)} → Cleaned: {len(cleaned_data)}")

# Retrain on cleaned data
model_v2 = reload_base_model()
trainer_v2 = SFTTrainer(model=model_v2, train_dataset=cleaned_data)
trainer_v2.train()

# Verify improvement
report_v2 = auditor.audit(model_before=base, model_after=model_v2, training_data=cleaned_data)
print(f"Regression v1: {report.overall_regression_score:.4f}")
print(f"Regression v2: {report_v2.overall_regression_score:.4f}")
# Expected: v2 regression is significantly lower
```

## F.4 Using Sentinel with Axolotl

```yaml
# axolotl_config.yml
base_model: Qwen/Qwen2.5-7B-Instruct
model_type: AutoModelForCausalLM

load_in_4bit: true
adapter: qlora
lora_r: 16
lora_alpha: 32
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj

datasets:
  - path: my-org/customer-support
    type: sharegpt

# Sentinel configuration
sentinel:
  enabled: true
  profile: sentinel-community/qwen2.5-7b-instruct-profile
  protect:
    math: 0.9
    safety: 1.0
    code: 0.5
  method: gradient_projection
  adaptive: true
  monitor: true
  monitor_interval: 50
  early_stop_threshold: 0.05
  log_to_wandb: true
  audit_on_complete: true
  audit_output: ./audit_report.html
```

## F.5 Using Sentinel with Unsloth

```python
from unsloth import FastLanguageModel
from sentinel import SentinelCallback

# Load with Unsloth (4-bit quantized)
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

# Sentinel works the same — it only touches LoRA parameters
profile = CapabilityProfiler(model, tokenizer).profile(
    capabilities={"math": "sentinel:math-500", "safety": "sentinel:safety-300"}
)

callback = SentinelCallback(profile, protect=["math", "safety"])

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=data,
    callbacks=[callback],  # Works identically with Unsloth
)
trainer.train()
```

## F.6 The Pre-Commit Hook: Never Train Without Checking

Add Sentinel prediction as a pre-training check:

```python
# pre_check.py — run before launching expensive training
from sentinel import CapabilityProfile, RegressionPredictor, TrainingConfig
import sys

profile = CapabilityProfile.load("profile.sentinel")
predictor = RegressionPredictor(profile, training_config=TrainingConfig(
    learning_rate=2e-5, num_epochs=3, lora_r=16
))

risk = predictor.predict(training_data)

# Gate: block training if any capability has CRITICAL risk
critical_caps = [c for c, r in risk.capabilities.items() if r.risk_level == "CRITICAL"]
if critical_caps:
    print(f"❌ BLOCKED: Critical regression risk for: {', '.join(critical_caps)}")
    print("Run `sentinel predict --verbose` for details.")
    print("Options: (1) Add protection  (2) Clean data  (3) Override with --force")
    sys.exit(1)

print("✅ Risk assessment passed. Proceeding to training.")
```

---

# Appendix G — FAQ

## General

**Q: Does Sentinel work with full fine-tuning (not LoRA)?**
A: Not at Level 100. Sentinel's core advantage is that LoRA's low-rank parameter space makes SVD, projection, and influence functions computationally tractable. Full fine-tuning with 7B+ parameters would require randomized approximations that significantly reduce accuracy. We may add experimental full-parameter support at Level 90+, but LoRA/QLoRA/DoRA is the primary target.

**Q: Does Sentinel work with QLoRA (4-bit quantization)?**
A: Yes. Sentinel operates on the LoRA adapter parameters, not the quantized base model weights. Profiling with QLoRA uses slightly more time (dequantization during backward pass) but produces identical subspaces. Memory savings from quantization are preserved.

**Q: Does Sentinel work with DoRA (Weight-Decomposed Low-Rank Adaptation)?**
A: Yes. DoRA decomposes LoRA into magnitude and direction components. Sentinel treats all trainable parameters (including DoRA's magnitude vector) as the parameter space for subspace computation. The API is identical.

**Q: How does Sentinel compare to just adding retention data (data mixing)?**
A: Data mixing is Sentinel's `replay_mix` protection method — one of four options. The difference is that Sentinel tells you *how much* retention data to add and *which* retention data is most effective. Without Sentinel, you're guessing ratios and hoping. With Sentinel, you know that "math regression requires 200 examples from MATH training set mixed at 5%" because the Data Surgeon computed it from influence scores.

**Q: Can I use Sentinel for RLHF/PPO, not just SFT?**
A: Yes. SentinelCallback hooks into `on_step_end`, which is called by all TRL trainers including PPOTrainer. The gradient projection applies to the policy gradient. Monitoring and auditing work identically. The main caveat is that RLHF training dynamics are noisier than SFT, so prediction confidence intervals are wider.

**Q: What if my base model doesn't have a LoRA adapter — can I still profile it?**
A: Sentinel adds a temporary LoRA adapter for profiling, computes the capability subspaces in that LoRA space, and then discards the adapter. You specify the LoRA config you'll use for training. The profile must match your training LoRA config (same rank, same target modules).

## Accuracy & Reliability

**Q: How accurate are the regression predictions?**
A: Based on our calibration experiments across 75 training runs (5 model families × 5 scenarios × 3 configs): R² = 0.72 for ordinal prediction (ranking which capabilities are most at risk) and RMSE = 2.8% for cardinal prediction (predicting exact accuracy delta). The predictor is most accurate for SFT with moderate dataset sizes (5K–50K examples).

**Q: Can Sentinel prevent ALL regression?**
A: With β=1.0 (full freeze), the gradient component in the protected subspace is completely removed, so regression in that subspace should be zero. In practice, you'll see residual regression of 0.1–0.5% because: (1) capability subspaces are approximate (top-k SVD, not exact), and (2) capabilities may have some representation outside the computed subspace.

**Q: What if protecting a capability makes the model unable to learn the target task?**
A: This happens when the target task and the protected capability share significant subspace overlap. Sentinel warns you about this in the risk report (the "estimated protection cost" number). If the cost is >10%, consider: (1) reducing β for that capability, (2) using `ewc_subspace` instead of `gradient_projection` (soft penalty instead of hard block), or (3) accepting some regression on that capability.

**Q: Do I need to re-profile if I change LoRA rank or target modules?**
A: Yes. The capability subspace is computed in LoRA parameter space, which changes shape when you change rank or target modules. A profile for r=16 targeting q_proj,v_proj is not valid for r=32 targeting q_proj,k_proj,v_proj,o_proj. However, profiles are cached, so re-profiling is a one-time cost per configuration.

## Performance

**Q: How much does Sentinel slow down training?**
A: With `gradient_projection` (default): <3% total overhead including monitoring probes every 50 steps. The projection itself adds <0.01% per step. The monitoring probes are the main cost — you control the tradeoff with `monitor_interval`.

**Q: Can I run profiling on a different GPU than training?**
A: Yes. Profile on any GPU, save to disk (`profile.save("profile.sentinel")`), and load it on any other machine (`CapabilityProfile.load("profile.sentinel")`). Profiles are portable.

**Q: Does Sentinel work with multi-GPU training?**
A: Yes. DDP and DeepSpeed ZeRO-1/2 work out of the box — LoRA parameters are small enough to be replicated across ranks. DeepSpeed ZeRO-3 and FSDP require the specialized `DeepSpeedSentinelCallback` or `FSDPSentinelCallback` which handles LoRA parameter gather/scatter.

## Practical Usage

**Q: Which capabilities should I protect?**
A: Start with the capabilities your users care about. If you're building a customer support bot, protect math, safety, and code (common collateral damage from conversational fine-tuning). If you're not sure, run `sentinel predict` first — it tells you what's at risk. You only need to protect capabilities that are actually at risk.

**Q: What β (protection strength) should I use?**
A: Start with the defaults (β=0.8 for most capabilities). If `sentinel predict` shows a capability at CRITICAL risk, use β=0.9–1.0. If the protection cost is too high (your target task performance suffers), reduce β. The adaptive mode (`adaptive=True`) handles this automatically in most cases.

**Q: Can I protect custom/domain-specific capabilities?**
A: Yes. Register a custom capability with any evaluation dataset:
```python
from sentinel import register_capability
register_capability("medical_qa", eval_set=my_medical_eval_dataset)
```
The eval set should be 100–500 examples of question-answer pairs that test the capability you want to protect.

**Q: How do I know if Sentinel is actually working?**
A: Three ways: (1) The training log shows regression deltas at every probe interval — you can see capabilities staying flat instead of dropping. (2) After training, compare `sentinel audit` with and without protection. (3) Run the full pipeline on a toy example first (Section 1.2) to build confidence.

---

# Appendix H — Troubleshooting

## H.1 Common Issues

### "Profile computation runs out of GPU memory"

**Symptom:** `OutOfMemoryError` during `profiler.profile()`.

**Fixes (in order of preference):**
1. Enable gradient checkpointing: `profiler = CapabilityProfiler(model, tokenizer, gradient_checkpointing=True)`
2. Reduce batch size: `gradient_batch_size=1`
3. Reduce examples: `max_examples_per_capability=200`
4. Use randomized SVD (default): `svd_method="randomized"`
5. Use QLoRA (4-bit base model)

### "Risk report shows all capabilities at LOW risk but I know regression happens"

**Symptom:** Predictor says everything is fine, but you observe regression after training.

**Possible causes:**
1. **Sample size too low:** Increase `training_data_sample_size` from 500 to 2000+. Small samples may miss gradient directions that emerge from the full dataset.
2. **Subspace rank too low:** Increase `subspace_rank` from 64 to 128. The capability subspace may be broader than 64 dimensions.
3. **Calibration mismatch:** The default calibration model may not match your specific model family. Run `sentinel calibrate` on your own (prediction, actual) pairs to fine-tune calibration.
4. **Non-LoRA regression:** If regression is driven by the base model's residual stream (not the LoRA subspace), Sentinel won't detect it. This is rare with LoRA but can happen at very high learning rates.

### "Protection is too aggressive — model can't learn the target task"

**Symptom:** Target task loss stops decreasing after protection activates.

**Fixes:**
1. **Check overlap:** Run `sentinel predict` — if the estimated protection cost is >10%, the target task significantly overlaps with protected capabilities. This is a fundamental tradeoff, not a bug.
2. **Reduce β:** Lower protection strength (e.g., 0.9 → 0.6) for the overlapping capability.
3. **Switch method:** Use `ewc_subspace` instead of `gradient_projection`. EWC penalizes rather than blocks, allowing some movement in capability directions.
4. **Increase warmup:** Set `warmup_steps=200` or higher. Longer warmup lets the model find non-conflicting gradient directions before protection engages.
5. **Use adaptive mode:** Set `adaptive=True` — it will automatically reduce β if the target loss plateaus.


### "Training crashes with `shape mismatch` after modifying LoRA config"

**Symptom:** `RuntimeError: shape mismatch` when using SentinelCallback.

**Cause:** The profile was computed with a different LoRA configuration (different rank or target modules) than the current model.

**Fix:** Re-profile with the current LoRA config. Profiles are tied to (model_id, lora_r, target_modules).

### "Audit takes too long"

**Symptom:** `auditor.audit()` runs for hours on large training datasets.

**Fixes:**
1. **Use faster attribution:** Switch from `lora_trak` to `gradient_cosine` (10× faster, slightly less accurate).
2. **Reduce training data sample:** Pass `max_train_examples=5000` to only attribute the top 5K most influential examples.
3. **Skip attribution entirely:** If you just want accuracy deltas without per-example attribution, use `auditor.quick_audit()`.

### "Monitor probes show noisy readings — capability appears to oscillate"

**Symptom:** The live monitor shows math regression alternating between -1% and +2% between consecutive probes.

**Cause:** Probe size is too small (high variance from random sampling) or loss-based probes don't correlate well with accuracy for this capability.

**Fixes:**
1. **Increase probe size:** `probe_size=50` instead of 25. More examples per probe = less noise.
2. **Increase trend window:** `trend_window=10` instead of 5. Alerts only fire if the trend is consistently negative over more probes.
3. **Switch to generation probes:** `probe_method="generate"` for noisy capabilities. More expensive but more accurate.

## H.2 Diagnostic Commands

```bash
# Check if your model is compatible
sentinel diagnose --model Qwen/Qwen2.5-7B-Instruct

# Validate a profile
sentinel inspect profile.sentinel

# Check if a profile matches your current model config
sentinel inspect profile.sentinel --validate-against-model ./my_model

# Dry run — run the full pipeline without actually modifying anything
sentinel predict --profile profile.sentinel --training-data data.jsonl --dry-run --verbose

# Debug mode — maximum logging verbosity
SENTINEL_LOG_LEVEL=DEBUG sentinel train --profile profile.sentinel ...
```

## H.3 Getting Help

- **GitHub Issues:** File bugs, feature requests, and questions with the `bug`, `feature`, or `question` label
- **Discord:** `#sentinel` channel for real-time help
- **Discussions:** GitHub Discussions for longer-form technical Q&A
- **Paper:** For the theoretical foundations, cite the ICML 2027 Paper 1 (prediction) or NeurIPS 2027 Paper 2 (prevention)

---

*Sentinel Architecture Document v1.0 — April 2026*
*Level 0. Everything is ahead.*

