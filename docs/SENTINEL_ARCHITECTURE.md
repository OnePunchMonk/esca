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
1. [Problem Statement](#1-problem-statement)
2. [Core Theory — Capability Subspaces](#2-core-theory--capability-subspaces)
3. [System Architecture](#3-system-architecture)
4. [Component I — Capability Profiler](#4-component-i--capability-profiler)
5. [Component II — Regression Predictor](#5-component-ii--regression-predictor)
6. [Component III — Constrained Optimizer](#6-component-iii--constrained-optimizer)
7. [Component IV — Live Monitor](#7-component-iv--live-monitor)
8. [Component V — Post-Training Auditor](#8-component-v--post-training-auditor)
9. [Component VI — Data Surgeon](#9-component-vi--data-surgeon)
10. [CLI & Developer Experience](#10-cli--developer-experience)
11. [Integration Layer](#11-integration-layer)
12. [Observability & Reporting](#12-observability--reporting)
13. [Advanced Features (Level 80–100)](#13-advanced-features-level-80100)
14. [Evaluation & Benchmarks](#14-evaluation--benchmarks)
15. [Research Program](#15-research-program)
16. [Repository Structure](#16-repository-structure)
17. [Roadmap — Level 0 → 100](#17-roadmap--level-0--100)
18. [Appendix A — Mathematical Foundations](#appendix-a--mathematical-foundations)
19. [Appendix B — Failure Modes](#appendix-b--failure-modes)
20. [Appendix C — Competitive Landscape](#appendix-c--competitive-landscape)

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

# 1. Problem Statement

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

## 1.4 What Sentinel Is Not

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
╚══════════════════════════════════════════════════════════════════╝
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
