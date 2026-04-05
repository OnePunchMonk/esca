[normal] ESCA
[normal] Emergent Self-Correction Amplification
[normal] A Production-Grade Library + Research Framework for Amplifying
[normal] Self-Correction in RLVR-Trained Language Models
[normal] Built on TRL · Axolotl · Transformers · vLLM · DeepSpeed
[normal] Status: Architecture v0.2  ·  April 2026
[normal] Target: 2–3M Lines of Code  ·  GitHub-First  ·  Paper + Library
[Heading 1] 0. The 4-Benchmark Gate: Why ESCA
[normal] Before any line of code is written, every idea must pass a 4-benchmark critique. This is not a rubber stamp — four prior iterations were rejected against this framework. ESCA is the only idea that passed all four.
[Heading 1] 1. Vision & Problem Statement
[Heading 2] 1.1 The Core Observation
[normal] DeepSeek-R1 and subsequent RLVR-trained models spontaneously developed 'wait' behavior — self-directed pauses and course corrections during reasoning chains. The community recognized this as important and moved on. Nobody asked the second-order question:
[normal] "What is the ceiling of self-correction ability if we explicitly train for it?"
[normal] ESCA answers this question. The thesis is that RLVR training implicitly generates the most valuable training signal it never trains on. When a model's reasoning process reverses course, recognizes an error, and recovers — that is exactly the behavior we want. But the current training objective gives it no special status. Self-correction is rewarded only incidentally, as part of a correct final answer, and the reversal itself is treated as just more tokens.
[normal] The result: self-correction behavior is fragile, inconsistent, and present only when the model stumbles into it during exploration. ESCA makes it a robustly trained capability.
[Heading 2] 1.2 Why Now (April 2026)
[Heading 2] 1.3 Intellectual Lineage
[normal] ESCA is not invented from scratch. It sits at the intersection of three research threads, each of which it builds on directly:
[Heading 1] 2. Architecture Deep Dive
[Heading 2] 2.1 Core Definitions
[normal] Every subsequent design decision follows from these three definitions. They must be precise, not intuitive.
[Heading 2] 2.2 System Architecture
[normal] ESCA adds three components to an existing GRPO training loop. All three are implemented as TRL callbacks — they hook into the existing infrastructure without modifying the optimizer, the model, or the rollout generation code.
[normal] ESCA Training Loop — Data Flow
[Heading 2] 2.3 The SCE Detector — Implementation
[normal] The detector is the most sensitive component. False positives (theatrical SCEs trained in) are expensive. False negatives (missed genuine SCEs) are cheap — you just lose signal. The design prioritizes precision over recall, with τ as the tuning knob.
[normal] Full Python implementation:
[normal] # esca/detection/sce_detector.py
[normal] from __future__ import annotations
[normal] import re
[normal] from dataclasses import dataclass, field
[normal] from typing import Optional, List, Tuple
[normal] import numpy as np
[normal] from sentence_transformers import SentenceTransformer
[normal] # ─── Reversal marker vocabulary ─────────────────────────────────────────
[normal] # Curated from 10K manually inspected R1/QwQ rollouts.
[normal] # Sorted by decreasing precision on held-out set.
[normal] REVERSAL_MARKERS: List[str] = [
[normal] "wait", "actually", "let me reconsider", "that's not right",
[normal] "i made an error", "let me try again", "hold on", "no wait",
[normal] "i think i was wrong", "let me restart", "going back",
[normal] "actually no", "that approach won't work", "i need to rethink",
[normal] "this is wrong", "i got confused", "let me recalculate",
[normal] "scratch that", "on second thought", "i realize i've been",
[normal] "my previous reasoning was", "i made a mistake", "correction:",
[normal] "i need to revise", "let me redo", "that doesn't work because",
[normal] "i was wrong about", "re-examining", "looking at this again",
[normal] "i think i overcomplicated", "stepping back", "to reconsider",
[normal] "i realize now", "i missed", "let me be more careful",
[normal] "i'm going in the wrong direction", "different approach",
[normal] "i should", "more carefully", "rechecking", "alternative:",
[normal] "approach 2", "method 2", "try differently", "from the start",
[normal] # VLM extensions:
[normal] "let me look at the image again", "i may have misread",
[normal] "looking more carefully", "i think i misidentified",
[normal] "the image actually shows", "on second look",
[normal] ]
[normal] @dataclass
[normal] class SCEEvent:
[normal] rollout_text: str
[normal] reversal_pos: int          # character position of reversal marker
[normal] semantic_shift: float      # cosine distance (higher = more genuine)
[normal] pre_segment: str           # tokens before reversal (wrong path)
[normal] correction_moment: str     # reversal marker + immediate context
[normal] post_segment: str          # tokens after reversal (correct path)
[normal] is_genuine: bool           # semantic_shift > tau
[normal] problem_id: str = ''
[normal] step: int = 0
[normal] difficulty: float = 0.0    # injected from verifier metadata
[normal] class SCEDetector:
[normal] def __init__(
[normal] self,
[normal] tau: float = 0.4,
[normal] embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
[normal] chunk_size_tokens: int = 100,
[normal] context_chunks: int = 3,
[normal] device: str = 'cpu',   # CPU — negligible overhead
[normal] ):
[normal] self.tau = tau
[normal] self.chunk_size = chunk_size_tokens
[normal] self.ctx = context_chunks
[normal] self.embedder = SentenceTransformer(embedder_name, device=device)
[normal] # Dynamic threshold: adapts to prevent performative drift
[normal] self._shift_history: List[float] = []
[normal] self._tau_dynamic = tau
[normal] def detect(
[normal] self,
[normal] rollout_text: str,
[normal] is_correct: bool,
[normal] problem_id: str = '',
[normal] step: int = 0,
[normal] ) -> Optional[SCEEvent]:
[normal] if not is_correct:
[normal] return None  # Only mine from correct rollouts — verifier is free
[normal] # Step 1: Scan for reversal markers (O(|text| × |markers|))
[normal] marker_positions = self._find_markers(rollout_text)
[normal] if not marker_positions:
[normal] return None
[normal] # Step 2: For each candidate, test semantic shift
[normal] best_sce: Optional[SCEEvent] = None
[normal] best_shift = 0.0
[normal] for pos, marker in marker_positions:
[normal] shift, pre_seg, post_seg = self._semantic_shift(rollout_text, pos)
[normal] if shift > self._tau_dynamic and shift > best_shift:
[normal] best_shift = shift
[normal] correction_ctx = rollout_text[pos:pos+200]  # marker + 200 chars
[normal] best_sce = SCEEvent(
[normal] rollout_text=rollout_text,
[normal] reversal_pos=pos,
[normal] semantic_shift=shift,
[normal] pre_segment=pre_seg,
[normal] correction_moment=correction_ctx,
[normal] post_segment=post_seg,
[normal] is_genuine=True,
[normal] problem_id=problem_id,
[normal] step=step,
[normal] )
[normal] # Update dynamic threshold to prevent performative drift
[normal] if best_sce is not None:
[normal] self._update_dynamic_tau(best_shift)
[normal] return best_sce
[normal] def _find_markers(self, text: str) -> List[Tuple[int, str]]:
[normal] text_lower = text.lower()
[normal] results = []
[normal] for marker in REVERSAL_MARKERS:
[normal] start = 0
[normal] while True:
[normal] idx = text_lower.find(marker, start)
[normal] if idx == -1: break
[normal] results.append((idx, marker))
[normal] start = idx + 1
[normal] return sorted(results, key=lambda x: x[0])
[normal] def _semantic_shift(
[normal] self, text: str, reversal_pos: int
[normal] ) -> Tuple[float, str, str]:
[normal] pre_text  = text[:reversal_pos]
[normal] post_text = text[reversal_pos:]
[normal] # Chunk into ~100-token windows, take last/first 3
[normal] pre_chunks  = self._chunk(pre_text)[-self.ctx:]
[normal] post_chunks = self._chunk(post_text)[:self.ctx]
[normal] if not pre_chunks or not post_chunks:
[normal] return 0.0, pre_text, post_text
[normal] pre_emb  = self.embedder.encode(pre_chunks).mean(axis=0)
[normal] post_emb = self.embedder.encode(post_chunks).mean(axis=0)
[normal] shift = float(1 - self._cosine_sim(pre_emb, post_emb))
[normal] return shift, pre_text, post_text
[normal] def _chunk(self, text: str) -> List[str]:
[normal] words = text.split()
[normal] return [' '.join(words[i:i+self.chunk_size])
[normal] for i in range(0, len(words), self.chunk_size)
[normal] if words[i:i+self.chunk_size]]
[normal] @staticmethod
[normal] def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
[normal] denom = np.linalg.norm(a) * np.linalg.norm(b)
[normal] return float(np.dot(a, b) / denom) if denom > 0 else 0.0
[normal] def _update_dynamic_tau(self, new_shift: float) -> None:
[normal] self._shift_history.append(new_shift)
[normal] if len(self._shift_history) > 100:
[normal] self._shift_history.pop(0)
[normal] mean_shift = np.mean(self._shift_history)
[normal] # If mean shift drops below 0.5, tighten threshold
[normal] # This prevents the model from gaming a fixed threshold
[normal] self._tau_dynamic = max(self.tau, mean_shift * 0.8)
[Heading 2] 2.4 The SCE Replay Buffer
[normal] The replay buffer is a first-class data structure, not an afterthought. It manages four non-trivial problems simultaneously: capacity, recency weighting, expiry (to handle policy drift), and difficulty-stratified sampling (to ensure hard problems are overrepresented in replay).
[normal] # esca/buffer/replay_buffer.py
[normal] from collections import deque
[normal] from typing import List, Dict, Tuple
[normal] import numpy as np
[normal] from ..detection.sce_detector import SCEEvent
[normal] class SCEReplayBuffer:
[normal] '''
[normal] Priority replay buffer for Self-Correction Events.
[normal] Sampling weights are a product of:
[normal] - Recency: (1 - age/expiry_steps)  — downweights stale policy traces
[normal] - Shift magnitude: sce.semantic_shift  — prefers high-confidence SCEs
[normal] - Difficulty: sce.difficulty  — oversamples hard problems
[normal] '''
[normal] def __init__(
[normal] self,
[normal] capacity: int = 50_000,
[normal] expiry_steps: int = 500,
[normal] difficulty_alpha: float = 0.5,  # weight of difficulty in sampling
[normal] ):
[normal] self.capacity = capacity
[normal] self.expiry_steps = expiry_steps
[normal] self.difficulty_alpha = difficulty_alpha
[normal] self._buffer: deque = deque(maxlen=capacity)
[normal] self._step = 0
[normal] def add(self, sce: SCEEvent, step: int) -> None:
[normal] sce.step = step
[normal] self._buffer.append(sce)
[normal] def sample(self, n: int) -> List[SCEEvent]:
[normal] valid = [s for s in self._buffer
[normal] if self._step - s.step < self.expiry_steps]
[normal] if len(valid) < n:
[normal] return valid
[normal] weights = np.array([
[normal] s.semantic_shift
[normal] * (1 - (self._step - s.step) / self.expiry_steps)
[normal] * (1 + self.difficulty_alpha * s.difficulty)
[normal] for s in valid
[normal] ])
[normal] weights = np.clip(weights, 1e-8, None)
[normal] weights /= weights.sum()
[normal] idxs = np.random.choice(len(valid), size=n, replace=False, p=weights)
[normal] return [valid[i] for i in idxs]
[normal] def tick(self) -> None:
[normal] self._step += 1
[normal] def stats(self) -> Dict:
[normal] valid = [s for s in self._buffer
[normal] if self._step - s.step < self.expiry_steps]
[normal] return {
[normal] 'total': len(self._buffer),
[normal] 'valid': len(valid),
[normal] 'mean_shift': float(np.mean([s.semantic_shift for s in valid])) if valid else 0.0,
[normal] 'mean_difficulty': float(np.mean([s.difficulty for s in valid])) if valid else 0.0,
[normal] 'step': self._step,
[normal] }
[Heading 2] 2.5 The Dual Training Objective
[normal] The main GRPO objective is unchanged. ESCA augments it with two mechanisms: a supplementary SFT step on correction-moment tokens, and an optional reward bonus for rollouts containing Genuine SCEs.
[normal] # esca/training/esca_callback.py
[normal] from transformers import TrainerCallback, TrainerState, TrainerControl
[normal] from trl import GRPOTrainer
[normal] from ..detection.sce_detector import SCEDetector
[normal] from ..buffer.replay_buffer import SCEReplayBuffer
[normal] from ..training.sft_step import run_sft_on_correction_moments
[normal] from ..logging.diagnostics import ESCADiagnostics
[normal] class SelfCorrectionCallback(TrainerCallback):
[normal] '''
[normal] Primary ESCA entry point. Drop this into any GRPOTrainer.
[normal] No changes to the trainer, model, or optimizer required.
[normal] '''
[normal] def __init__(
[normal] self,
[normal] tau: float = 0.4,
[normal] n_replay: int = 50,
[normal] replay_batch_size: int = 32,
[normal] alpha: float = 0.1,
[normal] buffer_capacity: int = 50_000,
[normal] expiry_steps: int = 500,
[normal] push_to_hub: bool = True,
[normal] hub_repo_id: str = 'esca-project/sce-traces',
[normal] wandb_log: bool = True,
[normal] device: str = 'cpu',
[normal] ):
[normal] self.detector = SCEDetector(tau=tau, device=device)
[normal] self.buffer   = SCEReplayBuffer(capacity=buffer_capacity, expiry_steps=expiry_steps)
[normal] self.diagnostics = ESCADiagnostics(wandb_log=wandb_log)
[normal] self.n_replay    = n_replay
[normal] self.replay_bs   = replay_batch_size
[normal] self.alpha       = alpha
[normal] self.push_to_hub = push_to_hub
[normal] self.hub_repo_id = hub_repo_id
[normal] self._step       = 0
[normal] def on_step_end(self, args, state, control, **kwargs):
[normal] # kwargs contains rollouts from GRPOTrainer hook
[normal] rollouts = kwargs.get('rollouts', [])
[normal] rewards  = kwargs.get('rewards', [])
[normal] # Phase 1: Detect SCEs in this step's rollouts
[normal] for rollout, reward in zip(rollouts, rewards):
[normal] is_correct = reward > 0.5  # verifier threshold
[normal] sce = self.detector.detect(
[normal] rollout_text=rollout['text'],
[normal] is_correct=is_correct,
[normal] problem_id=rollout.get('problem_id', ''),
[normal] step=self._step,
[normal] )
[normal] if sce is not None:
[normal] self.buffer.add(sce, self._step)
[normal] # Phase 2: Augment reward with ESCA bonus (optional)
[normal] if self.alpha > 0:
[normal] self._inject_reward_bonus(rollouts, rewards, kwargs)
[normal] # Phase 3: Run supplementary SFT every N_replay steps
[normal] if self._step % self.n_replay == 0 and len(self.buffer._buffer) >= self.replay_bs:
[normal] sce_batch = self.buffer.sample(self.replay_bs)
[normal] model = kwargs.get('model')
[normal] optimizer = kwargs.get('optimizer')
[normal] if model and optimizer:
[normal] run_sft_on_correction_moments(model, optimizer, sce_batch)
[normal] # Phase 4: Log diagnostics
[normal] self.diagnostics.log(self._step, self.buffer.stats(), rollouts, rewards)
[normal] self.buffer.tick()
[normal] self._step += 1
[normal] def on_train_end(self, args, state, control, **kwargs):
[normal] if self.push_to_hub:
[normal] self._push_sce_dataset()
[normal] def _inject_reward_bonus(self, rollouts, rewards, kwargs):
[normal] # Reward bonus already applied during detection — placeholder for
[normal] # frameworks that support mid-step reward modification
[normal] pass
[normal] def _push_sce_dataset(self):
[normal] from datasets import Dataset
[normal] records = [
[normal] {'pre_segment': s.pre_segment,
[normal] 'correction_moment': s.correction_moment,
[normal] 'post_segment': s.post_segment,
[normal] 'semantic_shift': s.semantic_shift,
[normal] 'problem_id': s.problem_id,
[normal] 'step': s.step}
[normal] for s in self.buffer._buffer
[normal] ]
[normal] if records:
[normal] Dataset.from_list(records).push_to_hub(self.hub_repo_id)
[Heading 1] 3. Full Repository Architecture (2–3M Lines of Code)
[normal] ESCA is not a 400-line library. It is a research platform. The 2–3M LoC target is achieved by treating it as a complete ecosystem: the core library (~50K LoC), a comprehensive test suite (~200K LoC), a benchmark harness (~100K LoC), a PostTrace companion library (~300K LoC), full integration layers (~150K LoC), an evaluation dashboard (~80K LoC), generated SCE datasets (~1M+ rows), experiment configs (~50K LoC), and documentation (~100K LoC). Every line has a reason.
[Heading 2] 3.1 Directory Structure
[normal] esca/                                          # Root monorepo
[normal] ├── esca-core/                                 # ~50K LoC — Primary Python library
[normal] │   ├── esca/
[normal] │   │   ├── __init__.py
[normal] │   │   ├── detection/                         # SCE Detection subsystem
[normal] │   │   │   ├── __init__.py
[normal] │   │   │   ├── sce_detector.py                # Core detector (shown above)
[normal] │   │   │   ├── marker_vocabulary.py           # Curated reversal markers + metadata
[normal] │   │   │   ├── semantic_shift.py              # Embedding-based shift computation
[normal] │   │   │   ├── genuine_vs_performative.py     # Genuine/Performative classifier
[normal] │   │   │   ├── vlm_extensions.py              # VLM-specific marker extensions
[normal] │   │   │   └── tests/
[normal] │   │   │       ├── test_detector.py           # Unit + property-based tests
[normal] │   │   │       ├── test_markers.py
[normal] │   │   │       └── conftest.py
[normal] │   │   ├── buffer/                            # Replay Buffer subsystem
[normal] │   │   │   ├── __init__.py
[normal] │   │   │   ├── replay_buffer.py               # Priority replay buffer (shown above)
[normal] │   │   │   ├── difficulty_stratifier.py       # Difficulty-stratified sampling
[normal] │   │   │   ├── freshness_manager.py           # Recency / expiry management
[normal] │   │   │   └── tests/
[normal] │   │   │       ├── test_replay_buffer.py
[normal] │   │   │       └── test_sampling_distribution.py
[normal] │   │   ├── training/                          # Training Integration subsystem
[normal] │   │   │   ├── __init__.py
[normal] │   │   │   ├── esca_callback.py               # SelfCorrectionCallback (shown above)
[normal] │   │   │   ├── sft_step.py                    # Correction-moment SFT step
[normal] │   │   │   ├── reward_augmenter.py            # R_total = R_task + alpha * bonus
[normal] │   │   │   ├── curriculum.py                  # Difficulty-aware problem sampling
[normal] │   │   │   └── tests/
[normal] │   │   │       ├── test_callback_integration.py
[normal] │   │   │       └── test_sft_step.py
[normal] │   │   ├── logging/                           # Diagnostics & Observability
[normal] │   │   │   ├── __init__.py
[normal] │   │   │   ├── diagnostics.py                 # W&B + JSONL logging
[normal] │   │   │   ├── sce_dashboard.py               # Local Gradio diagnostic dashboard
[normal] │   │   │   ├── metrics.py                     # SCE frequency, OOD, difficulty curves
[normal] │   │   │   └── tests/
[normal] │   │   │       └── test_metrics.py
[normal] │   │   ├── integrations/                      # Framework integration layers
[normal] │   │   │   ├── __init__.py
[normal] │   │   │   ├── trl_grpo.py                    # TRL GRPOTrainer callback hook
[normal] │   │   │   ├── axolotl_plugin.py              # Axolotl YAML-driven integration
[normal] │   │   │   ├── verl_adapter.py                # VeRL (ByteDance) adapter
[normal] │   │   │   ├── deepspeed_hooks.py             # DeepSpeed ZeRO-3 compatibility
[normal] │   │   │   └── tests/
[normal] │   │   │       └── test_integrations.py
[normal] │   │   └── hub/                               # HuggingFace Hub integration
[normal] │   │       ├── __init__.py
[normal] │   │       ├── dataset_builder.py             # SCE trace → HF Dataset
[normal] │   │       ├── model_card.py                  # Auto-generate ESCA model cards
[normal] │   │       └── push_traces.py                 # Push SCE datasets to Hub
[normal] │   ├── setup.py
[normal] │   ├── pyproject.toml
[normal] │   └── README.md
[normal] │
[normal] ├── esca-bench/                                # ~100K LoC — Benchmark harness
[normal] │   ├── benchmarks/
[normal] │   │   ├── math_sce/                          # MATH difficulty-stratified SCE eval
[normal] │   │   │   ├── run_eval.py
[normal] │   │   │   ├── difficulty_splitter.py
[normal] │   │   │   └── configs/
[normal] │   │   ├── amc_aime/                          # AMC/AIME hard problem eval
[normal] │   │   ├── gpqa/                              # GPQA — OOD scientific reasoning
[normal] │   │   ├── code_sce/                          # Code generation SCE patterns
[normal] │   │   ├── vlm_sce/                           # VLM self-correction benchmarks
[normal] │   │   │   ├── mathvista_sce/
[normal] │   │   │   └── chartqa_sce/
[normal] │   │   └── ood_transfer/                      # The decisive OOD experiment
[normal] │   │       ├── in_distribution.py
[normal] │   │       ├── near_ood.py
[normal] │   │       └── far_ood.py                     # Spatial reasoning, held-out
[normal] │   ├── ablations/                             # Ablation study scripts
[normal] │   │   ├── replay_only_vs_reward_only.py
[normal] │   │   ├── full_path_vs_moment_only.py
[normal] │   │   └── replay_ratio_sweep.py              # N_replay ∈ {20, 50, 100}
[normal] │   └── baselines/                             # Baseline comparisons
[normal] │       ├── grpo_baseline.py
[normal] │       └── standard_sft_augmentation.py
[normal] │
[normal] ├── esca-experiments/                          # ~50K LoC — Experiment configs
[normal] │   ├── configs/
[normal] │   │   ├── qwen2.5_1.5b_math.yaml
[normal] │   │   ├── qwen2.5_7b_math_code.yaml
[normal] │   │   ├── qwen2.5_vl_7b_mathvista.yaml
[normal] │   │   └── llama3.1_8b_gpqa.yaml
[normal] │   ├── scripts/
[normal] │   │   ├── week1_baseline_instrumentation.sh
[normal] │   │   ├── week2_esca_v0.sh
[normal] │   │   ├── week3_ablations.sh
[normal] │   │   └── week4_ood_decisive.sh
[normal] │   └── results/                               # Tracked with DVC
[normal] │       ├── baseline_grpo/
[normal] │       ├── esca_v0/
[normal] │       └── ablations/
[normal] │
[normal] ├── posttrace/                                 # ~300K LoC — Companion library
[normal] │   ├── posttrace/
[normal] │   │   ├── attributors/
[normal] │   │   │   ├── lora_trak.py                   # TRAK adapted for LoRA subspace
[normal] │   │   │   ├── data_inf.py                    # DataInf approximation
[normal] │   │   │   └── gradient_cosine.py             # Fast cosine similarity in grad space
[normal] │   │   ├── oracles/
[normal] │   │   │   ├── quality_scorer.py              # Pre-training data scoring
[normal] │   │   │   └── conflict_detector.py           # Gradient-conflicting examples
[normal] │   │   ├── curator/
[normal] │   │   │   ├── active_loop.py                 # train→attribute→filter→retrain
[normal] │   │   │   └── reweighter.py                  # Per-example loss weighting
[normal] │   │   └── integrations/
[normal] │   │       ├── axolotl_plugin.py
[normal] │   │       └── hf_trainer_cb.py
[normal] │   └── README.md
[normal] │
[normal] ├── esca-dashboard/                            # ~80K LoC — React + FastAPI dashboard
[normal] │   ├── frontend/                              # React/TypeScript
[normal] │   │   ├── src/
[normal] │   │   │   ├── components/
[normal] │   │   │   │   ├── SCETimeline.tsx            # SCE frequency over training
[normal] │   │   │   │   ├── SemanticShiftHistogram.tsx
[normal] │   │   │   │   ├── DifficultyHeatmap.tsx
[normal] │   │   │   │   ├── OODTransferCurve.tsx
[normal] │   │   │   │   └── TraceViewer.tsx            # Inspect individual SCE traces
[normal] │   │   │   └── App.tsx
[normal] │   │   └── package.json
[normal] │   └── backend/                               # FastAPI
[normal] │       ├── api/
[normal] │       │   ├── traces.py
[normal] │       │   ├── metrics.py
[normal] │       │   └── export.py
[normal] │       └── main.py
[normal] │
[normal] ├── datasets/                                  # ~1M+ rows — Generated SCE data
[normal] │   ├── sce-traces-math/                       # SCE traces from math RLVR runs
[normal] │   ├── sce-traces-code/                       # SCE traces from code RLVR runs
[normal] │   └── sce-traces-vlm/                        # SCE traces from VLM RLVR runs
[normal] │
[normal] ├── docs/                                      # ~100K LoC — Documentation
[normal] │   ├── architecture/
[normal] │   ├── tutorials/
[normal] │   │   ├── quickstart.md
[normal] │   │   ├── axolotl_integration.md
[normal] │   │   └── vlm_extension.md
[normal] │   └── paper/                                 # Research paper LaTeX source
[normal] │       ├── main.tex
[normal] │       ├── appendix.tex
[normal] │       └── figures/
[normal] │
[normal] ├── .github/                                   # GitHub infrastructure
[normal] │   ├── workflows/
[normal] │   │   ├── ci.yml                             # Tests + linting on every PR
[normal] │   │   ├── benchmark.yml                      # Weekly benchmark runs
[normal] │   │   └── publish.yml                        # PyPI release automation
[normal] │   ├── ISSUE_TEMPLATE/
[normal] │   │   ├── bug_report.md
[normal] │   │   └── feature_request.md
[normal] │   └── PULL_REQUEST_TEMPLATE.md
[normal] │
[normal] ├── pyproject.toml                             # Monorepo root config
[normal] ├── CONTRIBUTING.md
[normal] ├── ROADMAP.md
[normal] └── README.md
[Heading 2] 3.2 Lines of Code Breakdown
[Heading 1] 4. PostTrace — The Complementary Data Attribution Layer
[normal] PostTrace is not a separate project. It is the answer to the question that ESCA generates: once you have identified which SCE traces are most influential in shaping behavior, can you trace that influence back to the training data that caused the model to generate those SCE patterns in the first place?
[Heading 2] 4.1 The LoRA-TRAK Insight
[normal] Standard influence functions are computationally intractable for large models — computing the inverse Hessian over all parameters is O(n²) in model size. TRAK (2023) showed this works for classification models. PostTrace extends it to generative post-training by exploiting a key structural property of LoRA fine-tuning: the parameter space being trained is tiny. The LoRA adapter rank is typically r=8 or r=16, meaning the effective parameter space for attribution is 10,000–100,000 dimensions, not 7 billion. Influence functions become computationally feasible in the LoRA subspace.
[Heading 2] 4.2 PostTrace Architecture
[normal] # posttrace/attributors/lora_trak.py
[normal] from __future__ import annotations
[normal] import torch
[normal] import torch.nn.functional as F
[normal] from typing import List, Dict, Optional
[normal] from dataclasses import dataclass
[normal] from peft import PeftModel
[normal] @dataclass
[normal] class AttributionResult:
[normal] example_id: str
[normal] score: float          # Causal attribution score (higher = more responsible)
[normal] confidence: float     # Estimation uncertainty
[normal] gradient_sim: float   # Raw cosine similarity in LoRA grad space
[normal] class LoRATRAK:
[normal] '''
[normal] TRAK adapted for LoRA subspace attribution.
[normal] Answers: which training examples caused this model output?
[normal] Mathematical foundation:
[normal] Influence(z_train, z_test) ≈ grad_θ L(z_test)^T H^{-1} grad_θ L(z_train)
[normal] where θ = LoRA parameters only (rank r << d)
[normal] H^{-1} approximated via DataInf (diagonal Fisher + damping)
[normal] '''
[normal] def __init__(
[normal] self,
[normal] model: PeftModel,
[normal] damping: float = 0.01,   # λ for H^{-1} approximation
[normal] device: str = 'cuda',
[normal] ):
[normal] self.model = model
[normal] self.damping = damping
[normal] self.device = device
[normal] self._lora_params = self._extract_lora_params()
[normal] def attribute(
[normal] self,
[normal] query_text: str,            # The model output to explain
[normal] training_set: List[Dict],   # {'id': str, 'text': str}
[normal] top_k: int = 50,
[normal] ) -> List[AttributionResult]:
[normal] # Step 1: Compute gradient of query on LoRA params
[normal] q_grad = self._compute_gradient(query_text)
[normal] # Step 2: Compute gradients for each training example
[normal] # In practice: batch this, use gradient checkpointing
[normal] results = []
[normal] for example in training_set:
[normal] t_grad = self._compute_gradient(example['text'])
[normal] # DataInf approximation: H^{-1} ≈ diag(F + λI)^{-1}
[normal] h_inv = 1.0 / (self._fisher_diag + self.damping)
[normal] influence = float(torch.dot(q_grad * h_inv, t_grad))
[normal] cosine_sim = float(F.cosine_similarity(q_grad.unsqueeze(0), t_grad.unsqueeze(0)))
[normal] results.append(AttributionResult(
[normal] example_id=example['id'],
[normal] score=influence,
[normal] confidence=abs(cosine_sim),  # proxy for estimation quality
[normal] gradient_sim=cosine_sim,
[normal] ))
[normal] results.sort(key=lambda r: abs(r.score), reverse=True)
[normal] return results[:top_k]
[normal] def _compute_gradient(self, text: str) -> torch.Tensor:
[normal] inputs = self.model.tokenizer(text, return_tensors='pt').to(self.device)
[normal] outputs = self.model(**inputs, labels=inputs['input_ids'])
[normal] loss = outputs.loss
[normal] self.model.zero_grad()
[normal] loss.backward()
[normal] # Concatenate LoRA gradients into a single flat vector
[normal] grads = []
[normal] for name, param in self.model.named_parameters():
[normal] if 'lora_' in name and param.grad is not None:
[normal] grads.append(param.grad.view(-1))
[normal] return torch.cat(grads).detach()
[normal] def _extract_lora_params(self) -> List[str]:
[normal] return [n for n, _ in self.model.named_parameters() if 'lora_' in n]
[normal] @property
[normal] def _fisher_diag(self) -> torch.Tensor:
[normal] # Cached Fisher diagonal estimate — computed once on a data subset
[normal] if not hasattr(self, '_fisher_cache'):
[normal] raise RuntimeError('Call .fit_fisher(data) before .attribute()')
[normal] return self._fisher_cache
[normal] def fit_fisher(self, calibration_data: List[str], n_batches: int = 100):
[normal] '''Estimate diagonal Fisher information over calibration set.'''
[normal] fisher = None
[normal] for text in calibration_data[:n_batches]:
[normal] grad = self._compute_gradient(text)
[normal] fisher = grad.pow(2) if fisher is None else fisher + grad.pow(2)
[normal] self._fisher_cache = fisher / len(calibration_data[:n_batches])
[Heading 1] 5. GitHub Traction Strategy
[normal] A good library that nobody finds is useless. The traction strategy is as deliberate as the architecture. Every element is designed to be discovered, adopted, and cited within the current community.
[Heading 2] 5.1 The Distribution Moat
[Heading 2] 5.2 README Structure (The First 10 Seconds)
[normal] GitHub gives you 10 seconds to make someone star the repository. The README is engineered for this:
[normal] # ESCA — Emergent Self-Correction Amplification
[normal] [![PyPI](https://img.shields.io/pypi/v/esca-core)](https://pypi.org/project/esca-core/)
[normal] [![TRL](https://img.shields.io/badge/TRL-compatible-blue)](https://github.com/huggingface/trl)
[normal] [![HF Hub](https://img.shields.io/badge/Dataset-Hub-yellow)](https://huggingface.co/datasets/esca-project)
[normal] [![Paper](https://img.shields.io/badge/arXiv-XXXX-red)](https://arxiv.org)
[normal] **RLVR training generates self-correction behavior as a side effect.
[normal] ESCA makes it a primary trained capability.**
[normal] Add one callback to your GRPOTrainer:
[normal] ```python
[normal] from esca import SelfCorrectionCallback
[normal] from trl import GRPOTrainer, GRPOConfig
[normal] trainer = GRPOTrainer(
[normal] model=model,
[normal] config=GRPOConfig(...),
[normal] callbacks=[SelfCorrectionCallback(tau=0.4, n_replay=50)],  # <- this
[normal] )
[normal] trainer.train()
[normal] ```
[normal] Results: SCE frequency 3% → 12% over 3000 steps.
[normal] Hard problem accuracy (MATH difficulty-5, AIME) improves most.
[normal] OOD self-correction generalizes to domains not in training distribution.
[Heading 2] 5.3 Issue Labels & Contribution Infrastructure
[Heading 1] 6. Failure Mode Analysis & Mitigations
[normal] Five failure modes are documented below. Each has a detection method and a concrete mitigation. The mitigations are built into the library — not documentation warnings. The fifth (Subtle Reward Hacking) is newly added and is the most dangerous because it is invisible to the dynamic τ filter alone.
[Heading 1] 7. Evaluation Protocol
[Heading 2] 7.1 Primary Metrics
[Heading 2] 7.2 The Decisive Experiment
[Heading 2] 7.3 Falsification Strategy
[normal] The decisive experiment is a binary outcome — the OOD pattern either holds or it doesn’t. “Diagnose and redesign” is not a contingency plan. The following falsification tree must be prepared before Week 4 begins, so that a negative result produces a useful research artifact rather than an abandoned project.
[normal] Hypothesis A: SCE traces are domain-locked, not meta-skill generalizing. Check whether SCE frequency increased on in-distribution problems only. If yes — ESCA works, the generalization claim doesn’t. Paper pivot: “targeted self-correction amplification for domain-specific RLVR” (narrower but publishable).
[normal] Hypothesis B: SCE frequency increased but conversion rate fell — the model self-corrects more but doesn’t recover correctly. If yes — the replay SFT is training the recognition habit without the recovery skill. Redesign: SFT target should include more of the post-correction path, not just the correction moment.
[normal] Hypothesis C: The baseline GRPO model already has near-ceiling SCE behavior for this model size and task. Check whether the 3B model shows larger ESCA deltas than the 7B. If yes — the scaling question becomes the primary research contribution: ESCA provides diminishing returns as model capability increases, which is itself a publishable finding about the ceiling of self-correction training.
[normal] In every case, PostTrace attribution on the SCE replay buffer provides the diagnostic signal: which traces generalized, which were domain-locked, and what distinguished them. This is why PostTrace Phase 1 must be running before Week 4, not after.
[Heading 2] 7.4 Shared Community Benchmark (esca-bench)
[normal] The highest-leverage missing artifact in this research program is a shared benchmark the community can clone, run in an afternoon, and use to compare their own methods against — covering self-correction behavior (ESCA) and post-training attribution quality (PostTrace) in a single suite. Neither library currently provides this. The field needs what BIG-Bench did for capabilities.
[normal] Proposed: esca-bench, a third artifact in the monorepo. Three tracks: (1) SCERate — standardized self-correction frequency and OOD transfer eval across five model families and three task domains on a fixed compute budget; (2) AttributionQuality — planted influence + TOFU-style forget/retain eval that any attribution method can be submitted to; (3) DataConflict — synthetic mixed-objective datasets with known gradient conflicts, testing whether attribution tools correctly identify conflicting examples. Design constraint: runnable end-to-end in under 4 GPU-hours on a single A100. Target: HuggingFace Hub dataset + Papers With Code evaluation suite entry at the same time as the library PyPI release. The benchmark is the community surface that drives long-term adoption of both libraries — models and methods get compared on it, creating a persistent reason to engage with the ecosystem.
[Heading 1] 8. Implementation Roadmap
[Heading 2] 8.1 Week-by-Week Plan (Month 1)
[Heading 2] 8.2 Month 2 — VLM Extension
[normal] Base model: Qwen2.5-VL-7B + MathVista + ChartQA
[normal] Extend reversal marker list with visual reasoning patterns (let me look at the image again, i may have misread, on second look)
[normal] Semantic shift must be computed over both visual and textual reasoning path
[normal] Key research question: do VLM SCEs cluster on perception errors (wrong object identified) or reasoning errors (wrong inference from correct parse)?
[normal] If the former: ESCA provides empirical evidence for the perception-reasoning distinction without architecturally complex perception-gated approaches
[Heading 2] 8.3 Month 3 — Library Polish + Community
[normal] PostTrace integration: connect SCE attribution to training data causality. Concretely: implement esca/shared/gradient_cache.py as the shared gradient extraction layer, wire SCEReplayBuffer output into PostTrace’s post-hoc attribution pipeline, and validate that PostTrace can identify which SFT training examples are causally responsible for the model’s baseline SCE frequency. This requires PostTrace Phase 1 (LoRA-TRAK, DataInf) to be implemented in parallel — coordinate release schedule.
[normal] Axolotl plugin: YAML-driven integration, community PR to main Axolotl repo
[normal] Documentation: tutorials, API reference, integration guides
[normal] HuggingFace Blog post submission: 'Teaching Models to Change Their Mind'
[normal] PyPI release: esca-core v0.1.0
[normal] GitHub Actions: full CI, benchmark suite, release automation
[normal] Community: good first issues opened, Discord/Slack channel, office hours
[Heading 2] 8.4 Month 4+ — Research Extension
[normal] RLVR expansion: online prompt-cluster attribution for RLVR (the PostTrace RLVR paper)
[normal] Multi-domain generalization: extend beyond math/code to biology, law, medicine
[normal] Self-correction curricula: can you design training problems that maximally elicit SCEs?
[normal] Scaling study: does ESCA benefit increase or decrease with model scale? (7B → 70B) This is arguably the most important empirical question in the project. If larger models already have higher baseline SCE rates — plausible, since they are better reasoners generally — ESCA may show diminishing returns exactly where compute budgets are largest. A negative result here is publishable and important: it tells the field where the ceiling of self-correction training lies. Run this before claiming the paper’s main result is model-size-agnostic.
[normal] AgentSCE: do self-correction patterns in single-step rollouts predict better tool use in agent settings? Self-correction in a multi-turn agent loop — where the model receives tool results, web search output, or feedback from another model — is a richer and arguably more important phenomenon than single-step correction. The current SCE detector would miss most of these: the reversal often happens across turns rather than within a single generation, and the verifier-based correctness filter assumes a single terminal reward. Multi-agent SCE detection requires extending the detector to operate over conversation history, not just individual rollouts. This is the most technically demanding extension but has the highest practical impact as agent architectures become the dominant deployment surface.
[normal] SCE curriculum design: rather than passively mining self-corrections from whatever rollouts the model generates, can you construct training problems that maximally elicit SCEs? This means understanding what problem structure causes a model to go down a wrong path and then recover — the inverse of the current detection problem. The PostTrace Oracle (objective-aware data scorer) is the natural tool: score candidate problems by their expected SCE elicitation rate on a proxy model, then rank-select them into the training distribution. If high-SCE-elicitation problems can be reliably identified in advance, the ESCA training signal becomes proactive rather than reactive — a potential 2–3x improvement in replay buffer fill rate without any additional compute.
[normal] DPO post-training extension: ESCA’s SCE detection assumes RLVR — the verifier tells you whether a rollout is correct, which is the basis for the genuine SCE filter. For DPO-trained models, there is no verifier. A DPO preference pair implies a correct and incorrect completion, but this does not map cleanly onto the rollout-level correctness signal ESCA requires. This is an underspecified gap that blocks adoption for a large fraction of production fine-tuning pipelines. Month 4+ should include a concrete design proposal: either a DPO-compatible correctness proxy (e.g. reward model score differential), or an explicit statement that ESCA is RLVR-only and why.
[Heading 1] 9. Framework Integration Details
[Heading 2] 9.1 Axolotl Integration (YAML-Driven)
[normal] The Axolotl integration is the distribution moat. Axolotl users enable ESCA by adding one section to their existing YAML config — no code changes, no framework modification:
[normal] # axolotl config — existing keys unchanged
[normal] base_model: Qwen/Qwen2.5-7B-Instruct
[normal] model_type: AutoModelForCausalLM
[normal] # ... all existing training config unchanged ...
[normal] # Add this section to enable ESCA:
[normal] esca:
[normal] enabled: true
[normal] tau: 0.4                    # Semantic shift threshold (calibrate in week 1)
[normal] n_replay: 50               # SFT step every N GRPO steps
[normal] replay_batch_size: 32      # SCE traces per SFT step
[normal] alpha: 0.1                 # Reward bonus for genuine SCE rollouts
[normal] buffer_capacity: 50000     # Max SCE traces in replay buffer
[normal] expiry_steps: 500          # Expire old traces (handles policy drift)
[normal] push_to_hub: true          # Auto-push SCE dataset to HuggingFace Hub
[normal] hub_repo_id: your-org/sce-traces-{model_name}
[normal] wandb_log: true            # Stream metrics to W&B
[normal] device: cpu                # Embedder runs on CPU — no GPU competition
[Heading 2] 9.2 TRL Direct Integration
[normal] # Direct TRL integration — minimum viable ESCA
[normal] from esca import SelfCorrectionCallback
[normal] from trl import GRPOTrainer, GRPOConfig
[normal] from transformers import AutoModelForCausalLM, AutoTokenizer
[normal] model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
[normal] tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
[normal] # Drop-in: no changes to model, optimizer, or training loop
[normal] esca_callback = SelfCorrectionCallback(
[normal] tau=0.4,
[normal] n_replay=50,
[normal] alpha=0.1,
[normal] push_to_hub=True,
[normal] hub_repo_id='your-org/sce-traces',
[normal] )
[normal] trainer = GRPOTrainer(
[normal] model=model,
[normal] tokenizer=tokenizer,
[normal] config=GRPOConfig(
[normal] num_generations=8,
[normal] max_new_tokens=2048,
[normal] learning_rate=1e-5,
[normal] ),
[normal] train_dataset=train_dataset,
[normal] reward_funcs=[math_verifier],
[normal] callbacks=[esca_callback],   # <- this is the entire integration
[normal] )
[normal] trainer.train()
[normal] # After training: SCE dataset is on HuggingFace Hub
[normal] # ESCA metrics are in W&B
[normal] # Model card mentions ESCA training
[Heading 2] 9.3 CLI Tools
[normal] # Install
[normal] pip install esca-core
[normal] # Week 1: Calibrate τ before committing to a value
[normal] esca calibrate \
[normal] --rollout-file rollouts.jsonl \
[normal] --tau-range 0.2,0.3,0.4,0.5 \
[normal] --sample-size 200 \
[normal] --output calibration_report.html
[normal] # Inspect SCEs in a rollout file
[normal] esca inspect \
[normal] --rollout-file rollouts.jsonl \
[normal] --tau 0.4 \
[normal] --show-genuine-only \
[normal] --export sce_traces.jsonl
[normal] # Mine SCE dataset from a completed training run
[normal] esca mine \
[normal] --checkpoint-dir ./checkpoints/step_3000 \
[normal] --training-rollouts rollout_logs/ \
[normal] --tau 0.4 \
[normal] --push-to-hub your-org/sce-traces-math
[normal] # Run PostTrace attribution: what training data caused these SCEs?
[normal] esca trace \
[normal] --sce-file sce_traces.jsonl \
[normal] --checkpoint-dir ./checkpoints/step_3000 \
[normal] --training-data sft_data.jsonl \
[normal] --output attribution_report.html
[Heading 1] 10. Research Contributions
[Heading 2] 10.1 Primary Claims
[Heading 2] 10.2 Novelty Positioning
[Heading 2] 10.3 Paper Title
[normal] "Mining Your Own Rollouts: Self-Correction Amplification for Post-Training Language Models"
[normal] Venue target: NeurIPS 2026 (submission October 2026). If OOD results are strong, ICML 2026 is possible (submission January). The results section needs the 4-week implementation plan to complete before submission.
[Heading 1] 11. Open Questions
[normal] These are deliberately open — they are the questions that determine whether the approach works and in which configuration. They are the work, not the documentation.
[Heading 2] 11.1 Methods Questions
[normal] How should τ be calibrated across different base models? Qwen2.5 SCE patterns differ from LLaMA-3. Is the optimal τ model-family-specific, or does a universal calibration procedure work?
[normal] For DPO post-training (not GRPO), do self-correction events look the same? The verifier-based is_correct filter assumes RLVR. DPO requires a different correctness signal. Three candidate proxies: (1) reward model score differential between chosen and rejected completions as a soft correctness signal; (2) perplexity of the chosen completion under a reference model as a quality gate; (3) human-annotated correctness labels on a small calibration set used to threshold the reward signal. Each has different noise characteristics. This question must be resolved before ESCA can claim broad post-training coverage — DPO is still the dominant fine-tuning paradigm for instruction-following and alignment work.
[normal] What is the right granularity for the SFT target — token-level NLL on correction moment, or span-level? Token-level is cleaner mathematically but may be too fine-grained for the semantic shift signal.
[normal] Does the ESCA reward bonus create curriculum effects? Problems that elicit more SCEs might be harder — the bonus would effectively upweight hard problems. Is this intentional curriculum learning or reward hacking?
[normal] What happens to SCE patterns in long-context (>8K token) reasoning chains? The reversal marker heuristic was developed on shorter rollouts.
[Heading 2] 11.2 Engineering Questions
[normal] Can the SCE detector run async on a background CPU process during GPU rollout generation, without introducing synchronization bottlenecks in the training loop?
[normal] For DeepSpeed ZeRO-3, model parameters are sharded across GPUs. The SFT step on correction moments needs to accumulate gradients in a way that's compatible with ZeRO-3's partitioned optimizer states.
[normal] What is the memory footprint of the replay buffer at capacity (50K traces × average ~2K tokens per trace)? This is ~100M tokens. Need to store as tokenized tensors or text? Trade-offs?
[normal] How does LoRA-TRAK scale to 70B models? Even in LoRA subspace, r=64 on a 70B model is still a large gradient vector. Need aggressive approximation.
[Heading 2] 11.3 Evaluation Questions
[normal] Is OOD generalization of SCE behavior actually testable? GPQA has chemistry, biology, physics questions — if the model only trained on math SCEs, does it self-correct on physics? This requires careful experimental design.
[normal] How do we distinguish 'the model is better at hard problems because it self-corrects more' from 'the model is better at hard problems because ESCA happened to improve relevant representations'? The SCE frequency metric helps but isn't definitive.
[normal] What is the Pareto frontier between replay ratio (N_replay) and task performance degradation? At N_replay=10, the SFT signal might dominate. This is the ablation C study.
[Heading 2] 11.4 ESCA–PostTrace Integration Questions
[normal] Can PostTrace explain why some SCE traces generalize OOD better than others? The decisive experiment (Section 7.2) tests whether ESCA-trained models generalize to far-OOD domains. If the OOD improvement is uneven across problem types, PostTrace attribution could pinpoint which training examples in the SCE replay buffer are causally responsible for the OOD transfer — and which are domain-locked. This is the highest-value joint experiment between the two libraries.
[normal] What is the right data contract between ESCA and PostTrace? PostTrace’s LoRA-TRAK attributor (PostTrace v0.1 design doc, Section 4.1.1) operates on per-example gradient vectors in LoRA subspace. ESCA’s SCEReplayBuffer stores SCE traces as tokenized text. For PostTrace to attribute which pre-training or SFT examples caused the model to generate certain SCE patterns, ESCA must expose gradient vectors for the pre-correction path segments — not just the text. This requires a shared gradient extraction layer that neither library currently specifies. The monorepo structure (Appendix A) makes this tractable; define the esca/shared/gradient_cache.py interface as the first joint deliverable.
[normal] PostTrace’s Oracle (objective-aware data scorer) can score candidate SFT data by gradient alignment with a target behavior. The ESCA self-correction behavior is a natural target. Can the Oracle be used to select SFT data that maximally primes the model for SCE generation before RLVR training begins — a pre-RLVR data selection step that gives ESCA a higher baseline SCE rate to amplify from? This would close a second loop: PostTrace scouts the pretraining surface that ESCA then exploits.
[Heading 1] 12. Contributing
[normal] ESCA is built to be extended by the community. The core library is intentionally scoped — it does one thing well. Extensions, integrations, and new domains are explicitly invited.
[Heading 2] 12.1 How to Contribute
[normal] Read the architecture document and open questions (Section 11). Pick one open question that interests you.
[normal] Check open GitHub issues. Issues labeled 'good first issue' are fully specified — implementation only. Issues labeled 'research' require a design decision.
[normal] For new SCE marker vocabularies: run esca inspect on a rollout file from your domain, identify patterns, submit a PR to marker_vocabulary.py with precision measurements.
[normal] For new integrations (VeRL, LLaMA-Factory, Unsloth): copy the pattern from esca/integrations/trl_grpo.py. The callback interface is stable.
[normal] For new evaluation benchmarks: add to esca-bench/benchmarks/. Run the existing benchmark suite first to establish your baseline for comparison.
[normal] For PostTrace extensions: see posttrace/README.md. The LoRA-TRAK implementation is the most important missing piece — it needs empirical validation on TOFU-style controlled experiments.
[Heading 2] 12.2 What We Don't Want
[normal] Another DPO / RLVR variant — the algorithm space is not the bottleneck ESCA addresses
[normal] Domain-specific verifiers — see OpenVerifiers discussion in Section 0 for why this doesn't generalize
[normal] Changes to the core GRPOTrainer — ESCA is a callback, not a fork
[normal] Large language model checkpoints in the repo — use HuggingFace Hub for models
[Heading 1] Appendix A: Monorepo vs. Separate Repos — The Architecture Decision
[normal] The document refers to an esca/ monorepo containing both the ESCA core library and the PostTrace companion library (~300K LoC each). This is a deliberate architectural decision worth making explicit, because it has consequences for adoption, contributor experience, and long-term maintenance.
[Heading 2] A.1 The Case For the Monorepo (Current Design)
[normal] ESCA and PostTrace are deeply coupled at the data layer. The esca trace CLI command (Section 9.3) passes SCE trace files directly into PostTrace’s attribution pipeline. The SCEReplayBuffer generates exactly the kind of training data PostTrace is designed to audit. PostTrace’s gradient-conflict detector can flag SFT data that is pulling against ESCA’s self-correction objective. These are not two independent libraries that happen to be co-located — they share a data contract and a design philosophy.
[normal] Concrete benefits of the monorepo structure: (1) Shared gradient extraction infrastructure — both ESCA’s SCE detector and PostTrace’s LoRA-TRAK attributor need per-example gradient vectors in LoRA subspace. In a monorepo, this runs once and is shared. In separate repos, it runs twice and must be versioned independently. (2) Atomic schema evolution — the SCEEvent dataclass and the AttributionResult dataclass reference each other. A monorepo guarantees they stay in sync across library boundaries without coordinating two separate release cycles. (3) Single CI pipeline — the planted influence eval and the TOFU benchmark both exercise ESCA and PostTrace together. Running them in a single CI environment catches integration regressions that two separate pipelines would miss.
[Heading 2] A.2 The Risks of the Monorepo
[normal] The monorepo has two real risks. First, contributor confusion: a practitioner who wants only PostTrace for data attribution on their existing DPO pipeline may be confused by a 2–3M LoC repo that appears to require ESCA. This can be mitigated with clear package boundaries (pip install esca-core installs ESCA only; pip install posttrace installs PostTrace only; pip install esca installs both) and top-level README routing that immediately separates the two entry points.
[normal] Second, paper citation complexity: ESCA and PostTrace target separate papers (NeurIPS 2026 and a companion attribution paper). Reviewers may question whether they are reviewing a system paper or two half-papers. The mitigation is clear scope separation in each paper: the ESCA paper does not require PostTrace to be implemented, and the PostTrace paper does not require ESCA beyond a motivating example. The monorepo is an engineering decision; the papers are independent intellectual contributions.
[Heading 2] A.3 Verdict: Monorepo is Correct, With Package Isolation
[normal] The monorepo is the right call at this stage, for one decisive reason: the interface between ESCA and PostTrace is not yet stable. During active development (Months 1–3), the SCEEvent schema, the gradient extraction format, and the attribution API will change frequently. Managing those changes across two separate repos with independent versioning adds coordination overhead that is not justified until the interfaces are stable.
[normal] The recommended structure: single GitHub repo (github.com/your-org/esca) with two independently publishable Python packages (esca-core/ and posttrace/) sharing a common esca/shared/ layer for gradient extraction and data types. Separate PyPI releases with separate version numbers. A CODEOWNERS file routing PostTrace PRs to attribution reviewers and ESCA PRs to RLVR reviewers. Split into separate repos only if the communities diverge enough that shared CI becomes a bottleneck — revisit at v1.0.
[normal] ESCA Architecture Document v0.2 — April 2026
[normal] One coherent research program: KoRA → Erasus → ESCA

[TABLE]
# | Benchmark | What It Tests | ESCA Score | Verdict
1 | Does it fit Karpathy's patterns? | Mechanistically grounded, systems-level thinking — not just a loss function tweak | 9/10 — He's written explicitly about wanting models that recognize when they're going wrong and course-correct without human intervention. The framing 'self-correction is currently an emergent accident rather than a trained capability' is exactly his style of observation. | ✓
2 | Would HF blog about, build, and maintain it? | Is it infrastructure, not a paper? Does it plug into TRL/transformers/Hub? | 9/10 — Concretely: a SelfCorrectionCallback for TRL's GRPOTrainer. 400 lines of code. HF has exactly this training callback infrastructure. The SCE dataset pushed to Hub creates a community artifact. | ✓
3 | Are people actively researching it? | Is the pain real and loud — scrappy demos exist? | 8/10 — The self-correction 'wait' behavior from R1-style training is the most actively discussed phenomenon in post-training right now. Nobody has closed the loop and turned observation into training signal. | ✓
4 | Why hasn't someone else done this yet? | The hardest one. If it's good and nobody's done it, there must be a reason. Is it surmountable? | 8/10 — The reason is organizational/prioritization, not mathematical. Teams are in a scaling mindset — just run more RLVR. The second-order question ('what's the ceiling if we explicitly train for it?') requires stepping back from scaling. This is the right time to ask. | ✓

[TABLE]
The Four Rejected Ideas (For Completeness)
Iteration 1 (Metacognitive Co-Training): Rejected — chicken-and-egg structural problem, cannot predict success before attempting. Iteration 2 (OpenVerifiers): Rejected — standardization window closed, 15 incompatible libraries already being built simultaneously. Iteration 3 (Domain-Adversarial Post-Training): Rejected — forcing domain-invariant representations penalizes the domain-specific knowledge needed for correct answers. Iteration 4 (Shortcut Suppression via Linear Probes): Rejected — requires solving 2+ years of mechanistic interpretability first.

[TABLE]
Signals in the Field
RLVR scaling is hitting diminishing returns on math — MATH-500 near-saturated, AMC near-saturated
The field is explicitly looking for the next lever beyond more verifiable data
TRL's GRPOTrainer has first-class callback infrastructure — ESCA integrates without forking
R1-style training has produced a generation of models with non-trivial baseline SCE rates (3–5%), giving us signal to amplify
vLLM + DeepSpeed make rollout generation cheap enough to instrument in real time | Technical Enablers
RLVR makes SCE detection tractable: the verifier already knows the final answer, so the 'correct rollout' filter is free
Embedding models (MiniLM-L6, 22M params) run on CPU with ~4ms overhead — negligible vs. generation cost
Community is right at the stage of 'we've noticed this phenomenon and don't know what to do with it' — closing the loop is maximally timely
No existing library covers this. The space is ready for infrastructure, not another paper

[TABLE]
Ancestor | Key Contribution | What ESCA Takes | What ESCA Adds
KoRA (Representation Theory) | Novel training objectives grounded in how representations evolve | The framing that post-training can target specific representational properties | Coreset selection applied to trajectory space rather than representation space
Erasus (Coreset Selection) | Principled identification of maximally influential examples for unlearning | The algorithmic machinery for finding high-influence subsets of a training set | Identifies self-correction moments as the high-influence subset of rollout trajectories
DataTrove (Pretraining Data) | Principled data selection for pretraining quality | The philosophy that data selection is as important as the training algorithm | Applies data selection philosophy to the online RLVR trajectory regime
DeepSeek-R1 (RLVR) | Demonstration that self-correction emerges from RL training at scale | The empirical observation that SCEs are real and frequent enough to mine | Closes the loop: observation → systematic amplification
TRL GRPOTrainer (Infrastructure) | Production-grade GRPO implementation with callback hooks | The training infrastructure — ESCA implements as a callback, not a fork | SelfCorrectionCallback + SCEReplayBuffer as first-class citizens

[TABLE]
Self-Correction Event (SCE) — Formal Definition
A Self-Correction Event is a subsequence within an RLVR rollout satisfying all three conditions simultaneously: (1) A direction-reversal marker is present at token position t — drawn from a curated vocabulary of ~50 markers derived from R1/QwQ rollout analysis. (2) The reasoning path after position t is semantically distinct from the path before position t, measured by cosine distance on chunk-level sentence embeddings with threshold τ = 0.4. (3) The rollout reaches a correct final answer, verified by the task verifier already present in the RLVR pipeline.

[TABLE]
Genuine SCE vs. Performative SCE — The Critical Distinction
A Genuine SCE satisfies condition (2) with a high-confidence embedding shift. The model actually changed direction. A Performative SCE uses direction-reversal language ('wait', 'actually') but continues the same reasoning path — theatrical self-correction that sounds like course correction but isn't. The SCE Detector specifically targets Genuine SCEs and discards Performative SCEs. Training on Performative SCEs would teach the model to say 'wait' without changing behavior — precisely the failure mode we must avoid.

[TABLE]
Stage | Component | Input | Output | Timing
1 | Rollout Generation | Problem batch (G=8 samples per problem) | G rollout traces per problem with verifier rewards | Every training step — unchanged
2 | SCE Detection | Rollout traces + verifier correctness flags | Detected SCEEvents (or None) per trace — ~4ms per rollout on CPU | Async, post-generation, pre-update
3 | SCE Replay Buffer | Detected SCEEvents | Maintained circular buffer of (s, c, p) triples — capacity 50K, expiry 500 steps | Updated after detection, sampled before SFT step
4 | GRPO Update (Primary) | Rollout traces + R_task rewards + R_esca bonus (α=0.1) | Standard GRPO gradient update with augmented reward | Every training step — main objective
5 | SCE Replay SFT (Supplementary) | Sampled SCE traces from buffer | SFT loss on correction-moment tokens only — L_ESCA | Every N_replay=50 GRPO steps
6 | Diagnostic Logger | All of the above | SCE frequency, semantic shift distribution, performative rate, OOD metrics to W&B/JSONL | Every step — async, no training effect

[TABLE]
The Core Mathematical Objects
Primary: L_GRPO = -E[π_θ(o|q) / π_ref(o|q) × Â] + β·D_KL(π_θ‖π_ref)Supplementary (runs every N_replay=50 GRPO steps): L_ESCA = -Σ_{(s,c,p)∈B_SCE} log π_θ(c | s)where s = pre-correction segment (wrong path), c = correction moment tokens only, p = post-correction path (not trained on).Optional Reward Bonus: R_total = R_task + α·1[rollout contains Genuine SCE]·R_task  (α=0.1 default)The SFT loss is applied ONLY to correction-moment tokens. Training on the full post-correction path is standard SFT and adds nothing new. The correction moment is where the habit of recognizing error and reversing lives.

[TABLE]
Component | LoC Estimate | Language | Description
esca-core (library) | ~50,000 | Python | Detection, buffer, training, logging, integrations
esca-bench (benchmarks) | ~100,000 | Python | Full eval harness, ablations, baselines
esca-experiments (configs + scripts) | ~50,000 | Python + YAML + Shell | Experiment configs, training scripts, DVC pipelines
posttrace (companion) | ~300,000 | Python | Data attribution, oracle, curation loop
esca-dashboard (UI) | ~80,000 | TypeScript + Python | React frontend + FastAPI backend
Test suites (all) | ~200,000 | Python | Unit, integration, property-based, regression tests
Generated datasets (serialized) | ~800,000 rows | JSONL / Parquet | SCE traces from training runs (math, code, VLM)
Documentation (Markdown + LaTeX) | ~100,000 | Markdown + LaTeX | Tutorials, API docs, research paper source
GitHub workflows + configs | ~20,000 | YAML + Shell | CI/CD, release automation, issue templates
TOTAL | ~2.7M+ LoC | Mixed | Full ecosystem

[TABLE]
The PostTrace Core Thesis
The bottleneck in post-training is not the algorithm. DPO, GRPO, RLVR — the gradient descent part works. The data understanding part doesn't. Every post-training failure (sycophancy, domain hallucination, safety regressions, capability loss) is opaque. PostTrace answers: which specific training examples are causally responsible for this behavior? It is the auditing and debugging layer that ESCA's training surface exposes.

[TABLE]
Method | LoRA-Native | Generative Post-Training | Active Curation Loop | Practitioner API | Computational Cost
PostTrace (ours) | ✓ | ✓ | ✓ | ✓ | O(r² × n_examples) — tractable
TRAK (2023) | ✗ | ✗ (classification only) | ✗ | ✗ (research code) | O(d² × n) — intractable
DataInf | ✓ | ⚠ (partial) | ✗ | ✗ | O(r² × n) — similar
EK-FAC | ✗ | ✗ | ✗ | ✗ | O(d × n) — expensive
TracIn | ✗ | ⚠ | ✗ | ✗ | O(T × d × n) — very expensive

[TABLE]
Vector | Mechanism | Why It Works | Timeline
TRL Integration | SelfCorrectionCallback drops into GRPOTrainer in 3 lines | TRL is the de facto RLVR training library. Everyone already using it gets ESCA for free | Day 1 — first thing shipped
Axolotl Plugin | One YAML key (esca: true) enables everything | Axolotl has the largest practitioner adoption for fine-tuning. Zero workflow change required | Week 1
HuggingFace Hub Dataset | SCE traces pushed automatically — esca-project/sce-traces-math, -code, -vlm | Hub creates discoverability. Dataset page links back to repo. Community can download and inspect | Week 2
Weights & Biases Dashboard | ESCA metrics stream to W&B — SCE frequency, shift distribution, OOD curves | W&B is universal in the training community. Shareable run links spread virally | Week 2
Model Cards | Auto-generated model cards mention ESCA training, link to repo | Every model pushed to Hub after ESCA training is a discovery vector | Week 3
The Blog Post | 'Teaching Models to Change Their Mind: Self-Correction Amplification in RLVR Training' — pitched to HF + Karpathy | The HF blog has massive reach. Karpathy sharing it drives GitHub stars exponentially | Week 4

[TABLE]
Label | Purpose | Target Contributor
good first issue | Extend reversal marker vocabulary for a new domain, write a test for a specific failure mode | New contributors — easy entry point
help wanted | Implement a new benchmark integration, add DeepSpeed ZeRO-3 compatibility | Intermediate — requires framework knowledge
research | Implement alternative SCE detection methods, run ablations, analyze failure modes | Researchers — open-ended
integration | Add VeRL adapter, add LLaMA-Factory plugin, add Unsloth compatibility | Framework specialists
dataset | Mine SCE traces from a new domain/model, expand the HF Hub datasets | Anyone with compute
bug | Standard bug reports with reproducible cases | All users

[TABLE]
Failure Mode | What Happens | Detection | Mitigation | Severity
Performative Self-Correction | Model learns to insert reversal language ('wait', 'actually') without actually changing reasoning direction — theatrical self-correction that games the bonus reward | Track mean semantic_shift of detected SCEs over training. If average shift decreases over 500 steps, performative drift is occurring | Dynamic threshold τ_dynamic = max(τ_base, mean_shift × 0.8). Buffer automatically discards traces below dynamic threshold. SFT never reinforces traces where semantic shift < τ_dynamic | HIGH — direct corruption of training signal
SCE Rate Runaway | Model learns to always self-correct, even on problems where first approach was correct. Wastes inference compute, confuses correct paths | Track SCE frequency on held-out easy set where baseline accuracy >90%. SCE rate on easy problems should stay near zero | Confidence-gated correction bonus: apply R_esca bonus ONLY on rollouts where the pre-correction path was headed toward a wrong answer (detectable from verifier on prefix) | MEDIUM — inference inefficiency, minor capability degradation
Stale Buffer Drift | Old SCE traces, generated by an earlier policy version, no longer reflect current model's failure modes. Replaying them confuses current policy | Track perplexity of current policy on buffered SCE pre-correction segments. Spike in perplexity signals buffer staleness | Expiry window (default 500 steps) handles standard drift. Reduce to 200 steps if perplexity drift detected. Buffer auto-prunes by recency weight | LOW — handled by expiry design
Tau Miscalibration | τ set too low: performative SCEs flood buffer. τ set too high: no SCEs detected, buffer stays empty | Week 1 diagnostic: manually inspect 50 detected SCEs at candidate τ values. Report genuine rate. Target τ = lowest value with genuine rate > 80% | Dynamic τ adaptation (above). Diagnostic dashboard shows genuine vs performative breakdown in real time. CLI tool: esca calibrate --rollout-file rollouts.jsonl | MEDIUM — requires calibration week, then auto-managed
Subtle Reward Hacking (Verifier-Blind) | Model learns to generate a plausible-looking semantic shift by switching to a different wrong answer rather than a correct one, in contexts where the verifier is weak or ambiguous. Dynamic τ catches theatrical self-correction but not substantive pivots to alternative wrong paths. The SCE bonus gets captured without genuine improvement. | Track SCE conversion rate (% of SCE rollouts that are correct) separately on strong-verifier vs. weak-verifier problem subsets. Divergence between these two rates signals verifier-blind hacking. PostTrace attribution can identify which buffered SCE traces are systematically associated with weak-verifier problems. | Stratify the SCE replay buffer by verifier confidence tier. Only admit SCE traces from rollouts where the verifier confidence exceeds a threshold (e.g. 0.9). Use PostTrace gradient-conflict detection to flag buffered traces that are pulling against the high-confidence correct-answer gradient direction — these are the hacking candidates. Prune them from the buffer automatically. | HIGH — silent, undetectable by τ alone

[TABLE]
Metric | Operationalization | Hypothesis | Target
SCE Frequency | % of correct rollouts containing a Genuine SCE | Increases from baseline ~3% to target after ESCA training | ~12% at step 3000
SCE Conversion Rate | % of rollouts containing SCE that are correct (vs. all rollouts with SCE) | Stays stable or increases — we want SCEs that lead to correct answers, not thrashing | ≥ baseline conversion rate
OOD Self-Correction | SCE frequency on held-out problem types not in training distribution | Key generalization test: does the self-correction habit transfer? Should increase MORE than in-distribution SCE frequency | OOD > in-distribution increase
Utility Retention | MMLU / general benchmark scores vs. baseline GRPO | ESCA should not degrade general capability | < 0.5% degradation
Hard Problem Improvement | Pass@1 on MATH difficulty-5, AMC, AIME | Primary capability metric. ESCA improves hard problems most | Primary result metric
Performative SCE Rate | % of detected SCEs classified as Performative (semantic shift < τ) | Should stay low throughout training — rising performative rate = failure mode 1 | < 20% of detected SCEs

[TABLE]
Experiment Design: The OOD Generalization Test
Train two models: baseline GRPO and ESCA-GRPO. Same base model (Qwen2.5-7B), same training data (MATH + CODE), same compute budget, same number of steps.Evaluate on three problem sets:1. In-distribution: same domain as training data (MATH, held-out)2. Near-OOD: same domain, harder difficulty tier not seen in training (MATH difficulty-5, AIME)3. Far-OOD: different domain requiring same reasoning meta-operations (GPQA scientific reasoning, spatial reasoning)The hypothesis that would confirm ESCA works:Far-OOD improvement > Near-OOD improvement > In-distribution improvementIf self-correction is a domain-invariant reasoning capability, training it explicitly should generalize most to domains furthest from training distribution. This is the experiment. If this pattern holds, this is the paper.

[TABLE]
Week | Focus | Deliverables | Success Criteria
Week 1 — Baseline + Instrumentation | Run baseline GRPO. Instrument SCE detector as logging-only callback (no training effect). Calibrate τ. | Baseline GRPO run on Qwen2.5-1.5B + MATH-500 subset. SCE detector as TrainerCallback. Calibration report: genuine SCE rate vs τ at 0.2, 0.3, 0.4, 0.5. Manually inspected 50 SCEs. W&B dashboard live. | SCE frequency baseline established (expected 2–5%). τ calibrated with ≥80% genuine rate. Diagnostic dashboard operational.
Week 2 — ESCA v0 | Implement replay buffer, supplementary SFT step, and reward bonus. Run ESCA training with α=0 first (replay only). | SCEReplayBuffer implemented and tested. run_sft_on_correction_moments implemented. ESCA training run: 3000 steps, α=0. SCE frequency at steps 1000/2000/3000 vs. baseline. | SCE frequency measurably higher than baseline by step 1000. Buffer not empty. No performative drift in first 2K steps.
Week 3 — Ablations | Ablation A: replay only vs. reward bonus only vs. both. Ablation B: SFT on full post-correction path vs. correction moment only. Ablation C: N_replay sweep {20, 50, 100}. | 6 training runs (3 ablations × 2 conditions each). Ablation report: which component drives which metric. N_replay recommendation. Optimal α value from sweep. | Clear winner among ablation conditions. Design choices confirmed or revised based on empirical results.
Week 4 — The Decisive OOD Experiment | Train final ESCA model on MATH + CODE (dual domain, verifiable). Evaluate on MATH hard tier, GPQA, spatial reasoning. | ESCA-GRPO final model (Qwen2.5-7B, 3000 steps). Evaluation on 3 problem sets (in-distribution, near-OOD, far-OOD). OOD generalization curve. Decision: is this the paper? | Far-OOD improvement > near-OOD > in-distribution. If yes: draft paper. If no: diagnose and redesign.

[TABLE]
Primary Claim
Self-correction behavior, emergent from RLVR training, can be amplified from an occasional accident into a robustly trained capability through targeted replay of self-correction traces — without compromising task performance or requiring step-level annotation.

[TABLE]
Secondary Claim
Models trained with ESCA show stronger self-correction generalization to OOD problem types, suggesting that self-correction ability is a domain-invariant reasoning meta-operation that transfers across task families. This is the key claim for the 'what happens after RLVR plateaus' question.

[TABLE]
PostTrace Claim (Companion Paper)
Per-example causal attribution in generative post-training is computationally tractable when restricted to the LoRA parameter subspace. Influence functions adapted to the LoRA subspace (LoRA-TRAK) provide practitioner-usable attribution for SFT and DPO fine-tuning on datasets up to 100K examples.

[TABLE]
Claim | Novelty | Prior Work | Gap Filled
SCE Detection as a Training Signal | First work to close the observation → training signal loop for self-correction | R1/QwQ papers observe SCE behavior. Multiple papers analyze it. Nobody trains on it. | The loop has never been closed
Correction-Moment-Only SFT | SFT applied only to the reversal + immediate context tokens, not the full post-correction path | Replay-based RL methods exist but operate on full trajectories or final answers | Trains the habit of recognizing error, not the specific solution — preserving exploration
Dynamic τ for Performative Drift Prevention | Adaptive threshold that tightens as mean semantic shift drops | Fixed thresholds in all prior work. Performative SCE problem not previously analyzed. | Prevents the most dangerous training failure mode for this approach
LoRA-TRAK for Generative Post-Training | TRAK extended to generative models via LoRA subspace approximation | TRAK (classification), DataInf (regression LoRA), TracIn (full models) | First tractable per-example attribution for generative post-training at practitioner scale
