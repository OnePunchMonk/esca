"""Sentinel CLI entry point."""

from __future__ import annotations

import argparse
import sys


def cmd_profile(args: argparse.Namespace) -> None:
    """Profile a model's capability subspaces."""
    print(f"[Sentinel] Profiling {args.model} with LoRA r={args.lora_r}")
    print(f"[Sentinel] Capabilities: {args.capabilities}")
    print("[Sentinel] (Full implementation requires GPU — see Python API)")


def cmd_predict(args: argparse.Namespace) -> None:
    """Predict regression risk for a training dataset."""
    print(f"[Sentinel] Loading profile from {args.profile}")
    print(f"[Sentinel] Scanning training data at {args.training_data}")
    print("[Sentinel] (Full implementation requires GPU — see Python API)")


def cmd_inspect(args: argparse.Namespace) -> None:
    """Inspect a saved profile or report."""
    from ..profiler.profile import CapabilityProfile

    profile = CapabilityProfile.load(args.path)
    print(profile.summary())
    print()
    print(profile.overlap_report())


def cmd_audit(args: argparse.Namespace) -> None:
    """Run a post-training regression audit."""
    print(f"[Sentinel] Running post-training audit")
    print(f"[Sentinel] Profile:      {args.profile}")
    print(f"[Sentinel] Model before: {args.model_before}")
    print(f"[Sentinel] Model after:  {args.model_after}")
    print(f"[Sentinel] Output:       {args.output}")
    print()

    try:
        from ..profiler.profile import CapabilityProfile
        from ..auditor import RegressionAuditor

        profile = CapabilityProfile.load(args.profile)
        auditor = RegressionAuditor(
            profile,
            attribution_method=args.method,
            attribution_top_k=args.top_k,
        )

        # Stub: In a full CLI run this loads models from paths
        print("[Sentinel] (Full GPU audit: load models, run auditor.audit(), export report)")
        print(f"[Sentinel] Attribution method: {args.method}")
        print(f"[Sentinel] Output report: {args.output}")

    except Exception as e:
        print(f"[Sentinel] Error: {e}")
        print("[Sentinel] Tip: use the Python API for full audit control")
        raise SystemExit(1)


def cmd_data_scan(args: argparse.Namespace) -> None:
    """Scan a training dataset for regression-causing examples."""
    print(f"[Sentinel] Scanning training data: {args.training_data}")
    print(f"[Sentinel] Profile: {args.profile}")
    print(f"[Sentinel] Max examples to report: {args.top_k}")
    print()
    print("[Sentinel] Use the Python API for full data scanning:")
    print("""
  from sentinel import RegressionPredictor
  predictor = RegressionPredictor(profile, model=model, tokenizer=tokenizer)
  risk = predictor.predict(training_data)
  for cap, r in risk.capabilities.items():
      print(f"{cap}: {r.risk_level} — top examples:")
      for ex in r.contributing_examples:
          print(f"  #{ex.example_id}: {ex.example_text[:80]}")
""")


def cmd_surgery(args: argparse.Namespace) -> None:
    """Generate a surgery plan from an audit report."""
    import json

    print(f"[Sentinel] Loading audit report from {args.audit_report}")
    try:
        from ..auditor.report import AuditReport
        from ..surgeon import DataSurgeon

        with open(args.audit_report) as f:
            data = json.load(f)

        # Build a stub report for demo purposes
        print("[Sentinel] Surgery plan generation (stub):")
        print(f"  Degraded capabilities in report: {data.get('capabilities_degraded', [])}")
        print(f"  Strategy: {args.strategy}")
        print(f"  Max removals: {args.max_removals}")
        print(f"[Sentinel] Output plan: {args.output}")
        print("[Sentinel] Use Python API for full surgery plan execution.")

    except Exception as e:
        print(f"[Sentinel] Error: {e}")
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel",
        description="Sentinel — regression prevention for LLM fine-tuning",
    )
    sub = parser.add_subparsers(dest="command")

    # profile
    p_profile = sub.add_parser("profile", help="Profile a model's capabilities")
    p_profile.add_argument("--model", required=True)
    p_profile.add_argument("--lora-r", type=int, default=16)
    p_profile.add_argument("--capabilities", default="math,code,safety")
    p_profile.add_argument("--output", default="profile.sentinel")
    p_profile.add_argument("--device", default="cuda")

    # predict
    p_predict = sub.add_parser("predict", help="Predict regression risk")
    p_predict.add_argument("--profile", required=True)
    p_predict.add_argument("--training-data", required=True)
    p_predict.add_argument("--sample-size", type=int, default=1000)
    p_predict.add_argument("--quick", action="store_true")

    # inspect
    p_inspect = sub.add_parser("inspect", help="Inspect a profile or report")
    p_inspect.add_argument("path")

    # audit
    p_audit = sub.add_parser("audit", help="Run post-training regression audit")
    p_audit.add_argument("--profile", required=True, help="Path to .sentinel profile")
    p_audit.add_argument("--model-before", required=True, help="Base model path or HF ID")
    p_audit.add_argument("--model-after", required=True, help="Fine-tuned model path")
    p_audit.add_argument("--training-data", required=True, help="Training dataset path")
    p_audit.add_argument("--method", default="gradient_cosine",
                         choices=["gradient_cosine", "datainf"],
                         help="Attribution method")
    p_audit.add_argument("--top-k", type=int, default=50,
                         help="Top-k examples to attribute per capability")
    p_audit.add_argument("--output", default="audit_report.html",
                         help="Output file (html or json)")
    p_audit.add_argument("--device", default="cuda")

    # data-scan
    p_scan = sub.add_parser("data-scan", help="Scan training data for harmful examples")
    p_scan.add_argument("--profile", required=True)
    p_scan.add_argument("--training-data", required=True)
    p_scan.add_argument("--top-k", type=int, default=100)
    p_scan.add_argument("--output", default="scan_results.json")

    # surgery
    p_surgery = sub.add_parser("surgery", help="Generate a data surgery plan from an audit report")
    p_surgery.add_argument("--audit-report", required=True, help="Path to audit_report.json")
    p_surgery.add_argument("--strategy", default="smart",
                           choices=["remove", "reweight", "augment", "smart"])
    p_surgery.add_argument("--max-removals", type=int, default=500)
    p_surgery.add_argument("--output", default="surgery_plan.json")

    return parser


def app() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handlers = {
        "profile": cmd_profile,
        "predict": cmd_predict,
        "inspect": cmd_inspect,
        "audit": cmd_audit,
        "data-scan": cmd_data_scan,
        "surgery": cmd_surgery,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    app()
