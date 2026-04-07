"""Sentinel CLI entry point."""

from __future__ import annotations

import argparse
import json
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
    }
    handlers[args.command](args)


if __name__ == "__main__":
    app()
