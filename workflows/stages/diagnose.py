"""Diagnose stage: environment and pipeline health checks.

The ``env`` check is deliberately lightweight (no torch/ray imports).
The ``pipeline`` check imports the full ML stack and should only be run
when debugging issues.
"""

from __future__ import annotations

import os
import platform

from workflows.lifecycle import State, logger


def run_diagnose_env(state: State) -> None:
    """Fast environment check — no torch/ray imports."""
    logger.info(  # type: ignore[call-arg]
        "diagnose_env",
        python_version=platform.python_version(),
        platform=platform.platform(),
        model=state.model,
    )
    print("\n" + "=" * 60)
    print("ENVIRONMENT DIAGNOSTICS")
    print("=" * 60)
    print(f"  Python:       {platform.python_version()}")
    print(f"  Platform:     {platform.platform()}")
    print(f"  Model type:   {state.model}")
    print(f"  Run type:     {state.run_type}")
    print(f"  Device:       {state.device}")

    # Check NVIDIA LD_LIBRARY_PATH
    from sentimentizer.env import get_nvidia_ld_library_path

    nvidia_paths = get_nvidia_ld_library_path()
    if nvidia_paths:
        print(f"  NVIDIA lib:   {nvidia_paths[:80]}...")
    else:
        print("  NVIDIA lib:   (not found)")

    # Check Ray-related env vars
    ray_env_vars = [
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV",
        "RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION",
        "RAY_GRAFANA_HOST",
        "RAY_PROMETHEUS_HOST",
    ]
    for var in ray_env_vars:
        val = os.environ.get(var, "(not set)")
        print(f"  {var}: {val}")

    # Check if torch/ray are importable (without importing them)
    torch_available = False
    ray_available = False
    try:
        import importlib.util

        torch_available = importlib.util.find_spec("torch") is not None
        ray_available = importlib.util.find_spec("ray") is not None
    except Exception:
        pass
    print(f"  torch:        {'available' if torch_available else 'NOT available'}")
    print(f"  ray:          {'available' if ray_available else 'NOT available'}")
    print("=" * 60 + "\n")


def run_diagnose_pipeline(state: State) -> None:
    """Heavy pipeline check. Imports the ML stack."""
    import json

    from sentimentizer.agent.diagnose_model import diagnose_training_issues

    logger.info("running_diagnostics", model_type=state.model)
    result = diagnose_training_issues(model_type=state.model)

    # Print human-readable summary
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE DIAGNOSTICS")
    print("=" * 60)

    for check_name, check in result["checks"].items():
        status = (
            "PASS" if check.get("passed", False) else ("SKIP" if check.get("skipped") else "FAIL")
        )
        print(f"\n  [{status}] {check_name}")
        if "mismatch_rate" in check:
            print(
                f"    Mismatch rate: {check['mismatch_rate']:.1%} "
                f"({check.get('mismatches', '?')}/"
                f"{check.get('common_words', '?')} words)"
            )
        if "shape_matches" in check:
            print(
                f"    Shape matches: {check['shape_matches']} "
                f"(actual={check.get('actual_shape')}, "
                f"expected={check.get('expected_shape')})"
            )
        if "imbalance_ratio" in check:
            print(f"    Class imbalance ratio: {check['imbalance_ratio']}:1")
        if "invalid_token_count_in_sample" in check:
            print(f"    Invalid tokens in sample: {check['invalid_token_count_in_sample']}")
        if check.get("skipped"):
            print(f"    Skipped: {check.get('skip_reason', 'unknown')}")

    if result["critical_issues"]:
        print("\n  CRITICAL ISSUES:")
        for issue in result["critical_issues"]:
            print(f"    - {issue}")
    if result["warnings"]:
        print("\n  WARNINGS:")
        for warning in result["warnings"]:
            print(f"    - {warning}")
    if not result["critical_issues"] and not result["warnings"]:
        print("\n  All checks passed. No issues detected.")

    print("=" * 60 + "\n")

    # Also save full results as JSON
    diagnostics_path = "diagnostics_results.json"
    with open(diagnostics_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"diagnostics_saved_to_{diagnostics_path}")
