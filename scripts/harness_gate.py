from __future__ import annotations

import argparse
from pathlib import Path

from ai_core.agent.harness.gates import gate_artifact
from scripts.harness_run import run_harness


def run_gate(output_path: Path) -> int:
    report = run_harness(output_path)
    failed = 0
    passed = 0
    reasons: list[str] = []

    for item in report.get("queries", []):
        artifact = item.get("artifact", {})
        ok, failures = gate_artifact(artifact)
        if ok:
            passed += 1
        else:
            failed += 1
            reasons.extend(failures)

    print(f"passed={passed} failed={failed}")
    if reasons:
        for reason in reasons:
            print(f"- {reason}")

    return 0 if failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run harness gate checks.")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args(argv)

    output_path = Path(args.output)
    return run_gate(output_path)


if __name__ == "__main__":
    raise SystemExit(main())
