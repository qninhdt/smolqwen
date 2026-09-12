#!/usr/bin/env python
"""Run the selected-profile validation on one Colab L4 and download its bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REMOTE_BUNDLE = "/content/serving-validation.tgz"


def _colab(*args: str, check: bool = True, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["colab", "--auth=oauth2", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout,
    )


def _session_exists(session: str) -> bool:
    result = _colab("sessions", check=False, timeout=60)
    return any(line.startswith(f"[{session}]") for line in result.stdout.splitlines())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="smolqwen-l4-validation")
    parser.add_argument(
        "--profile", default="balanced", choices=("latency", "balanced", "throughput")
    )
    parser.add_argument("--archive", type=Path, default=Path("/tmp/smolqwen-serving-src.tgz"))
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"source archive not found: {args.archive}")
    if _session_exists(args.session):
        raise SystemExit(f"refusing to reuse active session {args.session}")

    profile_path = ROOT / "configs" / "serving" / f"{args.profile}.yaml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    config = {"profile_name": args.profile, "profile": profile}
    started = False
    try:
        _colab("new", "--session", args.session, "--gpu", "L4")
        started = True
        _colab(
            "upload",
            "-s",
            args.session,
            str(args.archive),
            "/content/smolqwen-l4-src.tgz",
        )
        with tempfile.TemporaryDirectory(prefix="smolqwen-validation-") as directory:
            config_path = Path(directory) / "serving-validation-config.json"
            config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")
            _colab(
                "upload",
                "-s",
                args.session,
                str(config_path),
                "/content/serving-validation-config.json",
            )
        print(f"running direct validation for {args.profile}", flush=True)
        result = _colab(
            "exec",
            "-s",
            args.session,
            "-f",
            "scripts/colab-serving-validation.py",
            "--timeout",
            "1800",
            check=False,
            timeout=2100,
        )
        bundle = Path("/tmp/smolqwen-serving-validation.tgz")
        download = _colab(
            "download",
            "-s",
            args.session,
            REMOTE_BUNDLE,
            str(bundle),
            check=False,
        )
        if download.returncode:
            raise RuntimeError("validation produced no downloadable bundle")
        subprocess.run(["tar", "-xzf", str(bundle), "-C", str(ROOT)], check=True)
        print("downloaded artifacts/serving/validation", flush=True)
        return result.returncode
    finally:
        if started and _session_exists(args.session):
            _colab("stop", "-s", args.session, check=False)


if __name__ == "__main__":
    raise SystemExit(main())
