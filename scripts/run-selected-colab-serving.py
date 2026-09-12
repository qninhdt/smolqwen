#!/usr/bin/env python
"""Run the frozen 16x2 official-Qwen study and download every point locally."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POINTS = ROOT / "benchmarks/serving/study-points.json"
REMOTE_BUNDLE = "/content/serving-point.tgz"
ARTIFACTS = ROOT / "artifacts/serving/sweeps/direct-vllm029-official-16x2"


def _colab(
    *args: str, check: bool = True, timeout_s: int = 120
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["colab", "--auth=oauth2", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout_s,
    )


def _session_exists(session: str) -> bool:
    result = _colab("sessions", check=False, timeout_s=60)
    return any(line.startswith(f"[{session}]") for line in (result.stdout or "").splitlines())


def _extract(bundle: Path) -> None:
    subprocess.run(["tar", "-xzf", str(bundle), "-C", str(ROOT)], check=True)


def _local_run(point: dict[str, object]) -> Path | None:
    point_dir = ARTIFACTS / str(point["_benchmark_name"])
    for path in sorted(point_dir.glob("SERVE--*/run=0.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("completed") == 200 and data.get("failed") == 0:
            return path
    return None


def _report(index: int, point: dict[str, object], path: Path | None, remote_rc: int) -> None:
    if path is None:
        state = "no raw result"
        status_path = ARTIFACTS / str(point["_benchmark_name"]) / "point-status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            state = str(status.get("status", state))
            if status.get("error"):
                state += f": {status['error']}"
        if remote_rc:
            state += f" (remote exit {remote_rc})"
        print(f"[{index}/32] result: {state}", flush=True)
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    completed = int(data.get("completed", 0))
    failed = int(data.get("failed", 0))
    p95_ttft = float(data.get("p95_ttft_ms", 0.0))
    p95_tpot = float(data.get("p95_tpot_ms", 0.0))
    throughput = float(data.get("output_throughput", 0.0))
    concurrency = int(data.get("max_concurrency", point["max_concurrency"]))
    if failed:
        assessment = "FAIL: exclude from selection"
    elif p95_ttft <= 200:
        assessment = "latency candidate"
    elif p95_ttft <= 500:
        assessment = "balanced candidate"
    else:
        assessment = "stress/high-latency candidate"
    status_note = ""
    status_path = ARTIFACTS / str(point["_benchmark_name"]) / "point-status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "passed":
            status_note = "; wrapper post-processing failed (raw measurement retained)"
    print(
        f"[{index}/32] result: {completed}/{completed + failed}, c{concurrency}, "
        f"{throughput:.2f} tok/s, p95 TTFT {p95_ttft:.1f} ms, "
        f"p95 TPOT {p95_tpot:.2f} ms; {assessment}{status_note}",
        flush=True,
    )


def _start_session(base: str, generation: int) -> str:
    session = base if generation == 0 else f"{base}-r{generation}"
    _colab("new", "--session", session, "--gpu", "L4", timeout_s=180)
    return session


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="smolqwen-l4-16x2")
    parser.add_argument("--archive", type=Path, default=Path("/tmp/smolqwen-serving-src.tgz"))
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"source archive not found: {args.archive}")
    points = json.loads(POINTS.read_text(encoding="utf-8"))
    if len(points) != 32:
        raise SystemExit(f"expected 32 study points, got {len(points)}")
    source_uploaded = False
    active_session = args.session
    session_generation = 0
    known_sessions = {args.session}
    try:
        downloaded = 0
        for index, point in enumerate(points, start=1):
            local_run = _local_run(point)
            if local_run is not None:
                downloaded += 1
                _report(index, point, local_run, 0)
                continue

            bundle_path: Path | None = None
            remote_rc = 1
            for attempt in range(1, 3):
                try:
                    if not _session_exists(active_session):
                        session_generation += 1
                        active_session = _start_session(args.session, session_generation)
                        known_sessions.add(active_session)
                        source_uploaded = False
                    if not source_uploaded:
                        _colab(
                            "upload",
                            "-s",
                            active_session,
                            str(args.archive),
                            "/content/smolqwen-l4-src.tgz",
                        )
                        source_uploaded = True
                    with tempfile.TemporaryDirectory(prefix="smolqwen-point-") as directory:
                        point_file = Path(directory) / "serving-point.json"
                        point_file.write_text(
                            json.dumps(point, sort_keys=True) + "\n", encoding="utf-8"
                        )
                        _colab(
                            "upload",
                            "-s",
                            active_session,
                            str(point_file),
                            "/content/serving-point.json",
                        )
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    source_uploaded = False
                    if attempt == 2:
                        raise RuntimeError(
                            f"could not prepare {point['_benchmark_name']} after two attempts"
                        ) from exc
                    continue

                print(
                    f"[{index}/32] running {point['_benchmark_name']} (attempt {attempt})",
                    flush=True,
                )
                result = _colab(
                    "exec",
                    "-s",
                    active_session,
                    "-f",
                    "scripts/colab-serving-point.py",
                    "--timeout",
                    "1800",
                    timeout_s=2100,
                    check=False,
                )
                remote_rc = result.returncode
                bundle = Path("/tmp") / f"serving-point-{index:02d}-attempt-{attempt}.tgz"
                download = _colab(
                    "download",
                    "-s",
                    active_session,
                    REMOTE_BUNDLE,
                    str(bundle),
                    check=False,
                )
                if download.returncode == 0:
                    _extract(bundle)
                    bundle_path = bundle
                    break
                source_uploaded = False
                if _session_exists(active_session):
                    if attempt == 2:
                        raise RuntimeError(
                            f"remote point {point['_benchmark_name']} produced no bundle"
                        )
                else:
                    session_generation += 1
                    active_session = f"{args.session}-r{session_generation}"
                    known_sessions.add(active_session)

            if bundle_path is not None:
                downloaded += 1
            _report(index, point, _local_run(point), remote_rc)
        print(f"downloaded {downloaded}/32 point bundles", flush=True)
        return 0 if downloaded == 32 else 1
    finally:
        for session in sorted(known_sessions):
            try:
                if _session_exists(session):
                    _colab("stop", "-s", session, check=False, timeout_s=180)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass


if __name__ == "__main__":
    raise SystemExit(main())
