"""Persistent `spawn`ed workers that own live environments, with bounded calls.

Four decisions here, each answering a failure that would otherwise be silent.

**`spawn`, not `fork`.** `huggingface_hub.get_token()` reads the Colab secrets
vault first and caches it in a *module global*, so a `fork`ed child inherits the
plaintext token in memory no matter what `os.environ` says. A `spawn`ed child is a
fresh interpreter with no inherited globals. On top of that the worker clears and
rebuilds `os.environ` before importing anything, sets
`HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, and redirects `HOME`/`HF_HOME` so
`~/.cache/huggingface/token` and `~/.netrc` are unreachable. The test asserts
`get_token() is None` inside a worker rather than inspecting `os.environ`, because
the env var is the one source that was never the problem.

**Compilation happens in the worker.** The parent holds raw JSON strings and never
`exec()`s dataset source — see `registry.py`. Each worker builds its own registry,
so "compile once" is once per worker.

**Two timeout layers, meaning different things.** Inside the worker a `SIGALRM`
bounds one call, so a runaway `check_func` loses its own episode and the worker
survives — that is `timeout`. In the parent a longer deadline catches a worker that
did not answer at all, which means it is wedged or dead — that is `worker_crash`.
Collapsing them would be wrong in a way that reaches training: Phase 6 drops
`worker_crash` episodes from the buffer, while a `timeout` is the model's own doing
and is scored.

**A crash takes every episode the worker held.** A worker owning N live episodes
that dies cannot lose one in isolation, so the pool reports all N failed and the
scheduler re-admits them. `env_episodes_per_worker` therefore sets the blast radius
as a chosen number rather than an emergent one.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Literal

# Env vars a worker may keep. Everything else is dropped, so a credential added to
# the parent's environment later does not silently become reachable.
_ALLOWED_ENV_KEYS: frozenset[str] = frozenset(
    {"PATH", "LANG", "LC_ALL", "TZ", "PYTHONPATH", "PYTHONHASHSEED", "TMPDIR"}
)

FailureReason = Literal["ok", "timeout", "worker_crash", "error"]

# How much longer than the in-worker alarm the parent waits before declaring the
# worker wedged. The alarm should always fire first; this margin is what
# distinguishes "the call ran too long" from "nobody is home".
PARENT_TIMEOUT_MARGIN_S = 5.0


class PoolError(Exception):
    """Raised when the pool cannot be started or a request cannot be routed."""


def minimal_environment(*, home: str) -> dict[str, str]:
    """The environment a worker runs with: no credentials, redirected HOME.

    Built by allow-list rather than by deleting known credential names. A deny-list
    is only correct until the next tool invents an env var.
    """
    env = {key: value for key, value in os.environ.items() if key in _ALLOWED_ENV_KEYS}
    env.update(
        {
            "HOME": home,
            "HF_HOME": os.path.join(home, "hf"),
            "XDG_CACHE_HOME": os.path.join(home, "cache"),
            # Stops huggingface_hub from reading a token it happens to find.
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HUB_OFFLINE": "1",
            "WANDB_MODE": "disabled",
            "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
        }
    )
    return env


@dataclass(frozen=True)
class Result:
    """One request's outcome. Never an exception crossing the process boundary."""

    episode_id: str
    reason: FailureReason
    value: Any = None
    detail: str | None = None
    lost_episode_ids: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.reason == "ok"

    @property
    def is_infrastructure_failure(self) -> bool:
        """Whether Phase 6 must drop this episode rather than score it.

        A crashed episode scored as a low reward teaches the model to avoid an
        infrastructure failure it did not cause.
        """
        return self.reason == "worker_crash"


@dataclass
class _WorkerHandle:
    """The parent's view of one worker: its pipes, its process, its episodes."""

    index: int
    process: Any
    requests: Any
    results: Any
    episodes: set[str] = field(default_factory=set)
    home: str = ""
    lost_episodes: tuple[str, ...] = ()
    # A worker executes its request queue serially. Serialising parent requests
    # too keeps one caller from consuming another caller's reply from `results`.
    lock: Any = field(default_factory=threading.Lock)

    def alive(self) -> bool:
        return bool(self.process.is_alive())


def _worker_main(
    requests: Any,
    results: Any,
    env: Mapping[str, str],
    metadata_path: str,
    metadata_sha256: str | None,
    scenario_path: str | None,
    scenario_sha256: str | None,
    call_timeout_s: float,
) -> None:  # pragma: no cover - runs in a child process
    """Worker entry point. Scrubs the environment *before* importing anything else."""
    os.environ.clear()
    os.environ.update(env)

    import signal

    from smolqwen.data.loader import verify_sha256
    from smolqwen.env.instance import EnvInstance
    from smolqwen.env.registry import EnvRegistry
    from smolqwen.env.verifier import compile_checklist, score

    class _CallTimeout(Exception):
        pass

    def _alarm(_signum: int, _frame: Any) -> None:
        raise _CallTimeout(f"call exceeded {call_timeout_s}s")

    signal.signal(signal.SIGALRM, _alarm)

    registry = EnvRegistry.from_metadata(metadata_path, sha256=metadata_sha256)
    if scenario_path is not None:
        verify_sha256(scenario_path, scenario_sha256)
    instances: dict[str, EnvInstance] = {}
    # Shared by every episode of the same scenario on this worker. `score` binds
    # a fresh deep-copied initial state before every invocation, so sharing the
    # compiled functions does not share episode state.
    checklists: dict[str, Any] = {}
    episode_checklist_keys: dict[str, str] = {}

    def _checklist_key(payload: Mapping[str, Any], checklist: Any) -> str:
        explicit = payload.get("checklist_id")
        if explicit is not None:
            return str(explicit)
        encoded = json.dumps(checklist, sort_keys=True, separators=(",", ":"), default=str)
        return sha256(encoded.encode("utf-8")).hexdigest()

    def _handle(command: str, payload: Mapping[str, Any]) -> Any:
        episode_id = str(payload.get("episode_id", ""))
        if command == "create":
            env_id = str(payload["env_id"])
            instances[episode_id] = EnvInstance.create(
                env_id=env_id,
                env_class=registry.env_class(env_id),
                env_class_name=str(payload["env_class_name"]),
                init_config=payload.get("init_config") or {},
                tools=registry.tools(env_id),
            )
            checklist = payload.get("checklist")
            if checklist is not None:
                create_checklist_key = _checklist_key(payload, checklist)
                if create_checklist_key not in checklists:
                    checklists[create_checklist_key] = compile_checklist(
                        create_checklist_key, checklist
                    )
                episode_checklist_keys[episode_id] = create_checklist_key
            return {"tools": list(instances[episode_id].tools)}
        if command == "step":
            return instances[episode_id].step(str(payload["name"]), payload.get("arguments") or {})
        if command == "finalize":
            return instances[episode_id].final_state()
        if command == "score":
            instance = instances[episode_id]
            checklist = payload.get("checklist")
            checklist_key: str | None = episode_checklist_keys.get(episode_id)
            if checklist is None and checklist_key is None:
                raise PoolError(f"{episode_id}: no checklist supplied at create time")
            if checklist is not None:
                checklist_key = _checklist_key(payload, checklist)
            assert checklist_key is not None
            compiled = checklists.get(checklist_key)
            if compiled is None:
                assert checklist is not None
                compiled = compile_checklist(checklist_key, checklist)
                checklists[checklist_key] = compiled
            return score(compiled, instance.initial_state, instance.final_state()).to_dict()
        if command == "destroy":
            instances.pop(episode_id, None)
            episode_checklist_keys.pop(episode_id, None)
            return True
        if command == "stats":
            return {
                "exec_count": registry.exec_count,
                "compiled_envs": registry.compiled_count(),
                "checklist_exec_count": sum(c.exec_count for c in checklists.values()),
                "live_episodes": len(instances),
                "pid": os.getpid(),
                "hf_token_visible": _hf_token_visible(),
            }
        raise PoolError(f"unknown command {command!r}")

    while True:
        message = requests.get()
        if message is None:
            return
        command, payload = message
        episode_id = str(payload.get("episode_id", ""))
        # The alarm bounds this one call. On expiry the episode fails and the
        # worker keeps running -- which is exactly what separates `timeout` from
        # `worker_crash` for the caller.
        timeout_s = float(payload.get("timeout_s", call_timeout_s))
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        try:
            value = _handle(command, payload)
            results.put(("ok", episode_id, value, None))
        except _CallTimeout as exc:
            results.put(("timeout", episode_id, None, str(exc)))
        except Exception as exc:  # noqa: BLE001 - never let a child die on dataset code
            results.put(("error", episode_id, None, f"{type(exc).__name__}: {exc}"))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)


def _hf_token_visible() -> bool:  # pragma: no cover - exercised inside a worker
    """Whether a Hub token is reachable from here, asked the way the Hub asks it."""
    try:
        from huggingface_hub import get_token
    except ImportError:
        return False
    try:
        return get_token() is not None
    except Exception:  # noqa: BLE001 - an unreadable token is an absent token
        return False


class WorkerPool:
    """N persistent workers, each owning several live environments."""

    def __init__(
        self,
        *,
        metadata_path: str,
        metadata_sha256: str | None = None,
        scenario_path: str | None = None,
        scenario_sha256: str | None = None,
        worker_count: int = 2,
        episodes_per_worker: int = 8,
        call_timeout_s: float = 10.0,
        create_timeout_s: float | None = None,
        step_timeout_s: float | None = None,
        verify_timeout_s: float | None = None,
    ) -> None:
        if worker_count < 1:
            raise PoolError("worker_count must be at least 1")
        self.metadata_path = metadata_path
        self.metadata_sha256 = metadata_sha256
        if (scenario_path is None) != (scenario_sha256 is None):
            raise PoolError("scenario_path and scenario_sha256 must be provided together")
        self.scenario_path = scenario_path
        self.scenario_sha256 = scenario_sha256
        self.worker_count = worker_count
        self.episodes_per_worker = episodes_per_worker
        self.call_timeout_s = call_timeout_s
        self.create_timeout_s = create_timeout_s or call_timeout_s
        self.step_timeout_s = step_timeout_s or call_timeout_s
        self.verify_timeout_s = verify_timeout_s or call_timeout_s
        # `spawn` is the security boundary, not a portability choice -- see the
        # module docstring on the forked-token cache.
        self._context = mp.get_context("spawn")
        self._workers: list[_WorkerHandle] = []
        self._owner: dict[str, int] = {}
        self._temp_homes: list[Any] = []
        self._started = False

    def __enter__(self) -> WorkerPool:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()

    def start(self) -> None:
        if self._started:
            return
        for index in range(self.worker_count):
            self._workers.append(self._spawn(index))
        self._started = True

    def _spawn(self, index: int) -> _WorkerHandle:
        home = tempfile.TemporaryDirectory(prefix=f"smolqwen-worker-{index}-")
        self._temp_homes.append(home)
        requests: Any = self._context.Queue()
        results: Any = self._context.Queue()
        process = self._context.Process(
            target=_worker_main,
            args=(
                requests,
                results,
                minimal_environment(home=home.name),
                self.metadata_path,
                self.metadata_sha256,
                self.scenario_path,
                self.scenario_sha256,
                self.call_timeout_s,
            ),
            daemon=True,
        )
        process.start()
        return _WorkerHandle(
            index=index, process=process, requests=requests, results=results, home=home.name
        )

    def _worker_for(self, episode_id: str) -> _WorkerHandle:
        index = self._owner.get(episode_id)
        if index is None:
            raise PoolError(f"episode {episode_id!r} was never created on this pool")
        return self._workers[index]

    def _least_loaded(self) -> _WorkerHandle:
        handle = min(self._workers, key=lambda worker: len(worker.episodes))
        if len(handle.episodes) >= self.episodes_per_worker:
            raise PoolError(
                f"every worker holds {self.episodes_per_worker} episodes; "
                "raise env_episodes_per_worker or env_worker_count"
            )
        return handle

    def _replace(self, handle: _WorkerHandle) -> list[str]:
        """Kill and rebuild a worker, returning every episode it held."""
        lost = sorted(handle.episodes)
        if handle.process.is_alive():
            handle.process.kill()
        handle.process.join(timeout=5)
        for episode_id in lost:
            self._owner.pop(episode_id, None)
        handle.episodes.clear()
        handle.lost_episodes = tuple(lost)
        replacement = self._spawn(handle.index)
        self._workers[handle.index] = replacement
        return lost

    def _request(
        self, handle: _WorkerHandle, command: str, payload: dict[str, Any], *, timeout_s: float
    ) -> Result:
        episode_id = str(payload.get("episode_id", ""))
        with handle.lock:
            if self._workers[handle.index] is not handle:
                return Result(
                    episode_id,
                    "worker_crash",
                    detail=(
                        f"worker {handle.index} was replaced; "
                        f"lost episodes {list(handle.lost_episodes)}"
                    ),
                    lost_episode_ids=handle.lost_episodes,
                )
            if not handle.alive():
                lost = tuple(self._replace(handle))
                return Result(
                    episode_id,
                    "worker_crash",
                    detail=f"worker {handle.index} was already dead; lost episodes {list(lost)}",
                    lost_episode_ids=lost,
                )

            payload["timeout_s"] = timeout_s
            handle.requests.put((command, payload))
            deadline = timeout_s + PARENT_TIMEOUT_MARGIN_S
            try:
                reason, returned_id, value, detail = handle.results.get(timeout=deadline)
            except queue.Empty:
                # The in-worker alarm should have answered by now. It did not, so the
                # worker is wedged or dead: every episode it held is lost.
                lost = tuple(self._replace(handle))
                return Result(
                    episode_id,
                    "worker_crash",
                    detail=(
                        f"worker {handle.index} did not respond within {deadline}s; "
                        f"lost episodes {list(lost)}"
                    ),
                    lost_episode_ids=lost,
                )
            if not handle.alive():
                lost = tuple(self._replace(handle))
                return Result(
                    episode_id,
                    "worker_crash",
                    detail=(
                        f"worker {handle.index} died during the call; lost episodes {list(lost)}"
                    ),
                    lost_episode_ids=lost,
                )
            return Result(returned_id or episode_id, reason, value=value, detail=detail)

    def create(
        self,
        episode_id: str,
        *,
        env_id: str,
        env_class_name: str,
        init_config: Mapping[str, Any] | None = None,
        checklist: Sequence[Mapping[str, Any]] | None = None,
        checklist_id: str | None = None,
    ) -> Result:
        """Build a live environment for one episode on the least-loaded worker."""
        if not self._started:
            self.start()
        if episode_id in self._owner:
            raise PoolError(f"episode {episode_id!r} already exists")
        handle = self._least_loaded()
        self._owner[episode_id] = handle.index
        handle.episodes.add(episode_id)
        result = self._request(
            handle,
            "create",
            {
                "episode_id": episode_id,
                "env_id": env_id,
                "env_class_name": env_class_name,
                "init_config": dict(init_config or {}),
                "checklist": list(checklist) if checklist is not None else None,
                "checklist_id": checklist_id,
            },
            timeout_s=self.create_timeout_s,
        )
        if not result.ok:
            handle.episodes.discard(episode_id)
            self._owner.pop(episode_id, None)
        return result

    def step(self, episode_id: str, name: str, arguments: Mapping[str, Any]) -> Result:
        return self._request(
            self._worker_for(episode_id),
            "step",
            {"episode_id": episode_id, "name": name, "arguments": dict(arguments)},
            timeout_s=self.step_timeout_s,
        )

    def finalize(self, episode_id: str) -> Result:
        return self._request(
            self._worker_for(episode_id),
            "finalize",
            {"episode_id": episode_id},
            timeout_s=self.step_timeout_s,
        )

    def score(
        self,
        episode_id: str,
        checklist: Sequence[Mapping[str, Any]] | None = None,
        *,
        checklist_id: str | None = None,
    ) -> Result:
        payload: dict[str, Any] = {"episode_id": episode_id}
        if checklist is not None:
            payload["checklist"] = list(checklist)
        if checklist_id is not None:
            payload["checklist_id"] = checklist_id
        return self._request(
            self._worker_for(episode_id), "score", payload, timeout_s=self.verify_timeout_s
        )

    def destroy(self, episode_id: str) -> Result:
        handle = self._worker_for(episode_id)
        result = self._request(
            handle, "destroy", {"episode_id": episode_id}, timeout_s=self.step_timeout_s
        )
        handle.episodes.discard(episode_id)
        self._owner.pop(episode_id, None)
        return result

    def worker_stats(self, worker_index: int = 0) -> Result:
        """Per-worker counters: `exec` calls, live episodes, token visibility.

        The exec counters are per worker deliberately: a global count would be
        satisfied by compiling in the parent, which is the thing the design forbids.
        """
        return self._request(
            self._workers[worker_index], "stats", {"episode_id": ""}, timeout_s=self.step_timeout_s
        )

    def episodes_of(self, worker_index: int) -> tuple[str, ...]:
        return tuple(sorted(self._workers[worker_index].episodes))

    def owner_of(self, episode_id: str) -> int | None:
        return self._owner.get(episode_id)

    def kill_worker(self, worker_index: int) -> list[str]:
        """Kill a worker outright and report every episode lost with it.

        Exists so the crash blast radius is testable without waiting for a real
        segfault: the caller gets the same episode list the pool would report.
        """
        return self._replace(self._workers[worker_index])

    def shutdown(self) -> None:
        for handle in self._workers:
            try:
                handle.requests.put(None)
            except Exception:  # noqa: BLE001 - a dead worker's queue may be closed
                pass
        for handle in self._workers:
            handle.process.join(timeout=5)
            if handle.process.is_alive():
                handle.process.kill()
                handle.process.join(timeout=5)
        self._workers.clear()
        self._owner.clear()
        for home in self._temp_homes:
            home.cleanup()
        self._temp_homes.clear()
        self._started = False
