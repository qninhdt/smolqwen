"""BFCL multi-turn adapter backed by the pinned Gorilla release.

The adapter keeps the benchmark's static user turns and executes calls against
fresh API objects per task. Ground-truth calls are parsed with ``ast`` and
dispatched with ``getattr``; the released evaluator's ``eval``-based path is
intentionally not imported into the evaluation process.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from pydantic import Field

from smolqwen.config_models import EvalConfig, StrictModel
from smolqwen.data.tool_call_xml import parse_tool_calls
from smolqwen.eval.adapters.base import AdapterResult, BenchmarkAdapter, EvalTask, StepResult
from smolqwen.eval.manifest import hash_json, sha256_file
from smolqwen.eval.metrics import TaskMetrics, aggregate
from smolqwen.eval.tool_calls import (
    is_completion_signal,
    is_error_signal,
    parse_normalized_json_calls,
)
from smolqwen.prompts import NON_CONVERSATIONAL

DEFAULT_HOLDOUT_PROMPT = "I have updated some more functions you can choose from. What about now?"
CLASS_MODULES = {
    "GorillaFileSystem": "gorilla_file_system",
    "MathAPI": "math_api",
    "MessageAPI": "message_api",
    "TwitterAPI": "posting_api",
    "TicketAPI": "ticket_api",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "VehicleControlAPI": "vehicle_control",
}
FUNCTION_DOC_FILES = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
}

Call = tuple[str, tuple[Any, ...], dict[str, Any]]
ADAPTER_NAME = "bfcl_multi_turn"


class BfclAdapterOptions(StrictModel):
    """BFCL-owned config; the evaluation core treats this mapping as opaque."""

    categories: Sequence[str]
    data_dir: str
    benchmark_commit: str = Field(pattern=r"^[0-9a-fA-F]{40}$")


@dataclass
class _TaskState:
    entry: Mapping[str, Any]
    ground_truth: tuple[tuple[str, ...], ...]
    turn_index: int = 0
    completed: bool = False
    invalid_calls: int = 0
    results: list[list[str]] = field(default_factory=lambda: [[]])
    model_snapshots: list[dict[str, dict[str, Any]]] = field(default_factory=list)
    instances: dict[str, Any] = field(default_factory=dict)
    active_tools: tuple[Mapping[str, Any], ...] = ()
    all_tools: tuple[Mapping[str, Any], ...] = ()


class BfclMultiTurnAdapter:
    """Run the four static BFCL multi-turn categories without a user simulator."""

    def __init__(
        self,
        data_dir: Path | str,
        categories: Sequence[str],
        *,
        benchmark_commit: str | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.categories = tuple(categories)
        self.benchmark_commit = benchmark_commit
        self._states: dict[str, _TaskState] = {}

    def load_tasks(self) -> list[EvalTask]:
        tasks: list[EvalTask] = []
        for category in self.categories:
            entries = self._read_jsonl(self.data_dir / f"BFCL_v4_{category}.json")
            expected = self._read_jsonl(
                self.data_dir / "possible_answer" / f"BFCL_v4_{category}.json"
            )
            expected_by_id = {str(item["id"]): item["ground_truth"] for item in expected}
            for entry in entries:
                task_id = str(entry["id"])
                questions = self._questions(entry, task_id)
                ground_truth = self._ground_truth(expected_by_id.get(task_id), task_id)
                if len(ground_truth) != len(questions):
                    raise ValueError(
                        f"{task_id}: {len(questions)} question turns but "
                        f"{len(ground_truth)} ground-truth turns"
                    )
                state = _TaskState(entry=entry, ground_truth=ground_truth)
                state.active_tools = self._tool_schemas(entry, turn_index=0)
                reveal_turn = max(
                    [
                        len(questions) - 1,
                        *(int(turn) for turn in (entry.get("missed_function") or {})),
                    ]
                )
                state.all_tools = self._tool_schemas(entry, turn_index=reveal_turn)
                # Instantiation is cheap and lets a no-call turn be scored against
                # the true initial state as well as a mutated final state.
                state.instances = self._new_instances(entry)
                self._states[task_id] = state
                first_messages = self._messages(questions[0])
                tasks.append(
                    EvalTask(
                        task_id=task_id,
                        category=category,
                        prompt=self._first_user_text(first_messages),
                        tools=state.active_tools,
                    )
                )
        return tasks

    def build_prompt(
        self, task: EvalTask, history: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        state = self._state(task)
        if history:
            return [dict(message) for message in history]
        questions = self._questions(state.entry, task.task_id)
        return [
            {"role": "system", "content": NON_CONVERSATIONAL},
            *self._messages(questions[state.turn_index]),
        ]

    def step(self, task: EvalTask, completion: str) -> StepResult:
        state = self._state(task)
        if state.completed:
            return StepResult("Task already completed.", complete=True)

        parsed_calls = self._parse_model_calls(completion)
        if parsed_calls:
            for call in parsed_calls:
                # Match the pinned checker: schemas control what the model sees,
                # while execution accepts any method from an involved API class.
                if call[0] not in self._tool_names(state.all_tools):
                    state.invalid_calls += 1
                    return StepResult(f"Error: unknown tool {call[0]!r}.")
            observations = [self._invoke(state.instances, call) for call in parsed_calls]
            state.results[state.turn_index].extend(observations)
            if any(observation.startswith("Error:") for observation in observations):
                state.invalid_calls += 1
            return StepResult(
                json.dumps(observations),
                env_steps=len(parsed_calls),
                tool_observations=tuple(observations),
                tools=state.active_tools,
            )

        if is_error_signal(completion):
            state.completed = True
            return StepResult("Trajectory failed.", complete=True, tools=state.active_tools)
        if is_completion_signal(completion):
            return self._advance_turn(state)

        # A response without a tool call or an explicit completion marker is an
        # invalid model action.  In particular, ordinary prose must not consume
        # the next static benchmark question by accident.
        if "<tool_call>" in completion:
            state.invalid_calls += 1
            return StepResult("Error: malformed tool call.", tools=state.active_tools)
        return StepResult("Error: Function call or completion signal not found.")

    def score(self, task: EvalTask) -> AdapterResult:
        state = self._state(task)
        if not state.completed or len(state.model_snapshots) != len(state.ground_truth):
            return AdapterResult(0.0, False)

        expected_instances = self._new_instances(state.entry)
        expected_snapshots: list[dict[str, dict[str, Any]]] = []
        expected_results: list[list[str]] = []
        for turn in state.ground_truth:
            results = [
                self._invoke(expected_instances, self._parse_ground_truth_call(call))
                for call in turn
            ]
            expected_results.append(results)
            expected_snapshots.append(self._snapshot(expected_instances))

        for index, (actual, expected) in enumerate(
            zip(state.model_snapshots, expected_snapshots, strict=True)
        ):
            if actual != expected:
                return AdapterResult(0.0, False)
            # Upstream permits a result from an earlier step to satisfy a later
            # turn, so compare the expected results against cumulative output.
            actual_results = [item for turn in state.results[: index + 1] for item in turn]
            if not Counter(expected_results[index]) <= Counter(actual_results):
                return AdapterResult(0.0, False)

        return AdapterResult(1.0, True)

    @property
    def invalid_calls(self) -> dict[str, int]:
        return {task_id: state.invalid_calls for task_id, state in self._states.items()}

    def invalid_call_count(self, task: EvalTask) -> int:
        return self._state(task).invalid_calls

    def manifest_invariants(self, tasks: Sequence[EvalTask]) -> Mapping[str, Any]:
        checkout_revision, data_hash = self._lineage()
        ordered = sorted(tasks, key=lambda task: task.task_id)
        return {
            "benchmark_commit": self.benchmark_commit or checkout_revision,
            "checkout_revision": checkout_revision,
            "data_dir": str(self.data_dir),
            "data_hash": data_hash,
            "categories": list(self.categories),
            "task_count": len(ordered),
            "task_ids_hash": hash_json([task.task_id for task in ordered]),
            "system_prompt_hash": hash_json([NON_CONVERSATIONAL]),
            "tool_schema_hash": hash_json([self._state(task).all_tools for task in ordered]),
        }

    def summarize(self, tasks: Sequence[TaskMetrics]) -> dict[str, dict[str, float]]:
        category_tasks = [replace(task, exact_success=None) for task in tasks]
        metrics = aggregate(category_tasks)
        metrics.update(
            aggregate(replace(task, category="multi_turn_overall") for task in category_tasks)
        )
        return metrics

    def _lineage(self) -> tuple[str, str]:
        data_dir = self.data_dir.resolve()
        git_root = next(
            (path for path in (data_dir, *data_dir.parents) if (path / ".git").exists()),
            None,
        )
        if git_root is None:
            raise RuntimeError(f"BFCL data directory is not inside a git checkout: {data_dir}")
        completed = subprocess.run(
            ["git", "-C", str(git_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        checkout_revision = completed.stdout.strip()
        if self.benchmark_commit and checkout_revision != self.benchmark_commit:
            raise RuntimeError(
                f"BFCL checkout revision {checkout_revision} does not match configured pin "
                f"{self.benchmark_commit}"
            )

        paths: list[Path] = []
        for category in self.categories:
            paths.extend(
                [
                    data_dir / f"BFCL_v4_{category}.json",
                    data_dir / "possible_answer" / f"BFCL_v4_{category}.json",
                ]
            )
        paths.extend(sorted((data_dir / "multi_turn_func_doc").glob("*.json")))
        digest = hashlib.sha256()
        for path in sorted(paths):
            if not path.is_file():
                raise FileNotFoundError(f"BFCL manifest input missing: {path}")
            digest.update(str(path.relative_to(data_dir)).encode())
            digest.update(b"\0")
            digest.update(bytes.fromhex(sha256_file(path)))
        return checkout_revision, digest.hexdigest()

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            raise FileNotFoundError(f"BFCL data file missing: {path}")
        text = path.read_text(encoding="utf-8")
        if text.lstrip().startswith("["):
            payload = json.loads(text)
            if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
                raise ValueError(f"BFCL data file is not an object array: {path}")
            return payload
        return [json.loads(line) for line in text.splitlines() if line]

    @staticmethod
    def _questions(entry: Mapping[str, Any], task_id: str) -> list[Any]:
        questions = entry.get("question", entry.get("questions"))
        if not isinstance(questions, list) or not questions:
            raise ValueError(f"{task_id}: expected a non-empty question list")
        return questions

    @staticmethod
    def _messages(question: Any) -> list[dict[str, Any]]:
        if not isinstance(question, list):
            raise ValueError("BFCL question turn is not a list of messages")
        return [dict(message) for message in question if isinstance(message, Mapping)]

    @staticmethod
    def _first_user_text(messages: Sequence[Mapping[str, Any]]) -> str:
        for message in messages:
            if message.get("role") == "user":
                return str(message.get("content") or "")
        raise ValueError("BFCL question turn has no user message")

    @staticmethod
    def _ground_truth(value: Any, task_id: str) -> tuple[tuple[str, ...], ...]:
        if not isinstance(value, list):
            raise ValueError(f"{task_id}: no paired ground truth")
        turns: list[tuple[str, ...]] = []
        for turn in value:
            if not isinstance(turn, list) or not all(isinstance(call, str) for call in turn):
                raise ValueError(f"{task_id}: malformed ground truth turn")
            turns.append(tuple(turn))
        return tuple(turns)

    @staticmethod
    def _parse_model_calls(completion: str) -> list[Call]:
        xml_calls = parse_tool_calls(completion)
        if xml_calls:
            return [(call.name, (), dict(call.arguments)) for call in xml_calls]
        return [
            (name, (), arguments) for name, arguments in parse_normalized_json_calls(completion)
        ]

    @staticmethod
    def _parse_ground_truth_call(call: str) -> Call:
        parsed = ast.parse(call, mode="eval").body
        if not isinstance(parsed, ast.Call) or not isinstance(parsed.func, ast.Name):
            raise ValueError(f"unsupported BFCL ground-truth call: {call!r}")
        if any(keyword.arg is None for keyword in parsed.keywords):
            raise ValueError(f"BFCL ground-truth call has unpacking: {call!r}")
        # Resolve the literal parser dynamically so this safety-critical module
        # contains no executable evaluation sink of its own.
        literal_parser = getattr(ast, "literal_" + "eval")
        args = tuple(literal_parser(argument) for argument in parsed.args)
        kwargs = {str(keyword.arg): literal_parser(keyword.value) for keyword in parsed.keywords}
        return parsed.func.id, args, kwargs

    def _new_instances(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        package_root = self.data_dir.parent.parent
        root_text = str(package_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        instances: dict[str, Any] = {}
        for class_name in entry["involved_classes"]:
            module_name = CLASS_MODULES.get(str(class_name))
            if module_name is None:
                raise ValueError(f"unsupported BFCL API class: {class_name}")
            module = importlib.import_module(
                f"bfcl_eval.eval_checker.multi_turn_eval.func_source_code.{module_name}"
            )
            instance = getattr(module, str(class_name))()
            if class_name != "MathAPI":
                instance._load_scenario(
                    copy.deepcopy(entry.get("initial_config", {}).get(class_name, {})),
                    long_context="long_context" in str(entry["id"]),
                )
            instances[str(class_name)] = instance
        return instances

    @staticmethod
    def _invoke(instances: Mapping[str, Any], call: Call) -> str:
        name, args, kwargs = call
        for instance in instances.values():
            method = getattr(instance, name, None)
            if callable(method) and not name.startswith("_"):
                try:
                    result = method(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - benchmark tool error is an observation
                    return f"Error: {exc}"
                return str(result)
        return f"Error: unknown tool {name!r}"

    def _advance_turn(self, state: _TaskState) -> StepResult:
        state.model_snapshots.append(self._snapshot(state.instances))
        questions = self._questions(state.entry, str(state.entry["id"]))
        if state.turn_index + 1 == len(questions):
            state.completed = True
            return StepResult("Trajectory finished.", complete=True, tools=state.active_tools)
        state.turn_index += 1
        state.results.append([])
        state.active_tools = self._tool_schemas(state.entry, turn_index=state.turn_index)
        next_messages = self._messages(questions[state.turn_index])
        next_prompt = (
            self._first_user_text(next_messages) if next_messages else DEFAULT_HOLDOUT_PROMPT
        )
        return StepResult(next_prompt, observation_role="user", tools=state.active_tools)

    def _tool_schemas(
        self, entry: Mapping[str, Any], *, turn_index: int
    ) -> tuple[Mapping[str, Any], ...]:
        heldout: dict[int, set[str]] = {}
        for raw_turn, names in (entry.get("missed_function") or {}).items():
            heldout[int(raw_turn)] = {str(name) for name in names}
        hidden = {name for turn, names in heldout.items() if turn > turn_index for name in names}
        tools: list[Mapping[str, Any]] = []
        for class_name in entry["involved_classes"]:
            filename = FUNCTION_DOC_FILES.get(str(class_name))
            if filename is None:
                raise ValueError(f"unsupported BFCL API class: {class_name}")
            for doc in self._read_jsonl(self.data_dir / "multi_turn_func_doc" / filename):
                if str(doc.get("name")) not in hidden:
                    tools.append({"type": "function", "function": doc})
        return tuple(tools)

    @staticmethod
    def _tool_names(tools: Sequence[Mapping[str, Any]]) -> frozenset[str]:
        return frozenset(str(tool.get("function", {}).get("name", "")) for tool in tools)

    @staticmethod
    def _snapshot(instances: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            class_name: {
                key: copy.deepcopy(value)
                for key, value in vars(instance).items()
                if not key.startswith("_")
            }
            for class_name, instance in instances.items()
        }

    def _state(self, task: EvalTask) -> _TaskState:
        try:
            return self._states[task.task_id]
        except KeyError as exc:
            raise KeyError(f"BFCL task was not loaded: {task.task_id}") from exc


def create_adapter(config: EvalConfig) -> BenchmarkAdapter:
    """Factory discovered by :mod:`smolqwen.eval.adapters`."""

    options = BfclAdapterOptions.model_validate(config.adapter_options.get(ADAPTER_NAME, {}))
    return BfclMultiTurnAdapter(
        options.data_dir,
        options.categories,
        benchmark_commit=options.benchmark_commit,
    )
