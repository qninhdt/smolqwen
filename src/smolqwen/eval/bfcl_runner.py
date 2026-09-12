"""Direct BFCL evaluation over a batched generation callback.

Supports both multi-turn (multi_turn_*) and single-turn (simple_*, parallel,
etc.) categories. Scoring uses BFCL's own upstream checkers."""

from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from smolqwen.config_models import EvalConfig
from smolqwen.console import progress_task
from smolqwen.data.loader import Message, ToolCall, parse_message
from smolqwen.data.render import render_prefix
from smolqwen.data.tool_call_xml import parse_tool_calls
from smolqwen.eval.manifest import hash_json
from smolqwen.eval.metrics import TaskMetrics, aggregate
from smolqwen.eval.tool_calls import parse_normalized_json_calls
from smolqwen.eval.trajectories import TrajectoryRecord
from smolqwen.inference.decoding import split_generation_continuation

BFCL_ROOT = (
    Path(__file__).resolve().parents[3]
    / "third_party"
    / "gorilla"
    / "berkeley-function-call-leaderboard"
)
MULTI_TURN_CATEGORIES = frozenset(
    {
        "multi_turn_base",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
        "multi_turn_long_context",
    }
)
DEFAULT_CATEGORIES: list[str] = ["multi_turn_base"]
HOLDOUT_PROMPT = "I have updated some more functions you can choose from. What about now?"
LENGTH_MARGIN = 8
_JSON_TOOL_BLOCK = re.compile(r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>", re.DOTALL)


def is_multi_turn(category: str) -> bool:
    return category in MULTI_TURN_CATEGORIES


@dataclass(frozen=True)
class BfclTask:
    task_id: str
    category: str
    entry: Mapping[str, Any]
    ground_truth: Any


@dataclass(frozen=True)
class BfclRequest:
    task_id: str
    prompt_ids: tuple[int, ...]
    max_new_tokens: int


@dataclass(frozen=True)
class BfclCompletion:
    task_id: str
    text: str
    generated_tokens: int
    finish_reason: str | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


GenerateBatch = Callable[[Sequence[BfclRequest]], Sequence[BfclCompletion]]


@dataclass
class _Episode:
    task: BfclTask
    messages: list[Message]
    tools: list[dict[str, Any]]
    responses: list[list[list[str]]]
    started_at: float = field(default_factory=time.monotonic)
    turn_index: int = 0
    turn_steps: int = 0
    generation_turns: int = 0
    env_steps: int = 0
    generated_tokens: int = 0
    invalid_calls: int = 0
    observations: list[str] = field(default_factory=list)
    completed: bool = False
    terminal_reason: str | None = None
    truncated: bool = False
    recorded: bool = False


@dataclass(frozen=True)
class BfclRun:
    metrics: dict[str, dict[str, float]]
    invariants: Mapping[str, Any]


def load_bfcl_tasks(
    categories: Sequence[str] = DEFAULT_CATEGORIES,
    expected_revision: str | None = None,
) -> tuple[list[BfclTask], str]:
    """Load BFCL's dataset and ground truth for the requested categories."""

    revision = _bfcl_revision()
    if expected_revision and revision != expected_revision:
        raise RuntimeError(
            f"BFCL checkout revision {revision} does not match configured pin {expected_revision}"
        )
    load_dataset_entry, load_ground_truth_entry, _, _ = _bfcl_api()
    tasks: list[BfclTask] = []
    for category in categories:
        entries = load_dataset_entry(category)
        expected = {
            str(row["id"]): row["ground_truth"] for row in load_ground_truth_entry(category)
        }
        for entry in entries:
            task_id = str(entry["id"])
            ground_truth = expected[task_id]
            if is_multi_turn(category):
                ground_truth = tuple(tuple(call for call in turn) for turn in ground_truth)
            tasks.append(BfclTask(task_id, category, entry, ground_truth))
    return tasks, revision


def evaluate_bfcl(
    config: EvalConfig,
    *,
    tokenizer: Any,
    generate: GenerateBatch,
    tasks: Sequence[BfclTask],
    benchmark_revision: str,
    record_sink: Callable[[TrajectoryRecord], None] | None = None,
) -> BfclRun:
    """Run BFCL's benchmark and grade each task with its upstream checker.

    Multi-turn categories use the turn-advancement loop and ``multi_turn_checker``;
    single-turn categories use single-shot generation and ``ast_checker``.
    """

    multi = [t for t in tasks if is_multi_turn(t.category)]
    single = [t for t in tasks if not is_multi_turn(t.category)]

    task_metrics: list[TaskMetrics] = []
    records: list[TrajectoryRecord] = []

    def _sink(record: TrajectoryRecord) -> None:
        records.append(record)
        if record_sink is not None:
            record_sink(record)

    if multi:
        task_metrics.extend(_evaluate_multi_turn(config, tokenizer, generate, multi, _sink))
    if single:
        task_metrics.extend(_evaluate_single_turn(config, tokenizer, generate, single, _sink))

    metrics = aggregate(task_metrics)
    categories = sorted({t.category for t in tasks})
    return BfclRun(
        metrics=metrics,
        invariants={
            "benchmark": "BFCL",
            "benchmark_commit": benchmark_revision,
            "categories": categories,
            "task_count": len(tasks),
            "task_ids_hash": hash_json([task.task_id for task in tasks]),
            "tool_schema_hash": hash_json([task.entry["function"] for task in tasks]),
            "system_prompt": None,
            "checker": "multi_turn_checker + ast_checker"
            if (multi and single)
            else "ast_checker"
            if single
            else "multi_turn_checker",
        },
    )


def _evaluate_multi_turn(
    config: EvalConfig,
    tokenizer: Any,
    generate: GenerateBatch,
    tasks: Sequence[BfclTask],
    record_sink: Callable[[TrajectoryRecord], None],
) -> list[TaskMetrics]:
    """Multi-turn loop: turn advancement, upstream ``multi_turn_checker``."""

    run_key = f"smolqwen_{uuid.uuid4().hex}"
    episodes = [_episode(task) for task in tasks]
    task_metrics: list[TaskMetrics] = []

    with progress_task("multi-turn", total=len(episodes), unit="tasks", every=1) as advance:
        while any(not episode.recorded for episode in episodes):
            ready = [
                episode
                for episode in episodes
                if not episode.recorded and episode.terminal_reason is None
            ][: config.profile.generation_concurrency]

            requests: list[BfclRequest] = []
            requested_episodes: list[_Episode] = []
            for episode in ready:
                prompt_ids = _render_ids(tokenizer, episode, config.enable_thinking)
                remaining = config.profile.max_seq_length - len(prompt_ids) - LENGTH_MARGIN
                if remaining < 1:
                    episode.terminal_reason = "step_cap"
                    continue
                requests.append(
                    BfclRequest(
                        episode.task.task_id,
                        tuple(prompt_ids),
                        min(config.decoding.max_new_tokens, remaining),
                    )
                )
                requested_episodes.append(episode)

            if requests:
                completions = list(generate(requests))
                if len(completions) != len(requests):
                    raise RuntimeError(
                        f"generation returned {len(completions)} rows for {len(requests)} requests"
                    )
                for episode, request, completion in zip(
                    requested_episodes, requests, completions, strict=True
                ):
                    if completion.task_id != request.task_id:
                        raise RuntimeError(
                            f"generation result {completion.task_id!r} does not match request "
                            f"{request.task_id!r}"
                        )
                    _advance_episode(
                        episode, completion, run_key, max_steps=config.max_steps_per_task
                    )

            finished = [
                episode
                for episode in episodes
                if not episode.recorded and episode.terminal_reason is not None
            ]
            for episode in finished:
                metric, record = _score_episode(episode, run_key)
                task_metrics.append(metric)
                record_sink(record)
                episode.recorded = True
                running = sum(item.score for item in task_metrics) / len(task_metrics)
                advance(f"mean {running:.3f}")

            if not requests and not finished:
                raise RuntimeError("BFCL evaluation made no progress")

    return task_metrics


def _evaluate_single_turn(
    config: EvalConfig,
    tokenizer: Any,
    generate: GenerateBatch,
    tasks: Sequence[BfclTask],
    record_sink: Callable[[TrajectoryRecord], None],
) -> list[TaskMetrics]:
    """Single-shot generation for one-turn categories, graded by ``ast_checker``."""

    task_metrics: list[TaskMetrics] = []

    with progress_task("single-turn", total=len(tasks), unit="tasks", every=1) as advance:
        for offset in range(0, len(tasks), config.profile.generation_concurrency):
            batch = tasks[offset : offset + config.profile.generation_concurrency]

            requests: list[BfclRequest] = []
            for task in batch:
                messages = _question_messages(task.entry, 0)
                prompt_ids = _render_prompt_ids(
                    tokenizer,
                    messages,
                    [_tool_schema(doc) for doc in task.entry["function"]],
                    config.enable_thinking,
                )
                remaining = config.profile.max_seq_length - len(prompt_ids) - LENGTH_MARGIN
                requests.append(
                    BfclRequest(
                        task.task_id,
                        tuple(prompt_ids),
                        min(config.decoding.max_new_tokens, remaining),
                    )
                )

            completions = list(generate(requests))
            by_id = {completion.task_id: completion for completion in completions}

            for task in batch:
                completion = by_id[task.task_id]
                calls = _parse_calls(completion.text, tools=task.entry["function"])
                model_output = [{call.name: dict(call.arguments)} for call in calls]
                result = _ast_check(
                    list(task.entry["function"]),
                    model_output,
                    list(task.ground_truth),
                    _ast_language(task.category),
                    task.category,
                    "smolqwen",
                )
                valid = bool(result.get("valid"))
                # Single-shot AST grading: exactly one generation step and no tool
                # execution, so there is no runtime invalid-call signal to count.
                metric = TaskMetrics(
                    category=task.category,
                    score=float(valid),
                    invalid_calls=0,
                    steps=1,
                    generated_tokens=completion.generated_tokens,
                    truncated=completion.truncated,
                    exact_success=valid,
                    diagnostics={"completion_rate": 1.0},
                )
                task_metrics.append(metric)
                running = sum(item.score for item in task_metrics) / len(task_metrics)
                advance(f"mean {running:.3f}")

    return task_metrics


def _episode(task: BfclTask) -> _Episode:
    entry = task.entry
    tools = [_tool_schema(doc) for doc in entry["function"]]
    messages = _question_messages(entry, 0)
    return _Episode(
        task=task,
        messages=messages,
        tools=tools,
        responses=[[] for _ in task.ground_truth],
    )


def _advance_episode(
    episode: _Episode, completion: BfclCompletion, run_key: str, *, max_steps: int
) -> None:
    episode.generation_turns += 1
    episode.generated_tokens += completion.generated_tokens
    episode.truncated = episode.truncated or completion.truncated
    reasoning, content = split_generation_continuation(completion.text, thinking=False)
    calls = _parse_calls(content, tools=episode.tools)
    if not calls:
        episode.messages.append(
            Message(role="assistant", content=content, reasoning_content=reasoning)
        )
        _next_turn(episode)
        return

    episode.messages.append(
        Message(role="assistant", content="", reasoning_content=reasoning, tool_calls=tuple(calls))
    )
    if episode.turn_steps >= max_steps:
        episode.terminal_reason = "step_cap"
        return
    known = _known_tool_names(episode.task.entry)
    executable = [_call_text(call) for call in calls if call.name in known]
    unknown = [call.name for call in calls if call.name not in known]
    results: list[str] = []
    if executable:
        results.extend(_execute(executable, episode.task.entry, run_key, episode.task.task_id))
        episode.responses[episode.turn_index].append(executable)
    for name in unknown:
        results.append(f"Error during execution: Unknown function {name!r}")
    episode.turn_steps += 1
    episode.env_steps += len(calls)
    episode.invalid_calls += sum(_is_error_result(result) for result in results)
    for result in results:
        episode.messages.append(Message(role="tool", content=result))
        episode.observations.append(result)


def _next_turn(episode: _Episode) -> None:
    if episode.turn_index + 1 == len(episode.task.ground_truth):
        episode.completed = True
        episode.terminal_reason = "final_answer"
        return
    episode.turn_index += 1
    episode.turn_steps = 0
    entry = episode.task.entry
    heldout = entry.get("missed_function", {})
    for doc in heldout.get(str(episode.turn_index), []):
        episode.tools.append(_tool_schema(doc))
    episode.messages.extend(_question_messages(entry, episode.turn_index))


def _question_messages(entry: Mapping[str, Any], turn_index: int) -> list[Message]:
    raw = entry["question"][turn_index]
    if raw:
        return [parse_message(message) for message in raw]
    return [Message(role="user", content=HOLDOUT_PROMPT)]


def _score_episode(episode: _Episode, run_key: str) -> tuple[TaskMetrics, TrajectoryRecord]:
    valid = False
    failure_reason: str | None = "multi_turn:force_terminated"
    if episode.completed:
        result = _check(
            episode.responses,
            [list(turn) for turn in episode.task.ground_truth],
            dict(episode.task.entry),
            run_key,
            episode.task.category,
        )
        valid = bool(result.get("valid"))
        failure_reason = None if valid else str(result.get("error_type") or "bfcl_check_failed")
    diagnostics = {"completion_rate": float(episode.completed)}
    metric = TaskMetrics(
        category=episode.task.category,
        score=float(valid),
        invalid_calls=episode.invalid_calls,
        steps=episode.env_steps,
        generated_tokens=episode.generated_tokens,
        truncated=episode.truncated,
        exact_success=valid,
        diagnostics=diagnostics,
        terminal_reason=episode.terminal_reason,
    )
    record = TrajectoryRecord(
        task_id=episode.task.task_id,
        category=episode.task.category,
        messages=[message.to_template_dict() for message in episode.messages],
        observations=list(episode.observations),
        score=float(valid),
        exact_success=valid,
        completed=episode.completed,
        failure_reason=failure_reason,
        failed_check_names=[] if failure_reason is None else [failure_reason],
        diagnostics=diagnostics,
        terminal_reason=episode.terminal_reason,
        generation_turns=episode.generation_turns,
        env_steps=episode.env_steps,
        generated_tokens=episode.generated_tokens,
        truncated=episode.truncated,
        invalid_calls=episode.invalid_calls,
        wall_s=time.monotonic() - episode.started_at,
    )
    return metric, record


def _parse_calls(text: str, *, tools: Sequence[Mapping[str, Any]] = ()) -> list[ToolCall]:
    calls = parse_tool_calls(text, tools=tools)
    if calls:
        return calls
    json_calls: list[ToolCall] = []
    for match in _JSON_TOOL_BLOCK.finditer(text):
        try:
            payload = json.loads(match.group("body"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping) and isinstance(payload.get("name"), str):
            arguments = payload.get("arguments", {})
            if isinstance(arguments, Mapping):
                json_calls.append(ToolCall(str(payload["name"]), dict(arguments)))
    if json_calls:
        return json_calls
    return [ToolCall(name, arguments) for name, arguments in parse_normalized_json_calls(text)]


def _render_ids(tokenizer: Any, episode: _Episode, enable_thinking: bool) -> list[int]:
    return _render_prompt_ids(tokenizer, episode.messages, episode.tools, enable_thinking)


def _render_prompt_ids(
    tokenizer: Any,
    messages: Sequence[Message],
    tools: Sequence[dict[str, Any]],
    enable_thinking: bool,
) -> list[int]:
    text = render_prefix(
        tokenizer,
        messages,
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    return [int(token) for token in ids]


def _tool_schema(doc: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "function", "function": dict(doc)}


def _known_tool_names(entry: Mapping[str, Any]) -> set[str]:
    docs = list(entry["function"])
    for heldout in entry.get("missed_function", {}).values():
        docs.extend(heldout)
    return {str(doc["name"]) for doc in docs}


def _call_text(call: ToolCall) -> str:
    arguments = ", ".join(f"{name}={value!r}" for name, value in call.arguments.items())
    return f"{call.name}({arguments})"


def _is_error_result(result: str) -> bool:
    if result.startswith("Error during execution:"):
        return True
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, Mapping) and "error" in payload


def _execute(calls: list[str], entry: Mapping[str, Any], run_key: str, task_id: str) -> list[str]:
    _, _, execute_multi_turn_func_call, _ = _bfcl_api()
    results, _ = execute_multi_turn_func_call(
        calls,
        dict(entry.get("initial_config", {})),
        list(entry["involved_classes"]),
        run_key,
        task_id,
        long_context=False,
        is_evaL_run=False,
    )
    return [str(result) for result in results]


def _check(
    responses: list[list[list[str]]],
    ground_truth: list[list[str]],
    entry: dict[str, Any],
    run_key: str,
    category: str,
) -> Mapping[str, Any]:
    _, _, _, multi_turn_checker = _bfcl_api()
    return cast(
        Mapping[str, Any],
        multi_turn_checker(responses, ground_truth, entry, category, f"{run_key}_score"),
    )


def _ast_check(
    func_description: list[dict[str, Any]],
    model_output: list[dict[str, Any]],
    possible_answer: list[dict[str, Any]],
    language: Any,
    test_category: str,
    model_name: str,
) -> Mapping[str, Any]:
    ast_checker_mod = _bfcl_ast_api()
    return cast(
        Mapping[str, Any],
        ast_checker_mod.ast_checker(
            func_description, model_output, possible_answer, language, test_category, model_name
        ),
    )


def _ast_language(category: str) -> Any:
    from bfcl_eval.constants.enums import Language

    if "java" in category:
        return Language.JAVA
    if "javascript" in category or "js" in category:
        return Language.JAVASCRIPT
    return Language.PYTHON


def _bfcl_ast_api() -> Any:
    root = str(BFCL_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    # ast_checker.py imports model_config → handler chain → tree_sitter.
    # We only need the checker functions; stub the unused import to avoid
    # pulling the full handler tree into the eval environment.
    from types import ModuleType

    mock = ModuleType("bfcl_eval.constants.model_config")
    mock.MODEL_CONFIG_MAPPING = {}  # type: ignore[attr-defined]
    sys.modules.setdefault("bfcl_eval.constants.model_config", mock)
    return importlib.import_module("bfcl_eval.eval_checker.ast_eval.ast_checker")


def _bfcl_api() -> tuple[Any, Any, Any, Any]:
    root = str(BFCL_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    checker = importlib.import_module("bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker")
    utilities = importlib.import_module("bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils")
    loaders = importlib.import_module("bfcl_eval.utils")

    return (
        loaders.load_dataset_entry,
        loaders.load_ground_truth_entry,
        utilities.execute_multi_turn_func_call,
        checker.multi_turn_checker,
    )


def _bfcl_revision() -> str:
    completed = subprocess.run(
        ["git", "-C", str(BFCL_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip()
