"""Build rendered agent-shaped traffic from the repository's pinned BFCL tasks."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from smolqwen.config_models import EvalConfig
from smolqwen.eval.adapters.bfcl import BfclAdapterOptions, BfclMultiTurnAdapter


def build_bfcl_agentic_workload(
    config: EvalConfig,
    *,
    tokenizer: Any,
    output_path: Path | str,
) -> tuple[Path, Path]:
    """Render each pinned BFCL multi-turn task's first agent request and tools.

    vLLM's built-in BFCL loader does not replay the multi-turn categories used by
    the quality evaluation. These custom rows preserve the system/user prompt and
    per-task tool schemas through the real Qwen chat template. They measure an
    agent-shaped serving request, not a BFCL score; Phase 5 remains the quality gate.
    """
    raw = config.adapter_options.get("bfcl_multi_turn", {})
    options = BfclAdapterOptions.model_validate(raw)
    adapter = BfclMultiTurnAdapter(
        options.data_dir,
        options.categories,
        benchmark_commit=options.benchmark_commit,
    )
    tasks = adapter.load_tasks()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    category_counts = Counter(task.category for task in tasks)
    tool_counts: list[int] = []
    with target.open("w", encoding="utf-8") as handle:
        for task in tasks:
            messages = adapter.build_prompt(task, [])
            prompt = tokenizer.apply_chat_template(
                messages,
                tools=list(task.tools),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            if not isinstance(prompt, str):
                raise TypeError("tokenizer returned a non-string rendered BFCL prompt")
            handle.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
            tool_counts.append(len(task.tools))

    composition = target.with_suffix(".composition.json")
    composition.write_text(
        json.dumps(
            {
                "source": "pinned BFCL multi_turn_* tasks",
                "shape": "first request per task, Qwen-rendered with per-task tool schema",
                "quality_claim": False,
                "task_count": len(tasks),
                "category_counts": dict(sorted(category_counts.items())),
                "min_tools": min(tool_counts, default=0),
                "max_tools": max(tool_counts, default=0),
                "builtin_loader_limitation": (
                    "vLLM's BFCL benchmark loader does not replay these multi-turn categories"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return target, composition
