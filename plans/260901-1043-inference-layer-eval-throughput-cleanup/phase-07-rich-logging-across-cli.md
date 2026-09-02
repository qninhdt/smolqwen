---
phase: 7
title: "Rich logging across every CLI"
status: pending
priority: P2
effort: "1.5d"
dependencies: [8]
---

# Phase 7: Rich logging across every CLI

## Overview

Give every command visible progress and readable errors through rich on stderr,
replacing bare `print()` calls, while leaving **every** machine-readable stdout
emitter byte-identical — there are at least six, and two are consumed by
committed notebooks.

## Requirements

- Functional: one console module owns rich setup; modules log through `logging`
  with a `smolqwen.<module>` logger.
- Functional: every machine-readable stdout emitter keeps byte-identical output,
  enumerated before any code changes.
- Functional: level controllable by `--verbose`, `--quiet`, `SMOLQWEN_LOG_LEVEL`.
- Functional: `transformers`, `trl`, and `vllm` loggers route through the same
  handler.
- Non-functional: the console module must not import torch or vllm.

## Architecture

**stdout is a contract, and it is wider than the first draft assumed.** That
version named three emitters and scheduled print replacement in five modules
containing five more. The real list:

| Emitter | Consumer |
|---|---|
| `eval/runner.py:185` report paths JSON | `test_cli_dry_run.py` |
| `cli.py:430` resolved config summary | `test_cli_dry_run.py` |
| `cli.py:256` + `:259` probe report and path | `test_cli_dry_run.py:73` |
| `env/selftest.py:197` JSON | — |
| `rollout/bench.py:290` JSON | — |
| `training/grpo.py:668` difficulty JSON | **`notebooks/03-grpo.ipynb`** |
| `training/merge.py:126` merge report JSON | **`notebooks/01-sft.ipynb`** |
| `serving/server.py:87` argv from `--print-command` | `docs/serving.md:33` |

Two are read by notebooks, and notebook changes are a non-goal — so those two
cannot move to stderr, full stop. `serve --print-command` exists to be captured
by a shell, and no test guards it today.

So the split: rich to stderr, and every row above stays on stdout unchanged.

```python
# src/smolqwen/console.py  — no torch, no vllm
console = Console(stderr=True)
logging.basicConfig(handlers=[RichHandler(console=console, rich_tracebacks=True)])
```

Today's silence is the concrete failure: `eval/runner.py` runs for hours and
emits one JSON line at the end, so a stalled run is indistinguishable from a slow
one. `data/cli_actions.py:70` already solved the notebook case — a rich bar plus
a periodic plain line, because Colab is not a TTY — and that pattern is promoted
to a shared helper rather than reinvented.

Error paths collapse: four handlers follow
`print(f"...: {exc}", file=sys.stderr); return 2`. Four, not six, because Phase 8
runs first and deletes the `bench` and `sweep` subcommands those two handlers
served — which is also why this phase depends on Phase 8 rather than running
beside it. Both editing `cli.py` concurrently was a conflict the first draft
authorized.

## Related Code Files

- Create: `src/smolqwen/console.py` — console, logging setup, progress helper,
  status table
- Create: `tests/test_console_stdout_contract.py` — captures stdout and stderr
  separately for **every** emitter in the table, including
  `serve --print-command`, `merge-adapter`, `profile-difficulty`, `env-selftest`
- Create: `tests/test_console_no_heavy_imports.py` — torch and vllm absent from
  `sys.modules` after import
- Modify: `src/smolqwen/cli.py` — `--verbose`/`--quiet`; init logging before
  dispatch; collapse the four remaining error handlers
- Modify: `src/smolqwen/data/cli_actions.py` — shared progress helper, drop the 15
  local prints
- Modify: `src/smolqwen/eval/runner.py` — per-task progress on stderr; stdout JSON
  untouched
- Modify: `training/sft.py`, `training/grpo.py`, `training/merge.py`,
  `env/selftest.py`, `serving/server.py`, `rollout/bench.py` — replace
  human-progress prints only; leave every JSON emitter alone
- Modify: `tests/test_cli_dry_run.py` — extend to assert stdout purity per command

## Implementation Steps

1. Enumerate emitters from `grep -n "print(" src/` and classify each as
   machine-readable or human-progress. Write the classification into the phase
   report **before** editing code. The table above is the starting point, not the
   final answer.
2. Write `console.py`: console on stderr, `configure_logging(level)` with
   `RichHandler(rich_tracebacks=True)`, `progress_task()` carrying the non-TTY
   fallback from `cli_actions.py:70`, `status_table()`.
3. Route `transformers`, `trl`, `vllm` loggers into the same handler.
4. Add `--verbose`/`--quiet` to `_add_common`; read `SMOLQWEN_LOG_LEVEL`; call
   `configure_logging` in `main()` before dispatch.
5. Replace human-progress prints module by module. After each module, run
   `test_console_stdout_contract.py`.
6. Collapse the four error handlers, preserving exit code 2.
7. Add per-task progress to the eval runner: task index, category, elapsed,
   running score.

## Success Criteria

- [ ] Every emitter in the classification has a stdout assertion, and captured
      stdout is byte-identical to pre-phase output for all of them
- [ ] `notebooks/03-grpo.ipynb` and `01-sft.ipynb` still see their JSON on stdout
- [ ] `serve --print-command` still prints argv to stdout, now with a test
- [ ] `import smolqwen.console` pulls in neither torch nor vllm
- [ ] `smolqwen --dry-run` works for every subcommand without vllm installed
- [ ] Non-TTY runs emit periodic plain lines instead of a broken bar
- [ ] Library logs route through the same handler
- [ ] Error paths log a rich traceback and exit 2
- [ ] CPU suite green

## Risk Assessment

The assumption is that step 1's enumeration is complete. The first draft assumed
three emitters and was wrong by at least a factor of two, so the failure mode is
demonstrated rather than hypothetical.

- Signal it broke: a notebook cell or a documented command starts seeing empty
  stdout.
- Response: the enumeration is a written deliverable checked before code changes,
  and every entry gets a test. Phase 8 running first also shrinks the surface by
  deleting two subcommands.

Second risk: rich progress on a non-TTY Colab cell floods output with redraw
frames — the exact problem `cli_actions.py:70` was written to avoid.

- Signal it broke: Colab output fills with repeated bar frames.
- Response: the shared helper carries that fallback by construction, but a local
  TTY cannot reproduce the failure — verify on a real Colab cell during Phase 10.
