#!/usr/bin/env python3
"""Kind-aware audit of declared-but-unreferenced symbols in `src/smolqwen`.

Counting name references across Python files answers the question for a function
and gets it wrong for everything else. `ProfileConfig.active_pool_multiplier` is
read exactly once inside `src/` -- from a `@property` body two lines below its own
declaration -- and set in two profile YAMLs that a Python-only pass never opens.
A Python-only reference count calls it dead; deleting it drops the GRPO prompt
pool from 16 to 8 without raising anything, and then `extra="forbid"` rejects both
profile files at load.

Framework-invoked names fail the same way from the other direction: nothing in
this repo calls `on_save`, because transformers does. So every declaration is
resolved by kind, and a kind carrying surfaces beyond Python is searched across
those surfaces too.

    python scripts/audit-dead-symbols.py
    python scripts/audit-dead-symbols.py --json > inventory.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "src" / "smolqwen"

# Consumer surfaces. `plans/` is excluded on purpose: a plan naming a symbol it
# intends to delete would otherwise keep that symbol alive forever.
SURFACE_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".md", ".ipynb", ".toml", ".json", ".sh"})
SURFACE_NAMES = frozenset({"Makefile", "Dockerfile", "docker-compose.yml"})
EXCLUDED_PREFIXES = ("artifacts/", "plans/", "third_party/")

# Invoked by protocol rather than by name from this repo, so a reference count is
# the wrong question -- the answer is always zero and the symbol is always live.
# transformers `TrainerCallback` hooks, `Trainer` overrides, the pydantic and
# dataclass surfaces, and the adapter-discovery contract in
# `eval/adapters/__init__.py`, which resolves modules through `pkgutil`.
# vLLM also resolves the worker extension by its fully-qualified class name.
FRAMEWORK_INVOKED = frozenset(
    {
        "on_init_end",
        "on_train_begin",
        "on_train_end",
        "on_epoch_begin",
        "on_epoch_end",
        "on_step_begin",
        "on_substep_end",
        "on_step_end",
        "on_optimizer_step",
        "on_pre_optimizer_step",
        "on_evaluate",
        "on_predict",
        "on_prediction_step",
        "on_save",
        "on_log",
        "compute_loss",
        "training_step",
        "prediction_step",
        "evaluation_loop",
        "get_train_dataloader",
        "get_eval_dataloader",
        "_get_train_sampler",
        "_get_eval_sampler",
        "create_optimizer",
        "create_scheduler",
        "model_config",
        "validate_runtime_pairings",
        "MemoryWorkerExtension",
        "ADAPTER_NAME",
        "ADAPTER_ROLE",
        "create_adapter",
    }
)

# Dunders are protocol by definition.
DUNDER = re.compile(r"^__.*__$")


@dataclass(frozen=True)
class Declaration:
    """One declared name, with the kind that decides how to resolve it."""

    name: str
    kind: str
    path: str
    line: int
    owner: str = ""

    @property
    def location(self) -> str:
        return f"{self.path}:{self.line}"


def is_strict_model(node: ast.ClassDef) -> bool:
    """A pydantic config model: its annotated attributes are settable from YAML."""
    return any(
        isinstance(base, ast.Name) and base.id in {"StrictModel", "BaseModel"}
        for base in node.bases
    )


def collect_declarations(path: Path) -> list[Declaration]:
    """Every top-level name plus every class member, tagged by kind."""
    relative = str(path.relative_to(REPO_ROOT))
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[Declaration] = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            found.append(Declaration(node.name, "function", relative, node.lineno))
        elif isinstance(node, ast.ClassDef):
            found.append(Declaration(node.name, "class", relative, node.lineno))
            found.extend(_class_members(node, relative))
        elif isinstance(node, ast.Assign):
            found.extend(
                Declaration(target.id, "module_constant", relative, node.lineno)
                for target in node.targets
                if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found.append(Declaration(node.target.id, "module_constant", relative, node.lineno))

    return found


def _class_members(node: ast.ClassDef, relative: str) -> list[Declaration]:
    """Class members, with pydantic fields separated from ordinary attributes.

    A pydantic field is reachable from any `configs/**/*.yaml` key and from a test
    constructor kwarg, so its kind must carry those surfaces. A `@property` is a
    read site for whatever its body names -- handled by the reference pass, which
    walks every function body including property bodies.
    """
    config_model = is_strict_model(node)
    members: list[Declaration] = []
    for child in node.body:
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            kind = "property" if _has_decorator(child, "property") else "method"
            members.append(Declaration(child.name, kind, relative, child.lineno, node.name))
        elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            kind = "pydantic_field" if config_model else "class_attribute"
            members.append(Declaration(child.target.id, kind, relative, child.lineno, node.name))
    return members


def _has_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == name:
            return True
        if isinstance(decorator, ast.Attribute) and decorator.attr == name:
            return True
    return False


def python_reference_lines(paths: list[Path]) -> dict[str, set[str]]:
    """Map every referenced name to the `path:line` sites that reference it.

    Loads, attribute accesses, keyword arguments and string-literal `getattr`
    targets all count. Walking every function body is what makes a `@property`
    read a reference; the previous pass walked only definitions and missed it.
    """
    sites: dict[str, set[str]] = {}

    def record(name: str, path: Path, line: int) -> None:
        sites.setdefault(name, set()).add(f"{path.relative_to(REPO_ROOT)}:{line}")

    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                record(node.id, path, node.lineno)
            elif isinstance(node, ast.Attribute):
                record(node.attr, path, node.lineno)
            elif isinstance(node, ast.keyword) and node.arg:
                record(node.arg, path, node.lineno)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                # `getattr(profile, "active_pool_multiplier")` and dict-key reads
                # are indistinguishable from prose here, so a bare string counts.
                if node.value.isidentifier():
                    record(node.value, path, node.lineno)
            elif isinstance(node, ast.alias):
                record(node.asname or node.name.rsplit(".", 1)[-1], path, node.lineno)
    return sites


def surface_files() -> list[Path]:
    """Every tracked non-plan file whose suffix or name can carry a reference."""
    listed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    keep: list[Path] = []
    for entry in listed:
        if entry.startswith(EXCLUDED_PREFIXES):
            continue
        path = REPO_ROOT / entry
        if not path.is_file():
            continue
        if path.suffix in SURFACE_SUFFIXES or path.name in SURFACE_NAMES:
            keep.append(path)
    return keep


def text_reference_lines(name: str, texts: dict[str, list[str]], *, skip: str) -> list[str]:
    """Word-boundary hits for `name` across non-Python surfaces.

    Deliberately textual: a YAML key, a Makefile target, a docs code block, and a
    notebook JSON cell are all plain text at this level, and a name reachable only
    from one of them is still live. `skip` drops the declaration's own file so a
    symbol never counts as its own consumer.
    """
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    hits: list[str] = []
    for relative, lines in texts.items():
        if relative == skip:
            continue
        for number, line in enumerate(lines, start=1):
            if pattern.search(line):
                hits.append(f"{relative}:{number}")
    return hits


# Kinds whose consumers can live outside Python. A pydantic field is set from a
# profile or base YAML and documented in prose; a module constant can be named in
# a Makefile recipe or a notebook cell.
TEXT_SEARCHED_KINDS = frozenset({"pydantic_field", "class_attribute", "module_constant"})

# One audited declaration: its identity, its verdict, and the sites that justify it.
Entry = dict[str, Any]


def resolve(
    declaration: Declaration,
    python_sites: dict[str, set[str]],
    texts: dict[str, list[str]],
) -> Entry:
    """Consumers of one declaration, and the verdict that follows from them."""
    if DUNDER.match(declaration.name) or declaration.name in FRAMEWORK_INVOKED:
        return {
            **_identity(declaration),
            "verdict": "framework",
            "consumers": [],
            "reason": "invoked by protocol, never by name from this repository",
        }

    consumers = sorted(
        site
        for site in python_sites.get(declaration.name, set())
        # A declaration's own line is not a consumer; other lines in the same file
        # are -- that is exactly the `@property` case.
        if site != declaration.location
    )
    if declaration.kind in TEXT_SEARCHED_KINDS:
        consumers.extend(text_reference_lines(declaration.name, texts, skip=declaration.path))

    private = declaration.name.startswith("_")
    return {
        **_identity(declaration),
        "verdict": "unreferenced" if not consumers else "live",
        "consumers": sorted(set(consumers)),
        "reason": "" if consumers else ("module-private" if private else "public"),
    }


def _identity(declaration: Declaration) -> Entry:
    return {
        "name": declaration.name,
        "kind": declaration.kind,
        "owner": declaration.owner,
        "declared_at": declaration.location,
    }


def _readable_lines(files: list[Path]) -> dict[str, list[str]]:
    """Non-Python surfaces, read once, so a text search is not O(files x symbols)."""
    texts: dict[str, list[str]] = {}
    for path in files:
        if path.suffix == ".py":
            continue
        try:
            texts[str(path.relative_to(REPO_ROOT))] = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
    return texts


def audit() -> list[Entry]:
    files = surface_files()
    python_sites = python_reference_lines([path for path in files if path.suffix == ".py"])
    texts = _readable_lines(files)

    results: list[Entry] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        for declaration in collect_declarations(path):
            results.append(resolve(declaration, python_sites, texts))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the full inventory as JSON")
    parser.add_argument(
        "--kind",
        action="append",
        default=None,
        help="restrict output to one kind; repeatable",
    )
    arguments = parser.parse_args()

    results = audit()
    if arguments.kind:
        results = [entry for entry in results if entry["kind"] in set(arguments.kind)]

    if arguments.json:
        json.dump(results, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    unreferenced = [entry for entry in results if entry["verdict"] == "unreferenced"]
    by_kind: dict[str, list[Entry]] = {}
    for entry in unreferenced:
        by_kind.setdefault(entry["kind"], []).append(entry)

    for kind in sorted(by_kind):
        print(f"{kind} ({len(by_kind[kind])} unreferenced)")
        for entry in sorted(by_kind[kind], key=lambda item: str(item["declared_at"])):
            owner = f"{entry['owner']}." if entry["owner"] else ""
            print(f"  {owner}{entry['name']:38} {entry['declared_at']}")
        print()

    counts = {"live": 0, "unreferenced": 0, "framework": 0}
    for entry in results:
        counts[entry["verdict"]] += 1
    print(
        f"{len(results)} declarations: {counts['live']} live, "
        f"{counts['unreferenced']} unreferenced, {counts['framework']} framework-invoked"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
