# CPU / Diff-Aware Validation Report — 2026-09-04

## Scope

- Branch: `feat/inference-layer-eval-throughput`; intentionally dirty worktree.
- Read-only validation; no source, test, config, commit, push, or
  `third_party/EnvScaler` edits made intentionally.
- `.codegraph/` absent; native `rg`, `git`, and test tooling used.

## Commands and Results

| Command | Result |
|---|---|
| `PYTHONDONTWRITEBYTECODE=1 uv run pytest -p no:cacheprovider tests/test_sft_device_runtime.py tests/test_grpo_args.py tests/test_gpu_validation_script.py tests/test_optim_toggles.py tests/test_inference_profiles.py tests/test_cli_dry_run.py tests/test_inference_engine_contract.py tests/test_eval_generation_path.py` | **98 passed**, 7.41s |
| `PYTHONDONTWRITEBYTECODE=1 uv run pytest -p no:cacheprovider -m 'not gpu and not dataset'` | **517 passed**, 18 deselected, 2 expected TRL experimental warnings, 85.97s |
| `uv run ruff check src tests && uv run ruff format --check src tests` | Pass; 160 files formatted |
| `uv run ruff check scripts/colab-gpu-validation.py && uv run ruff format --check scripts/colab-gpu-validation.py` | Pass; script formatted |
| `uv run mypy --cache-dir /tmp/smolqwen-mypy-check-current` | Pass; no issues in 160 files |
| `make smoke` | Pass; all 6 stages × 3 profiles plus `probe --no-write`; `smoke ok` |
| `git diff --check` | Pass |
| In-memory `compile()` of `scripts/colab-gpu-validation.py` | Pass |
| `uv run python scripts/colab-gpu-validation.py --help` | Pass; six phases and `--profile {auto,t4,l4,a100}` exposed |
| `uv run python scripts/colab-gpu-validation.py --profile h100 --help` | Expected exit 2; invalid profile rejected |
| `uv run python -c 'import importlib.util; ... vllm ...'` | `vllm=False` |
| `uv run python -c 'import torch; ...'`; `nvidia-smi ...` | CUDA available, 1 device; local RTX 3050 Laptop, sm86, 4096 MiB |

### Transient first full-run observation

The first full CPU command observed a concurrently changing dirty worktree:
`512 passed, 1 failed, 18 deselected`; the failure was a transient
`NameError: Toggle is not defined` in an intermediate version of the untracked
`tests/test_sft_device_runtime.py`. The live file then collected 11 tests and a
focused rerun passed; the same full command against the stable current tree
passed 517/517. No fix was applied.

## Structural Coverage

| Contract | Coverage | Assessment |
|---|---|---|
| sm75/T4 routing | `test_turing_falls_back_to_padded_fp16_sdpa` asserts `sdpa`, FP16, `fp16=True`, `bf16=False`, padded; `_sft_config` argument test also asserts FP16 + SDPA | CPU-injected capability coverage is direct; no real sm75 execution |
| sm80+ routing | Parametrized sm80/sm89 test asserts FA2, BF16, padding-free; missing FA2 wheel and pre-sm75 refusal are asserted | CPU-injected capability coverage is direct; no real Ampere GPU execution |
| CPU assembly | `test_cpu_assembly_keeps_its_existing_padding_free_contract` asserts padding-free and BF16 config behavior | Covered |
| Resolver edge cases | `test_optim_toggles.py` covers no CUDA, missing wheel, sm75, sm89, explicit SDPA; precision downgrade reason covered | Covered at pure-function level |
| Shipped profiles and dry-run | `test_inference_profiles.py`, `test_cli_dry_run.py`, and `make smoke` cover profile resolution. `test_module_import_pulls_in_neither_torch_nor_vllm` plus dry-run CUDA guard cover lazy imports; local environment confirms vLLM absent | Covered on CPU/config path |
| Validation-script profile contract | New `test_gpu_validation_script.py` covers auto mapping for sm75/T4, sm80/A100, sm89/L4 and explicit profile forwarding through `_controller` (4 passed) | Partial: no tests for `_run_child`, timeout/OOM classification, result persistence/resume, phase command bodies, or real GPU phases |
| GRPO per-card routing | `test_grpo_args.py` checks every shipped profile's TRL argument assembly and generation-group divisibility using injected toggles | Does not execute `build_grpo_trainer` hardware resolution or real GRPO/vLLM |

`pytest-cov` is not installed (`pytest_cov=False`), so no numerical coverage
report was produced. The matrix above is the meaningful contract coverage.

## Warnings / Gaps

- No target T4 or L4 session. The local sm86/4-GiB RTX 3050 is not evidence for
  the requested T4/L4 acceptance boundary, and vLLM is absent.
- GPU-marked vLLM adapter capability and worker-side SFT memory tests were not
  run. Do not claim GPU-only adapter acceptance, sleep/memory release, agreement,
  throughput, stability, in-training cost/envelope, or BFCL completion from this
  report.
- The validation script is syntax/lint/help tested and now has profile/controller
  unit tests, but its subprocess, timeout, OOM, artifact-resume, and six real
  phase contracts remain unverified without the Colab environment.
- Final read-only status showed the submodule dirty with an untracked
  `interact_with_env/agent/__pycache__/`; no intentional command entered or
  modified `third_party/EnvScaler`.

Status: DONE_WITH_CONCERNS
Summary: CPU implementation, profile routing contracts, dry-run matrix, and current diff checks are green. Target-GPU execution and deeper validation-controller runtime behavior remain open.
Concerns/Blockers: No target T4/L4 available; vLLM absent locally; GPU-only criteria remain unmeasured.
