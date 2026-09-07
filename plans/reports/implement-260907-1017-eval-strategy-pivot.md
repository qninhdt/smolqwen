# Implement: SFT train-only, GRPO upstream-style dev (BFCL base), prefix-caching fix

Date: 2026-09-07. Survey: `plans/reports/survey-260907-0826-envscaler-eval-strategy-kv-cache.md`. Plan approved tại session (plan mode).

## Đã làm

### A. Prefix-caching fix (bug KV cache)

`src/smolqwen/inference/engine.py` — `OfflineEngine.build()` truyền `enable_prefix_caching=True` vào `LLM(...)` và fail-closed sau build: đọc `llm_engine.vllm_config.cache_config.enable_prefix_caching`, không phải `True` → `EngineError`. Nguyên nhân: vLLM 0.26 resolve flag thành `false` cho Qwen3.5 hybrid khi không truyền (cùng default mà GRPO colocated phải override ở `_force_trl_prefix_caching`). Test mới: `test_build_requests_prefix_caching_and_fails_closed_on_a_change` (tests/test_inference_engine_contract.py) — kwarg được truyền + assert kích hoạt khi fake báo False.

### B. SFT train-only

- `src/smolqwen/training/sft.py`: `eval_strategy="no"`; `build_trainer` không còn `eval_dataset`, `_bench_engine`, bench callback, param `bench_engine`; `run_train_sft` không còn `trainer.evaluate()` cuối run + stdout JSON (`_is_number` xoá); `get_eval_dataloader` override xoá; `Assembled.bench_engine` xoá. `load_shards`/`Shards` giữ nguyên (val.jsonl vẫn được validate nếu có, không consume).
- `src/smolqwen/config_models.py`: xoá `SftConfig.bench_eval`.
- **Xoá** `src/smolqwen/training/checkpoint_eval.py` (CheckpointEngine/build_bench_eval_callback/eval_config_for — chỉ SFT dùng, verified consumers).
- `configs/base/sft.yaml`: bỏ `eval_steps` + block `bench_eval`; comment train-only upstream-parity.
- W&B: chỉ train loss + throughput (`sft/*` qua `ThroughputCallback`).

### C. GRPO upstream-style

- `src/smolqwen/config_models.py`: `CurriculumConfig` bỏ `heldout_env_count`/`heldout_scenarios_per_env`; `BenchEvalConfig` default adapter → `bfcl_multi_turn`, docstring mới.
- `src/smolqwen/training/grpo.py`: bỏ `select_heldout_scenarios` (candidates = toàn bộ scenarios, vẫn qua curriculum filter); bỏ `eval_dataset`; `GrpoAssembly.eval_task_ids` xoá; factory_oracle factories chỉ nhận train_scenarios.
- `src/smolqwen/training/bench_eval.py`: xoá `assert_dev_adapter` + `TEST_BENCHMARK_TOKENS` (dev=test là quyết định user); docstrings mới.
- `configs/base/grpo.yaml`: curriculum không còn heldout; `bench_eval: enabled: true, adapter: bfcl_multi_turn, every_steps: 20, task_limit: 128, baseline_at_step_zero: true, timeout_s: 2400.0`. Cadence 20 = save_steps (mỗi score gắn checkpoint-N) và dày hơn upstream so theo dữ liệu (upstream: 5 steps × 512 prompts/step).
- `configs/base/eval.yaml`: `adapters: [bfcl_multi_turn]`, categories `[multi_turn_base]` (200 tasks), xoá block envscaler_heldout.
- **Xoá** `tests/test_dev_test_integrity.py` (premise đã bị đảo có chủ đích).
- Update tests: `test_bench_eval_callback.py` (bỏ 2 test guard, trajectory path đọc theo `runner.adapter_name`), `test_sft_assembly.py` (`eval_dataset is None`), `test_bfcl_oracle_replay.py` (200 base tasks, không còn 800), `test_console_stdout_contract.py` (emitter SFT cuối biến mất), `test_eval_admission_capacity.py` (xoá test premise "shipped heldout > pool capacity").
- `tests/test_adapter_protocol.py`, `test_grpo_args.py`, `test_envscaler_heldout_selection.py`, `test_adapter_diagnostics.py` giữ pass (module envscaler_heldout giữ nguyên, chỉ rời config).

### D. Scripts L4

- `scripts/colab-l4-smoke.py`: eval smoke → `bfcl_multi_turn` (giữ max_steps_per_task=1, max_new_tokens=32), step rename `eval-smoke`.
- `scripts/colab-gpu-validation.py`: `_small_eval_config` → bfcl; report metric key `bfcl_multi_turn`; GRPO smoke kế thừa adapter mới từ grpo.yaml (override `bench_eval.enabled=true, task_limit=1` giữ nguyên).

### E. Docs + plan

- `docs/evaluation.md`: section "Dev and test" viết lại — dev=test=BFCL base 200, SFT train-only, GRPO bench cadence; sizing note điều chỉnh.
- `docs/grpo.md`: bỏ heldout/`assert_dev_adapter`, bench_eval config mới, bỏ 10% budget wording.
- `plans/260901-1043-.../plan.md`: "Decision update — 2026-09-07" ghi rõ supersede goal 5 + dev/test criteria (theo pattern của 2026-09-05), không sửa history.

## Deferred (không làm, ghi rõ)

- Module `src/smolqwen/eval/adapters/envscaler_heldout.py` + tests của nó giữ nguyên — ngừng dùng ở config là đủ; xoá hẳn là scope riêng kiểu phase-8 inventory.
- Conversion vẫn ghi `val.jsonl` (val_fraction 0.02) — training không còn consume.
- `TrainingConfig.eval_steps` giữ (GRPO vẫn có field, eval_strategy="no" nên vô hại).

## Verification

- `uv run pytest -m "not gpu and not dataset"` (đúng selection của `make test-ci`): **exit 0**.
- `uv run pytest tests/ --ignore=tests/test_vllm_adapter_capability.py`: exit 0 (623+ tests).
- `uv run ruff check` + `ruff format --check`: sạch.
- `smolqwen {train-sft,train-grpo,evaluate,profile-difficulty} --dry-run`: resolve không cần torch/vllm.
- GPU (còn mở, thuộc phase-10 L4): build `OfflineEngine` không raise prefix-caching assert; `grpo/bench_wall_s` của boundary đầu để chốt `timeout_s`/`task_limit`; W&B chỉ còn train loss (SFT) và reward + `grpo/bench_*` (GRPO).
