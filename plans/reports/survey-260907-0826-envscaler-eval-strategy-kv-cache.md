# Survey: Eval strategy đổi mới (SFT eval_loss, dev=test=BFCL base, bỏ env heldout) + bug vLLM KV cache

Date: 2026-09-07. Scope: khảo sát trước khi implement, không sửa code.

## TL;DR

1. **EnvScaler upstream KHÔNG có dev set riêng cho SFT** — SFT train full 9K trajectories, 3 epochs, không validation. Bộ dev duy nhất upstream dùng là **BFCL multi_turn_base trong RL validation** (mỗi 5 steps, 128 samples, temp 0, group_size=1).
2. SFT trong smolqwen **đã có sẵn eval_loss** trên `val.jsonl` (184 rows, 2% split) — chỉ cần không bật `bench_eval`, không cần viết gì thêm.
3. BFCL v4 multi_turn_base = **200 samples** (1–7 turns, avg 3.67). EnvScaler vendored data có id set giống hệt 200 samples này.
4. Bug vLLM KV cache **confirmed ở mức code**: `OfflineEngine` (eval + in-training bench path) không truyền `enable_prefix_caching=True`; với Qwen3.5 hybrid, vLLM 0.26 resolve flag này thành **false** (chính codebase đã ghi nhận hiện tượng này ở `_force_trl_prefix_caching` trong `grpo.py:554-580`). Mỗi turn = 1 request mới chứa toàn bộ conversation → prefill lại từ đầu. GRPO colocated đã patch bật; eval path thì không.

---

## 1. Upstream EnvScaler train setup (RUC-NLPIR/EnvScaler @ 87e6673)

Vendored submodule khớp upstream HEAD trừ `README.md` (verified bằng `gh api compare/87e6673...main` → only `README.md` changed). Upstream commit mới nhất 96ae8b0 chỉ sửa README.

### SFT (sft/README.md + qwen3_full_sft.yaml)

| Item | Upstream |
|---|---|
| Framework | LlamaFactory |
| Model | Qwen3-4B (example; released: 1.7B/4B/8B) |
| Dataset | EnvScaler-SFT-Traj-9K (9K trajectories), `mask_history: True` |
| Val/dev set | **KHÔNG có.** Không `val_dataset`, không `eval_steps`, chỉ `plot_loss: true` |
| Epochs | 3 (smolqwen: 2) |
| cutoff_len | 32768 (smolqwen cùng) |
| lr | 1e-6 full-FT (smolqwen 1e-4 LoRA) |

Step 2 của họ chia n-turn sample thành n sub-samples, chỉ supervise turn cuối (`mask_history`). smolqwen supervise tất cả assistant turns trong một trajectory — khác biệt đã biết và có chủ đích (shard format riêng).

### RL (rl/example/env_scaler/only_non_conv_qwen3_8gpu.yaml + ROLL framework)

| Item | Upstream |
|---|---|
| Train data | Toàn bộ `envscaler_rl_scenario_metadata.json` (2550 scenarios, 51 RL envs × 50) qua `SynEnvNonConversationTrain` |
| Heldout EnvScaler scenarios | **KHÔNG có** — không có khái niệm heldout RL scenario trong pipeline upstream |
| RL validation | **BFCL multi_turn_base**: `val_env_manager` tags `[BFCLEval]`, `env_type: bfcl`, `mode: multi_turn_base`, `num_env_groups: 128`, `group_size: 1` (temp 0 → greedy), `eval_steps: 5`, `val_batch_size: 128` |
| In-train reward | Verifier reward trên train scenarios (đúng như smolqwen) |
| max_steps / save / eval | 201 / 50 / 5 |

Kết luận so với yêu cầu mới của user: **hướng đi mới của user (RL train → theo dõi reward train; mỗi N steps eval BFCL base) khớp CHÍNH XÁC thiết kế upstream.** smolqwen's `envscaler_heldout` (10 env × 8 scenarios) là khái niệm tự chế trong plan cũ, upstream không có.

### DataReality check

- `191_env_metadata.json`: 140 `_sft` envs + 51 `_rl` envs. RL scenarios chỉ phủ 51 rl envs.
- Không env RL nào trong bộ released là conversational (checked tool schemas — không `chat_with_user` user-simulator tool; `env_146_rl` có `send_message` nhưng là tool messaging trong env, không phải user agent). Vậy upstream config "only_non_conv" phủ toàn bộ 2550 scenarios = smolqwen không bỏ lỡ env nào.
- BFCL: upstream eval guide dùng BFCL-v3 (thinking mode, temp 0.7, FC mode, 64K context); vendored `bfcl_env/data/data_multi_turn_base.json` = 200 entries, **id set identical** với smolqwen's `BFCL_v4_multi_turn_base.json` (gorilla submodule @ 6ea5797). Data tương đương; decoding khác (smolqwen: greedy, non-thinking — đã ghi trong eval.yaml invariant).

## 2. Hiện trạng smolqwen

### SFT (src/smolqwen/training/sft.py)

- `eval_loss` **đã có sẵn**: `_sft_config` đặt `eval_strategy="steps"`, `eval_steps: 100`; `build_trainer` truyền `eval_dataset=shards.eval`; `TokenBudgetSFTTrainer.get_eval_dataloader` dùng cùng token-budget sampler (không vLLM, teacher-forced). `run_train_sft` gọi `trainer.evaluate()` cuối run và in JSON metrics.
- Logging: SFT chạy `report_to=[]` trong Trainer, nhưng `ThroughputCallback.on_log` forward toàn bộ logs (bao gồm eval metrics) vào `Tracker` → W&B. Cần 1 lần verify trên card rằng `eval_loss` thực sự xuất hiện trong W&B (on_log cũng fire sau evaluate — kỳ vọng có).
- `val.jsonl`: 184 rows, đến từ **64 envs trong số 140 sft envs — 100% trùng với train envs** (split theo trajectory, không theo env). Cho mục đích overfit-check trên cùng distribution: hợp lệ. Ghi nhận để không nhầm với "held-out env".
- `bench_eval` (vLLM envscaler_heldout trong SFT) hiện `enabled: false` trong `sft.yaml:69-75`. Hướng mới = bỏ hẳn block; toàn bộ machinery `CheckpointEngine` / `build_bench_eval_callback` trong SFT path trở thành unused nếu không caller nào khác (GRPO dùng `build_bench_eval_callback` riêng qua `grpo.py:436`).

### GRPO (src/smolqwen/training/grpo.py)

- `select_heldout_scenarios` (10 env × 8 = 80 scenarios) → `eval_dataset` cho TRL (nhưng `eval_strategy="no"` nên không tự chạy) + `bench_eval.adapter: envscaler_heldout`.
- Train reward đã log qua TRL → W&B (`report_to=["wandb"]`).
- Muốn đổi: bỏ heldout selection, bench_eval đổi adapter → BFCL base với `every_steps: N`.
- **Blocker:** `assert_dev_adapter` (`bench_eval.py:132-147`) từ chối mọi adapter name chứa `bfcl|gorilla|berkeley` tại construction. Guard này bảo vệ success criterion cũ của plan ("BFCL là test, không in-training"). Yêu cầu mới của user đảo contract: dev = test = BFCL base. Cần bỏ/thay guard + cập nhật `test_dev_test_integrity.py`, `test_bench_eval_callback.py`, success criteria trong `plan.md`, docs (evaluation.md, grpo.md, sft.md).
- `GrpoAssembly.eval_task_ids` và curriculum heldout_env_count/heldout_scenarios_per_env cần dọn khi bỏ heldout.
- Lưu ý cost: BFCL greedy trên colocated engine — upstream dùng 128 samples mỗi 5 steps trên 8 GPU. Trên 1×L4, 200 tasks × ~4 turns × gen là nặng; `task_limit` (prefix of deterministic order) vẫn là knob phù hợp, nhưng prefix 200-task list là order theo category → prefix chỉ phủ multi_turn_base nếu adapter chỉ load category đó. → đổi `eval.yaml`-style options của bench_eval thành categories: [multi_turn_base].

### Đếm sample

- `BFCL_v4_multi_turn_base.json`: **200 tasks**, 1–7 turns (avg 3.67). possible_answer 200 khớp.
- 3 categories còn lại mỗi cái 200 (tổng 800 nếu chạy đủ 4 — nhưng hướng mới chỉ cần base).
- EnvScaler SFT scenarios: 4684; RL scenarios: 2550.

## 3. Bug vLLM KV cache — phân tích

Triệu chứng: multi-turn eval chậm bất thường, mỗi turn prefill lại toàn bộ conversation.

Call path: `evaluate_batched` → `TurnEngine._generate_ready` → `OfflineEngineBackend.generate` → `OfflineEngine.generate_ids` → `vllm.LLM.generate({prompt_token_ids: [toàn bộ conversation]})`.

- Turn engine **không có khái niệm session** — mỗi turn là 1 request độc lập chứa prompt_token_ids đầy đủ (đúng thiết kế; không cần session API nếu prefix caching hoạt động).
- Prefix caching là cơ chế vLLM match các KV block chung giữa requests: system prompt + tool schemas + lịch sử đầu của turn N trùng prefix của turn N-1 → chỉ prefill phần tăng thêm.
- `OfflineEngine.build` (`inference/engine.py:243-261`) **không truyền `enable_prefix_caching`**. Theo comment thực nghiệm trong `grpo.py:554-580`: vLLM 0.26 resolve `enable_prefix_caching=false` cho hybrid models như Qwen3.5 khi không truyền flag (GDN hybrid allocator). GRPO colocated đã phải patch `_force_trl_prefix_caching()` để buộc bật; **eval path không có patch tương đương** → prefix caching tắt → mỗi turn prefill full conversation. Với BFCL base (avg 3.67 turns, tool schemas ~4K+ tokens) và EnvScaler heldout, chi phí prefill nhân theo mỗi turn — khớp triệu chứng.
- GRPO bench theo checkpoint (CheckpointEngine) và `evaluate` CLI đều dùng `OfflineEngine` → cùng bệnh.

**Fix đề xuất (root-cause, 1 dòng):** `enable_prefix_caching=True` trong `LLM(...)` của `OfflineEngine.build`, cùng guard-assert kiểu `_assert_prefix_caching` khi build (fail-closed như GRPO đã làm). Không cần session/ KV-preserve API. Verify trên card: đo prefill tokens per turn trước/sau (vLLM metrics hoặc profile đã có `stage_intervals`).

## 4. Việc cần làm khi implement (file-level)

| # | Việc | File |
|---|---|---|
| 1 | Bật prefix caching trong OfflineEngine + assert | `src/smolqwen/inference/engine.py` |
| 2 | SFT: xoá block `bench_eval` khỏi config; giữ eval_loss | `configs/base/sft.yaml` |
| 3 | Bỏ heldout khỏi GRPO: xoá `select_heldout_scenarios` call, curriculum heldout fields, `eval_task_ids` | `src/smolqwen/training/grpo.py`, `configs/base/grpo.yaml`, `src/smolqwen/config_models.py` |
| 4 | GRPO bench_eval → bfcl_multi_turn_base, categories=[multi_turn_base], every_steps: N, task_limit ≤ 200 | `configs/base/grpo.yaml` + BenchEvalConfig |
| 5 | Gỡ/đổi `assert_dev_adapter` (dev=test giờ là chủ đích) | `src/smolqwen/training/bench_eval.py` |
| 6 | `evaluate` chỉ chạy bfcl base; envscaler_heldout adapter rời khỏi config (code giữ lại hoặc xoá theo inventory) | `configs/base/eval.yaml` |
| 7 | Cập nhật tests: `test_dev_test_integrity.py`, `test_bench_eval_callback.py`, `test_sft_bench_eval_boundary.py` | `tests/` |
| 8 | Docs: evaluation.md, grpo.md, sft.md, plan.md success criteria | `docs/`, `plans/.../plan.md` |
| 9 | Verify eval_loss & bench metrics thực sự xuất hiện trên W&B (GPU gate) | L4 session |

## Unresolved questions

1. `task_limit` cho BFCL base trong RL: chạy đủ 200 hay prefix nhỏ hơn (vd 50) để giữ 10% wall-time budget? Upstream dùng 128/8GPU mỗi 5 steps — trên 1×L4 nên bắt đầu nhỏ, đo `bench_wall_s` rồi tăng.
2. N (every_steps) cho RL dev: upstream 5; grpo.yaml hiện save_steps 20 — đề xuất every_steps 20 (trùng save boundary) hoặc 40.
3. Có xoá hẳn `envscaler_heldout` adapter + `CheckpointEngine` (SFT path) không, hay giữ code chỉ rời config? (Phase 8 inventory của plan cũ cần chạy lại nếu xoá.)
