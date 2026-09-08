# ============================================================
# Cell 1 — Clone repo, install stack, verify kernels (~5-8 min)
# ============================================================
%%bash
git clone https://github.com/qninhdt/smolqwen.git /content/smolqwen 2>/dev/null || (cd /content/smolqwen && git pull)
cd /content/smolqwen
bash scripts/setup_colab.sh

# ============================================================
# Cell 2 — Secrets + config override (edit before running)
# ============================================================
%env HF_TOKEN=hf_xxx                  # để tải Qwen/Qwen3.5-2B
%env WANDB_API_KEY=xxx                # logging loss/throughput/VRAM (đã fix hôm nay)
# checkpoint push lên Hub (bắt buộc nếu muốn --resume sau khi VM reclaim):
%env SFT_HUB_REPO=qninhdt/qwen3.5-2b-sft-adapter    # tạo repo trước trên HF

# ============================================================
# Cell 3 — Dry-run kiểm tra config (không đụng GPU)
# ============================================================
%%bash
cd /content/smolqwen
export HF_TOKEN WANDB_API_KEY
uv run --no-sync smolqwen train-sft --profile l4 \
  --override tracking.hub_repo_id="$SFT_HUB_REPO" \
  --dry-run

# ============================================================
# Cell 4 — TRAIN (chạy 2 epoch, checkpoint tự lưu mỗi epoch + push Hub)
# ============================================================
%%bash
cd /content/smolqwen
export HF_TOKEN WANDB_API_KEY
uv run --no-sync smolqwen train-sft --profile l4 \
  --override tracking.hub_repo_id="$SFT_HUB_REPO" \
  2>&1 | tee sft_train.log

# ============================================================
# Cell 5 — EVAL checkpoint mỗi epoch (chạy sau train hoặc sau khi Ctrl+C)
# ============================================================
%%bash
cd /content/smolqwen
export HF_TOKEN
uv run --no-sync smolqwen evaluate --profile l4 \
  --checkpoint artifacts/models/qwen3.5-2b-sft/checkpoint-<EPOCH1_STEP> \
  --tag sft-epoch1
# epoch 2:
# uv run --no-sync smolqwen evaluate --profile l4 \
#   --checkpoint .../checkpoint-<EPOCH2_STEP> --tag sft-epoch2

# ============================================================
# Cell 6 — RESUME nếu train bị ngắt giữa chừng (VM reclaim / Ctrl+C)
# ============================================================
%%bash
cd /content/smolqwen
export HF_TOKEN WANDB_API_KEY
uv run --no-sync smolqwen train-sft --profile l4 \
  --override tracking.hub_repo_id="$SFT_HUB_REPO" \
  --resume
