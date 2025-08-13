#!/usr/bin/env bash

set -euo pipefail
echo "[INFO] Strict mode (set -euo pipefail) enabled."

# Determine the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd)"
echo "[INFO] Script directory: $SCRIPT_DIR"

VENV_PATH="$SCRIPT_DIR/lerobot/.venv/bin/activate"
if [ ! -f "$VENV_PATH" ]; then
	echo "Virtual environment not found at $VENV_PATH. Please create it first."
	exit 1
fi
source "$VENV_PATH"
echo "[INFO] Virtual environment activated at: $VENV_PATH"


export TOKENIZERS_PARALLELISM=false
echo "[INFO] TOKENIZERS_PARALLELISM set to false. (to suppress warnings)"

echo "[INFO] Starting training script..."
uv run "$SCRIPT_DIR/lerobot/src/lerobot/scripts/train.py" \
	--policy.path=lerobot/smolvla_base \
	--dataset.repo_id=aractingi/il_gym0 \
	--batch_size=32 \
	--steps=20000 \
	--output_dir=outputs/train/my_smolvla \
	--job_name=my_smolvla_training \
	--policy.device=cuda \
	--wandb.enable=true \
	--policy.push_to_hub=false
echo "[INFO] Training script finished."