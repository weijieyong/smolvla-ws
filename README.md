# SmolVLA Quickstart (Internal)

SmolVLA is easy to use for fine-tuning or integration into robotics workflows.

## Prerequisites
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/):
  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- Make sure you have a working GPU and CUDA drivers (tested: Ubuntu 24.04, RTX 5080, CUDA 12.8).
- Compiled and build FFmpeg 7.1.1 from source ([guide](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#FFmpeg))

## Clone and Install
Clone the repo and install SmolVLA dependencies:
```sh
git clone https://github.com/huggingface/lerobot.git
cd lerobot
uv pip install -e ".[smolvla]"
```

## (Optional) For RTX 5080 GPUs
Use nightly builds for torch/torchcodec for compatibility:
```sh
uv pip uninstall torch torchvision torchcodec
uv pip install --pre torch torchvision torchcodec --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Authenticate for Model/Dataset Access
Login to HuggingFace and Weights & Biases:
```sh
hf auth login
wandb login
```

## (Optional) Suppress Tokenizer Warnings
```sh
export TOKENIZERS_PARALLELISM=false
```

## Fine-tune Example
Run fine-tuning on a base model with a HuggingFace dataset:
```sh
uv run src/lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=aractingi/il_gym0 \
  --batch_size=16 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=false
```

## Data Recording & Evaluation
- (TODO) Add instructions for recording data and running evaluation.

## Resources
- https://huggingface.co/blog/smolvla
- https://huggingface.co/docs/lerobot/en/smolvla
- https://huggingface.co/papers/2506.01844
- https://github.com/huggingface/lerobot