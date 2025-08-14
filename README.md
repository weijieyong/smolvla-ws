# SmolVLA Quickstart (Internal)

SmolVLA is easy to use for fine-tuning or integration into robotics workflows.

## Prerequisites
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/):
  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- Make sure you have a working GPU and CUDA drivers (tested: Ubuntu 24.04, RTX 5080, CUDA 12.8).
- Compiled and build FFmpeg 7.1.1 from source ([guide](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#FFmpeg))

> [!NOTE] Tested System Configuration
> - **OS:** Ubuntu 24.04.3 LTS (x86_64)
> - **Kernel:** 6.16.0-061600-generic
> - **GPU:** NVIDIA GeForce RTX 5080
> - **NVIDIA Driver:** 580.65.06
> - **CUDA:** 12.8
> - **Python:** 3.10
> - **uv:** 0.8.4
> - **FFmpeg:** 7.1.1

## Clone and Install
Clone the repo and install SmolVLA dependencies:
```sh
git clone https://github.com/weijieyong/lerobot.git
cd lerobot
uv venv --python 3.10
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
Adjust `batch_size` value based on your GPU's VRAM. to prevent OOM

```sh
uv run src/lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=aractingi/il_gym0 \
  --batch_size=48 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=false
```

more about the training script [here](https://github.com/huggingface/lerobot/blob/main/examples/4_train_policy_with_script.md)

## Data Recording
- https://huggingface.co/docs/lerobot/getting_started_real_world_robot?teleoperate_so101=Command#record-a-dataset
- 
## Visualizing dataset
- visualizing with rerun, on the lerobot/pusht dataset

```sh
uv run src/lerobot/scripts/visualize_dataset.py \
  --repo-id lerobot/aloha_static_coffee_new \
  --episode-index 0
```

## Eval



## Resources
- https://huggingface.co/blog/smolvla
- https://huggingface.co/docs/lerobot/en/smolvla
- https://huggingface.co/papers/2506.01844
- https://github.com/huggingface/lerobot
- https://deepwiki.com/huggingface/lerobot
