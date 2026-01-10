#!/usr/bin/env python3
"""
Upload nanochat model to HuggingFace Hub
Usage: python upload_to_hf.py --repo-id your-username/nanochat-d20 --checkpoint sft
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

def create_model_card(checkpoint_type, config_path):
    """Create a README.md model card for HuggingFace"""

    # Load meta.json to get model config
    with open(config_path, 'r') as f:
        meta = json.load(f)

    # Extract config
    model_config = meta.get('model_config', {})
    n_layer = model_config.get('n_layer', 20)
    vocab_size = model_config.get('vocab_size', 65536)
    n_embd = model_config.get('n_embd', 1280)

    # Calculate approximate parameters
    # Rough estimate: embedding + transformer layers + output
    num_params = vocab_size * n_embd * 2 + n_layer * (n_embd ** 2) * 12
    params_str = f"~{num_params/1e6:.1f}M"

    model_card = f"""---
language:
- en
license: mit
tags:
- text-generation
- pytorch
- nanochat
- gpt
datasets:
- karpathy/fineweb-edu-100b-shuffle
---

# nanochat-d{n_layer} ({checkpoint_type.upper()})

This is a nanochat model trained using the [nanochat](https://github.com/karpathy/nanochat) framework.

> The best ChatGPT that $100 can buy.

## Model Description

- **Model Type**: GPT-style Transformer
- **Architecture**: {n_layer} layers
- **Parameters**: {params_str}
- **Training Stage**: {checkpoint_type.upper()}
- **Vocab Size**: {vocab_size}

## Training Details

This model was trained as part of the nanochat project, which implements a full-stack LLM training pipeline including:
- BPE tokenizer training (vocab size: 65536)
- Base model pretraining on FineWeb-Edu
- {"Midtraining with conversation format" if checkpoint_type in ['mid', 'sft'] else ""}
- {"Supervised fine-tuning (SFT)" if checkpoint_type == 'sft' else ""}

## Usage

```python
import torch
from nanochat.gpt import GPT
from nanochat.tokenizer import Tokenizer
from nanochat.engine import Engine

# Load tokenizer
tokenizer = Tokenizer()
tokenizer.load("tokenizer.pkl")

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT.from_checkpoint("model.pt", device=device)

# Create inference engine
engine = Engine(model, tokenizer)

# Generate text
prompt = "Hello, how are you?"
response = engine.generate(prompt, max_tokens=100)
print(response)
```

## Files

- `model.pt`: PyTorch model checkpoint
- `meta.json`: Model configuration and metadata
- `tokenizer.pkl`: BPE tokenizer
- `token_bytes.pt`: Token byte mappings for the tokenizer

## Citation

```bibtex
@misc{{nanochat,
  author = {{Andrej Karpathy}},
  title = {{nanochat: The best ChatGPT that $100 can buy}},
  year = {{2025}},
  publisher = {{GitHub}},
  url = {{https://github.com/karpathy/nanochat}}
}}
```

## License

MIT License
"""
    return model_card


def prepare_upload_dir(checkpoint_type, base_dir="/workspace/nanochat_data"):
    """Prepare a directory with all files needed for upload"""

    base_dir = Path(base_dir)
    upload_dir = Path(f"/tmp/nanochat_upload_{checkpoint_type}")

    # Clean and create upload directory
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True)

    # Determine checkpoint directory
    checkpoint_dirs = {
        'base': base_dir / 'base_checkpoints' / 'd20',
        'mid': base_dir / 'mid_checkpoints' / 'd20',
        'sft': base_dir / 'chatsft_checkpoints' / 'd20'
    }

    if checkpoint_type not in checkpoint_dirs:
        raise ValueError(f"Invalid checkpoint type. Choose from: {list(checkpoint_dirs.keys())}")

    checkpoint_dir = checkpoint_dirs[checkpoint_type]

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Find the checkpoint files
    model_files = list(checkpoint_dir.glob("model_*.pt"))
    meta_files = list(checkpoint_dir.glob("meta_*.json"))

    if not model_files:
        raise FileNotFoundError(f"No model checkpoint found in {checkpoint_dir}")

    # Use the latest checkpoint
    model_file = sorted(model_files)[-1]
    meta_file = sorted(meta_files)[-1] if meta_files else None

    print(f"Found model checkpoint: {model_file}")
    print(f"Found meta file: {meta_file}")

    # Copy model files
    shutil.copy(model_file, upload_dir / "model.pt")
    if meta_file:
        shutil.copy(meta_file, upload_dir / "meta.json")

    # Copy tokenizer files
    tokenizer_dir = base_dir / 'tokenizer'
    if tokenizer_dir.exists():
        for tok_file in ['tokenizer.pkl', 'token_bytes.pt']:
            src = tokenizer_dir / tok_file
            if src.exists():
                shutil.copy(src, upload_dir / tok_file)
                print(f"Copied tokenizer file: {tok_file}")

    # Create model card
    if meta_file:
        model_card = create_model_card(checkpoint_type, upload_dir / "meta.json")
    else:
        # Create a basic model card without meta
        model_card = create_model_card(checkpoint_type, None)

    with open(upload_dir / "README.md", 'w') as f:
        f.write(model_card)

    print(f"\nPrepared upload directory: {upload_dir}")
    print(f"Files to upload:")
    for file in upload_dir.iterdir():
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name} ({size_mb:.2f} MB)")

    return upload_dir


def upload_to_huggingface(repo_id, upload_dir, private=False):
    """Upload the prepared directory to HuggingFace Hub"""

    api = HfApi()

    # Create repository if it doesn't exist
    try:
        print(f"\nCreating repository: {repo_id}")
        create_repo(repo_id, private=private, exist_ok=True)
        print(f"Repository created/verified: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Note: {e}")

    # Upload folder
    print(f"\nUploading files to {repo_id}...")
    api.upload_folder(
        folder_path=str(upload_dir),
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"\n✅ Upload complete!")
    print(f"🔗 View your model at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload nanochat model to HuggingFace")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., 'username/nanochat-d20-sft')"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        choices=['base', 'mid', 'sft'],
        default='sft',
        help="Which checkpoint to upload (default: sft)"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/workspace/nanochat_data",
        help="Base directory containing the checkpoints"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("nanochat → HuggingFace Upload Tool")
    print("=" * 60)

    # Step 1: Prepare upload directory
    print("\n[1/2] Preparing files for upload...")
    upload_dir = prepare_upload_dir(args.checkpoint, args.base_dir)

    # Step 2: Upload to HuggingFace
    print("\n[2/2] Uploading to HuggingFace Hub...")
    upload_to_huggingface(args.repo_id, upload_dir, args.private)

    print("\n" + "=" * 60)
    print("Done! Your model is now available on HuggingFace.")
    print("=" * 60)


if __name__ == "__main__":
    main()
