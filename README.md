# Transformer Language Model from Scratch

## Motivation

The goal is just to explore making a basic Transformer Language model from scratch.

## Features

### Model Components

- **Transformer Language Model**: Custom PyTorch module with from-scratch implementations:
  - Linear Layers
  - Embedding Layers
  - RMSNorm
  - SwiGLU Activation
  - Rotary Position Embedding (RoPE)
  - Scale Dot Product Attention (SDPA)
  - Multi-Headed Attention (MHA)
  - Softmax Function
  - Decoder-Only Transformer Blocks

### Training Utilities

- Custom implementation of optimization and regularization methods:
  - AdamW optimizer
  - Cosine annealing learning rate scheduler
  - Cross-entropy loss function
  - Gradient clipping
  - Model checkpoint saving/loading utilities
  - Lightweight logging with selectable backends (console or Weights & Biases)

- Config-driven CLI entry points (see Usage):
  - `transformerlm-train` for training via a TOML config
  - `transformerlm-make-data` for building memmap datasets
  - `transformerlm-train-tokenizer` for BPE training
  - `transformerlm-infer` for text generation

### Tokenizer

- Train a Byte-Pair Encoding (BPE) tokenizer:
  - GPT-2 style byte-level pretokenization
  - Chunked corpus scanning with parallel pretoken frequency counting
  - Generates `vocab.json` and `merges.txt` with support for special tokens
  - Merge search is simple/naive and open to further optimization

### Data Preparation

- CLI to preprocess raw text data into memory-mapped numpy arrays (`.memmap`), ensuring efficient data loading during training

### Inference

- Sampling-based text generation with adjustable parameters:
  - Temperature sampling
  - Top-p (nucleus) sampling
  - Optional EOS handling and context windowing

### Training Script

- Comprehensive training loop:
  - Logs loss, learning rate, grad L2 norm
  - Optionally logs activation and weight norms
  - Checkpointing at a configurable iteration frequency
  - Select logging backend via config (console by default; optional Wandb)

### Wandb Sweep

- A starter Wandb sweep YAML is included and can be adapted to the CLI/config workflow

## Installation

Requires Python 3.11–3.12 and PyTorch. Using [`uv`](https://github.com/astral-sh/uv) is recommended.

- Quick run without installing the package:

```bash
# Print an example resolved config
uv run transformerlm-train --config configs/train.toml --print-config
```

- Or install the package locally (editable):

```bash
uv pip install -e .
```

## Usage Examples

- Train the tokenizer (BPE):

```bash
uv run transformerlm-train-tokenizer --config configs/train_tokenizer.toml
```

- Start model training:

```bash
uv run transformerlm-train --config configs/train.toml
```

- Generate text:

```bash
uv run transformerlm-infer --config configs/infer.toml
```

- Build memmap datasets from raw text:

```bash
uv run transformerlm-make-data --config configs/make_data.toml
```

- Inspect effective configuration without running:

```bash
uv run transformerlm-train --config configs/train.toml --print-config
```

## Benchmarking

- Benchmarks live under `benchmarking/` and are TOML‑driven, similar to the CLI tools.
- Use the sample configs in `configs/` and run the scripts directly.
- Results are logged with the `ConsoleLogger` to stdout; no files are written.

- Inference latency:
  - Run: `python -m benchmarking.bench_infer_latency --config configs/bench_infer.toml`
  - Measures warmup and repeated decode steps (tokens/sec, latency).

- Tokenizer throughput:
  - Run: `python -m benchmarking.bench_tokenizer --config configs/bench_tokenizer.toml`
  - Measures encode and decode throughput over given texts.

## Tests

- Run tests: `uv run pytest`
- Markers:
  - `slow`: long-running tests. Deselect with `-m "not slow"`.
  - `gpu`: requires CUDA/GPU. Deselect with `-m "not gpu"`.
- Examples:
  - Quick CPU suite: `uv run pytest -m "not slow and not gpu"`
  - Select a file/test: `uv run pytest tests/tokenizer/test_tokenizer.py -q`
  - Filter by name: `uv run pytest -k tokenizer`

## Logging

- Backends:
  - `console` (default): prints structured JSON lines with metrics like `metrics.loss`, `metrics.lr`, `metrics.grad_l2_norm`, plus optional activation/weight norms.
  - `wandb`: logs to Weights & Biases and uploads artifacts (checkpoints, tokenizer files, optional inference outputs).
- Configure in `configs/train.toml` under `[logging]`:

```toml
[logging]
backend = "console"   # or "wandb"
run_name = ""         # optional; defaults to timestamp
architecture = "TransformerLM"
dataset = "TinyStoriesV2-GPT4"

# Optional if using Wandb
[wandb]
entity = "your-entity"
project = "your-project"
```

- Inference logs include sampling params and truncated text:
  - Keys: `params.temperature`, `params.p`, `params.eos_token_id`, `text.prompt`, `text.output`, `metrics.latency_ms`.
- Tip (console backend): Pipe to `jq` for readability:
  - `uv run transformerlm-train --config configs/train.toml | jq -r "."`

## Considerations

- Tokenizer training uses a simple/naive merge search; the pretokenization and frequency counting are chunked/parallelized, but the merge step itself can be further optimized.
