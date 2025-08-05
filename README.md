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
  - Scale Dot Product Attention (SPDA)
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

### Tokenizer

- Script to train a Byte-Pair Encoding (BPE) tokenizer:
  - Generates vocabulary and merges for tokenization
  - Naive token merging strategy currently used (open to enhancements)

### Data Preparation

- Scripts to preprocess raw text data into memory-mapped numpy arrays (`.memmap`), ensuring efficient data loading during training

### Inference

- Sampling-based text generation script with adjustable parameters:
  - Temperature sampling
  - Top-p (nucleus) sampling

### Training Script

- Comprehensive training loop script:
  - Integrated logging for weights, activations, and loss metrics
  - Utilizes [Weights & Biases (wandb)](https://wandb.ai/) for experiment tracking

### Wandb Sweep

- Ready-to-use Wandb sweep YAML configuration for hyperparameter optimization experiments

## Installation

Ensure you have [`uv`](https://github.com/astral-sh/uv) installed, then run scripts directly:

```bash
source scripts/*.sh
```

## Usage Examples

- Train the tokenizer:

```bash
source scripts/train_tokenizer.sh
```

- Start model training:

```bash
source scripts/train_transformer.sh
```

- Generate text:

```bash
source scripts/infer_transformer.sh
```

## Considerations

- Currently, tokenizer training employs a naive merging strategy.

##

