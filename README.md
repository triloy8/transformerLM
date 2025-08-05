# Usage
Make sure to have [`uv`](https://github.com/astral-sh/uv) installed and just use 
```bash
  uv run example.py
```

# Components
* transformer_lm with from scratch implementations of Linear, Embedding, RMSNorm, SwiGLU, RoPE, SPDA, softmax, MHA, Decoder Only Transformer block, Transformer LM
* train_utils contains a from scratch implementation  of AdamW, cosine annealing LR scheduler, cross entropy loss, gradient clipping and checkpointing utils  
* train tokenizer for vocab/merges generation for bpe tokenization
* infer script w/ temp and top p samping for token generation
* train_transformer script with a train loop, weight/activation logging, train/loss logging, relies on wandb 
* make data functions to make a numpy array memmap from txt file
* wandb sweep yaml boilerplate

# Things to consider
* naive token merging step for training for now