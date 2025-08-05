from tokenizer import Tokenizer
from transformer_lm import (TransformerLM,
                            Linear)
from train_transformer_utils import (AdamW,
                                     get_batch,
                                     cross_entropy,
                                     save_checkpoint,
                                     lr_cosine_schedule,
                                     gradient_clipping,)

from pathlib import Path
import numpy as np
import torch
import wandb
import datetime
import os

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# wandb
run = wandb.init(
    entity="yiltro8-org",
    project="transformer_lm",
    name=f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{wandb.util.generate_id()}",
    config={
        "architecture": "Transformer LM",
        "dataset": "TinyStoriesV2-GPT4",
        "vocab_size": 50257,
        "context_length": 256,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 16,
        "d_ff": 1344,
        "rope_theta": 10000.0,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.001,
        "grad_clip_max_l2_norm": 1.0,
        "max_learning_rate": 1e-3,
        "min_learning_rate": 1e-4,
        "warmup_iters": 50,
        "cosine_cycle_iters": 4500,
        "max_train_iteration": 5000,
        "max_val_iteration": 10,
        "val_freq_iteration": 125,
        "batch_size": 32,
        "device": "cuda",
        "dtype": "float32",
        "ckpting_save_iter": 1000,
    },
)
cfg  = run.config

# data
DATA_PATH = Path("./transformerLM/data")
RUNS_PATH = Path("./transformerLM/runs")
VOCAB_PATH = DATA_PATH / "gpt2_vocab.json"
MERGES_PATH = DATA_PATH / "gpt2_merges.txt"
NP_DAT_TRAIN_PATH = DATA_PATH / "TinyStoriesV2-GPT4-train.dat"
total_train_tokens = 547994686
NP_DAT_VALID_PATH = DATA_PATH / "TinyStoriesV2-GPT4-valid.dat"
total_val_tokens = 5535291

ckpting_save_folder = RUNS_PATH / run.name
if not os.path.exists(ckpting_save_folder):
    os.makedirs(ckpting_save_folder)

# init tokenizer, model, optim and data
tokenizer = Tokenizer.from_files(vocab_filepath=VOCAB_PATH, 
                                 merges_filepath=MERGES_PATH, 
                                 special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"])

model = TransformerLM(vocab_size=cfg.vocab_size, 
                      context_length=cfg.context_length, 
                      d_model=cfg.d_model, 
                      num_layers=cfg.num_layers, 
                      num_heads=cfg.num_heads, 
                      d_ff=cfg.d_ff, 
                      rope_theta=cfg.rope_theta,
                      device=cfg.device, 
                      dtype=DTYPES[cfg.dtype])

optimizer = AdamW(model.parameters(),
                  lr=0.003,
                  betas=cfg.betas,
                  eps=float(cfg.eps),
                  weight_decay=cfg.weight_decay)

np_arr_train_data = np.memmap(NP_DAT_TRAIN_PATH,
                              dtype=np.int32,
                              mode='r',
                              shape=(total_train_tokens,))

np_arr_valid_data = np.memmap(NP_DAT_VALID_PATH,
                              dtype=np.int32,
                              mode='r',
                              shape=(total_val_tokens,))

# weight/activation norm utils
def get_weight_norms(model):
    norms = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            norms[name] = param.data.norm().item()
    return norms
activation_norms = {}
def get_activation_norm_hook(name):
    def hook(module, input, output):
        activation_norms[name] = output.norm().item()
    return hook
for name, module in model.named_modules():
    if isinstance(module, Linear):
        module.register_forward_hook(get_activation_norm_hook(name))

train_iteration = 0
while True:
    model.train()
    train_batch_sampled_sequence, train_batch_sampled_ids = get_batch(dataset=np_arr_train_data, 
                                                                      batch_size=cfg.batch_size,
                                                                      context_length=cfg.context_length,
                                                                      device="cuda")

    # model call
    train_logits = model(train_batch_sampled_sequence)

    # log activation norms
    wandb.log({
        "activation_norms/mean": np.mean(list(activation_norms.values())),
        "activation_norms/max": np.max(list(activation_norms.values())),
        "activation_norms/min": np.min(list(activation_norms.values())),
        **{f"activation_norms/{k}": v for k, v in activation_norms.items()}
    }, step=train_iteration)

    # update params
    optimizer.zero_grad() # reset grads
    train_loss = cross_entropy(train_logits, train_batch_sampled_ids).mean() # compute loss

    wandb.log({'train_loss': train_loss.item()}, step=train_iteration)

    print(f"train_loss: {train_loss.item()}")

    train_loss.backward() # compute grads
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    l2_norm = torch.norm(
        torch.stack([g.detach().norm(2) for g in grads])
    )
    wandb.log({'grads': l2_norm.item()}, step=train_iteration)
    gradient_clipping(parameters=model.parameters(), max_l2_norm=cfg.grad_clip_max_l2_norm) # grad clip
    optimizer.step() # update params

    # log weight norms
    weight_norms = get_weight_norms(model)
    wandb.log({
        "weight_norms/mean": np.mean(list(weight_norms.values())),
        "weight_norms/max": np.max(list(weight_norms.values())),
        "weight_norms/min": np.min(list(weight_norms.values())),
        **{f"weight_norms/{k}": v for k, v in weight_norms.items()}
    }, step=train_iteration)

    # manually update the optimizer with the new LR
    new_lr = lr_cosine_schedule(train_iteration, cfg.max_learning_rate, cfg.min_learning_rate, cfg.warmup_iters, cfg.cosine_cycle_iters)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    wandb.log({'lr': new_lr}, step=train_iteration)

    # do validation
    if train_iteration % cfg.val_freq_iteration == 0:
        model.eval()
        val_iteration = 0
        running_val_loss = 0.0
        while True:
            with torch.no_grad():
                val_batch_sampled_sequence, val_batch_sampled_ids = get_batch(dataset=np_arr_valid_data, 
                                                                            batch_size=cfg.batch_size,
                                                                            context_length=cfg.context_length,
                                                                            device="cuda")
                val_logits = model(val_batch_sampled_sequence)
                val_loss = cross_entropy(val_logits, val_batch_sampled_ids).mean()
                running_val_loss += val_loss.item()
            val_iteration += 1    
            if cfg.max_val_iteration is not None and val_iteration >= cfg.max_val_iteration:
                break
        avg_val_loss = running_val_loss / cfg.max_val_iteration
        print(f"val_loss: {avg_val_loss}")
        wandb.log({'val_loss': avg_val_loss}, step=train_iteration)

    # save checkpoint
    if train_iteration > 0 and train_iteration % cfg.ckpting_save_iter == 0:
        ckpt_file_iter = ckpting_save_folder / f"{train_iteration}.ckpt"
        save_checkpoint(model, optimizer, train_iteration, ckpt_file_iter)

    # condition to stop
    if cfg.max_train_iteration is not None and train_iteration >= cfg.max_train_iteration:
        break

    # update iteration
    train_iteration += 1

wandb.finish()