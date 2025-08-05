from transformerlm.tokenizer.tokenizer import Tokenizer
from transformerlm.transformer.transformer_lm import (TransformerLM,
                                                      Linear)
from transformerlm.transformer.train_transformer_utils import (AdamW,
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
import argparse

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

def get_args():
    parser = argparse.ArgumentParser(
        description="Training script hyperparameters and options."
    )

    # ===== OPTIMIZER =====
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.95), help='Adam betas (beta1, beta2)')
    parser.add_argument('--eps', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--max_learning_rate', type=float, default=3e-4, help='Maximum learning rate')
    parser.add_argument('--min_learning_rate', type=float, default=3e-5, help='Minimum learning rate (default: 0.1 * max_learning_rate)')
    parser.add_argument('--warmup_iters', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--cosine_cycle_iters', type=int, default=90, help='Number of iterations for cosine annealing')
    parser.add_argument('--grad_clip_max_l2_norm', type=float, default=1.0, help='Maximum L2-norm for gradient clipping')

    # ===== MODEL =====
    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size')
    parser.add_argument('--context_length', type=int, default=1024, help='Max sequence/context length')
    parser.add_argument('--d_model', type=int, default=768, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=3072, help='Feedforward hidden dimension')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='Rotary embedding theta')

    # ===== GLOBAL =====
    parser.add_argument('--device', type=str, default='cuda', help='Device to train on (e.g., cuda, cpu)')
    parser.add_argument('--dtype', type=str, default='float32', help='Tensor dtype (e.g., float32, bfloat16)')
    parser.add_argument('--max_iteration', type=int, default=100000, help='Total number of iterations to train for')
    parser.add_argument('--ckpting_save_iter', type=int, default=1000, help='Checkpoint save interval')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--max_train_iteration', type=int, default=100, help='Total number of training iterations per run')
    parser.add_argument('--max_val_iteration', type=int, default=5, help='Total number of validation iterations per evaluation')
    parser.add_argument('--val_freq_iteration', type=int, default=25, help='Run validation every N training iterations')

    # ===== DATA / PATHS =====
    parser.add_argument( '--data_path', type=Path, default=Path('./data'), help='Root directory where all data files live')
    parser.add_argument( '--runs_path', type=Path, default=Path('./runs'), help='Directory for experiment runs / checkpoints / logs')
    parser.add_argument( '--vocab_path', type=Path, default=Path('./data/gpt2_vocab.json'), help='Path to GPT-2 tokenizer vocab file')
    parser.add_argument( '--merges_path', type=Path, default=Path('./data/gpt2_merges.txt'), help='Path to GPT-2 tokenizer merges file')
    parser.add_argument( '--np_dat_train_path', type=Path, default=Path('./data/TinyStoriesV2-GPT4-train.dat'), help='Memory-mapped (numpy) training dataset')
    parser.add_argument( '--total_train_tokens', type=int, default=547_994_686, help='Total number of tokens in the training set (for progress bars / LR schedules)')
    parser.add_argument( '--np_dat_valid_path', type=Path, default=Path('./data/TinyStoriesV2-GPT4-valid.dat'), help='Memory-mapped (numpy) validation dataset')
    parser.add_argument( '--total_val_tokens', type=int, default=5_535_291, help='Total number of tokens in the validation set')


    return parser.parse_args()

args = get_args()

# wandb config
run = wandb.init(
    entity="yiltro8-org",
    project="transformer_lm",
    name=f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{wandb.util.generate_id()}",
    config={
        "architecture": "Transformer LM",
        "dataset": "TinyStoriesV2-GPT4",
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
        "betas": args.betas,
        "eps": args.eps,
        "weight_decay": args.weight_decay,
        "grad_clip_max_l2_norm": args.grad_clip_max_l2_norm,
        "max_learning_rate": args.max_learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "warmup_iters": args.warmup_iters,
        "cosine_cycle_iters": args.cosine_cycle_iters,
        "max_train_iteration": args.max_train_iteration,
        "max_val_iteration": args.max_val_iteration,
        "val_freq_iteration": args.val_freq_iteration,
        "batch_size": args.batch_size,
        "device": args.device,
        "dtype": args.dtype,
        "ckpting_save_iter": args.ckpting_save_iter,
    },
)
cfg  = run.config

# data config
DATA_PATH = args.data_path            
RUNS_PATH = args.runs_path            
VOCAB_PATH = args.vocab_path           
MERGES_PATH = args.merges_path
NP_DAT_TRAIN_PATH = args.np_dat_train_path
total_train_tokens  = args.total_train_tokens
NP_DAT_VALID_PATH = args.np_dat_valid_path
total_val_tokens = args.total_val_tokens     

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