from transformerlm.transformer.transformer_lm import (TransformerLM,
                                                      Linear)
from transformerlm.training.optim import AdamW
from transformerlm.training.data import get_batch
from transformerlm.training.loss import cross_entropy
from transformerlm.training.checkpoint import save_checkpoint
from transformerlm.training.schedule import lr_cosine_schedule
from transformerlm.training.grad import gradient_clipping

from pathlib import Path
import numpy as np
import torch
import wandb
import datetime
import os
import argparse
import json

from transformerlm.config import (
    load_train_config,
    asdict_pretty,
)
from transformerlm.utils.dtypes import DTYPES

def train_transformer(args):
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

    ckpting_save_folder = args.runs_path / run.name
    if not os.path.exists(ckpting_save_folder):
        os.makedirs(ckpting_save_folder)

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

    np_arr_train_data = np.memmap(args.np_dat_train_path,
                                dtype=np.int32,
                                mode='r',
                                shape=(args.total_train_tokens,))

    np_arr_valid_data = np.memmap(args.np_dat_valid_path,
                                dtype=np.int32,
                                mode='r',
                                shape=(args.total_val_tokens,))

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

def _parse_only_config():
    parser = argparse.ArgumentParser(description="Training via config file only.", allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True, help="Path to train TOML config")
    parser.add_argument("--print-config", action="store_true", help="Print resolved config and exit")
    return parser.parse_args()

def main():
    # Config-only entry point
    args_cfg = _parse_only_config()
    cfg_dc = load_train_config(args_cfg.config)
    if args_cfg.print_config:
        print(json.dumps(asdict_pretty(cfg_dc), indent=2))
        return

    # Build an argparse-like namespace expected by existing code
    ns = argparse.Namespace(
        # optimizer
        betas=(cfg_dc.optimizer.betas[0], cfg_dc.optimizer.betas[1]),
        eps=cfg_dc.optimizer.eps,
        weight_decay=cfg_dc.optimizer.weight_decay,
        max_learning_rate=cfg_dc.optimizer.max_learning_rate,
        min_learning_rate=cfg_dc.optimizer.min_learning_rate,
        warmup_iters=cfg_dc.optimizer.warmup_iters,
        cosine_cycle_iters=cfg_dc.optimizer.cosine_cycle_iters,
        grad_clip_max_l2_norm=cfg_dc.optimizer.grad_clip_max_l2_norm,
        # model
        vocab_size=cfg_dc.model.vocab_size,
        context_length=cfg_dc.model.context_length,
        d_model=cfg_dc.model.d_model,
        num_layers=cfg_dc.model.num_layers,
        num_heads=cfg_dc.model.num_heads,
        d_ff=cfg_dc.model.d_ff,
        rope_theta=cfg_dc.model.rope_theta,
        # global
        device=cfg_dc.model.device,
        dtype=cfg_dc.model.dtype,
        max_iteration=None,
        ckpting_save_iter=cfg_dc.training.ckpting_save_iter,
        batch_size=cfg_dc.training.batch_size,
        max_train_iteration=cfg_dc.training.max_train_iteration,
        max_val_iteration=cfg_dc.training.max_val_iteration,
        val_freq_iteration=cfg_dc.training.val_freq_iteration,
        # data/paths
        runs_path=cfg_dc.data.runs_path,
        np_dat_train_path=cfg_dc.data.np_dat_train_path,
        total_train_tokens=cfg_dc.data.total_train_tokens,
        np_dat_valid_path=cfg_dc.data.np_dat_valid_path,
        total_val_tokens=cfg_dc.data.total_val_tokens,
    )
    train_transformer(ns)

if __name__ == "__main__":
    main()
