from transformerlm.models import (
    TransformerLM,
    Linear,
)
from transformerlm.training.optim import AdamW
from transformerlm.training.data import get_batch
from transformerlm.training.loss import cross_entropy
from transformerlm.training.checkpoint import save_checkpoint
from transformerlm.training.schedule import lr_cosine_schedule
from transformerlm.training.grad import gradient_clipping
from transformerlm.training.loop import train_iterations

import numpy as np
import wandb
import datetime
import os

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
    cfg = run.config

    ckpting_save_folder = args.runs_path / run.name
    if not os.path.exists(ckpting_save_folder):
        os.makedirs(ckpting_save_folder)

    model = TransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=cfg.rope_theta,
        device=cfg.device,
        dtype=DTYPES[cfg.dtype],
    )

    optimizer = AdamW(
        model.parameters(),
        lr=0.003,
        betas=cfg.betas,
        eps=float(cfg.eps),
        weight_decay=cfg.weight_decay,
    )

    np_arr_train_data = np.memmap(
        args.np_dat_train_path, dtype=np.int32, mode="r", shape=(args.total_train_tokens,)
    )

    np_arr_valid_data = np.memmap(
        args.np_dat_valid_path, dtype=np.int32, mode="r", shape=(args.total_val_tokens,)
    )

    # activation norm utils
    activation_norms = {}

    def get_activation_norm_hook(name):
        def hook(module, input, output):
            activation_norms[name] = output.norm().item()

        return hook

    for name, module in model.named_modules():
        if isinstance(module, Linear):
            module.register_forward_hook(get_activation_norm_hook(name))

    def _log_fn(data: dict, step: int):
        wandb.log(data, step=step)

    train_iterations(
        model,
        optimizer,
        np_arr_train_data=np_arr_train_data,
        np_arr_valid_data=np_arr_valid_data,
        batch_size=cfg.batch_size,
        context_length=cfg.context_length,
        device=str(cfg.device),
        max_learning_rate=cfg.max_learning_rate,
        min_learning_rate=cfg.min_learning_rate,
        warmup_iters=cfg.warmup_iters,
        cosine_cycle_iters=cfg.cosine_cycle_iters,
        max_train_iteration=cfg.max_train_iteration,
        max_val_iteration=cfg.max_val_iteration,
        val_freq_iteration=cfg.val_freq_iteration,
        grad_clip_max_l2_norm=cfg.grad_clip_max_l2_norm,
        ckpting_save_iter=cfg.ckpting_save_iter,
        ckpting_save_folder=ckpting_save_folder,
        get_batch=get_batch,
        cross_entropy=cross_entropy,
        lr_cosine_schedule=lr_cosine_schedule,
        gradient_clipping=gradient_clipping,
        save_checkpoint=save_checkpoint,
        log=_log_fn,
        activation_norms=activation_norms,
        log_activation_norms=True,
        log_weight_norms=True,
    )

    wandb.finish()

