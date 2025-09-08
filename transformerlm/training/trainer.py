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
from transformerlm.training.loop import train_loop

import numpy as np
import os

from transformerlm.utils.dtypes import DTYPES
from logger import Logger


def train_transformer(args, *, logger: Logger, run_name: str):
    # checkpoint folder based on run_name provided by logger
    cfg = args
    ckpting_save_folder = args.runs_path / run_name
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

    train_loop(
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
        logger=logger,
        activation_norms=activation_norms,
        log_activation_norms=True,
        log_weight_norms=True,
    )
