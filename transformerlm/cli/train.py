import argparse
from pathlib import Path
from datetime import datetime

from transformerlm.config import load_train_config
from transformerlm.training.trainer import train_transformer
from transformerlm.cli.utils import add_config_args, load_config_or_print
from transformerlm.logging.noop import NoOpLogger
from transformerlm.logging.wandb_logger import WandbLogger


def _parse_only_config():
    parser = argparse.ArgumentParser(description="Training via config file only.", allow_abbrev=False)
    add_config_args(parser, type_=Path)
    return parser.parse_args()


def main():
    # Config-only entry point
    args_cfg = _parse_only_config()
    cfg_dc = load_config_or_print(load_train_config, args_cfg.config, args_cfg.print_config)
    if cfg_dc is None:
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
    # Build run config payload (similar to prior wandb config)
    run_config = {
        "architecture": "Transformer LM",
        "dataset": "TinyStoriesV2-GPT4",
        "vocab_size": ns.vocab_size,
        "context_length": ns.context_length,
        "d_model": ns.d_model,
        "num_layers": ns.num_layers,
        "num_heads": ns.num_heads,
        "d_ff": ns.d_ff,
        "rope_theta": ns.rope_theta,
        "betas": ns.betas,
        "eps": ns.eps,
        "weight_decay": ns.weight_decay,
        "grad_clip_max_l2_norm": ns.grad_clip_max_l2_norm,
        "max_learning_rate": ns.max_learning_rate,
        "min_learning_rate": ns.min_learning_rate,
        "warmup_iters": ns.warmup_iters,
        "cosine_cycle_iters": ns.cosine_cycle_iters,
        "max_train_iteration": ns.max_train_iteration,
        "max_val_iteration": ns.max_val_iteration,
        "val_freq_iteration": ns.val_freq_iteration,
        "batch_size": ns.batch_size,
        "device": ns.device,
        "dtype": ns.dtype,
        "ckpting_save_iter": ns.ckpting_save_iter,
    }

    # Select logger backend
    logger = NoOpLogger()
    if cfg_dc.wandb and cfg_dc.wandb.entity and cfg_dc.wandb.project:
        # Stable default name; wandb may override formatting but keeps base name
        default_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logger = WandbLogger(entity=cfg_dc.wandb.entity, project=cfg_dc.wandb.project, name=default_name)

    info = logger.start_run(run_config)
    run_name = info.get("run_name") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    try:
        train_transformer(ns, logger=logger, run_name=run_name)
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
