from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .schemas import (
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
    DataConfig,
    WandbConfig,
    LoggingConfig,
    TrainConfig,
    MakeDataInputConfig,
    MakeDataOutputConfig,
    MakeDataConfig,
    TokenizerConfig,
    TrainTokenizerInputConfig,
    TrainTokenizerOutputConfig,
    TrainTokenizerConfig,
)
from .io import _as_path, _expect_keys, _load_toml
from .validate import _validate_model, _validate_optimizer, _validate_training, _validate_data, _validate_tokenizer


def load_train_config(path: Path | str) -> TrainConfig:
    cfg = _load_toml(_as_path(path))
    _expect_keys(cfg, "root", ["model", "optimizer", "training", "data"])

    m: Dict[str, Any] = cfg["model"]
    o: Dict[str, Any] = cfg["optimizer"]
    t: Dict[str, Any] = cfg["training"]
    d: Dict[str, Any] = cfg["data"]
    w: Dict[str, Any] = cfg.get("wandb", {})
    lg: Dict[str, Any] = cfg.get("logging", {})

    model = ModelConfig(
        vocab_size=int(m["vocab_size"]),
        context_length=int(m["context_length"]),
        d_model=int(m["d_model"]),
        num_layers=int(m["num_layers"]),
        num_heads=int(m["num_heads"]),
        d_ff=int(m["d_ff"]),
        rope_theta=float(m["rope_theta"]),
        device=str(m["device"]),
        dtype=str(m["dtype"]),
    )
    betas = o.get("betas", [0.9, 0.95])
    optimizer = OptimizerConfig(
        betas=(float(betas[0]), float(betas[1])),
        eps=float(o["eps"]),
        weight_decay=float(o["weight_decay"]),
        max_learning_rate=float(o["max_learning_rate"]),
        min_learning_rate=float(o["min_learning_rate"]),
        warmup_iters=int(o["warmup_iters"]),
        cosine_cycle_iters=int(o["cosine_cycle_iters"]),
        grad_clip_max_l2_norm=float(o["grad_clip_max_l2_norm"]),
    )
    training = TrainingConfig(
        batch_size=int(t["batch_size"]),
        max_train_iteration=int(t["max_train_iteration"]),
        max_val_iteration=int(t["max_val_iteration"]),
        val_freq_iteration=int(t["val_freq_iteration"]),
        ckpting_save_iter=int(t["ckpting_save_iter"]),
    )
    data = DataConfig(
        runs_path=_as_path(d["runs_path"]),
        np_dat_train_path=_as_path(d["np_dat_train_path"]),
        total_train_tokens=int(d["total_train_tokens"]),
        np_dat_valid_path=_as_path(d["np_dat_valid_path"]),
        total_val_tokens=int(d["total_val_tokens"]),
    )
    wandb: Optional[WandbConfig] = None
    if w:
        wandb = WandbConfig(
            entity=w.get("entity"),
            project=w.get("project"),
            architecture=w.get("architecture"),
            dataset=w.get("dataset"),
        )
    logging: Optional[LoggingConfig] = None
    if lg:
        logging = LoggingConfig(
            backend=lg.get("backend"),
            run_name=lg.get("run_name"),
            architecture=lg.get("architecture"),
            dataset=lg.get("dataset"),
        )

    _validate_model(model)
    _validate_optimizer(optimizer)
    _validate_training(training)
    _validate_data(data)

    return TrainConfig(model=model, optimizer=optimizer, training=training, data=data, wandb=wandb, logging=logging)


def load_make_data_config(path: Path | str) -> MakeDataConfig:
    cfg = _load_toml(_as_path(path))
    _expect_keys(cfg, "root", ["input", "output", "tokenizer"])
    i = cfg["input"]
    o = cfg["output"]
    t = cfg["tokenizer"]

    input_cfg = MakeDataInputConfig(
        input_filename=_as_path(i["input_filename"]),
        total_tokens=int(i["total_tokens"]),
    )
    output_cfg = MakeDataOutputConfig(output_filename=_as_path(o["output_filename"]))
    tokenizer = TokenizerConfig(
        merges_path=_as_path(t["merges_path"]),
        vocab_path=_as_path(t["vocab_path"]),
        special_tokens=list(t.get("special_tokens", [])),
    )

    if not input_cfg.input_filename.exists():
        raise FileNotFoundError(f"input_filename not found: {input_cfg.input_filename}")
    if input_cfg.total_tokens <= 0:
        raise ValueError("total_tokens must be > 0")
    _validate_tokenizer(tokenizer)

    return MakeDataConfig(input=input_cfg, output=output_cfg, tokenizer=tokenizer)


def load_train_tokenizer_config(path: Path | str) -> TrainTokenizerConfig:
    cfg = _load_toml(_as_path(path))
    _expect_keys(cfg, "root", ["input", "output"])
    i = cfg["input"]
    o = cfg["output"]

    input_cfg = TrainTokenizerInputConfig(
        input_path=_as_path(i["input_path"]),
        vocab_size=int(i["vocab_size"]),
        special_tokens=list(i.get("special_tokens", [])),
    )
    output_cfg = TrainTokenizerOutputConfig(
        merges_path=_as_path(o["merges_path"]),
        vocab_path=_as_path(o["vocab_path"]),
    )

    if not input_cfg.input_path.exists():
        raise FileNotFoundError(f"input_path not found: {input_cfg.input_path}")
    if input_cfg.vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")

    return TrainTokenizerConfig(input=input_cfg, output=output_cfg)
