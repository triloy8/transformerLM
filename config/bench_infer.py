from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .schemas import (
    TokenizerConfig,
    ModelConfig,
    CheckpointConfig,
    InferenceConfig,
    BenchParams,
    BenchInferConfig,
    LoggingConfig,
)
from .io import _as_path, _expect_keys, _load_toml
from .validate import (
    _validate_tokenizer,
    _validate_model,
    _validate_inference,
    _validate_bench_params,
)


def load_bench_infer_config(path: Path | str) -> BenchInferConfig:
    cfg = _load_toml(_as_path(path))
    _expect_keys(cfg, "root", ["tokenizer", "model", "checkpoint", "inference", "benchmark"])

    tok: Dict[str, Any] = cfg["tokenizer"]
    m: Dict[str, Any] = cfg["model"]
    c: Dict[str, Any] = cfg["checkpoint"]
    i: Dict[str, Any] = cfg["inference"]
    b: Dict[str, Any] = cfg["benchmark"]
    lg: Dict[str, Any] = cfg.get("logging", {})

    tokenizer = TokenizerConfig(
        merges_path=_as_path(tok["merges_path"]),
        vocab_path=_as_path(tok["vocab_path"]),
        special_tokens=list(tok.get("special_tokens", [])),
    )
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
    checkpoint = CheckpointConfig(ckpt_path=_as_path(c["ckpt_path"]))
    inference = InferenceConfig(
        text_list=list(i["text_list"]),
        temperature=float(i["temperature"]),
        p=float(i["p"]),
        eos_token_id=int(i["eos_token_id"]),
    )
    benchmark = BenchParams(
        warmup=int(b.get("warmup", 2)),
        repeats=int(b.get("repeats", 5)),
        steps=int(b.get("steps", model.context_length)),
        synchronize=bool(b.get("synchronize", True)),
        backward=bool(b.get("backward", False)),
    )

    _validate_tokenizer(tokenizer)
    _validate_model(model)
    _validate_inference(inference)
    _validate_bench_params(benchmark)
    if not checkpoint.ckpt_path.exists():
        raise FileNotFoundError(f"ckpt_path not found: {checkpoint.ckpt_path}")

    logging: Optional[LoggingConfig] = None
    if lg:
        logging = LoggingConfig(
            backend=lg.get("backend"),
            run_name=lg.get("run_name"),
            architecture=lg.get("architecture"),
            dataset=lg.get("dataset"),
        )
    return BenchInferConfig(
        tokenizer=tokenizer,
        model=model,
        checkpoint=checkpoint,
        inference=inference,
        benchmark=benchmark,
        logging=logging,
    )

