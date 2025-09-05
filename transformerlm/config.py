from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import tomllib


ALLOWED_DTYPES = {"float32", "float16", "bfloat16"}
ALLOWED_DEVICES = {"cpu", "cuda"}


# ===== Dataclasses (Schemas) =====

@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float
    device: str
    dtype: str


@dataclass
class OptimizerConfig:
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    max_learning_rate: float
    min_learning_rate: float
    warmup_iters: int
    cosine_cycle_iters: int
    grad_clip_max_l2_norm: float


@dataclass
class TrainingConfig:
    batch_size: int
    max_train_iteration: int
    max_val_iteration: int
    val_freq_iteration: int
    ckpting_save_iter: int


@dataclass
class DataConfig:
    runs_path: Path
    np_dat_train_path: Path
    total_train_tokens: int
    np_dat_valid_path: Path
    total_val_tokens: int


@dataclass
class WandbConfig:
    entity: Optional[str] = None
    project: Optional[str] = None


@dataclass
class TrainConfig:
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    data: DataConfig
    wandb: Optional[WandbConfig] = None


@dataclass
class TokenizerConfig:
    merges_path: Path
    vocab_path: Path
    special_tokens: List[str]


@dataclass
class CheckpointConfig:
    ckpt_path: Path


@dataclass
class InferenceConfig:
    text_list: List[str]
    temperature: float
    p: float
    eos_token_id: int


@dataclass
class InferConfig:
    tokenizer: TokenizerConfig
    model: ModelConfig
    checkpoint: CheckpointConfig
    inference: InferenceConfig


@dataclass
class MakeDataInputConfig:
    input_filename: Path
    total_tokens: int


@dataclass
class MakeDataOutputConfig:
    output_filename: Path


@dataclass
class MakeDataConfig:
    input: MakeDataInputConfig
    output: MakeDataOutputConfig
    tokenizer: TokenizerConfig


@dataclass
class TrainTokenizerInputConfig:
    input_path: Path
    vocab_size: int
    special_tokens: List[str]


@dataclass
class TrainTokenizerOutputConfig:
    merges_path: Path
    vocab_path: Path


@dataclass
class TrainTokenizerConfig:
    input: TrainTokenizerInputConfig
    output: TrainTokenizerOutputConfig


# ===== Helpers =====

def _as_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _expect_keys(d: Dict[str, Any], name: str, keys: List[str]) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"Missing keys in [{name}]: {missing}")


def _validate_model(m: ModelConfig) -> None:
    if m.d_model % m.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    if m.dtype not in ALLOWED_DTYPES:
        raise ValueError(f"dtype must be one of {sorted(ALLOWED_DTYPES)}")
    if m.device not in ALLOWED_DEVICES:
        raise ValueError(f"device must be one of {sorted(ALLOWED_DEVICES)}")
    for k in ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff"):
        if getattr(m, k) <= 0:
            raise ValueError(f"{k} must be > 0")
    if m.rope_theta <= 0:
        raise ValueError("rope_theta must be > 0")


def _validate_optimizer(o: OptimizerConfig) -> None:
    if len(o.betas) != 2:
        raise ValueError("betas must have 2 elements")
    if not (0 <= o.betas[0] < 1 and 0 <= o.betas[1] < 1):
        raise ValueError("betas must be in [0,1)")
    for k in ("eps", "max_learning_rate", "min_learning_rate", "grad_clip_max_l2_norm"):
        if getattr(o, k) <= 0:
            raise ValueError(f"{k} must be > 0")
    for k in ("warmup_iters", "cosine_cycle_iters"):
        if getattr(o, k) < 0:
            raise ValueError(f"{k} must be >= 0")
    if o.min_learning_rate > o.max_learning_rate:
        raise ValueError("min_learning_rate must be <= max_learning_rate")


def _validate_training(t: TrainingConfig) -> None:
    for k in ("batch_size", "max_train_iteration", "max_val_iteration", "val_freq_iteration", "ckpting_save_iter"):
        if getattr(t, k) <= 0:
            raise ValueError(f"{k} must be > 0")


def _validate_data(d: DataConfig) -> None:
    # Inputs should exist; create of outputs handled by caller.
    if not d.np_dat_train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {d.np_dat_train_path}")
    if not d.np_dat_valid_path.exists():
        raise FileNotFoundError(f"Validation dataset not found: {d.np_dat_valid_path}")
    if d.total_train_tokens <= 0 or d.total_val_tokens <= 0:
        raise ValueError("total_*_tokens must be > 0")


def _validate_inference(i: InferenceConfig) -> None:
    if i.temperature <= 0:
        raise ValueError("temperature must be > 0")
    if not (0 <= i.p <= 1):
        raise ValueError("p must be in [0, 1]")
    if i.eos_token_id < 0:
        raise ValueError("eos_token_id must be >= 0")
    if not i.text_list:
        raise ValueError("text_list must not be empty")


def _validate_tokenizer(tok: TokenizerConfig) -> None:
    if not tok.vocab_path.exists():
        raise FileNotFoundError(f"vocab_path not found: {tok.vocab_path}")
    if not tok.merges_path.exists():
        raise FileNotFoundError(f"merges_path not found: {tok.merges_path}")


# ===== Loaders =====

def _load_toml(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_train_config(path: Path | str) -> TrainConfig:
    cfg = _load_toml(_as_path(path))
    _expect_keys(cfg, "root", ["model", "optimizer", "training", "data"])

    m = cfg["model"]
    o = cfg["optimizer"]
    t = cfg["training"]
    d = cfg["data"]
    w = cfg.get("wandb", {})

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
    wandb = None
    if w:
        wandb = WandbConfig(entity=w.get("entity"), project=w.get("project"))

    _validate_model(model)
    _validate_optimizer(optimizer)
    _validate_training(training)
    _validate_data(data)

    return TrainConfig(model=model, optimizer=optimizer, training=training, data=data, wandb=wandb)


def load_infer_config(path: Path | str) -> InferConfig:
    cfg = _load_toml(_as_path(path))
    _expect_keys(cfg, "root", ["tokenizer", "model", "checkpoint", "inference"])
    tok = cfg["tokenizer"]
    m = cfg["model"]
    c = cfg["checkpoint"]
    i = cfg["inference"]

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

    _validate_tokenizer(tokenizer)
    _validate_model(model)
    _validate_inference(inference)
    if not checkpoint.ckpt_path.exists():
        raise FileNotFoundError(f"ckpt_path not found: {checkpoint.ckpt_path}")

    return InferConfig(tokenizer=tokenizer, model=model, checkpoint=checkpoint, inference=inference)


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


def asdict_pretty(dc) -> Dict[str, Any]:
    # Convert dataclass to dict with stringified Paths for printing
    def _stringify(obj):
        if isinstance(obj, Path):
            return str(obj)
        return obj
    d = asdict(dc)
    def walk(x):
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return _stringify(x)
    return walk(d)

