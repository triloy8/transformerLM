from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


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
    architecture: Optional[str] = None
    dataset: Optional[str] = None


@dataclass
class LoggingConfig:
    backend: Optional[str] = None  # "console" | "wandb" | "noop" | "jsonl"
    run_name: Optional[str] = None
    architecture: Optional[str] = None
    dataset: Optional[str] = None


@dataclass
class TrainConfig:
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    data: DataConfig
    wandb: Optional[WandbConfig] = None
    logging: Optional[LoggingConfig] = None


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
    logging: Optional[LoggingConfig] = None


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


# ===== Benchmark Schemas =====

@dataclass
class BenchParams:
    warmup: int
    repeats: int
    steps: int
    synchronize: bool = True
    backward: bool = False
    optimizer_step: bool = False


@dataclass
class BenchInferConfig:
    tokenizer: TokenizerConfig
    model: ModelConfig
    checkpoint: CheckpointConfig
    inference: InferenceConfig
    benchmark: BenchParams
    logging: Optional[LoggingConfig] = None
    optimizer: Optional["OptimizerBenchConfig"] = None


@dataclass
class OptimizerBenchConfig:
    lr: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    grad_clip_max_l2_norm: float = 0.0


@dataclass
class BenchTokenizerInput:
    text_list: List[str]


@dataclass
class BenchTokenizerParams:
    repeats: int


@dataclass
class BenchTokenizerConfig:
    tokenizer: TokenizerConfig
    input: BenchTokenizerInput
    benchmark: BenchTokenizerParams
    logging: Optional[LoggingConfig] = None
