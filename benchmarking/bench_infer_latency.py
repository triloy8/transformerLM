from __future__ import annotations

import argparse
from typing import List

import torch

from cli.utils import add_config_args, load_config_or_print
from config import load_bench_infer_config
from transformerlm.tokenizer.tokenizer import Tokenizer
from transformerlm.models import TransformerLM
from transformerlm.utils.dtypes import DTYPES
from transformerlm.training.loss import cross_entropy
from transformerlm.training.optim import AdamW
from transformerlm.training.grad import gradient_clipping
from logger import ConsoleLogger
from profiling import nvtx

from .common import measure, mean, stddev


def _parse_only_config():
    parser = argparse.ArgumentParser(description="Benchmark: inference latency via config.", allow_abbrev=False)
    add_config_args(parser, type_=str)
    return parser.parse_args()


def main():
    args_cfg = _parse_only_config()
    cfg = load_config_or_print(load_bench_infer_config, args_cfg.config, args_cfg.print_config)
    if cfg is None:
        return

    logger = ConsoleLogger()

    # Prepare tokenizer and inputs
    with nvtx.range("bench/setup/tokenizer"):
        tokenizer = Tokenizer.from_files(
            vocab_filepath=str(cfg.tokenizer.vocab_path),
            merges_filepath=str(cfg.tokenizer.merges_path),
            special_tokens=cfg.tokenizer.special_tokens,
        )
        ids: List[List[int]] = [tokenizer.encode(text) for text in cfg.inference.text_list]
    prompt_lens = [len(x) for x in ids]
    batch_size = len(ids)

    # Model
    with nvtx.range("bench/setup/model_load"):
        model = TransformerLM(
            vocab_size=cfg.model.vocab_size,
            context_length=cfg.model.context_length,
            d_model=cfg.model.d_model,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            d_ff=cfg.model.d_ff,
            rope_theta=cfg.model.rope_theta,
            device=cfg.model.device,
            dtype=DTYPES[cfg.model.dtype],
        )
        ckpt = torch.load(str(cfg.checkpoint.ckpt_path), map_location=cfg.model.device)
        model.load_state_dict(ckpt["model_state_dict"])  # type: ignore[index]
        # Snapshot initial model state on CPU for per-repeat resets
        initial_model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    # Snapshot initial model state on CPU for per-repeat resets
    initial_model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    in_indices = torch.tensor(ids, device=cfg.model.device)

    # Start run logging
    run_config = {
        "benchmark": "infer_latency",
        "device": cfg.model.device,
        "dtype": cfg.model.dtype,
        "batch_size": batch_size,
        "prompt_len_avg": float(mean([float(x) for x in prompt_lens])),
        "prompt_count": batch_size,
        "backward": bool(cfg.benchmark.backward),
        # model
        "vocab_size": cfg.model.vocab_size,
        "context_length": cfg.model.context_length,
        "d_model": cfg.model.d_model,
        "num_layers": cfg.model.num_layers,
        "num_heads": cfg.model.num_heads,
        "d_ff": cfg.model.d_ff,
        "rope_theta": cfg.model.rope_theta,
        # sampling
        "temperature": cfg.inference.temperature,
        "p": cfg.inference.p,
        "eos_token_id": cfg.inference.eos_token_id,
        # bench
        "warmup": cfg.benchmark.warmup,
        "repeats": cfg.benchmark.repeats,
        "steps": cfg.benchmark.steps,
        "synchronize": cfg.benchmark.synchronize,
    }
    # Optimizer (optional)
    optimizer = None
    if cfg.benchmark.optimizer_step:
        # If no optimizer config provided, use safe defaults per loader (lr=0.0, etc.)
        opt_cfg = cfg.optimizer
        if opt_cfg is None:
            raise ValueError("optimizer_step enabled but no optimizer config provided")
        with nvtx.range("bench/setup/optimizer"):
            optimizer = AdamW(
                model.parameters(),
                lr=opt_cfg.lr,
                betas=opt_cfg.betas,
                eps=opt_cfg.eps,
                weight_decay=opt_cfg.weight_decay,
            )

    # Annotate run with optimizer config if present
    if cfg.benchmark.optimizer_step and cfg.optimizer is not None:
        run_config.update(
            {
                "optimizer_step": True,
                "optimizer.lr": float(cfg.optimizer.lr),
                "optimizer.weight_decay": float(cfg.optimizer.weight_decay),
                "optimizer.betas": tuple(cfg.optimizer.betas),
                "optimizer.eps": float(cfg.optimizer.eps),
                "optimizer.grad_clip_max_l2_norm": float(cfg.optimizer.grad_clip_max_l2_norm),
            }
        )
    else:
        run_config.update({"optimizer_step": False})

    with nvtx.range("bench/setup/run_config"):
        info = logger.start_run(run_config)

    # Warmup: run the same loop shape as the timed section
    inputs = in_indices[:, :-1]
    targets = in_indices[:, 1:]
    clip_enabled = bool(cfg.optimizer is not None and cfg.optimizer.grad_clip_max_l2_norm > 0.0)
    with nvtx.range("bench/warmup"):
        for _ in range(cfg.benchmark.warmup):
            if not cfg.benchmark.backward:
                model.eval()
                with torch.no_grad():
                    for _ in range(cfg.benchmark.steps):
                        if nvtx.enabled("fine"):
                            with nvtx.range("bench/warmup/iter/forward"):
                                _ = model(inputs)
                        else:
                            _ = model(inputs)
            else:
                model.train()
                for _ in range(cfg.benchmark.steps):
                    model.zero_grad(set_to_none=True)
                    if nvtx.enabled("fine"):
                        with nvtx.range("bench/warmup/iter/forward"):
                            logits = model(inputs)
                        with nvtx.range("bench/warmup/iter/loss"):
                            loss = cross_entropy(logits, targets).mean()
                        with nvtx.range("bench/warmup/iter/backward"):
                            loss.backward()
                    else:
                        logits = model(inputs)
                        loss = cross_entropy(logits, targets).mean()
                        loss.backward()
                    if cfg.benchmark.optimizer_step and optimizer is not None:
                        # Optional gradient clipping
                        if clip_enabled:
                            if nvtx.enabled("fine"):
                                with nvtx.range("bench/warmup/iter/clip"):
                                    gradient_clipping(model.parameters(), cfg.optimizer.grad_clip_max_l2_norm)  # type: ignore[arg-type]
                            else:
                                gradient_clipping(model.parameters(), cfg.optimizer.grad_clip_max_l2_norm)  # type: ignore[arg-type]
                        if nvtx.enabled("fine"):
                            with nvtx.range("bench/warmup/iter/opt_step"):
                                optimizer.step()
                        else:
                            optimizer.step()

    # Timed repeats
    latencies_ms: List[float] = []
    tokens_per_sec: List[float] = []

    for r in range(cfg.benchmark.repeats):
        # Always reset model (and optimizer state) before each timed repeat
        with nvtx.range(f"bench/repeat[{r}]/reset"):
            model.load_state_dict(initial_model_state, strict=True)
            model.zero_grad(set_to_none=True)
            if cfg.benchmark.optimizer_step and cfg.optimizer is not None:
                optimizer = AdamW(
                    model.parameters(),
                    lr=cfg.optimizer.lr,
                    betas=cfg.optimizer.betas,
                    eps=cfg.optimizer.eps,
                    weight_decay=cfg.optimizer.weight_decay,
                )

        def _run():
            # Standardized micro-bench: repeat forward (and optional backward) on fixed inputs
            if not cfg.benchmark.backward:
                model.eval()
                with torch.no_grad():
                    last_logits = None
                    for _ in range(cfg.benchmark.steps):
                        if nvtx.enabled("fine"):
                            with nvtx.range("bench/timed/iter/forward"):
                                last_logits = model(inputs)
                        else:
                            last_logits = model(inputs)
                    return last_logits
            else:
                model.train()
                last_logits = None
                for _ in range(cfg.benchmark.steps):
                    model.zero_grad(set_to_none=True)
                    if nvtx.enabled("fine"):
                        with nvtx.range("bench/timed/iter/forward"):
                            last_logits = model(inputs)
                        with nvtx.range("bench/timed/iter/loss"):
                            loss = cross_entropy(last_logits, targets).mean()
                        with nvtx.range("bench/timed/iter/backward"):
                            loss.backward()
                    else:
                        last_logits = model(inputs)
                        loss = cross_entropy(last_logits, targets).mean()
                        loss.backward()
                    if cfg.benchmark.optimizer_step and optimizer is not None:
                        if clip_enabled:
                            if nvtx.enabled("fine"):
                                with nvtx.range("bench/timed/iter/clip"):
                                    gradient_clipping(model.parameters(), cfg.optimizer.grad_clip_max_l2_norm)  # type: ignore[arg-type]
                            else:
                                gradient_clipping(model.parameters(), cfg.optimizer.grad_clip_max_l2_norm)  # type: ignore[arg-type]
                        if nvtx.enabled("fine"):
                            with nvtx.range("bench/timed/iter/opt_step"):
                                optimizer.step()
                        else:
                            optimizer.step()
                return last_logits

        nvtx.mark("bench/measure_start")
        with nvtx.range(f"bench/repeat[{r}]/timed"):
            _, dt = measure(cfg.model.device, _run, synchronize=cfg.benchmark.synchronize)
        nvtx.mark("bench/measure_end")
        # Processed tokens: per iteration, each sequence processes (T_in - 1) positions
        T_in = int(in_indices.shape[1])
        processed_tokens = max(T_in - 1, 0) * batch_size * int(cfg.benchmark.steps)

        lat_ms = dt * 1000.0
        tps = (float(processed_tokens) / dt) if dt > 0 else 0.0
        latencies_ms.append(lat_ms)
        tokens_per_sec.append(tps)

        with nvtx.range(f"bench/repeat[{r}]/log"):
            logger.log(
                {
                    "phase": "bench_infer",
                    "metrics.latency_ms": lat_ms,
                    "metrics.tokens_sec": tps,
                    "metrics.processed_tokens": int(processed_tokens),
                    "metrics.batch_size": int(batch_size),
                    "metrics.backward": bool(cfg.benchmark.backward),
                },
                step=r,
            )

    # Summary
    with nvtx.range("bench/summary"):
        logger.log(
            {
                "phase": "bench_infer",
                "event": "summary",
                "metrics.latency_ms.mean": mean(latencies_ms),
                "metrics.tokens_sec.mean": mean(tokens_per_sec),
                "metrics.latency_ms.stddev": stddev(latencies_ms),
                "metrics.tokens_sec.stddev": stddev(tokens_per_sec),
                "metrics.iters": int(cfg.benchmark.repeats),
            }
        )

    logger.finish()


if __name__ == "__main__":
    main()
