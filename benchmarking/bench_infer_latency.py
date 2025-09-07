from __future__ import annotations

import argparse
from typing import List

import torch

from transformerlm.cli.utils import add_config_args, load_config_or_print
from transformerlm.config import load_bench_infer_config
from transformerlm.tokenizer.tokenizer import Tokenizer
from transformerlm.models import TransformerLM
from transformerlm.utils.dtypes import DTYPES
from transformerlm.training.loss import cross_entropy
from transformerlm.logging.console_logger import ConsoleLogger

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
    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(cfg.tokenizer.vocab_path),
        merges_filepath=str(cfg.tokenizer.merges_path),
        special_tokens=cfg.tokenizer.special_tokens,
    )
    ids: List[List[int]] = [tokenizer.encode(text) for text in cfg.inference.text_list]
    prompt_lens = [len(x) for x in ids]
    batch_size = len(ids)

    # Model
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
    info = logger.start_run(run_config)

    # Warmup: run the same loop shape as the timed section
    inputs = in_indices[:, :-1]
    targets = in_indices[:, 1:]
    for _ in range(cfg.benchmark.warmup):
        if not cfg.benchmark.backward:
            model.eval()
            with torch.no_grad():
                for _ in range(cfg.benchmark.steps):
                    _ = model(inputs)
        else:
            model.train()
            for _ in range(cfg.benchmark.steps):
                model.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = cross_entropy(logits, targets).mean()
                loss.backward()

    # Timed repeats
    latencies_ms: List[float] = []
    tokens_per_sec: List[float] = []

    for r in range(cfg.benchmark.repeats):
        def _run():
            # Standardized micro-bench: repeat forward (and optional backward) on fixed inputs
            if not cfg.benchmark.backward:
                model.eval()
                with torch.no_grad():
                    last_logits = None
                    for _ in range(cfg.benchmark.steps):
                        last_logits = model(inputs)
                    return last_logits
            else:
                model.train()
                last_logits = None
                for _ in range(cfg.benchmark.steps):
                    model.zero_grad(set_to_none=True)
                    last_logits = model(inputs)
                    loss = cross_entropy(last_logits, targets).mean()
                    loss.backward()
                return last_logits

        _, dt = measure(cfg.model.device, _run, synchronize=cfg.benchmark.synchronize)
        # Processed tokens: per iteration, each sequence processes (T_in - 1) positions
        T_in = int(in_indices.shape[1])
        processed_tokens = max(T_in - 1, 0) * batch_size * int(cfg.benchmark.steps)

        lat_ms = dt * 1000.0
        tps = (float(processed_tokens) / dt) if dt > 0 else 0.0
        latencies_ms.append(lat_ms)
        tokens_per_sec.append(tps)

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
