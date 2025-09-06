import argparse

from transformerlm.config import load_infer_config
from transformerlm.inference.predictor import infer_transformer
from transformerlm.cli.utils import add_config_args, load_config_or_print
from transformerlm.logging.console_logger import ConsoleLogger


def _parse_only_config():
    parser = argparse.ArgumentParser(description="Inference via config file only.", allow_abbrev=False)
    add_config_args(parser, type_=str)
    return parser.parse_args()


def main():
    args_cfg = _parse_only_config()
    cfg_dc = load_config_or_print(load_infer_config, args_cfg.config, args_cfg.print_config)
    if cfg_dc is None:
        return

    # Build args Namespace expected by infer_transformer
    ns = argparse.Namespace(
        # tokenizer
        merges_path=str(cfg_dc.tokenizer.merges_path),
        vocab_path=str(cfg_dc.tokenizer.vocab_path),
        special_tokens=list(cfg_dc.tokenizer.special_tokens),
        # model
        vocab_size=cfg_dc.model.vocab_size,
        context_length=cfg_dc.model.context_length,
        d_model=cfg_dc.model.d_model,
        num_layers=cfg_dc.model.num_layers,
        num_heads=cfg_dc.model.num_heads,
        d_ff=cfg_dc.model.d_ff,
        rope_theta=cfg_dc.model.rope_theta,
        device=cfg_dc.model.device,
        dtype=cfg_dc.model.dtype,
        # checkpoint
        ckpt_path=str(cfg_dc.checkpoint.ckpt_path),
        # inference
        text_list=list(cfg_dc.inference.text_list),
        temperature=cfg_dc.inference.temperature,
        p=cfg_dc.inference.p,
        eos_token_id=cfg_dc.inference.eos_token_id,
    )
    logger = ConsoleLogger()
    output_strings = infer_transformer(ns, logger=logger)
    print(output_strings)


if __name__ == "__main__":
    main()
