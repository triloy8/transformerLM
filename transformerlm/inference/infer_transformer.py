from transformerlm.tokenizer.tokenizer import Tokenizer
from transformerlm.transformer.transformer_lm import TransformerLM
import torch
import argparse
import json

from transformerlm.config import (
    load_infer_config,
    asdict_pretty,
)

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

def infer_transformer(args):
    tokenizer = Tokenizer.from_files(vocab_filepath=args.vocab_path, merges_filepath=args.merges_path, special_tokens=args.special_tokens)
    ids = [tokenizer.encode(text) for text in args.text_list]

    model = TransformerLM(vocab_size=args.vocab_size, 
                        context_length=args.context_length,
                        d_model=args.d_model,
                        num_layers=args.num_layers, 
                        num_heads=args.num_heads, 
                        d_ff=args.d_ff, 
                        rope_theta=args.rope_theta,
                        device=args.device, 
                        dtype=DTYPES[args.dtype])

    ckpt_dict = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt_dict["model_state_dict"])

    in_indices = torch.tensor(ids, device=args.device)

    out_indices = model.decode(in_indices=in_indices, 
                            context_length=args.context_length, 
                            temperature=args.temperature, 
                            p=args.p, 
                            eos_token_id=args.eos_token_id)

    output_strings = []
    for out_indices_ in out_indices: 
        out_indices_list = out_indices_.tolist()
        output_string = tokenizer.decode(out_indices_list)
        output_strings.append(output_string)

    return output_strings

def _parse_only_config():
    parser = argparse.ArgumentParser(description="Inference via config file only.", allow_abbrev=False)
    parser.add_argument("--config", type=str, required=True, help="Path to inference TOML config")
    parser.add_argument("--print-config", action="store_true", help="Print resolved config and exit")
    return parser.parse_args()

def main():
    args_cfg = _parse_only_config()
    cfg_dc = load_infer_config(args_cfg.config)
    if args_cfg.print_config:
        print(json.dumps(asdict_pretty(cfg_dc), indent=2))
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
    output_strings = infer_transformer(ns)
    print(output_strings)

if __name__ == "__main__":
    main()
