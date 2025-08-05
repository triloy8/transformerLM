from transformerlm.tokenizer.tokenizer import Tokenizer
from transformerlm.transformer.transformer_lm import TransformerLM
import torch
import argparse

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

def get_args():
    parser = argparse.ArgumentParser(
        description="Inference script for TransformerLM."
    )

    # ===== TOKENIZER =====
    parser.add_argument('--merges_path', type=str, required=True, help='Input path for merges file')
    parser.add_argument('--vocab_path', type=str, required=True, help='Input for vocab JSON file')
    parser.add_argument('--special_tokens', nargs='+',  type=str, help='List of special tokens (space separated)')

    # ===== MODEL =====
    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size')
    parser.add_argument('--context_length', type=int, default=1024, help='Max sequence/context length')
    parser.add_argument('--d_model', type=int, default=768, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=3072, help='Feedforward hidden dimension')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='Rotary embedding theta')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train on (e.g., cuda, cpu)')
    parser.add_argument('--dtype', type=str, default='float32', help='Tensor dtype (e.g., float32, bfloat16)')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Input path for model ckpt')

    # ===== INFERENCE =====
    parser.add_argument('--text_list', nargs='+',  type=str, help='Text list as input')
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--p", type=float, default=0.2, help="Top-p sampling value")
    parser.add_argument("--eos_token_id", type=int, default=50256, help="End-of-sequence token id")

    return parser.parse_args()

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

if __name__ == "__main__":
    args = get_args()
    output_strings = infer_transformer(args)
    print(output_strings)
