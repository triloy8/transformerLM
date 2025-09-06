from transformerlm.tokenizer.tokenizer import Tokenizer
from transformerlm.models import TransformerLM
from transformerlm.inference.generate import generate
import torch
from transformerlm.utils.dtypes import DTYPES


def infer_transformer(args):
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_path, merges_filepath=args.merges_path, special_tokens=args.special_tokens
    )
    ids = [tokenizer.encode(text) for text in args.text_list]

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
        dtype=DTYPES[args.dtype],
    )

    ckpt_dict = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt_dict["model_state_dict"])

    in_indices = torch.tensor(ids, device=args.device)

    out_indices = generate(
        model,
        in_indices=in_indices,
        steps=args.context_length,
        temperature=args.temperature,
        p=args.p,
        eos_token_id=args.eos_token_id,
        context_length=args.context_length,
    )

    output_strings = []
    for out_indices_ in out_indices:
        out_indices_list = out_indices_.tolist()
        output_string = tokenizer.decode(out_indices_list)
        output_strings.append(output_string)

    return output_strings

