from transformerlm.tokenizer.tokenizer import Tokenizer
from transformerlm.transformer.transformer_lm import TransformerLM
import torch

from pathlib import Path

DATA_PATH = Path("./data")
RUNS_PATH = Path("./runs")
VOCAB_PATH = DATA_PATH / "gpt2_vocab.json"
MERGES_PATH = DATA_PATH / "gpt2_merges.txt"
CKPT_PATH = RUNS_PATH / "2025-08-04_00-16-58_28jyumc2/4000.ckpt"

tokenizer = Tokenizer.from_files(vocab_filepath=VOCAB_PATH, merges_filepath=MERGES_PATH, special_tokens=["<|endoftext|>"])
ids = [tokenizer.encode("I love Lily"), tokenizer.encode("I love John"), tokenizer.encode("I love Judy")]

model = TransformerLM(vocab_size=50257, 
                      context_length=256, 
                      d_model=512, 
                      num_layers=4, 
                      num_heads=16, 
                      d_ff=1344, 
                      rope_theta=10000.0,
                      device="cuda", 
                      dtype=torch.float32)

ckpt_dict = torch.load(CKPT_PATH)
model.load_state_dict(ckpt_dict["model_state_dict"])

in_indices = torch.tensor(ids, device="cuda")

out_indices = model.decode(in_indices=in_indices, 
                           context_length=256, 
                           temperature=1.0, 
                           p=0.2, 
                           eos_token_id=50256)

output_strings = []
for out_indices_ in out_indices: 
    out_indices_list = out_indices_.tolist()
    output_string = tokenizer.decode(out_indices_list)
    output_strings.append(output_string)

print(output_strings)
