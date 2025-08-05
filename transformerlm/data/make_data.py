from transformerlm.tokenizer.tokenizer import Tokenizer
from pathlib import Path
import argparse
import time
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(
        description="Tokenize a text file and save tokens as a .dat numpy array."
    )

    parser.add_argument("--input_filename", type=str, required=True, help="Path to the input .txt file")
    parser.add_argument("--output_filename", type=str, required=True, help="Path to the output .dat file (numpy array of tokens)")
    parser.add_argument("--vocab_path", type=str, default=str("gpt2_vocab.json"), help="Path to the vocab JSON file")
    parser.add_argument("--merges_path", type=str, default=str("gpt2_merges.txt"), help="Path to the merges file")
    parser.add_argument("--total_tokens", type=int, required=True, default=None, help="Total number of tokens to write")
    parser.add_argument( "--special_tokens", nargs="+",  type=str, help="List of special tokens (space separated)")
    
    return parser.parse_args()

def count_total_tokens(tokenizer, input_filename):
    with open(input_filename, 'r') as f:
        total_tokens = 0
        start_time = time.time()
        last_time = start_time

        for token_id in tokenizer.encode_iterable(f):
            total_tokens += 1
            if total_tokens % 1_000_000 == 0:
                now = time.time()
                elapsed = now - start_time
                interval = now - last_time
                print(f"{total_tokens:,} tokens counted... "
                    f"({elapsed:.1f}s total, {interval:.1f}s since last million)")
                last_time = now

        print(f"Total tokens: {total_tokens:,} ({time.time() - start_time:.1f}s)")

def write_token_ids_to_memmap(tokenizer, input_filename, total_tokens, output_filename, dtype=np.int32):
    arr = np.memmap(output_filename, dtype=dtype, mode='w+', shape=(total_tokens,))
    with open(input_filename, 'r') as f:
        start_time = time.time()
        last_time = start_time
        for i, token_id in enumerate(tokenizer.encode_iterable(f)):
            arr[i] = token_id
            if i > 0 and i % 1_000_000 == 0:
                now = time.time()
                elapsed = now - start_time
                interval = now - last_time
                print(f"Wrote {i:,} tokens... "
                      f"({elapsed:.1f}s total, {interval:.1f}s since last million)")
                last_time = now
    arr.flush()
    total_time = time.time() - start_time
    print(f"Done writing {total_tokens:,} tokens to {output_filename} in {total_time:.1f}s "
          f"({total_time/60:.1f} min).")

if __name__ == "__main__":
    args = get_args()

    tokenizer = Tokenizer.from_files(vocab_filepath=args.vocab_path,
                                     merges_filepath=args.merges_path,
                                     special_tokens=["<|endoftext|>"])

    write_token_ids_to_memmap(tokenizer, args.input_filename, args.total_tokens, args.output_filename)