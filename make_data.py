from tokenizer import Tokenizer
from pathlib import Path
import time
import numpy as np

DATA_PATH = Path("./transformerLM/data")
VOCAB_PATH = DATA_PATH / "gpt2_vocab.json"
MERGES_PATH = DATA_PATH / "gpt2_merges.txt"
TINYSTORIES_TRAIN_PATH = DATA_PATH / "TinyStoriesV2-GPT4-train.txt" # total_tokens = 547994686
TINYSTORIES_VAL_PATH = DATA_PATH / "TinyStoriesV2-GPT4-valid.txt" # total_tokens = 5535291

tokenizer = Tokenizer.from_files(vocab_filepath=VOCAB_PATH, 
                                 merges_filepath=MERGES_PATH, 
                                 special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"])

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