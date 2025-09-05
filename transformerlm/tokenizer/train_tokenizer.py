from transformerlm.tokenizer.tokenizer_utils import (gpt2_bytes_to_unicode,
                                                     find_chunk_boundaries,
                                                     process_chunk_text,)
import os
import json
import argparse
import regex as re
from collections import Counter
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json as _json  # For printing config

from transformerlm.config import (
    load_train_tokenizer_config,
    asdict_pretty,
)


def get_args():
    parser = argparse.ArgumentParser(description="Process input path, vocab size, and special tokens.")

    # ===== TRAIN BPE INPUTS =====
    parser.add_argument( "--input_path", type=str, help="Path to the input file or directory")
    parser.add_argument( "--vocab_size", type=int, help="Vocabulary size")
    parser.add_argument( "--special_tokens", nargs="+",  type=str, help="List of special tokens (space separated)")

    # ===== TRAIN BPE OUTPUTS =====
    parser.add_argument('--merges_path', type=str, required=True, help='Output path for merges file')
    parser.add_argument('--vocab_path', type=str, required=True, help='Output path for vocab JSON file')

    
    args = parser.parse_args()
    args.input_path = os.path.expanduser(args.input_path)
    
    return args

def train_bpe(args):
    gpt2_byte_encoder = gpt2_bytes_to_unicode()
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

     # - initialize vocab with <|endoftext|> and gpt2 encoder
    vocab_list = [gpt2_byte_encoder[k] for k in gpt2_byte_encoder.keys()]
    initial_vocab = {k:i for i, k in enumerate(vocab_list)}

    # - open corpus, remove special token and pretokenize on chunks
    num_processes = 1
    pretoken_frequency = Counter()
    with open(args.input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f,
                                           num_processes,
                                           "<|endoftext|>".encode("utf-8"))
        
        with ProcessPoolExecutor() as executor:
            results = list(
                executor.map(
                    partial(process_chunk_text, input_path=args.input_path, special_tokens=args.special_tokens),
                    zip(boundaries[:-1], boundaries[1:])
                )
            )
        for partial_pretoken_frequency in results:
            pretoken_frequency.update(partial_pretoken_frequency)

    pretoken_frequency_dict = dict(pretoken_frequency)
    pretoken_frequency_tuple = {
        tuple(gpt2_byte_encoder[b] for b in word.encode('utf-8')): freq
        for word, freq in pretoken_frequency_dict.items()
    }

    # - compute bpe merges until vocab size is reached
    merges = []
    vocab = initial_vocab
    iter = 0
    iter_max = args.vocab_size - len(list(vocab.keys()))
    while iter < iter_max:
        # compute pair frequency
        pair_frequency = {}
        for freq_tuple, freq_value in pretoken_frequency_tuple.items():
            for i in range(len(freq_tuple)-1):
                pair = (freq_tuple[i], freq_tuple[i+1])
                if pair in pair_frequency: 
                    pair_frequency[pair] = pair_frequency[pair] + freq_value
                elif not pair in pair_frequency:
                    pair_frequency[pair] = freq_value
        
        # get biggest freq and update merges
        max_value = max(pair_frequency.values())
        max_pair_freq = [k for k, v in pair_frequency.items() if v == max_value]
        if len(max_pair_freq)>1:
            best_key = max(max_pair_freq, key=lambda p: ("".join([chr(gpt2_byte_decoder[c]) for c in p[0]]), "".join([chr(gpt2_byte_decoder[c]) for c in p[1]])))
        else:
            best_key = max_pair_freq[0]

        merges.append(best_key[0]+" "+best_key[1])

        # update pretoken frequency tuples with new merges
        new_merge = ''.join(best_key)
        new_pretoken_frequency_tuple = defaultdict(int)

        for key, value in pretoken_frequency_tuple.items():
            new_key = []
            i = 0
            while i < len(key):
                # Check for the merge pair
                if key[i:i+2] == best_key:
                    new_key.append(new_merge)
                    i += 2
                else:
                    new_key.append(key[i])
                    i += 1
            new_pretoken_frequency_tuple[tuple(new_key)] += value

        pretoken_frequency_tuple = dict(new_pretoken_frequency_tuple)

        iter += 1

    # - return vocab w/ additional merges and merges as bytes
    initial_vocab_size = len(list(vocab.keys()))
    for i, merge in enumerate(merges):
        merge_token = "".join(merge.split())
        vocab[merge_token] = initial_vocab_size + i
    for i, special_token in enumerate(args.special_tokens):
        vocab[special_token] = len(vocab)+i

    with open(args.merges_path, "w", encoding="utf-8") as f:
        for token in merges:
            f.write(token + "\n")

    with open(args.vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)


def _parse_only_config():
    parser = argparse.ArgumentParser(description="Train tokenizer via config file only.", allow_abbrev=False)
    parser.add_argument("--config", type=str, required=True, help="Path to train-tokenizer TOML config")
    parser.add_argument("--print-config", action="store_true", help="Print resolved config and exit")
    return parser.parse_args()

def main():
    args_cfg = _parse_only_config()
    cfg_dc = load_train_tokenizer_config(args_cfg.config)
    if args_cfg.print_config:
        print(_json.dumps(asdict_pretty(cfg_dc), indent=2))
        return

    # Build args Namespace compatible with existing train_bpe signature
    ns = argparse.Namespace(
        input_path=str(cfg_dc.input.input_path),
        vocab_size=cfg_dc.input.vocab_size,
        special_tokens=list(cfg_dc.input.special_tokens),
        merges_path=str(cfg_dc.output.merges_path),
        vocab_path=str(cfg_dc.output.vocab_path),
    )
    train_bpe(ns)

if __name__ == "__main__":
    main()
